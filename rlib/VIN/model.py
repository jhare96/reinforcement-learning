"""Value Iteration Network model."""

import torch
import torch.nn.functional as F

from rlib.agent import Agent, ModelConfig
from rlib.utils.utils import one_hot, tonumpy, totorch


class VINCNN(Agent):
    def __init__(self, input_size, action_size, k=10, lr=1e-3, device='cuda'):
        # VIN historically used a constant LR (no scheduler decay) and
        # no gradient clipping; encode that as a fixed config preset.
        config = ModelConfig(
            lr=lr,
            lr_final=lr,
            decay_steps=int(1e9),
            grad_clip=None,
            device=device,
        )
        super().__init__(config=config)
        channels, height, width = input_size
        self.action_size = action_size
        self.conv_enc = torch.nn.Conv2d(
            channels, 150, kernel_size=(3, 3), stride=(1, 1), padding=1
        ).to(device)  # φ(s)
        self.R_bar = torch.nn.Conv2d(
            150, 1, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False
        ).to(device)
        self.Q_bar = torch.nn.Conv2d(
            1, action_size, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
        ).to(device)
        self.w = torch.nn.Parameter(torch.zeros(action_size, 1, 3, 3), requires_grad=True).to(
            device
        )
        self.Q = torch.nn.Linear(action_size, action_size).to(device)
        self.k = k  # nsteps to plan with VIN
        self._build_optimiser(optim=torch.optim.RMSprop)

    def forward(self, img, x, y):
        hidden = self.conv_enc(img)
        R_bar = self.R_bar(hidden)
        Q_bar = self.Q_bar(R_bar)
        V_bar, _ = torch.max(Q_bar, dim=1, keepdim=True)
        batch_size = img.shape[0]
        psi = self._plan_ahead(R_bar, V_bar)[torch.arange(batch_size), :, x.long(), y.long()].view(
            batch_size, self.action_size
        )  # ψ(s)
        Qsa = self.Q(psi)
        return Qsa

    def evaluate(self, state, loc):
        with torch.no_grad():
            x, y = zip(*loc)
            x = torch.tensor(x).to(self.device)
            y = torch.tensor(y).to(self.device)
            Qsa = self.forward(totorch(state, self.device), x, y)
        return tonumpy(Qsa)

    def backprop(self, states, locs, R, actions):
        x, y = zip(*locs)
        Qsa = self.forward(
            totorch(states, self.device), torch.tensor(x).to(self.device), torch.tensor(y)
        ).to(self.device)
        actions_onehot = totorch(one_hot(actions, self.action_size), self.device)
        Qvalue = torch.sum(Qsa * actions_onehot, axis=1)
        loss = torch.mean(torch.square(totorch(R).float().cuda() - Qvalue))
        return self._train_step(loss)

    def value_iteration(self, r, V):
        return F.conv2d(
            # Stack reward with most recent value
            torch.cat([r, V], 1),
            # Convolve r->q weights to r, and v->q weights for v. These represent transition probabilities
            torch.cat([self.Q_bar.weight, self.w], 1),
            stride=1,
            padding=1,
        )

    def _plan_ahead(self, r, V):
        for _i in range(self.k):
            Q = self.value_iteration(r, V)
            V, _ = torch.max(Q, dim=1, keepdim=True)

        Q = self.value_iteration(r, V)
        return Q
