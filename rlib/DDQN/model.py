"""Q-network for DQN/DDQN."""

import numpy as np
import torch
import torch.nn.functional as F

from rlib.agent import Agent, ModelConfig
from rlib.utils.utils import totorch, totorch_many


class DQN(Agent):
    def __init__(
        self,
        model,
        input_shape,
        action_size,
        config: ModelConfig,
        *,
        optim: type[torch.optim.Optimizer] = torch.optim.Adam,
        optim_args: dict | None = None,
        **model_args,
    ):
        super().__init__(config=config)
        self.action_size = action_size

        self.model = model(input_shape, **model_args).to(self.device)
        self.Q = torch.nn.Linear(self.model.dense_size, action_size).to(self.device)

        self._build_optimiser(optim=optim, optim_args=optim_args)

    def loss(self, Qsa, R, action_onehot):
        Qvalue = torch.sum(Qsa * action_onehot, dim=1)
        loss = torch.mean(torch.square(R - Qvalue))
        return loss

    def backprop(self, state: np.ndarray, R: np.ndarray, action: np.ndarray):
        state, R, action = totorch_many(state, R, action, device=self.device)
        action_onehot = F.one_hot(action.long(), num_classes=self.action_size)
        Qsa = self.forward(state)
        loss = self.loss(Qsa, R, action_onehot)
        return self._train_step(loss)

    def forward(self, state):
        Qsa = self.Q(self.model(state))
        return Qsa

    def evaluate(self, state):
        with torch.no_grad():
            Qsa = self.forward(totorch(state, self.device))
        return Qsa.cpu().numpy()
