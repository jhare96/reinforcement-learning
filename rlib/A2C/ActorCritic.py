"""A2C-family agent models.

The class hierarchy mirrors the user-facing pattern requested for rlib:

* :class:`rlib.networks.Model` — abstract base providing the shared
  optimiser/scheduler/training-step boilerplate.
* :class:`A2CModel` — abstract intermediate that encodes the **A2C
  loss** (advantage actor-critic with entropy bonus) so every A2C-style
  variant inherits it for free.
* :class:`ActorCritic` — concrete feed-forward A2C model.
* :class:`ActorCritic_LSTM` — concrete recurrent A2C variant.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from rlib.networks import Model
from rlib.networks.networks import MaskedLSTMBlock
from rlib.utils.utils import tonumpy, tonumpy_many, totorch, totorch_many


class A2CModel(Model):
    """A2C-family base class: defines the actor-critic + entropy loss.

    Concrete subclasses (feed-forward, recurrent, ...) only need to
    implement ``forward``, :meth:`evaluate` and :meth:`backprop`; they
    all share the same loss function via :meth:`loss`.
    """

    def __init__(
        self,
        action_size: int,
        entropy_coeff: float = 0.01,
        value_coeff: float = 0.5,
        lr: float = 1e-3,
        lr_final: float = 1e-6,
        decay_steps: float = 6e5,
        grad_clip: float | None = 0.5,
        device: str = "cuda",
    ):
        super().__init__(
            lr=lr,
            lr_final=lr_final,
            decay_steps=decay_steps,
            grad_clip=grad_clip,
            device=device,
        )
        self.action_size = action_size
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff

    def loss(
        self,
        policy: torch.Tensor,
        R: torch.Tensor,
        V: torch.Tensor,
        actions_onehot: torch.Tensor,
    ) -> torch.Tensor:
        """Standard A2C/A3C actor–critic loss with entropy bonus.

        Combines:

        * a half-MSE value loss on ``R - V``,
        * the negative log-likelihood policy gradient using a *detached*
          advantage, and
        * an entropy bonus on the action distribution.

        ``policy`` is expected to be a normalised distribution
        (i.e. the output of a softmax); we numerically clip it before
        taking ``log`` so the loss stays finite for near-deterministic
        policies.
        """
        advantage = R - V
        value_loss = self.value_loss(R, V)

        log_policy = torch.log(torch.clip(policy, 1e-6, 0.999999))
        log_policy_actions = torch.sum(log_policy * actions_onehot, dim=1)
        policy_loss = torch.mean(-log_policy_actions * advantage.detach())

        entropy = torch.mean(torch.sum(policy * -log_policy, dim=1))
        return policy_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy


class ActorCritic(A2CModel):
    """Feed-forward A2C actor-critic."""

    def __init__(
        self,
        model,
        input_size,
        action_size,
        entropy_coeff=0.01,
        value_coeff=0.5,
        lr=1e-3,
        lr_final=1e-6,
        decay_steps=6e5,
        grad_clip=0.5,
        build_optimiser=True,
        optim=torch.optim.RMSprop,
        optim_args=None,
        device='cuda',
        **model_args,
    ):
        super().__init__(
            action_size=action_size,
            entropy_coeff=entropy_coeff,
            value_coeff=value_coeff,
            lr=lr,
            lr_final=lr_final,
            decay_steps=decay_steps,
            grad_clip=grad_clip,
            device=device,
        )

        self.model = model(input_size, **model_args).to(self.device)
        self.dense_size = self.model.dense_size
        self.policy_distrib = torch.nn.Linear(self.dense_size, action_size).to(self.device)  # Actor
        self.V = torch.nn.Linear(self.dense_size, 1).to(self.device)  # Critic

        if build_optimiser:
            self._build_optimiser(optim=optim, optim_args=optim_args)

    def forward(self, state):
        enc_state = self.model(state)
        policy = F.softmax(self.policy_distrib(enc_state), dim=-1)
        value = self.V(enc_state).view(-1)
        return policy, value

    def evaluate(self, state: np.ndarray):
        state = totorch(state, self.device)
        with torch.no_grad():
            policy, value = self.forward(state)
        return tonumpy(policy), tonumpy(value)

    def backprop(self, state, R, action):
        state, R, action = totorch_many(state, R, action, device=self.device)
        action_onehot = F.one_hot(action.long(), num_classes=self.action_size)
        policy, value = self.forward(state)
        loss = self.loss(policy, R, value, action_onehot)
        return self._train_step(loss)


class ActorCritic_LSTM(A2CModel):
    """Recurrent A2C actor-critic (masked LSTM body)."""

    def __init__(
        self,
        model,
        input_size,
        action_size,
        cell_size,
        entropy_coeff=0.01,
        value_coeff=0.5,
        lr=1e-3,
        lr_final=1e-6,
        decay_steps=6e5,
        grad_clip=0.5,
        build_optimiser=True,
        optim=torch.optim.RMSprop,
        optim_args=None,
        device='cuda',
        **model_args,
    ):
        super().__init__(
            action_size=action_size,
            entropy_coeff=entropy_coeff,
            value_coeff=value_coeff,
            lr=lr,
            lr_final=lr_final,
            decay_steps=decay_steps,
            grad_clip=grad_clip,
            device=device,
        )
        self.input_size = input_size
        self.cell_size = cell_size

        self.model = model(input_size, **model_args).to(self.device)
        self.dense_size = self.model.dense_size
        # self.lstm = MaskedRNN(MaskedLSTMCell(cell_size, self.dense_size), time_major=True)
        self.lstm = MaskedLSTMBlock(self.dense_size, cell_size, time_major=True).to(self.device)

        self.policy_distrib = torch.nn.Linear(cell_size, action_size, device=self.device)  # Actor
        self.V = torch.nn.Linear(cell_size, 1, device=self.device)  # Critic

        if build_optimiser:
            self._build_optimiser(optim=optim, optim_args=optim_args)

    def forward(self, state, hidden=None, done=None):
        T, num_envs = state.shape[:2]
        folded_state = state.view(-1, *self.input_size)
        enc_state = self.model(folded_state)
        folded_enc_state = enc_state.view(T, num_envs, self.dense_size)
        lstm_outputs, hidden = self.lstm(folded_enc_state, hidden, done)
        policy = F.softmax(self.policy_distrib(lstm_outputs), dim=-1).view(-1, self.action_size)
        value = self.V(lstm_outputs).view(-1)
        return policy, value, hidden

    def evaluate(self, state: np.ndarray, hidden: np.ndarray | None = None, done=None):
        state = totorch(state, self.device)
        hidden = totorch_many(*hidden, device=self.device) if hidden is not None else None
        with torch.no_grad():
            policy, value, hidden = self.forward(state, hidden, done)
        return tonumpy(policy), tonumpy(value), tonumpy_many(*hidden)

    def backprop(self, state, R, action, hidden, done):
        state, R, action, done = totorch_many(state, R, action, done, device=self.device)
        hidden = totorch_many(*hidden, device=self.device)
        action_onehot = F.one_hot(action.long(), num_classes=self.action_size)
        policy, value, hidden = self.forward(state, hidden, done)
        loss = self.loss(policy, R, value, action_onehot)
        return self._train_step(loss)

    def get_initial_hidden(self, batch_size):
        return np.zeros((1, batch_size, self.cell_size)), np.zeros((1, batch_size, self.cell_size))

    def mask_hidden(self, hidden, dones):
        mask = (1 - dones).reshape(-1, 1)
        return (hidden[0] * mask, hidden[1] * mask)
