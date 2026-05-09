from dataclasses import dataclass

import torch
import torch.nn.functional as F

from rlib.agent import Agent, ModelConfig
from rlib.utils.utils import tonumpy, totorch, totorch_many


@dataclass(frozen=True)
class PPOConfig(ModelConfig):
    """Hyperparameters for clipped-objective PPO-family agents (PPO / RND / DAAC policy).

    Attributes:
        entropy_coeff: Coefficient on the policy entropy bonus.
        policy_clip: Clipping parameter for PPO's clipped objective.
    """

    entropy_coeff: float = 0.01
    policy_clip: float = 0.1


class PPOModel(Agent):
    """PPO-family base class: defines the clipped-objective policy loss.

    Concrete subclasses (single-critic PPO, twin-critic PPOIntrinsic,
    DAAC's policy head, ...) only need to implement ``forward``,
    ``evaluate`` and ``backprop``; they all share the same clipped
    policy loss + entropy bonus via :meth:`ppo_clipped_policy_loss`.
    """

    config: PPOConfig

    def __init__(self, action_size: int, config: PPOConfig) -> None:
        super().__init__(config=config)
        self.action_size = action_size
        self.entropy_coeff = config.entropy_coeff
        self.policy_clip = config.policy_clip

    def ppo_clipped_policy_loss(self, policy, old_policy, action_onehot, advantage):
        """PPO clipped-objective policy loss + entropy bonus.

        Returns the ``(policy_loss, entropy)`` pair so subclasses can
        combine them with whatever value-function loss(es) and
        coefficients their agent uses.
        """
        policy_actions = torch.sum(policy * action_onehot, dim=1)
        old_policy_actions = torch.sum(old_policy * action_onehot, dim=1)
        ratio = policy_actions / old_policy_actions
        unclipped = ratio * -advantage
        clipped = torch.clip(ratio, 1 - self.policy_clip, 1 + self.policy_clip) * -advantage
        policy_loss = torch.mean(torch.maximum(unclipped, clipped))
        entropy = torch.mean(torch.sum(policy * -torch.log(policy), dim=1))
        return policy_loss, entropy


class PPO(PPOModel):
    """Single-critic PPO actor-critic."""

    def __init__(
        self,
        model,
        input_shape,
        action_size,
        config: PPOConfig,
        *,
        value_coeff: float = 1.0,
        build_optimiser: bool = True,
        optim: type[torch.optim.Optimizer] = torch.optim.Adam,
        optim_args: dict | None = None,
        **model_args,
    ):
        super().__init__(action_size=action_size, config=config)
        self.value_coeff = value_coeff

        self.model = model(input_shape, **model_args).to(self.device)
        dense_size = self.model.dense_size
        self.policy = torch.nn.Sequential(
            torch.nn.Linear(dense_size, action_size), torch.nn.Softmax(dim=-1)
        ).to(self.device)
        self.V = torch.nn.Linear(dense_size, 1).to(self.device)

        if build_optimiser:
            self._build_optimiser(optim=optim, optim_args=optim_args)

    def forward(self, state):
        state_enc = self.model(state)
        policy = self.policy(state_enc)
        value = self.V(state_enc).view(-1)
        return policy, value

    def evaluate(self, state):
        with torch.no_grad():
            policy, value = self.forward(totorch(state, self.device))
        return tonumpy(policy), tonumpy(value)

    def loss(self, policy, R, V, Adv, action_onehot, old_policy):
        policy_loss, entropy = self.ppo_clipped_policy_loss(policy, old_policy, action_onehot, Adv)
        value_loss = self.value_loss(R, V)
        return policy_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy

    def backprop(self, state, R, Adv, action, old_policy):
        state, action, R, Adv, old_policy = totorch_many(
            state, action, R, Adv, old_policy, device=self.device
        )
        action_onehot = F.one_hot(action.long(), self.action_size)
        policy, value = self.forward(state)
        loss = self.loss(policy, R, value, Adv, action_onehot, old_policy)
        return self._train_step(loss)
