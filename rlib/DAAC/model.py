import numpy as np
import torch
import torch.nn.functional as F

from rlib.networks import Model, ModelConfig, PPOConfig
from rlib.PPO.model import PPOModel
from rlib.utils.utils import tonumpy, totorch, totorch_many


class ValueModel(Model):
    def __init__(
        self,
        model,
        input_shape,
        action_size,
        config: ModelConfig,
        *,
        build_optimiser: bool = True,
        optim: type[torch.optim.Optimizer] = torch.optim.Adam,
        optim_args: dict | None = None,
        **model_args,
    ):
        super().__init__(config=config)
        self.action_size = action_size

        self.model = model(input_shape, **model_args).to(self.device)
        dense_size = self.model.dense_size
        self.V = torch.nn.Linear(dense_size, 1).to(self.device)

        if build_optimiser:
            self._build_optimiser(optim=optim, optim_args=optim_args)

    def forward(self, state):
        enc_state = self.model(state)
        value = self.V(enc_state).view(-1)
        return value

    def evaluate(self, state):
        with torch.no_grad():
            value = self.forward(totorch(state, self.device))
        return tonumpy(value)

    def loss(self, V, R):
        return self.value_loss(R, V)

    def backprop(self, state, R):
        state, R = totorch_many(state, R, device=self.device)
        value = self.forward(state)
        loss = self.loss(value, R)
        return self._train_step(loss)


class PolicyModel(PPOModel):
    """DAAC policy head: clipped PPO objective with a separate advantage prediction."""

    def __init__(
        self,
        model,
        input_shape,
        action_size,
        config: PPOConfig,
        *,
        adv_coeff: float = 0.25,
        build_optimiser: bool = True,
        optim: type[torch.optim.Optimizer] = torch.optim.Adam,
        optim_args: dict | None = None,
        **model_args,
    ):
        super().__init__(action_size=action_size, config=config)
        self.adv_coeff = adv_coeff

        self.model = model(input_shape, **model_args).to(self.device)
        dense_size = self.model.dense_size
        self.policy = torch.nn.Sequential(
            torch.nn.Linear(dense_size, action_size), torch.nn.Softmax(dim=-1)
        ).to(self.device)
        self.Adv = torch.nn.Linear(dense_size, 1).to(self.device)

        if build_optimiser:
            self._build_optimiser(optim=optim, optim_args=optim_args)

    def forward(self, state):
        enc_state = self.model(state)
        policy = self.policy(enc_state)
        Adv = self.Adv(enc_state).view(-1)
        return policy, Adv

    def evaluate(self, state):
        with torch.no_grad():
            policy, Adv = self.forward(totorch(state, self.device))
        return tonumpy(policy), tonumpy(Adv)

    def loss(self, policy, Adv_hat, Adv, action_onehot, old_policy):
        policy_loss, entropy = self.ppo_clipped_policy_loss(policy, old_policy, action_onehot, Adv)
        adv_loss = torch.square(Adv_hat - Adv).sum(dim=-1).mean()
        return policy_loss - self.entropy_coeff * entropy + self.adv_coeff * adv_loss

    def backprop(self, state, Adv, action, old_policy):
        state, action, Adv, old_policy = totorch_many(
            state, action, Adv, old_policy, device=self.device
        )
        policy, Adv_hat = self.forward(state)
        action_onehot = F.one_hot(action.long(), self.action_size)
        loss = self.loss(policy, Adv_hat, Adv, action_onehot, old_policy)
        return self._train_step(loss)


class DAAC(Model):
    # Decoupling Value and Policy for Generalization in Reinforcement Learning
    # https://arxiv.org/pdf/2102.10330.pdf
    def __init__(
        self,
        policy_model,
        value_model,
        input_shape,
        action_size,
        config: PPOConfig,
        *,
        adv_coeff: float = 0.25,
        policy_optim: type[torch.optim.Optimizer] = torch.optim.Adam,
        policy_optim_args: dict | None = None,
        policy_model_args: dict | None = None,
        value_optim: type[torch.optim.Optimizer] = torch.optim.Adam,
        value_optim_args: dict | None = None,
        value_model_args: dict | None = None,
    ):
        if value_model_args is None:
            value_model_args = {}
        if value_optim_args is None:
            value_optim_args = {}
        if policy_model_args is None:
            policy_model_args = {}
        if policy_optim_args is None:
            policy_optim_args = {}
        super().__init__(config=config)
        self.entropy_coeff = config.entropy_coeff
        self.adv_coeff = adv_coeff
        self.policy_clip = config.policy_clip

        self.value = ValueModel(
            value_model,
            input_shape,
            action_size,
            config=config,
            optim=value_optim,
            optim_args=value_optim_args,
            **value_model_args,
        )

        self.policy = PolicyModel(
            policy_model,
            input_shape,
            action_size,
            config=config,
            adv_coeff=adv_coeff,
            optim=policy_optim,
            optim_args=policy_optim_args,
            **policy_model_args,
        )

    def get_policy(self, state: np.ndarray):
        return self.policy.evaluate(state)

    def get_value(self, state: np.ndarray):
        return self.value.evaluate(state)

    def evaluate(self, state: np.ndarray):
        with torch.no_grad():
            policy, _ = self.policy.forward(totorch(state, self.policy.device))
            value = self.value.forward(totorch(state, self.value.device))
        return tonumpy(policy), tonumpy(value)

    def backprop(self, state, R, Adv, action, old_policy):
        # ``policy`` and ``value`` each own their own optimiser/scheduler,
        # so DAAC simply delegates and sums the two scalar losses.
        policy_loss = self.policy.backprop(state, Adv, action, old_policy)
        value_loss = self.value.backprop(state, R)
        return policy_loss + value_loss
