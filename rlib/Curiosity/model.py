"""Intrinsic Curiosity Module agent."""

import torch
import torch.nn.functional as F

from rlib.A2C.model import ActorCritic
from rlib.networks import A2CConfig, Model
from rlib.utils.utils import totorch_many


class ICM(torch.nn.Module):
    def __init__(
        self,
        model_head,
        input_size,
        action_size,
        forward_coeff,
        device='cuda',
        **model_head_args,
    ):
        super().__init__()
        self.action_size = action_size
        self.forward_coeff = forward_coeff
        self.phi = model_head(input_size, **model_head_args)
        dense_size = self.phi.dense_size
        self.device = device

        # forward model
        self.forward1 = torch.nn.Sequential(
            torch.nn.Linear(dense_size + action_size, dense_size), torch.nn.ReLU()
        ).to(device)
        self.pred_state = torch.nn.Linear(dense_size, dense_size).to(device)

        # inverse model
        self.inverse1 = torch.nn.Sequential(
            torch.nn.Linear(dense_size * 2, dense_size), torch.nn.ReLU()
        ).to(device)
        self.pred_action = torch.nn.Sequential(
            torch.nn.Linear(dense_size * 2, dense_size), torch.nn.ReLU()
        ).to(device)

    def intr_reward(self, phi, action_onehot, phi_next):
        f1 = self.forward1(torch.cat([phi, action_onehot], dim=1))
        phi_pred = self.pred_state(f1)
        intr_reward = 0.5 * torch.sum(
            torch.square(phi_pred - phi_next), dim=1
        )  # l2 distance metric ‖ˆφ(st+1)−φ(st+1)‖22
        return intr_reward

    def predict_action(self, phi1, phi2):
        phi_cat = torch.cat([phi1, phi2], dim=1)
        pred_action = self.pred_action(phi_cat)
        return pred_action

    def get_intr_reward(self, state, action, next_state):
        state, next_state, action = totorch_many(state, next_state, action, device=self.device)
        action = action.long()
        phi1 = self.phi(state)
        phi2 = self.phi(next_state)
        action_onehot = F.one_hot(action, self.action_size)
        with torch.no_grad():
            intr_reward = self.intr_reward(phi1, action_onehot, phi2)
        return intr_reward.cpu().numpy()

    def get_pred_action(self, state, next_state):
        state, next_state = totorch_many(state, next_state, device=self.device)
        return self.pred_action(state, next_state)

    def loss(self, state, action, next_state):
        action = action.long()
        phi1 = self.phi(state)
        phi2 = self.phi(next_state)
        action_onehot = F.one_hot(action, self.action_size)

        forward_loss = torch.mean(self.intr_reward(phi1, action_onehot, phi2))
        inverse_loss = F.cross_entropy(self.predict_action(phi1, phi2), action)
        return (1 - self.forward_coeff) * inverse_loss + self.forward_coeff * forward_loss


class Curiosity(Model):
    def __init__(
        self,
        policy_model,
        ICM_model,
        input_size,
        action_size,
        config: A2CConfig,
        *,
        forward_coeff: float,
        policy_importance: float,
        reward_scale: float,
        policy_args: dict | None = None,
        ICM_args: dict | None = None,
    ):
        if ICM_args is None:
            ICM_args = {}
        if policy_args is None:
            policy_args = {}
        super().__init__(config=config)
        self.reward_scale = reward_scale
        self.forward_coeff = forward_coeff
        self.policy_importance = policy_importance
        self.entropy_coeff = config.entropy_coeff
        self.action_size = action_size

        try:
            iter(input_size)
        except TypeError:
            input_size = (input_size,)

        self.ICM = ICM(
            ICM_model, input_size, action_size, forward_coeff, device=config.device, **ICM_args
        )
        self.AC = ActorCritic(
            policy_model,
            input_size,
            action_size,
            config=config,
            build_optimiser=False,
            **policy_args,
        )

        self._build_optimiser(optim=torch.optim.RMSprop)

    def forward(self, state):
        return self.AC.forward(state)

    def evaluate(self, state):
        return self.AC.evaluate(state)

    def intrinsic_reward(self, state, action, next_state):
        return self.ICM.get_intr_reward(state, action, next_state)

    def backprop(
        self,
        state,
        next_state,
        R,
        Adv,
        action,
        state_mean,
        state_std,
    ):
        state, next_state, R, Adv, action, state_mean, state_std = totorch_many(
            state, next_state, R, Adv, action, state_mean, state_std, device=self.device
        )
        policy, value = self.AC.forward(state)
        action_onehot = F.one_hot(action.long(), self.action_size)
        policy_loss = self.AC.loss(policy, R, value, action_onehot)
        ICM_loss = self.ICM.loss(
            (state - state_mean) / state_std, action, (next_state - state_mean) / state_std
        )
        loss = self.policy_importance * policy_loss + self.reward_scale * ICM_loss
        return self._train_step(loss)
