import numpy as np
import torch
import torch.nn.functional as F

from rlib.networks import Model, PPOConfig
from rlib.networks.networks import conv2d_outsize
from rlib.PPO.model import PPOModel
from rlib.utils.utils import tonumpy, totorch, totorch_many


class RewardForwardFilter:
    # https://github.com/openai/random-network-distillation
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems


class PPOIntrinsic(PPOModel):
    """Twin-critic PPO with extrinsic + intrinsic value heads."""

    def __init__(
        self,
        model,
        input_size,
        action_size,
        config: PPOConfig,
        *,
        extr_coeff: float = 2.0,
        intr_coeff: float = 1.0,
        build_optimiser: bool = True,
        optim: type[torch.optim.Optimizer] = torch.optim.Adam,
        optim_args: dict | None = None,
        **model_args,
    ):
        super().__init__(action_size=action_size, config=config)
        self.input_size = input_size
        self.extr_coeff = extr_coeff
        self.intr_coeff = intr_coeff

        self.model = model(input_size, **model_args).to(self.device)
        self.dense_size = dense_size = self.model.dense_size
        self.policy = torch.nn.Sequential(
            torch.nn.Linear(dense_size, action_size), torch.nn.Softmax(dim=-1)
        ).to(self.device)  # Actor
        self.Ve = torch.nn.Linear(dense_size, 1).to(self.device)  # Critic (Extrinsic)
        self.Vi = torch.nn.Linear(dense_size, 1).to(
            self.device
        )  # Intrinsic Value i.e. expected instrinsic value of state

        if build_optimiser:
            self._build_optimiser(optim=optim, optim_args=optim_args)

    def forward(self, state):
        state_enc = self.model(state)
        policy = self.policy(state_enc)
        value_extr = self.Ve(state_enc).view(-1)
        value_intr = self.Ve(state_enc).view(-1)
        return policy, value_extr, value_intr

    def evaluate(self, state):
        with torch.no_grad():
            policy, value_extr, value_intr = self.forward(totorch(state, self.device))
        return tonumpy(policy), tonumpy(value_extr), tonumpy(value_intr)

    def loss(
        self,
        policy,
        Re,
        Ri,
        Ve,
        Vi,
        Adv,
        action_onehot,
        old_policy,
    ):
        policy_loss, entropy = self.ppo_clipped_policy_loss(policy, old_policy, action_onehot, Adv)
        value_loss = self.extr_coeff * self.value_loss(Re, Ve) + self.intr_coeff * self.value_loss(
            Ri, Vi
        )
        return policy_loss + value_loss - self.entropy_coeff * entropy

    def backprop(self, state, Re, Ri, Adv, action, old_policy):
        state, action, Re, Ri, Adv, old_policy = totorch_many(
            state, action, Re, Ri, Adv, old_policy, device=self.device
        )
        action_onehot = F.one_hot(action.long(), self.action_size)
        policy, Ve, Vi = self.forward(state)
        loss = self.loss(policy, Re, Ri, Ve, Vi, Adv, action_onehot, old_policy)
        return self._train_step(loss)


class PredictorCNN(torch.nn.Module):
    def __init__(
        self,
        input_size,
        conv1_size=32,
        conv2_size=64,
        conv3_size=64,
        dense_size=512,
        padding=None,
        init_scale=np.sqrt(2),
        scale=True,
        trainable=True,
    ):
        # input_shape [channels, height, width]
        if padding is None:
            padding = (0, 0)
        super().__init__()
        self.scale = scale
        self.dense_size = dense_size
        self.input_size = input_size
        self.init_scale = init_scale
        self.h1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                input_size[0], conv1_size, kernel_size=(8, 8), stride=(4, 4), padding=padding
            ),
            torch.nn.LeakyReLU(),
        )
        self.h2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                conv1_size, conv2_size, kernel_size=(4, 4), stride=(2, 2), padding=padding
            ),
            torch.nn.LeakyReLU(),
        )
        self.h3 = torch.nn.Sequential(
            torch.nn.Conv2d(
                conv2_size, conv3_size, kernel_size=(3, 3), stride=(1, 1), padding=padding
            ),
            torch.nn.LeakyReLU(),
        )
        self.flatten = torch.nn.Flatten()
        c, h, w = self._conv_outsize()
        h * w * c
        if trainable:
            self.dense = torch.nn.Sequential(
                torch.nn.Linear(h * w * c, dense_size),
                torch.nn.ReLU(),
                torch.nn.Linear(dense_size, dense_size),
                torch.nn.ReLU(),
                torch.nn.Linear(dense_size, dense_size),
            )
        else:
            self.dense = torch.nn.Linear(h * w * c, dense_size)

        self.init_weights()
        self.set_trainable(trainable)

    def set_trainable(self, trainable):
        if not trainable:
            for param in self.parameters():
                param.requires_grad = False

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            torch.nn.init.orthogonal_(module.weight, gain=self.init_scale)

    def _conv_outsize(self):
        _, h, w = self.input_size
        h, w = conv2d_outsize(h, w, self.h1[0].kernel_size, self.h1[0].stride, self.h1[0].padding)
        h, w = conv2d_outsize(h, w, self.h2[0].kernel_size, self.h2[0].stride, self.h2[0].padding)
        h, w = conv2d_outsize(h, w, self.h3[0].kernel_size, self.h3[0].stride, self.h3[0].padding)
        return self.h3[0].out_channels, h, w

    def forward(self, x):
        x = x / 255 if self.scale else x
        x = self.h1(x)
        x = self.h2(x)
        x = self.h3(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x


class PredictorMLP(torch.nn.Module):
    def __init__(
        self,
        input_size,
        num_layers=2,
        dense_size=64,
        activation=torch.nn.LeakyReLU,
        init_scale=np.sqrt(2),
        trainable=True,
    ):
        # input_shape = feature_size
        super().__init__()
        self.dense_size = dense_size
        self.input_size = input_size
        self.init_scale = init_scale
        layers = []
        in_size = input_size
        for _l in range(num_layers):
            layers.append(torch.nn.Linear(in_size, dense_size))
            layers.append(activation())
            in_size = dense_size
        layers.append(torch.nn.Linear(dense_size, dense_size))
        self.layers = torch.nn.ModuleList(layers)

        self.init_weights()
        self.set_trainable(trainable)

    def set_trainable(self, trainable):
        if not trainable:
            for param in self.parameters():
                param.requires_grad = False

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            torch.nn.init.orthogonal_(module.weight, gain=self.init_scale)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class RND(Model):
    # EXPLORATION BY RANDOM NETWORK DISTILLATION
    # https://arxiv.org/pdf/1810.12894.pdf
    def __init__(
        self,
        policy_model,
        target_model,
        input_size,
        action_size,
        config: PPOConfig,
        *,
        intr_coeff: float = 0.5,
        extr_coeff: float = 1.0,
        policy_args: dict | None = None,
        RND_args: dict | None = None,
        optim: type[torch.optim.Optimizer] = torch.optim.Adam,
        optim_args: dict | None = None,
    ):
        if RND_args is None:
            RND_args = {}
        if policy_args is None:
            policy_args = {}
        super().__init__(config=config)
        self.intr_coeff = intr_coeff
        self.extr_coeff = extr_coeff
        self.entropy_coeff = config.entropy_coeff
        self.action_size = action_size

        target_size = (
            (1, input_size[1], input_size[2]) if len(input_size) == 3 else input_size
        )  # only use last frame in frame-stack for convolutions

        self.policy = PPOIntrinsic(
            policy_model,
            input_size,
            action_size,
            config=config,
            extr_coeff=extr_coeff,
            intr_coeff=intr_coeff,
            build_optimiser=False,
            **policy_args,
        )

        # randomly weighted and fixed neural network, acts as a random_id for each state
        self.target_model = target_model(target_size, trainable=False).to(config.device)

        # learns to predict target model
        # i.e. provides rewards based ability to predict a fixed random function, thus behaves as density map of explored areas
        self.predictor_model = target_model(target_size, trainable=True).to(config.device)

        self._build_optimiser(optim=optim, optim_args=optim_args)

    def forward(self, state):
        return self.policy.forward(state)

    def evaluate(self, state):
        return self.policy.evaluate(state)

    def _intr_reward(self, next_state, state_mean, state_std):
        norm_next_state = torch.clip((next_state - state_mean) / state_std, -5, 5)
        intr_reward = torch.square(
            self.predictor_model(norm_next_state) - self.target_model(norm_next_state).detach()
        ).sum(dim=-1)
        return intr_reward

    def intrinsic_reward(self, next_state: np.ndarray, state_mean: np.ndarray, state_std):
        next_state, state_mean, state_std = totorch_many(
            next_state, state_mean, state_std, device=self.device
        )
        with torch.no_grad():
            intr_reward = self._intr_reward(next_state, state_mean, state_std)
        return tonumpy(intr_reward)

    def backprop(
        self,
        state,
        next_state,
        R_extr,
        R_intr,
        Adv,
        actions,
        old_policy,
        state_mean,
        state_std,
    ):
        state, next_state, R_extr, R_intr, Adv, actions, old_policy, state_mean, state_std = (
            totorch_many(
                state,
                next_state,
                R_extr,
                R_intr,
                Adv,
                actions,
                old_policy,
                state_mean,
                state_std,
                device=self.device,
            )
        )
        policy, Ve, Vi = self.policy.forward(state)
        actions_onehot = F.one_hot(actions.long(), self.action_size)
        policy_loss = self.policy.loss(
            policy, R_extr, R_intr, Ve, Vi, Adv, actions_onehot, old_policy
        )

        predictor_loss = self._intr_reward(next_state, state_mean, state_std).mean()
        loss = policy_loss + predictor_loss
        return self._train_step(loss)
