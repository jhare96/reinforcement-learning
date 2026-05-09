import numpy as np
import torch
import torch.nn.functional as F

from rlib.agent import Agent
from rlib.PPO.model import PPOConfig
from rlib.RND.model import PPOIntrinsic
from rlib.utils.utils import tonumpy, totorch, totorch_many


def sign(x):
    if x < 0:
        return 2
    elif x == 0:
        return 0
    elif x > 0:
        return 1
    else:
        raise ValueError


class RANDAL(Agent):
    def __init__(
        self,
        policy_model,
        target_model,
        input_size,
        action_size,
        config: PPOConfig,
        *,
        pixel_control: bool = True,
        intr_coeff: float = 0.5,
        extr_coeff: float = 1.0,
        RP: float = 1,
        VR: float = 1,
        PC: float = 1,
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
        self.entropy_coeff = config.entropy_coeff
        self.intr_coeff = intr_coeff
        self.extr_coeff = extr_coeff
        self.pixel_control = pixel_control
        self.action_size = action_size
        self.RP = RP  # reward prediction
        self.VR = VR  # value replay
        self.PC = PC  # pixel control

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

        device = config.device
        target_size = (
            (1, input_size[1], input_size[2]) if len(input_size) == 3 else input_size
        )  # only use last frame in frame-stack for convolutions

        # randomly weighted and fixed neural network, acts as a random_id for each state
        self.target_model = target_model(target_size, trainable=False, **RND_args).to(device)

        # learns to predict target model
        # i.e. provides rewards based ability to predict a fixed random function, thus behaves as density map of explored areas
        self.predictor_model = target_model(target_size, trainable=True, **RND_args).to(device)

        if pixel_control:
            self.feat_map = torch.nn.Sequential(
                torch.nn.Linear(self.policy.dense_size, 32 * 8 * 8), torch.nn.ReLU()
            ).to(device)
            self.deconv1 = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(32, 32, kernel_size=(3, 3), stride=(1, 1)), torch.nn.ReLU()
            ).to(device)
            self.deconv_advantage = torch.nn.ConvTranspose2d(
                32, action_size, kernel_size=(3, 3), stride=(2, 2)
            ).to(device)
            self.deconv_value = torch.nn.ConvTranspose2d(
                32, 1, kernel_size=(3, 3), stride=(2, 2)
            ).to(device)

        # reward model
        self.r1 = torch.nn.Sequential(
            torch.nn.Linear(self.policy.dense_size, 128), torch.nn.ReLU()
        ).to(device)
        self.r2 = torch.nn.Linear(128, 3).to(device)

        self._build_optimiser(optim=optim, optim_args=optim_args)

    def forward(self, state):
        return self.policy.forward(state)

    def evaluate(self, state):
        return self.policy.evaluate(state)

    def Qaux(self, enc_state):
        # Auxillary Q value calculated via dueling network
        # Z. Wang, N. de Freitas, and M. Lanctot. Dueling Network Architectures for Deep ReinforcementLearning. https://arxiv.org/pdf/1511.06581.pdf
        batch_size = enc_state.shape[0]
        feat_map = self.feat_map(enc_state).view([batch_size, 32, 8, 8])
        deconv1 = self.deconv1(feat_map)
        deconv_adv = self.deconv_advantage(deconv1)
        deconv_value = self.deconv_value(deconv1)
        qaux = deconv_value + deconv_adv - torch.mean(deconv_adv, dim=1, keepdim=True)
        return qaux

    def get_pixel_control(self, state: np.ndarray):
        with torch.no_grad():
            enc_state = self.policy.model(totorch(state, self.device))
            Qaux = self.Qaux(enc_state)
        return tonumpy(Qaux)

    def pixel_loss(self, Qaux, Qaux_actions, Qaux_target):
        'Qaux_target temporal difference target for Q_aux'
        one_hot_actions = F.one_hot(Qaux_actions.long(), self.action_size)
        pixel_action = one_hot_actions.view([-1, self.action_size, 1, 1])
        Q_aux_action = torch.sum(Qaux * pixel_action, dim=1)
        pixel_loss = 0.5 * torch.mean(
            torch.square(Qaux_target - Q_aux_action)
        )  # l2 loss for Q_aux over all pixels and batch
        return pixel_loss

    def reward_loss(self, reward_states, reward_target):
        r1 = self.r1(self.policy.model(reward_states))
        pred_reward = self.r2(r1)
        reward_loss = torch.mean(
            F.cross_entropy(pred_reward, reward_target.long())
        )  # cross entropy over caterogical reward
        return reward_loss

    def replay_loss(self, R, V):
        return torch.mean(torch.square(R - V))

    def forward_loss(self, states, actions, Re, Ri, Adv, old_policy):
        states, actions, Re, Ri, Adv, old_policy = totorch_many(
            states, actions, Re, Ri, Adv, old_policy, device=self.device
        )
        actions_onehot = F.one_hot(actions.long(), self.action_size)
        policy, Ve, Vi = self.forward(states)
        forward_loss = self.policy.loss(policy, Re, Ri, Ve, Vi, Adv, actions_onehot, old_policy)
        return forward_loss

    def auxiliary_loss(
        self,
        reward_states,
        rewards,
        Qaux_target,
        Qaux_actions,
        replay_states,
        replay_R,
    ):
        reward_states, rewards, Qaux_target, Qaux_actions, replay_states, replay_R = totorch_many(
            reward_states,
            rewards,
            Qaux_target,
            Qaux_actions,
            replay_states,
            replay_R,
            device=self.device,
        )

        policy_enc = self.policy.model(replay_states)
        replay_values = self.policy.Ve(policy_enc)
        reward_loss = self.reward_loss(reward_states, rewards)
        replay_loss = self.replay_loss(replay_R, replay_values)
        aux_loss = self.RP * reward_loss + self.VR * replay_loss

        Qaux_actions = Qaux_actions.long()

        if self.pixel_control:
            Qaux = self.Qaux(policy_enc)
            pixel_loss = self.pixel_loss(Qaux, Qaux_actions, Qaux_target)
            aux_loss += self.PC * pixel_loss

        return aux_loss

    def predictor_loss(self, next_states, state_mean, state_std):
        'loss for predictor network'
        next_states, state_mean, state_std = totorch_many(
            next_states, state_mean, state_std, device=self.device
        )
        predictor_loss = self._intr_reward(next_states, state_mean, state_std).mean()
        return predictor_loss

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
        states,
        next_states,
        Re,
        Ri,
        Adv,
        actions,
        old_policy,
        reward_states,
        rewards,
        Qaux_target,
        Qaux_actions,
        replay_states,
        replay_R,
        state_mean,
        state_std,
    ):
        forward_loss = self.forward_loss(states, actions, Re, Ri, Adv, old_policy)
        aux_losses = self.auxiliary_loss(
            reward_states, rewards, Qaux_target, Qaux_actions, replay_states, replay_R
        )
        predictor_loss = self.predictor_loss(next_states, state_mean, state_std)

        loss = forward_loss + aux_losses + predictor_loss
        return self._train_step(loss)
