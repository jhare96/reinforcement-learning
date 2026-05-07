import contextlib
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from rlib.A2C.model import ActorCritic
from rlib.networks import A2CConfig, Model
from rlib.utils import TrainerConfig
from rlib.utils.SyncMultiEnvTrainer import SyncMultiEnvTrainer
from rlib.utils.utils import (
    GAE,
    RunningMeanStd,
    fastsample,
    fold_batch,
    stack_many,
    tonumpy,
    totorch,
    totorch_many,
)

# A2C-CNN version of Unsupervised Reinforcement Learning with Auxiliary Tasks (UNREAL) https://arxiv.org/abs/1611.05397
# Modifications:
#   no action-reward fed into policy
#   Use greyscaled images
#   deconvolute to pixel grid that overlaps FULL image
#   Generalised Advantage Estimation
#   Assumes input image size is 84x84

# torch.backends.cudnn.benchmark=True


def sign(x):
    if x < 0:
        return 2
    elif x == 0:
        return 0
    elif x > 0:
        return 1
    else:
        raise ValueError


class UnrealA2C2(Model):
    def __init__(
        self,
        policy_model,
        input_shape,
        action_size,
        config: A2CConfig,
        *,
        pixel_control: bool = True,
        RP: float = 1.0,
        PC: float = 1.0,
        VR: float = 1.0,
        policy_args: dict | None = None,
        optim: type[torch.optim.Optimizer] = torch.optim.RMSprop,
        optim_args: dict | None = None,
    ):
        if policy_args is None:
            policy_args = {}
        super().__init__(config=config)
        self.RP, self.PC, self.VR = RP, PC, VR
        self.entropy_coeff = config.entropy_coeff
        self.value_coeff = config.value_coeff
        self.pixel_control = pixel_control
        self.action_size = action_size

        with contextlib.suppress(TypeError):
            iter(input_shape)

        self.policy = ActorCritic(
            policy_model,
            input_shape,
            action_size,
            config=config,
            build_optimiser=False,
            **policy_args,
        )

        device = config.device
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
        # Qaux_target temporal difference target for Q_aux
        # print('max qaux actions', Qaux_actions)
        # print('action_size', self.action_size)
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

    def forward_loss(self, states, R, actions):
        states, R, actions = totorch_many(states, R, actions, device=self.device)
        actions_onehot = F.one_hot(actions.long(), num_classes=self.action_size)
        policies, values = self.forward(states)
        forward_loss = self.policy.loss(policies, R, values, actions_onehot)
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
        replay_values = self.policy.V(policy_enc)
        reward_loss = self.reward_loss(reward_states, rewards)
        replay_loss = self.replay_loss(replay_R, replay_values)
        aux_loss = self.RP * reward_loss + self.VR * replay_loss

        Qaux_actions = Qaux_actions.long()

        if self.pixel_control:
            Qaux = self.Qaux(policy_enc)
            pixel_loss = self.pixel_loss(Qaux, Qaux_actions, Qaux_target)
            aux_loss += self.PC * pixel_loss

        return aux_loss

    def backprop(
        self,
        states,
        R,
        actions,
        reward_states,
        rewards,
        Qaux_target,
        Qaux_actions,
        replay_states,
        replay_R,
    ):
        forward_loss = self.forward_loss(states, R, actions)
        aux_losses = self.auxiliary_loss(
            reward_states, rewards, Qaux_target, Qaux_actions, replay_states, replay_R
        )

        loss = forward_loss + aux_losses
        return self._train_step(loss)


@dataclass(frozen=True)
class UnrealTrainerConfig(TrainerConfig):
    """Hyperparameters for the feed-forward :class:`UnrealTrainer`."""

    normalise_obs: bool = True
    replay_length: int = 2000


class UnrealTrainer(SyncMultiEnvTrainer):
    """Trainer for the feed-forward UNREAL agent."""

    def __init__(
        self,
        envs,
        model: UnrealA2C2,
        val_envs,
        config: UnrealTrainerConfig,
    ):
        super().__init__(envs, model, val_envs, config=config)

        self.replay = deque([], maxlen=config.replay_length)  # replay length per actor
        self.action_size = self.model.action_size
        self.normalise_obs = config.normalise_obs

        if self.normalise_obs:
            self.obs_running = RunningMeanStd()
            self.state_mean = np.zeros_like(self.states)
            self.state_std = np.ones_like(self.states)
            self.aux_reward_rolling = RunningMeanStd()

    def populate_memory(self):
        for _t in range(2000 // self.nsteps):
            states, *_ = self.rollout()
            # self.state_mean, self.state_std = self.obs_running.update(fold_batch(states)[...,-1:])
            self.update_minmax(states)

    def update_minmax(self, obs):
        minima = obs.min()
        maxima = obs.max()
        if minima < self.state_min:
            self.state_min = minima
        if maxima > self.state_max:
            self.state_max = maxima

    def norm_obs(self, obs):
        '''normalise pixel intensity changes by recording min and max pixel observations
        not using per pixel normalisation because expected image is singular greyscale frame
        '''
        return (obs - self.state_min) * (1 / (self.state_max - self.state_min))

    def auxiliary_target(self, pixel_rewards, last_values, dones):
        T = len(pixel_rewards)
        R = np.zeros((T, *last_values.shape))
        dones = dones[:, :, np.newaxis, np.newaxis]
        R[-1] = last_values * (1 - dones[-1])

        for i in reversed(range(T - 1)):
            # restart score if done as BatchEnv automatically resets after end of episode
            R[i] = pixel_rewards[i] + 0.99 * R[i + 1] * (1 - dones[-1])

        return R

    def pixel_rewards(self, prev_state, states):
        # states of rank [T, B, channels, 84, 84]
        T = len(states)  # time length
        B = states.shape[1]  # batch size
        pixel_rewards = np.zeros((T, B, 21, 21))
        states = states[:, :, -1, :, :]
        prev_state = prev_state[:, -1, :, :]
        if self.normalise_obs:
            states = self.norm_obs(states)
            # print('states, max', states.max(), 'min', states.min(), 'mean', states.mean())
            prev_state = self.norm_obs(prev_state)

        pixel_rewards[0] = (
            np.abs(states[0] - prev_state).reshape(-1, 4, 4, 21, 21).mean(axis=(1, 2))
        )
        for i in range(1, T):
            pixel_rewards[i] = (
                np.abs(states[i] - states[i - 1]).reshape(-1, 4, 4, 21, 21).mean(axis=(1, 2))
            )
        # print('pixel reward',pixel_rewards.shape, 'max', pixel_rewards.max(), 'mean', pixel_rewards.mean())
        return pixel_rewards

    def sample_replay(self):
        workers = np.random.choice(
            self.num_envs, replace=False, size=2
        )  # randomly sample from one of n workers
        sample_start = np.random.randint(1, len(self.replay) - self.nsteps - 2)
        replay_sample = []
        for i in range(sample_start, sample_start + self.nsteps):
            replay_sample.append(self.replay[i])

        replay_states = np.stack([replay_sample[i][0][workers] for i in range(len(replay_sample))])
        replay_actions = np.stack([replay_sample[i][1][workers] for i in range(len(replay_sample))])
        replay_rewards = np.stack([replay_sample[i][2][workers] for i in range(len(replay_sample))])
        replay_values = np.stack([replay_sample[i][3][workers] for i in range(len(replay_sample))])
        replay_dones = np.stack([replay_sample[i][4][workers] for i in range(len(replay_sample))])
        # print('replay dones shape', replay_dones.shape)
        # print('replay_values shape', replay_values.shape)

        next_state = self.replay[sample_start + self.nsteps][0][workers]  # get state
        _, replay_last_values = self.model.evaluate(next_state)
        replay_R = (
            GAE(
                replay_rewards,
                replay_values,
                replay_last_values,
                replay_dones,
                gamma=0.99,
                lambda_=0.95,
            )
            + replay_values
        )

        if self.model.pixel_control:
            prev_states = self.replay[sample_start - 1][0][workers]
            Qaux_value = self.model.get_pixel_control(next_state)
            pixel_rewards = self.pixel_rewards(prev_states, replay_states)
            Qaux_target = self.auxiliary_target(
                pixel_rewards, np.max(Qaux_value, axis=1), replay_dones
            )
        else:
            Qaux_target = np.zeros(
                (len(replay_states), 1, 1, 1)
            )  # produce fake Qaux to save writing unecessary code

        return (
            fold_batch(replay_states),
            fold_batch(replay_actions),
            fold_batch(replay_R),
            fold_batch(Qaux_target),
            fold_batch(replay_dones),
        )
        # return replay_states, replay_actions, replay_R, Qaux_target, replay_dones

    def sample_reward(self):
        # worker = np.random.randint(0,self.num_envs) # randomly sample from one of n workers
        replay_rewards = np.array([self.replay[i][2] for i in range(len(self.replay))])
        worker = np.argmax(np.sum(replay_rewards, axis=0))  # sample experience from best worker
        nonzero_idxs = np.where(np.abs(replay_rewards) > 0)[0]  # idxs where |reward| > 0
        zero_idxs = np.where(replay_rewards == 0)[0]  # idxs where reward == 0

        if (
            len(nonzero_idxs) == 0 or len(zero_idxs) == 0
        ):  # if nonzero or zero idxs do not exist i.e. all rewards same sign
            idx = np.random.randint(len(replay_rewards))
        elif np.random.uniform() > 0.5:  # sample from zero and nonzero rewards equally
            # print('nonzero')
            idx = np.random.choice(nonzero_idxs)
        else:
            idx = np.random.choice(zero_idxs)

        reward_states = self.replay[idx][0][worker]
        reward = np.array([sign(replay_rewards[idx, worker])])  # source of error

        return reward_states[None], reward

    def _train_nstep(self):
        batch_size = self.num_envs * self.nsteps
        num_updates = self.total_steps // batch_size
        s = 0
        self.state_min = 0
        self.state_max = 0
        self.populate_memory()
        # main loop
        start = time.time()
        for t in range(1, num_updates + 1):
            states, actions, rewards, values, dones, last_values = self.rollout()

            # R = self.nstep_return(rewards, last_values, dones, clip=False)
            R = GAE(rewards, values, last_values, dones, gamma=0.99, lambda_=0.95) + values

            # stack all states, actions and Rs across all workers into a single batch
            states, actions, rewards, R = (
                fold_batch(states),
                fold_batch(actions),
                fold_batch(rewards),
                fold_batch(R),
            )

            # self.state_mean, self.state_std = self.obs_running.update(states[...,-1:]) # update state normalisation statistics
            self.update_minmax(states)

            reward_states, sample_rewards = self.sample_reward()
            replay_states, replay_actions, replay_R, Qaux_target, replay_dones = (
                self.sample_replay()
            )

            loss_value = self.model.backprop(
                states,
                R,
                actions,
                reward_states,
                sample_rewards,
                Qaux_target,
                replay_actions,
                replay_states,
                replay_R,
            )

            if (
                self.render_freq > 0
                and t % ((self.validate_freq // batch_size) * self.render_freq) == 0
            ):
                render = True
            else:
                render = False

            if self.validate_freq > 0 and t % (self.validate_freq // batch_size) == 0:
                self.validation_summary(t, loss_value, start, render)
                start = time.time()

            if self.save_freq > 0 and t % (self.save_freq // batch_size) == 0:
                s += 1
                self.save(self.s)
                print('saved model')

    def rollout(
        self,
    ):
        rollout = []
        for _t in range(self.nsteps):
            policies, values = self.model.evaluate(self.states)
            # Qaux = self.model.get_pixel_control(self.states, self.prev_hidden, self.prev_actions_rewards[np.newaxis])
            actions = fastsample(policies)
            next_states, rewards, dones, infos = self.env.step(actions)

            rollout.append((self.states, actions, rewards, values, dones))
            self.replay.append(
                (self.states, actions, rewards, values, dones)
            )  # add to replay memory
            self.states = next_states

        states, actions, rewards, values, dones = stack_many(*zip(*rollout))
        _, last_values = self.model.evaluate(next_states)
        return states, actions, rewards, values, dones, last_values

    def get_action(self, state):
        policy, value = self.model.evaluate(state)
        action = int(np.random.choice(policy.shape[1], p=policy[0]))
        return action
