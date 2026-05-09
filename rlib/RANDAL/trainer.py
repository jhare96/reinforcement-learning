import time
from collections import deque
from dataclasses import dataclass

import numpy as np
from tqdm.auto import tqdm

from rlib.RANDAL.model import RANDAL, sign
from rlib.RND.model import RewardForwardFilter
from rlib.RND.trainer import RNDTrainerConfig
from rlib.training import SyncMultiEnvTrainer
from rlib.training.returns import GAE
from rlib.utils.utils import (
    RunningMeanStd,
    fastsample,
    fold_batch,
    fold_many,
    stack_many,
)


@dataclass(frozen=True)
class RANDALTrainerConfig(RNDTrainerConfig):
    """Hyperparameters for :class:`RANDALTrainer`.

    Inherits the RND extra fields and adds the UNREAL replay buffer
    knobs.
    """

    replay_length: int = 2000
    norm_pixel_reward: bool = True


class RANDALTrainer(SyncMultiEnvTrainer):
    """Trainer for the RANDAL agent (RND + UNREAL auxiliary tasks)."""

    agent: RANDAL

    def __init__(
        self,
        envs,
        agent: RANDAL,
        val_envs,
        config: RANDALTrainerConfig,
    ):
        super().__init__(envs, agent, val_envs, config=config)

        self.gamma_intr = config.gamma_intr
        self.num_epochs = config.num_epochs
        self.num_minibatches = config.num_minibatches
        self.init_obs_steps = config.init_obs_steps
        self.replay_length = config.replay_length
        self.normalise_obs = config.norm_pixel_reward
        self.pred_prob = 1 / (self.num_envs / 32.0)
        self.state_obs = RunningMeanStd()
        self.forward_filter = RewardForwardFilter(config.gamma_intr)
        self.intr_rolling = RunningMeanStd()
        self.replay = deque([], maxlen=config.replay_length)  # replay length per actor

    def populate_memory(self):
        for _t in range(self.replay_length // self.nsteps):
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
        _, replay_last_values_extr, replay_last_values_intr = self.agent.evaluate(next_state)
        replay_R = (
            GAE(
                replay_rewards,
                replay_values,
                replay_last_values_extr,
                replay_dones,
                gamma=0.99,
                lambda_=0.95,
            )
            + replay_values
        )

        if self.agent.pixel_control:
            prev_states = self.replay[sample_start - 1][0][workers]
            Qaux_value = self.agent.get_pixel_control(next_state)
            pixel_rewards = self.pixel_rewards(prev_states, replay_states)
            Qaux_target = self.auxiliary_target(
                pixel_rewards, np.max(Qaux_value, axis=1), replay_dones
            )
        else:
            Qaux_target = np.zeros(
                (len(replay_states), 1, 1, 1)
            )  # produce fake Qaux to save writing unecessary code

        return replay_states, replay_actions, replay_R, Qaux_target, replay_dones

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

    def init_state_obs(self, num_steps):
        states = 0
        for _i in range(num_steps):
            rand_actions = np.random.randint(0, self.agent.action_size, size=self.num_envs)
            next_states, rewards, dones, infos = self.env.step(rand_actions)
            next_states = (
                next_states[:, -1] if len(next_states.shape) == 4 else next_states
            )  # [num_envs, channels, height, width] for convolutions, assume frame stack
            states += next_states
        # Reduce over the env axis too — running stats are a single per-pixel
        # mean shared across all envs, so it broadcasts cleanly against the
        # folded ``(T*B, ...)`` batch produced during training.
        return (states / num_steps).mean(axis=0)

    def _train_nstep(self):
        # stats for normalising states
        self.state_mean, self.state_std = self.state_obs.update(
            self.init_state_obs(self.init_obs_steps)
        )
        self.state_min, self.state_max = 0.0, 0.0
        self.populate_memory()  # populate experience replay with random actions
        self.states = self.env.reset()  # reset to state s_0

        batch_size = self.num_envs * self.nsteps
        num_updates = self.total_steps // batch_size
        s = 0
        mini_batch_size = self.nsteps // self.num_minibatches
        start = time.time()
        # main loop
        for t in self._progress(range(1, num_updates + 1), num_updates):
            (
                states,
                next_states,
                actions,
                extr_rewards,
                intr_rewards,
                values_extr,
                values_intr,
                last_values_extr,
                last_values_intr,
                old_policies,
                dones,
            ) = self.rollout()
            # update state normalisation statistics — fold (T, B, ...) into (T*B, ...)
            # so the running mean has one entry per pixel, not per env.
            self.update_minmax(states)
            self.state_mean, self.state_std = self.state_obs.update(fold_batch(next_states))
            mean, std = self.state_mean, self.state_std

            replay_states, replay_actions, replay_Re, Qaux_target, replay_dones = (
                self.sample_replay()
            )  # sample experience replay

            int_rff = np.array(
                [self.forward_filter.update(intr_rewards[i]) for i in range(len(intr_rewards))]
            )
            R_intr_mean, R_intr_std = self.intr_rolling.update(
                int_rff.ravel()
            )  # normalise intrinsic rewards
            intr_rewards /= R_intr_std

            Adv_extr = GAE(
                extr_rewards,
                values_extr,
                last_values_extr,
                dones,
                gamma=self.gamma,
                lambda_=self.lambda_,
            )
            Adv_intr = GAE(
                intr_rewards,
                values_intr,
                last_values_intr,
                dones,
                gamma=self.gamma_intr,
                lambda_=self.lambda_,
            )
            Re = Adv_extr + values_extr
            Ri = Adv_intr + values_intr
            total_Adv = Adv_extr + Adv_intr
            loss_value = 0

            # perform minibatch gradient descent for K epochs
            idxs = np.arange(len(states))
            for _epoch in range(self.num_epochs):
                reward_states, sample_rewards = (
                    self.sample_reward()
                )  # sample reward from replay memory
                np.random.shuffle(idxs)
                for batch in range(0, len(states), mini_batch_size):
                    batch_idxs = idxs[batch : batch + mini_batch_size]
                    # stack all states, actions and Rs across all workers into a single batch
                    mb_states, mb_nextstates, mb_actions, mb_Re, mb_Ri, mb_Adv, mb_old_policies = (
                        fold_many(
                            states[batch_idxs],
                            next_states[batch_idxs],
                            actions[batch_idxs],
                            Re[batch_idxs],
                            Ri[batch_idxs],
                            total_Adv[batch_idxs],
                            old_policies[batch_idxs],
                        )
                    )

                    mb_replay_states, mb_replay_actions, mb_replay_Rextr, mb_Qaux_target = (
                        fold_many(
                            replay_states[batch_idxs],
                            replay_actions[batch_idxs],
                            replay_Re[batch_idxs],
                            Qaux_target[batch_idxs],
                        )
                    )

                    mb_nextstates = mb_nextstates[
                        np.where(np.random.uniform(size=(mini_batch_size)) < self.pred_prob)
                    ]
                    # states, next_states, Re, Ri, Adv, actions, old_policy, reward_states, rewards, Qaux_target, Qaux_actions, replay_states, replay_R, state_mean, state_std
                    loss_value += self.agent.backprop(
                        mb_states.copy(),
                        mb_nextstates.copy(),
                        mb_Re.copy(),
                        mb_Ri.copy(),
                        mb_Adv.copy(),
                        mb_actions.copy(),
                        mb_old_policies.copy(),
                        reward_states.copy(),
                        sample_rewards.copy(),
                        mb_Qaux_target.copy(),
                        mb_replay_actions.copy(),
                        mb_replay_states.copy(),
                        mb_replay_Rextr.copy(),
                        mean.copy(),
                        std.copy(),
                    )

            loss_value /= self.num_epochs

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
                self.save(s)
                tqdm.write('saved model')

    def get_action(self, states):
        policies, values_extr, values_intr = self.agent.evaluate(states)
        actions = fastsample(policies)
        if states.shape[0] == 1:
            return int(actions.item())
        return actions

    def rollout(self):
        rollout = []
        for _t in range(self.nsteps):
            policies, values_extr, values_intr = self.agent.evaluate(self.states)
            actions = fastsample(policies)
            next_states, extr_rewards, dones, infos = self.env.step(actions)

            next_states__ = (
                next_states[:, -1:] if len(next_states.shape) == 4 else next_states
            )  # [num_envs, channels, height, width] for convolutions
            intr_rewards = self.agent.intrinsic_reward(
                next_states__, self.state_mean, self.state_std
            )

            rollout.append(
                (
                    self.states,
                    next_states__,
                    actions,
                    extr_rewards,
                    intr_rewards,
                    values_extr,
                    values_intr,
                    policies,
                    dones,
                )
            )
            self.replay.append(
                (self.states, actions, extr_rewards, values_extr, dones)
            )  # add to replay memory
            self.states = next_states

        (
            states,
            next_states,
            actions,
            extr_rewards,
            intr_rewards,
            values_extr,
            values_intr,
            policies,
            dones,
        ) = stack_many(*zip(*rollout))
        (
            last_policy,
            last_values_extr,
            last_values_intr,
        ) = self.agent.evaluate(self.states)
        return (
            states,
            next_states,
            actions,
            extr_rewards,
            intr_rewards,
            values_extr,
            values_intr,
            last_values_extr,
            last_values_intr,
            policies,
            dones,
        )
