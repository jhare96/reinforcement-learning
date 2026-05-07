import time
from dataclasses import dataclass

import numpy as np

from rlib.RND.model import RND, RewardForwardFilter
from rlib.training import SyncMultiEnvTrainer, TrainerConfig
from rlib.utils.utils import RunningMeanStd, fastsample, fold_many, stack_many


@dataclass(frozen=True)
class RNDTrainerConfig(TrainerConfig):
    """Hyperparameters for :class:`RNDTrainer`.

    ``gamma`` is reused as the *extrinsic* discount; the intrinsic
    discount is the new ``gamma_intr`` field.
    """

    gamma_intr: float = 0.99
    init_obs_steps: int = 600
    num_epochs: int = 4
    num_minibatches: int = 4


class RNDTrainer(SyncMultiEnvTrainer):
    """Trainer for the Random Network Distillation agent."""

    def __init__(
        self,
        envs,
        model: RND,
        val_envs,
        config: RNDTrainerConfig,
    ):
        super().__init__(envs, model, val_envs, config=config)

        self.gamma_intr = config.gamma_intr
        self.num_epochs = config.num_epochs
        self.num_minibatches = config.num_minibatches
        self.init_obs_steps = config.init_obs_steps
        self.pred_prob = 1 / (self.num_envs / 32.0)
        self.state_obs = RunningMeanStd()
        self.forward_filter = RewardForwardFilter(config.gamma_intr)
        self.intr_rolling = RunningMeanStd()

    def init_state_obs(self, num_steps):
        states = 0
        for _i in range(num_steps):
            rand_actions = np.random.randint(0, self.model.action_size, size=self.num_envs)
            next_states, rewards, dones, infos = self.env.step(rand_actions)
            next_states = (
                next_states[:, -1] if len(next_states.shape) == 4 else next_states
            )  # [num_envs, channels, height, width] for convolutions, assume frame stack
            states += next_states
        return states / num_steps

    def _train_nstep(self):
        # stats for normalising states
        self.state_mean, self.state_std = self.state_obs.update(
            self.init_state_obs(self.init_obs_steps)
        )
        self.states = self.env.reset()  # reset to state s_0

        batch_size = self.num_envs * self.nsteps
        num_updates = self.total_steps // batch_size
        s = 0
        mini_batch_size = self.nsteps // self.num_minibatches
        start = time.time()
        # main loop
        for t in range(1, num_updates + 1):
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
            self.state_mean, self.state_std = self.state_obs.update(
                next_states
            )  # update state normalisation statistics
            mean, std = self.state_mean, self.state_std

            int_rff = np.array(
                [self.forward_filter.update(intr_rewards[i]) for i in range(len(intr_rewards))]
            )
            R_intr_mean, R_intr_std = self.intr_rolling.update(
                int_rff.ravel()
            )  # normalise intrinsic rewards
            intr_rewards /= R_intr_std

            Adv_extr = self.GAE(
                extr_rewards,
                values_extr,
                last_values_extr,
                dones,
                gamma=self.gamma,
                lambda_=self.lambda_,
            )
            Adv_intr = self.GAE(
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

                    mb_nextstates = mb_nextstates[
                        np.where(np.random.uniform(size=(mini_batch_size)) < self.pred_prob)
                    ]
                    loss_value += self.model.backprop(
                        mb_states.copy(),
                        mb_nextstates.copy(),
                        mb_Re.copy(),
                        mb_Ri.copy(),
                        mb_Adv.copy(),
                        mb_actions.copy(),
                        mb_old_policies.copy(),
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
                print('saved model')

    def get_action(self, states):
        policies, values_extr, values_intr = self.model.evaluate(states)
        actions = fastsample(policies)
        return actions

    def rollout(self):
        rollout = []
        for _t in range(self.nsteps):
            policies, values_extr, values_intr = self.model.evaluate(self.states)
            actions = fastsample(policies)
            next_states, extr_rewards, dones, infos = self.env.step(actions)

            next_states__ = (
                next_states[:, -1:] if len(next_states.shape) == 4 else next_states
            )  # [num_envs, channels, height, width] for convolutions
            intr_rewards = self.model.intrinsic_reward(
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
        ) = self.model.evaluate(self.states)
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
