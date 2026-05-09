import time
from dataclasses import dataclass

import numpy as np

from rlib.PPO.model import PPO
from rlib.training import SyncMultiEnvTrainer, TrainerConfig
from rlib.training.returns import GAE
from rlib.utils.utils import fastsample, fold_many, stack_many


@dataclass(frozen=True)
class PPOTrainerConfig(TrainerConfig):
    """Hyperparameters for :class:`PPOTrainer`."""

    num_epochs: int = 4
    num_minibatches: int = 4


class PPOTrainer(SyncMultiEnvTrainer):
    """Trainer for the clipped-objective PPO model."""

    def __init__(
        self,
        envs,
        model: PPO,
        val_envs,
        config: PPOTrainerConfig,
    ):
        super().__init__(envs, model, val_envs, config=config)
        self.num_epochs = config.num_epochs
        self.num_minibatches = config.num_minibatches

    def _train_nstep(self):
        batch_size = self.num_envs * self.nsteps
        num_updates = self.total_steps // batch_size
        s = 0
        mini_batch_size = self.nsteps // self.num_minibatches
        start = time.time()
        # main loop
        for t in range(1, num_updates + 1):
            # rollout_start = time.time()
            states, actions, rewards, values, last_values, old_policies, dones = self.rollout()
            # print('rollout time', time.time()-rollout_start)
            Adv = GAE(rewards, values, last_values, dones, gamma=self.gamma, lambda_=self.lambda_)
            R = Adv + values
            loss_value = 0

            # backprop_time = time.time()
            idxs = np.arange(len(states))
            for _epoch in range(self.num_epochs):
                np.random.shuffle(idxs)
                for batch in range(0, len(states), mini_batch_size):
                    batch_idxs = idxs[batch : batch + mini_batch_size]
                    # stack all states, actions and Rs across all workers into a single batch
                    mb_states, mb_actions, mb_R, mb_Adv, mb_old_policies = fold_many(
                        states[batch_idxs],
                        actions[batch_idxs],
                        R[batch_idxs],
                        Adv[batch_idxs],
                        old_policies[batch_idxs],
                    )

                    loss_value += self.model.backprop(
                        mb_states.copy(),
                        mb_R.copy(),
                        mb_Adv.copy(),
                        mb_actions.copy(),
                        mb_old_policies.copy(),
                    )

            # print('backprop time', time.time()-backprop_time)
            loss_value /= self.num_epochs

            if (
                self.render_freq > 0
                and t % ((self.validate_freq // batch_size) * self.render_freq) == 0
            ):
                render = True
            else:
                render = False

            if self.validate_freq > 0 and t % (self.validate_freq // batch_size) == 0:
                # val_time = time.time()
                self.validation_summary(t, loss_value, start, render)
                # print('validation time', time.time()-val_time)
                start = time.time()

            if self.save_freq > 0 and t % (self.save_freq // batch_size) == 0:
                s += 1
                self.save(s)
                print('saved model')

    def get_action(self, states):
        policies, values = self.model.evaluate(states)
        actions = fastsample(policies)
        return actions

    def rollout(self):
        rollout = []
        for _t in range(self.nsteps):
            policies, values = self.model.evaluate(self.states)
            actions = fastsample(policies)
            next_states, rewards, dones, infos = self.env.step(actions)
            rollout.append((self.states, actions, rewards, values, policies, dones))
            self.states = next_states

        states, actions, rewards, values, policies, dones = stack_many(*zip(*rollout))
        (
            policy,
            last_values,
        ) = self.model.evaluate(next_states)
        return states, actions, rewards, values, last_values, policies, dones
