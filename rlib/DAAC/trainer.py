import time
from dataclasses import dataclass

import numpy as np
from tqdm.auto import tqdm

from rlib.DAAC.model import DAAC
from rlib.training import SyncMultiEnvTrainer, TrainerConfig
from rlib.training.returns import GAE, lambda_return
from rlib.utils.utils import fastsample, fold_many, stack_many


@dataclass(frozen=True)
class DAACTrainerConfig(TrainerConfig):
    """Hyperparameters for :class:`DAACTrainer`."""

    policy_epochs: int = 1
    value_epochs: int = 9
    num_minibatches: int = 8


class DAACTrainer(SyncMultiEnvTrainer):
    """Trainer for the Decoupled Advantage Actor-Critic agent."""

    agent: DAAC

    def __init__(
        self,
        envs,
        agent: DAAC,
        val_envs,
        config: DAACTrainerConfig,
    ):
        super().__init__(envs, agent, val_envs, config=config)
        self.policy_epochs = config.policy_epochs
        self.value_epochs = config.value_epochs
        self.num_minibatches = config.num_minibatches

    def _train_nstep(self):
        batch_size = self.num_envs * self.nsteps
        num_updates = self.total_steps // batch_size
        s = 0
        mini_batch_size = self.nsteps // self.num_minibatches
        start = time.time()
        # main loop
        for t in self._progress(range(1, num_updates + 1), num_updates):
            # rollout_start = time.time()
            states, actions, rewards, values, last_values, old_policies, dones = self.rollout()
            # print('rollout time', time.time()-rollout_start)
            Adv = GAE(rewards, values, last_values, dones, gamma=self.gamma, lambda_=self.lambda_)
            R = lambda_return(
                rewards, values, last_values, dones, gamma=self.gamma, lambda_=self.lambda_
            )
            loss_value = 0

            idxs = np.arange(len(states))
            value_loss = 0
            for _epoch in range(self.value_epochs):
                np.random.shuffle(idxs)
                for batch in range(0, len(states), mini_batch_size):
                    batch_idxs = idxs[batch : batch + mini_batch_size]
                    # stack all states, actions and Rs across all workers into a single batch
                    (
                        mb_states,
                        mb_Rs,
                    ) = fold_many(states[batch_idxs], R[batch_idxs])

                    value_loss += self.agent.value.backprop(mb_states.copy(), mb_Rs.copy())

            value_loss /= self.value_epochs

            idxs = np.arange(len(states))
            policy_loss = 0
            for _epoch in range(self.policy_epochs):
                np.random.shuffle(idxs)
                for batch in range(0, len(states), mini_batch_size):
                    batch_idxs = idxs[batch : batch + mini_batch_size]
                    # stack all states, actions and Rs across all workers into a single batch
                    mb_states, mb_actions, mb_Adv, mb_old_policies = fold_many(
                        states[batch_idxs],
                        actions[batch_idxs],
                        Adv[batch_idxs],
                        old_policies[batch_idxs],
                    )

                    policy_loss += self.agent.policy.backprop(
                        mb_states.copy(), mb_Adv.copy(), mb_actions.copy(), mb_old_policies.copy()
                    )

            policy_loss /= self.policy_epochs
            loss_value = policy_loss + value_loss

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
                tqdm.write('saved model')

    def get_action(self, states):
        policies, values = self.agent.evaluate(states)
        actions = fastsample(policies)
        if states.shape[0] == 1:
            return int(actions.item())
        return actions

    def rollout(self):
        rollout = []
        for _t in range(self.nsteps):
            policies, values = self.agent.evaluate(self.states)
            actions = fastsample(policies)
            next_states, rewards, dones, infos = self.env.step(actions)
            rollout.append((self.states, actions, rewards, values, policies, dones))
            self.states = next_states

        states, actions, rewards, values, policies, dones = stack_many(*zip(*rollout))
        (
            policy,
            last_values,
        ) = self.agent.evaluate(next_states)
        return states, actions, rewards, values, last_values, policies, dones
