import time

import numpy as np

from rlib.Curiosity.model import Curiosity
from rlib.training import SyncMultiEnvTrainer, TrainerConfig
from rlib.training.returns import nstep_return
from rlib.utils.utils import RunningMeanStd, fastsample, fold_batch, stack_many


class RollingObs:
    def __init__(self, mean=0):
        self.rolling = RunningMeanStd()

    def update(self, x):
        if len(x.shape) == 4:  # assume image obs
            return self.rolling.update(
                np.mean(x, axis=1, keepdims=True)
            )  # [time*batch,height,width,stack] -> [height, width]
        else:
            return self.rolling.update(x)  # [time*batch,*shape] -> [*shape]


class CuriosityTrainer(SyncMultiEnvTrainer):
    """Trainer for the Intrinsic Curiosity Module agent."""

    def __init__(
        self,
        envs,
        agent: Curiosity,
        val_envs,
        config: TrainerConfig,
    ):
        super().__init__(envs, agent, val_envs, config=config)
        self.state_obs = RollingObs()
        self.state_mean = None
        self.state_std = None
        self.lambda_ = 0.95

    def init_state_obs(self, num_steps):
        states = 0
        for _i in range(num_steps):
            rand_actions = np.random.randint(0, self.agent.action_size, size=self.num_envs)
            next_states, rewards, dones, infos = self.env.step(rand_actions)
            states += next_states
        return states / num_steps

    def _train_nstep(self):
        num_updates = self.total_steps // (self.num_envs * self.nsteps)
        s = 0
        self.state_mean, self.state_std = self.state_obs.update(
            self.init_state_obs(10000 // self.num_envs)
        )
        self.states = self.env.reset()
        print(self.state_mean.shape, self.state_std.shape)
        start = time.time()
        # main loop
        batch_size = self.num_envs * self.nsteps
        for t in range(1, num_updates + 1):
            states, next_states, actions, rewards, dones, values = self.rollout()
            _, last_values = self.agent.evaluate(next_states[-1])

            R = nstep_return(rewards, last_values, dones)
            Adv = R - values
            # delta = rewards + self.gamma * values[:-1] - values[1:]
            # Adv = self.multistep_target(delta, values[-1], dones, gamma=self.gamma*self.lambda_)

            # stack all states, next_states, actions and Rs across all workers into a single batch
            states, next_states, actions, R, Adv = (
                fold_batch(states),
                fold_batch(next_states),
                fold_batch(actions),
                fold_batch(R),
                fold_batch(Adv),
            )
            mean, std = self.state_mean, self.state_std

            loss_value = self.agent.backprop(states, next_states, R, Adv, actions, mean, std)

            # self.state_mean, self.state_std = self.state_obs.update(states)

            if self.render_freq > 0 and t % (self.validate_freq * self.render_freq) == 0:
                render = True
            else:
                render = False

            if self.validate_freq > 0 and t % (self.validate_freq // batch_size) == 0:
                self.validation_summary(t, loss_value, start, render)
                start = time.time()

            if self.save_freq > 0 and t % (self.save_freq // batch_size) == 0:
                s += 1
                self.saver.save(
                    self.sess, str(self.model_dir + self.current_time + '/' + str(s) + ".ckpt")
                )
                print('saved model')

    def get_action(self, state):
        policy, value = self.agent.evaluate(state)
        action = int(np.random.choice(policy.shape[1], p=policy[0]))
        return action

    def rollout(
        self,
    ):
        rollout = []
        for _t in range(self.nsteps):
            policies, values = self.agent.evaluate(self.states)
            actions = fastsample(policies)
            next_states, extr_rewards, dones, infos = self.env.step(actions)

            mean, std = self.state_mean[None], self.state_std[None]
            intr_rewards = self.agent.intrinsic_reward(
                (self.states - mean) / std, actions, (next_states - mean) / std
            )
            rewards = extr_rewards + intr_rewards
            rollout.append((self.states, next_states, actions, rewards, values, dones))
            self.states = next_states

        states, next_states, actions, rewards, values, dones = stack_many(*zip(*rollout))
        return states, next_states, actions, rewards, dones, values
