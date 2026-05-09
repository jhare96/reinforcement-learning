from dataclasses import dataclass

import numpy as np

from rlib.DDQN.model import DQN
from rlib.training import SyncMultiEnvTrainer, TrainerConfig
from rlib.utils.utils import fold_batch, one_hot, unfold_batch


@dataclass(frozen=True)
class DDQNTrainerConfig(TrainerConfig):
    """Hyperparameters for :class:`SyncDDQN`."""

    epsilon_start: float = 1.0
    epsilon_final: float = 0.01
    epsilon_steps: float = 1e6
    epsilon_test: float = 0.01


class SyncDDQN(SyncMultiEnvTrainer):
    """Synchronous Double-DQN trainer (n-step or one-step TD)."""

    agent: DQN

    def __init__(
        self,
        envs,
        agent: DQN,
        target_agent: DQN,
        val_envs,
        action_size,
        config: DDQNTrainerConfig,
    ):
        super().__init__(envs=envs, agent=agent, val_envs=val_envs, config=config)

        self.target_agent = self.TargetQ = target_agent
        self.Q = self.agent  # more readable alias
        self.epsilon = np.array([config.epsilon_start], dtype=np.float64)
        self.epsilon_final = config.epsilon_final
        self.epsilon_steps = config.epsilon_steps
        self.schedule = self.linear_schedule(
            self.epsilon, config.epsilon_final, int(config.epsilon_steps // self.num_envs)
        )
        self.epsilon_test = np.array(config.epsilon_test, dtype=np.float64)
        self.action_size = action_size

    class linear_schedule:
        def __init__(self, epsilon, epsilon_final, num_steps=1000000):
            self._counter = 0
            self._epsilon = epsilon
            self._epsilon_final = epsilon_final
            self._step = (epsilon - epsilon_final) / num_steps
            self._num_steps = num_steps

        def step(
            self,
        ):
            if self._counter < self._num_steps:
                self._epsilon -= self._step
                self._counter += 1
            else:
                self._epsilon[:] = self._epsilon_final

        def get_epsilon(
            self,
        ):
            return self._epsilon

    def get_action(self, state):
        q_values = self.agent.evaluate(state)
        if state.shape[0] == 1:
            if np.random.uniform() < self.epsilon_test:
                return int(np.random.randint(self.action_size))
            return int(np.argmax(q_values))
        actions = np.argmax(q_values, axis=-1)
        rand_mask = np.random.uniform(size=actions.shape) < self.epsilon_test
        actions = np.where(
            rand_mask, np.random.randint(self.action_size, size=actions.shape), actions
        )
        return actions

    def update_target(self):
        self.target_agent.load_state_dict(self.agent.state_dict())

    def local_attr(self, attr):
        attr['update_target_freq'] = self.target_freq
        return attr

    def rollout(self):
        rollout = []
        for _t in range(self.nsteps):
            Qsa = self.Q.evaluate(self.states)
            actions = np.argmax(Qsa, axis=1)
            random = np.random.uniform(size=(self.num_envs))
            random_actions = np.random.randint(self.action_size, size=(self.num_envs))
            actions = np.where(random < self.epsilon, random_actions, actions)
            next_states, rewards, dones, infos = self.env.step(actions)
            rollout.append((self.states, actions, rewards, dones, infos))
            self.states = next_states
            self.schedule.step()

        states, actions, rewards, dones, infos = zip(*rollout)
        states, actions, rewards, dones = (
            np.stack(states),
            np.stack(actions),
            np.stack(rewards),
            np.stack(dones),
        )
        TargetQsa = unfold_batch(
            self.TargetQ.evaluate(fold_batch(states)), self.nsteps, self.num_envs
        )  # Q(s,a; theta-1)
        values = np.sum(
            TargetQsa * one_hot(actions, self.action_size), axis=-1
        )  # Q(s, argmax_a Q(s,a; theta); theta-1)

        last_actions = np.argmax(self.Q.evaluate(next_states), axis=1)
        last_TargetQsa = self.TargetQ.evaluate(next_states)  # Q(s,a; theta-1)
        last_values = np.sum(
            last_TargetQsa * one_hot(last_actions, self.action_size), axis=-1
        )  # Q(s, argmax_a Q(s,a; theta); theta-1)
        return states, actions, rewards, dones, values, last_values
