import numpy as np

from rlib.DDQN.model import DQN
from rlib.utils import DDQNTrainerConfig
from rlib.utils.SyncMultiEnvTrainer import SyncMultiEnvTrainer
from rlib.utils.utils import fold_batch, one_hot, unfold_batch
from rlib.utils.wrappers import FireResetEnv, StackEnv


class SyncDDQN(SyncMultiEnvTrainer):
    """Synchronous Double-DQN trainer (n-step or one-step TD)."""

    def __init__(
        self,
        envs,
        model: DQN,
        target_model: DQN,
        val_envs,
        action_size,
        config: DDQNTrainerConfig,
    ):
        super().__init__(envs=envs, model=model, val_envs=val_envs, config=config)

        self.target_model = self.TargetQ = target_model
        self.Q = self.model  # more readable alias
        self.epsilon = np.array([config.epsilon_start], dtype=np.float64)
        self.epsilon_final = config.epsilon_final
        self.epsilon_steps = config.epsilon_steps
        self.schedule = self.linear_schedule(
            self.epsilon, config.epsilon_final, config.epsilon_steps // self.num_envs
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
        if np.random.uniform() < self.epsilon_test:
            action = np.random.randint(self.action_size)
        else:
            action = np.argmax(self.model.evaluate(state))
        return action

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

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
            self.TargetQ.evaluate(fold_batch(states)), self.num_steps, self.num_envs
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


def stackFireReset(env):
    return StackEnv(FireResetEnv(env))
