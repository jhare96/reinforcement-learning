import threading
import time

import numpy as np

from rlib.A2C.model import ActorCritic, ActorCritic_LSTM
from rlib.training import SyncMultiEnvTrainer, TrainerConfig
from rlib.utils.utils import fastsample, fold_batch, stack_many


class A2CTrainer(SyncMultiEnvTrainer):
    """Synchronous Advantage Actor-Critic trainer (feed-forward)."""

    agent: ActorCritic

    def __init__(
        self,
        envs,
        agent: ActorCritic,
        val_envs,
        config: TrainerConfig,
    ) -> None:
        super().__init__(envs, agent, val_envs, config=config)

    def get_action(self, state):
        policy, value = self.agent.evaluate(state)
        return int(fastsample(policy).item())

    def rollout(
        self,
    ):
        rollout = []
        for _t in range(self.nsteps):
            policies, values = self.agent.evaluate(self.states)
            actions = fastsample(policies)
            next_states, rewards, dones, infos = self.env.step(actions)
            rollout.append((self.states, actions, rewards, values, dones))
            self.states = next_states

        states, actions, rewards, values, dones = stack_many(*zip(*rollout))
        _, last_values = self.agent.evaluate(next_states)
        return states, actions, rewards, dones, values, last_values

    def _train_onestep(self):
        states = self.env.reset()
        y = np.zeros(self.num_envs)
        num_steps = self.total_steps // self.num_envs
        start = time.time()
        for t in range(1, num_steps + 1):
            policies, values = self.agent.evaluate(self.states)
            actions = fastsample(policies)
            next_states, rewards, dones, infos = self.env.step(actions)
            _, next_values = self.agent.evaluate(next_states)
            y = rewards + self.gamma * next_values * (1 - dones)

            loss_value = self.agent.backprop(states, y, actions)
            states = next_states

            if (
                self.render_freq > 0
                and t % ((self.validate_freq // self.num_envs) * self.render_freq) == 0
            ):
                render = True
            else:
                render = False

            if self.validate_freq > 0 and t % (self.validate_freq // self.num_envs) == 0:
                self.validation_summary(t, loss_value, start, render)
                start = time.time()
            if self.save_freq > 0 and t % (self.save_freq // self.num_envs) == 0:
                self.s += 1
                self.save(self.s)
                print('saved model')


class A2CLSTMTrainer(SyncMultiEnvTrainer):
    """Recurrent A2C trainer (LSTM hidden state propagated across rollouts)."""

    agent: ActorCritic_LSTM

    def __init__(
        self,
        envs,
        agent: ActorCritic_LSTM,
        val_envs,
        config: TrainerConfig,
    ) -> None:
        super().__init__(envs, agent, val_envs, config=config)
        self.prev_hidden = self.agent.get_initial_hidden(self.num_envs)

    def _validation_score(self, render: bool) -> float:
        # Recurrent: dispatch to the agent's own validate_sync /
        # validate_async so each episode resets the LSTM hidden state.
        if isinstance(self.val_envs, list):
            self.validate_rewards = []
            num_envs = len(self.val_envs)
            per_env = [self.num_val_episodes // num_envs for _ in range(num_envs)]
            per_env[-1] += self.num_val_episodes % num_envs
            threads = [
                threading.Thread(
                    daemon=True,
                    target=self._validate_async,
                    args=(self.val_envs[i], per_env[i], self.val_steps, render and i == 0),
                )
                for i in range(num_envs)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            return float(np.mean(self.validate_rewards)) if self.validate_rewards else 0.0
        return float(self.validate_sync(render))

    def _train_nstep(self):
        batch_size = self.num_envs * self.nsteps
        start = time.time()
        num_updates = self.total_steps // batch_size
        s = 0
        # main loop
        for t in range(1, num_updates + 1):
            states, actions, rewards, first_hidden, dones, values, last_values = self.rollout()

            R = self.returns(rewards, values, last_values, dones, self.gamma, self.lambda_)

            # stack all states, actions and Rs across all workers into a single batch
            actions, R = fold_batch(actions), fold_batch(R)
            loss_value = self.agent.backprop(states, R, actions, first_hidden, dones)

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
                self.save_model(s)
                print('saved model')

    def _validate_async(self, env, num_ep, max_steps, render=False):
        for _episode in range(num_ep):
            state = env.reset()
            episode_score = []
            hidden = self.agent.get_initial_hidden(1)
            for t in range(max_steps):
                policy, value, hidden = self.agent.evaluate(state[None, None], hidden)
                # print('policy', policy, 'value', value)
                action = int(fastsample(policy).item())
                next_state, reward, done, info = env.step(action)
                state = next_state

                episode_score.append(reward)

                if render:
                    with self.lock:
                        env.render()

                if done or t == max_steps - 1:
                    tot_reward = np.sum(episode_score)
                    with self.lock:
                        self.validate_rewards.append(tot_reward)

                    break
        if render:
            with self.lock:
                env.close()

    def validate_sync(self, render):
        episode_scores = []
        env = self.val_envs
        # _validation_score only dispatches here for non-list val_envs.
        assert not isinstance(env, list)
        for _episode in range(self.num_val_episodes // len(env)):
            states = env.reset()
            episode_score = []
            prev_hidden = self.agent.get_initial_hidden(len(self.val_envs))
            for t in range(self.val_steps):
                policies, values, hidden = self.agent.evaluate(states[None], prev_hidden)
                actions = fastsample(policies)
                next_states, rewards, dones, infos = env.step(actions)
                states = next_states

                episode_score.append(rewards * (1 - dones))

                if render:
                    with self.lock:
                        env.render()

                if dones.sum() == self.num_envs or t == self.val_steps - 1:
                    tot_reward = np.sum(np.stack(episode_score), axis=0)
                    episode_scores.append(tot_reward)
                    break

        return np.mean(episode_scores)

    def rollout(
        self,
    ):
        rollout = []
        first_hidden = self.prev_hidden
        for _t in range(self.nsteps):
            policies, values, hidden = self.agent.evaluate(self.states[None], self.prev_hidden)
            actions = fastsample(policies)
            next_states, rewards, dones, infos = self.env.step(actions)
            rollout.append((self.states, actions, rewards, values, dones))
            self.states = next_states
            self.prev_hidden = self.agent.mask_hidden(
                hidden, dones
            )  # reset hidden state at end of episode

        states, actions, rewards, values, dones = stack_many(*zip(*rollout))
        _, last_values, _ = self.agent.evaluate(self.states[None], self.prev_hidden)
        return states, actions, rewards, first_hidden, dones, values, last_values
