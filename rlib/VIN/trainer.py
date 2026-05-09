import os
import threading
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from rlib.training.returns import Returns
from rlib.utils.utils import fold_batch, one_hot, stack_many
from rlib.VIN.model import VINCNN


class VINTrainer:
    """Standalone trainer for the Value Iteration Network agent.

    Doesn't subclass :class:`SyncMultiEnvTrainer` because VIN's
    optimisation loop and observation interface are noticeably different
    (no value-targets / advantage estimates, location-aware action
    selection). Could be migrated in a future refactor.
    """

    def __init__(
        self,
        agent: VINCNN,
        envs,
        val_envs,
        epsilon=0.1,
        epsilon_final=0.1,
        epsilon_steps=1000000,
        epsilon_test=0.1,
        returns: Returns = Returns.NSTEP,
        log_dir='logs/',
        model_dir='models/',
        total_steps=50000000,
        nsteps=20,
        gamma=0.99,
        lambda_=0.95,
        validate_freq=1e6,
        save_freq=0,
        render_freq=0,
        update_target_freq=0,
        num_val_episodes=50,
        log_scalars=True,
    ):
        self.agent = agent
        self.env = envs
        self.num_envs = len(envs)
        self.val_envs = val_envs
        self.total_steps = total_steps
        self.action_size = self.agent.action_size
        self.epsilon = epsilon
        self.epsilon_test = epsilon_test
        self.states = self.env.reset()
        self.loc = self.get_locs()
        print('locs', self.loc)

        self.total_steps = int(total_steps)
        self.nsteps = nsteps
        self.returns = returns
        self.gamma = gamma
        self.lambda_ = lambda_

        self.validate_freq = int(validate_freq)
        self.num_val_episodes = num_val_episodes

        self.save_freq = int(save_freq)
        self.render_freq = render_freq
        self.target_freq = int(update_target_freq)
        self.t = 1

        self.validate_rewards = []
        self.lock = threading.Lock()
        self.scheduler = self.linear_schedule(epsilon, epsilon_final, epsilon_steps)

        self.log_scalars = log_scalars
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.s = 0  # number of saves made

        if log_scalars:
            # Tensorboard Variables
            train_log_dir = self.log_dir + '/train'
            self.train_writer = SummaryWriter(train_log_dir)

    def get_locs(self):
        locs = []
        for env in self.env.envs:
            locs.append(env.agent_loc)
        return locs

    def train(self):
        self.train_nstep()

    def train_nstep(self):
        batch_size = self.num_envs * self.nsteps
        num_updates = self.total_steps // batch_size
        # main loop
        start = time.time()
        for t in range(self.t, num_updates + 1):
            states, locs, actions, rewards, dones, infos, values, last_values = self.rollout()
            R = self.returns(rewards, values, last_values, dones, self.gamma, self.lambda_)
            # stack all states, actions and Rs from all workers into a single batch
            states, locs, actions, R = (
                fold_batch(states),
                fold_batch(locs),
                fold_batch(actions),
                fold_batch(R),
            )
            # print('locs', locs.shape)
            loss_value = self.agent.backprop(states, locs, R, actions)

            if self.validate_freq > 0 and t % (self.validate_freq // batch_size) == 0:
                self.validation_summary(t, loss_value, start, False)
                start = time.time()

            if self.save_freq > 0 and t % (self.save_freq // batch_size) == 0:
                self.s += 1
                self.save(self.s)
                print('saved model')

            if (
                self.target_freq > 0 and t % (self.target_freq // batch_size) == 0
            ):  # update target network (for value based learning e.g. DQN)
                self.update_target()

            self.t += 1

    def eval_state(self, state, loc):
        return self.agent.evaluate(state, loc)

    def save(self, s: int) -> None:
        """Save the agent weights to ``<model_dir>/<s>.pt``."""
        os.makedirs(self.model_dir, exist_ok=True)
        torch.save(self.agent.state_dict(), f"{self.model_dir}/{s}.pt")

    def update_target(self) -> None:
        """No-op: VIN has no target network. Defined so the training loop
        can call it unconditionally when ``update_target_freq > 0``."""
        return

    def rollout(self):
        rollout = []
        for _t in range(self.nsteps):
            Qsa = self.eval_state(self.states, self.loc)
            actions = np.argmax(Qsa, axis=1)
            random = np.random.uniform(size=(self.num_envs))
            random_actions = np.random.randint(self.action_size, size=(self.num_envs))
            actions = np.where(random < self.epsilon, random_actions, actions)
            next_states, rewards, dones, infos = self.env.step(actions)
            values = np.sum(Qsa * one_hot(actions, self.action_size), axis=-1)
            rollout.append((self.states, self.loc, actions, rewards, dones, infos, values))
            self.states = next_states
            self.epsilon = self.scheduler.step()
            self.loc = self.get_locs()

        states, locs, actions, rewards, dones, infos, values = stack_many(*zip(*rollout))

        last_Qsa = self.eval_state(next_states, self.loc)  # Q(s,a|theta)
        last_actions = np.argmax(last_Qsa, axis=1)
        last_values = np.sum(last_Qsa * one_hot(last_actions, self.action_size), axis=-1)
        return states, locs, actions, rewards, dones, infos, values, last_values

    def get_action(self, state, loc):
        Qsa = self.eval_state(state, loc)
        if np.random.uniform() < self.epsilon_test:
            action = np.random.choice(self.action_size)
        else:
            action = np.argmax(Qsa, axis=1)
        return action

    def validation_summary(self, t, loss, start, render):
        batch_size = self.num_envs * self.nsteps
        tot_steps = t * batch_size
        time_taken = time.time() - start
        frames_per_update = (self.validate_freq // batch_size) * batch_size
        fps = frames_per_update / time_taken
        num_val_envs = len(self.val_envs)
        num_val_eps = [self.num_val_episodes // num_val_envs for i in range(num_val_envs)]
        num_val_eps[-1] = num_val_eps[-1] + self.num_val_episodes % self.num_val_episodes // (
            num_val_envs
        )
        render_array = np.zeros(len(self.val_envs))
        render_array[0] = render
        threads = [
            threading.Thread(
                daemon=True,
                target=self.validate,
                args=(self.val_envs[i], num_val_eps[i], 10000, render_array[i]),
            )
            for i in range(num_val_envs)
        ]
        try:
            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join()

        except KeyboardInterrupt:
            for thread in threads:
                thread.join()

        score = np.mean(self.validate_rewards)
        self.validate_rewards = []
        print(
            f"update {t}, validation score {score:f}, total steps {tot_steps}, "
            f"loss {loss:f}, time taken for {frames_per_update} frames:{time_taken:f}s, "
            f"fps {fps:f}"
        )

        if self.log_scalars:
            self.train_writer.add_scalar('Validation/Score', score)
            self.train_writer.add_scalar('Training/Loss', loss)

    def validate(self, env, num_ep, max_steps, render=False):
        for _episode in range(num_ep):
            state = env.reset()
            loc = env.agent_loc
            episode_score = []
            for t in range(max_steps):
                action = self.get_action(state[np.newaxis], [loc])
                next_state, reward, done, info = env.step(action)
                state = next_state
                loc = env.agent_loc

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
                self._epsilon = self._epsilon_final

            return self._epsilon
