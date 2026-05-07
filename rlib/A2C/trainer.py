import time

import numpy as np

from rlib.A2C.model import ActorCritic, ActorCritic_LSTM
from rlib.utils import TrainerConfig
from rlib.utils.SyncMultiEnvTrainer import SyncMultiEnvTrainer
from rlib.utils.utils import fastsample, fold_batch, stack_many


class A2CTrainer(SyncMultiEnvTrainer):
    """Synchronous Advantage Actor-Critic trainer (feed-forward)."""

    def __init__(
        self,
        envs,
        model: ActorCritic,
        val_envs,
        config: TrainerConfig,
    ) -> None:
        super().__init__(envs, model, val_envs, config=config)

        hyperparas = {
            'learning_rate': model.lr,
            'learning_rate_final': model.lr_final,
            'lr_decay_steps': model.decay_steps,
            'grad_clip': model.grad_clip,
            'nsteps': config.nsteps,
            'num_workers': self.num_envs,
            'total_steps': config.total_steps,
            'entropy_coefficient': model.entropy_coeff,
            'value_coefficient': model.value_coeff,
            'return type': config.return_type,
            'gamma': config.gamma,
            'lambda': config.lambda_,
        }

        if config.log_scalars:
            filename = config.log_dir + '/' + 'hyperparameters.txt'
            self.save_hyperparameters(filename, **hyperparas)

    def get_action(self, state):
        policy, value = self.model.evaluate(state)
        action = int(fastsample(policy))
        return action

    def rollout(
        self,
    ):
        rollout = []
        for _t in range(self.nsteps):
            policies, values = self.model.evaluate(self.states)
            actions = fastsample(policies)
            next_states, rewards, dones, infos = self.env.step(actions)
            rollout.append((self.states, actions, rewards, values, dones))
            self.states = next_states

        states, actions, rewards, values, dones = stack_many(*zip(*rollout))
        _, last_values = self.model.evaluate(next_states)
        return states, actions, rewards, dones, values, last_values

    def _train_onestep(self):
        states = self.env.reset()
        y = np.zeros(self.num_envs)
        num_steps = self.total_steps // self.num_envs
        start = time.time()
        for t in range(1, num_steps + 1):
            policies, values = self.model.evaluate(self.states)
            actions = fastsample(policies)
            next_states, rewards, dones, infos = self.env.step(actions)
            y = rewards + self.gamma * self.model.get_value(next_states) * (1 - dones)

            loss_value = self.model.backprop(states, y, actions)
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

    def __init__(
        self,
        envs,
        model: ActorCritic_LSTM,
        val_envs,
        config: TrainerConfig,
    ) -> None:
        super().__init__(envs, model, val_envs, config=config)

        self.prev_hidden = self.model.get_initial_hidden(self.num_envs)

        hyper_params = {
            'learning_rate': model.lr,
            'learning_rate_final': model.lr_final,
            'lr_decay_steps': model.decay_steps,
            'grad_clip': model.grad_clip,
            'nsteps': self.nsteps,
            'num_workers': self.num_envs,
            'total_steps': self.total_steps,
            'entropy_coefficient': model.entropy_coeff,
            'value_coefficient': model.value_coeff,
            'gamma': self.gamma,
            'lambda': self.lambda_,
        }

        if self.log_scalars:
            filename = config.log_dir + '/hyperparameters.txt'
            self.save_hyperparameters(filename, **hyper_params)

    def _train_nstep(self):
        batch_size = self.num_envs * self.nsteps
        start = time.time()
        num_updates = self.total_steps // batch_size
        s = 0
        # main loop
        for t in range(1, num_updates + 1):
            states, actions, rewards, first_hidden, dones, values, last_values = self.rollout()

            if self.return_type == 'nstep':
                R = self.nstep_return(rewards, last_values, dones, gamma=self.gamma)
            elif self.return_type == 'GAE':
                R = (
                    self.GAE(
                        rewards, values, last_values, dones, gamma=self.gamma, lambda_=self.lambda_
                    )
                    + values
                )
            elif self.return_type == 'lambda':
                R = self.lambda_return(
                    rewards, values, last_values, dones, gamma=self.gamma, lambda_=self.lambda_
                )

            # stack all states, actions and Rs across all workers into a single batch
            actions, R = fold_batch(actions), fold_batch(R)
            loss_value = self.model.backprop(states, R, actions, first_hidden, dones)

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
                self.saver.save(self.sess, str(self.model_dir + str(s) + ".ckpt"))
                print('saved model')

    def _validate_async(self, env, num_ep, max_steps, render=False):
        for _episode in range(num_ep):
            state = env.reset()
            episode_score = []
            hidden = self.model.get_initial_hidden(1)
            for t in range(max_steps):
                policy, value, hidden = self.model.evaluate(state[None, None], hidden)
                # print('policy', policy, 'value', value)
                action = int(fastsample(policy))
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
        for _episode in range(self.num_val_episodes // len(env)):
            states = env.reset()
            episode_score = []
            prev_hidden = self.model.get_initial_hidden(len(self.val_envs))
            for t in range(self.val_steps):
                policies, values, hidden = self.model.evaluate(states[None], prev_hidden)
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
            policies, values, hidden = self.model.evaluate(self.states[None], self.prev_hidden)
            actions = fastsample(policies)
            next_states, rewards, dones, infos = self.env.step(actions)
            rollout.append((self.states, actions, rewards, values, dones))
            self.states = next_states
            self.prev_hidden = self.model.mask_hidden(
                hidden, dones
            )  # reset hidden state at end of episode

        states, actions, rewards, values, dones = stack_many(*zip(*rollout))
        _, last_values, _ = self.model.evaluate(self.states[None], self.prev_hidden)
        return states, actions, rewards, first_hidden, dones, values, last_values
