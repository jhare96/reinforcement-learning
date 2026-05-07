import json
import os
import threading
import time
from typing import Any

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from rlib.networks import Model
from rlib.utils.utils import fold_batch
from rlib.utils.VecEnv import BatchEnv, DummyBatchEnv


class SyncMultiEnvTrainer:
    """Synchronous multi-env training framework for any :class:`rlib.networks.Model`."""

    model: Model
    env: BatchEnv | DummyBatchEnv
    val_envs: Any
    validate_rewards: list[Any]
    train_writer: SummaryWriter
    train_log_dir: str

    def __init__(
        self,
        envs: BatchEnv | DummyBatchEnv,
        model: Model,
        val_envs: list | BatchEnv | DummyBatchEnv,
        train_mode: str = 'nstep',
        return_type: str = 'nstep',
        log_dir: str = 'logs/',
        model_dir: str = 'models/',
        total_steps: float = 50e6,
        nsteps: int = 5,
        gamma: float = 0.99,
        lambda_: float = 0.95,
        validate_freq: float = 1e6,
        save_freq: int = 0,
        render_freq: int = 0,
        update_target_freq: int = 0,
        num_val_episodes: int = 50,
        max_val_steps: int = 10000,
        log_scalars: bool = True,
    ) -> None:
        '''
        A synchronous multiple env training framework for pytorch

        Args:
            envs - BatchEnv | DummyBatchEnv: multiple synchronous training environments
            model - reinforcement learning model
            log_dir, log directory string for location of directory to log scalars log_dir='logs/', model_dir='models/',
            val_envs - use your own discretion to choose which validation mode you wan't, recommended BatchEnv or list for Atari and DummyBatchEnv for Classic Control like envs
                list: a list of envs for validation, uses threading to run environments asychronously
                BatchEnv: uses multiprocessing to run validation envs sychronously in parallel
                DummyBatchEnv: allows for sychronous env stepping without the overhead of multiprocessing, good for computationally cheap environments
            train_mode - 'nstep' or 'onestep' species whether training is done using multiple step TD learning or single step
            return_type - string to determine whether 'nstep', 'lambda' or 'GAE' returns are to be used
            total_steps - number of Total training steps across all environements
            nsteps - number of steps TD error is caluclated over
            validate_freq - number of steps across all environements before performing validating, 0 for no validation
            save_freq - number of steps across all environements before saving model, 0 for no saving
            render_freq - multiple of validate_freq before rendering (i.e. render every X validations), 0 for no rendering
            update_target_freq - number of steps across all environements before updating target model, 0 for no updating
            num_val_episodes - number of episodes to average over when validating
            max_val_steps - maximum number of steps for each validation episode (prevents infinite loops)
            log_scalars - boolean flag whether to log tensorboard scalars to log_dir
        '''
        self.env = envs
        if isinstance(envs, list):
            self.validate_func = self.validate_async
        else:
            self.validate_func = self.validate_sync
        if train_mode not in ['nstep', 'onestep']:
            raise ValueError(
                'train_mode {} is not a valid argument. Valid arguments are ... {}, {}'.format(
                    train_mode, 'nstep', 'onestep'
                )
            )
        assert num_val_episodes >= len(val_envs), (
            f'number of validation epsiodes {num_val_episodes} must be greater than or equal to the number of validation envs {len(val_envs)}'
        )
        if return_type not in ['nstep', 'lambda', 'GAE']:
            raise ValueError(
                'return_type {} is not a valid argument. Valid arguments are ... {}, {}, {}'.format(
                    return_type, 'nstep', 'lambda', 'GAE'
                )
            )
        self.train_mode = train_mode
        self.num_envs = len(envs)
        self.env_id = envs.spec.id
        self.val_envs = val_envs
        self.validate_rewards = []
        self.model = model

        self.total_steps = int(total_steps)
        self.nsteps = nsteps
        self.return_type = return_type
        self.gamma = gamma
        self.lambda_ = lambda_

        self.validate_freq = int(validate_freq)
        self.num_val_episodes = num_val_episodes
        self.val_steps = max_val_steps
        self.lock = threading.Lock()

        self.save_freq = int(save_freq)
        self.render_freq = render_freq
        self.target_freq = int(update_target_freq)
        self.s = 0  # number of saves made
        self.t = 1  # number of updates done
        self.log_scalars = log_scalars
        self.log_dir = log_dir
        self.model_dir = model_dir

        self.states = self.env.reset()

        if log_scalars:
            # Tensorboard Variables
            self.train_log_dir = self.log_dir + '/train'
            self.train_writer = SummaryWriter(self.train_log_dir)

        if not os.path.exists(self.model_dir) and save_freq > 0:
            os.makedirs(self.model_dir)

    def __del__(self):
        self.env.close()

    def train(self):
        if self.train_mode == 'nstep':
            self._train_nstep()
        elif self.train_mode == 'onestep':
            self._train_onestep()
        else:
            raise ValueError(f'{self.train_mode} is not a valid training mode')

    def _train_nstep(self) -> None:
        '''Default multi-step training loop for synchronous training over multiple environments.

        Most agents use this as-is and only override :meth:`rollout` and
        the model's ``backprop`` signature. Agents whose update doesn't
        fit this template (e.g. PPO with multiple epochs and minibatches)
        override this method directly.
        '''
        start = time.time()
        batch_size = self.num_envs * self.nsteps
        num_updates = self.total_steps // batch_size
        # main loop
        for t in range(self.t, num_updates + 1):
            states, actions, rewards, dones, values, last_values = self.rollout()
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
                    rewards,
                    values,
                    last_values,
                    dones,
                    gamma=self.gamma,
                    lambda_=self.lambda_,
                    clip=False,
                )
            # stack all states, actions and Rs from all workers into a single batch
            states, actions, R = fold_batch(states), fold_batch(actions), fold_batch(R)
            loss_value = self.model.backprop(states, R, actions)

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
                self.s += 1
                self.save(self.s)
                print('saved model')

            if (
                self.target_freq > 0 and t % (self.target_freq // batch_size) == 0
            ):  # update target network (for value based learning e.g. DQN)
                self.update_target()

            self.t += 1

    def rollout(self) -> Any:
        """Collect ``self.nsteps`` of experience and return whatever the agent's training loop expects."""
        raise NotImplementedError(f'{type(self).__name__} does not implement rollout')

    def nstep_return(self, rewards, last_values, dones, gamma=0.99, clip=False):
        if clip:
            rewards = np.clip(rewards, -1, 1)

        T = len(rewards)

        # Calculate R for advantage A = R - V
        R = np.zeros_like(rewards)
        R[-1] = last_values * (1 - dones[-1])

        for i in reversed(range(T - 1)):
            # restart score if done as BatchEnv automatically resets after end of episode
            R[i] = rewards[i] + gamma * R[i + 1] * (1 - dones[i])

        return R

    def lambda_return(
        self,
        rewards,
        values,
        last_values,
        dones,
        gamma=0.99,
        lambda_=0.8,
        clip=False,
    ):
        if clip:
            rewards = np.clip(rewards, -1, 1)
        T = len(rewards)
        # Calculate eligibility trace R^lambda
        R = np.zeros_like(rewards)
        R[-1] = last_values * (1 - dones[-1])
        for t in reversed(range(T - 1)):
            # restart score if done as BatchEnv automatically resets after end of episode
            R[t] = rewards[t] + gamma * (lambda_ * R[t + 1] + (1.0 - lambda_) * values[t + 1]) * (
                1 - dones[t]
            )

        return R

    def GAE(
        self,
        rewards,
        values,
        last_values,
        dones,
        gamma=0.99,
        lambda_=0.95,
        clip=False,
    ):
        if clip:
            rewards = np.clip(rewards, -1, 1)
        # Generalised Advantage Estimation
        Adv = np.zeros_like(rewards)
        Adv[-1] = rewards[-1] + gamma * last_values * (1 - dones[-1]) - values[-1]
        T = len(rewards)
        for t in reversed(range(T - 1)):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            Adv[t] = delta + gamma * lambda_ * Adv[t + 1] * (1 - dones[t])

        return Adv

    def validation_summary(self, t, loss, start, render):
        batch_size = self.num_envs * self.nsteps
        tot_steps = t * batch_size
        time_taken = time.time() - start
        frames_per_update = (self.validate_freq // batch_size) * batch_size
        fps = frames_per_update / time_taken

        score = self.validate_func(render)
        print(
            f"update {t}, validation score {score:f}, total steps {tot_steps}, "
            f"loss {loss:f}, time taken for {frames_per_update} frames:{time_taken:f}s, "
            f"fps {fps:f} \t\t\t"
        )

        if self.log_scalars:
            self.train_writer.add_scalar('validation/score', score, tot_steps)
            self.train_writer.add_scalar('train/loss', loss, tot_steps)

    def save_model(self, s):
        model_loc = f'{self.model_dir}/{s}.pt'
        # default saving method is to save session
        torch.save(self.model.state_dict(), model_loc)

    def load_model(self, modelname, model_dir="models/"):
        filename = model_dir + modelname + '.pt'
        if os.path.exists(filename):
            self.model.load_state_dict(torch.load(filename))
            print("loaded:", filename)
        else:
            print(filename, " does not exist")

    def base_attr(self):
        attributes = {
            'train_mode': self.train_mode,
            'total_steps': self.total_steps,
            'nsteps': self.nsteps,
            'return_type': self.return_type,
            'gamma': self.gamma,
            'lambda_': self.lambda_,
            'validate_freq': self.validate_freq,
            'num_val_episodes': self.num_val_episodes,
            'save_freq': self.save_freq,
            'render_freq': self.render_freq,
            'model_dir': self.model_dir,
            'train_log_dir': self.train_log_dir,
            's': self.s,
            't': self.t,
        }

        return attributes

    def local_attr(self, attr):
        # attr[variable] = z
        return attr

    def save(self, s):
        model_loc = str(self.model_dir + '/' + str(s) + '.trainer')
        attributes = self.base_attr()
        # add local variables to dict
        attributes = self.local_attr(attributes)
        with open(model_loc, 'w+') as file:
            json.dump(attributes, file)
        # save model
        self.save_model(s)

    def load(
        self,
        Class,
        model,
        model_checkpoint,
        envs,
        val_envs,
        filename,
        log_scalars=True,
        allow_gpu_growth=True,
        continue_train=True,
    ):
        with open(filename, 'r') as file:
            attrs = json.loads(file.read())
        s = attrs.pop('s')
        t = attrs.pop('t')
        attrs.pop('current_time')
        print(attrs)
        trainer = Class(
            envs=envs,
            model=model,
            val_envs=val_envs,
            log_scalars=log_scalars,
            gpu_growth=allow_gpu_growth,
            **attrs,
        )
        if continue_train:
            trainer.s = s
            trainer.t = t
        self.load_model(model_checkpoint, trainer.model_dir)
        return trainer

    def update_target(self) -> None:
        """Hook called every ``update_target_freq`` steps. No-op by default.

        Off-policy agents (e.g. DDQN) override this to copy weights from
        ``self.model`` to a target network. Pure on-policy agents leave
        ``update_target_freq=0`` and never call it.
        """
        raise NotImplementedError(f'{type(self).__name__} does not implement update_target')

    def _train_onestep(self) -> None:
        '''More efficient implementation of train_nstep when nsteps=1.'''
        raise NotImplementedError(f'{type(self).__name__} does not implement _train_onestep')

    def save_hyperparameters(self, filename, **kwargs):
        with open(filename, "w") as handle:
            for key, value in kwargs.items():
                handle.write(f"{key} = {value}\n")

    def validate_async(self, render=False):
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
                target=self._validate_async,
                args=(self.val_envs[i], num_val_eps[i], self.val_steps, render_array[i]),
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
        return score

    def _validate_async(self, env, num_ep, max_steps, render=False):
        'single env validation'
        from rlib.envs import wrap
        from rlib.envs.base import RLVecEnv

        rl_env = wrap(env)
        for _episode in range(num_ep):
            state, _info = rl_env.reset()
            episode_score = []
            for t in range(max_steps):
                action = self.get_action(state[np.newaxis])
                next_state, reward, terminated, truncated, info = rl_env.step(action)
                done = RLVecEnv.merge_done(terminated, truncated)
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

    def validate_sync(self, render=False):
        'batch env validation'
        episode_scores = []
        env = self.val_envs
        for _episode in range(self.num_val_episodes // len(env)):
            states = env.reset()
            episode_score = []
            for t in range(self.val_steps):
                actions = self.get_action(states)
                next_states, rewards, dones, infos = env.step(actions)
                states = next_states
                # print('state', state, 'action', action, 'reward', reward)

                episode_score.append(rewards * (1 - dones))

                if render:
                    with self.lock:
                        env.render()

                if dones.sum() == self.num_envs or t == self.val_steps - 1:
                    tot_reward = np.sum(np.stack(episode_score), axis=0)
                    episode_scores.append(tot_reward)
                    break

        return np.mean(episode_scores)

    def get_action(self, state: np.ndarray) -> Any:
        """Hook used by the default validation loops to pick an action.

        Concrete trainers should override this if they want to use
        :meth:`validate_sync` / :meth:`validate_async` directly.
        """
        raise NotImplementedError(
            'get_action method is required when using the default validation functions, '
            'check that this is implemented properly'
        )

    def fold_batch(self, x):
        rows, cols = x.shape[0], x.shape[1]
        y = x.reshape(rows * cols, *x.shape[2:])
        return y
