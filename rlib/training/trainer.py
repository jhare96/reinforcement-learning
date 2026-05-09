import dataclasses
import json
import os
import threading
import time
from typing import Any

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from rlib.envs.vec_env import BatchEnv, DummyBatchEnv
from rlib.networks import Model
from rlib.training.config import TrainerConfig, TrainMode
from rlib.training.validation import Validator, make_validator
from rlib.utils.utils import fold_batch


class SyncMultiEnvTrainer:
    """Synchronous multi-env training framework for any :class:`rlib.networks.Model`."""

    model: Model
    config: TrainerConfig
    validator: Validator
    train_writer: SummaryWriter
    train_log_dir: str

    def __init__(
        self,
        envs: BatchEnv | DummyBatchEnv,
        model: Model,
        val_envs: list | BatchEnv | DummyBatchEnv,
        config: TrainerConfig,
    ) -> None:
        '''Build a synchronous multi-env training loop.

        Args:
            envs: training environments (``BatchEnv`` or ``DummyBatchEnv``).
            model: an :class:`rlib.networks.Model` subclass.
            val_envs: validation envs — a ``list`` (uses threading),
                a ``BatchEnv`` (multiprocessing), or a ``DummyBatchEnv``
                (in-process).
            config: a :class:`TrainerConfig` carrying all training
                hyperparameters.
        '''
        self.config = config

        self.env = envs
        self.validator = make_validator(val_envs)
        assert config.num_val_episodes >= len(val_envs), (
            f'number of validation epsiodes {config.num_val_episodes} must be greater than or '
            f'equal to the number of validation envs {len(val_envs)}'
        )
        self.num_envs = len(envs)
        self.env_id = envs.spec.id
        self.val_envs = val_envs
        self.model = model

        # Mirror config fields onto ``self`` for ergonomics — most of
        # the legacy training-loop code reads ``self.gamma`` etc. directly.
        self.train_mode = config.train_mode
        self.total_steps = config.total_steps
        self.nsteps = config.nsteps
        self.returns = config.returns
        self.gamma = config.gamma
        self.lambda_ = config.lambda_
        self.validate_freq = config.validate_freq
        self.num_val_episodes = config.num_val_episodes
        self.val_steps = config.max_val_steps
        self.save_freq = config.save_freq
        self.render_freq = config.render_freq
        self.target_freq = config.update_target_freq
        self.log_scalars = config.log_scalars
        self.log_dir = config.log_dir
        self.model_dir = config.model_dir

        self.lock = threading.Lock()
        # Kept for backwards-compatibility: agents with custom recurrent
        # validation loops (A2C-LSTM, UNREAL-LSTM, VIN) push per-episode
        # scores onto this list under ``self.lock``.
        self.validate_rewards: list[Any] = []
        self.s = 0  # number of saves made
        self.t = 1  # number of updates done
        self.states = self.env.reset()

        if config.log_scalars:
            self.train_log_dir = config.log_dir + '/train'
            self.train_writer = SummaryWriter(self.train_log_dir)
            self._log_hyperparameters()

        if not os.path.exists(self.model_dir) and config.save_freq > 0:
            os.makedirs(self.model_dir)

    def _log_hyperparameters(self) -> None:
        """Dump the trainer + model configs to ``<log_dir>/hyperparameters.txt``.

        Uses :func:`dataclasses.asdict` so adding a field to any
        ``*TrainerConfig`` or ``*ModelConfig`` automatically shows up
        in the log without touching agent code.  Subclasses can override
        this hook if they want to add fields not present on the configs
        (e.g. derived values like ``num_workers``).
        """
        if not os.path.exists(self.config.log_dir):
            os.makedirs(self.config.log_dir, exist_ok=True)
        params: dict[str, Any] = {
            **dataclasses.asdict(self.config),
            'num_workers': self.num_envs,
        }
        if hasattr(self.model, 'config'):
            params.update(
                {f'model.{k}': v for k, v in dataclasses.asdict(self.model.config).items()}
            )
        filename = self.config.log_dir + '/hyperparameters.txt'
        self.save_hyperparameters(filename, **params)

    def __del__(self):
        self.env.close()

    def train(self):
        if self.train_mode is TrainMode.NSTEP:
            self._train_nstep()
        elif self.train_mode is TrainMode.ONESTEP:
            self._train_onestep()
        else:
            raise ValueError(f'{self.train_mode!r} is not a valid training mode')

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
        return_fn = self.config.returns
        # main loop
        for t in range(self.t, num_updates + 1):
            states, actions, rewards, dones, values, last_values = self.rollout()
            R = return_fn(rewards, values, last_values, dones, self.gamma, self.lambda_)
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

    def validation_summary(self, t, loss, start, render):
        batch_size = self.num_envs * self.nsteps
        tot_steps = t * batch_size
        time_taken = time.time() - start
        frames_per_update = (self.validate_freq // batch_size) * batch_size
        fps = frames_per_update / time_taken

        score = self._validation_score(render)
        print(
            f"update {t}, validation score {score:f}, total steps {tot_steps}, "
            f"loss {loss:f}, time taken for {frames_per_update} frames:{time_taken:f}s, "
            f"fps {fps:f} \t\t\t"
        )

        if self.log_scalars:
            self.train_writer.add_scalar('validation/score', score, tot_steps)
            self.train_writer.add_scalar('train/loss', loss, tot_steps)

    def _validation_score(self, render: bool) -> float:
        """Return a single mean validation score.

        Defaults to dispatching through ``self.validator``.  Recurrent
        agents whose validation loop needs hidden-state plumbing
        (A2C-LSTM, UNREAL-LSTM) override this to call their own custom
        validate_sync/validate_async methods, which still use
        ``self.lock`` and ``self.validate_rewards`` for thread-safe
        score collection.
        """
        return self.validator.run(
            self.get_action,
            num_episodes=self.num_val_episodes,
            max_steps=self.val_steps,
            render=render,
        )

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
            'returns': self.returns,
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

    def get_action(self, state: np.ndarray) -> Any:
        """Hook used by the validator to pick an action during evaluation.

        Concrete trainers must override this if validation is enabled
        (it's called by every :class:`~rlib.training.validation.Validator`).
        """
        raise NotImplementedError(
            'get_action method is required when validation is enabled, '
            'check that this is implemented properly'
        )
