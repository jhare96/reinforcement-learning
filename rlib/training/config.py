"""Frozen dataclass config for :class:`rlib.training.SyncMultiEnvTrainer`.

The trainer takes ~17 hyperparameter kwargs in its ``__init__``, all
forwarded by every concrete trainer subclass via
``super().__init__(...)``.  :class:`TrainerConfig` collects them into a
single immutable value object so:

* concrete trainers can declare just the few kwargs they actually
  customise and forward the rest as a single ``config`` argument;
* the full hyperparameter set is trivial to log / diff / pickle; and
* callers can build presets (Atari, classic-control, sparse-reward) and
  share them across agents.

Per-agent trainer configs (``PPOTrainerConfig``, ``RNDTrainerConfig``,
...) live next to their trainer class in ``rlib/<Agent>/trainer.py``
since each is 1-to-1 with a single trainer.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from rlib.training.returns import Returns

__all__ = ["TrainMode", "TrainerConfig"]


class TrainMode(str, enum.Enum):
    """Whether the trainer dispatches to ``_train_nstep`` or ``_train_onestep``."""

    NSTEP = "nstep"
    ONESTEP = "onestep"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class TrainerConfig:
    """All hyperparameters for :class:`rlib.training.SyncMultiEnvTrainer`.

    Attributes:
        train_mode: Whether to dispatch to the multi-step
            (:attr:`TrainMode.NSTEP`) or one-step
            (:attr:`TrainMode.ONESTEP`) training loop.
        returns: Return / advantage estimator (:class:`Returns` enum).
        total_steps: Total environment steps across all parallel envs.
        nsteps: Length of each n-step rollout.
        gamma: Discount factor.
        lambda_: GAE / λ-return weighting (ignored by ``Returns.NSTEP``).
        validate_freq: Env steps between validation passes; ``0``
            disables validation.
        num_val_episodes: Episodes averaged per validation pass.
        max_val_steps: Per-episode step cap during validation.
        log_dir: Directory for tensorboard scalars.
        model_dir: Directory for checkpoints.
        save_freq: Env steps between checkpoints; ``0`` disables saving.
        log_scalars: Whether to write tensorboard scalars at all.
        update_target_freq: Env steps between target-net syncs (off-policy
            agents only); ``0`` disables.
        render_freq: Multiple of ``validate_freq`` between renders;
            ``0`` disables rendering.
    """

    # Training schedule
    train_mode: TrainMode = TrainMode.NSTEP
    returns: Returns = Returns.NSTEP
    total_steps: int = 50_000_000
    nsteps: int = 5
    gamma: float = 0.99
    lambda_: float = 0.95

    # Validation
    validate_freq: int = 1_000_000
    num_val_episodes: int = 50
    max_val_steps: int = 10_000

    # IO / logging
    log_dir: str = "logs/"
    model_dir: str = "models/"
    save_freq: int = 0
    log_scalars: bool = True

    # Off-policy hooks
    update_target_freq: int = 0
    render_freq: int = 0
