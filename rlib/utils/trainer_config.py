"""Frozen dataclass config for :class:`rlib.utils.SyncMultiEnvTrainer`.

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

from dataclasses import dataclass
from typing import Literal

__all__ = ["ReturnType", "TrainerConfig", "TrainMode"]


TrainMode = Literal["nstep", "onestep"]
ReturnType = Literal["nstep", "lambda", "GAE"]


@dataclass(frozen=True)
class TrainerConfig:
    """All hyperparameters for :class:`rlib.utils.SyncMultiEnvTrainer`.

    Attributes:
        train_mode: Either ``"nstep"`` (multi-step TD updates) or
            ``"onestep"`` (single-step TD updates).
        return_type: Return estimator — ``"nstep"``, ``"lambda"`` or
            ``"GAE"``.
        total_steps: Total environment steps across all parallel envs.
        nsteps: Length of each n-step rollout.
        gamma: Discount factor.
        lambda_: GAE / λ-return weighting.
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
    train_mode: TrainMode = "nstep"
    return_type: ReturnType = "nstep"
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

    def __post_init__(self) -> None:
        if self.train_mode not in ("nstep", "onestep"):
            raise ValueError(f"train_mode must be 'nstep' or 'onestep', got {self.train_mode!r}")
        if self.return_type not in ("nstep", "lambda", "GAE"):
            raise ValueError(
                f"return_type must be 'nstep', 'lambda' or 'GAE', got {self.return_type!r}"
            )
