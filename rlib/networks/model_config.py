"""Frozen dataclass configs for :class:`rlib.networks.Model` subclasses.

Each agent model historically took 5–15 hyperparameter kwargs in its
``__init__``, all duplicated across :class:`Model`, :class:`A2CModel`,
:class:`PPOModel` and the concrete subclasses.  The dataclasses below
collect them into a single immutable value object that:

* gives one canonical place to set defaults,
* makes it trivial to pickle / log / diff hyperparameter sets, and
* keeps the model constructors short.

Backward compatibility: every model still accepts the legacy keyword
arguments (``lr=...``, ``entropy_coeff=...``, ...).  When ``config`` is
not given they're bundled into a fresh dataclass internally; when it is
given the legacy kwargs are ignored.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ModelConfig", "A2CConfig", "PPOConfig"]


@dataclass(frozen=True)
class ModelConfig:
    """Shared hyperparameters for every :class:`rlib.networks.Model`.

    Attributes:
        lr: Initial learning rate.
        lr_final: Final learning rate the polynomial scheduler decays to.
        decay_steps: Optimiser steps over which the LR decays from
            ``lr`` to ``lr_final``.  Use ``lr_final == lr`` for a
            constant LR.
        grad_clip: Maximum gradient norm; ``None`` disables clipping.
        device: Torch device string (``"cuda"``, ``"cpu"``, ...).
    """

    lr: float = 1e-3
    lr_final: float = 0.0
    decay_steps: int = 600_000
    grad_clip: float | None = 0.5
    device: str = "cuda"


@dataclass(frozen=True)
class A2CConfig(ModelConfig):
    """Hyperparameters for advantage actor-critic models (A2C / A3C / UNREAL)."""

    entropy_coeff: float = 0.01
    value_coeff: float = 0.5


@dataclass(frozen=True)
class PPOConfig(ModelConfig):
    """Hyperparameters for clipped-objective PPO-family models (PPO / RND / DAAC policy)."""

    entropy_coeff: float = 0.01
    policy_clip: float = 0.1
