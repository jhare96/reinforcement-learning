"""Training infrastructure: synchronous multi-env trainer, configs,
return estimators, and validation strategies.

The trainer + base config that every concrete agent extends:

* :class:`SyncMultiEnvTrainer` — the main training loop.
* :class:`TrainerConfig` — base hyperparameter dataclass.

Per-trainer configs (``PPOTrainerConfig``, ``RNDTrainerConfig``, ...)
live next to their trainer class in ``rlib/<Agent>/trainer.py``.

Internals exposed for advanced use / testing:

* :data:`RETURN_FUNCTIONS` — dispatch table for n-step / GAE / λ-return.
* :class:`Validator`, :class:`AsyncValidator`, :class:`SyncValidator`,
  :func:`make_validator` — validation strategy implementations.
"""

from rlib.training.config import ReturnType, TrainerConfig, TrainMode
from rlib.training.returns import GAE, RETURN_FUNCTIONS, lambda_return, nstep_return
from rlib.training.trainer import SyncMultiEnvTrainer
from rlib.training.validation import AsyncValidator, SyncValidator, Validator, make_validator

__all__ = [
    "GAE",
    "AsyncValidator",
    "RETURN_FUNCTIONS",
    "ReturnType",
    "SyncMultiEnvTrainer",
    "SyncValidator",
    "TrainMode",
    "TrainerConfig",
    "Validator",
    "lambda_return",
    "make_validator",
    "nstep_return",
]
