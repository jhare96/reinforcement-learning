"""Training infrastructure: synchronous multi-env trainer, configs,
return estimators, and validation strategies.

The trainer + base config that every concrete agent extends:

* :class:`SyncMultiEnvTrainer` — the main training loop.
* :class:`TrainerConfig` — base hyperparameter dataclass.

Per-trainer configs (``PPOTrainerConfig``, ``RNDTrainerConfig``, ...)
live next to their trainer class in ``rlib/<Agent>/trainer.py``.

Internals exposed for advanced use / testing:

* :class:`Returns` — enum of return / advantage estimators.
* :func:`GAE`, :func:`lambda_return`, :func:`nstep_return` — the
  underlying free functions.
* :class:`Validator`, :class:`AsyncValidator`, :class:`SyncValidator`,
  :func:`make_validator` — validation strategy implementations.
"""

from rlib.training.config import TrainerConfig, TrainMode
from rlib.training.returns import GAE, Returns, lambda_return, nstep_return
from rlib.training.trainer import SyncMultiEnvTrainer
from rlib.training.validation import AsyncValidator, SyncValidator, Validator, make_validator

__all__ = [
    "GAE",
    "AsyncValidator",
    "Returns",
    "SyncMultiEnvTrainer",
    "SyncValidator",
    "TrainMode",
    "TrainerConfig",
    "Validator",
    "lambda_return",
    "make_validator",
    "nstep_return",
]
