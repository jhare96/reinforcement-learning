"""Utilities: vectorised envs, wrappers, schedulers, replay memory.

The canonical, backend-agnostic environment API lives in
:mod:`rlib.envs` (``RLEnv``, ``RLEnvBase``, ``RLVecEnv``, ``make``,
``wrap``, ``register_backend``).
"""

from rlib.utils.trainer_config import ReturnType, TrainerConfig, TrainMode

__all__ = ["ReturnType", "TrainerConfig", "TrainMode"]
