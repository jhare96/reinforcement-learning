"""Utilities: vectorised envs, wrappers, schedulers, replay memory.

Note:
    The legacy :mod:`rlib.utils.gym_compat` shim is deprecated. Use
    :mod:`rlib.envs` for the canonical, backend-agnostic env API
    (``RLEnv``, ``RLEnvBase``, ``RLVecEnv``, ``make``,
    ``register_backend``).
"""
from rlib.utils.gym_compat import gym, step_compat, reset_compat, GYMNASIUM

__all__ = ["gym", "step_compat", "reset_compat", "GYMNASIUM"]
