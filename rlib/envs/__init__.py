"""Generic, backend-agnostic environment abstraction for rlib.

This subpackage replaces the previous ``rlib.utils.gym_compat`` shim
(now removed).
It exposes a single canonical environment contract — the modern
5-tuple Gymnasium API — together with one thin adapter per supported
backend.  See :mod:`rlib.envs.base` for the design rationale.

Public surface:

* :class:`RLEnv` — :class:`typing.Protocol` for type annotations.
* :class:`RLEnvBase` — abstract base that adapters and wrappers extend.
* :class:`RLVecEnv` — abstract base for vectorised env runners.
* :func:`make` — construct or wrap an env.
* :func:`wrap` — wrap an already-constructed env.
* :func:`register_backend` — teach rlib about a new env type.
"""

from rlib.envs.base import RLEnv, RLEnvBase, RLVecEnv
from rlib.envs.registry import make, register_backend, wrap

__all__ = [
    "RLEnv",
    "RLEnvBase",
    "RLVecEnv",
    "make",
    "wrap",
    "register_backend",
]
