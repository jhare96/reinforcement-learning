"""Generic, backend-agnostic environment abstraction for rlib.

This subpackage replaces the previous ``rlib.utils.gym_compat`` shim
(now removed).
It exposes a single canonical environment contract — the modern
5-tuple Gymnasium API — together with one thin adapter per supported
backend.  See :mod:`rlib.envs.base` for the design rationale.

Public surface:

* :class:`RLEnv` — :class:`typing.Protocol` for type annotations.
* :class:`RLEnv` — abstract base that adapters and wrappers extend.
* :class:`RLVecEnv` — abstract base for vectorised env runners.
* :func:`make` — construct or wrap an env.
* :func:`wrap` — wrap an already-constructed env.
* :func:`register_backend` — teach rlib about a new env type.

Concrete vec-env runners and the agent-suite wrappers live in
:mod:`rlib.envs.vec_env` and :mod:`rlib.envs.wrappers` respectively.
The most-used names are re-exported here for convenience.
"""

from rlib.envs.base import RLEnv, RLVecEnv
from rlib.envs.registry import make, register_backend, wrap
from rlib.envs.vec_env import BatchEnv, DummyBatchEnv

__all__ = [
    "BatchEnv",
    "DummyBatchEnv",
    "RLEnv",
    "RLEnv",
    "RLVecEnv",
    "make",
    "register_backend",
    "wrap",
]
