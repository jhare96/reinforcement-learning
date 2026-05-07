"""Compatibility shim between Gymnasium and the legacy Gym API.

The library has historically targeted ``gym``, which is unmaintained and
exposes the classic 4-tuple ``(obs, reward, done, info)`` step API.
``gymnasium`` is the maintained successor and uses a 5-tuple
``(obs, reward, terminated, truncated, info)`` step API together with a
``reset()`` that returns ``(obs, info)`` instead of just ``obs``.

To keep all existing agent rollout code working with both backends, this
module:

1. Re-exports the active backend as :data:`gym` (preferring ``gymnasium``).
2. Provides :func:`step_compat` and :func:`reset_compat` helpers that always
   return the legacy 4-tuple / single-observation form expected by the
   trainer and the agents.

Wrappers and :class:`~rlib.utils.VecEnv.Worker` use these helpers so that
agents continue to receive the original ``(obs, reward, done, info)``
contract regardless of the installed backend.
"""

from __future__ import annotations

from typing import Any, Tuple

try:  # pragma: no cover - exercised at import time
    import gymnasium as gym  # type: ignore

    GYMNASIUM = True
except ImportError:  # pragma: no cover - legacy fallback
    import gym  # type: ignore

    GYMNASIUM = False


def step_compat(env, action) -> Tuple[Any, Any, Any, dict]:
    """Step ``env`` and return the legacy 4-tuple ``(obs, reward, done, info)``.

    When ``env`` follows the gymnasium 5-tuple API, ``terminated`` and
    ``truncated`` are merged into a single ``done`` flag using a logical OR.
    """
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = bool(terminated) or bool(truncated)
        if isinstance(info, dict):
            info = dict(info)
            info.setdefault("TimeLimit.truncated", bool(truncated) and not bool(terminated))
        return obs, reward, done, info
    return result  # already 4-tuple (legacy gym)


def reset_compat(env, **kwargs) -> Any:
    """Reset ``env`` and return only the observation.

    Gymnasium's ``reset`` returns ``(obs, info)``; legacy ``gym`` returns just
    ``obs``. Callers that only need the observation can use this helper to
    stay backend-agnostic.
    """
    result = env.reset(**kwargs)
    if isinstance(result, tuple) and len(result) == 2:
        return result[0]
    return result


__all__ = ["gym", "GYMNASIUM", "step_compat", "reset_compat"]
