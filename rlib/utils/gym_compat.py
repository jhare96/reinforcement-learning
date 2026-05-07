"""Deprecated — use :mod:`rlib.envs` instead.

This module previously implemented a small Gymnasium ↔ legacy-Gym shim
exposing ``step_compat`` / ``reset_compat`` free functions that the
rest of the library called everywhere it touched an env.  That design
was awkward (a side channel of free functions callers had to remember
to use, freezing the codebase on the deprecated 4-tuple step API) and
did not generalise to other gym-like backends (PettingZoo, EnvPool,
``dm_env``, in-house simulators, ...).

The replacement lives in :mod:`rlib.envs`:

* :class:`rlib.envs.RLEnv` — :class:`typing.Protocol` for the
  canonical 5-tuple env contract.
* :class:`rlib.envs.RLEnvBase` — abstract base for adapters / wrappers.
* :func:`rlib.envs.make` / :func:`rlib.envs.wrap` — the single entry
  point for constructing or wrapping an env.
* :func:`rlib.envs.register_backend` — extension hook for new backends.

For backward compatibility this module still re-exports the active
``gym`` (preferring ``gymnasium``) so existing ``from
rlib.utils.gym_compat import gym`` imports keep working, and
``step_compat`` / ``reset_compat`` are still callable but emit a
``DeprecationWarning`` and forward to the new adapters.

This shim will be removed in a future release; please migrate to
``rlib.envs`` at your earliest convenience.
"""

from __future__ import annotations

import warnings
from typing import Any

try:  # pragma: no cover - exercised at import time
    import gymnasium as gym  # type: ignore

    GYMNASIUM = True
except ImportError:  # pragma: no cover - legacy fallback
    import gym  # type: ignore

    GYMNASIUM = False


_DEPRECATION_MSG = (
    "rlib.utils.gym_compat.{name} is deprecated and will be removed in a "
    "future release. Use rlib.envs.wrap(env) (returns an RLEnvBase that "
    "speaks the modern 5-tuple step / (obs, info) reset API) instead."
)


def step_compat(env: Any, action: Any) -> tuple[Any, Any, bool, dict]:
    """Deprecated wrapper around :class:`rlib.envs.RLEnvBase` step.

    Returns the legacy 4-tuple ``(obs, reward, done, info)`` for
    backward compatibility.
    """
    warnings.warn(
        _DEPRECATION_MSG.format(name="step_compat"),
        DeprecationWarning,
        stacklevel=2,
    )
    from rlib.envs import wrap
    from rlib.envs.base import RLVecEnv

    rl_env = wrap(env)
    obs, reward, terminated, truncated, info = rl_env.step(action)
    done = RLVecEnv.merge_done(terminated, truncated)
    info = RLVecEnv.merge_info(info, terminated, truncated)
    return obs, reward, done, info


def reset_compat(env: Any, **kwargs: Any) -> Any:
    """Deprecated wrapper around :class:`rlib.envs.RLEnvBase` reset.

    Returns only the observation (legacy single-value reset) for
    backward compatibility.
    """
    warnings.warn(
        _DEPRECATION_MSG.format(name="reset_compat"),
        DeprecationWarning,
        stacklevel=2,
    )
    from rlib.envs import wrap

    rl_env = wrap(env)
    obs, _info = rl_env.reset(**kwargs)
    return obs


__all__ = ["gym", "GYMNASIUM", "step_compat", "reset_compat"]
