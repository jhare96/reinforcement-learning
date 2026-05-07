"""Adapter from a Gymnasium ``Env`` to rlib's :class:`RLEnv` contract.

Gymnasium already exposes the canonical 5-tuple ``step`` and
``(obs, info)`` ``reset``, so this adapter is essentially a typed
pass-through.  It exists so that:

* The rlib codebase has a single, uniform entry point
  (``rlib.envs.make``) regardless of backend.
* ``isinstance(env, RLEnvBase)`` works for sniffing.
* Future backend-specific quirks have an obvious home.
"""

from __future__ import annotations

from typing import Any

from gymnasium import Env

from rlib.envs.base import RLEnvBase


class GymnasiumAdapter(RLEnvBase):
    """Wrap a ``gymnasium.Env`` as an :class:`~rlib.envs.RLEnv`."""

    def __init__(self, env: Env) -> None:
        self.env = env

    def reset(self, *, seed: Any = None, options: Any = None) -> tuple[Any, dict]:
        kwargs: dict[str, Any] = {}
        if seed is not None:
            kwargs["seed"] = seed
        if options is not None:
            kwargs["options"] = options
        result = self.env.reset(**kwargs)
        # Defensive: a custom env may still return only obs.
        if isinstance(result, tuple) and len(result) == 2:
            return result
        return result, {}

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        return self.env.step(action)


__all__ = ["GymnasiumAdapter"]
