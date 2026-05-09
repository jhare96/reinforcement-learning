"""Adapter from a legacy ``gym.Env`` to rlib's :class:`RLEnv` contract.

Legacy ``gym`` (pre-0.26) exposes a 4-tuple ``step`` and a single-obs
``reset``.  This adapter translates both to the modern 5-tuple /
``(obs, info)`` shape:

* ``truncated`` is inferred from ``info["TimeLimit.truncated"]`` when
  present (the convention used by ``gym.wrappers.TimeLimit``); otherwise
  it defaults to ``False``.
* ``terminated`` is set to ``done and not truncated`` so the two flags
  are mutually exclusive, matching Gymnasium semantics.
"""

from __future__ import annotations

import contextlib
from typing import Any

from gym import Env

from rlib.envs.base import RLEnv


class LegacyGymAdapter(RLEnv):
    """Wrap a legacy (pre-0.26) ``gym.Env`` as an :class:`~rlib.envs.RLEnv`."""

    def __init__(self, env: Env) -> None:
        self.env = env

    def reset(self, *, seed: Any = None, options: Any = None) -> tuple[Any, dict]:
        # Legacy gym ignores keyword args; pass only what we can.
        if seed is not None and hasattr(self.env, "seed"):
            with contextlib.suppress(Exception):
                self.env.seed(seed)
        result = self.env.reset()
        if isinstance(result, tuple) and len(result) == 2:
            # Some "legacy-ish" envs already return (obs, info).
            return result
        return result, {}

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        result = self.env.step(action)
        if len(result) == 5:
            # Already modern; pass through.
            return result
        obs, reward, done, info = result
        truncated = False
        if isinstance(info, dict) and info.get("TimeLimit.truncated"):
            truncated = True
        terminated = bool(done) and not truncated
        return obs, reward, terminated, truncated, info


__all__ = ["LegacyGymAdapter"]
