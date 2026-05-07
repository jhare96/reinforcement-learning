"""Tests for the lightweight wrappers in :mod:`rlib.utils.wrappers`."""

from __future__ import annotations

from typing import Any

import numpy as np

from rlib.envs import RLEnvBase, make
from rlib.envs.wrappers import (
    ClipRewardEnv,
    DummyEnv,
    NoRewardEnv,
    TimeLimitEnv,
)


class _ConstantRewardEnv(RLEnvBase):
    """Minimal hand-rolled env emitting a constant reward each step.

    ``observation_space`` / ``action_space`` are properties on
    :class:`RLEnvBase`; we override them as plain attributes via the
    class body below so the abstract base contract is satisfied.
    """

    observation_space = None
    action_space = None

    def __init__(self, reward: float = 5.0) -> None:
        self.env = None
        self._reward = reward

    def reset(self, *, seed: Any = None, options: Any = None) -> tuple[Any, dict]:
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        return np.zeros(2, dtype=np.float32), self._reward, False, False, {}


def test_dummy_env_passes_through() -> None:
    inner = _ConstantRewardEnv(reward=2.5)
    env = DummyEnv(inner)
    obs, info = env.reset()
    assert obs.shape == (2,)
    assert info == {}
    out = env.step(0)
    assert out[1] == 2.5


def test_clip_reward_env_clips_to_unit_interval() -> None:
    big = ClipRewardEnv(_ConstantRewardEnv(reward=42.0))
    small = ClipRewardEnv(_ConstantRewardEnv(reward=-100.0))
    assert big.step(0)[1] == 1.0
    assert small.step(0)[1] == -1.0


def test_no_reward_env_zeros_rewards() -> None:
    env = NoRewardEnv(_ConstantRewardEnv(reward=99.0))
    assert env.step(0)[1] == 0


def test_time_limit_env_truncates_after_limit() -> None:
    env = TimeLimitEnv(_ConstantRewardEnv(), time_limit=3)
    env.reset()
    truncs = []
    for _ in range(5):
        _obs, _r, terminated, truncated, _info = env.step(0)
        truncs.append((terminated, truncated))
    # First 3 calls increment _step to 1,2,3 (none > 3); 4th tick -> _step=4 > 3.
    assert truncs[3][1] is True
    assert truncs[4][1] is True


def test_time_limit_env_resets_counter() -> None:
    env = TimeLimitEnv(_ConstantRewardEnv(), time_limit=2)
    env.reset()
    for _ in range(3):
        env.step(0)
    # Should be truncated by now.
    assert env.step(0)[3] is True
    env.reset()
    # After reset, counter is back to zero.
    _o, _r, _term, trunc, _info = env.step(0)
    assert trunc is False


def test_clip_reward_env_works_on_real_env() -> None:
    inner = make("CartPole-v1")
    try:
        env = ClipRewardEnv(inner)
        env.reset(seed=0)
        # CartPole always emits +1 reward; clip leaves it at +1.
        assert env.step(0)[1] == 1.0
    finally:
        inner.close()
