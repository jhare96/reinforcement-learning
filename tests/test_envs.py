"""Tests for :mod:`rlib.envs` (canonical contract + registry)."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pytest

from rlib.envs import RLEnv, RLVecEnv, make, register_backend, wrap
from rlib.envs.adapters import GymnasiumAdapter

# ---------------------------------------------------------------------------
# Registry / adapter resolution
# ---------------------------------------------------------------------------


def test_make_from_env_id_returns_rlenvbase() -> None:
    env = make("CartPole-v1")
    try:
        assert isinstance(env, RLEnv)
        assert isinstance(env, GymnasiumAdapter)
    finally:
        env.close()


def test_make_from_env_object_round_trip() -> None:
    raw = gym.make("CartPole-v1")
    env = make(raw)
    try:
        assert isinstance(env, RLEnv)
        obs, info = env.reset(seed=0)
        assert obs.shape == raw.observation_space.shape
        assert isinstance(info, dict)
    finally:
        env.close()


def test_wrap_passes_through_existing_rlenvbase() -> None:
    env = make("CartPole-v1")
    try:
        again = wrap(env)
        assert again is env
    finally:
        env.close()


def test_unknown_explicit_backend_raises() -> None:
    raw = gym.make("CartPole-v1")
    try:
        with pytest.raises(ValueError, match="Unknown backend"):
            wrap(raw, backend="not-a-real-backend")
    finally:
        raw.close()


def test_register_custom_backend() -> None:
    class Toy:
        observation_space = "obs"
        action_space = "act"

        def reset(self, *, seed: Any = None, options: Any = None) -> tuple[Any, dict]:
            return 0, {}

        def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
            return 0, 0.0, False, False, {}

        def close(self) -> None:
            return None

    class ToyAdapter(RLEnv):
        def __init__(self, env: Toy) -> None:
            self.env = env

        def reset(self, *, seed: Any = None, options: Any = None) -> tuple[Any, dict]:
            return self.env.reset(seed=seed, options=options)

        def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
            return self.env.step(action)

    register_backend(lambda e: isinstance(e, Toy), ToyAdapter, name="toy", prepend=True)
    wrapped = wrap(Toy())
    assert isinstance(wrapped, ToyAdapter)


# ---------------------------------------------------------------------------
# Canonical contract / 5-tuple semantics
# ---------------------------------------------------------------------------


def test_gymnasium_adapter_returns_5_tuple_step() -> None:
    env = make("CartPole-v1")
    try:
        env.reset(seed=0)
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(reward, (int, float, np.floating))
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        assert isinstance(info, dict)
        assert obs.shape == env.observation_space.shape
    finally:
        env.close()


def test_rlenv_protocol_is_runtime_checkable() -> None:
    env = make("CartPole-v1")
    try:
        assert isinstance(env, RLEnv)
    finally:
        env.close()


def test_rlvecenv_merge_done_or_logic() -> None:
    assert RLVecEnv.merge_done(False, False) is False
    assert RLVecEnv.merge_done(True, False) is True
    assert RLVecEnv.merge_done(False, True) is True
    assert RLVecEnv.merge_done(True, True) is True


def test_rlvecenv_merge_info_sets_truncated_flag() -> None:
    info = RLVecEnv.merge_info({}, terminated=False, truncated=True)
    assert info["TimeLimit.truncated"] is True

    info = RLVecEnv.merge_info({}, terminated=True, truncated=True)
    # Terminated takes precedence: not flagged as a pure truncation.
    assert info["TimeLimit.truncated"] is False


def test_rlvecenv_merge_info_does_not_overwrite_existing_key() -> None:
    info = RLVecEnv.merge_info({"TimeLimit.truncated": "preserved"}, False, True)
    assert info["TimeLimit.truncated"] == "preserved"
