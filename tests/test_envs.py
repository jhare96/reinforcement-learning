"""Tests for :mod:`rlib.envs` (canonical contract + built-in envs)."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from rlib.envs import RLEnv, RLVecEnv, make

# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_make_is_gymnasium_make() -> None:
    """``rlib.envs.make`` is a re-export of :func:`gymnasium.make`."""
    assert make is gym.make


def test_make_cartpole_returns_gym_env() -> None:
    env = make("CartPole-v1")
    try:
        assert isinstance(env, gym.Env)
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Built-in env registration (ApplePicker)
# ---------------------------------------------------------------------------


def test_apple_picker_registered() -> None:
    env = make("ApplePicker-v0", num_objects=3)
    try:
        obs, info = env.reset()
        assert isinstance(info, dict)
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 5
        _, reward, terminated, truncated, info = result
        assert isinstance(reward, (int, float, np.floating))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    finally:
        env.close()


def test_apple_picker_deterministic_registered() -> None:
    env = make("ApplePickerDeterministic-v0", num_objects=3)
    try:
        obs1, _ = env.reset()
        obs2, _ = env.reset()
        # Deterministic: same start state and item layout across resets.
        np.testing.assert_array_equal(obs1, obs2)
    finally:
        env.close()


# ---------------------------------------------------------------------------
# RLVecEnv helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# RLEnv ABC contract
# ---------------------------------------------------------------------------


def test_rlenv_subclass_must_implement_reset_step() -> None:
    """An ``RLEnv`` subclass without ``reset`` / ``step`` cannot instantiate."""

    class Incomplete(RLEnv):
        pass

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]
