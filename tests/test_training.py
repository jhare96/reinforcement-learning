"""Tests for :mod:`rlib.training.validation` and :mod:`rlib.training.returns`."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from rlib.envs.vec_env import DummyBatchEnv
from rlib.training import (
    AsyncValidator,
    SyncValidator,
    Validator,
    make_validator,
)
from rlib.training.returns import GAE, Returns, lambda_return, nstep_return

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class _ZeroAction:
    """Minimal action picker that always returns action 0 (or a zero vector)."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, state: np.ndarray):
        self.calls += 1
        # state shape is (n_envs, ...) — return scalar 0 for the async path,
        # vector for the sync path. The vec env decides what shape it wants.
        if state.ndim == 1 or state.shape[0] == 1:
            return 0
        return np.zeros(state.shape[0], dtype=np.int64)


class TestMakeValidator:
    def test_dispatches_async_for_list(self) -> None:
        envs = [gym.make("CartPole-v1") for _ in range(2)]
        try:
            v = make_validator(envs)
            assert isinstance(v, AsyncValidator)
            assert isinstance(v, Validator)
        finally:
            for e in envs:
                e.close()

    def test_dispatches_sync_for_batched(self) -> None:
        envs = DummyBatchEnv(lambda e: e, "CartPole-v1", num_envs=2)
        try:
            v = make_validator(envs)
            assert isinstance(v, SyncValidator)
            assert isinstance(v, Validator)
        finally:
            envs.close()


class TestSyncValidator:
    def test_returns_finite_score(self) -> None:
        envs = DummyBatchEnv(lambda e: e, "CartPole-v1", num_envs=2)
        try:
            v = SyncValidator(envs)
            score = v.run(_ZeroAction(), num_episodes=2, max_steps=20)
        finally:
            envs.close()
        assert isinstance(score, float)
        assert score >= 0  # CartPole reward is non-negative

    def test_zero_episodes_returns_zero(self) -> None:
        envs = DummyBatchEnv(lambda e: e, "CartPole-v1", num_envs=2)
        try:
            v = SyncValidator(envs)
            score = v.run(_ZeroAction(), num_episodes=0, max_steps=10)
        finally:
            envs.close()
        assert score == 0.0


class TestAsyncValidator:
    def test_returns_finite_score(self) -> None:
        envs = [gym.make("CartPole-v1") for _ in range(2)]
        try:
            v = AsyncValidator(envs)
            score = v.run(_ZeroAction(), num_episodes=2, max_steps=20)
        finally:
            for e in envs:
                e.close()
        assert isinstance(score, float)
        assert score >= 0

    def test_distributes_episodes_across_envs(self) -> None:
        envs = [gym.make("CartPole-v1") for _ in range(2)]
        try:
            v = AsyncValidator(envs)
            v.run(_ZeroAction(), num_episodes=4, max_steps=10)
        finally:
            for e in envs:
                e.close()

    def test_rejects_non_list(self) -> None:
        envs = DummyBatchEnv(lambda e: e, "CartPole-v1", num_envs=2)
        try:
            with pytest.raises(TypeError, match="list of envs"):
                AsyncValidator(envs)  # type: ignore[arg-type]
        finally:
            envs.close()


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------


class TestReturnFunctions:
    def test_enum_has_all_three_members(self) -> None:
        assert {m.name for m in Returns} == {"NSTEP", "GAE", "LAMBDA"}

    def test_nstep_enum_matches_direct_call(self) -> None:
        rewards = np.array([[1.0], [0.5], [0.0]], dtype=np.float32)
        values = np.array([[0.1], [0.2], [0.3]], dtype=np.float32)
        last_values = np.array([0.5], dtype=np.float32)
        dones = np.zeros_like(rewards)
        direct = nstep_return(rewards, last_values, dones, gamma=0.95)
        via_enum = Returns.NSTEP(rewards, values, last_values, dones, 0.95, 0.95)
        np.testing.assert_array_equal(direct, via_enum)

    def test_gae_enum_returns_targets_not_advantages(self) -> None:
        rewards = np.array([[1.0], [0.5], [0.0]], dtype=np.float32)
        values = np.array([[0.1], [0.2], [0.3]], dtype=np.float32)
        last_values = np.array([0.5], dtype=np.float32)
        dones = np.zeros_like(rewards)
        targets = Returns.GAE(rewards, values, last_values, dones, 0.95, 0.95)
        # Targets = advantages + values; advantages alone should be smaller in magnitude
        adv = GAE(rewards, values, last_values, dones, gamma=0.95, lambda_=0.95)
        np.testing.assert_allclose(targets, adv + values, rtol=1e-6)

    def test_lambda_enum_matches_direct_call(self) -> None:
        rewards = np.array([[1.0], [0.5], [0.0]], dtype=np.float32)
        values = np.array([[0.1], [0.2], [0.3]], dtype=np.float32)
        last_values = np.array([0.5], dtype=np.float32)
        dones = np.zeros_like(rewards)
        direct = lambda_return(rewards, values, last_values, dones, gamma=0.99, lambda_=0.9)
        via_enum = Returns.LAMBDA(rewards, values, last_values, dones, 0.99, 0.9)
        np.testing.assert_allclose(direct, via_enum, rtol=1e-6)

    def test_str_returns_name(self) -> None:
        assert str(Returns.GAE) == "GAE"
        assert str(Returns.NSTEP) == "NSTEP"
        assert str(Returns.LAMBDA) == "LAMBDA"
