"""Tests for :class:`rlib.utils.VecEnv.DummyBatchEnv`.

The subprocess-backed ``BatchEnv`` is intentionally not exercised here
to keep CI fast and avoid ``multiprocessing`` flakiness on shared
runners.
"""

from __future__ import annotations

import numpy as np

from rlib.envs import RLEnvBase, wrap
from rlib.utils.VecEnv import DummyBatchEnv


def _identity_constructor(env: object) -> RLEnvBase:
    return wrap(env)


def test_dummy_batch_env_reset_stacks_observations() -> None:
    vec = DummyBatchEnv(_identity_constructor, "CartPole-v1", num_envs=4)
    try:
        obs = vec.reset()
        assert obs.shape == (4, 4)
        assert obs.dtype.kind == "f"
    finally:
        vec.close()


def test_dummy_batch_env_step_returns_legacy_4_tuple() -> None:
    vec = DummyBatchEnv(_identity_constructor, "CartPole-v1", num_envs=3)
    try:
        vec.reset()
        actions = np.array([0, 1, 0], dtype=np.int64)
        out = vec.step(actions)
        assert len(out) == 4
        obs, rewards, dones, infos = out
        assert obs.shape == (3, 4)
        assert rewards.shape == (3,)
        assert dones.shape == (3,)
        assert dones.dtype == bool
        assert isinstance(infos, tuple)
        assert len(infos) == 3
        assert all(isinstance(i, dict) for i in infos)
    finally:
        vec.close()


def test_dummy_batch_env_len() -> None:
    vec = DummyBatchEnv(_identity_constructor, "CartPole-v1", num_envs=2)
    try:
        assert len(vec) == 2
    finally:
        vec.close()
