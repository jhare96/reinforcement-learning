"""Tests for the pure-numpy helpers in :mod:`rlib.utils.utils`."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rlib.utils.utils import (
    GAE,
    RunningMeanStd,
    Welfords_algorithm,
    fold_batch,
    fold_many,
    lambda_return,
    normalise,
    nstep_return,
    one_hot,
    stack_many,
    tonumpy,
    tonumpy_many,
    totorch,
    totorch_many,
    unfold_batch,
)

# ---------------------------------------------------------------------------
# Shape utilities
# ---------------------------------------------------------------------------


def test_fold_and_unfold_batch_round_trip() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 8, 3, 5)).astype(np.float32)
    folded = fold_batch(x)
    assert folded.shape == (32, 3, 5)
    restored = unfold_batch(folded, length=4, batch_size=8)
    assert restored.shape == x.shape
    np.testing.assert_array_equal(restored, x)


def test_fold_many_handles_multiple_arrays() -> None:
    a = np.zeros((2, 3, 4))
    b = np.zeros((2, 3))
    folded_a, folded_b = fold_many(a, b)
    assert folded_a.shape == (6, 4)
    assert folded_b.shape == (6,)


def test_stack_many_returns_tuple_of_arrays() -> None:
    a = [np.array([1, 2]), np.array([3, 4])]
    b = [np.array([0]), np.array([1])]
    sa, sb = stack_many(a, b)
    np.testing.assert_array_equal(sa, np.array([[1, 2], [3, 4]]))
    np.testing.assert_array_equal(sb, np.array([[0], [1]]))


def test_one_hot_encodes_class_indices() -> None:
    out = one_hot(np.array([0, 2, 1]), num_classes=3)
    expected = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=float)
    np.testing.assert_array_equal(out, expected)


def test_normalise_standardises_data() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    out = normalise(x, mean=x.mean(), std=x.std())
    assert pytest.approx(0.0, abs=1e-7) == out.mean()
    assert pytest.approx(1.0, rel=1e-6) == out.std()


# ---------------------------------------------------------------------------
# Torch interop
# ---------------------------------------------------------------------------


def test_totorch_and_back_round_trips_on_cpu() -> None:
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    t = totorch(x, device="cpu")
    assert isinstance(t, torch.Tensor)
    assert t.dtype == torch.float32
    assert t.device.type == "cpu"
    np.testing.assert_array_equal(tonumpy(t), x)


def test_totorch_many_and_tonumpy_many() -> None:
    a = np.zeros((2, 2), dtype=np.float32)
    b = np.ones((3,), dtype=np.float32)
    ta, tb = totorch_many(a, b, device="cpu")
    assert ta.shape == a.shape and tb.shape == b.shape
    na, nb = tonumpy_many(ta, tb)
    np.testing.assert_array_equal(na, a)
    np.testing.assert_array_equal(nb, b)


# ---------------------------------------------------------------------------
# Welford / RunningMeanStd
# ---------------------------------------------------------------------------


def test_welfords_algorithm_streams_per_batch_mean() -> None:
    rng = np.random.default_rng(42)
    data = rng.standard_normal((1000, 4)).astype(np.float64)
    welford = Welfords_algorithm(mean=np.zeros(4))
    for batch in data.reshape(100, 10, 4):
        welford.update(batch)
    # The estimator is biased by the ``epsilon`` initialisation, but
    # converges to the overall mean as more batches arrive.
    np.testing.assert_allclose(
        welford.mean,
        data.mean(axis=0),
        rtol=1e-3,
        atol=1e-3,
    )


def test_running_mean_std_matches_numpy() -> None:
    rng = np.random.default_rng(123)
    data = rng.standard_normal((1000, 3)).astype(np.float64)
    rms = RunningMeanStd(shape=(3,))
    for batch in data.reshape(20, 50, 3):
        rms.update(batch)
    np.testing.assert_allclose(rms.mean, data.mean(axis=0), rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(np.sqrt(rms.var), data.std(axis=0), rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# Returns / advantages
# ---------------------------------------------------------------------------


def _bootstrap_returns_reference(
    rewards: np.ndarray,
    last_values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
) -> np.ndarray:
    T = len(rewards)
    R = np.zeros_like(rewards)
    R[-1] = last_values * (1 - dones[-1])
    for i in reversed(range(T - 1)):
        R[i] = rewards[i] + gamma * R[i + 1] * (1 - dones[i])
    return R


def test_nstep_return_matches_reference_implementation() -> None:
    rewards = np.array([[1.0], [0.5], [0.0], [2.0]], dtype=np.float32)
    last_values = np.array([0.7], dtype=np.float32)
    dones = np.zeros_like(rewards, dtype=np.float32)
    out = nstep_return(rewards, last_values, dones, gamma=0.9)
    expected = _bootstrap_returns_reference(rewards, last_values, dones, gamma=0.9)
    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)


def test_nstep_return_zeroes_future_contribution_when_done_set() -> None:
    rewards = np.array([[1.0], [1.0], [1.0]], dtype=np.float32)
    last_values = np.array([5.0], dtype=np.float32)
    # done at t=1 means the step-2 return must not feed back through
    # the recursion at t=1 (we treat it as a fresh episode boundary).
    dones = np.array([[0.0], [1.0], [0.0]], dtype=np.float32)
    out = nstep_return(rewards, last_values, dones, gamma=0.99)
    # R[2] = 5; R[1] = 1 + 0.99*5*(1-1) = 1; R[0] = 1 + 0.99*1*(1-0) = 1.99
    np.testing.assert_allclose(
        out.flatten(),
        np.array([1.99, 1.0, 5.0]),
        rtol=1e-6,
        atol=1e-6,
    )


def test_nstep_return_clip_caps_rewards() -> None:
    # Use a bootstrap value so the final timestep contributes to R[0]
    # (the function overwrites R[-1] with last_values, discarding
    # rewards[-1]).
    rewards = np.array([[10.0], [10.0]], dtype=np.float32)
    last_values = np.array([10.0], dtype=np.float32)
    dones = np.zeros_like(rewards, dtype=np.float32)
    unclipped = nstep_return(rewards, last_values, dones, gamma=1.0, clip=False)
    clipped = nstep_return(rewards, last_values, dones, gamma=1.0, clip=True)
    # Unclipped R[0] = 10 + 1.0 * 10 = 20.
    # Clipped:  R[0] = 1  + 1.0 * 10 = 11 (last_values is not clipped).
    assert unclipped[0, 0] == pytest.approx(20.0)
    assert clipped[0, 0] == pytest.approx(11.0)
    assert clipped[0, 0] < unclipped[0, 0]


def test_lambda_return_collapses_to_nstep_when_lambda_is_one() -> None:
    rewards = np.array([[0.5], [0.0], [1.0], [0.2]], dtype=np.float32)
    values = np.array([[0.1], [0.2], [0.3], [0.4]], dtype=np.float32)
    last_values = np.array([0.5], dtype=np.float32)
    dones = np.zeros_like(rewards, dtype=np.float32)
    lam_one = lambda_return(rewards, values, last_values, dones, gamma=0.95, lambda_=1.0)
    nstep = nstep_return(rewards, last_values, dones, gamma=0.95)
    np.testing.assert_allclose(lam_one, nstep, rtol=1e-6, atol=1e-6)


def test_gae_recovers_one_step_td_when_lambda_is_zero() -> None:
    rewards = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    values = np.array([[0.5], [1.0], [1.5]], dtype=np.float32)
    last_values = np.array([2.0], dtype=np.float32)
    dones = np.zeros_like(rewards, dtype=np.float32)
    adv = GAE(rewards, values, last_values, dones, gamma=0.9, lambda_=0.0)
    expected = np.zeros_like(rewards)
    expected[-1] = rewards[-1] + 0.9 * last_values - values[-1]
    for t in reversed(range(len(rewards) - 1)):
        expected[t] = rewards[t] + 0.9 * values[t + 1] - values[t]
    np.testing.assert_allclose(adv, expected, rtol=1e-6, atol=1e-6)
