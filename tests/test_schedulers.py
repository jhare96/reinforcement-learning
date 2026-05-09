"""Tests for the LR scheduler helpers in :mod:`rlib.utils.schedulers`."""

from __future__ import annotations

import pytest
import torch

from rlib.utils.schedulers import polynomial_sheduler


def _make_optimiser(lr: float) -> torch.optim.Optimizer:
    param = torch.nn.Parameter(torch.zeros(1))
    return torch.optim.SGD([param], lr=lr)


def test_polynomial_scheduler_starts_at_initial_lr() -> None:
    opt = _make_optimiser(lr=1.0)
    sched = polynomial_sheduler(opt, lr_final=0.0, decay_steps=10, power=1)
    assert sched.get_last_lr()[0] == pytest.approx(1.0)


def test_polynomial_scheduler_decays_linearly() -> None:
    opt = _make_optimiser(lr=1.0)
    sched = polynomial_sheduler(opt, lr_final=0.0, decay_steps=10, power=1)
    lrs = []
    for _ in range(11):
        opt.step()
        sched.step()
        lrs.append(sched.get_last_lr()[0])
    # Linear decay from 1.0 -> 0.0 over 10 steps.
    assert lrs[0] == pytest.approx(0.9, abs=1e-6)
    assert lrs[4] == pytest.approx(0.5, abs=1e-6)
    assert lrs[9] == pytest.approx(0.0, abs=1e-6)


def test_polynomial_scheduler_clamps_to_final_after_decay_steps() -> None:
    opt = _make_optimiser(lr=1.0)
    sched = polynomial_sheduler(opt, lr_final=0.1, decay_steps=5, power=2)
    for _ in range(20):
        opt.step()
        sched.step()
    assert sched.get_last_lr()[0] == pytest.approx(0.1, abs=1e-6)


def test_polynomial_scheduler_rejects_increasing_schedule() -> None:
    opt = _make_optimiser(lr=0.1)
    with pytest.raises(AssertionError):
        polynomial_sheduler(opt, lr_final=1.0, decay_steps=10)


def test_polynomial_scheduler_allows_constant_lr() -> None:
    """``lr_final == lr_init`` is the canonical "no scheduler" config."""
    opt = _make_optimiser(lr=0.5)
    sched = polynomial_sheduler(opt, lr_final=0.5, decay_steps=10)
    for _ in range(20):
        opt.step()
        sched.step()
    assert sched.get_last_lr()[0] == pytest.approx(0.5)
