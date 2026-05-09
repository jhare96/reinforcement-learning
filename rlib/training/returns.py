"""Return / advantage estimators used by the synchronous trainers.

These are pure numpy functions over a ``(T, B, ...)`` rollout (``T``
timesteps, ``B`` parallel envs).  Splitting them out of
:class:`rlib.training.SyncMultiEnvTrainer` keeps the trainer focused on
the training loop and makes the estimators trivially unit-testable.

Dispatch
========

The trainer chooses between estimators via the :class:`Returns` enum:
``self.config.returns(rewards, values, last_values, dones, gamma, lambda_)``
calls into the function the enum member wraps.

The enum's *name* is the canonical CLI / log string (``"NSTEP"``,
``"GAE"``, ``"LAMBDA"``); the *value* is the callable.  Adding a new
estimator is therefore a two-line change: write the function, add an
enum member.
"""

from __future__ import annotations

import enum

import numpy as np

__all__ = [
    "GAE",
    "Returns",
    "lambda_return",
    "nstep_return",
]


# ---------------------------------------------------------------------------
# Underlying estimator functions
# ---------------------------------------------------------------------------


def nstep_return(
    rewards: np.ndarray,
    last_values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    clip: bool = False,
) -> np.ndarray:
    r"""N-step bootstrapped return :math:`R_t = r_t + \gamma R_{t+1}`.

    The recursion is reset to zero whenever ``dones[t]`` is set so the
    next episode's rewards don't bleed into the current one.
    """
    if clip:
        rewards = np.clip(rewards, -1, 1)

    T = len(rewards)
    R = np.zeros_like(rewards)
    R[-1] = last_values * (1 - dones[-1])

    for i in reversed(range(T - 1)):
        # restart score if done as BatchEnv automatically resets after end of episode
        R[i] = rewards[i] + gamma * R[i + 1] * (1 - dones[i])

    return R


def lambda_return(
    rewards: np.ndarray,
    values: np.ndarray,
    last_values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    lambda_: float = 0.8,
    clip: bool = False,
) -> np.ndarray:
    r"""λ-return :math:`R^\lambda_t = r_t + \gamma((1-\lambda) V_{t+1} + \lambda R^\lambda_{t+1})`.

    With ``lambda_ == 1.0`` collapses to the n-step return; with
    ``lambda_ == 0.0`` collapses to one-step TD.
    """
    if clip:
        rewards = np.clip(rewards, -1, 1)
    T = len(rewards)
    R = np.zeros_like(rewards)
    R[-1] = last_values * (1 - dones[-1])
    for t in reversed(range(T - 1)):
        R[t] = rewards[t] + gamma * (lambda_ * R[t + 1] + (1.0 - lambda_) * values[t + 1]) * (
            1 - dones[t]
        )
    return R


def GAE(
    rewards: np.ndarray,
    values: np.ndarray,
    last_values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    lambda_: float = 0.95,
    clip: bool = False,
) -> np.ndarray:
    """Generalised Advantage Estimation (Schulman et al. 2015).

    Returns the *advantage* sequence (``A_t``); the value targets are
    recovered as ``A_t + V_t``.
    """
    if clip:
        rewards = np.clip(rewards, -1, 1)
    Adv = np.zeros_like(rewards)
    Adv[-1] = rewards[-1] + gamma * last_values * (1 - dones[-1]) - values[-1]
    T = len(rewards)
    for t in reversed(range(T - 1)):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        Adv[t] = delta + gamma * lambda_ * Adv[t + 1] * (1 - dones[t])
    return Adv


# ---------------------------------------------------------------------------
# Uniform-signature wrappers used by the trainer's main loop.
#
# All three accept the full ``(rewards, values, last_values, dones,
# gamma, lambda_)`` signature and return *value targets* (so the trainer
# can pass the result straight to ``model.backprop`` without further
# arithmetic).  The N-step branch ignores ``values`` and ``lambda_``;
# the GAE branch adds ``+ values`` to the advantage to obtain targets.
# ---------------------------------------------------------------------------


def _nstep_targets(
    rewards: np.ndarray,
    values: np.ndarray,
    last_values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lambda_: float,
) -> np.ndarray:
    return nstep_return(rewards, last_values, dones, gamma=gamma)


def _gae_targets(
    rewards: np.ndarray,
    values: np.ndarray,
    last_values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lambda_: float,
) -> np.ndarray:
    return GAE(rewards, values, last_values, dones, gamma=gamma, lambda_=lambda_) + values


def _lambda_targets(
    rewards: np.ndarray,
    values: np.ndarray,
    last_values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lambda_: float,
) -> np.ndarray:
    return lambda_return(rewards, values, last_values, dones, gamma=gamma, lambda_=lambda_)


# ---------------------------------------------------------------------------
# Enum
# ---------------------------------------------------------------------------


class Returns(enum.Enum):
    """Return / advantage estimators.

    The enum *name* is the canonical CLI / log string (``"NSTEP"``,
    ``"GAE"``, ``"LAMBDA"``).  The *value* is the wrapped callable, so
    members are directly callable::

        targets = Returns.GAE(rewards, values, last_values, dones, 0.99, 0.95)

    The :func:`enum.member` wrapper around each value is required so
    Python's enum metaclass treats the function as an enum member rather
    than as an instance method bound on the class.
    """

    NSTEP = enum.member(_nstep_targets)
    GAE = enum.member(_gae_targets)
    LAMBDA = enum.member(_lambda_targets)

    def __call__(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        last_values: np.ndarray,
        dones: np.ndarray,
        gamma: float,
        lambda_: float,
    ) -> np.ndarray:
        return self.value(rewards, values, last_values, dones, gamma, lambda_)

    def __str__(self) -> str:
        return self.name
