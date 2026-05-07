"""Return / advantage estimators used by the synchronous trainers.

These are pure numpy functions over a ``(T, B, ...)`` rollout (``T``
timesteps, ``B`` parallel envs).  Splitting them out of
:class:`rlib.training.SyncMultiEnvTrainer` keeps the trainer focused
on the training loop and makes the estimators trivially unit-testable.

The trainer dispatches to one of these via the :data:`RETURN_FUNCTIONS`
table keyed on :class:`~rlib.training.config.ReturnType`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

__all__ = [
    "GAE",
    "RETURN_FUNCTIONS",
    "lambda_return",
    "nstep_return",
]


def nstep_return(
    rewards: np.ndarray,
    last_values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    clip: bool = False,
) -> np.ndarray:
    """N-step bootstrapped return :math:`R_t = r_t + \\gamma R_{t+1}`.

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
    """λ-return :math:`R^\\lambda_t = r_t + \\gamma((1-\\lambda) V_{t+1} + \\lambda R^\\lambda_{t+1})`.

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


# ``GAE`` returns the advantage sequence; the trainer's default loop
# expects targets, so the dispatch wrapper adds ``+ values`` for it.
def _gae_targets(
    rewards: np.ndarray,
    values: np.ndarray,
    last_values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lambda_: float,
) -> np.ndarray:
    return GAE(rewards, values, last_values, dones, gamma=gamma, lambda_=lambda_) + values


def _nstep_targets(
    rewards: np.ndarray,
    values: np.ndarray,
    last_values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lambda_: float,
) -> np.ndarray:
    # values + lambda_ ignored — n-step doesn't use them.
    return nstep_return(rewards, last_values, dones, gamma=gamma)


def _lambda_targets(
    rewards: np.ndarray,
    values: np.ndarray,
    last_values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lambda_: float,
) -> np.ndarray:
    return lambda_return(rewards, values, last_values, dones, gamma=gamma, lambda_=lambda_)


_ReturnFn = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float], np.ndarray]

#: Dispatch table from :data:`~rlib.training.config.ReturnType` to the
#: function the default n-step training loop should call.  All entries
#: have the same uniform signature ``(rewards, values, last_values,
#: dones, gamma, lambda_) -> targets`` so the trainer can call them
#: without knowing which estimator was chosen.
RETURN_FUNCTIONS: dict[str, _ReturnFn] = {
    "nstep": _nstep_targets,
    "GAE": _gae_targets,
    "lambda": _lambda_targets,
}
