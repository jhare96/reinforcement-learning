"""Shared building blocks for the paper-reproduction recipes.

* :class:`MLP`         — small MLP body for low-dimensional classic control.
* :func:`NatureCNNBody` — re-exported NatureCNN from :mod:`rlib.models`
                          for Atari pixel inputs.
* Env id constants for the three classic-control and three Atari
  benchmarks the paper reports.
* :func:`classic_envs` — build training + validation env tuples for
  classic control with the paper's 32 actors.
* :func:`atari_envs`   — same for the Atari benchmarks.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import torch

from rlib.envs.vec_env import BatchEnv, DummyBatchEnv
from rlib.envs.wrappers import AtariEnv
from rlib.models import NatureCNN  # noqa: F401  (re-exported)

# ---------------------------------------------------------------------------
# Env id constants (matching the paper's experiment grid)
# ---------------------------------------------------------------------------

CLASSIC_ENVS = ("Acrobot-v1", "CartPole-v1", "MountainCar-v0")
ATARI_ENVS = (
    "FreewayDeterministic-v4",
    "MontezumaRevengeDeterministic-v4",
    "SpaceInvadersDeterministic-v4",
)

#: Number of parallel actors used throughout the paper.
NUM_WORKERS = 32

#: Steps used for validation episodes (large enough to play out a
#: classic control episode; Atari uses the env's own time limit).
CLASSIC_VAL_STEPS = 500
ATARI_VAL_STEPS = 4_500


# ---------------------------------------------------------------------------
# Network bodies
# ---------------------------------------------------------------------------


class MLP(torch.nn.Module):
    """Two-layer Tanh MLP body for low-dimensional classic-control envs."""

    def __init__(self, input_size: tuple[int, ...] | int, hidden_size: int = 64) -> None:
        super().__init__()
        in_dim = int(input_size[0]) if hasattr(input_size, "__len__") else int(input_size)
        self.dense_size = hidden_size
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Env factories
# ---------------------------------------------------------------------------


def classic_envs(
    env_id: str,
    num_envs: int = NUM_WORKERS,
) -> tuple[DummyBatchEnv, list[Any]]:
    """Build (training BatchEnv, list of validation envs) for classic control."""
    train_envs = DummyBatchEnv(lambda e: e, env_id, num_envs=num_envs)
    val_envs = [gym.make(env_id) for _ in range(min(8, num_envs))]
    return train_envs, val_envs


def atari_envs(
    env_id: str,
    num_envs: int = NUM_WORKERS,
    *,
    val_count: int = 4,
    blocking: bool = False,
) -> tuple[BatchEnv, list[Any]]:
    """Build (training BatchEnv, list of validation envs) for Atari.

    Training envs use the standard Atari pre-processing stack
    (FireReset, Noop, ClipReward, EpisodicLife, 84x84 greyscale rescale,
    4-frame stack); validation envs use the same wrappers but with
    ``episodic=False`` and ``clip_reward=False`` so the score reflects
    the true game return.
    """
    train_envs = BatchEnv(
        AtariEnv,
        env_id,
        num_envs=num_envs,
        blocking=blocking,
        k=4,
        episodic=True,
        reset=True,
        clip_reward=True,
    )
    val_envs = [
        AtariEnv(gym.make(env_id), k=4, episodic=False, reset=True, clip_reward=False)
        for _ in range(val_count)
    ]
    return train_envs, val_envs


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def env_input_shape(envs: BatchEnv | DummyBatchEnv) -> tuple[int, ...]:
    """Return the observation shape after the env wrappers have been applied."""
    return (
        envs.envs[0].reset()[0].shape if hasattr(envs.envs[0], "reset") else envs.reset().shape[1:]
    )
