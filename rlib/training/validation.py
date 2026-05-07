"""Validation strategies for :class:`rlib.training.SyncMultiEnvTrainer`.

Two implementations cover the two ways the agent suite passes validation
environments to the trainer:

* :class:`AsyncValidator` — a ``list`` of single envs, run in parallel
  threads (one episode per env per call to :meth:`run`).
* :class:`SyncValidator` — a single batched env (``BatchEnv`` /
  ``DummyBatchEnv``) stepped synchronously in the calling thread.

Both implement the same :class:`Validator` :class:`typing.Protocol`, so
the trainer just calls ``self.validator.run(self.get_action, render)``
and doesn't need to know which strategy is in use.

Pulling these out of the trainer:

* shrinks the trainer file by ~100 lines,
* lets us test each validator in isolation against a fake action
  function without spinning up a whole agent, and
* removes the bug-prone ``self.validate_func`` instance-method binding.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import numpy as np

from rlib.envs import RLVecEnv, wrap
from rlib.utils.VecEnv import BatchEnv, DummyBatchEnv

__all__ = ["AsyncValidator", "SyncValidator", "Validator", "make_validator"]


# Action picker has the same shape across both strategies: numpy state ->
# numpy action(s).  Trainers expose this as ``self.get_action``.
ActionFn = Callable[[np.ndarray], Any]


@runtime_checkable
class Validator(Protocol):
    """Strategy for running validation episodes and returning a mean score."""

    def run(
        self,
        get_action: ActionFn,
        num_episodes: int,
        max_steps: int,
        render: bool = False,
    ) -> float:
        """Run ``num_episodes`` of validation; return the mean total reward."""


# ---------------------------------------------------------------------------
# Concrete validators
# ---------------------------------------------------------------------------


class AsyncValidator:
    """Validate against a ``list`` of envs using one daemon thread per env.

    Each thread runs its share of episodes and pushes the per-episode
    total reward into a shared list (guarded by ``self._lock``).  After
    every thread joins, the mean across all collected scores is
    returned.
    """

    def __init__(self, envs: list) -> None:
        if not isinstance(envs, list):
            raise TypeError(f"AsyncValidator expects a list of envs, got {type(envs).__name__}")
        self.envs = envs
        self._lock = threading.Lock()
        self._scores: list[float] = []

    def run(
        self,
        get_action: ActionFn,
        num_episodes: int,
        max_steps: int,
        render: bool = False,
    ) -> float:
        n = len(self.envs)
        # Distribute episodes evenly; remainder goes to the last env.
        per_env = [num_episodes // n for _ in range(n)]
        per_env[-1] += num_episodes % n
        # Only the first env renders, mirroring the original behaviour.
        render_flags = [bool(render and i == 0) for i in range(n)]

        self._scores = []
        threads = [
            threading.Thread(
                daemon=True,
                target=self._run_single,
                args=(self.envs[i], get_action, per_env[i], max_steps, render_flags[i]),
            )
            for i in range(n)
        ]
        try:
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        except KeyboardInterrupt:
            for t in threads:
                t.join()

        score = float(np.mean(self._scores)) if self._scores else 0.0
        self._scores = []
        return score

    def _run_single(
        self,
        env: Any,
        get_action: ActionFn,
        num_episodes: int,
        max_steps: int,
        render: bool,
    ) -> None:
        rl_env = wrap(env)
        for _episode in range(num_episodes):
            state, _info = rl_env.reset()
            episode_reward = 0.0
            for t in range(max_steps):
                action = get_action(state[np.newaxis])
                next_state, reward, terminated, truncated, _info = rl_env.step(action)
                done = RLVecEnv.merge_done(terminated, truncated)
                state = next_state
                episode_reward += float(reward)

                if render:
                    with self._lock:
                        env.render()

                if done or t == max_steps - 1:
                    with self._lock:
                        self._scores.append(episode_reward)
                    break
        if render:
            with self._lock:
                env.close()


class SyncValidator:
    """Validate against a single batched env (``BatchEnv`` / ``DummyBatchEnv``)."""

    def __init__(self, envs: BatchEnv | DummyBatchEnv) -> None:
        self.envs = envs
        self.num_envs = len(envs)

    def run(
        self,
        get_action: ActionFn,
        num_episodes: int,
        max_steps: int,
        render: bool = False,
    ) -> float:
        episode_scores: list[np.ndarray] = []
        # ``num_episodes // num_envs`` is what the original loop ran;
        # keep that semantics so existing scores don't shift.
        for _episode in range(num_episodes // self.num_envs):
            states = self.envs.reset()
            episode_reward: list[np.ndarray] = []
            for t in range(max_steps):
                actions = get_action(states)
                next_states, rewards, dones, _infos = self.envs.step(actions)
                states = next_states
                episode_reward.append(rewards * (1 - dones))

                if render:
                    self.envs.render()

                if dones.sum() == self.num_envs or t == max_steps - 1:
                    episode_scores.append(np.sum(np.stack(episode_reward), axis=0))
                    break

        return float(np.mean(episode_scores)) if episode_scores else 0.0


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_validator(val_envs: list | BatchEnv | DummyBatchEnv) -> Validator:
    """Pick the appropriate :class:`Validator` for the given env(s)."""
    if isinstance(val_envs, list):
        return AsyncValidator(val_envs)
    return SyncValidator(val_envs)
