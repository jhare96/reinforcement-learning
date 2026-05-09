"""Replay buffers for off-policy agents.

Currently houses :class:`NumpyReplayMemory`, a fixed-capacity ring
buffer storing ``(state, action, reward, next_state, done)`` transitions
in pre-allocated numpy arrays.

The legacy ``FrameBuffer`` / ``stack_frames`` helpers (which depended
on ``scipy.misc.imresize``, removed from SciPy 1.3) have been retired
in favour of the frame-stacking wrappers in :mod:`rlib.envs.wrappers`.
"""

from __future__ import annotations

import numpy as np


class NumpyReplayMemory:
    """Fixed-capacity ring buffer of ``(s, a, r, s', done)`` transitions."""

    def __init__(self, replaysize: int, shape: tuple[int, ...]) -> None:
        self._idx = 0
        self._full_flag = False
        self._replay_length = replaysize
        self._states = np.zeros((replaysize, *shape), dtype=np.uint8)
        self._actions = np.zeros((replaysize,), dtype=np.int64)
        self._rewards = np.zeros((replaysize,), dtype=np.float32)
        self._next_states = np.zeros((replaysize, *shape), dtype=np.uint8)
        self._dones = np.zeros((replaysize,), dtype=bool)

    def addMemory(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self._states[self._idx] = state
        self._actions[self._idx] = action
        self._rewards[self._idx] = reward
        self._next_states[self._idx] = next_state
        self._dones[self._idx] = done
        if self._idx + 1 >= self._replay_length:
            self._idx = 0
            self._full_flag = True
        else:
            self._idx += 1

    def __len__(self) -> int:
        return self._replay_length if self._full_flag else self._idx

    def sample(
        self, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        upper = self._replay_length if self._full_flag else self._idx
        idxs = np.random.choice(upper, size=batch_size, replace=False)
        return (
            self._states[idxs],
            self._actions[idxs],
            self._rewards[idxs],
            self._next_states[idxs],
            self._dones[idxs],
            idxs,
        )
