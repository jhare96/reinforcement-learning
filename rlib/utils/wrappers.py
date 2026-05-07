"""Environment wrappers used by the rlib agent suite.

All wrappers subclass :class:`rlib.envs.RLEnvBase` and use the **modern
5-tuple** ``(obs, reward, terminated, truncated, info)`` step API
together with ``(obs, info)`` reset.  Backend translation (legacy gym
4-tuple, dm_env, ...) happens once in :mod:`rlib.envs.adapters`; from
the wrappers' point of view the input env is always canonical.

Wrappers can compose freely (``StackEnv(GreyScaleEnv(env))``).  Each
wrapper just has to implement ``reset`` / ``step`` against the modern
contract; everything else (``observation_space``, ``unwrapped``,
``__getattr__`` forwarding, ``close``, ...) is provided by
:class:`RLEnvBase`.
"""

# Code was inspired from or modified from OpenAI baselines
# https://github.com/openai/baselines/tree/master/baselines/common

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import torch
from PIL import Image

from rlib.envs import RLEnvBase, wrap


def _ensure_rlenv(env) -> RLEnvBase:
    """Coerce a raw backend env into an :class:`RLEnvBase` if needed."""
    if isinstance(env, RLEnvBase):
        return env
    return wrap(env)


def AtariValidate(env) -> RLEnvBase:
    env = FireResetEnv(env)
    env = NoopResetEnv(env, max_op=3000)
    env = StackEnv(env)
    return env


class RescaleEnv(RLEnvBase):
    def __init__(self, env, size: int):
        self.env = _ensure_rlenv(env)
        self.size = size

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        frame = np.array(Image.fromarray(frame).resize([self.size, self.size]))
        frame = np.dot(frame[..., :3], np.array([0.299, 0.587, 0.114])).astype(dtype=np.uint8)
        return frame[:, :, np.newaxis]

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.preprocess(obs), reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        return self.preprocess(obs), info


class AtariRescale42x42(RLEnvBase):
    def __init__(self, env):
        self.env = _ensure_rlenv(env)

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        frame = np.array(Image.fromarray(frame).resize([84, 110]))[110 - 84 :, 0:84, :]
        frame = np.dot(frame[..., :3], np.array([0.299, 0.587, 0.114])).astype(dtype=np.uint8)
        frame = np.array(Image.fromarray(frame).resize([42, 42])).astype(dtype=np.uint8)
        return frame[:, :, np.newaxis]

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.preprocess(obs), reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        return self.preprocess(obs), info


class AtariRescaleEnv(RLEnvBase):
    def __init__(self, env):
        self.env = _ensure_rlenv(env)

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        frame = np.array(Image.fromarray(frame).resize([84, 110]))[110 - 84 :, 0:84, :]
        frame = np.dot(frame[..., :3], np.array([0.299, 0.587, 0.114])).astype(dtype=np.uint8)
        return frame[:, :, np.newaxis]

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.preprocess(obs), reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        return self.preprocess(obs), info


class AtariRescaleColour(RLEnvBase):
    def __init__(self, env):
        self.env = _ensure_rlenv(env)

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        frame = np.array(Image.fromarray(frame).resize([84, 110]))[110 - 84 :, 0:84, :]
        return frame

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.preprocess(obs), reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        return self.preprocess(obs), info


class DummyEnv(RLEnvBase):
    """No-op wrapper. Mostly useful as an explicit conversion to ``RLEnvBase``."""

    def __init__(self, env):
        self.env = _ensure_rlenv(env)

    def step(self, action) -> tuple[Any, float, bool, bool, dict]:
        return self.env.step(action)

    def reset(self, *, seed=None, options=None) -> tuple[Any, dict]:
        return self.env.reset(seed=seed, options=options)


class NoopResetEnv(RLEnvBase):
    def __init__(self, env, max_op: int = 7):
        self.env = _ensure_rlenv(env)
        self.max_op = max_op

    def reset(self, *, seed=None, options=None) -> tuple[Any, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        noops = np.random.randint(0, self.max_op)
        for _ in range(noops):
            obs, _reward, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset()
        return obs, info

    def step(self, action) -> tuple[Any, float, bool, bool, dict]:
        return self.env.step(action)


class ClipRewardEnv(RLEnvBase):
    def __init__(self, env):
        self.env = _ensure_rlenv(env)

    def step(self, action) -> tuple[Any, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = np.clip(reward, -1, 1)
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None) -> tuple[Any, dict]:
        return self.env.reset(seed=seed, options=options)


class NoRewardEnv(RLEnvBase):
    def __init__(self, env):
        self.env = _ensure_rlenv(env)

    def step(self, action) -> tuple[Any, float, bool, bool, dict]:
        obs, _reward, terminated, truncated, info = self.env.step(action)
        return obs, 0, terminated, truncated, info

    def reset(self, *, seed=None, options=None) -> tuple[Any, dict]:
        return self.env.reset(seed=seed, options=options)


class FireResetEnv(RLEnvBase):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        self.env = _ensure_rlenv(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, *, seed=None, options=None) -> tuple[Any, dict]:
        self.env.reset(seed=seed, options=options)
        obs, _, terminated, truncated, _ = self.env.step(1)
        if terminated or truncated:
            self.env.reset()
        obs, _, terminated, truncated, info = self.env.step(2)
        if terminated or truncated:
            obs, info = self.env.reset()
        return obs, info

    def step(self, ac) -> tuple[Any, float, bool, bool, dict]:
        return self.env.step(ac)


class EpisodicLifeEnv(RLEnvBase):
    def __init__(self, env):
        self.env = _ensure_rlenv(env)
        self.lives = 0
        self.end_of_episode = True

    def step(self, action) -> tuple[Any, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.end_of_episode = bool(terminated) or bool(truncated)
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives:
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None) -> tuple[Any, dict]:
        if self.end_of_episode:
            obs, info = self.env.reset(seed=seed, options=options)
        else:
            obs, _, _, _, info = self.env.step(0)
        return obs, info


class TimeLimitEnv(RLEnvBase):
    def __init__(self, env, time_limit: int):
        self.env = _ensure_rlenv(env)
        self._time_limit = time_limit
        self._step = 0

    def step(self, action) -> tuple[Any, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step += 1
        if self._step > self._time_limit:
            truncated = True
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None) -> tuple[Any, dict]:
        self._step = 0
        return self.env.reset(seed=seed, options=options)


class StackEnv(RLEnvBase):
    def __init__(self, env, k: int = 4):
        self.env = _ensure_rlenv(env)
        self._stacked_frames: deque[np.ndarray] = deque([], maxlen=k)
        self.k = k

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self.stack_frames(obs)
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        return self.stack_frames(obs, True), info

    def stack_frames(self, frame: np.ndarray, reset: bool = False) -> np.ndarray:
        if reset:
            for _ in range(self.k):
                self._stacked_frames.append(frame)
        else:
            self._stacked_frames.append(frame)
        return np.concatenate(self._stacked_frames, axis=2)


class AutoResetEnv(RLEnvBase):
    def __init__(self, env):
        self.env = _ensure_rlenv(env)

    def step(self, action) -> tuple[Any, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            obs, _info = self.env.reset()
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None) -> tuple[Any, dict]:
        return self.env.reset(seed=seed, options=options)


class ChannelsFirstEnv(RLEnvBase):
    def __init__(self, env):
        self.env = _ensure_rlenv(env)

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs.transpose(2, 0, 1), reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        return obs.transpose(2, 0, 1), info


class GreyScaleEnv(RLEnvBase):
    def __init__(self, env):
        self.env = _ensure_rlenv(env)

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        frame = np.dot(frame[..., :3], np.array([0.299, 0.587, 0.114])).astype(dtype=np.uint8)
        return frame[:, :, None]

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.preprocess(obs), reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        return self.preprocess(obs), info


class ToTorchEnv(RLEnvBase):
    def __init__(self, env, device: str = 'cuda:0'):
        self.env = _ensure_rlenv(env)
        self.device = device

    def step(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action.cpu().numpy())
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.tensor(reward, device=self.device, dtype=torch.float32)
        terminated_t = torch.tensor(terminated, device=self.device)
        truncated_t = torch.tensor(truncated, device=self.device)
        return obs, reward, terminated_t, truncated_t, info

    def reset(self, *, seed=None, options=None) -> tuple[torch.Tensor, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        return torch.from_numpy(obs).float().to(self.device), info


def apple_pickgame(
    env,
    k: int = 1,
    grey_scale: bool = False,
    auto_reset: bool = False,
    max_steps: int | None = 1000,
    channels_first: bool = True,
) -> RLEnvBase:
    if auto_reset:
        env = AutoResetEnv(env)
    if max_steps is not None:
        env = TimeLimitEnv(env, time_limit=max_steps)
    if grey_scale:
        env = GreyScaleEnv(env)
    if k > 1:
        env = StackEnv(env, k)
    if channels_first:
        env = ChannelsFirstEnv(env)
    return env


def AtariEnv(
    env,
    k: int = 4,
    rescale: int = 84,
    episodic: bool = True,
    reset: bool = True,
    clip_reward: bool = True,
    Noop: bool = True,
    time_limit: int | None = None,
    channels_first: bool = True,
    auto_reset: bool = False,
) -> RLEnvBase:
    """Wrapper function for Deterministic Atari env.

    ``assert 'Deterministic' in env.spec.id``
    """
    if reset:
        env = FireResetEnv(env)

    if Noop:
        if 'NoFrameskip' in env.spec.id:
            max_op = 30
        else:
            max_op = 7
        env = NoopResetEnv(env, max_op)

    if clip_reward:
        env = ClipRewardEnv(env)

    if episodic:
        env = EpisodicLifeEnv(env)

    if rescale == 42:
        env = AtariRescale42x42(env)
    elif rescale == 84:
        env = AtariRescaleEnv(env)
    else:
        raise ValueError('84 or 42 are valid rescale sizes')

    if k > 1:
        env = StackEnv(env, k)

    if time_limit is not None:
        env = TimeLimitEnv(env, time_limit)

    if auto_reset:
        env = AutoResetEnv(env)

    if channels_first:
        env = ChannelsFirstEnv(env)

    return env
