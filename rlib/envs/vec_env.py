"""Vectorised environment runners for rlib.

The single-env :class:`~rlib.envs.RLEnvBase` contract uses the modern
5-tuple ``(obs, reward, terminated, truncated, info)``.  Agent rollout
code in this library, however, has historically consumed the legacy
4-tuple ``(obs, rewards, dones, infos)``.  We keep that agent-facing
shape on purpose — the per-env 5-tuple lives on the wrapper side, and
the vec env runners below are the **single boundary** where
``done = terminated or truncated`` is computed (via
:meth:`RLVecEnv.merge_done`).

Adding a new backend therefore never requires touching this file: just
ship a new :class:`~rlib.envs.RLEnvBase` adapter and the vec runners
will consume it without changes.
"""

# Code was inspired from or modified from OpenAI baselines
# https://github.com/openai/baselines/tree/master/baselines/common

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable, Iterable, Iterator
from itertools import chain
from typing import Any

import numpy as np

from rlib.envs.base import RLEnvBase, RLVecEnv
from rlib.envs.registry import make as _make_env
from rlib.envs.registry import wrap as _wrap_env

# ---------------------------------------------------------------------------
# Single-env subprocess wrapper
# ---------------------------------------------------------------------------


class Env:
    """Run a single :class:`RLEnvBase` in its own subprocess.

    The worker process speaks the modern 5-tuple internally; this
    parent-side class collapses ``(terminated, truncated)`` into a
    single ``done`` flag before returning to the agent so consumers see
    the legacy 4-tuple.
    """

    def __init__(self, env: RLEnvBase, worker_id: int = 0):
        self.parent, self.child = mp.Pipe()
        self.worker = Worker(worker_id, env, self.child)
        self.worker.daemon = True
        self.worker.start()
        self.open = True

    def __del__(self):
        self.close()
        self.parent.close()
        self.child.close()

    def __getattr__(self, name: str):
        attribute = self._send_step('getattr', name)
        return attribute()

    def _send_step(self, cmd: str, payload) -> Callable[[], Any]:
        self.parent.send((cmd, payload))
        return self._recieve

    def _recieve(self):
        return self.parent.recv()

    def step(self, action, blocking: bool = True) -> Callable[[], Any]:
        return self._send_step('step', action)

    def reset(self):
        results = self._send_step('reset', None)
        return results()

    def close(self):
        if self.open:
            self.open = False
            self._send_step('close', None)
            self.worker.join()

    def render(self):
        self._send_step('render', None)


class Worker(mp.Process):
    def __init__(self, worker_id: int, env: RLEnvBase, connection):
        np.random.seed()
        mp.Process.__init__(self)
        self.env = _wrap_env(env)
        self.worker_id = worker_id
        self.connection = connection

    def _step(self):
        try:
            while True:
                cmd, a = self.connection.recv()
                if cmd == 'step':
                    obs, r, terminated, truncated, info = self.env.step(a)
                    done = RLVecEnv.merge_done(terminated, truncated)
                    info = RLVecEnv.merge_info(info, terminated, truncated)
                    self.connection.send((obs, r, done, info))
                elif cmd == 'render':
                    self.env.render()
                elif cmd == 'reset':
                    obs, _info = self.env.reset()
                    self.connection.send(obs)
                elif cmd == 'getattr':
                    self.connection.send(getattr(self.env, a))
                elif cmd == 'close':
                    self.env.close()
                    break
        except KeyboardInterrupt:
            print("closing worker", self.worker_id)
        finally:
            self.env.close()

    def run(self):
        self._step()


# ---------------------------------------------------------------------------
# Vectorised batch envs
# ---------------------------------------------------------------------------


class BatchEnv(RLVecEnv):
    """Run ``num_envs`` envs in parallel, one subprocess each."""

    def __init__(
        self,
        env_constructor: Callable[..., RLEnvBase],
        env_id: str,
        num_envs: int,
        blocking: bool = False,
        make_args: dict | None = None,
        **env_args,
    ):
        make_args = make_args or {}
        self.envs: list[Env] = []
        for _ in range(num_envs):
            inner = _make_env(env_id, **make_args)
            self.envs.append(Env(env_constructor(inner, **env_args)))
        self.blocking = blocking

    def __len__(self) -> int:
        return len(self.envs)

    def __getattr__(self, name: str):
        return getattr(self.envs[0], name)

    def step(
        self, actions: Iterable[Any]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[dict, ...]]:
        if self.blocking:
            results = [env.step(action, True) for env, action in zip(self.envs, actions)]
        else:
            results = [env.step(action, False) for env, action in zip(self.envs, actions)]
            results = [result() for result in results]

        obs, rewards, done, info = zip(*results)
        return np.stack(obs), np.stack(rewards), np.stack(done), info

    def reset(self) -> np.ndarray:
        obs = [env.reset() for env in self.envs]
        return np.stack(obs)

    def close(self):
        for env in self.envs:
            env.close()


def chunks(seq: list[Any], n: int) -> Iterator[list[Any]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


class ChunkEnv(RLVecEnv):
    def __init__(self, env_id: str, num_workers: int, num_chunks: int):
        self.num_workers = num_workers
        self.num_chunks = num_chunks
        self.env_id = env_id

        self.workers: list[ChunkWorker] = []
        self.parents: list[Any] = []
        for _ in range(num_workers):
            parent, child = mp.Pipe()
            worker = ChunkWorker(env_id, num_chunks, child)
            self.parents.append(parent)
            self.workers.append(worker)

        try:
            for worker in self.workers:
                worker.start()
        except KeyboardInterrupt:
            self.close()
            exit()

    def __len__(self) -> int:
        return self.num_workers * self.num_chunks

    def _send_step(self, cmd: str, actions) -> Callable[[], list[Any]]:
        for parent, action_chunk in zip(self.parents, chunks(actions, self.num_chunks)):
            parent.send((cmd, action_chunk))
        return self._recieve

    def _recieve(self) -> list[Any]:
        return [parent.recv() for parent in self.parents]

    def step(
        self, actions: Iterable[Any], blocking: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[dict, ...]] | Callable[[], list[Any]]:
        results = self._send_step('step', actions)
        if blocking:
            results = list(chain.from_iterable(results()))
            obs, rewards, dones, infos = zip(*results)
            return np.stack(obs), np.stack(rewards), np.stack(dones), infos
        return results

    def reset(self) -> np.ndarray:
        results = self._send_step('reset', np.zeros(self.num_chunks * self.num_workers))
        results = list(chain.from_iterable(results()))
        return np.stack(results)

    def close(self):
        self._send_step('close', np.zeros(self.num_chunks * self.num_workers))
        for worker in self.workers:
            worker.join()


class ChunkWorker(mp.Process):
    def __init__(
        self,
        env_id: str,
        num_chunks: int,
        connection,
        render: bool = False,
    ):
        mp.Process.__init__(self)
        self.envs = [_make_env(env_id) for _ in range(num_chunks)]
        self.connection = connection
        self.render = render

    def run(self):
        while True:
            cmd, actions = self.connection.recv()
            if cmd == 'step':
                results = []
                for a, env in zip(actions, self.envs):
                    obs, r, terminated, truncated, info = env.step(a)
                    done = RLVecEnv.merge_done(terminated, truncated)
                    info = RLVecEnv.merge_info(info, terminated, truncated)
                    if self.render:
                        env.render()
                    results.append((obs, r, done, info))
                self.connection.send(results)
            elif cmd == 'reset':
                results = []
                for _a, env in zip(actions, self.envs):
                    obs, _info = env.reset()
                    results.append(obs)
                self.connection.send(results)
            elif cmd == 'close':
                for env in self.envs:
                    env.close()
                self.connection.send(1)
                break


class DummyBatchEnv(RLVecEnv):
    """Synchronous (in-process) vec env runner.

    Lower overhead than :class:`BatchEnv` for cheap envs where
    multi-processing is not worth it.
    """

    def __init__(
        self,
        env_constructor: Callable[..., RLEnvBase],
        env_id: str,
        num_envs: int,
        make_args: dict | None = None,
        **env_args,
    ):
        make_args = make_args or {}
        self.envs: list[RLEnvBase] = [
            env_constructor(_make_env(env_id, **make_args), **env_args) for _ in range(num_envs)
        ]

    def __len__(self) -> int:
        return len(self.envs)

    def __getattr__(self, name: str):
        return getattr(self.envs[0], name)

    def step(
        self, actions: Iterable[Any]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[dict, ...]]:
        results: list[tuple[Any, Any, bool, dict]] = []
        for env, action in zip(self.envs, actions):
            obs, r, terminated, truncated, info = env.step(action)
            done = RLVecEnv.merge_done(terminated, truncated)
            info = RLVecEnv.merge_info(info, terminated, truncated)
            results.append((obs, r, done, info))
        obs, rewards, done, info = zip(*results)
        return (
            np.stack(obs).copy(),
            np.stack(rewards).copy(),
            np.stack(done).copy(),
            info,
        )

    def reset(self) -> np.ndarray:
        obs = [env.reset()[0] for env in self.envs]
        return np.stack(obs).copy()

    def close(self):
        for env in self.envs:
            env.close()
