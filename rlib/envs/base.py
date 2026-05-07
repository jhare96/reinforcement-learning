"""Canonical environment contract for rlib.

This module defines a single, backend-agnostic environment API that the
rest of the library targets:

* :class:`RLEnv` — a :class:`typing.Protocol` describing the canonical
  contract.  Use it for type annotations.  Any object that *structurally*
  matches it (e.g. a third-party simulator that already exposes the
  modern Gymnasium signature) is automatically an ``RLEnv``.
* :class:`RLEnvBase` — an abstract base class providing the same contract
  with helpful defaults (``__getattr__`` delegation, ``unwrapped``,
  context-manager support, ...).  Backend adapters and wrappers shipped
  with rlib inherit from it.
* :class:`RLVecEnv` — abstract base for vectorised env implementations.
  Concrete vec envs (``BatchEnv``/``DummyBatchEnv``) collapse the
  per-env 5-tuple into the legacy 4-tuple at this single boundary so
  agent rollouts never have to know about ``terminated``/``truncated``.

The canonical per-env contract is the **modern 5-tuple**:

* ``reset(*, seed=None, options=None) -> (obs, info)``
* ``step(action) -> (obs, reward, terminated, truncated, info)``
* ``close()``
* ``observation_space`` / ``action_space`` attributes

This deliberately matches Gymnasium so the most common backend is a
zero-cost pass-through, while other backends (legacy ``gym``,
``dm_env``, PettingZoo, EnvPool, ...) only need a thin adapter file.
"""

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

__all__ = ["RLEnv", "RLEnvBase", "RLVecEnv"]


@runtime_checkable
class RLEnv(Protocol):
    """Structural type describing rlib's canonical single-env contract.

    Any object exposing these members satisfies the protocol; no
    inheritance is required.  Use this in annotations:

    .. code-block:: python

        def my_wrapper(env: RLEnv) -> RLEnv: ...
    """

    observation_space: Any
    action_space: Any

    def reset(self, *, seed: Any = None, options: Any = None) -> tuple[Any, dict]: ...

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]: ...

    def close(self) -> None: ...


class RLEnvBase(ABC):
    """Concrete-friendly base class for rlib adapters and wrappers.

    Subclasses *must* implement :meth:`reset` and :meth:`step` to honour
    the canonical 5-tuple / ``(obs, info)`` contract.  All other
    convenience members (``unwrapped``, ``__getattr__`` forwarding,
    context-manager support, ``render``, ``close``, ``spec``, ...)
    delegate to ``self.env`` when present so wrapper classes get sane
    defaults for free.

    A subclass that does not wrap another env (e.g. a hand-written
    simulator) should set ``self.env = None`` and override the relevant
    members directly.
    """

    #: The wrapped underlying environment, or ``None`` for leaf envs.
    env: Any = None
    #: Default empty metadata dict, mirroring Gymnasium.
    metadata: dict = {}

    @abstractmethod
    def reset(self, *, seed: Any = None, options: Any = None) -> tuple[Any, dict]:
        """Reset the environment and return ``(obs, info)``."""

    @abstractmethod
    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        """Step the environment and return the modern 5-tuple."""

    # ----- helpful defaults that forward to the wrapped env --------

    @property
    def observation_space(self) -> Any:
        return self.env.observation_space

    @property
    def action_space(self) -> Any:
        return self.env.action_space

    @property
    def spec(self) -> Any:
        return getattr(self.env, "spec", None)

    @property
    def unwrapped(self) -> Any:
        """Walk through nested wrappers to the bottom-most env."""
        inner = self.env
        if inner is None:
            return self
        return getattr(inner, "unwrapped", inner)

    def render(self, *args: Any, **kwargs: Any) -> Any:
        if self.env is None:
            raise NotImplementedError("render() must be implemented by leaf envs")
        return self.env.render(*args, **kwargs)

    def close(self) -> None:
        if self.env is not None:
            self.env.close()

    def seed(self, seed: Any = None) -> Any:
        # Best-effort: prefer underlying env's seed() if it exists, else
        # fall back to passing the seed through reset().
        if self.env is not None and hasattr(self.env, "seed"):
            return self.env.seed(seed)
        return None

    def __enter__(self) -> "RLEnvBase":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __getattr__(self, name: str) -> Any:
        # Only called when normal attribute lookup fails.  Avoid
        # recursion for ``self.env`` itself and for dunder names.
        if name.startswith("_") or name == "env":
            raise AttributeError(name)
        env = self.__dict__.get("env", None)
        if env is None:
            raise AttributeError(name)
        return getattr(env, name)


class RLVecEnv(ABC):
    """Abstract base for rlib's vectorised environment runners.

    Agent rollout code in this library has historically consumed the
    legacy 4-tuple ``(obs, rewards, dones, infos)``.  We keep that
    *agent-facing* shape on purpose — the per-env 5-tuple lives on the
    wrapper side, and ``RLVecEnv`` implementations are responsible for
    collapsing ``terminated``/``truncated`` into a single ``done`` flag
    in **one place** (this base class' :meth:`merge_done` helper).
    """

    @abstractmethod
    def reset(self) -> Any:
        """Return a stacked batch of initial observations."""

    @abstractmethod
    def step(self, actions: Any) -> Any:
        """Step every sub-env and return ``(obs, rewards, dones, infos)``."""

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def __len__(self) -> int: ...

    @staticmethod
    def merge_done(terminated: bool, truncated: bool) -> bool:
        """Single canonical place where ``done = terminated or truncated``.

        Centralised so future agents that want to distinguish the two
        (e.g. for correct value-bootstrapping on truncation) only need
        to change call sites here.
        """
        return bool(terminated) or bool(truncated)

    @staticmethod
    def merge_info(info: dict, terminated: bool, truncated: bool) -> dict:
        """Annotate ``info`` with the legacy ``TimeLimit.truncated`` key.

        Mirrors Gymnasium's behaviour so any agent that inspects the
        info dict for truncation sees the same value regardless of
        backend.
        """
        if not isinstance(info, dict):
            return info
        out = dict(info)
        out.setdefault(
            "TimeLimit.truncated",
            bool(truncated) and not bool(terminated),
        )
        return out
