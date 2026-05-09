"""Backend registry and the public ``make()`` entry point.

A *backend* is a pair ``(predicate, adapter_cls)`` where ``predicate``
is a callable that returns ``True`` for env objects the adapter knows
how to wrap.  Backends are tried in registration order; the first
matching adapter wins.

Built-in backends (Gymnasium and legacy ``gym``) are registered at
import time.  Users can teach rlib about new backends — e.g. ``dm_env``,
PettingZoo single-agent slices, EnvPool, an in-house simulator — with
:func:`register_backend` *without* modifying the library:

.. code-block:: python

    from rlib.envs import register_backend, RLEnv

    class MySimAdapter(RLEnv):
        def __init__(self, env): self.env = env
        def reset(self, *, seed=None, options=None): ...
        def step(self, action): ...

    register_backend(lambda e: isinstance(e, MySim), MySimAdapter)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rlib.envs.adapters import GymnasiumAdapter, LegacyGymAdapter
from rlib.envs.base import RLEnv

__all__ = ["make", "register_backend", "wrap"]


_Predicate = Callable[[Any], bool]
_Backends: list[tuple[str, _Predicate, type[RLEnv]]] = []


def register_backend(
    predicate: _Predicate,
    adapter_cls: type[RLEnv],
    *,
    name: str = "",
    prepend: bool = False,
) -> None:
    """Register an adapter for env objects matched by ``predicate``.

    Args:
        predicate: Callable taking a candidate env and returning
            ``True`` if ``adapter_cls`` can wrap it.
        adapter_cls: An :class:`RLEnv` subclass whose ``__init__``
            takes the candidate env as its only positional argument.
        name: Optional human-readable identifier (used by
            ``backend="..."`` in :func:`make` and for debugging).
        prepend: If ``True``, insert at the front of the registry so it
            takes precedence over earlier registrations.
    """
    entry = (name or adapter_cls.__name__, predicate, adapter_cls)
    if prepend:
        _Backends.insert(0, entry)
    else:
        _Backends.append(entry)


def _looks_like_gymnasium(env: Any) -> bool:
    """Heuristic: env is a Gymnasium-style 5-tuple stepper."""
    cls_path = type(env).__module__
    # Duck-type: a real gymnasium env's class lives in the gymnasium package.
    return cls_path.startswith("gymnasium")


def _looks_like_legacy_gym(env: Any) -> bool:
    cls_path = type(env).__module__
    return cls_path.startswith("gym.") or cls_path == "gym"


# Built-in backends: try Gymnasium first, then legacy gym.
register_backend(_looks_like_gymnasium, GymnasiumAdapter, name="gymnasium")
register_backend(_looks_like_legacy_gym, LegacyGymAdapter, name="gym")


def wrap(env: Any, *, backend: str = "auto") -> RLEnv:
    """Wrap an existing env object in the appropriate :class:`RLEnv`.

    If ``env`` is already an :class:`RLEnv`, it is returned
    unchanged.  Otherwise the registry is consulted; pass an explicit
    ``backend="<name>"`` to bypass auto-detection.
    """
    if isinstance(env, RLEnv):
        return env

    if backend != "auto":
        for name, _pred, adapter_cls in _Backends:
            if name == backend:
                return adapter_cls(env)  # type: ignore[call-arg]
        raise ValueError(f"Unknown backend {backend!r}. Registered: {[n for n, _, _ in _Backends]}")

    for _name, predicate, adapter_cls in _Backends:
        try:
            if predicate(env):
                return adapter_cls(env)  # type: ignore[call-arg]
        except Exception:
            continue

    # Last-resort: assume it already follows the modern API.  This lets
    # a hand-rolled ``RLEnv``-shaped object through without forcing the
    # user to register a backend.
    return GymnasiumAdapter(env)


def make(
    env_or_id: Any,
    *,
    backend: str = "auto",
    **kwargs: Any,
) -> RLEnv:
    """Construct (if needed) and wrap an environment.

    * If ``env_or_id`` is a string, it is passed to
      ``gymnasium.make`` (or legacy ``gym.make`` if Gymnasium is not
      installed) and the result is wrapped.
    * Otherwise ``env_or_id`` is treated as an already-constructed env
      and wrapped directly via :func:`wrap`.
    """
    if isinstance(env_or_id, str):
        # Lazy import to avoid a hard dependency at module-load time.
        try:
            import gymnasium as _gym  # type: ignore
        except ImportError:  # pragma: no cover - legacy fallback
            import gym as _gym  # type: ignore
        env = _gym.make(env_or_id, **kwargs)
        return wrap(env, backend=backend)
    return wrap(env_or_id, backend=backend)
