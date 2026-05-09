"""YAML-driven object instantiation for ``python -m rlib.<Agent>`` runners.

Inspired by Hydra / OmegaConf but kept tiny on purpose: ~1 file, no
runtime framework. The YAML config *is* the constructor graph — any
mapping with a ``constructor`` key is recursively instantiated by
importing the target and passing the remaining keys as kwargs.

YAML schema
===========

.. code-block:: yaml

    env:
      constructor: rlib._cli.classic_envs    # or atari_envs, or your own callable
      id: CartPole-v1
      num_envs: 8
      num_val_envs: 4

    agent:
      constructor: rlib.A2C.ActorCritic
      model:                                  # body class, not instance
        constructor: rlib.models.MLP
        partial: true
        hidden_size: 64
      input_size: ${input_shape}
      action_size: ${action_size}
      config:
        constructor: rlib.A2C.A2CConfig
        lr: 7.0e-4
        device: ${device}

    trainer:
      constructor: rlib.A2C.A2CTrainer
      envs: ${train_envs}
      val_envs: ${val_envs}
      agent: ${agent}
      config:
        constructor: rlib.training.TrainerConfig
        total_steps: 100_000
        returns: GAE                          # str → Returns enum (auto-coerced)

Conventions
-----------

* ``constructor: dotted.path.to.Class`` — import & instantiate. Remaining
  keys are recursively instantiated then passed as kwargs.
* ``partial: true`` — return :func:`functools.partial(target, **kwargs)`
  instead of calling. Useful when the host constructor wants the *class*
  (e.g. ``ActorCritic`` takes a body class and instantiates it itself
  with the env-derived ``input_size``).
* ``${name}`` — string-only interpolation, resolved against a namespace
  populated by the runner: ``device``, plus whatever the env factory
  returns (``input_shape``, ``action_size``, ``train_envs``,
  ``val_envs``), plus ``agent`` once it's been built.

Extension
---------

Replace any node by pointing ``constructor`` at your own class — no rlib
changes needed::

    agent:
      constructor: my_pkg.experiments.MyAgent
      input_shape: ${input_shape}
      action_size: ${action_size}
      ...

Custom env factories follow the same protocol — return a dict with
``train_envs``, ``val_envs``, ``input_shape``, ``action_size``.
"""

from __future__ import annotations

import argparse
import ast
import functools
import importlib
import re
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch
import yaml

from rlib.envs.vec_env import BatchEnv, DummyBatchEnv
from rlib.envs.wrappers import AtariEnv
from rlib.training import Returns, TrainMode

__all__ = [
    "atari_envs",
    "auto_device",
    "build_runner_parser",
    "classic_envs",
    "clone_module",
    "instantiate",
    "load_yaml",
    "run_from_yaml",
]

_INTERP_RE = re.compile(r"^\$\{([a-zA-Z_][a-zA-Z0-9_.]*)\}$")
_ENUM_FIELDS: dict[str, type] = {"returns": Returns, "train_mode": TrainMode}


# ---------------------------------------------------------------------------
# Devices + helpers
# ---------------------------------------------------------------------------


def auto_device() -> str:
    """Pick the best available torch device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def clone_module(module: torch.nn.Module) -> torch.nn.Module:
    """Deep-copy a torch module and sync its weights from the source.

    Used from YAML configs that need a sibling network (e.g. the DDQN
    target network)::

        target_agent:
          constructor: rlib._cli.clone_module
          module: ${agent}
    """
    import copy

    clone = copy.deepcopy(module)
    clone.load_state_dict(module.state_dict())
    return clone


# ---------------------------------------------------------------------------
# Built-in env factories (return a dict the runner unpacks into the
# interpolation namespace).
# ---------------------------------------------------------------------------


def classic_envs(
    id: str,  # noqa: A002 — matches YAML "id" key
    num_envs: int = 8,
    num_val_envs: int = 4,
) -> dict[str, Any]:
    """Build a vector-obs gym env bundle (no preprocessing)."""
    train_envs = DummyBatchEnv(lambda env: env, id, num_envs=num_envs)
    val_envs = [gym.make(id) for _ in range(num_val_envs)]
    return {
        "train_envs": train_envs,
        "val_envs": val_envs,
        "input_shape": train_envs.envs[0].observation_space.shape,
        "action_size": train_envs.envs[0].action_space.n,
    }


def atari_envs(
    id: str,  # noqa: A002
    num_envs: int = 8,
    num_val_envs: int = 4,
    frame_stack: int = 4,
    episodic: bool = True,
    clip_reward: bool = True,
    auto_reset: bool = False,
    fire_reset: bool = False,
) -> dict[str, Any]:
    """Build an Atari env bundle with the standard frame-stack wrapper.

    Auto-registers the ALE namespace with Gymnasium on first call so
    ``id="ALE/Breakout-v5"`` (and similar) resolve out of the box when
    ``ale-py`` is installed via the ``[atari]`` extra.

    Args:
        fire_reset: If True, wrap each env in :class:`FireResetEnv` so
            games that need the FIRE button to begin (Breakout, Pong,
            ...) auto-press it after every reset.
    """
    try:
        import ale_py  # type: ignore[import-not-found]

        gym.register_envs(ale_py)
    except ImportError:
        pass  # gym.make() will raise a clearer error if needed

    probe = AtariEnv(
        gym.make(id), k=frame_stack, episodic=False, reset=fire_reset, clip_reward=False
    )
    input_shape = probe.reset()[0].shape
    action_size = probe.action_space.n
    probe.close()
    train_envs = BatchEnv(
        AtariEnv,
        id,
        num_envs=num_envs,
        blocking=False,
        k=frame_stack,
        episodic=episodic,
        reset=fire_reset,
        clip_reward=clip_reward,
        auto_reset=auto_reset,
    )
    val_envs = [
        AtariEnv(gym.make(id), k=frame_stack, episodic=False, reset=fire_reset, clip_reward=False)
        for _ in range(num_val_envs)
    ]
    return {
        "train_envs": train_envs,
        "val_envs": val_envs,
        "input_shape": input_shape,
        "action_size": action_size,
    }


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


def _import_target(path: str) -> Any:
    """Resolve ``"pkg.module.attr"`` to the attribute itself."""
    if ":" in path:
        module_path, _, attr = path.partition(":")
    elif "." in path:
        module_path, _, attr = path.rpartition(".")
    else:
        raise ValueError(f"constructor must be a dotted path, got {path!r}")
    module = importlib.import_module(module_path)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise ImportError(f"Cannot find {attr!r} in {module_path!r}") from exc


def _resolve_interpolation(text: str, interp: dict[str, Any]) -> Any:
    match = _INTERP_RE.match(text)
    if not match:
        return text
    key = match.group(1)
    cursor: Any = interp
    for part in key.split("."):
        if isinstance(cursor, dict):
            if part not in cursor:
                raise KeyError(f"interpolation ${{{key}}} not found in namespace")
            cursor = cursor[part]
        else:
            cursor = getattr(cursor, part)
    return cursor


def instantiate(node: Any, interp: dict[str, Any] | None = None) -> Any:
    """Recursively instantiate a YAML node.

    See module docstring for the semantics of ``constructor``,
    ``partial`` and ``${name}``.
    """
    interp = interp or {}
    if isinstance(node, str):
        return _resolve_interpolation(node, interp)
    if isinstance(node, list):
        return [instantiate(item, interp) for item in node]
    if not isinstance(node, dict):
        return node

    if "constructor" not in node:
        return {k: instantiate(v, interp) for k, v in node.items()}

    spec = dict(node)
    target = _import_target(spec.pop("constructor"))
    partial = bool(spec.pop("partial", False))
    kwargs = {k: instantiate(v, interp) for k, v in spec.items()}

    # Auto-coerce string enums where the host expects a Returns / TrainMode.
    for field, enum_cls in _ENUM_FIELDS.items():
        value = kwargs.get(field)
        if isinstance(value, str):
            kwargs[field] = (
                enum_cls[value.upper()] if enum_cls is Returns else enum_cls(value.lower())
            )

    if partial:
        return functools.partial(target, **kwargs) if kwargs else target
    return target(**kwargs)


# ---------------------------------------------------------------------------
# YAML loading + --set overrides
# ---------------------------------------------------------------------------


def load_yaml(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Read ``path`` and apply ``key.path=value`` overrides in order.

    Override values are parsed with :func:`ast.literal_eval` so e.g.
    ``--set trainer.config.total_steps=1_000_000`` works without
    quoting; bare strings fall back to the raw value.
    """
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML in {path} must be a mapping, got {type(data).__name__}")
    for override in overrides or []:
        _apply_override(data, override)
    return data


def _apply_override(data: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ValueError(f"--set override must be 'key.path=value', got {override!r}")
    key_path, raw_value = override.split("=", 1)
    try:
        value: Any = ast.literal_eval(raw_value)
    except (ValueError, SyntaxError):
        value = raw_value
    keys = key_path.split(".")
    cursor: Any = data
    for k in keys[:-1]:
        if not isinstance(cursor, dict) or k not in cursor or not isinstance(cursor[k], dict):
            cursor.setdefault(k, {})
        cursor = cursor[k]
    cursor[keys[-1]] = value


# ---------------------------------------------------------------------------
# Runner entry point
# ---------------------------------------------------------------------------


def build_runner_parser(prog: str, description: str) -> argparse.ArgumentParser:
    """Standard CLI surface for every ``python -m rlib.<Agent>`` runner."""
    parser = argparse.ArgumentParser(prog=prog, description=description)
    parser.add_argument(
        "config",
        type=Path,
        help="Path to YAML config (required).",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="overrides",
        metavar="KEY.PATH=VALUE",
        help="Override a YAML field; may be passed multiple times.",
    )
    return parser


def run_from_yaml(prog: str, argv: list[str] | None = None) -> None:
    """Build & train an agent from a YAML spec.

    Workflow:

    1. Parse CLI / load YAML.
    2. Instantiate ``env`` (callable returning ``train_envs``,
       ``val_envs``, ``input_shape``, ``action_size``); merge into the
       interpolation namespace.
    3. Instantiate ``agent`` and bind to ``${agent}``.
    4. Instantiate ``trainer`` (typically referencing ``${train_envs}``,
       ``${val_envs}``, ``${agent}``).
    5. Call ``trainer.train()``.
    """
    parser = build_runner_parser(prog=prog, description=__doc__ or "")
    args = parser.parse_args(argv)
    spec = load_yaml(args.config, args.overrides)

    interp: dict[str, Any] = {"device": auto_device()}

    env_bundle = instantiate(spec["env"], interp)
    if not isinstance(env_bundle, dict):
        raise TypeError(
            "env factory must return a dict with train_envs / val_envs / input_shape / action_size, "
            f"got {type(env_bundle).__name__}"
        )
    interp.update(env_bundle)

    interp["agent"] = instantiate(spec["agent"], interp)

    trainer = instantiate(spec["trainer"], interp)
    trainer.train()
