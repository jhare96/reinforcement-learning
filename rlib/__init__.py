"""rlib — a small PyTorch reinforcement learning library.

Exposes the package version and convenience submodule aliases. Heavy
agent implementations are imported lazily on attribute access so that
users only pay the import cost (and torch/gymnasium loading cost) for
the agents they actually use.

Example:

    from rlib import A2C, PPO  # lazy imports of the agent submodules
    from rlib.envs import make, RLEnv  # canonical, backend-agnostic env API
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__version__ = "3.1.0"

# Mapping of attribute name -> dotted submodule path
_LAZY_SUBMODULES = {
    "A2C": "rlib.A2C",
    "A3C": "rlib.A3C",
    "PPO": "rlib.PPO",
    "DDQN": "rlib.DDQN",
    "RND": "rlib.RND",
    "RANDAL": "rlib.RANDAL",
    "Curiosity": "rlib.Curiosity",
    "Unreal": "rlib.Unreal",
    "DAAC": "rlib.DAAC",
    "VIN": "rlib.VIN",
    "envs": "rlib.envs",
    "agent": "rlib.agent",
    "training": "rlib.training",
    "utils": "rlib.utils",
}


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        mod = importlib.import_module(_LAZY_SUBMODULES[name])
        globals()[name] = mod
        return mod
    raise AttributeError(f"module 'rlib' has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_SUBMODULES.keys()))


if TYPE_CHECKING:  # pragma: no cover - type checkers only
    from rlib import (  # noqa: F401  # noqa: F401  # noqa: F401
        A2C,
        A3C,
        DAAC,
        DDQN,
        PPO,
        RANDAL,
        RND,
        VIN,
        Curiosity,
        Unreal,
        envs,
        utils,
    )


__all__ = ["__version__", *list(_LAZY_SUBMODULES.keys())]
