"""Tests for top-level package metadata and lazy submodule imports."""

from __future__ import annotations

import importlib

import pytest

import rlib


def test_version_is_pep440_string() -> None:
    assert isinstance(rlib.__version__, str)
    assert rlib.__version__.count(".") >= 2


# Submodules whose imports are guaranteed by the core dependency set.
# Other agent submodules pull in optional deps (scipy, matplotlib, ...)
# which are not installed in the slim CI image.
_CORE_SUBMODULES = ["A3C", "DDQN", "RND", "RANDAL", "DAAC", "VIN", "envs", "networks", "utils"]
_OPTIONAL_SUBMODULES = ["A2C", "PPO", "Curiosity", "Unreal"]


@pytest.mark.parametrize("attr", _CORE_SUBMODULES)
def test_lazy_submodule_attribute_resolves(attr: str) -> None:
    mod = getattr(rlib, attr)
    assert mod is importlib.import_module(f"rlib.{attr}")


@pytest.mark.parametrize("attr", _OPTIONAL_SUBMODULES)
def test_lazy_submodule_attribute_resolves_optional(attr: str) -> None:
    try:
        mod = importlib.import_module(f"rlib.{attr}")
    except ModuleNotFoundError as exc:
        pytest.skip(f"optional dependency for rlib.{attr} not installed: {exc}")
    assert getattr(rlib, attr) is mod


def test_unknown_attribute_raises() -> None:
    with pytest.raises(AttributeError):
        rlib.this_attribute_does_not_exist  # noqa: B018
