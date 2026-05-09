"""Tests for the YAML instantiation helpers in :mod:`rlib._cli`."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import pytest
import torch

from rlib._cli import (
    auto_device,
    classic_envs,
    clone_module,
    instantiate,
    load_yaml,
)
from rlib.A2C import A2CConfig
from rlib.agent import ModelConfig
from rlib.PPO import PPOConfig
from rlib.training import Returns, TrainerConfig, TrainMode

# ---------------------------------------------------------------------------
# instantiate
# ---------------------------------------------------------------------------


class TestInstantiate:
    def test_passthrough_primitives(self) -> None:
        assert instantiate(42) == 42
        assert instantiate(None) is None
        assert instantiate(3.14) == 3.14
        assert instantiate(True) is True

    def test_passthrough_dict_without_constructor(self) -> None:
        assert instantiate({"a": 1, "b": 2}) == {"a": 1, "b": 2}

    def test_passthrough_list(self) -> None:
        assert instantiate([1, "x", 3.0]) == [1, "x", 3.0]

    def test_constructor_instantiates(self) -> None:
        cfg = instantiate({"constructor": "rlib.agent.ModelConfig", "lr": 1e-3, "device": "cpu"})
        assert isinstance(cfg, ModelConfig)
        assert cfg.lr == 1e-3
        assert cfg.device == "cpu"

    def test_constructor_a2c_config_subclass(self) -> None:
        cfg = instantiate(
            {"constructor": "rlib.A2C.A2CConfig", "entropy_coeff": 0.05, "device": "cpu"}
        )
        assert isinstance(cfg, A2CConfig)
        assert cfg.entropy_coeff == 0.05

    def test_partial_returns_partial(self) -> None:
        result = instantiate(
            {
                "constructor": "rlib.models.MLP",
                "partial": True,
                "hidden_size": 64,
            }
        )
        assert isinstance(result, partial)
        body = result((4,))
        assert isinstance(body, torch.nn.Module)
        assert body.dense_size == 64

    def test_partial_no_kwargs_returns_class(self) -> None:
        result = instantiate({"constructor": "rlib.models.MLP", "partial": True})
        from rlib.models import MLP

        assert result is MLP

    def test_recursive_instantiation_with_enum_coercion(self) -> None:
        spec = {
            "constructor": "rlib.training.TrainerConfig",
            "total_steps": 1000,
            "returns": "GAE",
        }
        cfg = instantiate(spec)
        assert isinstance(cfg, TrainerConfig)
        assert cfg.total_steps == 1000
        assert cfg.returns is Returns.GAE

    def test_train_mode_string_coerced(self) -> None:
        cfg = instantiate({"constructor": "rlib.training.TrainerConfig", "train_mode": "onestep"})
        assert cfg.train_mode is TrainMode.ONESTEP

    def test_unknown_constructor_raises(self) -> None:
        with pytest.raises(ImportError):
            instantiate({"constructor": "rlib.does.not.Exist"})

    def test_constructor_must_be_dotted(self) -> None:
        with pytest.raises(ValueError, match="dotted path"):
            instantiate({"constructor": "noModule"})


class TestInterpolation:
    def test_simple_interpolation(self) -> None:
        assert instantiate("${x}", {"x": 7}) == 7

    def test_nested_dict_interpolation(self) -> None:
        result = instantiate({"value": "${x}"}, {"x": "hi"})
        assert result == {"value": "hi"}

    def test_dotted_interpolation(self) -> None:
        assert instantiate("${a.b}", {"a": {"b": 42}}) == 42

    def test_attribute_interpolation(self) -> None:
        class Box:
            n = 99

        assert instantiate("${box.n}", {"box": Box()}) == 99

    def test_missing_key_raises(self) -> None:
        with pytest.raises(KeyError):
            instantiate("${missing}", {"x": 1})

    def test_non_interpolation_string_passthrough(self) -> None:
        # Strings that don't fully match the interpolation regex are kept verbatim.
        assert instantiate("hello") == "hello"
        assert instantiate("logs/${x}/run") == "logs/${x}/run"


# ---------------------------------------------------------------------------
# load_yaml + --set overrides
# ---------------------------------------------------------------------------


class TestLoadYaml:
    def test_basic_load(self, tmp_path: Path) -> None:
        cfg = tmp_path / "c.yaml"
        cfg.write_text("a: 1\nb:\n  c: 2\n")
        assert load_yaml(cfg) == {"a": 1, "b": {"c": 2}}

    def test_override_int(self, tmp_path: Path) -> None:
        cfg = tmp_path / "c.yaml"
        cfg.write_text("trainer:\n  total_steps: 100\n")
        assert load_yaml(cfg, ["trainer.total_steps=1_000_000"]) == {
            "trainer": {"total_steps": 1_000_000}
        }

    def test_override_string_fallback(self, tmp_path: Path) -> None:
        cfg = tmp_path / "c.yaml"
        cfg.write_text("env:\n  id: CartPole-v1\n")
        assert load_yaml(cfg, ["env.id=Acrobot-v1"]) == {"env": {"id": "Acrobot-v1"}}

    def test_override_creates_nested(self, tmp_path: Path) -> None:
        cfg = tmp_path / "c.yaml"
        cfg.write_text("{}\n")
        assert load_yaml(cfg, ["a.b.c=3"]) == {"a": {"b": {"c": 3}}}

    def test_override_invalid_format(self, tmp_path: Path) -> None:
        cfg = tmp_path / "c.yaml"
        cfg.write_text("{}\n")
        with pytest.raises(ValueError, match="must be 'key.path=value'"):
            load_yaml(cfg, ["bogus"])

    def test_top_level_must_be_mapping(self, tmp_path: Path) -> None:
        cfg = tmp_path / "c.yaml"
        cfg.write_text("- 1\n- 2\n")
        with pytest.raises(ValueError, match="must be a mapping"):
            load_yaml(cfg)


# ---------------------------------------------------------------------------
# Env factories + helpers
# ---------------------------------------------------------------------------


class TestEnvFactories:
    def test_classic_envs_returns_bundle(self) -> None:
        bundle = classic_envs(id="CartPole-v1", num_envs=2, num_val_envs=1)
        try:
            assert set(bundle) == {"train_envs", "val_envs", "input_shape", "action_size"}
            assert bundle["action_size"] == 2
            assert bundle["input_shape"] == (4,)
            assert len(bundle["val_envs"]) == 1
        finally:
            bundle["train_envs"].close()
            for env in bundle["val_envs"]:
                env.close()


class TestCloneModule:
    def test_clone_is_independent(self) -> None:
        src = torch.nn.Linear(4, 2)
        clone = clone_module(src)
        assert clone is not src
        for a, b in zip(src.parameters(), clone.parameters(), strict=False):
            assert torch.equal(a, b)
        with torch.no_grad():
            clone.weight.zero_()
        assert not torch.equal(src.weight, clone.weight)


class TestAutoDevice:
    def test_returns_known_device(self) -> None:
        assert auto_device() in {"cuda", "mps", "cpu"}


# ---------------------------------------------------------------------------
# PPOConfig instantiation via constructor
# ---------------------------------------------------------------------------


class TestPPOConfigInstantiation:
    """Sanity-check that PPOConfig (a ModelConfig subclass) works through ``constructor``."""

    def test_round_trip(self) -> None:
        cfg = instantiate(
            {
                "constructor": "rlib.PPO.PPOConfig",
                "policy_clip": 0.2,
                "entropy_coeff": 0.02,
                "device": "cpu",
            }
        )
        assert isinstance(cfg, PPOConfig)
        assert cfg.policy_clip == 0.2
