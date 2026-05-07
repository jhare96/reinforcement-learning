"""Tests for the model and trainer config dataclasses."""

from __future__ import annotations

import dataclasses

import pytest

from rlib.networks import A2CConfig, ModelConfig, PPOConfig
from rlib.utils import ReturnType, TrainerConfig, TrainMode


class TestModelConfig:
    def test_default_values(self) -> None:
        cfg = ModelConfig()
        assert cfg.lr == 1e-3
        assert cfg.lr_final == 0.0
        assert cfg.decay_steps == 600_000
        assert cfg.grad_clip == 0.5
        assert cfg.device == "cuda"

    def test_frozen(self) -> None:
        cfg = ModelConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.lr = 1e-2  # type: ignore[misc]

    def test_asdict_round_trip(self) -> None:
        cfg = ModelConfig(lr=2e-4, decay_steps=100, device="cpu")
        d = dataclasses.asdict(cfg)
        assert d == {
            "lr": 2e-4,
            "lr_final": 0.0,
            "decay_steps": 100,
            "grad_clip": 0.5,
            "device": "cpu",
        }
        assert ModelConfig(**d) == cfg


class TestA2CConfig:
    def test_inherits_modelconfig_fields(self) -> None:
        cfg = A2CConfig()
        assert isinstance(cfg, ModelConfig)
        assert cfg.lr == 1e-3
        assert cfg.entropy_coeff == 0.01
        assert cfg.value_coeff == 0.5

    def test_overrides(self) -> None:
        cfg = A2CConfig(lr=7e-4, entropy_coeff=0.02, value_coeff=0.25, device="cpu")
        assert cfg.lr == 7e-4
        assert cfg.entropy_coeff == 0.02
        assert cfg.value_coeff == 0.25
        assert cfg.device == "cpu"


class TestPPOConfig:
    def test_inherits_modelconfig_fields(self) -> None:
        cfg = PPOConfig()
        assert isinstance(cfg, ModelConfig)
        assert cfg.entropy_coeff == 0.01
        assert cfg.policy_clip == 0.1


class TestTrainerConfig:
    def test_default_values(self) -> None:
        cfg = TrainerConfig()
        assert cfg.train_mode == "nstep"
        assert cfg.return_type == "nstep"
        assert cfg.gamma == 0.99
        assert cfg.lambda_ == 0.95
        assert cfg.log_scalars is True

    def test_frozen(self) -> None:
        cfg = TrainerConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.gamma = 0.9  # type: ignore[misc]

    @pytest.mark.parametrize("mode", ["foo", "TD", "", "n-step"])
    def test_invalid_train_mode_rejected(self, mode: str) -> None:
        with pytest.raises(ValueError, match="train_mode"):
            TrainerConfig(train_mode=mode)  # type: ignore[arg-type]

    @pytest.mark.parametrize("ret", ["TD", "advantage", "", "nsteps"])
    def test_invalid_return_type_rejected(self, ret: str) -> None:
        with pytest.raises(ValueError, match="return_type"):
            TrainerConfig(return_type=ret)  # type: ignore[arg-type]

    def test_valid_modes_accepted(self) -> None:
        for mode in ("nstep", "onestep"):
            TrainerConfig(train_mode=mode)  # type: ignore[arg-type]
        for ret in ("nstep", "lambda", "GAE"):
            TrainerConfig(return_type=ret)  # type: ignore[arg-type]

    def test_asdict_round_trip(self) -> None:
        cfg = TrainerConfig(total_steps=10_000, gamma=0.95, log_scalars=False)
        d = dataclasses.asdict(cfg)
        assert TrainerConfig(**d) == cfg

    def test_typing_aliases_exist(self) -> None:
        # Smoke-check that the Literal aliases are importable.
        assert TrainMode is not None
        assert ReturnType is not None


class TestModelConfigIntegration:
    """Verify that the Model base class accepts a config and reads its fields."""

    def test_model_accepts_config(self) -> None:
        import torch

        from rlib.networks import Model

        class _Concrete(Model):
            def __init__(self, **kw):
                super().__init__(**kw)
                self.lin = torch.nn.Linear(2, 1).to(self.device)
                self._build_optimiser(torch.optim.SGD)

            def forward(self, x):
                return self.lin(x)

            def evaluate(self, x):
                return self.forward(torch.from_numpy(x).float().to(self.device)).detach().numpy()

            def backprop(self, x, y):
                pass

        cfg = ModelConfig(lr=2e-4, decay_steps=10, grad_clip=None, device="cpu")
        model = _Concrete(config=cfg)
        assert model.config is cfg
        assert model.lr == 2e-4
        assert model.grad_clip is None
        assert model.device == "cpu"

    def test_model_requires_config(self) -> None:
        """Without a config, Model raises a clear TypeError."""
        from rlib.networks import Model

        class _Concrete(Model):
            def forward(self, x):
                return x

            def evaluate(self, x):
                pass

            def backprop(self, x, y):
                pass

        with pytest.raises(TypeError):
            # Missing required positional arg
            _Concrete()  # type: ignore[call-arg]
