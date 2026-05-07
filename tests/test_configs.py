"""Tests for the model and trainer config dataclasses."""

from __future__ import annotations

import dataclasses

import pytest

from rlib.networks import A2CConfig, ModelConfig, PPOConfig
from rlib.utils import (
    DAACTrainerConfig,
    DDQNTrainerConfig,
    PPOTrainerConfig,
    RANDALTrainerConfig,
    ReturnType,
    RNDTrainerConfig,
    TrainerConfig,
    TrainMode,
    UnrealTrainerConfig,
)


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


class TestPerTrainerConfigs:
    """Each per-trainer config subclass adds its own fields and inherits TrainerConfig."""

    @pytest.mark.parametrize(
        "cls",
        [
            PPOTrainerConfig,
            RNDTrainerConfig,
            RANDALTrainerConfig,
            DAACTrainerConfig,
            DDQNTrainerConfig,
            UnrealTrainerConfig,
        ],
    )
    def test_inherits_trainer_config(self, cls: type[TrainerConfig]) -> None:
        cfg = cls()
        assert isinstance(cfg, TrainerConfig)
        # Inherited base field is still accessible.
        assert cfg.gamma == 0.99
        # Validators on the base still fire.
        with pytest.raises(ValueError, match="train_mode"):
            cls(train_mode="bogus")  # type: ignore[arg-type]

    def test_ppo_extra_fields(self) -> None:
        cfg = PPOTrainerConfig(num_epochs=2, num_minibatches=8)
        assert cfg.num_epochs == 2
        assert cfg.num_minibatches == 8

    def test_rnd_extra_fields(self) -> None:
        cfg = RNDTrainerConfig(gamma_intr=0.95, init_obs_steps=100)
        assert cfg.gamma_intr == 0.95
        assert cfg.init_obs_steps == 100

    def test_randal_inherits_rnd_fields(self) -> None:
        assert issubclass(RANDALTrainerConfig, RNDTrainerConfig)
        cfg = RANDALTrainerConfig(gamma_intr=0.95, replay_length=500, norm_pixel_reward=False)
        # RND fields
        assert cfg.gamma_intr == 0.95
        # RANDAL-only fields
        assert cfg.replay_length == 500
        assert cfg.norm_pixel_reward is False

    def test_ddqn_extra_fields(self) -> None:
        cfg = DDQNTrainerConfig(epsilon_start=0.5, epsilon_final=0.05, epsilon_steps=1e5)
        assert cfg.epsilon_start == 0.5
        assert cfg.epsilon_final == 0.05
        assert cfg.epsilon_steps == 1e5

    def test_daac_extra_fields(self) -> None:
        cfg = DAACTrainerConfig(policy_epochs=2, value_epochs=4, num_minibatches=2)
        assert cfg.policy_epochs == 2
        assert cfg.value_epochs == 4
        assert cfg.num_minibatches == 2

    def test_unreal_extra_fields(self) -> None:
        cfg = UnrealTrainerConfig(normalise_obs=False, replay_length=1000)
        assert cfg.normalise_obs is False
        assert cfg.replay_length == 1000

    def test_asdict_round_trip(self) -> None:
        cfg = RANDALTrainerConfig(total_steps=42, gamma_intr=0.7, replay_length=1)
        d = dataclasses.asdict(cfg)
        assert RANDALTrainerConfig(**d) == cfg


class TestAutoLoggedHyperparameters:
    """SyncMultiEnvTrainer.__init__ writes a hyperparameters.txt with both
    config and model.config fields."""

    def test_hyperparameters_file_written(self, tmp_path) -> None:
        import gymnasium as gym
        import torch

        from rlib.A2C import A2CTrainer, ActorCritic
        from rlib.networks import A2CConfig
        from rlib.utils.VecEnv import DummyBatchEnv

        class _MLP(torch.nn.Module):
            dense_size = 16

            def __init__(self, input_size):
                super().__init__()
                self.net = torch.nn.Sequential(torch.nn.Linear(input_size[0], 16), torch.nn.Tanh())

            def forward(self, x):
                return self.net(x)

        envs = DummyBatchEnv(lambda e: e, "CartPole-v1", num_envs=2)
        try:
            val = [gym.make("CartPole-v1") for _ in range(2)]
            model = ActorCritic(_MLP, (4,), 2, config=A2CConfig(device="cpu"))
            cfg = TrainerConfig(
                total_steps=10,
                nsteps=5,
                num_val_episodes=2,
                max_val_steps=10,
                log_dir=str(tmp_path / "logs"),
                model_dir=str(tmp_path / "models"),
            )
            A2CTrainer(envs, model, val, config=cfg)
        finally:
            envs.close()

        hp = tmp_path / "logs" / "hyperparameters.txt"
        assert hp.exists()
        content = hp.read_text()
        # Trainer config fields:
        assert "total_steps = 10" in content
        assert "gamma = 0.99" in content
        # Trainer-derived field:
        assert "num_workers = 2" in content
        # Model config fields (prefixed):
        assert "model.lr = 0.001" in content
        assert "model.entropy_coeff = 0.01" in content
        assert "model.value_coeff = 0.5" in content
