"""Train an A2C agent on CartPole-v1.

Usage::

    pip install -e ".[classic]"
    python examples/cartpole_a2c.py

TensorBoard logs land in ``logs/A2C/CartPole`` and model checkpoints in
``models/A2C/CartPole``::

    tensorboard --logdir logs/
"""

from __future__ import annotations

import gymnasium as gym
import torch

from rlib.A2C import A2C, ActorCritic
from rlib.networks import A2CConfig
from rlib.utils import TrainerConfig
from rlib.utils.VecEnv import DummyBatchEnv


class MLP(torch.nn.Module):
    """A tiny MLP body for low-dimensional state spaces (e.g. CartPole)."""

    def __init__(self, input_size, hidden_size: int = 64):
        super().__init__()
        in_dim = int(input_size[0]) if hasattr(input_size, "__len__") else int(input_size)
        self.dense_size = hidden_size
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    env_id = "CartPole-v1"
    num_envs = 8
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Classic control envs need no preprocessing — pass them through.
    def make(env):
        return env

    train_envs = DummyBatchEnv(make, env_id, num_envs=num_envs)
    val_envs = [gym.make(env_id) for _ in range(4)]

    input_size = train_envs.envs[0].observation_space.shape
    num_actions = train_envs.envs[0].action_space.n

    model = ActorCritic(
        MLP,
        input_size=input_size,
        action_size=num_actions,
        config=A2CConfig(
            lr=7e-4,
            lr_final=0.0,
            decay_steps=int(1e5),
            grad_clip=0.5,
            entropy_coeff=0.01,
            value_coeff=0.5,
            device=device,
        ),
    )

    trainer = A2C(
        envs=train_envs,
        model=model,
        val_envs=val_envs,
        config=TrainerConfig(
            total_steps=int(1e5),
            nsteps=5,
            validate_freq=int(2e4),
            num_val_episodes=10,
            max_val_steps=500,
            log_dir="logs/A2C/CartPole",
            model_dir="models/A2C/CartPole",
        ),
    )

    trainer.train()


if __name__ == "__main__":
    main()
