"""A2C Atari baseline (Table: A2C Atari) — Hare 2019, Sec. 3.1.

Hyperparameters reproduce :ref:`tbl:A2C Atari`::

    Optimiser            = Adam
    Learning rate        = 1e-3
    Number of actors     = 32
    Entropy coefficient  = 0.01
    Value coefficient    = 0.5
    n-step period        = 5
    Discount factor γ    = 0.99
    Gradient norm clip   = 0.5

Run::

    python examples/paper/atari_a2c.py SpaceInvadersDeterministic-v4
"""

from __future__ import annotations

import sys

import torch

from examples.paper.common import (
    ATARI_ENVS,
    ATARI_VAL_STEPS,
    NatureCNN,
    atari_envs,
    get_device,
)
from rlib.A2C import A2CTrainer, ActorCritic
from rlib.networks import A2CConfig
from rlib.training import TrainerConfig


def main(env_id: str = "SpaceInvadersDeterministic-v4") -> None:
    device = get_device()
    train_envs, val_envs = atari_envs(env_id)

    input_shape = train_envs.envs[0].reset()[0].shape
    action_size = train_envs.envs[0].action_space.n

    model = ActorCritic(
        NatureCNN,
        input_size=input_shape,
        action_size=action_size,
        config=A2CConfig(
            lr=1e-3,
            lr_final=1e-3,  # constant LR (paper does no decay for A2C baseline)
            decay_steps=int(1e9),
            grad_clip=0.5,
            entropy_coeff=0.01,
            value_coeff=0.5,
            device=device,
        ),
        optim=torch.optim.Adam,
    )

    trainer = A2CTrainer(
        envs=train_envs,
        model=model,
        val_envs=val_envs,
        config=TrainerConfig(
            total_steps=50_000_000,
            nsteps=5,
            gamma=0.99,
            validate_freq=1_000_000,
            num_val_episodes=8,
            max_val_steps=ATARI_VAL_STEPS,
            log_dir=f"logs/paper/A2C/{env_id}",
            model_dir=f"models/paper/A2C/{env_id}",
        ),
    )
    trainer.train()


if __name__ == "__main__":
    env_id = sys.argv[1] if len(sys.argv) > 1 else "SpaceInvadersDeterministic-v4"
    if env_id == "all":
        for e in ATARI_ENVS:
            main(e)
    else:
        main(env_id)
