"""UNREAL-A2C2 Atari (Table: UNREAL Atari) — Hare 2019, Sec. 3.3.

Hyperparameters reproduce :ref:`tbl:UNREAL Atari`::

    Optimiser                      = Adam
    Learning rate                  = 1e-3
    Number of actors               = 32
    Entropy coefficient            = 0.001
    Value coefficient              = 0.5
    n-step period                  = 20
    Discount factor γ              = 0.99
    Gradient norm clip             = 0.5
    Reward prediction coefficient  = 1
    Value replay coefficient       = 1
    Pixel control coefficient      = 1
    Replay length per actor        = 2000

Run::

    python examples/paper/scripts/atari_unreal.py MontezumaRevengeDeterministic-v4
"""

from __future__ import annotations

import sys

from examples.paper.scripts.common import (
    ATARI_ENVS,
    ATARI_VAL_STEPS,
    NatureCNN,
    atari_envs,
    get_device,
)
from rlib.A2C.model import A2CConfig
from rlib.training import Returns
from rlib.Unreal import UnrealA2C2, UnrealTrainer, UnrealTrainerConfig


def main(env_id: str = "MontezumaRevengeDeterministic-v4") -> None:
    device = get_device()
    train_envs, val_envs = atari_envs(env_id)

    input_shape = train_envs.envs[0].reset()[0].shape
    action_size = train_envs.envs[0].action_space.n

    agent = UnrealA2C2(
        NatureCNN,
        input_shape=input_shape,
        action_size=action_size,
        config=A2CConfig(
            lr=1e-3,
            lr_final=1e-4,
            decay_steps=int(50e6 // (32 * 20)),
            grad_clip=0.5,
            entropy_coeff=0.001,
            value_coeff=0.5,
            device=device,
        ),
        pixel_control=True,
        RP=1.0,
        VR=1.0,
        PC=1.0,
    ).to(device)

    trainer = UnrealTrainer(
        envs=train_envs,
        agent=agent,
        val_envs=val_envs,
        config=UnrealTrainerConfig(
            total_steps=50_000_000,
            nsteps=20,
            gamma=0.99,
            lambda_=0.95,
            returns=Returns.GAE,
            validate_freq=1_000_000,
            num_val_episodes=8,
            max_val_steps=ATARI_VAL_STEPS,
            normalise_obs=True,
            replay_length=2000,
            log_dir=f"logs/paper/UNREAL/{env_id}",
            model_dir=f"models/paper/UNREAL/{env_id}",
        ),
    )
    trainer.train()


if __name__ == "__main__":
    env_id = sys.argv[1] if len(sys.argv) > 1 else "MontezumaRevengeDeterministic-v4"
    if env_id == "all":
        for e in ATARI_ENVS:
            main(e)
    else:
        main(env_id)
