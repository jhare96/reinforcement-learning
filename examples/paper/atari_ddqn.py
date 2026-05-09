"""Synchronous n-step Double-DQN Atari baseline (Table: DDQN Atari) — Sec. 3.1.

Hyperparameters reproduce :ref:`tbl:DDQN Atari`::

    Optimiser              = Adam
    Learning rate          = 1e-3
    Number of actors       = 32
    Target network period  = 10,000 steps
    Initial ε              = 1.0
    Final ε                = 0.01
    ε_test                 = 0.01
    n-step period          = 5
    Discount factor γ      = 0.99
    Gradient norm clip     = 0.5

Run::

    python examples/paper/atari_ddqn.py SpaceInvadersDeterministic-v4
"""

from __future__ import annotations

import sys

import torch

from examples.paper.common import ATARI_ENVS, ATARI_VAL_STEPS, NatureCNN, atari_envs, get_device
from rlib.agent import ModelConfig
from rlib.DDQN import DQN, DDQNTrainerConfig, SyncDDQN


def main(env_id: str = "SpaceInvadersDeterministic-v4") -> None:
    device = get_device()
    train_envs, val_envs = atari_envs(env_id)

    input_shape = train_envs.envs[0].reset()[0].shape
    action_size = train_envs.envs[0].action_space.n

    model_cfg = ModelConfig(
        lr=1e-3,
        lr_final=1e-3,
        decay_steps=int(1e9),
        grad_clip=0.5,
        device=device,
    )
    model = DQN(NatureCNN, input_shape, action_size, config=model_cfg, optim=torch.optim.Adam)
    target_model = DQN(
        NatureCNN, input_shape, action_size, config=model_cfg, optim=torch.optim.Adam
    )

    trainer = SyncDDQN(
        envs=train_envs,
        model=model,
        target_model=target_model,
        val_envs=val_envs,
        action_size=action_size,
        config=DDQNTrainerConfig(
            total_steps=50_000_000,
            nsteps=5,
            gamma=0.99,
            update_target_freq=10_000,
            validate_freq=1_000_000,
            num_val_episodes=8,
            max_val_steps=ATARI_VAL_STEPS,
            epsilon_start=1.0,
            epsilon_final=0.01,
            epsilon_steps=2_000_000,  # paper anneals ε over 2M steps
            epsilon_test=0.01,
            log_dir=f"logs/paper/DDQN/{env_id}",
            model_dir=f"models/paper/DDQN/{env_id}",
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
