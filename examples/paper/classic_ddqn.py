"""Synchronous n-step Double-DQN Classic Control — Sec. 3.4.

Uses RMSProp (paper's preferred optimiser for low-dim envs).
The ε schedule anneals over half of training (1M steps) per the paper.

Run::

    python examples/paper/classic_ddqn.py CartPole-v1
"""

from __future__ import annotations

import sys

import torch

from examples.paper.common import CLASSIC_ENVS, CLASSIC_VAL_STEPS, MLP, classic_envs, get_device
from rlib.agent import ModelConfig
from rlib.DDQN import DQN, DDQNTrainerConfig, SyncDDQN
from rlib.training import Returns

TOTAL_STEPS = 2_000_000


def main(env_id: str = "CartPole-v1") -> None:
    device = get_device()
    train_envs, val_envs = classic_envs(env_id)

    input_size = train_envs.envs[0].observation_space.shape
    action_size = train_envs.envs[0].action_space.n

    model_cfg = ModelConfig(
        lr=1e-3,
        lr_final=1e-3,
        decay_steps=int(1e9),
        grad_clip=0.5,
        device=device,
    )
    agent = DQN(MLP, input_size, action_size, config=model_cfg, optim=torch.optim.RMSprop)
    target_agent = DQN(MLP, input_size, action_size, config=model_cfg, optim=torch.optim.RMSprop)

    trainer = SyncDDQN(
        envs=train_envs,
        agent=agent,
        target_agent=target_agent,
        val_envs=val_envs,
        action_size=action_size,
        config=DDQNTrainerConfig(
            total_steps=TOTAL_STEPS,
            nsteps=5,
            gamma=0.99,
            lambda_=0.95,
            returns=Returns.GAE,
            update_target_freq=10_000,
            validate_freq=100_000,
            num_val_episodes=8,
            max_val_steps=CLASSIC_VAL_STEPS,
            epsilon_start=1.0,
            epsilon_final=0.1,  # paper finds 0.1 best for classic
            epsilon_steps=1_000_000,  # anneal over half of training
            epsilon_test=0.01,
            log_dir=f"logs/paper/DDQN/{env_id}",
            model_dir=f"models/paper/DDQN/{env_id}",
        ),
    )
    trainer.train()


if __name__ == "__main__":
    env_id = sys.argv[1] if len(sys.argv) > 1 else "CartPole-v1"
    if env_id == "all":
        for e in CLASSIC_ENVS:
            main(e)
    else:
        main(env_id)
