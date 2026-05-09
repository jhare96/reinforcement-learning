"""A2C Classic Control — Hare 2019, Sec. 3.4 (Adam vs RMSProp results).

Uses the RMSProp baseline because the paper found it more stable than
Adam for low-dimensional inputs. Hyperparameters follow the original
Atari A2C table (1e-3 / 32 / entropy 0.01 / value 0.5 / nstep 5).

Run::

    python examples/paper/classic_a2c.py CartPole-v1
"""

from __future__ import annotations

import sys

import torch

from examples.paper.common import CLASSIC_ENVS, CLASSIC_VAL_STEPS, MLP, classic_envs, get_device
from rlib.A2C import A2CTrainer, ActorCritic
from rlib.A2C.model import A2CConfig
from rlib.training import Returns, TrainerConfig

TOTAL_STEPS = 5_000_000


def main(env_id: str = "CartPole-v1") -> None:
    device = get_device()
    train_envs, val_envs = classic_envs(env_id)

    input_size = train_envs.envs[0].observation_space.shape
    action_size = train_envs.envs[0].action_space.n

    agent = ActorCritic(
        MLP,
        input_size=input_size,
        action_size=action_size,
        config=A2CConfig(
            lr=1e-3,
            lr_final=1e-3,
            decay_steps=int(1e9),
            grad_clip=0.5,
            entropy_coeff=0.01,
            value_coeff=0.5,
            device=device,
        ),
        optim=torch.optim.RMSprop,
    )

    trainer = A2CTrainer(
        envs=train_envs,
        agent=agent,
        val_envs=val_envs,
        config=TrainerConfig(
            total_steps=TOTAL_STEPS,
            nsteps=5,
            gamma=0.99,
            lambda_=0.95,
            returns=Returns.GAE,
            validate_freq=100_000,
            num_val_episodes=8,
            max_val_steps=CLASSIC_VAL_STEPS,
            log_dir=f"logs/paper/A2C/{env_id}",
            model_dir=f"models/paper/A2C/{env_id}",
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
