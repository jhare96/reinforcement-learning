"""PPO Classic Control (Table: Control_PPO) — Hare 2019, Sec. 3.4.

Hyperparameters reproduce Table ``Control_PPO``::

    Optimiser              = RMSProp
    Learning rate          = 1e-3
    Number of actors       = 32
    Entropy coefficient    = 0.01
    Value coefficient      = 0.5
    n-step period          = 5
    Number of epochs       = 4
    Number of minibatches  = 1
    Discount factor γ      = 0.99
    PPO clip range         = [0.9, 1.1]
    Gradient norm clip     = 0.5

Run::

    python examples/paper/classic_ppo.py CartPole-v1
"""

from __future__ import annotations

import sys

import torch

from examples.paper.common import CLASSIC_ENVS, CLASSIC_VAL_STEPS, MLP, classic_envs, get_device
from rlib.networks import PPOConfig
from rlib.PPO import PPO, PPOTrainer, PPOTrainerConfig

TOTAL_STEPS = 2_000_000


def main(env_id: str = "CartPole-v1") -> None:
    device = get_device()
    train_envs, val_envs = classic_envs(env_id)

    input_size = train_envs.envs[0].observation_space.shape
    action_size = train_envs.envs[0].action_space.n

    model = PPO(
        MLP,
        input_shape=input_size,
        action_size=action_size,
        config=PPOConfig(
            lr=1e-3,
            lr_final=1e-3,
            decay_steps=int(1e9),
            grad_clip=0.5,
            entropy_coeff=0.01,
            policy_clip=0.1,
            device=device,
        ),
        value_coeff=0.5,
        optim=torch.optim.RMSprop,
    )

    trainer = PPOTrainer(
        envs=train_envs,
        model=model,
        val_envs=val_envs,
        config=PPOTrainerConfig(
            total_steps=TOTAL_STEPS,
            nsteps=5,
            gamma=0.99,
            lambda_=0.95,
            return_type="GAE",
            num_epochs=4,
            num_minibatches=1,
            validate_freq=100_000,
            num_val_episodes=8,
            max_val_steps=CLASSIC_VAL_STEPS,
            log_dir=f"logs/paper/PPO/{env_id}",
            model_dir=f"models/paper/PPO/{env_id}",
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
