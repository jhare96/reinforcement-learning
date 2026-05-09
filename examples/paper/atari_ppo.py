"""PPO Atari (Table: PPO Atari) — Hare 2019, Sec. 3.2.

Hyperparameters reproduce :ref:`tbl:PPO Atari`::

    Optimiser              = Adam
    Learning rate          = 1e-4
    Number of actors       = 32
    Entropy coefficient    = 0.01
    Value coefficient      = 0.5
    n-step period          = 128
    Number of epochs       = 4
    Number of minibatches  = 4
    Discount factor γ      = 0.99
    PPO clip range         = [0.9, 1.1]   (i.e. policy_clip = 0.1)
    Gradient norm clip     = 0.5

Run::

    python examples/paper/atari_ppo.py SpaceInvadersDeterministic-v4
"""

from __future__ import annotations

import sys

from examples.paper.common import ATARI_ENVS, ATARI_VAL_STEPS, NatureCNN, atari_envs, get_device
from rlib.networks import PPOConfig
from rlib.PPO import PPO, PPOTrainer, PPOTrainerConfig
from rlib.training import Returns

NSTEPS = 128
NUM_WORKERS = 32
TOTAL_STEPS = 50_000_000


def main(env_id: str = "SpaceInvadersDeterministic-v4") -> None:
    device = get_device()
    train_envs, val_envs = atari_envs(env_id, num_envs=NUM_WORKERS)

    input_shape = train_envs.envs[0].reset()[0].shape
    action_size = train_envs.envs[0].action_space.n

    model = PPO(
        NatureCNN,
        input_shape=input_shape,
        action_size=action_size,
        config=PPOConfig(
            lr=1e-4,
            lr_final=1e-4,
            decay_steps=TOTAL_STEPS // (NUM_WORKERS * NSTEPS),
            grad_clip=0.5,
            entropy_coeff=0.01,
            policy_clip=0.1,
            device=device,
        ),
        value_coeff=0.5,
    )

    trainer = PPOTrainer(
        envs=train_envs,
        model=model,
        val_envs=val_envs,
        config=PPOTrainerConfig(
            total_steps=TOTAL_STEPS,
            nsteps=NSTEPS,
            gamma=0.99,
            lambda_=0.95,
            returns=Returns.GAE,
            num_epochs=4,
            num_minibatches=4,
            validate_freq=1_000_000,
            num_val_episodes=8,
            max_val_steps=ATARI_VAL_STEPS,
            log_dir=f"logs/paper/PPO/{env_id}",
            model_dir=f"models/paper/PPO/{env_id}",
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
