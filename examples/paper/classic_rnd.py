"""RND with PPO Classic Control — Hare 2019, Sec. 3.4 sparse reward solutions.

Per the paper:
    > All PPO-based agents (RANDAL, RND, ICM) use the same base policy
    > hyperparameters as the Classic Control PPO agents seen in
    > Table Control_PPO, with the γ_e and γ_i and other algorithm
    > specific hyperparameters from Tables tbl:RND Atari and tbl:RANDAL Atari.

Run::

    python examples/paper/classic_rnd.py MountainCar-v0
"""

from __future__ import annotations

import sys

import torch

from examples.paper.common import CLASSIC_ENVS, CLASSIC_VAL_STEPS, MLP, classic_envs, get_device
from rlib.networks import PPOConfig
from rlib.RND import RND, PredictorMLP, RNDTrainer, RNDTrainerConfig

TOTAL_STEPS = 2_000_000


def main(env_id: str = "MountainCar-v0") -> None:
    device = get_device()
    train_envs, val_envs = classic_envs(env_id)

    input_size = train_envs.envs[0].observation_space.shape
    action_size = train_envs.envs[0].action_space.n

    model = RND(
        policy_model=MLP,
        target_model=PredictorMLP,
        input_size=input_size,
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
        intr_coeff=1.0,
        extr_coeff=2.0,
        optim=torch.optim.RMSprop,
    ).to(device)

    trainer = RNDTrainer(
        envs=train_envs,
        model=model,
        val_envs=val_envs,
        config=RNDTrainerConfig(
            total_steps=TOTAL_STEPS,
            nsteps=5,
            gamma=0.999,
            gamma_intr=0.99,
            lambda_=0.95,
            return_type="GAE",
            init_obs_steps=250,
            num_epochs=4,
            num_minibatches=1,
            validate_freq=100_000,
            num_val_episodes=8,
            max_val_steps=CLASSIC_VAL_STEPS,
            log_dir=f"logs/paper/RND/{env_id}",
            model_dir=f"models/paper/RND/{env_id}",
        ),
    )
    trainer.train()


if __name__ == "__main__":
    env_id = sys.argv[1] if len(sys.argv) > 1 else "MountainCar-v0"
    if env_id == "all":
        for e in CLASSIC_ENVS:
            main(e)
    else:
        main(env_id)
