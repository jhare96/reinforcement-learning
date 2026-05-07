"""Train RND on Montezuma's Revenge — a sparse-reward Atari benchmark.

Usage::

    pip install -e ".[atari]"
    python examples/montezuma_rnd.py

This script is the canonical playground for the RND-family agents in this
library. It demonstrates how RND combines a PPO policy with an intrinsic
reward derived from prediction error against a fixed random network — the
core method behind RANDAL as well.

This recipe is geared toward smoke-testing and demonstration; for paper-grade
runs increase ``total_steps`` substantially (the original RND paper trained
for billions of frames) and, ideally, run on GPU.
"""

from __future__ import annotations

import torch

from rlib.RND import RND, RNDTrainer, PredictorCNN
from rlib.networks.networks import UniverseCNN
from rlib.utils.VecEnv import BatchEnv
from rlib.utils.gym_compat import gym
from rlib.utils.wrappers import AtariEnv


def main() -> None:
    env_id = "MontezumaRevengeDeterministic-v4"
    num_envs = 16
    nsteps = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"

    probe = AtariEnv(gym.make(env_id), k=4, episodic=False, reset=False, clip_reward=False)
    input_shape = probe.reset().shape  # (4, 84, 84) by default
    action_size = probe.action_space.n
    probe.close()

    train_envs = BatchEnv(AtariEnv, env_id, num_envs=num_envs, blocking=False,
                          k=4, reset=False, episodic=False, clip_reward=True, auto_reset=True)
    val_envs = BatchEnv(AtariEnv, env_id, num_envs=4, blocking=False,
                        k=4, reset=False, episodic=False, clip_reward=False, auto_reset=True)

    model = RND(
        policy_model=UniverseCNN,
        target_model=PredictorCNN,
        input_size=input_shape,
        action_size=action_size,
        lr=1e-4,
        lr_final=0.0,
        decay_steps=int(1e7) // (num_envs * nsteps),
        grad_clip=0.5,
        intr_coeff=1.0,
        extr_coeff=2.0,
        entropy_coeff=0.001,
        device=device,
    ).to(device)

    trainer = RNDTrainer(
        envs=train_envs,
        model=model,
        val_envs=val_envs,
        total_steps=int(1e7),
        nsteps=nsteps,
        validate_freq=int(5e5),
        num_val_episodes=8,
        log_dir=f"logs/RND/{env_id}",
        model_dir=f"models/RND/{env_id}",
    )
    trainer.train()


if __name__ == "__main__":
    main()
