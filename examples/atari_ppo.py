"""Train PPO on an Atari game (SpaceInvaders by default).

Usage::

    pip install -e ".[atari]"
    python examples/atari_ppo.py

Pass a different env id on the command line, e.g.::

    python examples/atari_ppo.py BreakoutDeterministic-v4

Notes:
    - This is a *minimal* recipe geared towards a quick smoke run rather
      than a paper-strength reproduction. Tune ``total_steps``, ``nsteps``
      and the network size for serious experiments.
    - Atari ROMs are bundled with Gymnasium when installed via the
      ``[atari]`` extra (which sets ``accept-rom-license``).
"""

from __future__ import annotations

import sys

import gymnasium as gym

from rlib.envs.vec_env import BatchEnv
from rlib.envs.wrappers import AtariEnv
from rlib.models import UniverseCNN
from rlib.PPO import PPO, PPOTrainer, PPOTrainerConfig
from rlib.PPO.model import PPOConfig


def main(env_id: str = "SpaceInvadersDeterministic-v4") -> None:
    num_envs = 8
    nsteps = 128

    # Probe a single env to discover input shape and action space.
    probe = AtariEnv(gym.make(env_id), k=4, episodic=False, reset=False, clip_reward=False)
    input_shape = probe.reset().shape
    action_size = probe.action_space.n
    probe.close()

    train_envs = BatchEnv(
        AtariEnv,
        env_id,
        num_envs=num_envs,
        blocking=False,
        k=4,
        episodic=True,
        reset=False,
        clip_reward=True,
    )
    val_envs = [
        AtariEnv(gym.make(env_id), k=4, episodic=False, reset=False, clip_reward=False)
        for _ in range(4)
    ]

    agent = PPO(
        UniverseCNN,
        input_shape=input_shape,
        action_size=action_size,
        config=PPOConfig(
            lr=2.5e-4,
            lr_final=0.0,
            decay_steps=int(1e7) // (num_envs * nsteps),
            grad_clip=0.5,
            entropy_coeff=0.01,
            policy_clip=0.1,
        ),
        value_coeff=0.5,
    )

    trainer = PPOTrainer(
        envs=train_envs,
        agent=agent,
        val_envs=val_envs,
        config=PPOTrainerConfig(
            total_steps=int(1e7),
            nsteps=nsteps,
            validate_freq=int(2e5),
            num_val_episodes=8,
            log_dir=f"logs/PPO/{env_id}",
            model_dir=f"models/PPO/{env_id}",
            num_epochs=4,
            num_minibatches=4,
        ),
    )
    trainer.train()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "SpaceInvadersDeterministic-v4")
