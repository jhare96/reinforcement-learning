"""RND with PPO Atari (Table: RND Atari) — Hare 2019, Sec. 3.3.

Hyperparameters reproduce :ref:`tbl:RND Atari`::

    Optimiser              = Adam
    Learning rate          = 1e-4
    Number of actors       = 32
    Entropy coefficient    = 0.001
    Value coefficient      = 0.5
    n-step period          = 128
    Number of epochs       = 4
    Number of minibatches  = 4
    Extrinsic adv. coeff.  = 2.0
    Intrinsic adv. coeff.  = 1.0
    Extrinsic γ_e          = 0.999
    Intrinsic γ_i          = 0.99
    PPO clip range         = [0.9, 1.1]
    Gradient norm clip     = 0.5
    Initial obs steps      = 6400

Run::

    python examples/paper/atari_rnd.py MontezumaRevengeDeterministic-v4
"""

from __future__ import annotations

import sys

from examples.paper.common import ATARI_ENVS, ATARI_VAL_STEPS, NatureCNN, atari_envs, get_device
from rlib.PPO.model import PPOConfig
from rlib.RND import RND, PredictorCNN, RNDTrainer, RNDTrainerConfig
from rlib.training import Returns

NSTEPS = 128
NUM_WORKERS = 32
TOTAL_STEPS = 50_000_000


def main(env_id: str = "MontezumaRevengeDeterministic-v4") -> None:
    device = get_device()
    train_envs, val_envs = atari_envs(env_id, num_envs=NUM_WORKERS)

    input_shape = train_envs.envs[0].reset()[0].shape
    action_size = train_envs.envs[0].action_space.n

    agent = RND(
        policy_model=NatureCNN,
        target_model=PredictorCNN,
        input_size=input_shape,
        action_size=action_size,
        config=PPOConfig(
            lr=1e-4,
            lr_final=1e-4,
            decay_steps=TOTAL_STEPS // (NUM_WORKERS * NSTEPS),
            grad_clip=0.5,
            entropy_coeff=0.001,
            policy_clip=0.1,
            device=device,
        ),
        intr_coeff=1.0,
        extr_coeff=2.0,
    ).to(device)

    trainer = RNDTrainer(
        envs=train_envs,
        agent=agent,
        val_envs=val_envs,
        config=RNDTrainerConfig(
            total_steps=TOTAL_STEPS,
            nsteps=NSTEPS,
            gamma=0.999,  # extrinsic discount γ_e
            gamma_intr=0.99,  # intrinsic discount γ_i
            lambda_=0.95,
            returns=Returns.GAE,
            init_obs_steps=6400,
            num_epochs=4,
            num_minibatches=4,
            validate_freq=1_000_000,
            num_val_episodes=8,
            max_val_steps=ATARI_VAL_STEPS,
            log_dir=f"logs/paper/RND/{env_id}",
            model_dir=f"models/paper/RND/{env_id}",
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
