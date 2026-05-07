# Paper reproduction recipes

Training scripts for the experiments in [Hare 2019, _Dealing with Sparse Rewards
in Reinforcement Learning_ (arXiv:1910.09281)](https://arxiv.org/abs/1910.09281).

Each script has the canonical hyperparameters from the paper baked in (cited
back to the relevant table or section in the docstring). All logs land under
`logs/paper/<Agent>/<env_id>/`.

## Experiment grid

|                     | Acrobot | CartPole | MountainCar | Freeway | Montezuma | SpaceInvaders |
|---------------------|:-------:|:--------:|:-----------:|:-------:|:---------:|:-------------:|
| **A2C** baseline    |  ✓ classic_a2c     |  ✓                |  ✓               |  ✓ atari_a2c       |  ✓                  |  ✓                       |
| **DDQN** baseline   |  ✓ classic_ddqn    |  ✓                |  ✓               |  ✓ atari_ddqn      |  ✓                  |  ✓                       |
| **PPO**             |  ✓ classic_ppo     |  ✓                |  ✓               |  ✓ atari_ppo       |  ✓                  |  ✓                       |
| **RND**             |  ✓ classic_rnd     |  ✓                |  ✓               |  ✓ atari_rnd       |  ✓                  |  ✓                       |
| **UNREAL-A2C2**     |  —                 |  —                |  —               |  ✓ atari_unreal    |  ✓                  |  ✓                       |
| **RANDAL** (novel)  |  ✓ classic_randal  |  ✓                |  ✓               |  ✓ atari_randal    |  ✓                  |  ✓                       |

The classic-control UNREAL row is intentionally skipped — the paper notes the
pixel-control auxiliary task doesn't apply to low-dimensional inputs.

## Running

Each script accepts the env id as the first CLI argument; pass `all` to run
the agent across every env in its category sequentially.

```bash
# Single env
python examples/paper/atari_rnd.py MontezumaRevengeDeterministic-v4

# All Atari envs for a given agent (sequential)
python examples/paper/atari_rnd.py all

# Single classic-control env
python examples/paper/classic_ppo.py MountainCar-v0
```

## Compute notes

The Atari runs use the paper's `total_steps = 50,000,000` and 32 parallel
actors — this is multi-GPU-day training. For a quick smoke test, edit
`TOTAL_STEPS` at the top of the relevant script (and shrink `validate_freq`
proportionally) before launching.

Classic-control runs are much cheaper (`total_steps = 2,000,000` for the
sparse-reward / PPO methods, `5,000,000` for A2C); they fit on a single GPU
in tens of minutes.

## TensorBoard

```bash
tensorboard --logdir logs/paper/
```
