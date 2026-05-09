# Paper reproduction recipes

Training recipes for the experiments in [Hare 2019, _Dealing with Sparse
Rewards in Reinforcement Learning_ (arXiv:1910.09281)](https://arxiv.org/abs/1910.09281).

Two equivalent ways to launch each run:

|                | YAML config (`configs/`)                                 | Python script (`scripts/`)                          |
|----------------|----------------------------------------------------------|-----------------------------------------------------|
| **Use case**   | Quick smoke runs, share/diff configs, sweep via `--set`  | Need custom code (custom optimisers, callbacks)     |
| **Override**   | `--set trainer.config.total_steps=5_000_000`             | edit the file or wrap `main()`                      |

Both paths produce identical training. Pick whichever fits your workflow.

## Layout

```
examples/paper/
├── configs/                       ← 11 YAML configs (one per script)
│   ├── atari_a2c.yaml
│   ├── atari_ddqn.yaml
│   ├── atari_ppo.yaml
│   ├── atari_randal.yaml
│   ├── atari_rnd.yaml
│   ├── atari_unreal.yaml
│   ├── classic_a2c.yaml
│   ├── classic_ddqn.yaml
│   ├── classic_ppo.yaml
│   ├── classic_randal.yaml
│   └── classic_rnd.yaml
└── scripts/                       ← matching Python entry points
    ├── common.py                  ← shared env factories + body networks
    ├── atari_a2c.py
    ├── ...
    └── classic_rnd.py
```

## Experiment grid

|                     | Acrobot | CartPole | MountainCar | Freeway | Montezuma | SpaceInvaders |
|---------------------|:-------:|:--------:|:-----------:|:-------:|:---------:|:-------------:|
| **A2C** baseline    |   ✓     |    ✓     |     ✓       |   ✓     |     ✓     |       ✓       |
| **DDQN** baseline   |   ✓     |    ✓     |     ✓       |   ✓     |     ✓     |       ✓       |
| **PPO**             |   ✓     |    ✓     |     ✓       |   ✓     |     ✓     |       ✓       |
| **RND**             |   ✓     |    ✓     |     ✓       |   ✓     |     ✓     |       ✓       |
| **UNREAL-A2C2**     |   —     |    —     |     —       |   ✓     |     ✓     |       ✓       |
| **RANDAL** (novel)  |   ✓     |    ✓     |     ✓       |   ✓     |     ✓     |       ✓       |

The classic-control UNREAL row is intentionally skipped — the paper notes the
pixel-control auxiliary task doesn't apply to low-dimensional inputs.

## Running via YAML config

The configs ship with the canonical paper hyperparameters baked in. Use
`--set key.path=value` to override individual fields without editing the file.

```bash
# RANDAL on Montezuma — paper defaults (50M steps, 32 actors)
python -m rlib.RANDAL examples/paper/configs/atari_randal.yaml

# PPO on Breakout instead of SpaceInvaders
python -m rlib.PPO examples/paper/configs/atari_ppo.yaml --set env.id=ALE/Breakout-v5

# Quick smoke: 100k steps, 4 envs, CPU
python -m rlib.A2C examples/paper/configs/classic_a2c.yaml \
  --set trainer.config.total_steps=100_000 \
  --set env.num_envs=4 \
  --set agent.config.device=cpu

# Sweep all 3 paper Atari envs for one agent
for env in ALE/Freeway-v5 ALE/MontezumaRevenge-v5 ALE/SpaceInvaders-v5; do
  python -m rlib.RND examples/paper/configs/atari_rnd.yaml --set env.id=$env
done
```

## Running via Python script

The scripts accept the env id as the first CLI argument and support `all`
to run sequentially across the agent's whole category:

```bash
# Single env
python examples/paper/scripts/atari_rnd.py MontezumaRevengeDeterministic-v4

# All Atari envs for a given agent (sequential)
python examples/paper/scripts/atari_rnd.py all

# Single classic-control env
python examples/paper/scripts/classic_ppo.py MountainCar-v0
```

To customise hyperparameters, edit the constants at the top of the script
(e.g. `TOTAL_STEPS`) or the kwargs in `main()`.

## Compute notes

The Atari runs use the paper's `total_steps = 50,000,000` and 32 parallel
actors — this is multi-GPU-day training. For a quick smoke test, override
`trainer.config.total_steps` (and shrink `validate_freq` proportionally) on
the YAML path, or edit the constant at the top of the script.

Classic-control runs are much cheaper (`total_steps = 2,000,000` for the
sparse-reward / PPO methods, `5,000,000` for A2C); they fit on a single GPU
in tens of minutes.

## TensorBoard

```bash
tensorboard --logdir logs/paper/
```
