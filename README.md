# rlib — a small PyTorch reinforcement learning library

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://github.com/jhare96/reinforcement-learning/blob/master/pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-EE4C2C.svg)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29%2B-007ACC.svg)](https://gymnasium.farama.org/)

`rlib` is a small PyTorch-based reinforcement learning library, originally
developed for the MSc dissertation [*Dealing with sparse rewards in
reinforcement learning*](https://arxiv.org/abs/1910.09281).

It implements a number of well-known on-policy and off-policy agents in a
consistent API, and pairs them with a synchronous multi-environment trainer
and a small zoo of Atari/classic-control environment wrappers.

## Implemented agents

| Agent | Module | Reference |
|-------|--------|-----------|
| Advantage Actor Critic (A2C) | `rlib.A2C` | <https://openai.com/blog/baselines-acktr-a2c/> |
| Asynchronous A3C | `rlib.A3C` | <https://arxiv.org/abs/1602.01783> |
| Synchronous n-step Double DQN | `rlib.DDQN` | <https://arxiv.org/abs/1509.06461> |
| Proximal Policy Optimisation (PPO) | `rlib.PPO` | <https://arxiv.org/abs/1707.06347> |
| Random Network Distillation (RND) | `rlib.RND` | <https://arxiv.org/abs/1810.12894> |
| Intrinsic Curiosity Module (ICM) | `rlib.Curiosity` | <https://arxiv.org/abs/1705.05363> |
| UNREAL-A2C / A2C2 | `rlib.Unreal` | <https://deepmind.com/blog/article/reinforcement-learning-unsupervised-auxiliary-tasks> |
| Decoupled Advantage Actor-Critic (DAAC) | `rlib.DAAC` | <https://arxiv.org/abs/2102.10330> |
| Value Iteration Networks (VIN) | `rlib.VIN` | <https://arxiv.org/abs/1602.02867> |
| **RANDAL** (RND + UNREAL combination, novel) | `rlib.RANDAL` | <https://arxiv.org/abs/1910.09281> |

## Installation

```bash
git clone https://github.com/jhare96/reinforcement-learning.git
cd reinforcement-learning
pip install -e .

# Optional extras for specific environment families:
pip install -e ".[classic]"   # Classic control envs (CartPole, MountainCar, ...)
pip install -e ".[atari]"     # Atari with ROM auto-license
pip install -e ".[mujoco]"    # MuJoCo continuous-control envs
pip install -e ".[docs]"      # Build the local documentation
```

`rlib` targets **Python 3.11+**, **PyTorch 1.13+** and
[**Gymnasium**](https://gymnasium.farama.org/) 0.29+. The `rlib.envs`
package provides the canonical 5-tuple env contract (`RLEnv` ABC,
`RLVecEnv` ABC, `BatchEnv` / `DummyBatchEnv` runners, `AtariEnv` /
classic-control wrappers).

A `Dockerfile` is provided for fully-reproducible setups (see below).

## Quickstart

The fastest way to train an agent is the YAML CLI — every agent module is
runnable as `python -m rlib.<Agent> path/to/config.yaml`:

```bash
python -m rlib.A2C  examples/paper/configs/classic_a2c.yaml
python -m rlib.PPO  examples/paper/configs/atari_ppo.yaml
python -m rlib.RND  examples/paper/configs/atari_rnd.yaml
```

Override any field on the command line:

```bash
python -m rlib.A2C examples/paper/configs/classic_a2c.yaml \
    --set env.id=Acrobot-v1 \
    --set trainer.config.total_steps=1_000_000 \
    --set agent.config.lr=3e-4
```

Or drive everything from Python (see
[`examples/cartpole_a2c.py`](https://github.com/jhare96/reinforcement-learning/blob/master/examples/cartpole_a2c.py)
for the runnable version):

```python
import torch
import gymnasium as gym

from rlib.A2C import A2C, A2CConfig, ActorCritic
from rlib.envs import DummyBatchEnv
from rlib.models import MLP
from rlib.training import TrainerConfig

env_id, num_envs = "CartPole-v1", 8
train_envs = DummyBatchEnv(lambda e: e, env_id, num_envs=num_envs)
val_envs = [gym.make(env_id) for _ in range(4)]
device = "cuda" if torch.cuda.is_available() else "cpu"

agent = ActorCritic(
    MLP,
    input_shape=train_envs.observation_space.shape,
    action_size=train_envs.action_space.n,
    config=A2CConfig(lr=7e-4, decay_steps=int(1e5), grad_clip=0.5, device=device),
)

A2C(
    envs=train_envs,
    agent=agent,
    val_envs=val_envs,
    config=TrainerConfig(
        total_steps=int(1e5),
        nsteps=5,
        validate_freq=int(2e4),
        log_dir="logs/A2C/CartPole",
        model_dir="models/A2C/CartPole",
    ),
).train()
```

Inspect training curves with TensorBoard:

```bash
tensorboard --logdir logs/
```

More runnable examples — including Atari PPO and Montezuma's Revenge with
RND — live under [`examples/`](examples/).

For full reproductions of the experiments from the
[*Dealing with sparse rewards*](https://arxiv.org/abs/1910.09281) paper see
[`examples/paper/`](examples/paper/), which has one script per (agent, env class)
pair with the paper's hyperparameters baked in.

## Repository layout

```
rlib/
├── agent.py     # Agent base class + ModelConfig
├── models.py    # NatureCNN, MLP, MaskedLSTMBlock, …
├── _cli.py      # Hydra-style YAML runner used by `python -m rlib.<Agent>`
├── A2C/         # A2C and A2C-LSTM
├── A3C/         # Asynchronous A3C
├── PPO/         # PPO
├── DDQN/        # Synchronous n-step Double DQN
├── RND/         # Random Network Distillation
├── RANDAL/      # RANDAL (RND + UNREAL)
├── Curiosity/   # ICM-based curiosity agent
├── Unreal/      # UNREAL feedforward + LSTM
├── DAAC/        # Decoupled Advantage Actor-Critic
├── VIN/         # Value Iteration Networks
├── envs/        # RLEnv / RLVecEnv ABCs, BatchEnv, wrappers, ApplePicker
├── training/    # SyncMultiEnvTrainer, TrainerConfig, Returns, Validator
└── utils/       # ReplayMemory, schedulers, play, helpers
```

## Documentation

A static documentation site can be built locally with MkDocs:

```bash
pip install -e ".[docs]"
mkdocs serve
```

The Markdown sources live under [`docs/`](docs/).

## Contributing

Bug reports, feature requests and pull requests are very welcome — please see
[`CONTRIBUTING.md`](https://github.com/jhare96/reinforcement-learning/blob/master/CONTRIBUTING.md) for guidelines.

## Citation

If you use `rlib` in academic work, please cite the original RANDAL paper:

```bibtex
@article{hare2019dealing,
  title   = {Dealing with sparse rewards in reinforcement learning},
  author  = {Hare, Joshua},
  journal = {arXiv preprint arXiv:1910.09281},
  year    = {2019}
}
```

To cite this repository directly:

```bibtex
@misc{Hare_rlib,
  author       = {Joshua Hare},
  title        = {rlib: a PyTorch reinforcement learning library},
  year         = {2019--2026},
  version      = {3.0.0},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/jhare96/reinforcement-learning}}
}
```

## License

Released under the [Apache License 2.0](https://github.com/jhare96/reinforcement-learning/blob/master/LICENSE). See [`NOTICE`](https://github.com/jhare96/reinforcement-learning/blob/master/NOTICE) for
attribution of code adapted from third parties (notably OpenAI Baselines).
