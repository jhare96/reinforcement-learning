# rlib — a small PyTorch reinforcement learning library

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
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

`rlib` targets **Python 3.10+**, **PyTorch 1.13+** and
[**Gymnasium**](https://gymnasium.farama.org/) (the maintained successor to
OpenAI Gym). The :mod:`rlib.envs` package provides a backend-agnostic env
abstraction (`RLEnv` Protocol, `RLEnvBase` ABC, `make`/`wrap`/`register_backend`)
so the library also works against legacy `gym` and is easy to extend to other
gym-like backends.

A `Dockerfile` is provided for fully-reproducible setups (see below).

## Quickstart

Train an A2C agent on CartPole-v1 in ~40 lines (see [`examples/cartpole_a2c.py`](examples/cartpole_a2c.py)
for the runnable version):

```python
import torch
from rlib.A2C import A2C, ActorCritic
from rlib.utils.VecEnv import DummyBatchEnv
import gymnasium as gym


class MLP(torch.nn.Module):
    """A tiny MLP body for low-dimensional state spaces."""
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        in_dim = int(input_size[0]) if hasattr(input_size, "__len__") else int(input_size)
        self.dense_size = hidden_size
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_size), torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, hidden_size), torch.nn.Tanh(),
        )
    def forward(self, x): return self.net(x)


env_id, num_envs = "CartPole-v1", 8
train_envs = DummyBatchEnv(lambda e: e, env_id, num_envs=num_envs)
val_envs   = [gym.make(env_id) for _ in range(4)]

model = ActorCritic(
    MLP,
    input_size=train_envs.envs[0].observation_space.shape,
    action_size=train_envs.envs[0].action_space.n,
    lr=7e-4, decay_steps=int(1e5), grad_clip=0.5,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

A2C(envs=train_envs, model=model, val_envs=val_envs,
    total_steps=int(1e5), nsteps=5, validate_freq=int(2e4),
    log_dir="logs/A2C/CartPole", model_dir="models/A2C/CartPole",
).train()
```

Inspect training curves with TensorBoard:

```bash
tensorboard --logdir logs/
```

More runnable examples — including Atari PPO and Montezuma's Revenge with
RND — live under [`examples/`](examples/).

## Repository layout

```
rlib/
├── A2C/        # A2C and A2C-LSTM
├── A3C/        # Asynchronous A3C
├── PPO/        # PPO
├── DDQN/       # Synchronous n-step Double DQN
├── RND/        # Random Network Distillation
├── RANDAL/     # RANDAL (RND + UNREAL)
├── Curiosity/  # ICM-based curiosity agent
├── Unreal/     # UNREAL-A2C / A2C2
├── DAAC/       # Decoupled Advantage Actor-Critic
├── VIN/        # Value Iteration Networks
├── networks/   # Reusable CNN / MLP / masked-RNN building blocks
└── utils/      # VecEnv, wrappers, replay memory, schedulers, gym compat
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
[`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines.

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

Released under the [Apache License 2.0](LICENSE). See [`NOTICE`](NOTICE) for
attribution of code adapted from third parties (notably OpenAI Baselines).
