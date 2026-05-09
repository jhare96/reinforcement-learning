# Agents

`rlib` ships with the following agents. All agents (except A3C) subclass
[`SyncMultiEnvTrainer`](api/training.md) and therefore share a common
`train()` / validation interface, plus a `python -m rlib.<Agent>
<config>.yaml` CLI entry point.

| Agent | Module | Trainer class | Off-/on-policy | Best for |
|-------|--------|---------------|----------------|----------|
| Advantage Actor Critic | `rlib.A2C` | `A2C` | on-policy | Classic control, baseline Atari |
| A2C with LSTM | `rlib.A2C` | `A2CLSTMTrainer` | on-policy | Partially observable tasks |
| Asynchronous A3C | `rlib.A3C` | (custom) | on-policy | CPU-only multi-worker setups |
| Synchronous n-step Double DQN | `rlib.DDQN` | `SyncDDQN` | off-policy | Discrete-action Atari |
| Proximal Policy Optimisation | `rlib.PPO` | `PPOTrainer` | on-policy | Strong general-purpose baseline |
| Random Network Distillation | `rlib.RND` | `RNDTrainer` | on-policy + intrinsic | Hard-exploration / sparse rewards |
| Intrinsic Curiosity Module | `rlib.Curiosity` | `CuriosityTrainer` | on-policy + intrinsic | Sparse-reward exploration |
| UNREAL-A2C / A2C2 | `rlib.Unreal` | `UnrealTrainer` | on-policy + auxiliary | Sample-efficient pixel learning |
| Decoupled Advantage AC | `rlib.DAAC` | `DAACTrainer` | on-policy | Generalisation in procgen-style envs |
| Value Iteration Networks | `rlib.VIN` | `VINTrainer` | imitation / planning | Grid-world planning tasks |
| **RANDAL** | `rlib.RANDAL` | `RANDALTrainer` | RND + UNREAL combo | Hard-exploration with auxiliary tasks |

## Common training pattern

Via the YAML CLI (recommended):

```bash
python -m rlib.A2C examples/paper/configs/classic_a2c.yaml
```

Or from Python:

```python
import gymnasium as gym
from rlib.A2C import A2CTrainer, A2CConfig, ActorCritic
from rlib.envs import DummyBatchEnv
from rlib.models import MLP
from rlib.training import TrainerConfig

train_envs = DummyBatchEnv(lambda e: e, "CartPole-v1", num_envs=8)
val_envs = [gym.make("CartPole-v1") for _ in range(4)]

agent = ActorCritic(
    MLP,
    input_shape=train_envs.observation_space.shape,
    action_size=train_envs.action_space.n,
    config=A2CConfig(lr=7e-4, decay_steps=int(1e5), grad_clip=0.5),
)
A2CTrainer(
    envs=train_envs,
    agent=agent,
    val_envs=val_envs,
    config=TrainerConfig(total_steps=int(1e5), nsteps=5),
).train()
```

Full-fidelity reproductions of the experiments from
[arXiv:1910.09281](https://arxiv.org/abs/1910.09281) live under
[`examples/paper/`](https://github.com/jhare96/reinforcement-learning/tree/master/examples/paper).
For per-class API details see the [API reference](api/agent.md).
