# Agents

`rlib` ships with the following agents. All agents (except A3C and VIN)
subclass [`SyncMultiEnvTrainer`](https://github.com/jhare96/reinforcement-learning/blob/master/rlib/utils/SyncMultiEnvTrainer.py)
and therefore share a common `train()` / `validate()` interface.

| Agent | Module | Trainer class | Off-/on-policy | Best for |
|-------|--------|---------------|----------------|----------|
| Advantage Actor Critic | `rlib.A2C` | `A2C` | on-policy | Classic control, baseline Atari |
| A2C with LSTM | `rlib.A2C` | `A2CLSTM_Trainer` | on-policy | Partially observable tasks |
| Asynchronous A3C | `rlib.A3C` | (custom) | on-policy | CPU-only multi-worker setups |
| Synchronous n-step Double DQN | `rlib.DDQN` | `SyncDDQN` | off-policy | Discrete-action Atari |
| Proximal Policy Optimisation | `rlib.PPO` | `PPOTrainer` | on-policy | Strong general-purpose baseline |
| Random Network Distillation | `rlib.RND` | `RNDTrainer` | on-policy + intrinsic | Hard-exploration / sparse rewards |
| Intrinsic Curiosity Module | `rlib.Curiosity` | `Curiosity_Trainer` | on-policy + intrinsic | Sparse-reward exploration |
| UNREAL-A2C / A2C2 | `rlib.Unreal` | `UnrealTrainer` | on-policy + auxiliary | Sample-efficient pixel learning |
| Decoupled Advantage AC | `rlib.DAAC` | `DAACTrainer` | on-policy | Generalisation in procgen-style envs |
| Value Iteration Networks | `rlib.VIN` | `VINTrainer` | imitation / planning | Grid-world planning tasks |
| **RANDAL** | `rlib.RANDAL` | `RANDALTrainer` | RND + UNREAL combo | Hard-exploration with auxiliary tasks |

## Common training pattern

```python
from rlib.A2C import A2C, ActorCritic
from rlib.utils.VecEnv import DummyBatchEnv
from rlib.utils.gym_compat import gym

train_envs = DummyBatchEnv(lambda e: e, "CartPole-v1", num_envs=8)
val_envs   = [gym.make("CartPole-v1") for _ in range(4)]

model   = ActorCritic(input_size=4, num_actions=2, lr=7e-4)
trainer = A2C(envs=train_envs, model=model, val_envs=val_envs,
              total_steps=int(1e5))
trainer.train()
```

See [`examples/`](https://github.com/jhare96/reinforcement-learning/tree/master/examples)
for runnable variants (CartPole-A2C, Atari-PPO, Montezuma-RND).
