# rlib

`rlib` is a small PyTorch-based reinforcement learning library. It provides
clean implementations of several on-policy and off-policy agents together
with a synchronous multi-environment trainer and a small zoo of Atari /
classic-control wrappers.

The library was originally developed for the MSc dissertation [*Dealing with
sparse rewards in reinforcement learning*](https://arxiv.org/abs/1910.09281),
which introduced the **RANDAL** agent included here.

## Quick links

- [Installation and quickstart](https://github.com/jhare96/reinforcement-learning#installation)
- [Agent reference](agents.md)
- [Environment integration](environments.md)
- [Wrapper reference](wrappers.md)
- [Changelog](../CHANGELOG.md)

## Design overview

```
┌───────────────────────────┐      ┌────────────────────────────────┐
│   gymnasium / gym envs    │◄────►│  rlib.envs adapters            │
└───────────────────────────┘      └────────────────────────────────┘
                                              │
                                              ▼
                                ┌─────────────────────────────────┐
                                │  rlib.utils.wrappers (Atari…)   │
                                └─────────────────────────────────┘
                                              │
                                              ▼
                                ┌─────────────────────────────────┐
                                │  rlib.utils.VecEnv              │
                                │  (BatchEnv / DummyBatchEnv)     │
                                └─────────────────────────────────┘
                                              │
                                              ▼
                                ┌─────────────────────────────────┐
                                │  rlib.utils.SyncMultiEnvTrainer │
                                │  (subclassed by every agent)    │
                                └─────────────────────────────────┘
                                              │
                                              ▼
                                ┌─────────────────────────────────┐
                                │  Agents: A2C / PPO / RND / …    │
                                └─────────────────────────────────┘
```

The compat shim transparently translates between Gymnasium's 5-tuple
`step()` API and the legacy Gym 4-tuple, so all higher layers work with the
classic `(obs, reward, done, info)` contract regardless of backend.
