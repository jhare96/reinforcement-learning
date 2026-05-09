---8<--- "README.md"


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
