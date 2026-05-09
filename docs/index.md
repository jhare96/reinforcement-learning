---8<--- "README.md"


## Design overview

```
┌───────────────────────────┐
│   gymnasium envs          │
└───────────────────────────┘
              │  (obs, reward, terminated, truncated, info)
              ▼
┌───────────────────────────────────┐
│  rlib.envs.wrappers (Atari, …)    │   subclass RLEnv
└───────────────────────────────────┘
              │
              ▼
┌───────────────────────────────────┐
│  rlib.envs.BatchEnv /             │   subclass RLVecEnv
│  rlib.envs.DummyBatchEnv          │   merge_done / merge_info
└───────────────────────────────────┘
              │  (obs, rewards, dones, infos)
              ▼
┌───────────────────────────────────┐
│  rlib.training.SyncMultiEnvTrainer│   subclassed by every agent
└───────────────────────────────────┘
              │
              ▼
┌───────────────────────────────────┐
│  Agents: A2C / PPO / RND / …      │   each is a `rlib.agent.Agent`
└───────────────────────────────────┘
```

The Gymnasium 5-tuple is the canonical contract throughout `rlib.envs`.
The single boundary that collapses `(terminated, truncated)` into the
legacy `done` flag for agent rollouts lives in `RLVecEnv.merge_done` /
`merge_info`, so agent code sees a clean
`(obs, rewards, dones, infos)` API.
