# Environments

`rlib` is built directly on [Gymnasium](https://gymnasium.farama.org/) — the
maintained successor to OpenAI Gym. The canonical env contract
(`RLEnv` / `RLVecEnv` ABCs, `BatchEnv` / `DummyBatchEnv` runners,
wrappers, the `ApplePicker` exploration env) lives in
[`rlib.envs`](https://github.com/jhare96/reinforcement-learning/tree/master/rlib/envs).

## The 5-tuple contract

```python
import gymnasium as gym

env = gym.make("CartPole-v1")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

All wrappers and vec-env runners shipped with `rlib` consume this tuple
internally. The single boundary that collapses
`(terminated, truncated)` into the legacy `done` flag for agent rollouts
lives in `RLVecEnv.merge_done` / `merge_info`, so agents see a clean
`(obs, rewards, dones, infos)` API.

## Vectorised environments

Two runners are provided in `rlib.envs`:

- **`BatchEnv`** — each env runs in its own subprocess via
  `multiprocessing.Pipe`. Use this for expensive envs (e.g. Atari).
- **`DummyBatchEnv`** — all envs run in-process. Use this for cheap envs
  (e.g. classic control), where multiprocessing overhead dominates.

```python
from rlib.envs import BatchEnv, DummyBatchEnv
from rlib.envs.wrappers import AtariEnv

envs = BatchEnv(AtariEnv, "ALE/Pong-v5", num_envs=16, k=4)
```

The `rlib._cli` runner exposes two convenience factories,
`atari_envs(id, num_envs, num_val_envs, frame_stack, episodic, ...)` and
`classic_envs(id, num_envs, num_val_envs)`, which build the train + val
pair from a single declaration; YAML configs under
[`examples/paper/configs/`](https://github.com/jhare96/reinforcement-learning/tree/master/examples/paper/configs)
show the typical wiring.

## Built-in env: ApplePicker

The `ApplePicker-v0` / `ApplePickerDeterministic-v0` exploration grid-world
from the RANDAL paper is registered automatically when `rlib.envs` is
imported.

## Supported environment families

| Family | Install extra | Notes |
|--------|---------------|-------|
| Classic control | `pip install -e ".[classic]"` | CartPole, MountainCar, Acrobot, ... |
| Atari | `pip install -e ".[atari]"` | ROMs auto-licensed via `gymnasium[atari,accept-rom-license]` |
| MuJoCo | `pip install -e ".[mujoco]"` | Continuous control |

Other Gymnasium-compatible suites (e.g.
[MiniGrid](https://github.com/Farama-Foundation/Minigrid),
[Procgen](https://github.com/openai/procgen)) work as long as their
observation/action spaces are compatible with the chosen agent.
