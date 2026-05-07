# Environments

`rlib` is built on top of [Gymnasium](https://gymnasium.farama.org/) — the
maintained successor to OpenAI Gym. To keep older user code working, the
library also tolerates the legacy `gym` package via the
[`rlib.utils.gym_compat`](https://github.com/jhare96/reinforcement-learning/blob/master/rlib/utils/gym_compat.py)
shim.

## Choosing a backend

```python
# Preferred: pull `gym` from the compat shim. It will resolve to gymnasium
# if installed, otherwise to the legacy gym package.
from rlib.utils.gym_compat import gym

env = gym.make("CartPole-v1")
```

The shim also provides `step_compat(env, action)` and `reset_compat(env)`
helpers which always return the legacy 4-tuple form
`(obs, reward, done, info)` — useful when integrating raw environments into
custom training loops. Internally, all built-in wrappers and vectorised
runners use these helpers so the rest of the library stays backend-agnostic.

## Vectorised environments

Two vectorised runners are provided:

- **`rlib.utils.VecEnv.BatchEnv`** — runs each environment in its own
  subprocess via `multiprocessing.Pipe`. Use this for environments where
  stepping is expensive (e.g. Atari).
- **`rlib.utils.VecEnv.DummyBatchEnv`** — runs all environments in the same
  process. Use this for cheap environments (e.g. classic control), where
  the overhead of multiprocessing dominates.

```python
from rlib.utils.VecEnv import BatchEnv, DummyBatchEnv
from rlib.utils.wrappers import AtariEnv

# Atari with 4-frame stacking and reward clipping, 16 parallel workers.
envs = BatchEnv(AtariEnv, "PongNoFrameskip-v4", num_envs=16, k=4)
```

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
