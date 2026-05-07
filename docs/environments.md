# Environments

`rlib` is built on top of [Gymnasium](https://gymnasium.farama.org/) — the
maintained successor to OpenAI Gym. Backend-agnostic env adapters (Gymnasium,
legacy `gym`, and any user-registered backend) live in
[`rlib.envs`](https://github.com/jhare96/reinforcement-learning/tree/master/rlib/envs).

## Choosing a backend

```python
# Preferred: rlib.envs.make wraps an env id (or an existing env) into the
# canonical RLEnvBase contract.
from rlib.envs import make

env = make("CartPole-v1")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

For a raw Gymnasium env you can also just `import gymnasium as gym` and call
`gym.make(...)` directly — `rlib.envs.wrap(env)` will lift it into the
canonical contract on demand. Wrappers in `rlib.utils.wrappers` and the
vectorised runners in `rlib.utils.VecEnv` use this contract internally, so
the rest of the library is backend-agnostic.

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
