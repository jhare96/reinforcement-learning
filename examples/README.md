# Examples

This directory contains runnable training scripts that exercise the agents
shipped with `rlib`. Each script is intentionally small and self-contained.

| Script | Agent | Environment | Install extras |
|--------|-------|-------------|----------------|
| [`cartpole_a2c.py`](cartpole_a2c.py) | A2C | `CartPole-v1` | `[classic]` |
| [`atari_ppo.py`](atari_ppo.py) | PPO | `SpaceInvadersDeterministic-v4` (override on CLI) | `[atari]` |
| [`montezuma_rnd.py`](montezuma_rnd.py) | RND | `MontezumaRevengeDeterministic-v4` | `[atari]` |

After running, point TensorBoard at the `logs/` directory to inspect
training curves:

```bash
tensorboard --logdir logs/
```
