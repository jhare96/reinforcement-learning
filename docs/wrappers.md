# Wrappers

`rlib.envs.wrappers` provides a small set of environment wrappers,
inspired by (and originally adapted from) the OpenAI Baselines wrappers.

All wrappers subclass
[`rlib.envs.RLEnv`](api/envs.md)
and use the **modern 5-tuple step API** —
`(obs, reward, terminated, truncated, info)` — together with
`reset(*, seed=None, options=None) -> (obs, info)`.

The 5→4-tuple collapse for agent rollout code happens at a single
boundary inside the vectorised env runners (`RLVecEnv.merge_done` /
`merge_info`), so existing
`(obs, rewards, dones, infos) = env.step(actions)` agent code keeps
working unchanged.

## Reference

| Wrapper | Purpose |
|---------|---------|
| `RescaleEnv(env, size)` | Resize and grey-scale frames to `size × size`. |
| `AtariRescale42x42(env)` / `AtariRescaleEnv(env)` | Atari-specific 42×42 / 84×84 grey-scale preprocessing. |
| `AtariRescaleColour(env)` | Atari 84×84 colour preprocessing. |
| `NoopResetEnv(env, max_op)` | Take a random number of no-op actions on reset. |
| `FireResetEnv(env)` | Press FIRE on reset (for envs that need it to start). |
| `EpisodicLifeEnv(env)` | Terminate the episode whenever a life is lost. |
| `ClipRewardEnv(env)` | Clip rewards to `[-1, +1]`. |
| `NoRewardEnv(env)` | Zero out rewards (useful for pure exploration). |
| `TimeLimitEnv(env, time_limit)` | Hard episode-length cap. |
| `StackEnv(env, k)` | Stack the last `k` frames along the channel axis. |
| `AutoResetEnv(env)` | Auto-reset on `done` so the worker never blocks. |
| `ChannelsFirstEnv(env)` | Transpose `(H, W, C) → (C, H, W)` for PyTorch. |
| `GreyScaleEnv(env)` | RGB → single-channel luminance. |
| `ToTorchEnv(env, device)` | Convert obs/reward/done to `torch.Tensor` on a given device. |

## Convenience factories

- **`AtariEnv(env, k=4, rescale=84, episodic=True, ...)`** — composes the
  most common Atari pipeline (FIRE-reset, no-op reset, reward clipping,
  episodic-life, rescaling, frame stacking, time-limit, auto-reset,
  channels-first).
- **`apple_pickgame(env, k=1, grey_scale=False, auto_reset=False, ...)`** —
  preprocessing pipeline used in the RANDAL paper for the
  `ApplePicker` exploration environments.
