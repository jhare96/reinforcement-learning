# Changelog

All notable changes to **rlib** are documented in this file. The format is
loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/).

## [3.0.0] - Unreleased

A large modernisation of the library covering packaging, the env layer, the
agent / trainer split, configuration, the CLI, docs and CI. **It contains
breaking changes for users still on legacy `gym` and for anything that
imported from `rlib.networks` or `rlib.utils.SyncMultiEnvTrainer`.**

### Added

#### Packaging, licensing and tooling

- **Apache 2.0 license** (`LICENSE` + `NOTICE` recording attribution to
  OpenAI Baselines), **PEP 621 packaging** in `pyproject.toml` with
  optional extras `[atari]`, `[classic]`, `[mujoco]`, `[docs]`, `[dev]`, a
  **`Dockerfile`** for reproducible CPU runtimes, a **`requirements.txt`**
  for `pip install -r` workflows, and a **PEP 561 `py.typed`** marker.
- **GitHub Actions CI** (`.github/workflows/ci.yml`): `ruff` lint +
  format-check, `mypy` over `rlib/envs` and `rlib/utils/schedulers.py`,
  `pytest` matrix on Python 3.11 + 3.12, `python -m build` + `twine check`
  artefact upload.
- **Docs CI** (`.github/workflows/docs.yml`): `mkdocs build --strict`
  artefact + `actions/deploy-pages@v4` deploy on push to `master` or `v3`.
- **Makefile** mirroring the CI workflow (`make install / lint / format /
  format-check / typecheck / test / build / docs / ci / clean`) so the
  same commands run locally and in CI. `TYPECHECK_PATHS` is shared.
- **`pre-commit` configuration** running `ruff` + `ruff-format`.

#### Environment layer

- **`rlib.envs` package** — single canonical, backend-agnostic env layer:
  - `RLEnv` — abstract base class for the modern 5-tuple `(obs, reward,
    terminated, truncated, info)` step / `(obs, info)` reset contract,
    with helpful defaults (`unwrapped`, `__getattr__` forwarding,
    context-manager support, `render`, `close`, `spec`).
  - `RLVecEnv` — abstract base for vectorised env runners. Static helpers
    `merge_done(terminated, truncated)` and `merge_info(info, terminated,
    truncated)` are the **single canonical place** in the codebase that
    collapses the 5-tuple into the legacy `done` flag.
  - `make(env_or_id, **kwargs)` — re-exports `gymnasium.make` so import
    sites do not need to touch `gymnasium` directly.
- **`rlib.envs.vec_env`** — `BatchEnv` (multiprocessing), `DummyBatchEnv`
  (in-process) and `ChunkEnv` (sharded multiprocessing), all subclassing
  `RLVecEnv`.
- **`rlib.envs.wrappers`** — Atari + classic preprocessing wrappers
  (`AtariEnv`, `RescaleEnv`, `NoopResetEnv`, `FireResetEnv`,
  `EpisodicLifeEnv`, `ClipRewardEnv`, `StackEnv`, `AutoResetEnv`,
  `ChannelsFirstEnv`, `GreyScaleEnv`, `ToTorchEnv`, …) all subclassing
  `RLEnv`.
- **`rlib.envs.apple_picker`** — the `ApplePicker-v0` /
  `ApplePickerDeterministic-v0` exploration envs from the RANDAL paper,
  ported to the Gymnasium 5-tuple API and auto-registered at import
  time.

#### Training infrastructure

- **`rlib.training` package** — promoted from `rlib.utils` so the trainer,
  configs and validation code are first-class:
  - `SyncMultiEnvTrainer` — synchronous multi-env trainer.
  - `TrainerConfig` + per-agent subclasses (`PPOTrainerConfig`,
    `RNDTrainerConfig`, `RANDALTrainerConfig`, `DDQNTrainerConfig`,
    `DAACTrainerConfig`, `UnrealTrainerConfig`) co-located with their
    trainer modules.
  - `TrainMode` (`StrEnum`: `NSTEP` / `ONESTEP`).
  - `Returns` — enum-of-functions wrapping `nstep_return`, `lambda_return`
    and `GAE`. Replaces all the previous `return_type="GAE"` string
    arguments and centralises the maths in `rlib.training.returns`.
  - `Validator` / `make_validator(val_envs)` — pluggable thread / vec /
    in-process validation runner used by every trainer.
  - **`tqdm` progress bars** on every training loop (base trainer + every
    subclass), with live `score / loss / fps` postfix metrics.
    `validation_summary` and saved-model logs use `tqdm.write`.
  - **Auto-logged hyperparameters**: `dataclasses.asdict` of the trainer
    + agent configs is written to `<log_dir>/hyperparameters.txt` at the
    start of every run.

#### Agent layer

- **`rlib.agent.Agent`** + **`rlib.agent.ModelConfig`** — single base
  class for every agent, with the dataclass-config pattern (frozen
  configs co-located in each agent's `model.py`).
- Every agent split into `model.py` (network + config) and `trainer.py`
  (training loop + per-trainer config). Each subpackage exports a
  curated `__all__`.
- **`rlib.models`** — promoted from `rlib.networks` and now houses every
  reusable building block (`NatureCNN`, `MaskedLSTMBlock`, MLP heads,
  …). `rlib.networks` is gone.

#### CLI and YAML configs

- **`rlib._cli`** — Hydra-style YAML runner usable as
  `python -m rlib.<Agent> path/to/config.yaml [--set key=value ...]`.
  Supports `constructor: dotted.path` instantiation, `partial: true` for
  `functools.partial`, `${name}` interpolation against runtime namespace
  values (`device`, `agent`, env-factory bundle), and helpers
  `atari_envs(...)` / `classic_envs(...)` / `clone_module(...)` for
  building the train + val env pair from a single declaration.
  Auto-coerces `returns: "GAE"` and `train_mode: "nstep"` strings to the
  matching enum members.
- **Per-agent `__main__`** shims (`python -m rlib.A2C config.yaml`, etc.)
  for every agent.

#### Examples and docs

- **`examples/`** — runnable scripts (`cartpole_a2c.py`, `atari_ppo.py`,
  `montezuma_rnd.py`).
- **`examples/paper/`** — full reproductions of the experiments from
  [arXiv:1910.09281](https://arxiv.org/abs/1910.09281):
  - `examples/paper/scripts/{atari,classic}_*.py` — 11 Python recipe
    scripts, one per (agent, env class) pair.
  - `examples/paper/configs/*.yaml` — 11 matching YAML configs that
    reproduce each script's hyperparameters via the `rlib._cli` runner.
- **MkDocs site** with `mkdocs-material`, including:
  - Hand-written overviews (`index.md` includes `README.md` via
    `pymdownx.snippets`, plus `agents.md`, `environments.md`,
    `wrappers.md`).
  - **API reference** auto-generated by `mkdocstrings[python]` covering
    every public agent package, the `Agent` base, `rlib.training`,
    `rlib.envs`, `rlib.models` and `rlib.utils`.
  - Deployed to <https://jhare96.github.io/reinforcement-learning/>.

### Changed

- **Python 3.11+ required** (was 3.8+). Driven by `enum.member` (used by
  the `Returns` enum) being a 3.11 feature. New code uses PEP 604 `X | Y`
  unions and modern built-in generic aliases.
- **PyTorch 1.13+**, **Gymnasium 0.29+**, **PyYAML ≥ 6**, **tqdm ≥ 4.60**.
- **Trainers consume agents through `self.agent`** (was `self.model`).
  Every trainer subclass has a class-level `agent: <ConcreteSubclass>`
  annotation so type-checkers narrow correctly.
- **`SyncMultiEnvTrainer` is now config-only** — instead of accepting
  every hyperparameter as a kwarg, you pass a single `TrainerConfig`
  (or per-agent subclass).
- **Agents are now config-only** — `ModelConfig` (or per-agent subclass)
  carries `lr`, `decay_steps`, `grad_clip`, `entropy`, `value`, … so the
  network constructor signature stays small.
- **Vec envs cleaned up**: `BatchEnv` / `DummyBatchEnv` / `ChunkEnv`
  consume the wrappers' 5-tuple internally and produce the legacy
  4-tuple at exactly one boundary, so agent rollout code is unchanged.
- **README** rewritten with installation, quickstart (using YAML CLI),
  agent overview, citation and contribution sections; the docs home
  page now includes `README.md` directly.

### Removed

- **Legacy `gym` support**. Anywhere that previously imported `gym`
  must `import gymnasium as gym` (or use `rlib.envs.make`). The
  `LegacyGymAdapter` and `register_backend` extension points have been
  removed; all envs now flow through the canonical Gymnasium API.
- **`rlib.utils.gym_compat`** (already deprecated in v3 development) —
  use `rlib.envs.make` / `rlib.envs.wrap` directly.
- **`rlib.networks` package** — replaced by `rlib.agent` (base class +
  `ModelConfig`) plus `rlib.models` (network building blocks).
- **`rlib.utils.SyncMultiEnvTrainer`** — moved to `rlib.training`.
- **Demo `def main()` blocks** previously embedded in each agent module
  — replaced by the YAML CLI + `examples/`.

### Fixed

- Several latent bugs in trainers, models and replay memory found while
  the library was being refactored (`fastsample().item()` on 0-d
  tensors, RND / RANDAL observation-shape mismatches, a `nsteps` typo
  in the n-step DDQN target, `get_value` shape, …).
- mypy clean across `rlib/envs` and `rlib/utils/schedulers.py`; ruff +
  ruff-format clean across the whole repository.
- `mkdocs --strict` passes (the changelog and examples links are now
  absolute URLs).

### Migration notes

```diff
-from rlib.utils.SyncMultiEnvTrainer import SyncMultiEnvTrainer
+from rlib.training import SyncMultiEnvTrainer, TrainerConfig

-from rlib.utils.VecEnv import BatchEnv, DummyBatchEnv
+from rlib.envs import BatchEnv, DummyBatchEnv

-from rlib.utils.wrappers import AtariEnv
+from rlib.envs.wrappers import AtariEnv

-from rlib.networks.networks import NatureCNN
+from rlib.models import NatureCNN

-from rlib.utils.gym_compat import gym
+import gymnasium as gym
```

`Trainer(envs, model=..., total_steps=..., nsteps=..., ...)` becomes:

```python
trainer = A2C(envs, agent, val_envs, config=TrainerConfig(
    total_steps=int(1e5), nsteps=5, validate_freq=int(2e4),
    log_dir="logs/A2C/CartPole", model_dir="models/A2C/CartPole",
))
trainer.train()
```

For a fully declarative setup, prefer the YAML CLI:

```bash
python -m rlib.A2C examples/paper/configs/classic_a2c.yaml
```

## [2.0] - 2021

- PyTorch port of the original library (previously TensorFlow-based).

## [1.0] - 2019

- Initial public release accompanying the
  [RANDAL paper](https://arxiv.org/abs/1910.09281).

[3.0.0]: https://github.com/jhare96/reinforcement-learning/releases/tag/v3.0.0
[2.0]: https://github.com/jhare96/reinforcement-learning/releases/tag/v2.0
[1.0]: https://github.com/jhare96/reinforcement-learning/releases/tag/v1.0
