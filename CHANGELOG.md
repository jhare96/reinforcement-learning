# Changelog

All notable changes to **rlib** are documented in this file. The format is
loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/).

## [3.1.0] - Unreleased

This release replaces the awkward `step_compat` / `reset_compat` free-function
shim with a generic, scaling environment abstraction and bumps the minimum
Python to 3.10.

### Added

- **`rlib.envs` package** — the new canonical, backend-agnostic environment
  layer. Public surface:
  - `RLEnv` — `typing.Protocol` (runtime-checkable) describing the modern
    5-tuple `(obs, reward, terminated, truncated, info)` step / `(obs, info)`
    reset contract. Use it for type annotations.
  - `RLEnvBase` — abstract base class providing helpful defaults
    (`unwrapped`, `__getattr__` forwarding, context-manager support,
    `render`, `close`, `spec`, ...). Adapters and wrappers shipped with
    rlib inherit from it.
  - `RLVecEnv` — abstract base for vectorised env runners. The
    `merge_done(terminated, truncated)` and `merge_info(...)` helpers are
    the **single canonical place** in the codebase that collapses the
    5-tuple into the legacy `done` flag, so future work on truncation
    handling has one place to change.
  - `make(env_or_id, *, backend="auto", **kwargs)` — single entry point that
    wraps a string env id or an already-constructed env into an
    `RLEnvBase`.
  - `wrap(env, *, backend="auto")` — wrap a pre-built env.
  - `register_backend(predicate, adapter_cls, ...)` — extension hook so
    users can teach rlib about new env types (`dm_env`, PettingZoo
    single-agent slices, EnvPool, in-house simulators, ...) **without**
    modifying the library.
- **Backend adapters** under `rlib.envs.adapters`:
  - `GymnasiumAdapter` — pass-through for the modern Gymnasium API.
  - `LegacyGymAdapter` — translates the legacy 4-tuple into the canonical
    5-tuple, inferring `truncated` from `info["TimeLimit.truncated"]`.

### Changed

- **Python 3.10+ required** (was 3.8+). New code uses PEP 604 `X | Y`
  union syntax and modern built-in generic aliases.
- `rlib.utils.wrappers` rewritten to subclass `RLEnvBase` and use the
  modern 5-tuple internally. All wrappers are now backend-agnostic
  through a single `_ensure_rlenv()` coercion at the wrapper boundary.
- `rlib.utils.VecEnv` (`BatchEnv`, `DummyBatchEnv`, `ChunkEnv`) now
  inherit from `RLVecEnv` and consume the wrappers' 5-tuple, collapsing
  to the legacy 4-tuple at this single boundary so existing agent
  rollouts (`A2C`, `PPO`, `RND`, ...) keep working without changes.
- `rlib.utils.SyncMultiEnvTrainer._validate_async`, `rlib.utils.play`
  and `rlib.utils.random_agent` use `rlib.envs.wrap` / the modern
  5-tuple directly.

### Deprecated

- `rlib.utils.gym_compat.step_compat` and `reset_compat` — still
  callable for backward compatibility but emit a `DeprecationWarning`
  and forward to `rlib.envs.wrap`. Will be removed in a future release.
- The `from rlib.utils.gym_compat import gym` import still works
  (re-exports the active backend) but is superseded by
  `from rlib.envs import make, wrap`.

## [3.0.0] - 2026-05-07

This release modernises the library's packaging, dependencies and developer
experience. **It contains breaking changes for users still on the legacy
`gym` package.**

### Added

- **Apache 2.0 license** — the full Apache 2.0 license text is now included
  in `LICENSE`, together with a `NOTICE` file recording attribution for code
  adapted from OpenAI Baselines.
- **PEP 621 packaging** via `pyproject.toml` with explicit
  `dependencies` (`torch`, `numpy`, `gymnasium`, `Pillow`, `tensorboard`)
  and optional extras: `[atari]`, `[classic]`, `[mujoco]`, `[docs]`, `[dev]`.
- **`requirements.txt`** mirroring the runtime dependencies, for users who
  prefer a `pip install -r` workflow (e.g. inside Docker).
- **`Dockerfile`** providing a reproducible CPU-only runtime image.
- **PEP 561 `py.typed` marker** so downstream type-checkers can consume
  `rlib`'s gradually-introduced type hints.
- **Lazy top-level imports** in `rlib/__init__.py` exposing
  `rlib.A2C`, `rlib.PPO`, `rlib.RND`, ... and a `rlib.__version__` constant.
- **`rlib.utils.gym_compat`** — a thin Gymnasium ↔ legacy-Gym shim
  providing `step_compat()` / `reset_compat()` helpers so all wrappers and
  vectorised env runners can transparently consume both APIs.
- **Runnable examples** under [`examples/`](examples/):
  `cartpole_a2c.py`, `atari_ppo.py`, `montezuma_rnd.py`.
- **Documentation scaffold** under [`docs/`](docs/) with `mkdocs.yml`,
  `index.md`, `agents.md`, `environments.md` and `wrappers.md`.
- **Community files**: `CONTRIBUTING.md`,
  `.github/ISSUE_TEMPLATE/{bug_report,feature_request}.md`,
  `.github/PULL_REQUEST_TEMPLATE.md`.

### Changed

- **Migrated from `gym` to `gymnasium`** as the recommended backend. All
  built-in environment wrappers (`rlib/utils/wrappers.py`) and the
  `BatchEnv` / `DummyBatchEnv` / `ChunkEnv` runners
  (`rlib/utils/VecEnv.py`) now route their `step()` and `reset()` calls
  through the compat shim. Agents continue to receive the legacy
  `(obs, reward, done, info)` 4-tuple API; `terminated` and `truncated`
  are merged into `done` with a logical OR.
- **`setup.py`** is reduced to a thin shim — all metadata moved to
  `pyproject.toml`.
- **Removed the editor-specific `rlib/.vscode/`** directory and added
  `.vscode/` / `.idea/` to `.gitignore`. `requirements*.txt` is explicitly
  whitelisted in `.gitignore`.
- **Cleaned up an outdated TensorFlow docstring** in
  `rlib/networks/networks.py::MaskedRNN` (the project has been pure-PyTorch
  for a while now).
- **README** rewritten with installation, quickstart, agent overview,
  citation and contribution sections.
- **Citation BibTeX** updated to the v3.0.0 release year and version.

### Migration notes

If you depend on the legacy `gym` package, the easiest path forward is:

```bash
pip install gymnasium
```

and replace `import gym` with either:

```python
import gymnasium as gym
```

or

```python
from rlib.utils.gym_compat import gym  # automatically picks the best backend
```

If you cannot migrate yet, `rlib` will fall back to legacy `gym` if
`gymnasium` is not installed.

## [2.0] - 2021

- PyTorch port of the original library (previously TensorFlow-based).

## [1.0] - 2019

- Initial public release accompanying the
  [RANDAL paper](https://arxiv.org/abs/1910.09281).

[3.0.0]: https://github.com/jhare96/reinforcement-learning/releases/tag/v3.0.0
[2.0]: https://github.com/jhare96/reinforcement-learning/releases/tag/v2.0
[1.0]: https://github.com/jhare96/reinforcement-learning/releases/tag/v1.0
