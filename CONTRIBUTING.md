# Contributing to rlib

Thanks for your interest in improving **rlib**! This document outlines the
typical workflow for contributors.

## Reporting issues

Before opening a new issue, please search the
[existing issues](https://github.com/jhare96/reinforcement-learning/issues)
to avoid duplicates. When filing a bug, please use the bug report template
and include:

- The version of `rlib`, `torch` and `gymnasium` you are running.
- The exact command you ran and the full traceback.
- A minimal reproducible example, ideally on a classic-control env.

## Development setup

```bash
git clone https://github.com/jhare96/reinforcement-learning.git
cd reinforcement-learning
python -m venv .venv && source .venv/bin/activate
pip install -e ".[classic,dev]"
```

For Atari work, also install `pip install -e ".[atari]"`.

## Coding style

- Target **Python 3.10+**.
- Match the surrounding code style. The codebase is gradually being typed —
  please add type hints (PEP 604 `X | Y` union syntax, built-in generics
  like `list[int]`/`dict[str, Any]`) to any new public function.
- Keep changes focused: one feature or fix per PR.
- Avoid adding new top-level dependencies unless strictly necessary; prefer
  putting heavy or environment-specific deps behind an optional extra in
  `pyproject.toml`.

## Environment abstraction

When touching wrappers, vectorised env runners, or adding support for a new
gym-like backend (`dm_env`, PettingZoo, EnvPool, an in-house simulator, ...),
build on the [`rlib.envs`](rlib/envs/) package:

- Subclass [`RLEnvBase`](rlib/envs/base.py) (or, for protocol-only typing,
  use the `RLEnv` Protocol) and implement the modern 5-tuple
  `step(action) -> (obs, reward, terminated, truncated, info)` and
  `reset(*, seed=None, options=None) -> (obs, info)`.
- For a new backend, drop a small adapter file under
  [`rlib/envs/adapters/`](rlib/envs/adapters/) and register it with
  `register_backend(predicate, adapter_cls)` so `rlib.envs.make` /
  `rlib.envs.wrap` will auto-pick it up.

The legacy `rlib.utils.gym_compat` shim has been removed; use
`rlib.envs.wrap` (or just `import gymnasium as gym`) in new code.

## Submitting a pull request

1. Fork the repo and create a feature branch from `master`.
2. Make your changes in small, well-scoped commits.
3. If your change affects user-facing behaviour, update the README and add
   an entry under the *Unreleased* section of `CHANGELOG.md`.
4. Open a pull request using the PR template and describe the motivation,
   the change, and how you verified it.

## Adding a new agent

New agents should:

- Live under `rlib/<AgentName>/`.
- Subclass `rlib.utils.SyncMultiEnvTrainer.SyncMultiEnvTrainer` (or document
  why a custom trainer is needed).
- Be re-exported from `rlib/<AgentName>/__init__.py`.
- Ship with a runnable example under `examples/`.
- Be referenced from the README's *Implemented agents* table and the
  `docs/agents.md` page.

## License

By contributing, you agree that your contributions will be licensed under the
[Apache License 2.0](LICENSE).
