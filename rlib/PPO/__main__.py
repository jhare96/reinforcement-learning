"""Run the PPO example: ``python -m rlib.PPO``."""

import runpy
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
runpy.run_path(str(_REPO_ROOT / "examples" / "atari_ppo.py"), run_name="__main__")
