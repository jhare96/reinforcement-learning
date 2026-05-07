"""Run the RND example: ``python -m rlib.RND``."""

import runpy
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
runpy.run_path(str(_REPO_ROOT / "examples" / "montezuma_rnd.py"), run_name="__main__")
