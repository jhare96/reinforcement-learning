"""Run the A2C example: ``python -m rlib.A2C``.

Delegates to :mod:`examples.cartpole_a2c` so the demo and the library
share a single source of truth.
"""

import runpy
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
runpy.run_path(str(_REPO_ROOT / "examples" / "cartpole_a2c.py"), run_name="__main__")
