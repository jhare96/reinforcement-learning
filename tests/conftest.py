"""Pytest configuration shared across the rlib test suite."""

from __future__ import annotations

import os

# Keep CI runs deterministic and CPU-only by default.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTHONHASHSEED", "0")
