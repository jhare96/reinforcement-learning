"""Setup script for the rlib reinforcement learning library.

Most metadata is declared in ``pyproject.toml``. This shim is kept so that the
historical ``pip install -e .`` workflow documented in the README continues to
work on older versions of pip/setuptools that do not yet honour PEP 621
metadata for editable installs.
"""

from setuptools import setup

setup()
