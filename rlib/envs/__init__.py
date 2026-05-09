"""rlib's environment subpackage.

Targets the modern Gymnasium 5-tuple API directly:

* :class:`RLEnv` — abstract base class for rlib wrappers (provides
  ``__getattr__`` delegation, ``unwrapped``, context-manager support).
* :class:`RLVecEnv` — abstract base for vectorised env runners
  (``BatchEnv`` and ``DummyBatchEnv``).
* :func:`make` — re-export of :func:`gymnasium.make`.

Built-in custom envs (``ApplePicker-v0``, ``ApplePickerDeterministic-v0``)
are registered with Gymnasium at import time so ``gymnasium.make("ApplePicker-v0")``
works out of the box.
"""

from gymnasium import make
from gymnasium.envs.registration import register

from rlib.envs.base import RLEnv, RLVecEnv
from rlib.envs.vec_env import BatchEnv, DummyBatchEnv

__all__ = ["BatchEnv", "DummyBatchEnv", "RLEnv", "RLVecEnv", "make"]


# ---------------------------------------------------------------------------
# Built-in env registration
# ---------------------------------------------------------------------------

register(
    id="ApplePicker-v0",
    entry_point="rlib.envs.apple_picker:ApplePicker",
)
register(
    id="ApplePickerDeterministic-v0",
    entry_point="rlib.envs.apple_picker:ApplePickerDeterministic",
)
