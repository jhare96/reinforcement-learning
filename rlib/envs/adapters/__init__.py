"""Backend adapter implementations.

Each backend is a self-contained file exposing one
:class:`~rlib.envs.base.RLEnvBase` subclass.  Add a new backend by
dropping a new file here and registering it via
:func:`rlib.envs.register_backend`.
"""

from rlib.envs.adapters.gymnasium_adapter import GymnasiumAdapter
from rlib.envs.adapters.legacy_gym_adapter import LegacyGymAdapter

__all__ = ["GymnasiumAdapter", "LegacyGymAdapter"]
