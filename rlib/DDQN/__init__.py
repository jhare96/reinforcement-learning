"""Synchronous n-step Double DQN."""

from rlib.DDQN.model import DQN
from rlib.DDQN.trainer import SyncDDQN

__all__ = ["DQN", "SyncDDQN"]
