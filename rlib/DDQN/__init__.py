"""Synchronous n-step Double DQN."""

from rlib.DDQN.model import DQN
from rlib.DDQN.trainer import DDQNTrainerConfig, SyncDDQN

__all__ = ["DDQNTrainerConfig", "DQN", "SyncDDQN"]
