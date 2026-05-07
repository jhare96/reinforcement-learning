"""Decoupled Advantage Actor-Critic (DAAC)."""

from rlib.DAAC.model import DAAC, PolicyModel, ValueModel
from rlib.DAAC.trainer import DAACTrainer

__all__ = ["DAAC", "DAACTrainer", "PolicyModel", "ValueModel"]
