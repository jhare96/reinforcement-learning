"""Decoupled Advantage Actor-Critic (DAAC)."""

from rlib.DAAC.model import DAAC, PolicyModel, ValueModel
from rlib.DAAC.trainer import DAACTrainer, DAACTrainerConfig

__all__ = ["DAAC", "DAACTrainer", "DAACTrainerConfig", "PolicyModel", "ValueModel"]
