"""Random Network Distillation with Auxiliary Learning (RANDAL)."""

from rlib.RANDAL.model import RANDAL
from rlib.RANDAL.trainer import RANDALTrainer, RANDALTrainerConfig

__all__ = ["RANDAL", "RANDALTrainer", "RANDALTrainerConfig"]
