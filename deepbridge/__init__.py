"""
DeepBridge - A library for model distillation and validation.
"""

from deepbridge.db_data import DBDataset
from deepbridge.experiment import Experiment
from deepbridge.auto_distiller import AutoDistiller

__version__ = "0.1.8"
__author__ = "Team DeepBridge"

__all__ = [
    "DBDataset",
    "Experiment",
    "AutoDistiller",
]