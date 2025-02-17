"""
DeepBridge - A library for model distillation and validation.
"""

from deepbridge.db_data import DBDataset
from deepbridge.experiment import Experiment
from deepbridge.model_distiller import ModelDistiller

__version__ = "0.2.6"
__author__ = "Team DeepBridge"

__all__ = [
    "DBDataset",
    "Experiment",
    "ModelDistiller"
]