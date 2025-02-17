"""
DeepBridge - A library for model distillation and validation.
"""

from deepbridge.db_data import DBDataset
from deepbridge.experiment import Experiment
from deepbridge.model_distiller import ModelType, ModelConfig, ModelDistiller

__version__ = "0.2.4"
__author__ = "Team DeepBridge"

# Expor as principais classes no namespace raiz
__all__ = [
    "DBDataset",
    "Experiment",
    "ModelType",
    "ModelConfig",
    "ModelDistiller",

]