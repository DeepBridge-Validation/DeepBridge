"""
DeepBridge - Advanced Machine Learning Model Validation and Distillation

DeepBridge provides tools for model validation, distillation,
and performance analysis to create efficient machine learning models.
"""

# Version information
__version__ = '1.63.0'
__author__ = 'Team DeepBridge'

# Deprecation warning for v1.x
import warnings
warnings.warn(
    "DeepBridge v1.x is deprecated. "
    "Please upgrade to v2.0: pip install --upgrade deepbridge\n"
    "Migration guide: https://github.com/DeepBridge-Validation/DeepBridge/blob/master/desenvolvimento/refatoracao/GUIA_RAPIDO_MIGRACAO.md",
    DeprecationWarning,
    stacklevel=2
)

# Core components
from deepbridge.core.db_data import DBDataset
from deepbridge.core.experiment import Experiment

# Distillation components
from deepbridge.distillation.auto_distiller import AutoDistiller
from deepbridge.distillation.techniques.knowledge_distillation import (
    KnowledgeDistillation,
)
from deepbridge.distillation.techniques.surrogate import SurrogateModel

# Utils
from deepbridge.utils.model_registry import ModelType

# Import CLI app
try:
    from deepbridge.cli.commands import app as cli_app
except ImportError:
    cli_app = None

__all__ = [
    # Core components
    'DBDataset',
    'Experiment',
    # Distillation components
    'AutoDistiller',
    'SurrogateModel',
    'KnowledgeDistillation',
    'ModelType',
    # CLI
    'cli_app',
]
