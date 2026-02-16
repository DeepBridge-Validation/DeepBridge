"""
DeepBridge - Model Validation Toolkit

DeepBridge v2.0 focuses on comprehensive model validation.

For additional features:
- Model Distillation: pip install deepbridge-distillation
- Synthetic Data: pip install deepbridge-synthetic

Migration Guide: https://github.com/DeepBridge-Validation/DeepBridge/blob/master/desenvolvimento/refatoracao/GUIA_RAPIDO_MIGRACAO.md
"""

# Version information
__version__ = '2.0.0-alpha.1'
__author__ = 'Team DeepBridge'

# Core components
from deepbridge.core.db_data import DBDataset
from deepbridge.core.experiment import Experiment

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
    'ModelType',
    # CLI
    'cli_app',
]
