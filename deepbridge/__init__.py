"""
DeepBridge - Model Validation Toolkit

DeepBridge v2.0 focuses on comprehensive model validation: robustness,
uncertainty, resilience, hyperparameter and fairness testing.

Migration Guide: https://github.com/DeepBridge-Validation/DeepBridge/blob/master/desenvolvimento/refatoracao/GUIA_RAPIDO_MIGRACAO.md
"""

# Version information
__version__ = '2.0.0'
__author__ = 'Team DeepBridge'

# Core components
from deepbridge.core.db_data import DBDataset
from deepbridge.core.experiment import Experiment

# Utils
from deepbridge.utils.model_registry import ModelType

# CLI app (imported eagerly: a broken CLI must fail loudly, not become None)
from deepbridge.cli.commands import app as cli_app

__all__ = [
    # Core components
    'DBDataset',
    'Experiment',
    'ModelType',
    # CLI
    'cli_app',
]
