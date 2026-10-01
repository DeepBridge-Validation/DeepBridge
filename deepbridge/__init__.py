"""
DeepBridge - Model Validation Toolkit

DeepBridge v2.0 focuses on comprehensive model validation: robustness,
uncertainty, resilience, hyperparameter and fairness testing.

Migration Guide: https://github.com/DeepBridge-Validation/DeepBridge/blob/master/desenvolvimento/refatoracao/GUIA_RAPIDO_MIGRACAO.md
"""

# Version information
#
# O master trazia 1.63.0, o ultimo release da linha v1.x, com um aviso de
# depreciacao pedindo que o usuario migrasse para a 2.0. Esta linha de codigo
# JA E a 2.x, entao o aviso foi removido no merge: mantido, ele diria a quem
# importa a 2.1 que atualizasse para a 2.0. O numero 1.63.0 tambem nao
# sobrevive aqui, apesar de ser maior que 2.0.0 na ordem de publicacao: ele
# versiona a linha antiga, nao esta.
__version__ = '2.1.0'
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
