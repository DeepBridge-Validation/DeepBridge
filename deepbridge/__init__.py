"""
DeepBridge - Advanced Machine Learning Model Validation and Distillation

DeepBridge provides tools for model validation, distillation, 
and performance analysis to create efficient machine learning models.
"""

# Version information
__version__ = "0.1.12"
__author__ = "Team DeepBridge"

# Core components
from deepbridge.core.db_data import DBDataset
from deepbridge.core.experiment import Experiment

# Distillation components
from deepbridge.distillation.auto_distiller import AutoDistiller
from deepbridge.distillation.techniques.surrogate import SurrogateModel
from deepbridge.distillation.techniques.knowledge_distillation import KnowledgeDistillation

# Utils
from deepbridge.utils.model_registry import ModelType

# Import CLI app
try:
    from deepbridge.cli.commands import app as cli_app
except ImportError:
    cli_app = None

__all__ = [
    # Core components
    "DBDataset",
    "Experiment",
    
    # Distillation components
    "AutoDistiller",
    "SurrogateModel",
    "KnowledgeDistillation",
    "ModelType",
    
    # CLI
    "cli_app"
]