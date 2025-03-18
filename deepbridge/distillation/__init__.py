"""
Model distillation components and algorithms.
"""

from deepbridge.distillation.auto_distiller import AutoDistiller
from deepbridge.distillation.experiment_runner import ExperimentRunner
from deepbridge.distillation.techniques.knowledge_distillation import KnowledgeDistillation
from deepbridge.distillation.techniques.surrogate import SurrogateModel
from deepbridge.distillation.techniques.ensemble import EnsembleDistillation

__all__ = [
    "AutoDistiller",
    "ExperimentRunner",
    "KnowledgeDistillation", 
    "SurrogateModel",
    "EnsembleDistillation"
]