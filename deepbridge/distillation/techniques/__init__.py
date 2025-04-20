"""
Specific distillation techniques implementations.
"""

from deepbridge.distillation.techniques.knowledge_distillation import KnowledgeDistillation
from deepbridge.distillation.techniques.surrogate import SurrogateModel
from deepbridge.distillation.techniques.ensemble import EnsembleDistillation

__all__ = ["KnowledgeDistillation", "SurrogateModel", "EnsembleDistillation"]