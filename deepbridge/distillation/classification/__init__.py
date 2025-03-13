"""
Model distillation techniques and utilities.
Includes classification-specific implementations.
"""

from deepbridge.distillation.classification.knowledge_distillation import KnowledgeDistillation
from deepbridge.distillation.classification.model_registry import ModelRegistry, ModelType
from deepbridge.distillation.classification.surrogate import SurrogateModel


__all__ = [
    "KnowledgeDistillation",
    "ModelRegistry",
    "ModelType",
    "SurrogateModel"
]