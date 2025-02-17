"""
Model distillation techniques and utilities.
Includes classification-specific implementations.
"""

from deepbridge.distillation.classification.knowledge_distillation import KnowledgeDistillation
from deepbridge.distillation.classification.model_registry import ModelRegistry, ModelType
# from deepbridge.distillation.classification.ensambledistillation import EnsembleDistillation
# from deepbridge.distillation.classification.pruning import Pruning
# from deepbridge.distillation.classification.quantization import Quantization
# from deepbridge.distillation.classification.temperature_scaling import TemperatureScaling

# __all__ = [
#     "KnowledgeDistillation",
#     "ModelRegistry",
#     "ModelType",
#     "EnsembleDistillation",
#     "Pruning",
#     "Quantization",
#     "TemperatureScaling"
# ]

__all__ = [
    "KnowledgeDistillation",
    "ModelRegistry",
    "ModelType"
]