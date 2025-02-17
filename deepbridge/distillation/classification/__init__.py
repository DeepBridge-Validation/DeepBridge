# """
# Model distillation techniques and utilities.
# Includes classification-specific implementations.
# """

# from deepbridge.distillation.classification import (
#     EnsembleDistillation,
#     KnowledgeDistillation,
#     ModelRegistry,
#     ModelType,
#     Pruning,
#     Quantization,
#     TemperatureScaling
# )

# __all__ = [
#     "EnsembleDistillation",
#     "KnowledgeDistillation",
#     "ModelRegistry",
#     "ModelType",
#     "Pruning",
#     "Quantization",
#     "TemperatureScaling"
# ]

"""
Model distillation techniques and utilities.
Includes classification-specific implementations.
"""

from deepbridge.distillation.classification import (
    KnowledgeDistillation,
    ModelRegistry,
    ModelType
)

__all__ = [
    "KnowledgeDistillation",
    "ModelRegistry",
    "ModelType"
]