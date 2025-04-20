"""
Optimization techniques for model distillation.
"""

from deepbridge.distillation.techniques.optimization.pruning import Pruning
from deepbridge.distillation.techniques.optimization.quantization import Quantization
from deepbridge.distillation.techniques.optimization.temperature_scaling import TemperatureScaling

__all__ = ["Pruning", "Quantization", "TemperatureScaling"]