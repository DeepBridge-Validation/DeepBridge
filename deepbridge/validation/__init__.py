"""
Validation module for DeepBridge.

This module contains tools for model validation, reliability testing,
and robustness evaluation.
"""

from deepbridge.validation.robustness_test import RobustnessTest
from deepbridge.validation.perturbation import (
    BasePerturbation, 
    RawPerturbation, 
    QuantilePerturbation, 
    CategoricalPerturbation
)
from deepbridge.validation.robustness_metrics import (
    get_metric_function,
    is_metric_higher_better,
    RobustnessScore
)

__all__ = [
    # Robustness testing
    "RobustnessTest",
    
    # Perturbation methods
    "BasePerturbation",
    "RawPerturbation", 
    "QuantilePerturbation", 
    "CategoricalPerturbation",
    
    # Metrics
    "get_metric_function",
    "is_metric_higher_better",
    "RobustnessScore"
]