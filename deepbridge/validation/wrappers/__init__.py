"""
Wrapper module for dataset-based validation testing.

This module provides wrapper classes for running various validation tests
using DBDataset objects, offering a simplified and consistent interface.
"""

from .base_wrapper import BaseWrapper
from .feature_perturbation import FeaturePerturbationTests
from .outlier_robustness import OutlierRobustnessTests
from .distribution_shift import DistributionShiftTests
from .adversarial_robustness import AdversarialRobustnessTests
from .robustness_suite import RobustnessSuite

__all__ = [
    'BaseWrapper',
    'FeaturePerturbationTests',
    'OutlierRobustnessTests',
    'DistributionShiftTests',
    'AdversarialRobustnessTests',
    'RobustnessSuite'
]