"""
Wrapper module for dataset-based validation testing.

This module provides wrapper classes for running various validation tests
using DBDataset objects, offering a simplified and consistent interface.
"""

from .robustness_suite import RobustnessSuite
from .uncertainty_suite import UncertaintySuite

__all__ = [
    'RobustnessSuite',
    'UncertaintySuite'
]