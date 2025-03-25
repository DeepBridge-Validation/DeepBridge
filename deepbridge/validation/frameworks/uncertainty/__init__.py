"""
Uncertainty estimation and calibration framework.

This module provides tools for estimating and validating model uncertainty,
including calibration analysis and conformal prediction.
"""

from .conformal_prediction import (
    ConformalPredictor,
    compute_prediction_sets,
    get_conformal_score
)

from .calibration_analyzer import (
    CalibrationAnalyzer,
    calibrate_probabilities,
    isotonic_calibration,
    platt_scaling
)

from .uncertainty_metrics import (
    expected_calibration_error,
    maximum_calibration_error,
    brier_score,
    negative_log_likelihood,
    compute_uncertainty_metrics
)

from .bootstrap import (
    BootstrapUncertainty
)

__all__ = [
    'ConformalPredictor',
    'compute_prediction_sets',
    'get_conformal_score',
    'CalibrationAnalyzer',
    'calibrate_probabilities',
    'isotonic_calibration',
    'platt_scaling',
    'expected_calibration_error',
    'maximum_calibration_error',
    'brier_score',
    'negative_log_likelihood',
    'compute_uncertainty_metrics',
    'BootstrapUncertainty'
]