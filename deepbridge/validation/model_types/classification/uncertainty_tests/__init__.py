"""
Uncertainty tests for classification models.

This module provides specialized tools for estimating and validating
uncertainty in classification model predictions.
"""

from .classifier_prediction_intervals import (
    ClassificationUncertaintyTests,
    calibrate_classifier_probabilities,
    evaluate_calibration_metrics,
    compute_confidence_intervals,
    plot_calibration_curve
)

__all__ = [
    'ClassificationUncertaintyTests',
    'calibrate_classifier_probabilities',
    'evaluate_calibration_metrics',
    'compute_confidence_intervals',
    'plot_calibration_curve'
]