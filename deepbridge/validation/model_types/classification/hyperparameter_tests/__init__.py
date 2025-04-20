"""
Hyperparameter testing for classification models.

This module provides tools and tests for hyperparameter optimization
and analysis specifically designed for classification models.
"""

from .classifier_tuning import (
    ClassificationHyperparameterTests,
    tune_classification_hyperparameters,
    get_classification_param_grid,
    evaluate_classifier_hyperparameters
)

__all__ = [
    'ClassificationHyperparameterTests',
    'tune_classification_hyperparameters',
    'get_classification_param_grid',
    'evaluate_classifier_hyperparameters'
]