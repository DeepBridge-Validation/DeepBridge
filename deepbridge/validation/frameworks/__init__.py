"""
Core validation frameworks.

This module provides access to various validation frameworks for evaluating
different aspects of machine learning models, including robustness,
uncertainty, and hyperparameter importance.
"""

# Import from robustness framework
from .robustness import (
    RobustnessTester,
    run_robustness_test_suite,
    OutlierRobustnessTester,
    detect_outliers,
    ResilienceAnalyzer,
    compute_resilience_score
)

# Import from uncertainty framework
from .uncertainty import (
    ConformalPredictor,
    compute_prediction_sets,
    CalibrationAnalyzer,
    calibrate_probabilities,
    expected_calibration_error,
    maximum_calibration_error,
    BootstrapUncertainty
)

# Import from hyperparameters framework
from .hyperparameters import (
    HyperparameterImportance,
    compute_importance_fanova,
    compute_importance_permutation,
    EfficientTuner,
    bayesian_optimization_tuner,
    successive_halving_tuner
)

__all__ = [
    # Robustness
    'RobustnessTester',
    'run_robustness_test_suite',
    'OutlierRobustnessTester',
    'detect_outliers',
    'ResilienceAnalyzer',
    'compute_resilience_score',
    
    # Uncertainty
    'ConformalPredictor',
    'compute_prediction_sets',
    'CalibrationAnalyzer',
    'calibrate_probabilities',
    'expected_calibration_error',
    'maximum_calibration_error',
    'BootstrapUncertainty',
    
    # Hyperparameters
    'HyperparameterImportance',
    'compute_importance_fanova',
    'compute_importance_permutation',
    'EfficientTuner',
    'bayesian_optimization_tuner',
    'successive_halving_tuner'
]