"""
Parameter standardization and naming conventions for DeepBridge.
This module defines standard parameter names and types to ensure consistency across the codebase.
It also serves as a central repository for configuration templates used by different test types.
"""

import typing as t
from enum import Enum, auto

# Standard parameter names
class ParameterNames:
    """
    Standard parameter names to be used consistently throughout the codebase.
    Uppercase because these are constants.
    """
    # Dataset and features parameters
    DATASET = "dataset"
    FEATURES = "features" 
    FEATURE_SUBSET = "feature_subset"  # Consistently use feature_subset instead of features_select
    TARGET = "target"
    X_TRAIN = "X_train"
    X_TEST = "X_test"
    Y_TRAIN = "y_train"
    Y_TEST = "y_test"
    
    # Configuration parameters
    CONFIG_NAME = "config_name"  # Consistently use config_name instead of suite
    EXPERIMENT_TYPE = "experiment_type"
    TESTS = "tests"
    VERBOSE = "verbose"
    
    # Test specific parameters
    TEST_TYPE = "test_type"
    METRIC = "metric"
    N_TRIALS = "n_trials"
    N_ITERATIONS = "n_iterations"  # Multiple iterations for robustness testing
    
    # Model parameters
    MODEL = "model"
    MODEL_TYPE = "model_type"
    HYPERPARAMETERS = "hyperparameters"
    
    # Splitting parameters
    TEST_SIZE = "test_size"
    RANDOM_STATE = "random_state"
    
    # Performance metric parameters
    ACCURACY = "accuracy"
    AUC = "auc"  # Use lowercase to be consistent
    F1 = "f1"
    PRECISION = "precision"
    RECALL = "recall"

# Test type enum
class TestType(Enum):
    """Enum for standardized test types"""
    ROBUSTNESS = "robustness"
    UNCERTAINTY = "uncertainty"
    RESILIENCE = "resilience"
    HYPERPARAMETERS = "hyperparameters"
    
    def __str__(self):
        return self.value

# Config type enum
class ConfigName(Enum):
    """Enum for standardized configuration names"""
    QUICK = "quick"
    MEDIUM = "medium"
    FULL = "full"
    
    def __str__(self):
        return self.value

# Experiment type enum
class ExperimentType(Enum):
    """Enum for standardized experiment types"""
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    FORECASTING = "forecasting"
    
    def __str__(self):
        return self.value

# Standard type aliases
DatasetType = t.TypeVar('DatasetType')  # Type of the dataset object
ModelType = t.TypeVar('ModelType')      # Type of model objects

# Dictionary of features to feature importance
FeatureImportanceDict = t.Dict[str, float]

# Test config dictionary type
TestConfigDict = t.Dict[str, t.Any]

# Test results dictionary type
TestResultsDict = t.Dict[str, t.Any]

# Type for a list of test types
TestTypeList = t.List[str]

# Standard function signatures

def standardize_feature_names(feature_names: t.List[str]) -> t.List[str]:
    """
    Standardize feature names by replacing spaces with underscores and making lowercase.
    
    Args:
        feature_names: List of feature names to standardize
        
    Returns:
        List of standardized feature names
    """
    return [name.lower().replace(' ', '_') for name in feature_names]

def get_test_types() -> t.List[str]:
    """
    Get list of all supported test types.
    
    Returns:
        List of standardized test type strings
    """
    return [test_type.value for test_type in TestType]

def get_config_names() -> t.List[str]:
    """
    Get list of all supported configuration names.
    
    Returns:
        List of standardized configuration name strings
    """
    return [config.value for config in ConfigName]

def get_experiment_types() -> t.List[str]:
    """
    Get list of all supported experiment types.
    
    Returns:
        List of standardized experiment type strings
    """
    return [exp_type.value for exp_type in ExperimentType]

def is_valid_test_type(test_type: str) -> bool:
    """
    Check if a test type string is valid.
    
    Args:
        test_type: Test type string to check
        
    Returns:
        True if valid, False otherwise
    """
    return test_type in get_test_types()

def is_valid_config_name(config_name: str) -> bool:
    """
    Check if a configuration name string is valid.
    
    Args:
        config_name: Configuration name string to check
        
    Returns:
        True if valid, False otherwise
    """
    return config_name in get_config_names()

def is_valid_experiment_type(experiment_type: str) -> bool:
    """
    Check if an experiment type string is valid.

    Args:
        experiment_type: Experiment type string to check

    Returns:
        True if valid, False otherwise
    """
    return experiment_type in get_experiment_types()

# ---------------------------------------------------------------
# Centralized Configuration Templates for Different Test Types
# ---------------------------------------------------------------

# Configuration templates for robustness testing
ROBUSTNESS_CONFIGS = {
    ConfigName.QUICK.value: {
        'perturbation_methods': ['raw', 'quantile'],
        'levels': [0.1, 0.2],
        'n_trials': 3
    },
    ConfigName.MEDIUM.value: {
        'perturbation_methods': ['raw', 'quantile', 'adversarial'],
        'levels': [0.1, 0.2, 0.4],
        'n_trials': 6
    },
    ConfigName.FULL.value: {
        'perturbation_methods': ['raw', 'quantile', 'adversarial', 'custom'],
        'levels': [0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
        'n_trials': 10
    }
}

# Configuration templates for uncertainty testing
UNCERTAINTY_CONFIGS = {
    ConfigName.QUICK.value: [
        {'method': 'crqr', 'params': {'alpha': 0.1, 'test_size': 0.3, 'calib_ratio': 1/3}},
        {'method': 'crqr', 'params': {'alpha': 0.2, 'test_size': 0.3, 'calib_ratio': 1/3}}
    ],
    ConfigName.MEDIUM.value: [
        {'method': 'crqr', 'params': {'alpha': 0.05, 'test_size': 0.3, 'calib_ratio': 1/3}},
        {'method': 'crqr', 'params': {'alpha': 0.1, 'test_size': 0.3, 'calib_ratio': 1/3}},
        {'method': 'crqr', 'params': {'alpha': 0.2, 'test_size': 0.3, 'calib_ratio': 1/3}}
    ],
    ConfigName.FULL.value: [
        {'method': 'crqr', 'params': {'alpha': 0.01, 'test_size': 0.3, 'calib_ratio': 1/3}},
        {'method': 'crqr', 'params': {'alpha': 0.05, 'test_size': 0.3, 'calib_ratio': 1/3}},
        {'method': 'crqr', 'params': {'alpha': 0.1, 'test_size': 0.3, 'calib_ratio': 1/3}},
        {'method': 'crqr', 'params': {'alpha': 0.2, 'test_size': 0.3, 'calib_ratio': 1/3}},
        {'method': 'crqr', 'params': {'alpha': 0.3, 'test_size': 0.3, 'calib_ratio': 1/3}}
    ]
}

# Configuration templates for resilience testing
RESILIENCE_CONFIGS = {
    ConfigName.QUICK.value: {
        'drift_types': ['covariate', 'label'],
        'drift_intensities': [0.1, 0.2]
    },
    ConfigName.MEDIUM.value: {
        'drift_types': ['covariate', 'label', 'concept'],
        'drift_intensities': [0.05, 0.1, 0.2]
    },
    ConfigName.FULL.value: {
        'drift_types': ['covariate', 'label', 'concept', 'temporal'],
        'drift_intensities': [0.01, 0.05, 0.1, 0.2, 0.3]
    }
}

# Configuration templates for hyperparameter testing
HYPERPARAMETER_CONFIGS = {
    ConfigName.QUICK.value: {
        'n_trials': 10,
        'optimization_metric': 'accuracy'
    },
    ConfigName.MEDIUM.value: {
        'n_trials': 30,
        'optimization_metric': 'accuracy'
    },
    ConfigName.FULL.value: {
        'n_trials': 100,
        'optimization_metric': 'accuracy'
    }
}

# Master configuration dictionary mapping test types to their configurations
TEST_CONFIGS = {
    TestType.ROBUSTNESS.value: ROBUSTNESS_CONFIGS,
    TestType.UNCERTAINTY.value: UNCERTAINTY_CONFIGS,
    TestType.RESILIENCE.value: RESILIENCE_CONFIGS,
    TestType.HYPERPARAMETERS.value: HYPERPARAMETER_CONFIGS
}

def get_test_config(test_type: str, config_name: str) -> t.Dict[str, t.Any]:
    """
    Get configuration options for a specific test type and configuration level.

    Args:
        test_type: Type of test ('robustness', 'uncertainty', etc.)
        config_name: Configuration level ('quick', 'medium', 'full')

    Returns:
        Dictionary with configuration options

    Raises:
        ValueError: If test_type or config_name is invalid
    """
    if not is_valid_test_type(test_type):
        raise ValueError(f"Invalid test type: {test_type}. Valid options: {get_test_types()}")

    if not is_valid_config_name(config_name):
        raise ValueError(f"Invalid configuration name: {config_name}. Valid options: {get_config_names()}")

    if test_type not in TEST_CONFIGS:
        raise ValueError(f"No configuration template defined for test type: {test_type}")

    test_config = TEST_CONFIGS[test_type]
    if config_name not in test_config:
        raise ValueError(f"No {config_name} configuration defined for test type: {test_type}")

    return test_config[config_name]