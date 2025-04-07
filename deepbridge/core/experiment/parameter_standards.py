"""
Parameter standardization and naming conventions for DeepBridge.
This module defines standard parameter names and types to ensure consistency across the codebase.
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