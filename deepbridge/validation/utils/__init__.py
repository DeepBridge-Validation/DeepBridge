"""
Utility functions for the validation framework.

This module provides common utility functions for data handling,
model inspection, and general validation tasks.
"""

from .data_handling import (
    load_dataset,
    split_dataset,
    stratified_kfold,
    preprocess_data,
    handle_missing_values,
    create_validation_set
)

from .model_inspection import (
    get_model_info,
    extract_model_parameters,
    extract_feature_importance,
    get_model_coefficients,
    is_classifier,
    is_regressor,
    get_model_type
)

from .validation_utils import (
    setup_logger,
    format_validation_results,
    save_validation_results,
    load_validation_results,
    create_validation_summary,
    get_default_metrics
)

__all__ = [
    # Data handling
    'load_dataset',
    'split_dataset',
    'stratified_kfold',
    'preprocess_data',
    'handle_missing_values',
    'create_validation_set',
    
    # Model inspection
    'get_model_info',
    'extract_model_parameters',
    'extract_feature_importance',
    'get_model_coefficients',
    'is_classifier',
    'is_regressor',
    'get_model_type',
    
    # Validation utilities
    'setup_logger',
    'format_validation_results',
    'save_validation_results',
    'load_validation_results',
    'create_validation_summary',
    'get_default_metrics'
]