"""
Interface for the validation framework.

This module provides high-level interfaces and simplified APIs
for using the validation framework components.
"""

from .simplified_api import (
    validate_model,
    evaluate_robustness,
    evaluate_uncertainty,
    tune_hyperparameters,
    generate_report
)

from .validator_factory import (
    create_validator,
    create_robustness_validator,
    create_uncertainty_validator,
    create_hyperparameter_validator
)

from .report_formatter import (
    ReportFormatter,
    format_report,
    export_report
)

__all__ = [
    # Simplified API
    'validate_model',
    'evaluate_robustness',
    'evaluate_uncertainty',
    'tune_hyperparameters',
    'generate_report',
    
    # Validator factory
    'create_validator',
    'create_robustness_validator',
    'create_uncertainty_validator',
    'create_hyperparameter_validator',
    
    # Report formatter
    'ReportFormatter',
    'format_report',
    'export_report'
]