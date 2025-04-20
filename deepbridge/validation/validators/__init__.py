"""
Validator interfaces for the validation framework.

This module provides validator classes that implement specific
validation strategies for machine learning models.
"""

from .robustness_validator import RobustnessValidator
from .adversarial_validator import AdversarialValidator
from .uncertainty_validator import UncertaintyValidator
from .hyperparameter_validator import HyperparameterValidator

# Custom validator registry
VALIDATOR_REGISTRY = {
    'robustness': RobustnessValidator,
    'adversarial': AdversarialValidator,
    'uncertainty': UncertaintyValidator,
    'hyperparameter': HyperparameterValidator
}

def get_validator(validator_type: str, **kwargs):
    """
    Get a validator instance by type.
    
    Parameters:
    -----------
    validator_type : str
        Type of validator to create
    **kwargs : dict
        Additional parameters for the validator
        
    Returns:
    --------
    BaseValidator : Validator instance
    """
    if validator_type not in VALIDATOR_REGISTRY:
        raise ValueError(f"Unknown validator type: {validator_type}")
        
    return VALIDATOR_REGISTRY[validator_type](**kwargs)

__all__ = [
    'RobustnessValidator',
    'AdversarialValidator',
    'UncertaintyValidator',
    'HyperparameterValidator',
    'get_validator'
]