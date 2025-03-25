"""
Perturbation functions for robustness testing.

This module provides tools for applying various perturbations to data
to test model robustness.
"""

from .gaussian_perturbation import apply_gaussian_noise, GaussianNoisePerturber
from .salt_pepper_perturbation import apply_salt_pepper_noise, SaltPepperNoisePerturber
from .quantile_perturbation import apply_feature_perturbation, FeaturePerturber

__all__ = [
    'apply_gaussian_noise',
    'GaussianNoisePerturber',
    'apply_salt_pepper_noise',
    'SaltPepperNoisePerturber',
    'apply_feature_perturbation',
    'FeaturePerturber',
    'get_perturbation_function'
]

def get_perturbation_function(perturbation_type: str):
    """
    Get a perturbation function by name.
    
    Parameters:
    -----------
    perturbation_type : str
        Type of perturbation function:
        - 'gaussian': Gaussian noise
        - 'salt_pepper': Salt and pepper noise
        - 'feature': Feature-specific perturbation
        
    Returns:
    --------
    callable : Perturbation function
    """
    if perturbation_type == 'gaussian':
        return apply_gaussian_noise
    elif perturbation_type == 'salt_pepper':
        return apply_salt_pepper_noise
    elif perturbation_type == 'feature':
        return apply_feature_perturbation
    else:
        raise ValueError(f"Unknown perturbation type: {perturbation_type}")