"""
Salt and pepper noise perturbation for robustness testing.

This module provides functions for applying salt and pepper noise
to test model robustness.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple


class SaltPepperNoisePerturber:
    """
    Apply salt and pepper noise perturbation to data.
    
    This class provides methods for adding salt and pepper noise to data
    to test model robustness.
    """
    
    def __init__(
        self,
        noise_ratio: float = 0.05,
        salt_value: Optional[float] = None,
        pepper_value: Optional[float] = None,
        salt_prob: float = 0.5,
        seed: Optional[int] = None
    ):
        """
        Initialize the salt and pepper noise perturber.
        
        Parameters:
        -----------
        noise_ratio : float
            Fraction of values to perturb
        salt_value : float or None
            Value to use for salt noise (None uses max value)
        pepper_value : float or None
            Value to use for pepper noise (None uses min value)
        salt_prob : float
            Probability of salt vs. pepper
        seed : int or None
            Random seed for reproducibility
        """
        self.noise_ratio = noise_ratio
        self.salt_value = salt_value
        self.pepper_value = pepper_value
        self.salt_prob = salt_prob
        self.seed = seed
        
        if self.seed is not None:
            np.random.seed(self.seed)
    
    def perturb(
        self,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Apply salt and pepper noise perturbation.
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Data to perturb
            
        Returns:
        --------
        array-like or DataFrame : Perturbed data
        """
        # Check if X is a DataFrame
        is_dataframe = isinstance(X, pd.DataFrame)
        
        # Convert to numpy array if necessary
        if is_dataframe:
            columns = X.columns
            index = X.index
            X_values = X.values
        else:
            X_values = X
            
        # Create copy
        X_perturbed = X_values.copy()
        
        # Determine salt and pepper values
        if self.salt_value is None:
            salt_value = np.max(X_values)
        else:
            salt_value = self.salt_value
            
        if self.pepper_value is None:
            pepper_value = np.min(X_values)
        else:
            pepper_value = self.pepper_value
            
        # Generate mask for noisy values
        mask = np.random.random(X_values.shape) < self.noise_ratio
        
        # Generate mask for salt vs. pepper
        salt_mask = np.random.random(X_values.shape) < self.salt_prob
        
        # Apply salt noise
        X_perturbed[mask & salt_mask] = salt_value
        
        # Apply pepper noise
        X_perturbed[mask & (~salt_mask)] = pepper_value
        
        # Convert back to DataFrame if necessary
        if is_dataframe:
            X_perturbed = pd.DataFrame(X_perturbed, columns=columns, index=index)
            
        return X_perturbed


def apply_salt_pepper_noise(
    X: Union[np.ndarray, pd.DataFrame],
    noise_ratio: float = 0.05,
    salt_value: Optional[float] = None,
    pepper_value: Optional[float] = None,
    salt_prob: float = 0.5,
    seed: Optional[int] = None
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Apply salt and pepper noise perturbation to data.
    
    Parameters:
    -----------
    X : array-like or DataFrame
        Data to perturb
    noise_ratio : float
        Fraction of values to perturb
    salt_value : float or None
        Value to use for salt noise (None uses max value)
    pepper_value : float or None
        Value to use for pepper noise (None uses min value)
    salt_prob : float
        Probability of salt vs. pepper
    seed : int or None
        Random seed for reproducibility
        
    Returns:
    --------
    array-like or DataFrame : Perturbed data
    """
    perturber = SaltPepperNoisePerturber(
        noise_ratio=noise_ratio,
        salt_value=salt_value,
        pepper_value=pepper_value,
        salt_prob=salt_prob,
        seed=seed
    )
    
    return perturber.perturb(X)