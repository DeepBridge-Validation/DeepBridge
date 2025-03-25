"""
Gaussian noise perturbation for robustness testing.

This module provides functions for applying Gaussian noise perturbations
to test model robustness.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple


class GaussianNoisePerturber:
    """
    Apply Gaussian noise perturbation to data.
    
    This class provides methods for adding Gaussian noise to data
    to test model robustness.
    """
    
    def __init__(
        self,
        noise_level: float = 0.1,
        feature_wise: bool = False,
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize the Gaussian noise perturber.
        
        Parameters:
        -----------
        noise_level : float
            Standard deviation of the Gaussian noise
            (as fraction of data standard deviation)
        feature_wise : bool
            Whether to apply different noise levels to each feature
        clip_min : float or None
            Minimum value to clip perturbed data
        clip_max : float or None
            Maximum value to clip perturbed data
        seed : int or None
            Random seed for reproducibility
        """
        self.noise_level = noise_level
        self.feature_wise = feature_wise
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.seed = seed
        
        if self.seed is not None:
            np.random.seed(self.seed)
    
    def perturb(
        self,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Apply Gaussian noise perturbation.
        
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
            
        # Calculate standard deviation
        if self.feature_wise:
            std_dev = np.std(X_values, axis=0) * self.noise_level
        else:
            std_dev = np.std(X_values) * self.noise_level
            
        # Generate noise
        noise = np.random.normal(0, std_dev, X_values.shape)
        
        # Apply perturbation
        X_perturbed = X_values + noise
        
        # Clip values if specified
        if self.clip_min is not None or self.clip_max is not None:
            X_perturbed = np.clip(X_perturbed, self.clip_min, self.clip_max)
            
        # Convert back to DataFrame if necessary
        if is_dataframe:
            X_perturbed = pd.DataFrame(X_perturbed, columns=columns, index=index)
            
        return X_perturbed


def apply_gaussian_noise(
    X: Union[np.ndarray, pd.DataFrame],
    noise_level: float = 0.1,
    feature_wise: bool = False,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
    seed: Optional[int] = None
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Apply Gaussian noise perturbation to data.
    
    Parameters:
    -----------
    X : array-like or DataFrame
        Data to perturb
    noise_level : float
        Standard deviation of the Gaussian noise
        (as fraction of data standard deviation)
    feature_wise : bool
        Whether to apply different noise levels to each feature
    clip_min : float or None
        Minimum value to clip perturbed data
    clip_max : float or None
        Maximum value to clip perturbed data
    seed : int or None
        Random seed for reproducibility
        
    Returns:
    --------
    array-like or DataFrame : Perturbed data
    """
    perturber = GaussianNoisePerturber(
        noise_level=noise_level,
        feature_wise=feature_wise,
        clip_min=clip_min,
        clip_max=clip_max,
        seed=seed
    )
    
    return perturber.perturb(X)