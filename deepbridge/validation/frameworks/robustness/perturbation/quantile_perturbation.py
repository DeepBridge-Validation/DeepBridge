"""
Quantile-based feature perturbation for robustness testing.

This module provides functions for applying feature-specific perturbations
based on quantiles to test model robustness.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple


class FeaturePerturber:
    """
    Apply feature-specific perturbation to data.
    
    This class provides methods for perturbing specific features in the data
    to test model robustness.
    """
    
    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        perturbation_type: str = 'quantile',
        perturbation_level: float = 0.1,
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize the feature perturber.
        
        Parameters:
        -----------
        feature_names : list of str or None
            Names of features to perturb (None perturbs all)
        perturbation_type : str
            Type of perturbation to apply:
            - 'quantile': Move values to their quantiles
            - 'zero': Set values to zero
            - 'flip': Flip binary values
            - 'shuffle': Randomly shuffle values
        perturbation_level : float
            Level of perturbation to apply
        clip_min : float or None
            Minimum value to clip perturbed data
        clip_max : float or None
            Maximum value to clip perturbed data
        seed : int or None
            Random seed for reproducibility
        """
        self.feature_names = feature_names
        self.perturbation_type = perturbation_type
        self.perturbation_level = perturbation_level
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
        Apply feature-specific perturbation.
        
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
        
        if not is_dataframe and self.feature_names is not None:
            raise ValueError("Feature names can only be used with DataFrame input")
            
        # Create copy
        if is_dataframe:
            X_perturbed = X.copy()
            
            # Determine features to perturb
            if self.feature_names is None:
                features_to_perturb = X.columns
            else:
                features_to_perturb = [f for f in self.feature_names if f in X.columns]
                
            # Apply perturbation to each feature
            for feature in features_to_perturb:
                X_perturbed[feature] = self._perturb_feature(X[feature])
                
        else:
            X_perturbed = X.copy()
            
            # Apply perturbation to each column
            for j in range(X.shape[1]):
                X_perturbed[:, j] = self._perturb_feature(X[:, j])
                
        # Clip values if specified
        if self.clip_min is not None or self.clip_max is not None:
            if is_dataframe:
                X_perturbed = X_perturbed.clip(lower=self.clip_min, upper=self.clip_max)
            else:
                X_perturbed = np.clip(X_perturbed, self.clip_min, self.clip_max)
                
        return X_perturbed
    
    def _perturb_feature(self, feature_values: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """
        Apply perturbation to a single feature.
        
        Parameters:
        -----------
        feature_values : array-like or Series
            Feature values to perturb
            
        Returns:
        --------
        numpy.ndarray : Perturbed feature values
        """
        # Convert to numpy array if necessary
        if isinstance(feature_values, pd.Series):
            values = feature_values.values
        else:
            values = feature_values
            
        # Create copy
        perturbed = values.copy()
        
        if self.perturbation_type == 'quantile':
            # Move values to their quantiles
            quantiles = np.quantile(values, [self.perturbation_level, 1 - self.perturbation_level])
            
            # Determine mask for lower and upper quantiles
            mask = np.random.random(len(values)) < self.perturbation_level
            
            # Apply perturbation
            mask_lower = mask & (np.random.random(len(values)) < 0.5)
            mask_upper = mask & (~mask_lower)
            
            perturbed[mask_lower] = quantiles[0]
            perturbed[mask_upper] = quantiles[1]
            
        elif self.perturbation_type == 'zero':
            # Set values to zero
            mask = np.random.random(len(values)) < self.perturbation_level
            perturbed[mask] = 0
            
        elif self.perturbation_type == 'flip':
            # Flip binary values
            unique_values = np.unique(values)
            
            if len(unique_values) == 2:
                # For binary features
                mask = np.random.random(len(values)) < self.perturbation_level
                
                # Map each value to the other
                value_map = {unique_values[0]: unique_values[1], unique_values[1]: unique_values[0]}
                
                # Apply perturbation
                for i in np.where(mask)[0]:
                    perturbed[i] = value_map[values[i]]
            else:
                # Not a binary feature
                pass
                
        elif self.perturbation_type == 'shuffle':
            # Randomly shuffle values
            mask = np.random.random(len(values)) < self.perturbation_level
            
            # Get values to shuffle
            values_to_shuffle = values[mask]
            
            # Shuffle and replace
            if len(values_to_shuffle) > 1:
                np.random.shuffle(values_to_shuffle)
                perturbed[mask] = values_to_shuffle
                
        else:
            raise ValueError(f"Unknown perturbation type: {self.perturbation_type}")
            
        return perturbed


def apply_feature_perturbation(
    X: Union[np.ndarray, pd.DataFrame],
    feature_names: Optional[List[str]] = None,
    perturbation_type: str = 'quantile',
    perturbation_level: float = 0.1,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
    seed: Optional[int] = None
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Apply feature-specific perturbation to data.
    
    Parameters:
    -----------
    X : array-like or DataFrame
        Data to perturb
    feature_names : list of str or None
        Names of features to perturb (None perturbs all)
    perturbation_type : str
        Type of perturbation to apply
    perturbation_level : float
        Level of perturbation to apply
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
    perturber = FeaturePerturber(
        feature_names=feature_names,
        perturbation_type=perturbation_type,
        perturbation_level=perturbation_level,
        clip_min=clip_min,
        clip_max=clip_max,
        seed=seed
    )
    
    return perturber.perturb(X)