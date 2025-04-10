"""
Module for data perturbation methods used in robustness testing.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Union, Dict, Any

class DataPerturber:
    """
    Handles data perturbation for robustness testing.
    Extracted from RobustnessSuite to separate data perturbation responsibilities.
    """
    
    def __init__(self):
        """Initialize the data perturber."""
        self.rng = np.random.RandomState()
        
    def set_random_state(self, seed: Optional[int] = None) -> None:
        """Set random seed for reproducibility."""
        self.rng = np.random.RandomState(seed)
    
    def perturb_data(self, 
                     X: Union[pd.DataFrame, np.ndarray], 
                     perturb_method: str, 
                     level: float, 
                     perturb_features: Optional[List[str]] = None) -> Union[pd.DataFrame, np.ndarray]:
        """
        Perturb data using specified method and level.
        
        Parameters:
        -----------
        X : DataFrame or ndarray
            Feature data to perturb
        perturb_method : str
            Method to use ('raw' or 'quantile')
        level : float
            Level of perturbation to apply
        perturb_features : List[str] or None
            Specific features to perturb (None for all)
            
        Returns:
        --------
        DataFrame or ndarray : Perturbed data
        """
        # Create a copy of the original data
        X_perturbed = X.copy()
        
        # If perturb_features is None, perturb all features
        if perturb_features is None:
            perturb_features = X.columns if isinstance(X, pd.DataFrame) else range(X.shape[1])
        
        # Only perturb the specified features, keeping all others unchanged
        for feature in perturb_features:
            if isinstance(X, pd.DataFrame):
                # Skip if feature doesn't exist in the DataFrame
                if feature not in X.columns:
                    continue
                col = X.columns.get_loc(feature)
            else:
                # Skip if column index is out of bounds
                if isinstance(feature, int) and (feature < 0 or feature >= X.shape[1]):
                    continue
                col = feature
            
            if perturb_method == 'raw':
                self._apply_raw_perturbation(X, X_perturbed, col, level)
            elif perturb_method == 'quantile':
                self._apply_quantile_perturbation(X, X_perturbed, col, level)
            else:
                raise ValueError(f"Unknown perturbation method: {perturb_method}")
                
        return X_perturbed
    
    def _apply_raw_perturbation(self, 
                               X: Union[pd.DataFrame, np.ndarray], 
                               X_perturbed: Union[pd.DataFrame, np.ndarray], 
                               col: Union[int, str], 
                               level: float) -> None:
        """
        Apply Gaussian noise perturbation proportional to standard deviation.
        
        Parameters:
        -----------
        X : original data
        X_perturbed : data to be perturbed (modified in-place)
        col : column index or name
        level : perturbation level
        """
        # Apply Gaussian noise proportional to standard deviation
        feature_values = X.iloc[:, col] if isinstance(X, pd.DataFrame) else X[:, col]
        feature_std = np.std(feature_values)
        noise = self.rng.normal(0, level * feature_std, X_perturbed.shape[0])
        
        if isinstance(X, pd.DataFrame):
            # Get the data type of the original column
            col_dtype = X.iloc[:, col].dtype
            # Convert the noise to the appropriate type to avoid pandas dtype warning
            if pd.api.types.is_integer_dtype(col_dtype):
                # For integer columns, we need to convert the result to the same type
                X_perturbed.iloc[:, col] = (X_perturbed.iloc[:, col] + noise).astype(col_dtype)
            else:
                X_perturbed.iloc[:, col] += noise
        else:
            X_perturbed[:, col] += noise
                
    def _apply_quantile_perturbation(self, 
                                    X: Union[pd.DataFrame, np.ndarray], 
                                    X_perturbed: Union[pd.DataFrame, np.ndarray], 
                                    col: Union[int, str], 
                                    level: float) -> None:
        """
        Apply quantile-based perturbation.
        
        Parameters:
        -----------
        X : original data
        X_perturbed : data to be perturbed (modified in-place)
        col : column index or name
        level : perturbation level
        """
        # Apply quantile-based perturbation
        feature_values = X.iloc[:, col] if isinstance(X, pd.DataFrame) else X[:, col]
        quantiles = np.quantile(feature_values, [0.25, 0.75])
        perturbation = self.rng.uniform(
            quantiles[0] * (1 - level), 
            quantiles[1] * (1 + level), 
            X_perturbed.shape[0]
        )
        
        if isinstance(X, pd.DataFrame):
            # Get the data type of the original column
            col_dtype = X.iloc[:, col].dtype
            # Convert the perturbation to the appropriate type to avoid pandas dtype warning
            if pd.api.types.is_integer_dtype(col_dtype):
                # For integer columns, we need to convert the result to the same type
                X_perturbed.iloc[:, col] = perturbation.astype(col_dtype)
            else:
                X_perturbed.iloc[:, col] = perturbation
        else:
            X_perturbed[:, col] = perturbation
            
    def perturb_features_individually(self, 
                                     X: Union[pd.DataFrame, np.ndarray], 
                                     perturb_method: str, 
                                     level: float, 
                                     feature_subset: Optional[List[str]] = None) -> Dict[str, Union[pd.DataFrame, np.ndarray]]:
        """
        Perturb each feature individually.
        
        Parameters:
        -----------
        X : DataFrame or ndarray
            Feature data to perturb
        perturb_method : str
            Method to use ('raw' or 'quantile')
        level : float
            Level of perturbation to apply
        feature_subset : List[str] or None
            Specific features to perturb (None for all)
            When specified, only these features will be perturbed one at a time
            
        Returns:
        --------
        Dict[str, DataFrame or ndarray] : Dictionary mapping feature names to perturbed datasets
        """
        # Determine which features to perturb individually
        if feature_subset is None:
            features_to_perturb = X.columns if isinstance(X, pd.DataFrame) else range(X.shape[1])
        else:
            features_to_perturb = feature_subset
            
        perturbed_datasets = {}
        
        for feature in features_to_perturb:
            # Create a copy where only this one feature is perturbed
            perturbed_datasets[feature] = self.perturb_data(
                X, 
                perturb_method, 
                level, 
                [feature]  # Only perturb this specific feature
            )
            
        return perturbed_datasets