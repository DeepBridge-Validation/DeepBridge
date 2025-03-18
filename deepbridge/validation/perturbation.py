import numpy as np
import pandas as pd
from typing import List, Union, Optional
from abc import ABC, abstractmethod

class BasePerturbation(ABC):
    """
    Abstract base class for different perturbation methods.
    
    This class defines the common interface that all perturbation
    methods should implement for consistent usage.
    """
    
    @abstractmethod
    def perturb(self, X, perturb_size, **kwargs):
        """
        Apply perturbation to the input data.
        
        Args:
            X: Input data to perturb
            perturb_size: Magnitude of perturbation to apply
            **kwargs: Additional method-specific parameters
            
        Returns:
            Perturbed data
        """
        pass

class RawPerturbation(BasePerturbation):
    """
    Applies raw perturbation (adds Gaussian noise) to data.
    
    This method adds noise proportional to the variance of each feature,
    scaled by the perturbation size parameter.
    """
    
    def perturb(self, X: np.ndarray, perturb_size: float, **kwargs) -> np.ndarray:
        """
        Apply raw Gaussian noise perturbation to input data.
        
        Args:
            X: Matrix of features
            perturb_size: Size of perturbation (λ)
            **kwargs: Additional parameters (not used)
            
        Returns:
            np.ndarray: Perturbed data
        """
        # Calculate variance of each feature
        variances = np.var(X, axis=0)
        
        # Generate Gaussian noise proportional to variance
        noise = np.random.normal(0, np.sqrt(perturb_size * variances), size=X.shape)
        
        # Add noise to original data
        X_perturbed = X + noise
        
        return X_perturbed

class QuantilePerturbation(BasePerturbation):
    """
    Applies perturbation based on quantiles to the data.
    
    This method transforms data to quantile space, adds uniform noise,
    and transforms back, preserving the original distribution shape.
    """
    
    def perturb(self, X: np.ndarray, perturb_size: float, **kwargs) -> np.ndarray:
        """
        Apply quantile-based perturbation to input data.
        
        Args:
            X: Matrix of features
            perturb_size: Size of perturbation (λ)
            **kwargs: Additional parameters (not used)
            
        Returns:
            np.ndarray: Perturbed data
        """
        X_perturbed = X.copy()
        
        # For each feature
        for j in range(X.shape[1]):
            x = X[:, j]
            
            # If the feature has enough distinct values to make sense using quantiles
            unique_vals = np.unique(x)
            if len(unique_vals) > 5:
                # Convert to quantile space
                quantiles = np.zeros(len(x))
                for i, val in enumerate(x):
                    quantiles[i] = np.mean(x <= val)
                
                # Add uniform noise to quantiles
                perturbed_quantiles = np.clip(
                    quantiles + np.random.uniform(-0.5 * perturb_size, 0.5 * perturb_size, size=len(quantiles)),
                    0, 1
                )
                
                # Convert back to original space
                for i in range(len(x)):
                    # Find the value corresponding to the perturbed quantile
                    X_perturbed[i, j] = np.quantile(x, perturbed_quantiles[i])
            else:
                # For discrete data with few values, keep the original value
                X_perturbed[:, j] = x
        
        return X_perturbed

class CategoricalPerturbation(BasePerturbation):
    """
    Applies perturbation to categorical variables.
    
    This method randomly changes categorical values according to
    their frequency distribution with a probability based on the
    perturbation size.
    """
    
    def perturb(
        self, 
        X: pd.DataFrame, 
        cat_columns: List[str], 
        perturb_size: float, 
        **kwargs
    ) -> pd.DataFrame:
        """
        Apply perturbation to categorical variables.
        
        Args:
            X: DataFrame with the data
            cat_columns: List of categorical columns
            perturb_size: Probability of perturbing each sample
            **kwargs: Additional parameters (not used)
            
        Returns:
            pd.DataFrame: Data with perturbed categories
        """
        X_perturbed = X.copy()
        
        for col in cat_columns:
            # Calculate frequency of each category
            value_counts = X[col].value_counts(normalize=True)
            
            # For each sample
            for idx in X.index:
                # Decide if we'll perturb this sample
                if np.random.random() < perturb_size:
                    # If perturbing, choose a category according to frequencies
                    X_perturbed.at[idx, col] = np.random.choice(
                        value_counts.index,
                        p=value_counts.values
                    )
        
        return X_perturbed