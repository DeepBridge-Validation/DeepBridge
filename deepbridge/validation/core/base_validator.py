"""
Base validator module defining the foundational Validator class.

This module provides the abstract base class that all specific validators
should inherit from, ensuring a consistent interface across different
validation types.
"""

from typing import Dict, List, Optional, Union, Any, Tuple, Type
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class BaseValidator:
    """
    Abstract base class for all validators.
    
    This class defines the interface that all validators should implement
    and provides common utility methods for validation tasks.
    """
    
    def __init__(self, dataset=None, model=None):
        """
        Initialize the validator.
        
        Parameters:
        -----------
        dataset : Any or None
            Dataset containing training and test data
        model : Any or None
            Trained machine learning model
        """
        self.dataset = dataset
        self.model = model if model is not None else (dataset.model if dataset is not None else None)
        self.results = {}
    
    def _get_data(self, test_set: bool = True) -> Tuple:
        """
        Get data from the dataset.
        
        Parameters:
        -----------
        test_set : bool
            Whether to return test data (True) or training data (False)
            
        Returns:
        --------
        Tuple : Data tuple, typically (X, y)
        """
        if self.dataset is None:
            raise ValueError("Dataset must be provided to get data.")
            
        try:
            if hasattr(self.dataset, 'get_feature_data'):
                if test_set:
                    X = self.dataset.get_feature_data('test')
                    y = self.dataset.get_target_data('test')
                else:
                    X = self.dataset.get_feature_data('train')
                    y = self.dataset.get_target_data('train')
            else:
                # For non-standard datasets, try common patterns
                if test_set and hasattr(self.dataset, 'X_test'):
                    X = self.dataset.X_test
                    y = self.dataset.y_test if hasattr(self.dataset, 'y_test') else None
                elif not test_set and hasattr(self.dataset, 'X_train'):
                    X = self.dataset.X_train
                    y = self.dataset.y_train if hasattr(self.dataset, 'y_train') else None
                else:
                    # Assume dataset is a tuple (X, y)
                    X = self.dataset[0]
                    y = self.dataset[1] if len(self.dataset) > 1 else None
                    
            # Ensure X is a DataFrame if possible
            if not isinstance(X, pd.DataFrame) and hasattr(X, 'columns'):
                X = pd.DataFrame(X)
                
            # Convert y to numpy array if it's a pandas Series
            if hasattr(y, 'values'):
                y = y.values
                
            return X, y
            
        except Exception as e:
            raise ValueError(f"Error getting data from dataset: {str(e)}")
    
    def get_result(self, key: str) -> Any:
        """
        Get a stored validation result.
        
        Parameters:
        -----------
        key : str
            Key of the result to retrieve
            
        Returns:
        --------
        Any : The stored result
        """
        if key not in self.results:
            raise KeyError(f"No result found with key '{key}'")
        return self.results[key]
    
    def list_results(self) -> List[str]:
        """
        List all available results.
        
        Returns:
        --------
        List[str] : List of result keys
        """
        return list(self.results.keys())
    
    def validate(self) -> Dict:
        """
        Perform validation.
        
        This is an abstract method that should be implemented by subclasses.
        
        Returns:
        --------
        Dict : Validation results
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def visualize(self, result_key: Optional[str] = None, **kwargs) -> plt.Figure:
        """
        Visualize validation results.
        
        This is an abstract method that should be implemented by subclasses.
        
        Parameters:
        -----------
        result_key : str or None
            Key of the result to visualize (if None, use the most recent)
        **kwargs : dict
            Additional visualization parameters
            
        Returns:
        --------
        matplotlib.Figure : Visualization figure
        """
        raise NotImplementedError("Subclasses must implement this method")