import pandas as pd
import numpy as np
from typing import Tuple, Optional

class DatabaseProbabilityManager:
    """
    Helper class to manage probability extraction from DBDataset objects.
    Ensures consistent probability handling throughout the distillation process.
    """
    
    def __init__(self, dataset: 'DBDataset', verbose: bool = True):
        """
        Initialize the probability manager.
        
        Args:
            dataset: DBDataset instance containing features, target, and probabilities
            verbose: Whether to print debug information
        """
        self.dataset = dataset
        self.verbose = verbose
        self._validate_dataset()
        
    def _validate_dataset(self):
        """Validate that the dataset has the required properties."""
        if not hasattr(self.dataset, 'original_prob') or self.dataset.original_prob is None:
            if self.verbose:
                print("WARNING: Dataset has no probability information")
            return
            
        if self.verbose:
            print(f"Dataset probability info - type: {type(self.dataset.original_prob)}")
            if isinstance(self.dataset.original_prob, pd.DataFrame):
                print(f"Probability columns: {self.dataset.original_prob.columns.tolist()}")
                print(f"First 3 rows of probabilities:\n{self.dataset.original_prob.head(3)}")
    
    def get_train_test_probabilities(self, test_indices=None):
        """
        Get the train and test probabilities from the dataset.
        
        Args:
            test_indices: Optional indices for test set if splitting is needed
            
        Returns:
            Tuple of (train_probabilities, test_probabilities)
        """
        if not hasattr(self.dataset, 'original_prob') or self.dataset.original_prob is None:
            if self.verbose:
                print("No probabilities available in dataset")
            return None, None
            
        probs = self.dataset.original_prob
        
        # If already split into train and test
        if hasattr(self.dataset, 'train_data') and hasattr(self.dataset, 'test_data'):
            if self.verbose:
                print("Using pre-split train/test probabilities")
            train_indices = self.dataset.train_data.index
            test_indices = self.dataset.test_data.index
            
            train_probs = probs.loc[train_indices]
            test_probs = probs.loc[test_indices]
            
        # Otherwise split using provided test_indices
        elif test_indices is not None:
            if self.verbose:
                print(f"Splitting probabilities using provided test indices (n={len(test_indices)})")
            train_indices = [i for i in range(len(probs)) if i not in test_indices]
            train_probs = probs.iloc[train_indices]
            test_probs = probs.iloc[test_indices]
        else:
            if self.verbose:
                print("No splitting information available, returning all probabilities as train")
            train_probs = probs
            test_probs = None
            
        if self.verbose:
            print(f"Train probabilities shape: {train_probs.shape if train_probs is not None else 'None'}")
            print(f"Test probabilities shape: {test_probs.shape if test_probs is not None else 'None'}")
            
        return train_probs, test_probs
    
    @staticmethod
    def extract_binary_probabilities(probs: pd.DataFrame) -> np.ndarray:
        """
        Extract binary class probabilities from DataFrame.
        
        Args:
            probs: DataFrame containing probability columns
            
        Returns:
            Numpy array with probabilities for both classes
        """
        if probs is None:
            return None
            
        # Check for standard prob_class_X format
        if 'prob_class_0' in probs.columns and 'prob_class_1' in probs.columns:
            return probs[['prob_class_0', 'prob_class_1']].values
            
        # Handle other common naming patterns
        if 'class_0_prob' in probs.columns and 'class_1_prob' in probs.columns:
            return probs[['class_0_prob', 'class_1_prob']].values
            
        # If columns have probability in name
        prob_cols = [c for c in probs.columns if 'prob' in c.lower()]
        if len(prob_cols) == 2:
            return probs[prob_cols].values
            
        # If one probability column, assume it's for positive class
        if len(prob_cols) == 1:
            pos_probs = probs[prob_cols[0]].values
            return np.column_stack([1 - pos_probs, pos_probs])
            
        # Last resort: use last two columns
        if probs.shape[1] >= 2:
            return probs.iloc[:, -2:].values
            
        # If only one column, assume it's probability of positive class
        if probs.shape[1] == 1:
            pos_probs = probs.iloc[:, 0].values
            return np.column_stack([1 - pos_probs, pos_probs])
            
        return None