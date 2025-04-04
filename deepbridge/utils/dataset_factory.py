"""
Factory module for creating DBDataset objects consistently.
This centralizes dataset creation to reduce code duplication.
"""

import typing as t
import pandas as pd
from pathlib import Path

from deepbridge.core.db_data import DBDataset

class DBDatasetFactory:
    """
    Factory class for creating DBDataset objects with consistent parameters.
    This class helps eliminate duplicate code for DBDataset creation across the library.
    """
    
    @staticmethod
    def create_from_model(
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        target_column: str,
        model: t.Any,
        categorical_features: t.Optional[t.List[str]] = None,
        dataset_name: t.Optional[str] = None,
        **kwargs
    ) -> DBDataset:
        """
        Create a DBDataset with model.
        
        Args:
            train_data: Training data DataFrame
            test_data: Test data DataFrame
            target_column: Name of the target column
            model: Model instance
            categorical_features: Optional list of categorical feature names
            dataset_name: Optional name for the dataset
            **kwargs: Additional arguments to pass to DBDataset constructor
            
        Returns:
            DBDataset instance
        """
        return DBDataset(
            train_data=train_data,
            test_data=test_data,
            target_column=target_column,
            model=model,
            categorical_features=categorical_features,
            dataset_name=dataset_name,
            **kwargs
        )
    
    @staticmethod
    def create_from_probabilities(
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        target_column: str,
        train_predictions: pd.DataFrame,
        test_predictions: t.Optional[pd.DataFrame] = None,
        prob_cols: t.Optional[t.List[str]] = None,
        categorical_features: t.Optional[t.List[str]] = None,
        dataset_name: t.Optional[str] = None,
        **kwargs
    ) -> DBDataset:
        """
        Create a DBDataset with probability predictions.
        
        Args:
            train_data: Training data DataFrame
            test_data: Test data DataFrame
            target_column: Name of the target column
            train_predictions: DataFrame with probability predictions for training data
            test_predictions: Optional DataFrame with probability predictions for test data
            prob_cols: Optional list of probability column names
            categorical_features: Optional list of categorical feature names
            dataset_name: Optional name for the dataset
            **kwargs: Additional arguments to pass to DBDataset constructor
            
        Returns:
            DBDataset instance
        """
        return DBDataset(
            train_data=train_data,
            test_data=test_data,
            target_column=target_column,
            train_predictions=train_predictions,
            test_predictions=test_predictions,
            prob_cols=prob_cols,
            categorical_features=categorical_features,
            dataset_name=dataset_name,
            **kwargs
        )
    
    @staticmethod
    def create_for_alternative_model(
        original_dataset: DBDataset,
        model: t.Any,
        **kwargs
    ) -> DBDataset:
        """
        Create a DBDataset for an alternative model with the same data as the original dataset.
        
        Args:
            original_dataset: Original DBDataset instance
            model: Alternative model instance
            **kwargs: Additional arguments to pass to DBDataset constructor
            
        Returns:
            DBDataset instance
        """
        return DBDataset(
            train_data=original_dataset.train_data,
            test_data=original_dataset.test_data,
            target_column=original_dataset.target_name,
            model=model,
            categorical_features=(
                original_dataset.categorical_features 
                if hasattr(original_dataset, 'categorical_features') else None
            ),
            **kwargs
        )