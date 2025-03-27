"""
Base wrapper module for dataset-based validation tests.

This module provides the base class for all validation test wrappers
that work with DBDataset objects, abstracting common functionality
and providing a consistent interface.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
import matplotlib.pyplot as plt

class BaseWrapper:
    """
    Base class for all validation test wrappers that use DBDataset.
    
    This class provides common functionality for working with the DBDataset
    class, automatically extracting model, data, and problem type information.
    All specific test wrapper classes should inherit from this base class.
    """
    
    def __init__(self, dataset, verbose: bool = False):
        """
        Initialize the base wrapper with a DBDataset object.
        
        Parameters:
        -----------
        dataset : DBDataset
            Dataset object containing training/test data and model
        verbose : bool
            Whether to print progress information
        """
        if dataset is None:
            raise ValueError("Dataset cannot be None")
            
        self.dataset = dataset
        self.verbose = verbose
        
        # Extract model from dataset
        self.model = self.dataset.model
        if self.model is None:
            raise ValueError("Dataset must have a model")
            
        # Detect problem type
        self._problem_type = self._detect_problem_type()
        
        # Store features for easier access
        self._features = self.dataset.features
        
        if self.verbose:
            print(f"Initialized {self.__class__.__name__} with {self._problem_type} problem type")
            print(f"Dataset has {len(self._features)} features")
    
    def _detect_problem_type(self) -> str:
        """
        Detect if the problem is classification or regression.
        
        This method analyzes the target values to determine the type
        of problem we're dealing with (classification or regression).
        
        Returns:
        --------
        str : 'classification' or 'regression'
        """
        try:
            # Obter o nome da coluna target do DBDataset
            target_name = self.dataset.target_name
            
            # Verificar se existe um hint no dataset sobre o tipo do problema
            if hasattr(self.dataset, 'problem_type') and self.dataset.problem_type is not None:
                if self.verbose:
                    print(f"Using problem type from dataset: {self.dataset.problem_type}")
                return self.dataset.problem_type
            
            # Obter valores target
            y = self._get_target_data()
            y_values = y.values if hasattr(y, 'values') else y
            
            # Verificar se target é categórico
            unique_values = np.unique(y_values)
            n_unique = len(unique_values)
            
            # Verificações explícitas para classificação - mais rígida
            if n_unique <= 2:
                # Binário é quase sempre classificação
                if self.verbose:
                    print(f"Detected binary classification problem with {n_unique} unique values: {unique_values}")
                return 'classification'
            elif n_unique <= 10:
                # Verificar se todos os valores são inteiros
                is_all_integer = all(isinstance(val, (int, bool)) or 
                                (isinstance(val, (float, np.float64)) and val.is_integer()) 
                                for val in unique_values)
                
                if is_all_integer:
                    if self.verbose:
                        print(f"Detected multi-class classification with {n_unique} unique integer values")
                    return 'classification'
            
            # Se chegou até aqui, provavelmente é regressão
            if self.verbose:
                print(f"Detected regression problem with {n_unique} unique values")
            return 'regression'
        
        except Exception as e:
            if self.verbose:
                print(f"Error detecting problem type: {str(e)}. Using explicit 'classification' as default")
            return 'classification'  # Forçar classificação como default
    
    def _get_feature_data(self, dataset_type: str = 'test') -> pd.DataFrame:
        """
        Get feature data from the dataset.
        
        Parameters:
        -----------
        dataset_type : str
            Type of dataset to get ('train' or 'test')
            
        Returns:
        --------
        pandas.DataFrame : Feature data
        """
        if dataset_type not in ['train', 'test']:
            raise ValueError("dataset_type must be either 'train' or 'test'")
            
        try:
            return self.dataset.get_feature_data(dataset_type)
        except Exception as e:
            raise ValueError(f"Error getting feature data: {str(e)}")
    
    def _get_target_data(self, dataset_type: str = 'test') -> pd.Series:
        """
        Get target data from the dataset.
        
        Parameters:
        -----------
        dataset_type : str
            Type of dataset to get ('train' or 'test')
            
        Returns:
        --------
        pandas.Series : Target data
        """
        if dataset_type not in ['train', 'test']:
            raise ValueError("dataset_type deve ser 'train' ou 'test'")
            
        try:
            # Usar o método correto do DBDataset para obter dados do target
            return self.dataset.get_target_data(dataset_type)
        except Exception as e:
            if self.verbose:
                print(f"Erro target use get_target_data(): {str(e)}")
                print("Tentando método alternativo...")
            
            try:
                # Método alternativo: acessar através do atributo target
                if dataset_type == 'train':
                    return self.dataset.train_data[self.dataset.target_name]
                else:
                    return self.dataset.test_data[self.dataset.target_name]
            except Exception as e2:
                raise ValueError(f"Erro  target: {str(e2)}")
    
    def _validate_feature(self, feature_name: Optional[str] = None) -> str:
        """
        Validate feature name and return it.
        
        If feature_name is None, return the first feature.
        
        Parameters:
        -----------
        feature_name : str or None
            Name of the feature to validate
            
        Returns:
        --------
        str : Validated feature name
        """
        if feature_name is None:
            # Use first feature as default
            return self._features[0]
            
        if feature_name not in self._features:
            raise ValueError(f"Feature '{feature_name}' not found in dataset. "
                           f"Available features: {self._features}")
            
        return feature_name
    
    def _validate_data_consistency(self) -> None:
        """
        Validate that data in the dataset is consistent.
        
        This method checks if feature data and target data have the same
        number of samples, to catch potential issues early.
        """
        X_train = self._get_feature_data('train')
        y_train = self._get_target_data('train')
        
        if len(X_train) != len(y_train):
            raise ValueError(f"Training feature data ({len(X_train)} samples) and "
                           f"target data ({len(y_train)} samples) have different lengths")
                           
        X_test = self._get_feature_data('test')
        y_test = self._get_target_data('test')
        
        if len(X_test) != len(y_test):
            raise ValueError(f"Test feature data ({len(X_test)} samples) and "
                           f"target data ({len(y_test)} samples) have different lengths")
    
    def _validate_model(self) -> None:
        """
        Validate that the model has the required methods.
        
        This method checks if the model has the necessary methods like
        predict(), predict_proba() (for classifiers), etc.
        """
        if not hasattr(self.model, 'predict'):
            raise ValueError("Model must have a predict method")
            
        # For classification models, check if predict_proba is available
        if self._problem_type == 'classification':
            has_proba = hasattr(self.model, 'predict_proba')
            if self.verbose and not has_proba:
                print("Warning: Classification model doesn't have predict_proba method. "
                     "Some tests may not work as expected.")
    
    def is_classification(self) -> bool:
        """
        Check if the problem is classification.
        
        Returns:
        --------
        bool : True if classification, False otherwise
        """
        return self._problem_type == 'classification'
    
    def is_regression(self) -> bool:
        """
        Check if the problem is regression.
        
        Returns:
        --------
        bool : True if regression, False otherwise
        """
        return self._problem_type == 'regression'
    
    def get_problem_type(self) -> str:
        """
        Get the detected problem type.
        
        Returns:
        --------
        str : 'classification' or 'regression'
        """
        return self._problem_type
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names.
        
        Returns:
        --------
        list of str : Feature names
        """
        return self._features.copy()
    
    def get_categorical_features(self) -> List[str]:
        """
        Get list of categorical feature names.
        
        Returns:
        --------
        list of str : Categorical feature names
        """
        if hasattr(self.dataset, 'categorical_features'):
            return self.dataset.categorical_features
        else:
            # Try to infer categorical features
            X = self._get_feature_data()
            return [f for f in self._features if len(X[f].unique()) < 10]
    
    def get_numerical_features(self) -> List[str]:
        """
        Get list of numerical feature names.
        
        Returns:
        --------
        list of str : Numerical feature names
        """
        if hasattr(self.dataset, 'numerical_features'):
            return self.dataset.numerical_features
        else:
            # Subtract categorical features from all features
            categorical = set(self.get_categorical_features())
            return [f for f in self._features if f not in categorical]
    
    def summary(self) -> Dict[str, Any]:
        """
        Get a summary of the dataset and problem.
        
        Returns:
        --------
        dict : Summary information
        """
        X_train = self._get_feature_data('train')
        X_test = self._get_feature_data('test')
        
        return {
            'problem_type': self._problem_type,
            'n_features': len(self._features),
            'categorical_features': self.get_categorical_features(),
            'numerical_features': self.get_numerical_features(),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'model_type': type(self.model).__name__
        }