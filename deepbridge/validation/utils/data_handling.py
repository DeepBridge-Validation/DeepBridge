"""
Data handling utilities for the validation framework.

This module provides functions for loading, preprocessing, and
splitting datasets for validation tasks.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
import os
import pickle
import json


def load_dataset(
    source: Union[str, pd.DataFrame, Tuple],
    target_column: Optional[str] = None,
    file_format: Optional[str] = None,
    **kwargs
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Load a dataset from various sources.
    
    Parameters:
    -----------
    source : str, DataFrame, or tuple
        Source of the dataset:
        - str: Path to file or name of sklearn dataset
        - DataFrame: Already loaded DataFrame
        - tuple: (X, y) data
    target_column : str or None
        Name of target column in DataFrame or file
    file_format : str or None
        Format of the file ('csv', 'excel', 'pickle', etc.)
        If None, inferred from file extension
    **kwargs : dict
        Additional parameters for reading files
        
    Returns:
    --------
    tuple : (features_df, target_series)
    """
    # Handle various source types
    if isinstance(source, pd.DataFrame):
        # Source is already a DataFrame
        data = source
        
        if target_column is not None:
            # Split into features and target
            if target_column in data.columns:
                X = data.drop(columns=[target_column])
                y = data[target_column]
                return X, y
            else:
                raise ValueError(f"Target column '{target_column}' not found in DataFrame")
        else:
            # Return only features
            return data, None
            
    elif isinstance(source, tuple) and len(source) == 2:
        # Source is a tuple of (X, y)
        X, y = source
        
        # Convert X to DataFrame if it's not already
        if not isinstance(X, pd.DataFrame):
            if hasattr(X, 'feature_names'):
                # Handle scikit-learn datasets with feature_names attribute
                feature_names = X.feature_names
            else:
                # Create generic feature names
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
                
            X = pd.DataFrame(X, columns=feature_names)
            
        # Convert y to Series if it's not already
        if y is not None and not isinstance(y, pd.Series):
            if target_column is not None:
                y = pd.Series(y, name=target_column)
            else:
                y = pd.Series(y, name='target')
                
        return X, y
        
    elif isinstance(source, str):
        # Source is a file path or sklearn dataset name
        if os.path.exists(source):
            # Source is a file path
            return _load_from_file(source, target_column, file_format, **kwargs)
        else:
            # Try to load as a sklearn dataset
            return _load_from_sklearn(source, **kwargs)
    else:
        raise ValueError("Unsupported source type. Must be DataFrame, tuple of (X, y), or file path.")


def _load_from_file(
    file_path: str,
    target_column: Optional[str] = None,
    file_format: Optional[str] = None,
    **kwargs
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Load dataset from a file.
    
    Parameters:
    -----------
    file_path : str
        Path to the file
    target_column : str or None
        Name of target column
    file_format : str or None
        Format of the file
    **kwargs : dict
        Additional parameters for reading files
        
    Returns:
    --------
    tuple : (features_df, target_series)
    """
    # Infer file format from extension if not provided
    if file_format is None:
        _, ext = os.path.splitext(file_path)
        file_format = ext.lstrip('.').lower()
        
    # Load file based on format
    if file_format == 'csv':
        data = pd.read_csv(file_path, **kwargs)
    elif file_format in ['xls', 'xlsx', 'excel']:
        data = pd.read_excel(file_path, **kwargs)
    elif file_format in ['pickle', 'pkl']:
        data = pd.read_pickle(file_path, **kwargs)
    elif file_format in ['json']:
        data = pd.read_json(file_path, **kwargs)
    elif file_format in ['parquet']:
        data = pd.read_parquet(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")
        
    # Split into features and target if target column is provided
    if target_column is not None:
        if target_column in data.columns:
            X = data.drop(columns=[target_column])
            y = data[target_column]
            return X, y
        else:
            raise ValueError(f"Target column '{target_column}' not found in file")
    else:
        # Return only features
        return data, None


def _load_from_sklearn(
    dataset_name: str,
    **kwargs
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load a dataset from scikit-learn.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the sklearn dataset
    **kwargs : dict
        Additional parameters for loading
        
    Returns:
    --------
    tuple : (features_df, target_series)
    """
    try:
        from sklearn import datasets
        
        # Map dataset name to loader function
        dataset_loaders = {
            'iris': datasets.load_iris,
            'boston': datasets.load_boston,
            'diabetes': datasets.load_diabetes,
            'digits': datasets.load_digits,
            'wine': datasets.load_wine,
            'breast_cancer': datasets.load_breast_cancer
        }
        
        # Check if dataset exists
        if dataset_name.lower() in dataset_loaders:
            # Load dataset
            loader = dataset_loaders[dataset_name.lower()]
            dataset = loader(**kwargs)
            
            # Create DataFrame and Series
            X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
            y = pd.Series(dataset.target, name='target')
            
            return X, y
        else:
            raise ValueError(f"Unknown scikit-learn dataset: {dataset_name}")
            
    except ImportError:
        raise ImportError("scikit-learn is required to load sklearn datasets")


def split_dataset(
    X: Union[np.ndarray, pd.DataFrame],
    y: Optional[Union[np.ndarray, pd.Series]] = None,
    test_size: float = 0.2,
    val_size: Optional[float] = None,
    stratify: bool = True,
    random_state: Optional[int] = None,
    shuffle: bool = True
) -> Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Split dataset into train and test (and optionally validation) sets.
    
    Parameters:
    -----------
    X : array-like or DataFrame
        Features
    y : array-like or Series, optional
        Target variable for stratification
    test_size : float
        Proportion of data to use for testing
    val_size : float or None
        Proportion of data to use for validation
        If None, only train-test split is performed
    stratify : bool
        Whether to use stratified splitting
    random_state : int or None
        Random seed for reproducibility
    shuffle : bool
        Whether to shuffle data before splitting
        
    Returns:
    --------
    tuple : Train-test or train-validation-test split
    """
    # Convert to pandas objects if they're not already
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
        
    if y is not None and not isinstance(y, pd.Series):
        y = pd.Series(y)
        
    # Determine stratify parameter
    stratify_param = y if stratify and y is not None else None
    
    if val_size is None:
        # Simple train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_param,
            shuffle=shuffle
        )
        
        # Return split datasets
        if y is not None:
            return (X_train, y_train), (X_test, y_test)
        else:
            return X_train, X_test
    else:
        # Calculate test_proportion relative to full dataset
        # and validation_proportion relative to train set
        test_proportion = test_size
        validation_proportion = val_size / (1 - test_size)
        
        # First split into train+val and test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y,
            test_size=test_proportion,
            random_state=random_state,
            stratify=stratify_param,
            shuffle=shuffle
        )
        
        # Determine stratify parameter for second split
        stratify_param2 = y_trainval if stratify and y is not None else None
        
        # Split train+val into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=validation_proportion,
            random_state=random_state,
            stratify=stratify_param2,
            shuffle=shuffle
        )
        
        # Return split datasets
        if y is not None:
            return (X_train, y_train), (X_val, y_val), (X_test, y_test)
        else:
            return X_train, X_val, X_test


def stratified_kfold(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: Optional[int] = None
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create stratified K-fold cross-validation splits.
    
    Parameters:
    -----------
    X : array-like or DataFrame
        Features
    y : array-like or Series
        Target variable for stratification
    n_splits : int
        Number of folds
    shuffle : bool
        Whether to shuffle data before splitting
    random_state : int or None
        Random seed for reproducibility
        
    Returns:
    --------
    list : List of (train_indices, test_indices) tuples
    """
    # Check if stratification is possible
    if np.unique(y).size < 2:
        # If not enough classes for stratification, use regular KFold
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        return list(kf.split(X))
    else:
        # Use StratifiedKFold
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        return list(skf.split(X, y))


def preprocess_data(
    X: pd.DataFrame,
    categorical_features: Optional[List[str]] = None,
    numerical_features: Optional[List[str]] = None,
    scaling: Optional[str] = None,
    encoding: Optional[str] = None,
    handle_missing: Optional[str] = None
) -> pd.DataFrame:
    """
    Preprocess data for validation.
    
    Parameters:
    -----------
    X : DataFrame
        Features to preprocess
    categorical_features : list of str or None
        Names of categorical features
        If None, inferred from data types
    numerical_features : list of str or None
        Names of numerical features
        If None, inferred from data types
    scaling : str or None
        Scaling method for numerical features:
        - 'standard': StandardScaler
        - 'minmax': MinMaxScaler
        - 'robust': RobustScaler
        - None: No scaling
    encoding : str or None
        Encoding method for categorical features:
        - 'onehot': One-hot encoding
        - 'label': Label encoding
        - None: No encoding
    handle_missing : str or None
        Method for handling missing values:
        - 'drop': Drop rows with missing values
        - 'mean': Fill numeric with mean, categorical with mode
        - 'median': Fill numeric with median, categorical with mode
        - 'constant': Fill with specified constant
        - None: Keep missing values
        
    Returns:
    --------
    DataFrame : Preprocessed features
    """
    try:
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
        from sklearn.preprocessing import OneHotEncoder, LabelEncoder
        from sklearn.impute import SimpleImputer
    except ImportError:
        raise ImportError("scikit-learn is required for preprocessing")
        
    # Make a copy to avoid modifying the original
    X_processed = X.copy()
    
    # Infer feature types if not provided
    if categorical_features is None and numerical_features is None:
        # Get categorical features (object and category dtypes)
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Get numerical features (numeric dtypes)
        numerical_features = X.select_dtypes(include=['number']).columns.tolist()
    elif categorical_features is None:
        # Infer categorical features as non-numerical features
        categorical_features = [col for col in X.columns if col not in numerical_features]
    elif numerical_features is None:
        # Infer numerical features as non-categorical features
        numerical_features = [col for col in X.columns if col not in categorical_features]
    
    # Handle missing values
    if handle_missing is not None:
        X_processed = handle_missing_values(
            X_processed,
            method=handle_missing,
            categorical_features=categorical_features,
            numerical_features=numerical_features
        )
    
    # Scale numerical features
    if scaling is not None and numerical_features:
        if scaling == 'standard':
            scaler = StandardScaler()
        elif scaling == 'minmax':
            scaler = MinMaxScaler()
        elif scaling == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {scaling}")
            
        # Apply scaling
        if numerical_features:
            X_processed[numerical_features] = scaler.fit_transform(X_processed[numerical_features])
    
    # Encode categorical features
    if encoding is not None and categorical_features:
        if encoding == 'onehot':
            # Get dummies for all categorical features
            X_dummies = pd.get_dummies(
                X_processed[categorical_features],
                drop_first=False,
                dummy_na=False
            )
            
            # Remove original categorical columns and join with dummies
            X_processed = X_processed.drop(columns=categorical_features)
            X_processed = pd.concat([X_processed, X_dummies], axis=1)
            
        elif encoding == 'label':
            # Apply label encoding to each categorical feature
            for col in categorical_features:
                le = LabelEncoder()
                X_processed[col] = le.fit_transform(X_processed[col].astype(str))
        else:
            raise ValueError(f"Unknown encoding method: {encoding}")
    
    return X_processed


def handle_missing_values(
    X: pd.DataFrame,
    method: str = 'mean',
    categorical_features: Optional[List[str]] = None,
    numerical_features: Optional[List[str]] = None,
    fill_value: Any = None
) -> pd.DataFrame:
    """
    Handle missing values in dataset.
    
    Parameters:
    -----------
    X : DataFrame
        Features with missing values
    method : str
        Method for handling missing values:
        - 'drop': Drop rows with missing values
        - 'mean': Fill numeric with mean, categorical with mode
        - 'median': Fill numeric with median, categorical with mode
        - 'constant': Fill with specified constant
    categorical_features : list of str or None
        Names of categorical features
        If None, inferred from data types
    numerical_features : list of str or None
        Names of numerical features
        If None, inferred from data types
    fill_value : any
        Value to use for 'constant' method
        
    Returns:
    --------
    DataFrame : Features with handled missing values
    """
    # Make a copy to avoid modifying the original
    X_handled = X.copy()
    
    # Infer feature types if not provided
    if categorical_features is None and numerical_features is None:
        # Get categorical features (object and category dtypes)
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Get numerical features (numeric dtypes)
        numerical_features = X.select_dtypes(include=['number']).columns.tolist()
    elif categorical_features is None:
        # Infer categorical features as non-numerical features
        categorical_features = [col for col in X.columns if col not in numerical_features]
    elif numerical_features is None:
        # Infer numerical features as non-categorical features
        numerical_features = [col for col in X.columns if col not in categorical_features]
    
    if method == 'drop':
        # Drop rows with missing values
        X_handled = X_handled.dropna()
        
    elif method == 'mean':
        # Fill numerical features with mean
        for col in numerical_features:
            X_handled[col] = X_handled[col].fillna(X_handled[col].mean())
            
        # Fill categorical features with mode
        for col in categorical_features:
            mode_value = X_handled[col].mode().iloc[0] if not X_handled[col].mode().empty else 'missing'
            X_handled[col] = X_handled[col].fillna(mode_value)
            
    elif method == 'median':
        # Fill numerical features with median
        for col in numerical_features:
            X_handled[col] = X_handled[col].fillna(X_handled[col].median())
            
        # Fill categorical features with mode
        for col in categorical_features:
            mode_value = X_handled[col].mode().iloc[0] if not X_handled[col].mode().empty else 'missing'
            X_handled[col] = X_handled[col].fillna(mode_value)
            
    elif method == 'constant':
        # Fill all features with the same constant
        X_handled = X_handled.fillna(fill_value)
        
    else:
        raise ValueError(f"Unknown missing value handling method: {method}")
        
    return X_handled


def create_validation_set(
    X: Union[np.ndarray, pd.DataFrame],
    y: Optional[Union[np.ndarray, pd.Series]] = None,
    validation_size: float = 0.2,
    stratify: bool = True,
    random_state: Optional[int] = None
) -> Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]]:
    """
    Create a validation set from the dataset.
    
    Parameters:
    -----------
    X : array-like or DataFrame
        Features
    y : array-like or Series, optional
        Target variable for stratification
    validation_size : float
        Proportion of data to use for validation
    stratify : bool
        Whether to use stratified splitting
    random_state : int or None
        Random seed for reproducibility
        
    Returns:
    --------
    tuple : Training and validation data
    """
    # Determine stratify parameter
    stratify_param = y if stratify and y is not None else None
    
    # Split into train and validation sets
    if y is not None:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=validation_size,
            random_state=random_state,
            stratify=stratify_param
        )
        
        return X_train, y_train, X_val, y_val
    else:
        X_train, X_val = train_test_split(
            X,
            test_size=validation_size,
            random_state=random_state
        )
        
        return X_train, X_val