import typing as t
from pathlib import Path

import numpy as np
import pandas as pd


def validate_dataset(data: pd.DataFrame) -> bool:
    """
    Validate that a dataset meets basic requirements.

    Args:
        data: DataFrame to validate

    Returns:
        True if valid, raises exception otherwise
    """
    # Check if data is empty
    if data is None or data.empty:
        raise ValueError('Dataset is empty or None')

    # Check for at least some rows
    if len(data) < 1:
        raise ValueError('Dataset must have at least one row')

    # Check for at least some columns
    if len(data.columns) < 1:
        raise ValueError('Dataset must have at least one column')

    # Check for duplicate column names
    if len(data.columns) != len(set(data.columns)):
        raise ValueError('Dataset contains duplicate column names')

    return True


def validate_columns(
    data: pd.DataFrame,
    numerical_columns: t.Optional[t.List[str]] = None,
    categorical_columns: t.Optional[t.List[str]] = None,
    target_column: t.Optional[str] = None,
) -> bool:
    """
    Validate that specified columns exist in the dataset.

    Args:
        data: DataFrame to validate
        numerical_columns: List of numerical column names
        categorical_columns: List of categorical column names
        target_column: Name of target column

    Returns:
        True if valid, raises exception otherwise
    """
    all_columns = data.columns.tolist()

    # Check numerical columns
    if numerical_columns:
        missing_num_cols = set(numerical_columns) - set(all_columns)
        if missing_num_cols:
            raise ValueError(
                f'Numerical columns not found in dataset: {missing_num_cols}'
            )

    # Check categorical columns
    if categorical_columns:
        missing_cat_cols = set(categorical_columns) - set(all_columns)
        if missing_cat_cols:
            raise ValueError(
                f'Categorical columns not found in dataset: {missing_cat_cols}'
            )

    # Check target column
    if target_column and target_column not in all_columns:
        raise ValueError(
            f"Target column '{target_column}' not found in dataset"
        )

    # Check for overlap between numerical and categorical columns
    if numerical_columns and categorical_columns:
        overlap = set(numerical_columns) & set(categorical_columns)
        if overlap:
            raise ValueError(
                f'Columns cannot be both numerical and categorical: {overlap}'
            )

    return True


def validate_generator_params(method: str, num_samples: int, **kwargs) -> bool:
    """
    Validate parameters for synthetic data generation.

    Args:
        method: Generation method
        num_samples: Number of samples to generate
        **kwargs: Additional parameters

    Returns:
        True if valid, raises exception otherwise
    """
    # Validate method
    valid_methods = ['gaussian', 'ctgan', 'tvae', 'smote']
    if method not in valid_methods:
        raise ValueError(
            f'Unknown method: {method}. Valid methods are: {valid_methods}'
        )

    # Validate num_samples
    if num_samples <= 0:
        raise ValueError('Number of samples must be positive')

    # Validate specific method parameters
    if method == 'gaussian':
        # No specific requirements for gaussian method
        pass
    elif method == 'ctgan':
        # Validate CTGAN-specific parameters (to be implemented)
        # For future implementation
        raise NotImplementedError('CTGAN method is not yet implemented')
    elif method == 'tvae':
        # Validate TVAE-specific parameters (to be implemented)
        # For future implementation
        raise NotImplementedError('TVAE method is not yet implemented')
    elif method == 'smote':
        # Validate SMOTE-specific parameters (to be implemented)
        # For future implementation
        raise NotImplementedError('SMOTE method is not yet implemented')

    return True


def validate_file_path(
    path: t.Union[str, Path], must_exist: bool = False
) -> Path:
    """
    Validate a file path.

    Args:
        path: Path to validate
        must_exist: Whether the file must already exist

    Returns:
        Path object if valid, raises exception otherwise
    """
    path_obj = Path(path)

    # Check if file must exist
    if must_exist and not path_obj.exists():
        raise FileNotFoundError(f'File not found: {path}')

    # Check if parent directory exists
    if not path_obj.parent.exists():
        raise ValueError(f'Parent directory does not exist: {path_obj.parent}')

    return path_obj
