"""
Base interface for synthetic data generators.
"""

import typing as t
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseGenerator(ABC):
    """
    Abstract base class for all synthetic data generators.
    Defines the common interface that all generators should implement.
    """

    def __init__(
        self,
        random_state: t.Optional[int] = None,
        verbose: bool = False,
        preserve_dtypes: bool = True,
    ):
        """
        Initialize the base generator.

        Args:
            random_state: Random seed for reproducibility
            verbose: Whether to print progress information
            preserve_dtypes: Whether to preserve original data types
        """
        self.random_state = random_state
        self.verbose = verbose
        self.preserve_dtypes = preserve_dtypes

        # Initialize random number generator
        self.rng = np.random.RandomState(random_state)

        # Store column information
        self.categorical_columns = None
        self.numerical_columns = None
        self.target_column = None

        # Store original data types
        self.dtypes = None

        # Fitted flag
        self.fitted = False

    @abstractmethod
    def fit(
        self,
        data: pd.DataFrame,
        target_column: t.Optional[str] = None,
        categorical_columns: t.Optional[t.List[str]] = None,
        numerical_columns: t.Optional[t.List[str]] = None,
        **kwargs,
    ) -> 'BaseGenerator':
        """
        Fit the generator to data.

        Args:
            data: Training data
            target_column: Name of the target column
            categorical_columns: List of categorical column names
            numerical_columns: List of numerical column names
            **kwargs: Additional fitting parameters

        Returns:
            self: Fitted generator
        """
        # Store column information
        self.target_column = target_column

        # Infer categorical and numerical columns if not provided
        if categorical_columns is None or numerical_columns is None:
            self._infer_column_types(data)

        if categorical_columns is not None:
            self.categorical_columns = categorical_columns

        if numerical_columns is not None:
            self.numerical_columns = numerical_columns

        # Store original data types
        if self.preserve_dtypes:
            self.dtypes = data.dtypes.to_dict()

        # Set fitted flag
        self.fitted = True

        return self

    @abstractmethod
    def generate(self, num_samples: int, **kwargs) -> pd.DataFrame:
        """
        Generate synthetic data.

        Args:
            num_samples: Number of samples to generate
            **kwargs: Additional generation parameters

        Returns:
            DataFrame of synthetic data
        """
        if not self.fitted:
            raise ValueError('Generator not fitted. Call fit() first.')

        # Subclasses should implement this method
        pass

    def _infer_column_types(self, data: pd.DataFrame) -> None:
        """
        Infer categorical and numerical columns from data.

        Args:
            data: DataFrame to analyze
        """
        # Initialize lists
        categorical_columns = []
        numerical_columns = []

        # Exclude target column if specified
        columns_to_check = [
            col for col in data.columns if col != self.target_column
        ]

        for column in columns_to_check:
            # Get column dtype
            dtype = data[column].dtype

            # Check if categorical
            if pd.api.types.is_categorical_dtype(
                dtype
            ) or pd.api.types.is_object_dtype(dtype):
                categorical_columns.append(column)
            # Check if boolean (treat as categorical)
            elif pd.api.types.is_bool_dtype(dtype):
                categorical_columns.append(column)
            # Check if numerical
            elif pd.api.types.is_numeric_dtype(dtype):
                numerical_columns.append(column)
            else:
                # Default to categorical for other types
                categorical_columns.append(column)

        # Store results
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns

        if self.verbose:
            print(
                f'Inferred {len(categorical_columns)} categorical columns and {len(numerical_columns)} numerical columns'
            )

    def _restore_dtypes(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Restore original data types to generated data.

        Args:
            data: Generated data

        Returns:
            DataFrame with restored data types
        """
        if not self.preserve_dtypes or self.dtypes is None:
            return data

        # Make a copy to avoid modifying the original
        result = data.copy()

        for column, dtype in self.dtypes.items():
            if column in result.columns:
                try:
                    # Handle categorical types specially
                    if pd.api.types.is_categorical_dtype(dtype):
                        result[column] = pd.Categorical(result[column])
                    else:
                        result[column] = result[column].astype(dtype)
                except:
                    # If conversion fails, keep the original type
                    if self.verbose:
                        print(
                            f"Could not convert column '{column}' to {dtype}"
                        )

        return result

    def log(self, message: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)
