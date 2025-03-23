from abc import ABC, abstractmethod
import typing as t
import pandas as pd
import numpy as np
from pathlib import Path
import gc

class BaseSyntheticGenerator(ABC):
    """
    Base abstract class for all synthetic data generation methods.
    
    This serves as the foundation for all data generation techniques
    and defines the common interface that all implementations must follow.
    """
    
    def __init__(
        self,
        random_state: t.Optional[int] = None,
        preserve_dtypes: bool = True,
        preserve_constraints: bool = True,
        verbose: bool = True
    ):
        """
        Initialize the base synthetic data generator.
        
        Args:
            random_state: Seed for random number generation to ensure reproducibility
            preserve_dtypes: Whether to preserve the original data types in synthetic data
            preserve_constraints: Whether to enforce constraints from original data (ranges, unique values, etc.)
            verbose: Whether to print progress and information during processing
        """
        self.random_state = random_state
        self.preserve_dtypes = preserve_dtypes
        self.preserve_constraints = preserve_constraints
        self.verbose = verbose
        self._is_fitted = False
        self.numerical_columns = []
        self.categorical_columns = []
        self.target_column = None
        
        # Set random seed if provided
        if random_state is not None:
            np.random.seed(random_state)
    
    @abstractmethod
    def fit(
        self,
        data: pd.DataFrame,
        target_column: t.Optional[str] = None,
        categorical_columns: t.Optional[t.List[str]] = None,
        numerical_columns: t.Optional[t.List[str]] = None,
        **kwargs
    ) -> None:
        """
        Fit the generator to the input data.
        
        Args:
            data: The dataset to fit the generator on
            target_column: The name of the target variable column
            categorical_columns: List of categorical column names
            numerical_columns: List of numerical column names
            **kwargs: Additional parameters specific to the implementation
        """
        pass
    
    @abstractmethod
    def generate(
        self, 
        num_samples: int = 1000,
        chunk_size: t.Optional[int] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Generate synthetic data based on the fitted model.
        
        Args:
            num_samples: Number of synthetic samples to generate
            chunk_size: Size of chunks to use when generating large datasets
            **kwargs: Additional parameters specific to the implementation
            
        Returns:
            DataFrame containing the generated synthetic data
        """
        pass
    
    def _validate_columns(
        self, 
        data: pd.DataFrame,
        categorical_columns: t.Optional[t.List[str]] = None,
        numerical_columns: t.Optional[t.List[str]] = None
    ) -> t.Tuple[t.List[str], t.List[str]]:
        """
        Validate and infer column types if not explicitly provided.
        
        Args:
            data: The dataset to validate
            categorical_columns: List of categorical column names
            numerical_columns: List of numerical column names
            
        Returns:
            Tuple of (categorical_columns, numerical_columns)
        """
        all_columns = data.columns.tolist()
        
        # If categorical columns are provided, validate them
        if categorical_columns is not None:
            invalid_cols = set(categorical_columns) - set(all_columns)
            if invalid_cols:
                raise ValueError(f"Categorical columns {invalid_cols} not found in data")
        else:
            # Infer categorical columns
            categorical_columns = []
            for col in all_columns:
                if data[col].dtype == 'object' or pd.api.types.is_categorical_dtype(data[col]):
                    categorical_columns.append(col)
                elif pd.api.types.is_bool_dtype(data[col]):
                    categorical_columns.append(col)
                elif pd.api.types.is_integer_dtype(data[col]) and data[col].nunique() < 20:
                    # Treat integers with few unique values as categorical
                    categorical_columns.append(col)
        
        # If numerical columns are provided, validate them
        if numerical_columns is not None:
            invalid_cols = set(numerical_columns) - set(all_columns)
            if invalid_cols:
                raise ValueError(f"Numerical columns {invalid_cols} not found in data")
        else:
            # Infer numerical columns (all columns that are not categorical)
            numerical_columns = [col for col in all_columns if col not in categorical_columns]
        
        return categorical_columns, numerical_columns
    
    def _memory_optimize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize memory usage of DataFrame by downcasting data types.
        
        Args:
            df: DataFrame to optimize
            
        Returns:
            Memory-optimized DataFrame
        """
        if not self.preserve_dtypes:
            # Downcast numeric columns
            for col in self.numerical_columns:
                if col in df.columns:
                    if pd.api.types.is_integer_dtype(df[col]):
                        df[col] = pd.to_numeric(df[col], downcast='integer')
                    elif pd.api.types.is_float_dtype(df[col]):
                        df[col] = pd.to_numeric(df[col], downcast='float')
            
            # Convert categorical columns to appropriate types
            for col in self.categorical_columns:
                if col in df.columns and not pd.api.types.is_categorical_dtype(df[col]):
                    df[col] = df[col].astype('category')
        
        return df
    
    def _enforce_constraints(
        self, 
        df: pd.DataFrame, 
        original_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Enforce constraints from original data to synthetic data.
        
        Args:
            df: Synthetic DataFrame to apply constraints to
            original_data: Original DataFrame to derive constraints from
            
        Returns:
            DataFrame with constraints enforced
        """
        if not self.preserve_constraints:
            return df
        
        # Handle value ranges for numerical columns
        for col in self.numerical_columns:
            if col in df.columns:
                min_val = original_data[col].min()
                max_val = original_data[col].max()
                
                # Clip values to enforce range constraints
                df[col] = df[col].clip(min_val, max_val)
        
        # Handle categorical values
        for col in self.categorical_columns:
            if col in df.columns:
                # Get allowed values from original data
                allowed_values = set(original_data[col].unique())
                
                # Map values outside of the allowed set to values within the set
                mask = ~df[col].isin(allowed_values)
                if mask.any():
                    # Replace with random values from allowed set
                    replacement_values = np.random.choice(
                        list(allowed_values), 
                        size=mask.sum(),
                        replace=True
                    )
                    df.loc[mask, col] = replacement_values
        
        return df
    
    def log(self, message: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)
            
    def __repr__(self) -> str:
        """Return string representation of the generator."""
        status = "fitted" if self._is_fitted else "not fitted"
        return f"{self.__class__.__name__}(random_state={self.random_state}, {status})"