"""
Base interface for data processing components.
"""

import typing as t
import pandas as pd
from abc import ABC, abstractmethod

class BaseProcessor(ABC):
    """
    Abstract base class for all data processing components.
    Defines the common interface that all processors should implement.
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the base processor.
        
        Args:
            verbose: Whether to print progress information
        """
        self.verbose = verbose
    
    @abstractmethod
    def process(self, data: t.Any, **kwargs) -> t.Any:
        """
        Process the data.
        
        Args:
            data: Data to process
            **kwargs: Additional processing parameters
            
        Returns:
            Processed data
        """
        pass
    
    def log(self, message: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def validate_dataframe(self, 
                         df: pd.DataFrame, 
                         required_columns: t.Optional[t.List[str]] = None,
                         required_types: t.Optional[t.Dict[str, t.Type]] = None) -> bool:
        """
        Validate a DataFrame against required columns and types.
        
        Args:
            df: DataFrame to validate
            required_columns: Optional list of required column names
            required_types: Optional dict mapping column names to expected types
            
        Returns:
            bool: Whether the DataFrame is valid
        
        Raises:
            ValueError: If the DataFrame is invalid
        """
        # Check for required columns
        if required_columns:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for required types
        if required_types:
            for col, expected_type in required_types.items():
                if col not in df.columns:
                    continue
                    
                actual_type = df[col].dtype
                # Check if the column type is compatible with expected type
                if not self._is_compatible_dtype(actual_type, expected_type):
                    raise ValueError(f"Column '{col}' has type {actual_type}, expected {expected_type}")
        
        return True
    
    def _is_compatible_dtype(self, actual_dtype, expected_type):
        """Check if a pandas dtype is compatible with an expected type."""
        # Convert pandas dtype to Python type as needed
        if expected_type == float:
            return pd.api.types.is_float_dtype(actual_dtype)
        elif expected_type == int:
            return pd.api.types.is_integer_dtype(actual_dtype)
        elif expected_type == bool:
            return pd.api.types.is_bool_dtype(actual_dtype)
        elif expected_type == str:
            return pd.api.types.is_string_dtype(actual_dtype) or pd.api.types.is_object_dtype(actual_dtype)
        elif expected_type == 'categorical':
            return pd.api.types.is_categorical_dtype(actual_dtype)
        else:
            # For more complex types, use isinstance check
            return True