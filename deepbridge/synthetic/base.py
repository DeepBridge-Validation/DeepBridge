"""
Base classes for synthetic data generation.

This module provides the abstract base classes that define the interfaces
for all synthetic data generators in the DeepBridge library.
"""

import abc
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np

class BaseSyntheticGenerator(abc.ABC):
    """
    Abstract base class for all synthetic data generators.
    
    This class defines the common interface that all synthetic data generators
    must implement, ensuring consistent behavior across different generation approaches.
    """
    
    def __init__(
        self,
        random_state: Optional[int] = None,
        preserve_dtypes: bool = True,
        preserve_constraints: bool = True
    ):
        """
        Initialize the base synthetic generator.
        
        Args:
            random_state: Random seed for reproducibility
            preserve_dtypes: Whether to preserve data types of original data
            preserve_constraints: Whether to preserve constraints (e.g., min/max values)
        """
        self.random_state = random_state
        self.preserve_dtypes = preserve_dtypes
        self.preserve_constraints = preserve_constraints
        self.fitted = False
        self._metadata = {}
        self._column_stats = {}
    
    @abc.abstractmethod
    def fit(
        self, 
        data: pd.DataFrame, 
        target_column: Optional[str] = None,
        categorical_columns: Optional[List[str]] = None,
        numerical_columns: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
        **kwargs
    ) -> 'BaseSyntheticGenerator':
        """
        Fit the generator to the input data.
        
        Args:
            data: Input DataFrame
            target_column: Name of the target column
            categorical_columns: List of categorical columns
            numerical_columns: List of numerical columns
            metadata: Optional metadata dictionary with additional information
            **kwargs: Additional parameters specific to each generator
            
        Returns:
            self: The fitted generator instance
        """
        pass
    
    @abc.abstractmethod
    def generate(
        self, 
        num_samples: int, 
        conditions: Optional[Dict] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Generate synthetic data.
        
        Args:
            num_samples: Number of synthetic samples to generate
            conditions: Optional conditions to apply during generation
            **kwargs: Additional parameters specific to each generator
            
        Returns:
            DataFrame of generated synthetic data
        """
        pass
    
    def fit_generate(
        self,
        data: pd.DataFrame, 
        num_samples: int,
        target_column: Optional[str] = None,
        categorical_columns: Optional[List[str]] = None,
        numerical_columns: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
        conditions: Optional[Dict] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fit the generator and generate synthetic data in one step.
        
        Args:
            data: Input DataFrame
            num_samples: Number of synthetic samples to generate
            target_column: Name of the target column
            categorical_columns: List of categorical columns
            numerical_columns: List of numerical columns
            metadata: Optional metadata dictionary with additional information
            conditions: Optional conditions to apply during generation
            **kwargs: Additional parameters for both fit and generate
            
        Returns:
            DataFrame of generated synthetic data
        """
        self.fit(
            data=data,
            target_column=target_column,
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            metadata=metadata,
            **kwargs
        )
        
        return self.generate(
            num_samples=num_samples,
            conditions=conditions,
            **kwargs
        )
    
    def generate_adversarial(
        self, 
        model,
        num_samples: int,
        target_value: Optional[Any] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Generate synthetic adversarial data that performs poorly on the given model.
        
        Args:
            model: The model to generate adversarial examples for
            num_samples: Number of adversarial samples to generate
            target_value: Optional target value to fool the model into predicting
            **kwargs: Additional parameters for generation
            
        Returns:
            DataFrame of generated adversarial examples
        """
        raise NotImplementedError(
            "This synthetic generator does not support adversarial generation. "
            "Please use a generator that supports this feature."
        )
    
    def generate_novel(
        self, 
        num_samples: int,
        novelty_degree: float = 0.5,
        **kwargs
    ) -> pd.DataFrame:
        """
        Generate synthetic data that contains novel patterns not present in training data.
        
        Args:
            num_samples: Number of novel samples to generate
            novelty_degree: Degree of novelty (0.0 to 1.0, higher means more novel)
            **kwargs: Additional parameters for generation
            
        Returns:
            DataFrame of generated novel examples
        """
        raise NotImplementedError(
            "This synthetic generator does not support novel data generation. "
            "Please use a generator that supports this feature."
        )
    
    def save(self, path: str) -> None:
        """
        Save the synthetic generator to a file.
        
        Args:
            path: Path to save the generator
        """
        import joblib
        joblib.dump(self, path)
    
    @classmethod
    def load(cls, path: str) -> 'BaseSyntheticGenerator':
        """
        Load a synthetic generator from a file.
        
        Args:
            path: Path to the saved generator
            
        Returns:
            Loaded synthetic generator
        """
        import joblib
        return joblib.load(path)
    
    def _validate_columns(
        self,
        data: pd.DataFrame,
        categorical_columns: Optional[List[str]] = None,
        numerical_columns: Optional[List[str]] = None
    ) -> tuple:
        """
        Validate and infer categorical and numerical columns.
        
        Args:
            data: Input DataFrame
            categorical_columns: List of categorical columns
            numerical_columns: List of numerical columns
            
        Returns:
            Tuple of (categorical_columns, numerical_columns)
        """
        # If both are None, infer from data
        if categorical_columns is None and numerical_columns is None:
            categorical_cols = []
            numerical_cols = []
            
            for col in data.columns:
                if data[col].dtype == 'object' or data[col].dtype == 'category' or (data[col].dtype == 'int64' and data[col].nunique() < 20):
                    categorical_cols.append(col)
                else:
                    numerical_cols.append(col)
            
            return categorical_cols, numerical_cols
        
        # If one is provided but not the other, infer the other
        if categorical_columns is not None and numerical_columns is None:
            numerical_cols = [col for col in data.columns if col not in categorical_columns]
            return categorical_columns, numerical_cols
        
        if numerical_columns is not None and categorical_columns is None:
            categorical_cols = [col for col in data.columns if col not in numerical_columns]
            return categorical_cols, numerical_columns
        
        # Both are provided, just return them
        return categorical_columns, numerical_columns
    
    def _collect_metadata(
        self,
        data: pd.DataFrame,
        categorical_columns: List[str],
        numerical_columns: List[str],
        target_column: Optional[str] = None
    ) -> Dict:
        """
        Collect metadata about the dataset for later use.
        
        Args:
            data: Input DataFrame
            categorical_columns: List of categorical columns
            numerical_columns: List of numerical columns
            target_column: Name of the target column
            
        Returns:
            Tuple of (metadata_dict, column_stats_dict)
        """
        metadata = {
            'n_samples': len(data),
            'columns': list(data.columns),
            'categorical_columns': categorical_columns,
            'numerical_columns': numerical_columns,
            'target_column': target_column,
            'dtypes': {col: str(data[col].dtype) for col in data.columns}
        }
        
        # Collect statistics for each column
        column_stats = {}
        
        # Numerical columns
        for col in numerical_columns:
            try:
                column_stats[col] = {
                    'mean': data[col].mean(),
                    'std': data[col].std(),
                    'min': data[col].min(),
                    'max': data[col].max(),
                    'median': data[col].median()
                }
            except (TypeError, ValueError) as e:
                # Handle columns that might look numerical but contain non-numeric data
                print(f"Warning: Could not calculate statistics for column '{col}'. It might contain non-numeric data. Error: {str(e)}")
                # Adding to categorical columns instead
                categorical_columns.append(col)
                if col in numerical_columns:
                    numerical_columns.remove(col)
        
        # Categorical columns
        for col in categorical_columns:
            try:
                value_counts = data[col].value_counts(normalize=True).to_dict()
                unique_values = list(data[col].unique())
                
                column_stats[col] = {
                    'unique_values': unique_values,
                    'value_counts': value_counts,
                    'most_common': data[col].mode()[0] if not data[col].mode().empty else None
                }
            except Exception as e:
                print(f"Warning: Error processing categorical column '{col}': {str(e)}")
                # Add minimal information
                column_stats[col] = {
                    'unique_values': [],
                    'value_counts': {},
                    'most_common': None
                }
        
        # Update metadata with potentially revised column categorizations
        metadata['categorical_columns'] = categorical_columns
        metadata['numerical_columns'] = numerical_columns
        
        return metadata, column_stats
    
    def _format_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Format the output DataFrame to match original data types.
        
        Args:
            df: DataFrame to format
            
        Returns:
            Formatted DataFrame
        """
        if not self.preserve_dtypes or not hasattr(self, '_metadata') or not self._metadata:
            return df
        
        result = df.copy()
        
        # Convert to original data types
        for col, dtype in self._metadata.get('dtypes', {}).items():
            if col in result.columns:
                try:
                    if 'int' in dtype and not 'interval' in dtype:
                        result[col] = result[col].round().astype(dtype)
                    elif 'float' in dtype:
                        result[col] = result[col].astype(dtype)
                    elif dtype == 'category':
                        result[col] = result[col].astype('category')
                    elif dtype == 'object':
                        result[col] = result[col].astype('object')
                except Exception:
                    # If conversion fails, keep current dtype
                    pass
        
        return result


class SequentialSyntheticGenerator(BaseSyntheticGenerator):
    """
    Base class for sequential synthetic data generators.
    Extends the base class with methods specific to sequential data.
    """
    
    def __init__(
        self,
        random_state: Optional[int] = None,
        preserve_dtypes: bool = True,
        preserve_constraints: bool = True,
        sequence_key: Optional[str] = None,
        sequence_index: Optional[str] = None
    ):
        """
        Initialize the sequential synthetic generator.
        
        Args:
            random_state: Random seed for reproducibility
            preserve_dtypes: Whether to preserve data types of original data
            preserve_constraints: Whether to preserve constraints (e.g., min/max values)
            sequence_key: Column name that identifies different sequences
            sequence_index: Column name that defines the order within a sequence
        """
        super().__init__(random_state, preserve_dtypes, preserve_constraints)
        self.sequence_key = sequence_key
        self.sequence_index = sequence_index
    
    @abc.abstractmethod
    def fit(
        self, 
        data: pd.DataFrame, 
        target_column: Optional[str] = None,
        categorical_columns: Optional[List[str]] = None,
        numerical_columns: Optional[List[str]] = None,
        sequence_key: Optional[str] = None,
        sequence_index: Optional[str] = None,
        context_columns: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
        **kwargs
    ) -> 'SequentialSyntheticGenerator':
        """
        Fit the generator to the sequential input data.
        
        Args:
            data: Input DataFrame
            target_column: Name of the target column
            categorical_columns: List of categorical columns
            numerical_columns: List of numerical columns
            sequence_key: Column name that identifies different sequences
            sequence_index: Column name that defines the order within a sequence
            context_columns: Columns that remain constant for a sequence
            metadata: Optional metadata dictionary with additional information
            **kwargs: Additional parameters specific to each generator
            
        Returns:
            self: The fitted generator instance
        """
        pass
    
    @abc.abstractmethod
    def generate(
        self, 
        num_sequences: int, 
        sequence_length: Optional[int] = None,
        context_values: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Generate synthetic sequential data.
        
        Args:
            num_sequences: Number of synthetic sequences to generate
            sequence_length: Length of each sequence (None for variable length)
            context_values: Optional DataFrame with context values for each sequence
            **kwargs: Additional parameters specific to each generator
            
        Returns:
            DataFrame of generated synthetic sequential data
        """
        pass
    
    def fit_generate(
        self,
        data: pd.DataFrame, 
        num_sequences: int,
        target_column: Optional[str] = None,
        categorical_columns: Optional[List[str]] = None,
        numerical_columns: Optional[List[str]] = None,
        sequence_key: Optional[str] = None,
        sequence_index: Optional[str] = None,
        sequence_length: Optional[int] = None,
        context_columns: Optional[List[str]] = None,
        context_values: Optional[pd.DataFrame] = None,
        metadata: Optional[Dict] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fit the generator and generate synthetic sequential data in one step.
        
        Args:
            data: Input DataFrame
            num_sequences: Number of synthetic sequences to generate
            target_column: Name of the target column
            categorical_columns: List of categorical columns
            numerical_columns: List of numerical columns
            sequence_key: Column name that identifies different sequences
            sequence_index: Column name that defines the order within a sequence
            sequence_length: Length of each sequence (None for variable length)
            context_columns: Columns that remain constant for a sequence
            context_values: Optional DataFrame with context values for each sequence
            metadata: Optional metadata dictionary with additional information
            **kwargs: Additional parameters for both fit and generate
            
        Returns:
            DataFrame of generated synthetic sequential data
        """
        self.fit(
            data=data,
            target_column=target_column,
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            sequence_key=sequence_key,
            sequence_index=sequence_index,
            context_columns=context_columns,
            metadata=metadata,
            **kwargs
        )
        
        return self.generate(
            num_sequences=num_sequences,
            sequence_length=sequence_length,
            context_values=context_values,
            **kwargs
        )