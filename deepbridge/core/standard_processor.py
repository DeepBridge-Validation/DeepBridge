"""
Standard implementation of the BaseProcessor interface.
"""

import typing as t
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer

from .base_processor import BaseProcessor

class StandardProcessor(BaseProcessor):
    """
    Standard implementation of the BaseProcessor interface.
    Provides common data processing operations for numerical and categorical data.
    """
    
    def __init__(self, 
                verbose: bool = False,
                scaler_type: str = 'standard',
                handle_missing: bool = True,
                handle_outliers: bool = False,
                categorical_encoding: str = 'onehot'):
        """
        Initialize the standard processor.
        
        Args:
            verbose: Whether to print progress information
            scaler_type: Type of scaler to use ('standard', 'minmax', 'robust', or None)
            handle_missing: Whether to handle missing values
            handle_outliers: Whether to handle outliers
            categorical_encoding: Type of encoding for categorical features
                                 ('onehot', 'label', 'ordinal', or None)
        """
        super().__init__(verbose=verbose)
        self.scaler_type = scaler_type
        self.handle_missing = handle_missing
        self.handle_outliers = handle_outliers
        self.categorical_encoding = categorical_encoding
        
        # Initialize transformers
        self.scaler = None
        self.imputer = None
        self.encoders = {}
        
        # Initialize fitted state
        self.fitted = False
        
        # Store column information
        self.numerical_columns = None
        self.categorical_columns = None
        
        # Create the appropriate scaler
        if self.scaler_type:
            if scaler_type == 'standard':
                self.scaler = StandardScaler()
            elif scaler_type == 'minmax':
                self.scaler = MinMaxScaler()
            elif scaler_type == 'robust':
                self.scaler = RobustScaler()
    
    def process(self, data: t.Union[pd.DataFrame, np.ndarray], **kwargs) -> t.Union[pd.DataFrame, np.ndarray]:
        """
        Process the data.
        
        Args:
            data: Data to process
            **kwargs: Additional processing parameters
                - fit: Whether to fit transformers (default: False)
                - numerical_columns: List of numerical column names
                - categorical_columns: List of categorical column names
                - target_column: Target column name (excluded from processing)
                
        Returns:
            Processed data
        """
        # Convert array to DataFrame if needed
        array_input = False
        if isinstance(data, np.ndarray):
            array_input = True
            if data.ndim == 1:
                data = pd.DataFrame(data.reshape(-1, 1))
            else:
                data = pd.DataFrame(data)
        
        # Extract parameters
        fit = kwargs.get('fit', False)
        numerical_columns = kwargs.get('numerical_columns')
        categorical_columns = kwargs.get('categorical_columns')
        target_column = kwargs.get('target_column')
        
        # Get columns to process (exclude target)
        process_columns = list(data.columns)
        if target_column and target_column in process_columns:
            process_columns.remove(target_column)
        
        # Determine column types if not provided
        if numerical_columns is None or categorical_columns is None:
            self._infer_column_types(data, process_columns, target_column)
            numerical_columns = self.numerical_columns
            categorical_columns = self.categorical_columns
        else:
            self.numerical_columns = numerical_columns
            self.categorical_columns = categorical_columns
        
        # Create a copy of the data to avoid modifying the original
        result = data.copy()
        
        # Process numerical columns
        if numerical_columns:
            result = self._process_numerical(result, numerical_columns, fit)
        
        # Process categorical columns
        if categorical_columns:
            result = self._process_categorical(result, categorical_columns, fit)
        
        # Set fitted state
        if fit:
            self.fitted = True
        
        # Return as array if input was array
        if array_input:
            return result.values
        
        return result
    
    def _process_numerical(self, data: pd.DataFrame, columns: t.List[str], fit: bool) -> pd.DataFrame:
        """
        Process numerical columns.
        
        Args:
            data: DataFrame to process
            columns: Numerical column names
            fit: Whether to fit transformers
            
        Returns:
            Processed DataFrame
        """
        # Handle missing values if requested
        if self.handle_missing:
            # Initialize imputer if needed
            if fit or self.imputer is None:
                self.imputer = SimpleImputer(strategy='mean')
                self.imputer.fit(data[columns])
            
            # Apply imputation
            data.loc[:, columns] = self.imputer.transform(data[columns])
        
        # Handle outliers if requested
        if self.handle_outliers:
            for col in columns:
                if col in data.columns:
                    # Use IQR method
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # Cap outliers
                    data.loc[data[col] < lower_bound, col] = lower_bound
                    data.loc[data[col] > upper_bound, col] = upper_bound
        
        # Apply scaling if requested
        if self.scaler:
            if fit:
                self.scaler.fit(data[columns])
            
            # Apply scaling
            data.loc[:, columns] = self.scaler.transform(data[columns])
        
        return data
    
    def _process_categorical(self, data: pd.DataFrame, columns: t.List[str], fit: bool) -> pd.DataFrame:
        """
        Process categorical columns.
        
        Args:
            data: DataFrame to process
            columns: Categorical column names
            fit: Whether to fit encoders
            
        Returns:
            Processed DataFrame
        """
        # Skip if no encoding requested
        if not self.categorical_encoding:
            return data
        
        result = data.copy()
        
        # Process each categorical column
        for col in columns:
            if col not in data.columns:
                continue
                
            # Handle missing values (fill with most common value)
            if self.handle_missing and result[col].isna().any():
                most_common = result[col].mode()[0]
                result[col] = result[col].fillna(most_common)
            
            # Apply encoding
            if self.categorical_encoding == 'onehot':
                # One-hot encoding
                if fit or col not in self.encoders:
                    # Get unique values
                    unique_values = result[col].unique()
                    self.encoders[col] = {val: i for i, val in enumerate(unique_values)}
                
                # Create dummies
                dummies = pd.get_dummies(result[col], prefix=col)
                
                # Drop original column and add dummies
                result = pd.concat([result.drop(col, axis=1), dummies], axis=1)
                
            elif self.categorical_encoding == 'label':
                # Label encoding
                if fit or col not in self.encoders:
                    # Get unique values
                    unique_values = result[col].unique()
                    self.encoders[col] = {val: i for i, val in enumerate(unique_values)}
                
                # Apply encoding
                result[col] = result[col].map(self.encoders[col])
                
                # Handle values not seen during fit
                if result[col].isna().any():
                    result[col] = result[col].fillna(-1)
        
        return result
    
    def _infer_column_types(self, data: pd.DataFrame, columns: t.List[str], target_column: t.Optional[str] = None) -> None:
        """
        Infer column types from data.
        
        Args:
            data: DataFrame to analyze
            columns: Column names to process
            target_column: Target column name (excluded from processing)
        """
        # Initialize lists
        numerical_columns = []
        categorical_columns = []
        
        # Process each column
        for col in columns:
            if col == target_column:
                continue
                
            # Check column type
            dtype = data[col].dtype
            
            # Detect categorical columns
            if pd.api.types.is_categorical_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
                categorical_columns.append(col)
            # Detect boolean columns (treat as categorical)
            elif pd.api.types.is_bool_dtype(dtype):
                categorical_columns.append(col)
            # Detect numeric columns
            elif pd.api.types.is_numeric_dtype(dtype):
                # Check cardinality to detect categorical integers
                unique_count = data[col].nunique()
                if unique_count <= 10:  # Arbitrary threshold for categorical integers
                    categorical_columns.append(col)
                else:
                    numerical_columns.append(col)
            else:
                # Default to categorical for other types
                categorical_columns.append(col)
        
        # Store results
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        
        if self.verbose:
            print(f"Inferred {len(numerical_columns)} numerical columns and {len(categorical_columns)} categorical columns")
            print(f"Numerical columns: {numerical_columns}")
            print(f"Categorical columns: {categorical_columns}")