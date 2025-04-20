import typing as t
import pandas as pd

class DataValidator:
    """Handles data validation operations for datasets."""
    
    @staticmethod
    def validate_data_input(
        data: t.Optional[t.Any],
        train_data: t.Optional[t.Any],
        test_data: t.Optional[t.Any],
        target_column: t.Optional[str]
    ) -> None:
        """
        Validate input data configurations.
        
        Args:
            data: Unified dataset (can be DataFrame or scikit-learn dataset)
            train_data: Training dataset
            test_data: Test dataset
            target_column: Name of the target column
        
        Raises:
            ValueError: If input data configuration is invalid
        """
        if data is not None and (train_data is not None or test_data is not None):
            raise ValueError("Cannot provide both data and train/test data. Choose one option.")
        
        if data is None and (train_data is None or test_data is None):
            raise ValueError("Must provide either data or both train_data and test_data")
        
        if target_column is None:
            # Allow None for scikit-learn datasets, as we'll extract it later
            if (isinstance(data, pd.DataFrame) or 
                isinstance(train_data, pd.DataFrame) or 
                isinstance(test_data, pd.DataFrame)):
                raise ValueError("target_column must be provided when using DataFrame inputs")

    @staticmethod
    def validate_features(
        features: t.Optional[t.List[str]],
        data: t.Any,
        target_column: str,
        prob_cols: t.Optional[t.List[str]] = None
    ) -> t.List[str]:
        """
        Validate and/or infer feature names from data.
        
        Args:
            features: List of feature names (can be None to infer)
            data: Input data (DataFrame or scikit-learn dataset)
            target_column: Name of target column
            prob_cols: List of probability column names to exclude
        
        Returns:
            List of validated feature names
        
        Raises:
            ValueError: If features are invalid or cannot be inferred
        """
        # Handle scikit-learn datasets
        if not isinstance(data, pd.DataFrame):
            if hasattr(data, 'feature_names'):
                if features is None:
                    return list(data.feature_names)
                else:
                    # Validate provided features against dataset's feature_names
                    dataset_features = set(data.feature_names)
                    missing_features = set(features) - dataset_features
                    if missing_features:
                        raise ValueError(f"Features {missing_features} not found in data")
                    return features
            
            # Try to convert to DataFrame for further processing
            try:
                if hasattr(data, 'data'):
                    # Convert scikit-learn dataset to DataFrame
                    if hasattr(data, 'feature_names'):
                        data = pd.DataFrame(data.data, columns=data.feature_names)
                    else:
                        data = pd.DataFrame(data.data)
                else:
                    # Convert other data types to DataFrame
                    data = pd.DataFrame(data)
            except Exception as e:
                raise ValueError(f"Cannot validate features: {str(e)}")
        
        # If features not provided, infer from data
        if features is None:
            columns_to_exclude = [target_column]
            if prob_cols:
                columns_to_exclude.extend(prob_cols)
            
            # Get all columns except target and prob_cols
            return [col for col in data.columns if col not in columns_to_exclude]
        
        # Validate provided features exist in data
        missing_features = set(features) - set(data.columns)
        if missing_features:
            raise ValueError(f"Features {missing_features} not found in data")
            
        return features