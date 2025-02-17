import typing as t
import pandas as pd

class DataValidator:
    """Handles data validation operations for datasets."""
    
    @staticmethod
    def validate_data_input(
        data: t.Optional[pd.DataFrame],
        train_data: t.Optional[pd.DataFrame],
        test_data: t.Optional[pd.DataFrame],
        target_column: str
    ) -> None:
        if data is not None and (train_data is not None or test_data is not None):
            raise ValueError("Cannot provide both data and train/test data. Choose one option.")
        
        if data is None and (train_data is None or test_data is None):
            raise ValueError("Must provide either data or both train_data and test_data")
        
        if target_column is None:
            raise ValueError("target_column must be provided")

    @staticmethod
    def validate_features(
        features: t.List[str],
        data: pd.DataFrame,
        target_column: str,
        prob_cols: t.Optional[t.List[str]] = None
    ) -> t.List[str]:
        if features is None:
            return [col for col in data.columns if col != target_column 
                    and (prob_cols is None or col not in prob_cols)]
        
        missing_features = set(features) - set(data.columns)
        if missing_features:
            raise ValueError(f"Features {missing_features} not found in data")
        return features