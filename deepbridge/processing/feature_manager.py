import typing as t
import warnings
import pandas as pd

class FeatureManager:
    """Manages feature-related operations."""
    
    def __init__(self, data: pd.DataFrame, features: t.List[str]):
        self._data = data
        self._features = features
        self._categorical_features = []
        
    def infer_categorical_features(self, max_categories: t.Optional[int] = None) -> t.List[str]:
        """Infer categorical features based on data types and unique values."""
        categorical_features = []
        
        for feature in self._features:
            is_object = self._data[feature].dtype == 'object'
            n_unique = self._data[feature].nunique()
            
            if is_object or (max_categories and n_unique <= max_categories):
                categorical_features.append(feature)
        
        if categorical_features:
            warnings.warn(
                f"Inferred {len(categorical_features)} categorical features: "
                f"{', '.join(categorical_features[:5])}"
                f"{' ...' if len(categorical_features) > 5 else ''}"
            )
        
        self._categorical_features = categorical_features
        return categorical_features
    
    @property
    def numerical_features(self) -> t.List[str]:
        """Return list of numerical feature names."""
        return [f for f in self._features if f not in self._categorical_features]
    
    @property
    def features(self) -> t.List[str]:
        """Return list of all features."""
        return self._features
    
    @property
    def categorical_features(self) -> t.List[str]:
        """Return list of categorical features."""
        return self._categorical_features