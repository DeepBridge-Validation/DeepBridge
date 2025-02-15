import typing as t
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import load

class DBDataset:
    """
    DBDataset wraps training and test datasets along with optional model and predictions.
    
    The MLDataset class provides a structured way to handle machine learning datasets,
    models, and predictions while maintaining data integrity and providing useful utilities.

    Attributes:
        train_data (pd.DataFrame): Training dataset containing features and target
        test_data (pd.DataFrame): Test dataset containing features and target
        target_column (str): Name of the target column in both datasets
        features (t.Optional[t.List[str]]): List of feature column names. If None, all columns except target will be used. Defaults to None.
        model_path (t.Optional[t.Union[str, Path]]): Path to the saved model file (joblib format). Defaults to None.
        predictions (t.Optional[t.Union[np.ndarray, pd.Series]]): Array or series containing model predictions for the test set. Defaults to None.
        categorical_features (t.Optional[t.List[str]]): List of categorical feature names. If None, will attempt to infer. Defaults to None.
        max_categories (t.Optional[int]): Maximum number of unique values for a column to be considered categorical. Defaults to None.
        dataset_name (t.Optional[str]): Name identifier for the dataset. Defaults to None.
    """

    def __init__(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        target_column: str,
        features: t.Optional[t.List[str]] = None,
        model_path: t.Optional[t.Union[str, Path]] = None,
        predictions: t.Optional[t.Union[np.ndarray, pd.Series]] = None,
        categorical_features: t.Optional[t.List[str]] = None,
        max_categories: t.Optional[int] = None,
        dataset_name: t.Optional[str] = None
    ):
        # Validate inputs
        if len(train_data) == 0 or len(test_data) == 0:
            raise ValueError("Training and test datasets cannot be empty")

        # Store original data
        self._train_data = train_data.copy()
        self._test_data = test_data.copy()
        self._target_column = target_column
        self._dataset_name = dataset_name

        # Validate target column exists in both datasets
        if target_column not in train_data.columns:
            raise ValueError(f"Target column '{target_column}' not found in training data")
        if target_column not in test_data.columns:
            raise ValueError(f"Target column '{target_column}' not found in test data")

        # Set features
        if features is None:
            self._features = [col for col in train_data.columns if col != target_column]
        else:
            missing_features = set(features) - set(train_data.columns)
            if missing_features:
                raise ValueError(f"Features {missing_features} not found in training data")
            self._features = features

        # Set categorical features
        if categorical_features is None:
            self._categorical_features = self._infer_categorical_features(
                max_categories=max_categories
            )
        else:
            invalid_features = set(categorical_features) - set(self._features)
            if invalid_features:
                raise ValueError(f"Categorical features {invalid_features} not found in features list")
            self._categorical_features = categorical_features

        # Load model if provided
        self._model = None
        if model_path is not None:
            try:
                self._model = load(model_path)
            except Exception as e:
                warnings.warn(f"Failed to load model from {model_path}: {str(e)}")

        # Set predictions
        self._predictions = None
        if predictions is not None:
            if len(predictions) != len(test_data):
                raise ValueError("Length of predictions must match length of test data")
            self._predictions = (
                predictions.values if isinstance(predictions, pd.Series) else predictions
            )

    def _infer_categorical_features(self, max_categories: t.Optional[int] = None) -> t.List[str]:
        """Infer categorical features based on data types and unique values.

        Parameters
        ----------
        max_categories : t.Optional[int]
            Maximum number of unique values for a column to be considered categorical

        Returns
        -------
        t.List[str]
            List of inferred categorical feature names
        """
        categorical_features = []
        
        for feature in self._features:
            # Check if feature is object type or has limited unique values
            is_object = self._train_data[feature].dtype == 'object'
            n_unique = self._train_data[feature].nunique()
            
            if is_object or (max_categories and n_unique <= max_categories):
                categorical_features.append(feature)
        
        if categorical_features:
            warnings.warn(
                f"Inferred {len(categorical_features)} categorical features: "
                f"{', '.join(categorical_features[:5])}"
                f"{' ...' if len(categorical_features) > 5 else ''}"
            )
        
        return categorical_features

    @property
    def train_data(self) -> pd.DataFrame:
        """Return the training dataset."""
        return self._train_data

    @property
    def test_data(self) -> pd.DataFrame:
        """Return the test dataset."""
        return self._test_data

    @property
    def features(self) -> t.List[str]:
        """Return list of feature names."""
        return self._features

    @property
    def categorical_features(self) -> t.List[str]:
        """Return list of categorical feature names."""
        return self._categorical_features

    @property
    def numerical_features(self) -> t.List[str]:
        """Return list of numerical feature names."""
        return [f for f in self._features if f not in self._categorical_features]

    @property
    def target(self) -> str:
        """Return name of target column."""
        return self._target_column

    @property
    def model(self) -> t.Any:
        """Return the loaded model if available."""
        return self._model

    @property
    def predictions(self) -> t.Optional[np.ndarray]:
        """Return predictions if available."""
        return self._predictions

    def get_feature_data(self, dataset: str = 'train') -> pd.DataFrame:
        """Get feature columns from specified dataset.

        Parameters
        ----------
        dataset : str
            Either 'train' or 'test'

        Returns
        -------
        pd.DataFrame
            DataFrame containing only feature columns
        """
        if dataset.lower() not in ['train', 'test']:
            raise ValueError("dataset must be either 'train' or 'test'")
        
        data = self._train_data if dataset.lower() == 'train' else self._test_data
        return data[self._features]

    def get_target_data(self, dataset: str = 'train') -> pd.Series:
        """Get target column from specified dataset.

        Parameters
        ----------
        dataset : str
            Either 'train' or 'test'

        Returns
        -------
        pd.Series
            Series containing target values
        """
        if dataset.lower() not in ['train', 'test']:
            raise ValueError("dataset must be either 'train' or 'test'")
        
        data = self._train_data if dataset.lower() == 'train' else self._test_data
        return data[self._target_column]

    def set_predictions(self, predictions: t.Union[np.ndarray, pd.Series]) -> None:
        """Set predictions for the test dataset.

        Parameters
        ----------
        predictions : Union[np.ndarray, pd.Series]
            Array or series containing predictions
        """
        if len(predictions) != len(self._test_data):
            raise ValueError("Length of predictions must match length of test data")
        
        self._predictions = (
            predictions.values if isinstance(predictions, pd.Series) else predictions
        )

    def set_model(self, model_path: t.Union[str, Path]) -> None:
        """Load and set a model from file.

        Parameters
        ----------
        model_path : Union[str, Path]
            Path to the saved model file (joblib format)
        """
        try:
            self._model = load(model_path)
        except Exception as e:
            raise ValueError(f"Failed to load model from {model_path}: {str(e)}")

    @classmethod
    def from_single_dataset(
        cls,
        data: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        random_state: int = None,
        **kwargs
    ) -> 'MLDataset':
        """Create MLDataset by splitting a single dataset into train and test.

        Parameters
        ----------
        data : pd.DataFrame
            Complete dataset to split
        target_column : str
            Name of the target column
        test_size : float
            Proportion of dataset to include in the test split
        random_state : int
            Random state for reproducibility
        **kwargs
            Additional arguments to pass to MLDataset constructor

        Returns
        -------
        MLDataset
            New MLDataset instance with split data
        """
        train_data, test_data = train_test_split(
            data,
            test_size=test_size,
            random_state=random_state
        )
        return cls(
            train_data=train_data,
            test_data=test_data,
            target_column=target_column,
            **kwargs
        )

    def __len__(self) -> int:
        """Return total number of samples (train + test)."""
        return len(self._train_data) + len(self._test_data)

    def __repr__(self) -> str:
        """Return string representation of the dataset."""
        name = f"'{self._dataset_name}' " if self._dataset_name else ""
        return (
            f"MLDataset({name}with {len(self._train_data)} training samples and "
            f"{len(self._test_data)} test samples)\n"
            f"Features: {len(self._features)} total "
            f"({len(self._categorical_features)} categorical, "
            f"{len(self.numerical_features)} numerical)\n"
            f"Target: '{self._target_column}'\n"
            f"Model: {'loaded' if self._model is not None else 'not loaded'}\n"
            f"Predictions: "
            f"{'available' if self._predictions is not None else 'not available'}"
        )