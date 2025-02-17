import typing as t
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from processing.data_validator import DataValidator
from processing.feature_manager import FeatureManager
from processing.model_handler import ModelHandler
from results.dataset_formatter import DatasetFormatter

class DBDataset:
    """
    DBDataset wraps training and test datasets along with optional model and predictions.
    """
    
    def __init__(
        self,
        data: t.Optional[pd.DataFrame] = None,
        train_data: t.Optional[pd.DataFrame] = None,
        test_data: t.Optional[pd.DataFrame] = None,
        target_column: str = None,
        features: t.Optional[t.List[str]] = None,
        model_path: t.Optional[t.Union[str, Path]] = None,
        train_predictions: t.Optional[pd.DataFrame] = None,
        test_predictions: t.Optional[pd.DataFrame] = None,
        prob_cols: t.Optional[t.List[str]] = None,
        categorical_features: t.Optional[t.List[str]] = None,
        max_categories: t.Optional[int] = None,
        dataset_name: t.Optional[str] = None,
        test_size: float = 0.2,
        random_state: int = None
    ):
        # Initialize helper classes
        self._validator = DataValidator()
        self._model_handler = ModelHandler()
        
        # Validate input data
        self._validator.validate_data_input(data, train_data, test_data, target_column)
        
        # Process and store data
        if data is not None:
            self._process_unified_data(data, target_column, features, prob_cols, test_size)
        else:
            self._process_split_data(train_data, test_data, target_column, features, prob_cols)
        
        self._target_column = target_column
        self._dataset_name = dataset_name
        
        # Initialize feature manager and process features
        self._feature_manager = FeatureManager(self._data, self._features)
        self._categorical_features = (
            self._feature_manager.infer_categorical_features(max_categories)
            if categorical_features is None
            else self._validate_categorical_features(categorical_features)
        )
        
        # Load model if provided
        if model_path is not None:
            self._model_handler.load_model(
                model_path,
                features=self._features,
                data={'train': self._train_data, 'test': self._test_data}
            )
        
        self._formatter = DatasetFormatter(
            dataset_name=dataset_name,
            feature_manager=self._feature_manager,
            model_handler=self._model_handler,
            target_column=self._target_column
        )

        # Initialize model handler with predictions only if prob_cols provided
        if prob_cols is not None:
            self._model_handler.set_predictions(
                self._train_data,
                self._test_data,
                train_predictions,
                test_predictions,
                prob_cols
            )

    def _process_unified_data(
        self,
        data: pd.DataFrame,
        target_column: str,
        features: t.List[str],
        prob_cols: t.List[str],
        test_size: float
    ) -> None:
        """Process unified dataset."""
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        self._features = self._validator.validate_features(features, data, target_column, prob_cols)
        
        self._original_data = data.copy()
        self._data = data[self._features].copy()
        
        train_idx = int(len(data) * (1 - test_size))
        self._train_data = data.iloc[:train_idx].copy()
        self._test_data = data.iloc[train_idx:].copy()

    def _process_split_data(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        target_column: str,
        features: t.List[str],
        prob_cols: t.List[str]
    ) -> None:
        """Process split train/test datasets."""
        if len(train_data) == 0 or len(test_data) == 0:
            raise ValueError("Training and test datasets cannot be empty")
        
        for df, name in [(train_data, 'train'), (test_data, 'test')]:
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in {name} data")
        
        self._features = self._validator.validate_features(
            features, 
            pd.concat([train_data, test_data]), 
            target_column, 
            prob_cols
        )
        
        self._train_data = train_data.copy()
        self._test_data = test_data.copy()
        self._original_data = pd.concat([train_data, test_data], ignore_index=True)
        self._data = self._original_data[self._features].copy()

    def _validate_categorical_features(self, categorical_features: t.List[str]) -> t.List[str]:
        """Validate provided categorical features."""
        invalid_features = set(categorical_features) - set(self._features)
        if invalid_features:
            raise ValueError(f"Categorical features {invalid_features} not found in features list")
        return categorical_features


    @property
    def X(self) -> pd.DataFrame:
        """Return the unified dataset with only the selected features."""
        return self._data
    
    @property
    def original_prob(self) -> t.Optional[pd.DataFrame]:
        """Return predictions DataFrame if available."""
        return self._model_handler.predictions

    @property
    def target(self) -> pd.Series:
        """Return the target column values."""
        return self._original_data[self._target_column]

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
        return self._feature_manager.features

    @property
    def categorical_features(self) -> t.List[str]:
        """Return list of categorical feature names."""
        return self._feature_manager.categorical_features

    @property
    def numerical_features(self) -> t.List[str]:
        """Return list of numerical feature names."""
        return self._feature_manager.numerical_features

    @property
    def target_name(self) -> str:
        """Return name of target column."""
        return self._target_column

    @property
    def model(self) -> t.Any:
        """Return the loaded model if available."""
        return self._model_handler.model
    

    def get_feature_data(self, dataset: str = 'train') -> pd.DataFrame:
        """Get feature columns from specified dataset."""
        if dataset.lower() not in ['train', 'test']:
            raise ValueError("dataset must be either 'train' or 'test'")
        
        data = self._train_data if dataset.lower() == 'train' else self._test_data
        return data[self._features]

    def get_target_data(self, dataset: str = 'train') -> pd.Series:
        """Get target column from specified dataset."""
        if dataset.lower() not in ['train', 'test']:
            raise ValueError("dataset must be either 'train' or 'test'")
        
        data = self._train_data if dataset.lower() == 'train' else self._test_data
        return data[self._target_column]

    def set_model(self, model_path: t.Union[str, Path]) -> None:
        """Load and set a model from file."""
        self._model_handler.load_model(model_path)

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self._data)
    
    def __repr__(self) -> str:
        """Return string representation of the dataset."""
        return self._formatter.format_dataset_info(
            data=self._data if hasattr(self, '_data') else None,
            train_data=self._train_data,
            test_data=self._test_data
        )