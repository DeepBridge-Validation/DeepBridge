import typing as t
import pandas as pd

class DatasetFormatter:
    """
    Handles the string representation formatting for DBDataset.
    Responsible for creating human-readable string representations of dataset information.
    """
    def __init__(
        self,
        dataset_name: t.Optional[str],
        feature_manager: t.Any,
        model_handler: t.Any,
        target_column: str
    ):
        """
        Initialize the DatasetFormatter.

        Args:
            dataset_name: Optional name of the dataset
            feature_manager: Instance managing feature information
            model_handler: Instance managing model and predictions
            target_column: Name of the target column
        """
        self._dataset_name = dataset_name
        self._feature_manager = feature_manager
        self._model_handler = model_handler
        self._target_column = target_column

    def format_dataset_info(
        self,
        data: t.Optional[pd.DataFrame] = None,
        train_data: t.Optional[pd.DataFrame] = None,
        test_data: t.Optional[pd.DataFrame] = None
    ) -> str:
        """
        Format dataset information into a readable string.

        Args:
            data: Optional unified dataset
            train_data: Optional training dataset
            test_data: Optional test dataset

        Returns:
            Formatted string containing dataset information
        """
        name = f"'{self._dataset_name}' " if self._dataset_name else ""
        
        # Check if data is unified or split
        data_description = (
            f"with {len(data)} samples (not split)" 
            if data is not None else 
            f"with {len(train_data)} training samples and {len(test_data)} test samples"
        )
        
        return (
            f"DBDataset({name}{data_description})\n"
            f"Features: {len(self._feature_manager.features)} total "
            f"({len(self._feature_manager.categorical_features)} categorical, "
            f"{len(self._feature_manager.numerical_features)} numerical)\n"
            f"Target: '{self._target_column}'\n"
            f"Model: {'loaded' if self._model_handler.model is not None else 'not loaded'}\n"
            f"Predictions: {'available' if self._model_handler.predictions is not None else 'not available'}"
        )