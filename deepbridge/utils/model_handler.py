import typing as t
from pathlib import Path
import pandas as pd
from joblib import load

class ModelHandler:
    """Handles model loading and prediction operations."""
    
    def __init__(self):
        """Initialize the model handler."""
        self._model = None
        self._predictions = None
        self._prob_cols = None
        self._initialize_predictions = False
        self._original_predictions = None  # Store original predictions for synthetic data
    
    @property
    def model(self) -> t.Any:
        """Return the loaded model."""
        return self._model
    
    @model.setter
    def model(self, model: t.Any) -> None:
        """Set the model directly."""
        self._model = model
    
    def generate_predictions(self, data: t.Dict[str, pd.DataFrame], features: t.List[str]) -> None:
        """
        Generate predictions using the loaded model.
        
        Args:
            data: Dictionary containing 'train' and/or 'test' DataFrames
            features: List of feature names to use for prediction
        """
        if self._model is None:
            raise ValueError("No model available to generate predictions.")
        
        prob_cols = None
        train_preds = None
        test_preds = None
        
        # Generate predictions for each dataset
        for dataset_name, dataset in data.items():
            if dataset is None or len(dataset) == 0:
                continue
                
            try:
                # Make predictions using the model
                proba = self._model.predict_proba(dataset[features])
                
                # Create DataFrame with probability columns
                cols = [f'prob_class_{i}' for i in range(proba.shape[1])]
                preds_df = pd.DataFrame(proba, columns=cols, index=dataset.index)
                
                # Store predictions
                if prob_cols is None:
                    prob_cols = cols
                
                # Set predictions for the specific dataset
                if dataset_name == 'train':
                    train_preds = preds_df
                else:
                    test_preds = preds_df
            except Exception as e:
                raise ValueError(f"Failed to generate predictions for {dataset_name} data: {str(e)}")
        
        # Set predictions for all datasets
        self.set_predictions(
            data.get('train'),
            data.get('test'),
            train_preds,
            test_preds,
            prob_cols
        )
    
    def load_model(self, model_path: t.Union[str, Path], features: t.List[str] = None, 
                  data: t.Optional[t.Dict[str, pd.DataFrame]] = None) -> None:
        """
        Load model from file and generate predictions if data is provided.
        
        Args:
            model_path: Path to the saved model file
            features: List of feature names to use for prediction
            data: Dictionary containing 'train' and 'test' DataFrames for prediction
        """
        try:
            self._model = load(model_path)
            
            # Generate predictions if model is loaded and data is provided
            if self._model is not None and data is not None and features is not None:
                self.generate_predictions(data, features)
                
        except Exception as e:
            raise ValueError(f"Failed to load model from {model_path}: {str(e)}")
    
    def set_predictions(
        self,
        train_data: t.Optional[pd.DataFrame] = None,
        test_data: t.Optional[pd.DataFrame] = None,
        train_predictions: t.Optional[pd.DataFrame] = None,
        test_predictions: t.Optional[pd.DataFrame] = None,
        prob_cols: t.Optional[t.List[str]] = None
    ) -> None:
        """
        Set and validate predictions.
        
        Args:
            train_data: Training data DataFrame
            test_data: Test data DataFrame
            train_predictions: Predictions for training data
            test_predictions: Predictions for test data
            prob_cols: List of probability column names
        """
        if prob_cols is None and self._prob_cols is None:
            raise ValueError("prob_cols must be provided when setting predictions for the first time")
        
        prob_cols = prob_cols if prob_cols is not None else self._prob_cols
        self._prob_cols = prob_cols
        
        # Check if we should initialize empty predictions
        if not self._initialize_predictions and train_data is not None and test_data is not None:
            # First time initializing with prob_cols
            if all(col in train_data.columns for col in prob_cols) and \
               all(col in test_data.columns for col in prob_cols):
                train_predictions = train_data[prob_cols]
                test_predictions = test_data[prob_cols]
                self._initialize_predictions = True
        
        # Process predictions
        train_pred_df = pd.DataFrame() if train_predictions is None else train_predictions.copy()
        test_pred_df = pd.DataFrame() if test_predictions is None else test_predictions.copy()
        
        # Validate predictions if train/test data are provided
        if not train_pred_df.empty and train_data is not None:
            self._validate_predictions(train_pred_df, train_data, prob_cols, 'train')
        if not test_pred_df.empty and test_data is not None:
            self._validate_predictions(test_pred_df, test_data, prob_cols, 'test')
        
        # Combine predictions
        prediction_dfs = []
        if not train_pred_df.empty:
            prediction_dfs.append(train_pred_df[prob_cols])
        if not test_pred_df.empty:
            prediction_dfs.append(test_pred_df[prob_cols])
            
        if prediction_dfs:
            self._predictions = pd.concat(prediction_dfs, ignore_index=True)
        else:
            self._predictions = pd.DataFrame(columns=prob_cols)
    
    @staticmethod
    def _validate_predictions(
        pred_df: pd.DataFrame,
        data_df: pd.DataFrame,
        prob_cols: t.List[str],
        name: str
    ) -> None:
        """
        Validate prediction data.
        
        Args:
            pred_df: Predictions DataFrame
            data_df: Data DataFrame
            prob_cols: List of probability column names
            name: Name of the dataset ('train' or 'test')
        """
        if not isinstance(pred_df, pd.DataFrame):
            raise ValueError(f"{name}_predictions must be a pandas DataFrame")
        if not all(col in pred_df.columns for col in prob_cols):
            raise ValueError(f"Probability columns {prob_cols} not found in {name}_predictions")
        if len(pred_df) != len(data_df):
            raise ValueError(f"Length of {name}_predictions ({len(pred_df)}) must match length of {name} data ({len(data_df)})")
    
    @property
    def predictions(self) -> t.Optional[pd.DataFrame]:
        """Return predictions if available."""
        return self._predictions
    
    @property
    def prob_cols(self) -> t.Optional[t.List[str]]:
        """Return probability column names."""
        return self._prob_cols
    
    def reset(self) -> None:
        """Reset the handler to its initial state."""
        self._model = None
        self._predictions = None
        self._prob_cols = None
        self._initialize_predictions = False
        self._original_predictions = None