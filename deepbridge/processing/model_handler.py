import typing as t
from pathlib import Path
import pandas as pd
from joblib import load

class ModelHandler:
    """Handles model loading and prediction operations."""
    
    def __init__(self):
        self._model = None
        self._predictions = None
        self._prob_cols = None
        self._initialize_predictions = False
    
    def load_model(self, model_path: t.Union[str, Path], features: t.List[str] = None, data: t.Optional[t.Dict[str, pd.DataFrame]] = None) -> None:
        """Load model from file and generate predictions if data is provided.
        
        Parameters
        ----------
        model_path : Union[str, Path]
            Path to the saved model file
        features : List[str]
            List of feature names to use for prediction
        data : Optional[Dict[str, pd.DataFrame]]
            Dictionary containing 'train' and 'test' DataFrames for prediction
        """
        try:
            self._model = load(model_path)
            
            # Generate predictions if model is loaded and data is provided
            if self._model is not None and data is not None and features is not None:
                # Generate predictions for train and test sets
                train_proba = self._model.predict_proba(data['train'][features])
                test_proba = self._model.predict_proba(data['test'][features])
                
                # Create DataFrames with probability columns
                train_predictions = pd.DataFrame(
                    train_proba,
                    columns=[f'prob_{i}' for i in range(train_proba.shape[1])]
                )
                test_predictions = pd.DataFrame(
                    test_proba,
                    columns=[f'prob_{i}' for i in range(test_proba.shape[1])]
                )
                
                # Set predictions with generated probability columns
                self.set_predictions(
                    data['train'],
                    data['test'],
                    train_predictions,
                    test_predictions,
                    prob_cols=[f'prob_{i}' for i in range(train_proba.shape[1])]
                )
                
        except Exception as e:
            raise ValueError(f"Failed to load model from {model_path}: {str(e)}")
    
    def set_predictions(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        train_predictions: t.Optional[pd.DataFrame] = None,
        test_predictions: t.Optional[pd.DataFrame] = None,
        prob_cols: t.Optional[t.List[str]] = None
    ) -> None:
        """Set and validate predictions."""
        if prob_cols is None and self._prob_cols is None:
            raise ValueError("prob_cols must be provided when setting predictions for the first time")
        
        prob_cols = prob_cols if prob_cols is not None else self._prob_cols
        self._prob_cols = prob_cols
        
        # Check if we should initialize empty predictions
        if not self._initialize_predictions:
            # First time initializing with prob_cols
            if all(col in train_data.columns for col in prob_cols) and \
               all(col in test_data.columns for col in prob_cols):
                train_predictions = train_data[prob_cols]
                test_predictions = test_data[prob_cols]
                self._initialize_predictions = True
                
        # Process predictions
        train_pred_df = pd.DataFrame() if train_predictions is None else train_predictions.copy()
        test_pred_df = pd.DataFrame() if test_predictions is None else test_predictions.copy()
        
        if not train_pred_df.empty:
            self._validate_predictions(train_pred_df, train_data, prob_cols, 'train')
        if not test_pred_df.empty:
            self._validate_predictions(test_pred_df, test_data, prob_cols, 'test')
        
        self._predictions = pd.concat([
            train_pred_df[prob_cols] if not train_pred_df.empty else pd.DataFrame(),
            test_pred_df[prob_cols] if not test_pred_df.empty else pd.DataFrame()
        ], ignore_index=True)
    
    @staticmethod
    def _validate_predictions(
        pred_df: pd.DataFrame,
        data_df: pd.DataFrame,
        prob_cols: t.List[str],
        name: str
    ) -> None:
        """Validate prediction data."""
        if not pred_df.empty:
            if not isinstance(pred_df, pd.DataFrame):
                raise ValueError(f"{name}_predictions must be a pandas DataFrame")
            if not all(col in pred_df.columns for col in prob_cols):
                raise ValueError(f"Probability columns {prob_cols} not found in {name}_predictions")
            if len(pred_df) != len(data_df):
                raise ValueError(f"Length of {name}_predictions must match length of {name} data")
    
    @property
    def model(self) -> t.Any:
        """Return the loaded model."""
        return self._model
    
    @property
    def predictions(self) -> t.Optional[pd.DataFrame]:
        """Return predictions if available."""
        return self._predictions
    
    @property
    def prob_cols(self) -> t.Optional[t.List[str]]:
        """Return probability column names."""
        return self._prob_cols