from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, 
    accuracy_score, 
    classification_report,
    precision_recall_curve,
    average_precision_score
)
from sklearn.base import BaseEstimator
from scipy.special import logit, expit
import xgboost as xgb
from typing import Union, Tuple, Optional, Dict, Any, List
import logging
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelType(str, Enum):
    """Supported model types for distillation"""
    GBM = 'gbm'
    XGB = 'xgb'
    MLP = 'mlp'

@dataclass
class ModelConfig:
    """Configuration for model initialization"""
    model_class: type
    default_params: Dict[str, Any]

class ModelDistiller:
    """
    A model distillation class that creates simpler models that mimic the behavior 
    of more complex models by learning from their probability outputs.
    
    Features:
    - Support for multiple model types (GBM, XGB, MLP)
    - Automatic input validation and preprocessing
    - Comprehensive metrics calculation
    - Model persistence capabilities
    - Logging support
    """
    
    # Model configurations
    SUPPORTED_MODELS: Dict[str, ModelConfig] = {
        ModelType.GBM: ModelConfig(
            GradientBoostingRegressor,
            {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'random_state': 42
            }
        ),
        ModelType.XGB: ModelConfig(
            xgb.XGBRegressor,
            {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'random_state': 42,
                'objective': 'reg:squarederror'
            }
        ),
        ModelType.MLP: ModelConfig(
            MLPRegressor,
            {
                'hidden_layer_sizes': (100, 50),
                'max_iter': 1000,
                'random_state': 42,
                'early_stopping': True,
                'validation_fraction': 0.1
            }
        )
    }

    def __init__(
        self, 
        model_type: Union[str, ModelType] = ModelType.GBM,
        model_params: Optional[Dict[str, Any]] = None,
        save_path: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the distiller with the chosen model type.
        
        Args:
            model_type: Type of model to use ('gbm', 'xgb', or 'mlp')
            model_params: Optional dictionary of model parameters to override defaults
            save_path: Optional path to save model artifacts
        
        Raises:
            ValueError: If model_type is not supported
        """
        self.model_type = ModelType(model_type)
        self.save_path = Path(save_path) if save_path else None
        
        if self.model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model type. Choose from: {[m.value for m in ModelType]}"
            )
        
        config = self.SUPPORTED_MODELS[self.model_type]
        params = config.default_params.copy()
        
        if model_params:
            params.update(model_params)
            
        self.model = config.model_class(**params)
        self.is_fitted = False
        self.training_history: Dict[str, List[float]] = {
            'train_roc_auc': [],
            'test_roc_auc': []
        }
        
        logger.info(f"Initialized {self.model_type} distiller")

    def _validate_input(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        probas: Union[pd.DataFrame, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validate and preprocess input data.
        
        Args:
            X: Feature matrix
            probas: Probability predictions from original model
            
        Returns:
            Tuple of processed X and probas as numpy arrays
            
        Raises:
            ValueError: If inputs are invalid
        """
        try:
            X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
            probas_arr = probas.values if isinstance(probas, pd.DataFrame) else np.asarray(probas)
            
            if X_arr.ndim != 2:
                raise ValueError(f"X must be a 2D array, got shape {X_arr.shape}")
                
            if probas_arr.ndim == 1:
                probas_arr = probas_arr.reshape(-1, 1)
            elif probas_arr.shape[1] != 2 and probas_arr.shape[1] != 1:
                raise ValueError(
                    f"probas must have shape (n_samples,) or (n_samples, 2), "
                    f"got shape {probas_arr.shape}"
                )
                
            if len(X_arr) != len(probas_arr):
                raise ValueError(
                    f"X and probas must have same number of samples. "
                    f"Got X: {len(X_arr)}, probas: {len(probas_arr)}"
                )
                
            return X_arr, probas_arr
            
        except Exception as e:
            logger.error(f"Input validation failed: {str(e)}")
            raise

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        probas: Union[pd.DataFrame, np.ndarray],
        test_size: float = 0.2,
        random_state: Optional[int] = None,
        verbose: bool = True
    ) -> 'ModelDistiller':
        """
        Train the distilled model using original features and model probabilities.
        
        Args:
            X: Feature matrix
            probas: Probability predictions from original model
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            verbose: Whether to print training progress
            
        Returns:
            self: The fitted distiller instance
            
        Raises:
            ValueError: If inputs are invalid
        """
        logger.info("Starting model fitting")
        X_arr, probas_arr = self._validate_input(X, probas)
        
        # Split data
        X_train, X_test, probas_train, probas_test = train_test_split(
            X_arr, probas_arr, 
            test_size=test_size,
            random_state=random_state or 42
        )
        
        # Extract class 1 probabilities if 2D
        if probas_arr.shape[1] == 2:
            probas_train = probas_train[:, 1]
            probas_test = probas_test[:, 1]
        
        # Convert to logit space and fit
        y_train = logit(np.clip(probas_train, 1e-7, 1 - 1e-7))
        
        try:
            self.model.fit(X_train, y_train)
            self.is_fitted = True
            logger.info("Model fitting completed successfully")
        except Exception as e:
            logger.error(f"Model fitting failed: {str(e)}")
            raise
        
        # Calculate and store metrics
        metrics = self._calculate_metrics(X_train, X_test, probas_train, probas_test)
        self._update_training_history(metrics)
        
        if verbose:
            self._print_training_summary(metrics)
        
        if self.save_path:
            self.save()
        
        return self

    def _calculate_metrics(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        probas_train: np.ndarray,
        probas_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive model performance metrics.
        
        Returns:
            Dictionary containing calculated metrics
        """
        train_probas = self.predict(X_train)
        test_probas = self.predict(X_test)
        
        metrics = {
            'train_roc_auc': roc_auc_score(
                (probas_train > 0.5).astype(int), 
                train_probas
            ),
            'test_roc_auc': roc_auc_score(
                (probas_test > 0.5).astype(int),
                test_probas
            ),
            'train_accuracy': accuracy_score(
                (probas_train > 0.5).astype(int),
                train_probas > 0.5
            ),
            'test_accuracy': accuracy_score(
                (probas_test > 0.5).astype(int),
                test_probas > 0.5
            ),
            'train_avg_precision': average_precision_score(
                (probas_train > 0.5).astype(int),
                train_probas
            ),
            'test_avg_precision': average_precision_score(
                (probas_test > 0.5).astype(int),
                test_probas
            )
        }
        
        return metrics

    def _update_training_history(self, metrics: Dict[str, float]) -> None:
        """Update training history with new metrics"""
        for metric in ['train_roc_auc', 'test_roc_auc']:
            self.training_history[metric].append(metrics[metric])

    def _print_training_summary(self, metrics: Dict[str, float]) -> None:
        """Print a summary of training metrics"""
        print("\nDistilled Model Performance:")
        print(f"Train ROC AUC: {metrics['train_roc_auc']:.4f}")
        print(f"Test ROC AUC: {metrics['test_roc_auc']:.4f}")
        print(f"Train Accuracy: {metrics['train_accuracy']:.4f}")
        print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"Train Avg Precision: {metrics['train_avg_precision']:.4f}")
        print(f"Test Avg Precision: {metrics['test_avg_precision']:.4f}")

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make probability predictions with the distilled model.
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            Array of predicted probabilities
            
        Raises:
            ValueError: If model is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        try:
            X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
            return expit(self.model.predict(X_arr))
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        Save the distilled model and its metadata.
        
        Args:
            path: Optional path to save the model. If None, uses the instance save_path
            
        Raises:
            ValueError: If no save path is specified
        """
        save_path = Path(path) if path else self.save_path
        if not save_path:
            raise ValueError("No save path specified")
            
        save_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save the model
            model_path = save_path / 'distilled_model.joblib'
            pd.to_pickle(self.model, model_path)
            
            # Save metadata
            metadata = {
                'model_type': self.model_type.value,
                'is_fitted': self.is_fitted,
                'training_history': self.training_history
            }
            
            with open(save_path / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"Model saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'ModelDistiller':
        """
        Load a saved distilled model.
        
        Args:
            path: Path to the saved model directory
            
        Returns:
            Loaded ModelDistiller instance
            
        Raises:
            FileNotFoundError: If model files are not found
        """
        path = Path(path)
        
        try:
            # Load metadata
            with open(path / 'metadata.json', 'r') as f:
                metadata = json.load(f)
                
            # Create instance
            instance = cls(model_type=metadata['model_type'])
            
            # Load model
            instance.model = pd.read_pickle(path / 'distilled_model.joblib')
            instance.is_fitted = metadata['is_fitted']
            instance.training_history = metadata['training_history']
            
            logger.info(f"Model loaded from {path}")
            return instance
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

def calculate_detailed_metrics(
    original_probas: np.ndarray,
    distilled_probas: np.ndarray,
    y_true: np.ndarray
) -> Dict[str, Any]:
    """
    Calculate comprehensive comparison metrics between original and distilled models.
    
    Args:
        original_probas: Probability predictions from original model
        distilled_probas: Probability predictions from distilled model
        y_true: True labels
        
    Returns:
        Dictionary of calculated metrics
    """
    metrics = {
        'original_roc_auc': roc_auc_score(y_true, original_probas),
        'distilled_roc_auc': roc_auc_score(y_true, distilled_probas),
        'original_accuracy': accuracy_score(y_true, original_probas > 0.5),
        'distilled_accuracy': accuracy_score(y_true, distilled_probas > 0.5),
        'original_avg_precision': average_precision_score(y_true, original_probas),
        'distilled_avg_precision': average_precision_score(y_true, distilled_probas),
        'classification_report': classification_report(
            y_true,
            distilled_probas > 0.5,
            output_dict=True
        )
    }
    
    # Calculate probability calibration metrics
    probs_bins = np.linspace(0, 1, 11)
    metrics['probability_distribution'] = {
        'original': np.histogram(original_probas, bins=probs_bins)[0].tolist(),
        'distilled': np.histogram(distilled_probas, bins=probs_bins)[0].tolist(),
        'bin_edges': probs_bins.tolist()
    }
    
    # Calculate prediction differences
    prediction_diff = np.abs(original_probas - distilled_probas)
    metrics['prediction_differences'] = {
        'mean_difference': float(np.mean(prediction_diff)),
        'max_difference': float(np.max(prediction_diff)),
        'std_difference': float(np.std(prediction_diff)),
        'percentiles': {
            '25': float(np.percentile(prediction_diff, 25)),
            '50': float(np.percentile(prediction_diff, 50)),
            '75': float(np.percentile(prediction_diff, 75)),
            '95': float(np.percentile(prediction_diff, 95))
        }
    }
    
    return metrics