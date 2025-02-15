from typing import Optional, Union, Dict
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
import joblib
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelValidation:
    """
    Class for managing model validation experiments, including model versioning,
    metrics tracking, and surrogate model support.
    
    Attributes:
        experiment_name (str): Name of the experiment
        save_path (Path): Path where experiment files are saved
        models (Dict[str, BaseEstimator]): Dictionary of main models
        surrogate_models (Dict[str, BaseEstimator]): Dictionary of surrogate models
        metrics (Dict[str, Dict]): Dictionary of model metrics
    """
    
    def __init__(
        self,
        experiment_name: str = "default_experiment",
        save_path: Optional[str] = None
    ):
        """
        Initialize a new validation experiment.

        Args:
            experiment_name: Name for experiment identification
            save_path: Path to save experiment files (optional)
        """
        self.experiment_name = experiment_name
        self.save_path = Path(save_path) if save_path else Path("experiments") / experiment_name
        self.models: Dict[str, BaseEstimator] = {}
        self.data = {
            "X_train": None,
            "y_train": None,
            "X_test": None,
            "y_test": None
        }
        self.metrics: Dict[str, Dict] = {}
        self.surrogate_models: Dict[str, BaseEstimator] = {}
        self._setup_experiment()
        logger.info(f"Created experiment: {experiment_name}")

    def _setup_experiment(self) -> None:
        """Configure experiment directory structure."""
        try:
            # Create main experiment directory
            self.save_path.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            for subdir in ['models', 'metrics', 'surrogate_models']:
                (self.save_path / subdir).mkdir(exist_ok=True)
                
            logger.info(f"Created experiment directories at {self.save_path}")
        except Exception as e:
            logger.error(f"Failed to create experiment directories: {e}")
            raise

    def add_data(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_test: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_test: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> None:
        """
        Add training and test data to the experiment.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features (optional)
            y_test: Test labels (optional)
        """
        try:
            # Validate data shapes
            if len(X_train) != len(y_train):
                raise ValueError("X_train and y_train must have same number of samples")
            
            if X_test is not None and y_test is not None:
                if len(X_test) != len(y_test):
                    raise ValueError("X_test and y_test must have same number of samples")
            
            # Store data
            self.data["X_train"] = X_train
            self.data["y_train"] = y_train
            self.data["X_test"] = X_test
            self.data["y_test"] = y_test
            
            logger.info("Added data to experiment")
            
        except Exception as e:
            logger.error(f"Failed to add data: {e}")
            raise

    def add_model(
        self,
        model: BaseEstimator,
        model_name: str,
        is_surrogate: bool = False
    ) -> None:
        """
        Add a model to the experiment.

        Args:
            model: Scikit-learn compatible model
            model_name: Unique identifier for the model
            is_surrogate: Whether this is a surrogate model
        """
        try:
            # Validate model type
            if not isinstance(model, BaseEstimator):
                raise ValueError("Model must be a scikit-learn compatible estimator")
            
            # Check if model name already exists
            target_dict = self.surrogate_models if is_surrogate else self.models
            if model_name in target_dict:
                raise ValueError(f"Model '{model_name}' already exists")
            
            # Add model
            target_dict[model_name] = model
            logger.info(f"Added {'surrogate ' if is_surrogate else ''}model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to add model: {e}")
            raise

    def save_model(self, model_name: str, is_surrogate: bool = False) -> None:
        """
        Save a model to disk.

        Args:
            model_name: Name of the model to save
            is_surrogate: Whether this is a surrogate model
        """
        try:
            # Get correct model dictionary
            model_dict = self.surrogate_models if is_surrogate else self.models
            model = model_dict.get(model_name)
            
            if model is None:
                raise ValueError(f"Model '{model_name}' not found")
            
            # Create save path
            folder = "surrogate_models" if is_surrogate else "models"
            save_path = self.save_path / folder / f"{model_name}.joblib"
            
            # Save model
            joblib.dump(model, save_path)
            logger.info(f"Saved model to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

    def load_model(self, model_name: str, is_surrogate: bool = False) -> BaseEstimator:
        """
        Load a saved model.

        Args:
            model_name: Name of the model to load
            is_surrogate: Whether this is a surrogate model

        Returns:
            Loaded model instance
        """
        try:
            # Determine model path
            folder = "surrogate_models" if is_surrogate else "models"
            model_path = self.save_path / folder / f"{model_name}.joblib"
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Load model
            model = joblib.load(model_path)
            
            # Store in appropriate dictionary
            target_dict = self.surrogate_models if is_surrogate else self.models
            target_dict[model_name] = model
            
            logger.info(f"Loaded model from {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def save_metrics(self, metrics: Dict, model_name: str) -> None:
        """
        Save metrics for a specific model.

        Args:
            metrics: Dictionary of metrics to save
            model_name: Name of the model these metrics belong to
        """
        try:
            # Store metrics in instance
            self.metrics[model_name] = metrics
            
            # Save to disk
            metrics_path = self.save_path / "metrics" / f"{model_name}_metrics.joblib"
            joblib.dump(metrics, metrics_path)
            
            logger.info(f"Saved metrics for model {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
            raise

    def get_experiment_info(self) -> Dict:
        """
        Get information about the experiment.

        Returns:
            Dictionary containing experiment metadata and metrics
        """
        try:
            info = {
                "experiment_name": self.experiment_name,
                "save_path": str(self.save_path),
                "n_models": len(self.models),
                "n_surrogate_models": len(self.surrogate_models),
                "data_shapes": {
                    name: (data.shape if isinstance(data, (pd.DataFrame, np.ndarray)) else None)
                    for name, data in self.data.items()
                },
                "metrics": self.metrics,
                "models": list(self.models.keys()),
                "surrogate_models": list(self.surrogate_models.keys())
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get experiment info: {e}")
            raise

    def __str__(self) -> str:
        """String representation of the experiment."""
        info = self.get_experiment_info()
        return (
            f"Experiment: {info['experiment_name']}\n"
            f"Models: {len(info['models'])}\n"
            f"Surrogate Models: {len(info['surrogate_models'])}\n"
            f"Metrics Available: {len(info['metrics'])}"
        )