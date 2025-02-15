from dataclasses import dataclass, field
from typing import Optional, Union, Dict, Any, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
import joblib
import logging
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Enumeration for model types"""
    STANDARD = "standard"
    SURROGATE = "surrogate"

@dataclass
class ExperimentData:
    """Data class to store experiment data"""
    X_train: Optional[Union[pd.DataFrame, np.ndarray]] = None
    y_train: Optional[Union[pd.Series, np.ndarray]] = None
    X_test: Optional[Union[pd.DataFrame, np.ndarray]] = None
    y_test: Optional[Union[pd.Series, np.ndarray]] = None

    def validate(self) -> None:
        """Validate data consistency"""
        if self.X_train is not None and self.y_train is not None:
            if len(self.X_train) != len(self.y_train):
                raise ValueError("X_train and y_train must have the same number of samples")
        
        if self.X_test is not None and self.y_test is not None:
            if len(self.X_test) != len(self.y_test):
                raise ValueError("X_test and y_test must have the same number of samples")

class ModelValidation:
    """
    Class for model validation experiments, including Surrogate Models 
    and Model Distillation.
    
    Features:
    - Improved type hints and error handling
    - Path handling using pathlib
    - Data validation
    - Logging support
    - Model type enumeration
    - Consistent error messages
    """
    
    def __init__(
        self,
        experiment_name: str = "default_experiment",
        save_path: Optional[Union[str, Path]] = None,
        auto_setup: bool = True
    ):
        """
        Initialize a new validation experiment.

        Args:
            experiment_name: Name for experiment identification
            save_path: Path to save experiment assets
            auto_setup: Whether to automatically create directory structure
        
        Raises:
            ValueError: If experiment_name is empty or contains invalid characters
        """
        if not experiment_name or not experiment_name.strip():
            raise ValueError("experiment_name cannot be empty")
        
        self.experiment_name = experiment_name
        self.save_path = Path(save_path) if save_path else Path("experiments") / experiment_name
        self.models: Dict[str, BaseEstimator] = {}
        self.data = ExperimentData()
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.surrogate_models: Dict[str, BaseEstimator] = {}
        
        if auto_setup:
            self._setup_experiment()
            
        logger.info(f"Initialized experiment: {experiment_name}")

    def _setup_experiment(self) -> None:
        """Configure experiment directory structure"""
        try:
            for folder in ["models", "metrics", "surrogate_models"]:
                (self.save_path / folder).mkdir(parents=True, exist_ok=True)
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
            
        Raises:
            ValueError: If data validation fails
        """
        self.data = ExperimentData(X_train, y_train, X_test, y_test)
        self.data.validate()
        logger.info("Added data to experiment")

    def add_model(
        self,
        model: BaseEstimator,
        model_name: str,
        model_type: ModelType = ModelType.STANDARD
    ) -> None:
        """
        Add a model to the experiment.
        
        Args:
            model: Scikit-learn compatible model
            model_name: Unique identifier for the model
            model_type: Type of model (standard or surrogate)
            
        Raises:
            ValueError: If model_name already exists
        """
        if not isinstance(model, BaseEstimator):
            raise ValueError("Model must be a scikit-learn compatible estimator")
            
        target_dict = (
            self.surrogate_models if model_type == ModelType.SURROGATE else self.models
        )
        
        if model_name in target_dict:
            raise ValueError(f"Model '{model_name}' already exists")
            
        target_dict[model_name] = model
        logger.info(f"Added {model_type.value} model: {model_name}")

    def save_model(self, model_name: str, model_type: ModelType = ModelType.STANDARD) -> Path:
        """
        Save a specific model from the experiment.
        
        Args:
            model_name: Name of the model to save
            model_type: Type of model (standard or surrogate)
            
        Returns:
            Path: Path where the model was saved
            
        Raises:
            ValueError: If model is not found
        """
        target_dict = (
            self.surrogate_models if model_type == ModelType.SURROGATE else self.models
        )
        
        model = target_dict.get(model_name)
        if model is None:
            raise ValueError(f"Model '{model_name}' not found")

        folder = "surrogate_models" if model_type == ModelType.SURROGATE else "models"
        save_path = self.save_path / folder / f"{model_name}.joblib"
        
        try:
            joblib.dump(model, save_path)
            logger.info(f"Saved model to {save_path}")
            return save_path
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

    def load_model(
        self, 
        model_name: str, 
        model_type: ModelType = ModelType.STANDARD
    ) -> BaseEstimator:
        """
        Load a saved model.
        
        Args:
            model_name: Name of the model to load
            model_type: Type of model (standard or surrogate)
            
        Returns:
            BaseEstimator: Loaded model
            
        Raises:
            FileNotFoundError: If model file is not found
        """
        folder = "surrogate_models" if model_type == ModelType.SURROGATE else "models"
        load_path = self.save_path / folder / f"{model_name}.joblib"
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        try:
            model = joblib.load(load_path)
            target_dict = (
                self.surrogate_models if model_type == ModelType.SURROGATE else self.models
            )
            target_dict[model_name] = model
            logger.info(f"Loaded model from {load_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def save_metrics(self, metrics: Dict[str, Any], model_name: str) -> Path:
        """
        Save evaluation metrics for a specific model.
        
        Args:
            metrics: Dictionary of metrics to save
            model_name: Name of the model
            
        Returns:
            Path: Path where metrics were saved
        """
        self.metrics[model_name] = metrics
        save_path = self.save_path / "metrics" / f"{model_name}_metrics.joblib"
        
        try:
            joblib.dump(metrics, save_path)
            logger.info(f"Saved metrics for model {model_name}")
            return save_path
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
            raise

    def get_experiment_info(self) -> Dict[str, Any]:
        """
        Return general information about the experiment.
        
        Returns:
            Dict containing experiment metadata and statistics
        """
        return {
            "experiment_name": self.experiment_name,
            "save_path": str(self.save_path),
            "n_models": len(self.models),
            "n_surrogate_models": len(self.surrogate_models),
            "data_shapes": {
                name: getattr(self.data, name).shape 
                if hasattr(self.data, name) and 
                   isinstance(getattr(self.data, name), (pd.DataFrame, np.ndarray)) 
                else None
                for name in ["X_train", "y_train", "X_test", "y_test"]
            },
            "metrics": self.metrics,
            "models": list(self.models.keys()),
            "surrogate_models": list(self.surrogate_models.keys())
        }