from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Type, Any

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator
import xgboost as xgb

class ModelType(Enum):
    """Supported model types for knowledge distillation."""
    DECISION_TREE = auto()
    LOGISTIC_REGRESSION = auto()
    GBM = auto()
    XGB = auto()
    MLP = auto()

@dataclass
class ModelConfig:
    """Configuration for a machine learning model."""
    model_class: Type[BaseEstimator]
    default_params: Dict[str, Any]

class ModelRegistry:
    """Registry for supported student models in knowledge distillation."""
    
    # Model configurations
    SUPPORTED_MODELS: Dict[str, ModelConfig] = {
        ModelType.DECISION_TREE: ModelConfig(
            DecisionTreeClassifier,
            {
                'max_depth': 5,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            }
        ),
        ModelType.LOGISTIC_REGRESSION: ModelConfig(
            LogisticRegression,
            {
                'C': 1.0,
                'max_iter': 1000,
                'random_state': 42,
                'solver': 'lbfgs',
                'multi_class': 'auto'
            }
        ),
        ModelType.GBM: ModelConfig(
            GradientBoostingClassifier,
            {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'random_state': 42
            }
        ),
        ModelType.XGB: ModelConfig(
            xgb.XGBClassifier,
            {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'random_state': 42,
                'objective': 'binary:logistic'
            }
        ),
        ModelType.MLP: ModelConfig(
            MLPClassifier,
            {
                'hidden_layer_sizes': (100, 50),
                'max_iter': 1000,
                'random_state': 42,
                'early_stopping': True,
                'validation_fraction': 0.1
            }
        )
    }
    
    @classmethod
    def get_model(cls, model_type: ModelType, custom_params: Dict[str, Any] = None) -> BaseEstimator:
        """
        Get an instance of a model with specified parameters.
        
        Args:
            model_type: Type of model to instantiate
            custom_params: Custom parameters to override defaults
            
        Returns:
            Instantiated model
        """
        if model_type not in cls.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        config = cls.SUPPORTED_MODELS[model_type]
        params = config.default_params.copy()
        
        if custom_params:
            params.update(custom_params)
            
        return config.model_class(**params)