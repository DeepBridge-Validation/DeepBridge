from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Type, Any, Callable
import optuna

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.base import BaseEstimator
import xgboost as xgb

class ModelType(Enum):
    """Supported model types for knowledge distillation."""
    DECISION_TREE = auto()
    LOGISTIC_REGRESSION = auto()
    GBM = auto()
    XGB = auto()
    MLP = auto()

class ModelMode(Enum):
    """Supported model modes."""
    CLASSIFICATION = auto()
    REGRESSION = auto()

@dataclass
class ModelConfig:
    """Configuration for a machine learning model."""
    classifier_class: Type[BaseEstimator]
    regressor_class: Type[BaseEstimator]
    default_params: Dict[str, Any]
    param_space_fn: Callable[[optuna.Trial], Dict[str, Any]]

class ModelRegistry:
    """Registry for supported student models in knowledge distillation."""
    
    @staticmethod
    def _logistic_regression_param_space(trial: optuna.Trial) -> Dict[str, Any]:
        """Define parameter space for logistic regression."""
        return {
            'C': trial.suggest_float('C', 1e-3, 10, log=True),
            'solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear']),
            'max_iter': trial.suggest_categorical('max_iter', [500, 1000, 2000])
            # multi_class é omitido para evitar FutureWarning
        }
    
    @staticmethod
    def _linear_regression_param_space(trial: optuna.Trial) -> Dict[str, Any]:
        """Define parameter space for linear regression."""
        return {
            'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
            'positive': trial.suggest_categorical('positive', [True, False]),
        }
    
    @staticmethod
    def _decision_tree_param_space(trial: optuna.Trial) -> Dict[str, Any]:
        """Define parameter space for decision tree."""
        return {
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
        }
    
    @staticmethod
    def _gbm_param_space(trial: optuna.Trial) -> Dict[str, Any]:
        """Define parameter space for gradient boosting machine."""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0)
        }
    
    @staticmethod
    def _xgb_param_space(trial: optuna.Trial) -> Dict[str, Any]:
        """Define parameter space for XGBoost."""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
        }
    
    # Model configurations
    SUPPORTED_MODELS: Dict[ModelType, ModelConfig] = {
        ModelType.DECISION_TREE: ModelConfig(
            classifier_class=DecisionTreeClassifier,
            regressor_class=DecisionTreeRegressor,
            default_params={
                'max_depth': 5,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            },
            param_space_fn=_decision_tree_param_space
        ),
        ModelType.LOGISTIC_REGRESSION: ModelConfig(
            classifier_class=LogisticRegression,
            regressor_class=LinearRegression,
            default_params={
                'C': 1.0,
                'max_iter': 1000,
                'random_state': 42,
                'solver': 'lbfgs'
                # multi_class foi removido para evitar FutureWarning
            },
            param_space_fn=_logistic_regression_param_space
        ),
        ModelType.GBM: ModelConfig(
            classifier_class=GradientBoostingClassifier,
            regressor_class=GradientBoostingRegressor,
            default_params={
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'random_state': 42
            },
            param_space_fn=_gbm_param_space
        ),
        ModelType.XGB: ModelConfig(
            classifier_class=xgb.XGBClassifier,
            regressor_class=xgb.XGBRegressor,
            default_params={
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'random_state': 42,
                'objective': 'binary:logistic'  # Vai ser substituído para regressão
            },
            param_space_fn=_xgb_param_space
        )
    }
    
    @classmethod
    def get_model(
        cls, 
        model_type: ModelType, 
        custom_params: Dict[str, Any] = None,
        mode: ModelMode = ModelMode.CLASSIFICATION
    ) -> BaseEstimator:
        """
        Get an instance of a model with specified parameters.
        
        Args:
            model_type: Type of model to instantiate
            custom_params: Custom parameters to override defaults
            mode: Whether to return a classifier or regressor
            
        Returns:
            Instantiated model
        """
        if model_type not in cls.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        config = cls.SUPPORTED_MODELS[model_type]
        params = config.default_params.copy()
        
        if custom_params:
            params.update(custom_params)
            
        # Ajustar parâmetros específicos para cada modo
        if mode == ModelMode.REGRESSION:
            # Para XGBoost, mudar o objetivo para regressão
            if model_type == ModelType.XGB and 'objective' in params:
                params['objective'] = 'reg:squarederror'
                
            # Para LogisticRegression -> LinearRegression, remover parâmetros incompatíveis
            if model_type == ModelType.LOGISTIC_REGRESSION:
                params_to_remove = ['C', 'max_iter', 'solver', 'multi_class']
                for param in params_to_remove:
                    if param in params:
                        del params[param]
                
            # Escolher a classe de regressor
            model_class = config.regressor_class
        else:
            # Escolher a classe de classificador
            model_class = config.classifier_class
            
        return model_class(**params)
    
    @classmethod
    def get_param_space(
        cls, 
        model_type: ModelType, 
        trial: optuna.Trial,
        mode: ModelMode = ModelMode.CLASSIFICATION
    ) -> Dict[str, Any]:
        """
        Get parameter space for the specified model type.
        
        Args:
            model_type: Type of model to get parameter space for
            trial: Optuna trial instance
            mode: Whether to get parameters for classifier or regressor
            
        Returns:
            Dictionary of parameters to optimize
        """
        if model_type not in cls.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        config = cls.SUPPORTED_MODELS[model_type]
        
        # Obter espaço de parâmetros básico
        param_space = config.param_space_fn(trial)
        
        # Adicionar random_state para garantir reprodutibilidade
        if 'random_state' in config.default_params:
            param_space['random_state'] = config.default_params['random_state']
            
        # Ajustar para modo de regressão
        if mode == ModelMode.REGRESSION:
            # Para XGBoost, garantir que objective seja para regressão
            if model_type == ModelType.XGB:
                param_space['objective'] = 'reg:squarederror'
                
            # Para LogisticRegression -> LinearRegression, remover parâmetros incompatíveis
            if model_type == ModelType.LOGISTIC_REGRESSION:
                params_to_remove = ['C', 'solver', 'max_iter', 'multi_class']
                for param in params_to_remove:
                    if param in param_space:
                        param_space.pop(param)
        else:
            # Para XGBoost, garantir que objective seja mantido para classificação
            if model_type == ModelType.XGB and 'objective' in config.default_params:
                param_space['objective'] = config.default_params['objective']
            
        # Para MLP, garantir que validation_fraction seja mantido
        if model_type == ModelType.MLP and 'validation_fraction' in config.default_params:
            param_space['validation_fraction'] = config.default_params['validation_fraction']
            
        # Remover parâmetros temporários usados apenas para a otimização
        if model_type == ModelType.MLP and 'n_layers' in param_space:
            param_space.pop('n_layers')
        
        return param_space