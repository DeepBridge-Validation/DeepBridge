from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Type, Any, Callable, List, Optional, Tuple
import optuna

from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np

# Importações dos modelos de séries temporais
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import lightgbm as lgbm

# Adapters para uniformizar a interface entre diferentes modelos
class BaseTimeSeriesAdapter(BaseEstimator, RegressorMixin):
    """
    Base adapter para modelos de séries temporais.
    Converte diferentes interfaces de modelos para um padrão comum.
    """
    def __init__(self, model, **kwargs):
        self.model = model
        self.is_fitted = False
        self.kwargs = kwargs
        
    def fit(self, X, y):
        """
        Treina o modelo adaptado
        
        Args:
            X: Features de entrada (matriz com lag features, etc)
            y: Valores alvo da série temporal
        """
        raise NotImplementedError("Método fit deve ser implementado na classe filha")
        
    def predict(self, X):
        """
        Realiza previsões com o modelo adaptado
        
        Args:
            X: Features de entrada 
            
        Returns:
            Array de previsões
        """
        raise NotImplementedError("Método predict deve ser implementado na classe filha")
    
    @property
    def feature_importances_(self):
        """Retorna importâncias de features se o modelo suportar"""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None


class SklearnTSAdapter(BaseTimeSeriesAdapter):
    """Adapter para modelos do scikit-learn usados em séries temporais"""
    
    def fit(self, X, y):
        self.model.fit(X, y)
        self.is_fitted = True
        return self
        
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("O modelo deve ser treinado antes de fazer previsões")
        return self.model.predict(X)


class ARIMATSAdapter(BaseTimeSeriesAdapter):
    """Adapter para modelos ARIMA do statsmodels"""
    
    def fit(self, X, y):
        # ARIMA utiliza apenas a série temporal y
        # X contém features incorporadas, mas ARIMA é univariado
        # Extraímos os parâmetros p, d, q dos kwargs
        p = self.kwargs.get('order', (1, 0, 0))[0]
        d = self.kwargs.get('order', (1, 0, 0))[1]
        q = self.kwargs.get('order', (1, 0, 0))[2]
        
        # Criamos e treinamos o modelo
        self.model = ARIMA(y, order=(p, d, q))
        self.fitted_model = self.model.fit()
        self.is_fitted = True
        return self
        
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("O modelo deve ser treinado antes de fazer previsões")
        
        # Para previsões em ARIMA, precisamos do tamanho da amostra
        steps = X.shape[0] if hasattr(X, "shape") else len(X)
        
        # Usar a função de forecast padrão do ARIMA
        forecast = self.fitted_model.forecast(steps=steps)
        return forecast


class TSModelType(Enum):
    """Tipos de modelos suportados para destilação de séries temporais"""
    RIDGE = auto()
    ELASTIC_NET = auto()
    LASSO = auto()
    RANDOM_FOREST = auto()
    GRADIENT_BOOSTING = auto()
    XGB = auto()
    LGBM = auto()
    ARIMA = auto()


@dataclass
class TSModelConfig:
    """Configuração para modelos de séries temporais"""
    model_class: Type[BaseEstimator]
    adapter_class: Type[BaseTimeSeriesAdapter]
    default_params: Dict[str, Any]
    param_space_fn: Callable[[optuna.Trial], Dict[str, Any]]
    is_univariate: bool = False  # Indica se o modelo é univariado (só usa a série)


class TSModelRegistry:
    """
    Registro de modelos para destilação de séries temporais.
    Centraliza a criação e configuração de diferentes tipos de modelos.
    """
    
    @staticmethod
    def _ridge_param_space(trial: optuna.Trial) -> Dict[str, Any]:
        """Define espaço de parâmetros para Ridge"""
        return {
            'alpha': trial.suggest_float('alpha', 0.01, 10.0, log=True),
            'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
            'solver': trial.suggest_categorical('solver', ['auto', 'svd', 'cholesky', 'lsqr']),
            'max_iter': trial.suggest_categorical('max_iter', [1000, 3000, 5000])
        }
    
    @staticmethod
    def _elastic_net_param_space(trial: optuna.Trial) -> Dict[str, Any]:
        """Define espaço de parâmetros para ElasticNet"""
        return {
            'alpha': trial.suggest_float('alpha', 0.01, 10.0, log=True),
            'l1_ratio': trial.suggest_float('l1_ratio', 0.1, 0.9),
            'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
            'max_iter': trial.suggest_categorical('max_iter', [1000, 3000, 5000])
        }
    
    @staticmethod
    def _lasso_param_space(trial: optuna.Trial) -> Dict[str, Any]:
        """Define espaço de parâmetros para Lasso"""
        return {
            'alpha': trial.suggest_float('alpha', 0.01, 10.0, log=True),
            'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
            'max_iter': trial.suggest_categorical('max_iter', [1000, 3000, 5000])
        }
    
    @staticmethod
    def _random_forest_param_space(trial: optuna.Trial) -> Dict[str, Any]:
        """Define espaço de parâmetros para Random Forest"""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 25),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
        }
    
    @staticmethod
    def _gradient_boosting_param_space(trial: optuna.Trial) -> Dict[str, Any]:
        """Define espaço de parâmetros para Gradient Boosting"""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0)
        }
    
    @staticmethod
    def _xgb_param_space(trial: optuna.Trial) -> Dict[str, Any]:
        """Define espaço de parâmetros para XGBoost"""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True)
        }
    
    @staticmethod
    def _lgbm_param_space(trial: optuna.Trial) -> Dict[str, Any]:
        """Define espaço de parâmetros para LightGBM"""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True)
        }
    
    @staticmethod
    def _arima_param_space(trial: optuna.Trial) -> Dict[str, Any]:
        """Define espaço de parâmetros para ARIMA"""
        return {
            'order': (
                trial.suggest_int('p', 0, 5),
                trial.suggest_int('d', 0, 2),
                trial.suggest_int('q', 0, 5)
            )
        }
    
    # Registro dos modelos suportados com suas configurações
    SUPPORTED_MODELS: Dict[TSModelType, TSModelConfig] = {
        TSModelType.RIDGE: TSModelConfig(
            model_class=Ridge,
            adapter_class=SklearnTSAdapter,
            default_params={
                'alpha': 1.0,
                'fit_intercept': True,
                'random_state': 42
            },
            param_space_fn=_ridge_param_space
        ),
        TSModelType.ELASTIC_NET: TSModelConfig(
            model_class=ElasticNet,
            adapter_class=SklearnTSAdapter,
            default_params={
                'alpha': 1.0,
                'l1_ratio': 0.5,
                'fit_intercept': True,
                'random_state': 42
            },
            param_space_fn=_elastic_net_param_space
        ),
        TSModelType.LASSO: TSModelConfig(
            model_class=Lasso,
            adapter_class=SklearnTSAdapter,
            default_params={
                'alpha': 1.0,
                'fit_intercept': True,
                'random_state': 42
            },
            param_space_fn=_lasso_param_space
        ),
        TSModelType.RANDOM_FOREST: TSModelConfig(
            model_class=RandomForestRegressor,
            adapter_class=SklearnTSAdapter,
            default_params={
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            },
            param_space_fn=_random_forest_param_space
        ),
        TSModelType.GRADIENT_BOOSTING: TSModelConfig(
            model_class=GradientBoostingRegressor,
            adapter_class=SklearnTSAdapter,
            default_params={
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'random_state': 42
            },
            param_space_fn=_gradient_boosting_param_space
        ),
        TSModelType.XGB: TSModelConfig(
            model_class=XGBRegressor,
            adapter_class=SklearnTSAdapter,
            default_params={
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'random_state': 42
            },
            param_space_fn=_xgb_param_space
        ),
        TSModelType.LGBM: TSModelConfig(
            model_class=lgbm.LGBMRegressor,
            adapter_class=SklearnTSAdapter,
            default_params={
                'n_estimators': 100,
                'learning_rate': 0.1,
                'num_leaves': 31,
                'random_state': 42
            },
            param_space_fn=_lgbm_param_space
        ),
        TSModelType.ARIMA: TSModelConfig(
            model_class=ARIMA,
            adapter_class=ARIMATSAdapter,
            default_params={
                'order': (1, 0, 1)  # (p, d, q)
            },
            param_space_fn=_arima_param_space,
            is_univariate=True
        )
    }
    
    @classmethod
    def get_model(cls, model_type: TSModelType, custom_params: Dict[str, Any] = None) -> BaseTimeSeriesAdapter:
        """
        Cria uma instância de modelo com os parâmetros especificados.
        
        Args:
            model_type: Tipo de modelo a ser instanciado
            custom_params: Parâmetros personalizados (sobrepõe os padrões)
            
        Returns:
            Modelo adaptado pronto para treinamento
        """
        if model_type not in cls.SUPPORTED_MODELS:
            raise ValueError(f"Tipo de modelo não suportado: {model_type}")
            
        config = cls.SUPPORTED_MODELS[model_type]
        params = config.default_params.copy()
        
        if custom_params:
            params.update(custom_params)
            
        # Criação do modelo base
        model_instance = config.model_class(**params)
        
        # Criação do adapter
        adapter = config.adapter_class(model_instance, **params)
        
        return adapter
    
    @classmethod
    def get_param_space(cls, model_type: TSModelType, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Obtém espaço de parâmetros para otimização do modelo.
        
        Args:
            model_type: Tipo de modelo
            trial: Instância Optuna trial
            
        Returns:
            Dicionário de parâmetros para o trial
        """
        if model_type not in cls.SUPPORTED_MODELS:
            raise ValueError(f"Tipo de modelo não suportado: {model_type}")
            
        config = cls.SUPPORTED_MODELS[model_type]
        param_space = config.param_space_fn(trial)
        
        # Adicionar random_state para reprodutibilidade se o modelo suportar
        if 'random_state' in config.default_params:
            param_space['random_state'] = config.default_params['random_state']
            
        return param_space
    
    @classmethod
    def is_univariate(cls, model_type: TSModelType) -> bool:
        """
        Verifica se o modelo é univariado (usa apenas a série temporal).
        
        Args:
            model_type: Tipo de modelo
            
        Returns:
            True se o modelo for univariado, False caso contrário
        """
        if model_type not in cls.SUPPORTED_MODELS:
            raise ValueError(f"Tipo de modelo não suportado: {model_type}")
            
        return cls.SUPPORTED_MODELS[model_type].is_univariate