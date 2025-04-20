import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

class Quantization(BaseEstimator, ClassifierMixin):
    def __init__(self, base_model, n_bits=8):
        """
        Construtor da classe.
        
        Args:
            base_model: Modelo base a ser quantizado.
            n_bits: Número de bits para quantização (padrão: 8).
        """
        self.base_model = base_model
        self.n_bits = n_bits

    def fit(self, X, y):
        """
        Treina o modelo base e aplica quantização aos seus pesos.
        """
        # Treinar o modelo base
        self.base_model_ = clone(self.base_model)
        self.base_model_.fit(X, y)
        
        # Quantizar os coeficientes do modelo
        self._quantize_weights()
        return self

    def _quantize_weights(self):
        """
        Aplica quantização aos pesos do modelo.
        """
        # Coeficientes do modelo
        coef = self.base_model_.coef_
        intercept = self.base_model_.intercept_

        # Quantização para inteiros de n_bits
        scale = (np.max(coef) - np.min(coef)) / (2**self.n_bits - 1)
        self.quantized_coef_ = np.round((coef - np.min(coef)) / scale).astype(np.int32)
        self.quantized_intercept_ = np.round((intercept - np.min(intercept)) / scale).astype(np.int32)

        # Desquantização (simulação para uso em previsões)
        self.base_model_.coef_ = self.quantized_coef_ * scale + np.min(coef)
        self.base_model_.intercept_ = self.quantized_intercept_ * scale + np.min(intercept)

    def predict(self, X):
        """
        Faz previsões usando o modelo quantizado.
        """
        check_is_fitted(self, 'base_model_')
        return self.base_model_.predict(X)

    def predict_proba(self, X):
        """
        Retorna probabilidades (se o modelo base suportar).
        """
        check_is_fitted(self, 'base_model_')
        return self.base_model_.predict_proba(X)