import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
from scipy.special import softmax


class TemperatureScaling(BaseEstimator, ClassifierMixin):
    def __init__(self, base_model, temperature=1.0):
        """
        Construtor da classe.

        Args:
            base_model: Modelo base para aplicar o escalonamento de temperatura.
            temperature: Temperatura para ajustar as probabilidades.
        """
        self.base_model = base_model
        self.temperature = temperature

    def fit(self, X, y):
        """
        Treina o modelo base (opcional, se o modelo já estiver pré-treinado).
        """
        self.base_model.fit(X, y)
        return self

    def predict_proba(self, X):
        """
        Aplica temperature scaling às probabilidades do modelo base.
        """
        logits = self.base_model.decision_function(X)
        scaled_logits = logits / self.temperature

        # Handle 1D logits for binary classification
        if scaled_logits.ndim == 1:
            scaled_logits = np.column_stack([-scaled_logits, scaled_logits])

        return softmax(scaled_logits, axis=1)

    def predict(self, X):
        """
        Retorna as previsões após o temperature scaling.
        """
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

    def calibrate_temperature(self, X_val, y_val):
        """
        Calibra a temperatura para melhorar a calibração das probabilidades.
        """
        # Implementação simplificada (exemplo usando otimização numérica)
        from scipy.optimize import minimize_scalar

        def objective(t):
            self.temperature = t
            y_proba = self.predict_proba(X_val)[:, 1]
            return -roc_auc_score(y_val, y_proba)  # Maximizar ROC AUC

        result = minimize_scalar(
            objective, bounds=(0.1, 10.0), method='bounded'
        )
        self.temperature = result.x
        return self
