import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from scipy.special import softmax
from sklearn.metrics import accuracy_score, roc_auc_score

class EnsembleDistillation(BaseEstimator, ClassifierMixin):
    def __init__(self, teacher_models, student_model, temperature=1.0):
        """
        Construtor da classe.
        
        Args:
            teacher_models: Lista de modelos professores (ensemble) pré-treinados.
            student_model: Modelo aluno a ser destilado.
            temperature: Temperatura para suavizar as probabilidades do ensemble.
        """
        self.teacher_models = teacher_models
        self.student_model = student_model
        self.temperature = temperature

    def fit(self, X, y):
        """
        Treina o modelo aluno usando as previsões do ensemble de professores.
        """
        # Gerar previsões suavizadas do ensemble
        ensemble_probas = self._get_ensemble_probas(X)
        
        # Treinar o modelo aluno nas previsões do ensemble
        self.student_model.fit(X, ensemble_probas)
        return self

    def _get_ensemble_probas(self, X):
        """
        Combina as previsões dos modelos professores usando média das probabilidades.
        """
        probas = []
        for model in self.teacher_models:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)
            else:
                # Se o modelo não retornar probabilidades, usar one-hot encoding
                pred = model.predict(X)
                proba = np.zeros((len(pred), 2))
                proba[np.arange(len(pred)), pred] = 1
            probas.append(proba)
        
        # Média das probabilidades do ensemble com temperature scaling
        avg_probas = np.mean(probas, axis=0)
        scaled_probas = softmax(avg_probas / self.temperature)
        return scaled_probas

    def predict(self, X):
        """
        Faz previsões usando o modelo aluno.
        """
        check_is_fitted(self, 'student_model')
        return self.student_model.predict(X)

    def predict_proba(self, X):
        """
        Retorna as probabilidades do modelo aluno.
        """
        check_is_fitted(self, 'student_model')
        return self.student_model.predict_proba(X)

    def evaluate(self, X, y_true):
        """
        Avalia o desempenho do modelo aluno.
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "ROC AUC": roc_auc_score(y_true, y_proba)
        }