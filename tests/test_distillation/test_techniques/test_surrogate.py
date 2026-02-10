"""
Testes para deepbridge.distillation.techniques.surrogate.SurrogateModel

Objetivo: Elevar coverage de 52.83% para 90%+
Foco: Inicialização, fit, predict, predict_proba, evaluate, e _process_probabilities
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from deepbridge.distillation.techniques.surrogate import SurrogateModel
from deepbridge.utils.model_registry import ModelType


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def binary_classification_data():
    """Cria dados para classificação binária."""
    np.random.seed(42)
    n_samples = 200

    X = np.random.rand(n_samples, 5)
    y = (X[:, 0] + X[:, 1] > 1.0).astype(int)

    # Treinar teacher model para gerar probabilidades
    teacher = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=3)
    teacher.fit(X, y)
    probas = teacher.predict_proba(X)[:, 1]  # Positive class probabilities

    return X, y, probas


@pytest.fixture
def surrogate_model():
    """Cria modelo surrogate não treinado."""
    return SurrogateModel(model_type=ModelType.GBM)


# ============================================================================
# Testes de Inicialização
# ============================================================================

class TestSurrogateModelInitialization:
    """Testes de inicialização do SurrogateModel."""

    def test_init_with_default_params(self):
        """Testa inicialização com parâmetros padrão."""
        model = SurrogateModel()

        assert model.model_type == ModelType.GBM
        assert model.model_params == {}
        assert model.is_fitted == False
        assert model.model is not None
        assert model.metrics_calculator is not None

    def test_init_with_custom_model_type(self):
        """Testa inicialização com tipo de modelo customizado."""
        model = SurrogateModel(model_type=ModelType.DECISION_TREE)

        assert model.model_type == ModelType.DECISION_TREE

    def test_init_with_custom_params(self):
        """Testa inicialização com parâmetros customizados."""
        custom_params = {'max_depth': 5, 'random_state': 123}
        model = SurrogateModel(
            model_type=ModelType.DECISION_TREE,
            model_params=custom_params
        )

        assert model.model_params == custom_params


# ============================================================================
# Testes de Fit
# ============================================================================

class TestSurrogateModelFit:
    """Testes do método fit."""

    def test_fit_with_numpy_arrays(self, binary_classification_data):
        """Testa fit com arrays numpy."""
        X, y, probas = binary_classification_data

        model = SurrogateModel()
        result = model.fit(X, probas, verbose=False)

        # Verificar que retorna self
        assert result is model

        # Verificar que modelo foi treinado
        assert model.is_fitted == True

    def test_fit_with_pandas_dataframe(self, binary_classification_data):
        """Testa fit com DataFrame pandas."""
        X, y, probas = binary_classification_data

        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        probas_series = pd.Series(probas)

        model = SurrogateModel()
        model.fit(X_df, probas_series, verbose=False)

        assert model.is_fitted == True

    def test_fit_with_verbose_output(self, binary_classification_data, capsys):
        """Testa fit com verbose=True."""
        X, y, probas = binary_classification_data

        model = SurrogateModel()
        model.fit(X, probas, verbose=True)

        # Verificar que imprimiu métricas
        captured = capsys.readouterr()
        assert 'Surrogate Model Training Results' in captured.out
        assert 'Train metrics' in captured.out
        assert 'Test metrics' in captured.out

    def test_fit_stores_metrics_when_verbose(self, binary_classification_data):
        """Testa que fit armazena métricas quando verbose=True."""
        X, y, probas = binary_classification_data

        model = SurrogateModel()
        model.fit(X, probas, verbose=True)

        # Verificar que métricas foram armazenadas
        assert hasattr(model, 'train_metrics')
        assert hasattr(model, 'test_metrics')
        assert 'accuracy' in model.train_metrics
        assert 'auc_roc' in model.test_metrics

    def test_fit_with_custom_test_size(self, binary_classification_data):
        """Testa fit com test_size customizado."""
        X, y, probas = binary_classification_data

        model = SurrogateModel()
        model.fit(X, probas, test_size=0.3, verbose=False)

        assert model.is_fitted == True

    def test_fit_with_custom_random_state(self, binary_classification_data):
        """Testa fit com random_state customizado."""
        X, y, probas = binary_classification_data

        model1 = SurrogateModel()
        model1.fit(X, probas, random_state=42, verbose=False)

        model2 = SurrogateModel()
        model2.fit(X, probas, random_state=42, verbose=False)

        # Mesmas predições com mesmo random_state
        preds1 = model1.predict(X[:10])
        preds2 = model2.predict(X[:10])

        np.testing.assert_array_equal(preds1, preds2)


# ============================================================================
# Testes de Predict
# ============================================================================

class TestSurrogateModelPredict:
    """Testes dos métodos predict e predict_proba."""

    def test_predict_returns_binary(self, binary_classification_data):
        """Testa que predict retorna valores binários."""
        X, y, probas = binary_classification_data

        model = SurrogateModel()
        model.fit(X, probas, verbose=False)

        predictions = model.predict(X)

        # Verificar formato
        assert predictions.shape == (len(X),)

        # Verificar que são 0 ou 1
        assert np.all(np.isin(predictions, [0, 1]))

    def test_predict_proba_returns_two_columns(self, binary_classification_data):
        """Testa que predict_proba retorna 2 colunas."""
        X, y, probas = binary_classification_data

        model = SurrogateModel()
        model.fit(X, probas, verbose=False)

        probabilities = model.predict_proba(X)

        # Verificar formato
        assert probabilities.shape == (len(X), 2)

        # Verificar que somam 1
        np.testing.assert_array_almost_equal(
            probabilities.sum(axis=1),
            np.ones(len(X)),
            decimal=5
        )

    def test_predict_without_fit_raises_error(self, binary_classification_data):
        """Testa que predict sem fit levanta erro."""
        X, y, probas = binary_classification_data

        model = SurrogateModel()

        with pytest.raises(ValueError, match="must be fitted"):
            model.predict(X)

    def test_predict_proba_without_fit_raises_error(self, binary_classification_data):
        """Testa que predict_proba sem fit levanta erro."""
        X, y, probas = binary_classification_data

        model = SurrogateModel()

        with pytest.raises(ValueError, match="must be fitted"):
            model.predict_proba(X)


# ============================================================================
# Testes de _process_probabilities
# ============================================================================

class TestProcessProbabilities:
    """Testes do método _process_probabilities."""

    def test_process_dataframe_with_prob_class_1(self):
        """Testa processamento de DataFrame com coluna 'prob_class_1'."""
        model = SurrogateModel()

        df = pd.DataFrame({
            'prob_class_0': [0.3, 0.4, 0.2],
            'prob_class_1': [0.7, 0.6, 0.8]
        })

        result = model._process_probabilities(df)

        np.testing.assert_array_equal(result, np.array([0.7, 0.6, 0.8]))

    def test_process_dataframe_with_prob_1(self):
        """Testa processamento de DataFrame com coluna 'prob_1'."""
        model = SurrogateModel()

        df = pd.DataFrame({
            'prob_0': [0.3, 0.4, 0.2],
            'prob_1': [0.7, 0.6, 0.8]
        })

        result = model._process_probabilities(df)

        np.testing.assert_array_equal(result, np.array([0.7, 0.6, 0.8]))

    def test_process_dataframe_default_second_column(self):
        """Testa processamento de DataFrame com coluna padrão."""
        model = SurrogateModel()

        df = pd.DataFrame({
            'col0': [0.3, 0.4, 0.2],
            'col1': [0.7, 0.6, 0.8]
        })

        result = model._process_probabilities(df)

        # Deve usar segunda coluna por padrão
        np.testing.assert_array_equal(result, np.array([0.7, 0.6, 0.8]))

    def test_process_dataframe_single_column(self):
        """Testa processamento de DataFrame com uma coluna."""
        model = SurrogateModel()

        df = pd.DataFrame({'proba': [0.7, 0.6, 0.8]})

        result = model._process_probabilities(df)

        np.testing.assert_array_equal(result, np.array([0.7, 0.6, 0.8]))

    def test_process_series(self):
        """Testa processamento de Series."""
        model = SurrogateModel()

        series = pd.Series([0.7, 0.6, 0.8])

        result = model._process_probabilities(series)

        np.testing.assert_array_equal(result, np.array([0.7, 0.6, 0.8]))

    def test_process_numpy_2d_array(self):
        """Testa processamento de array 2D numpy."""
        model = SurrogateModel()

        array = np.array([[0.3, 0.7], [0.4, 0.6], [0.2, 0.8]])

        result = model._process_probabilities(array)

        # Deve retornar segunda coluna
        np.testing.assert_array_equal(result, np.array([0.7, 0.6, 0.8]))

    def test_process_numpy_1d_array(self):
        """Testa processamento de array 1D numpy."""
        model = SurrogateModel()

        array = np.array([0.7, 0.6, 0.8])

        result = model._process_probabilities(array)

        np.testing.assert_array_equal(result, np.array([0.7, 0.6, 0.8]))

    def test_process_invalid_type_raises_error(self):
        """Testa que tipo inválido levanta erro."""
        model = SurrogateModel()

        with pytest.raises(ValueError, match="Unsupported probability format"):
            model._process_probabilities([0.7, 0.6, 0.8])  # List não é suportado


# ============================================================================
# Testes de Evaluate
# ============================================================================

class TestSurrogateModelEvaluate:
    """Testes do método evaluate."""

    def test_evaluate_returns_metrics(self, binary_classification_data):
        """Testa que evaluate retorna métricas."""
        X, y, probas = binary_classification_data

        model = SurrogateModel()
        model.fit(X, probas, verbose=False)

        metrics = model.evaluate(X, y)

        # Verificar que contém métricas esperadas
        assert 'accuracy' in metrics
        assert 'auc_roc' in metrics
        assert isinstance(metrics['accuracy'], (int, float))
        assert isinstance(metrics['auc_roc'], (int, float))

    def test_evaluate_with_teacher_prob(self, binary_classification_data):
        """Testa evaluate com teacher probabilities."""
        X, y, probas = binary_classification_data

        model = SurrogateModel()
        model.fit(X, probas, verbose=False)

        metrics = model.evaluate(X, y, teacher_prob=probas)

        # Deve retornar métricas
        assert 'accuracy' in metrics

    def test_evaluate_without_fit_raises_error(self, binary_classification_data):
        """Testa que evaluate sem fit levanta erro."""
        X, y, probas = binary_classification_data

        model = SurrogateModel()

        with pytest.raises(ValueError, match="must be fitted"):
            model.evaluate(X, y)


# ============================================================================
# Testes de from_probabilities (Class Method)
# ============================================================================

class TestFromProbabilities:
    """Testes do método from_probabilities."""

    def test_from_probabilities_creates_model(self):
        """Testa que from_probabilities cria modelo."""
        probas = np.array([[0.3, 0.7], [0.4, 0.6]])

        model = SurrogateModel.from_probabilities(probas)

        assert isinstance(model, SurrogateModel)
        assert model.model_type == ModelType.GBM

    def test_from_probabilities_with_custom_model_type(self):
        """Testa from_probabilities com tipo customizado."""
        probas = np.array([[0.3, 0.7], [0.4, 0.6]])

        model = SurrogateModel.from_probabilities(
            probas,
            student_model_type=ModelType.DECISION_TREE
        )

        assert model.model_type == ModelType.DECISION_TREE

    def test_from_probabilities_with_custom_params(self):
        """Testa from_probabilities com parâmetros customizados."""
        probas = np.array([[0.3, 0.7], [0.4, 0.6]])

        custom_params = {'max_depth': 5}
        model = SurrogateModel.from_probabilities(
            probas,
            student_params=custom_params,
            random_state=42
        )

        # Verificar que random_state foi adicionado
        assert 'random_state' in model.model_params
        assert model.model_params['random_state'] == 42

    def test_from_probabilities_with_compatibility_params(self):
        """Testa from_probabilities com parâmetros de compatibilidade."""
        probas = np.array([[0.3, 0.7], [0.4, 0.6]])

        # Estes parâmetros são ignorados mas incluídos para compatibilidade
        model = SurrogateModel.from_probabilities(
            probas,
            validation_split=0.3,
            n_trials=5
        )

        assert isinstance(model, SurrogateModel)


# ============================================================================
# Testes de Edge Cases
# ============================================================================

class TestSurrogateModelEdgeCases:
    """Testes de casos extremos."""

    def test_fit_with_extreme_probabilities(self):
        """Testa fit com probabilidades extremas."""
        np.random.seed(42)
        X = np.random.rand(100, 5)

        # Probabilidades muito próximas de 0 e 1
        probas = np.array([0.01] * 50 + [0.99] * 50)

        model = SurrogateModel()
        model.fit(X, probas, verbose=False)

        assert model.is_fitted == True

    def test_predict_with_pandas_input(self, binary_classification_data):
        """Testa predict com DataFrame pandas."""
        X, y, probas = binary_classification_data

        model = SurrogateModel()
        model.fit(X, probas, verbose=False)

        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        predictions = model.predict(X_df)

        assert predictions.shape == (len(X),)

    def test_different_model_types(self, binary_classification_data):
        """Testa com diferentes tipos de modelo."""
        X, y, probas = binary_classification_data

        for model_type in [ModelType.GBM, ModelType.DECISION_TREE]:
            model = SurrogateModel(model_type=model_type)
            model.fit(X, probas, verbose=False)

            predictions = model.predict(X)
            assert predictions.shape == (len(X),)
