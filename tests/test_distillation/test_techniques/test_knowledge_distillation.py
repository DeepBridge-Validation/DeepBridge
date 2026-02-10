"""
Testes para deepbridge.distillation.techniques.knowledge_distillation.KnowledgeDistillation

Objetivo: Elevar coverage de 7.67% para 90%+
Foco: Inicialização, fit (com/sem Optuna), predict, predict_proba, evaluate,
      from_probabilities, temperature scaling, e processamento de probabilidades
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from deepbridge.distillation.techniques.knowledge_distillation import KnowledgeDistillation
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

    return X, y


@pytest.fixture
def trained_teacher_model(binary_classification_data):
    """Cria e treina um modelo professor."""
    X, y = binary_classification_data

    teacher = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=3)
    teacher.fit(X, y)

    return teacher


@pytest.fixture
def teacher_probabilities(trained_teacher_model, binary_classification_data):
    """Gera probabilidades do professor."""
    X, y = binary_classification_data

    probas = trained_teacher_model.predict_proba(X)
    return probas


# ============================================================================
# Testes de Inicialização
# ============================================================================

class TestKnowledgeDistillationInitialization:
    """Testes de inicialização do KnowledgeDistillation."""

    def test_init_with_teacher_model(self, trained_teacher_model):
        """Testa inicialização com modelo professor."""
        kd = KnowledgeDistillation(teacher_model=trained_teacher_model)

        assert kd.teacher_model is trained_teacher_model
        assert kd.teacher_probabilities is None
        assert kd.student_model_type == ModelType.LOGISTIC_REGRESSION
        assert kd.temperature == 1.0
        assert kd.alpha == 0.5
        assert kd.n_trials == 50
        assert kd.validation_split == 0.2
        assert kd.random_state == 42
        assert kd.student_model is None
        assert kd.best_params is None

    def test_init_with_teacher_probabilities(self, teacher_probabilities):
        """Testa inicialização com probabilidades pré-calculadas."""
        kd = KnowledgeDistillation(teacher_probabilities=teacher_probabilities)

        assert kd.teacher_model is None
        assert kd.teacher_probabilities is not None
        np.testing.assert_array_equal(kd.teacher_probabilities, teacher_probabilities)

    def test_init_with_custom_parameters(self, trained_teacher_model):
        """Testa inicialização com parâmetros customizados."""
        custom_params = {'max_iter': 200, 'C': 0.5}

        kd = KnowledgeDistillation(
            teacher_model=trained_teacher_model,
            student_model_type=ModelType.LOGISTIC_REGRESSION,
            student_params=custom_params,
            temperature=2.0,
            alpha=0.7,
            n_trials=10,
            validation_split=0.3,
            random_state=123
        )

        assert kd.student_params == custom_params
        assert kd.temperature == 2.0
        assert kd.alpha == 0.7
        assert kd.n_trials == 10
        assert kd.validation_split == 0.3
        assert kd.random_state == 123

    def test_init_without_teacher_raises_error(self):
        """Testa que inicialização sem professor levanta erro."""
        with pytest.raises(ValueError, match="Either teacher_model or teacher_probabilities must be provided"):
            KnowledgeDistillation()

    def test_init_with_both_teacher_and_probabilities(self, trained_teacher_model, teacher_probabilities):
        """Testa inicialização com ambos (deve usar probabilities)."""
        kd = KnowledgeDistillation(
            teacher_model=trained_teacher_model,
            teacher_probabilities=teacher_probabilities
        )

        # Ambos devem estar presentes
        assert kd.teacher_model is not None
        assert kd.teacher_probabilities is not None


# ============================================================================
# Testes de from_probabilities (Class Method)
# ============================================================================

class TestFromProbabilities:
    """Testes do método from_probabilities."""

    def test_from_probabilities_with_numpy_array(self):
        """Testa criação com array numpy."""
        probas = np.array([[0.3, 0.7], [0.4, 0.6], [0.2, 0.8]])

        kd = KnowledgeDistillation.from_probabilities(probas)

        assert isinstance(kd, KnowledgeDistillation)
        assert kd.teacher_model is None
        assert kd.teacher_probabilities is not None
        np.testing.assert_array_almost_equal(kd.teacher_probabilities, probas)

    def test_from_probabilities_with_dataframe(self):
        """Testa criação com DataFrame pandas."""
        df = pd.DataFrame({
            'prob_class_0': [0.3, 0.4, 0.2],
            'prob_class_1': [0.7, 0.6, 0.8]
        })

        kd = KnowledgeDistillation.from_probabilities(df)

        assert kd.teacher_probabilities is not None
        expected = np.array([[0.3, 0.7], [0.4, 0.6], [0.2, 0.8]])
        np.testing.assert_array_almost_equal(kd.teacher_probabilities, expected)

    def test_from_probabilities_with_single_column(self):
        """Testa criação com probabilidades de uma coluna."""
        # 1D array deve ser convertido para 2D com 1 coluna primeiro
        probas = np.array([[0.7], [0.6], [0.8]])

        kd = KnowledgeDistillation.from_probabilities(probas)

        # Deve converter para 2 colunas
        assert kd.teacher_probabilities.shape == (3, 2)
        expected = np.array([[0.3, 0.7], [0.4, 0.6], [0.2, 0.8]])
        np.testing.assert_array_almost_equal(kd.teacher_probabilities, expected)

    def test_from_probabilities_with_custom_params(self):
        """Testa from_probabilities com parâmetros customizados."""
        probas = np.array([[0.3, 0.7], [0.4, 0.6]])

        kd = KnowledgeDistillation.from_probabilities(
            probas,
            student_model_type=ModelType.GBM,
            temperature=3.0,
            alpha=0.8,
            n_trials=5,
            validation_split=0.25,
            random_state=999
        )

        assert kd.student_model_type == ModelType.GBM
        assert kd.temperature == 3.0
        assert kd.alpha == 0.8
        assert kd.n_trials == 5
        assert kd.validation_split == 0.25
        assert kd.random_state == 999

    def test_from_probabilities_normalizes_if_needed(self):
        """Testa que from_probabilities normaliza probabilidades."""
        # Probabilidades que não somam 1
        probas = np.array([[0.6, 1.4], [0.8, 1.2]])

        kd = KnowledgeDistillation.from_probabilities(probas)

        # Deve normalizar para somar 1
        row_sums = kd.teacher_probabilities.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(len(probas)))


# ============================================================================
# Testes de Fit
# ============================================================================

class TestKnowledgeDistillationFit:
    """Testes do método fit."""

    def test_fit_with_teacher_model(self, trained_teacher_model, binary_classification_data, capsys):
        """Testa fit com modelo professor."""
        X, y = binary_classification_data

        # Use n_trials pequeno para acelerar o teste
        kd = KnowledgeDistillation(
            teacher_model=trained_teacher_model,
            n_trials=2,
            random_state=42
        )

        result = kd.fit(X, y, verbose=False)

        # Verificar que retorna self
        assert result is kd

        # Verificar que student_model foi treinado
        assert kd.student_model is not None
        assert kd.best_params is not None

    def test_fit_with_teacher_probabilities(self, teacher_probabilities, binary_classification_data):
        """Testa fit com probabilidades pré-calculadas."""
        X, y = binary_classification_data

        kd = KnowledgeDistillation(
            teacher_probabilities=teacher_probabilities,
            n_trials=2,
            random_state=42
        )

        kd.fit(X, y, verbose=False)

        assert kd.student_model is not None

    def test_fit_with_provided_student_params(self, trained_teacher_model, binary_classification_data):
        """Testa fit com parâmetros fornecidos (sem Optuna)."""
        X, y = binary_classification_data

        custom_params = {'max_iter': 100, 'C': 1.0}
        kd = KnowledgeDistillation(
            teacher_model=trained_teacher_model,
            student_params=custom_params,
            random_state=42
        )

        kd.fit(X, y, verbose=False)

        # Verificar que não rodou Optuna (best_params == student_params)
        assert kd.best_params == custom_params
        assert kd.student_model is not None

    def test_fit_with_verbose_true(self, trained_teacher_model, binary_classification_data, capsys):
        """Testa fit com verbose=True."""
        X, y = binary_classification_data

        kd = KnowledgeDistillation(
            teacher_model=trained_teacher_model,
            n_trials=2,
            random_state=42
        )

        kd.fit(X, y, verbose=True)

        # Verificar que imprimiu algo
        captured = capsys.readouterr()
        assert 'Best hyperparameters found' in captured.out

    def test_fit_with_different_temperatures(self, trained_teacher_model, binary_classification_data):
        """Testa fit com diferentes temperaturas."""
        X, y = binary_classification_data

        for temp in [0.5, 1.0, 2.0, 5.0]:
            kd = KnowledgeDistillation(
                teacher_model=trained_teacher_model,
                temperature=temp,
                n_trials=2,
                random_state=42
            )

            kd.fit(X, y, verbose=False)
            assert kd.student_model is not None

    def test_fit_with_different_alphas(self, trained_teacher_model, binary_classification_data):
        """Testa fit com diferentes valores de alpha."""
        X, y = binary_classification_data

        for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
            kd = KnowledgeDistillation(
                teacher_model=trained_teacher_model,
                alpha=alpha,
                n_trials=2,
                random_state=42
            )

            kd.fit(X, y, verbose=False)
            assert kd.student_model is not None


# ============================================================================
# Testes de Predict
# ============================================================================

class TestKnowledgeDistillationPredict:
    """Testes dos métodos predict e predict_proba."""

    def test_predict_after_fit(self, trained_teacher_model, binary_classification_data):
        """Testa predict após fit."""
        X, y = binary_classification_data

        kd = KnowledgeDistillation(
            teacher_model=trained_teacher_model,
            n_trials=2,
            random_state=42
        )
        kd.fit(X, y, verbose=False)

        predictions = kd.predict(X)

        # Verificar formato
        assert predictions.shape == (len(X),)

        # Verificar que são 0 ou 1
        assert np.all(np.isin(predictions, [0, 1]))

    def test_predict_proba_after_fit(self, trained_teacher_model, binary_classification_data):
        """Testa predict_proba após fit."""
        X, y = binary_classification_data

        kd = KnowledgeDistillation(
            teacher_model=trained_teacher_model,
            n_trials=2,
            random_state=42
        )
        kd.fit(X, y, verbose=False)

        probas = kd.predict_proba(X)

        # Verificar formato
        assert probas.shape == (len(X), 2)

        # Verificar que somam 1
        np.testing.assert_array_almost_equal(
            probas.sum(axis=1),
            np.ones(len(X)),
            decimal=5
        )

    def test_predict_without_fit_raises_error(self, trained_teacher_model, binary_classification_data):
        """Testa que predict sem fit levanta erro."""
        X, y = binary_classification_data

        kd = KnowledgeDistillation(teacher_model=trained_teacher_model)

        with pytest.raises(RuntimeError, match="Model not trained"):
            kd.predict(X)

    def test_predict_proba_without_fit_raises_error(self, trained_teacher_model, binary_classification_data):
        """Testa que predict_proba sem fit levanta erro."""
        X, y = binary_classification_data

        kd = KnowledgeDistillation(teacher_model=trained_teacher_model)

        with pytest.raises(RuntimeError, match="Model not trained"):
            kd.predict_proba(X)


# ============================================================================
# Testes de Evaluate
# ============================================================================

class TestKnowledgeDistillationEvaluate:
    """Testes do método evaluate."""

    def test_evaluate_returns_metrics(self, trained_teacher_model, binary_classification_data):
        """Testa que evaluate retorna métricas."""
        X, y = binary_classification_data

        kd = KnowledgeDistillation(
            teacher_model=trained_teacher_model,
            n_trials=2,
            random_state=42
        )
        kd.fit(X, y, verbose=False)

        metrics = kd.evaluate(X, y, return_predictions=False)

        # Verificar que contém métricas esperadas
        assert 'accuracy' in metrics
        assert 'best_params' in metrics
        assert isinstance(metrics['accuracy'], (int, float))

    def test_evaluate_with_return_predictions(self, trained_teacher_model, binary_classification_data):
        """Testa evaluate com return_predictions=True."""
        X, y = binary_classification_data

        kd = KnowledgeDistillation(
            teacher_model=trained_teacher_model,
            n_trials=2,
            random_state=42
        )
        kd.fit(X, y, verbose=False)

        result = kd.evaluate(X, y, return_predictions=True)

        # Verificar estrutura
        assert 'metrics' in result
        assert 'predictions' in result
        assert isinstance(result['predictions'], pd.DataFrame)

        # Verificar colunas do DataFrame
        df = result['predictions']
        assert 'y_true' in df.columns
        assert 'y_pred' in df.columns
        assert 'y_prob' in df.columns
        assert 'teacher_prob' in df.columns

    def test_evaluate_without_fit_raises_error(self, trained_teacher_model, binary_classification_data):
        """Testa que evaluate sem fit levanta erro."""
        X, y = binary_classification_data

        kd = KnowledgeDistillation(teacher_model=trained_teacher_model)

        with pytest.raises(RuntimeError, match="Model not trained"):
            kd.evaluate(X, y)

    def test_evaluate_from_dataframe(self, trained_teacher_model, binary_classification_data):
        """Testa evaluate_from_dataframe."""
        X, y = binary_classification_data

        kd = KnowledgeDistillation(
            teacher_model=trained_teacher_model,
            n_trials=2,
            random_state=42
        )
        kd.fit(X, y, verbose=False)

        # Criar DataFrame
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        df['target'] = y

        metrics = kd.evaluate_from_dataframe(
            data=df,
            features_columns=[f'feature_{i}' for i in range(X.shape[1])],
            target_column='target',
            return_predictions=False
        )

        assert 'accuracy' in metrics


# ============================================================================
# Testes de Métodos Internos
# ============================================================================

class TestKnowledgeDistillationInternalMethods:
    """Testes de métodos internos."""

    def test_get_teacher_soft_labels_with_model(self, trained_teacher_model, binary_classification_data):
        """Testa _get_teacher_soft_labels com modelo."""
        X, y = binary_classification_data

        kd = KnowledgeDistillation(
            teacher_model=trained_teacher_model,
            random_state=42
        )

        soft_labels = kd._get_teacher_soft_labels(X[:10])

        # Verificar formato
        assert soft_labels.shape == (10, 2)

        # Verificar que somam ~1
        row_sums = soft_labels.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(10), decimal=5)

    def test_get_teacher_soft_labels_with_probabilities(self, teacher_probabilities, binary_classification_data):
        """Testa _get_teacher_soft_labels com probabilidades."""
        X, y = binary_classification_data

        # Use as mesmas primeiras 10 amostras para X e probabilities
        kd = KnowledgeDistillation(
            teacher_probabilities=teacher_probabilities[:10],
            random_state=42
        )

        soft_labels = kd._get_teacher_soft_labels(X[:10])

        assert soft_labels.shape == (10, 2)

    def test_get_teacher_soft_labels_with_dataframe_probabilities(self, binary_classification_data):
        """Testa _get_teacher_soft_labels com DataFrame."""
        X, y = binary_classification_data

        # Criar DataFrame com colunas nomeadas
        probas_df = pd.DataFrame({
            'prob_class_0': [0.3] * 10,
            'prob_class_1': [0.7] * 10
        })

        kd = KnowledgeDistillation(
            teacher_probabilities=probas_df,
            random_state=42
        )

        soft_labels = kd._get_teacher_soft_labels(X[:10])

        assert soft_labels.shape == (10, 2)

    def test_get_teacher_soft_labels_with_mismatched_length_raises_error(self, teacher_probabilities, binary_classification_data):
        """Testa que comprimentos diferentes levantam erro."""
        X, y = binary_classification_data

        # Usar menos probabilidades que amostras
        kd = KnowledgeDistillation(
            teacher_probabilities=teacher_probabilities[:50],
            random_state=42
        )

        with pytest.raises(ValueError, match="doesn't match number of samples"):
            kd._get_teacher_soft_labels(X)

    def test_kl_divergence(self, trained_teacher_model):
        """Testa cálculo de KL divergence."""
        kd = KnowledgeDistillation(teacher_model=trained_teacher_model)

        p = np.array([[0.3, 0.7], [0.4, 0.6]])
        q = np.array([[0.35, 0.65], [0.45, 0.55]])

        kl = kd._kl_divergence(p, q)

        # KL divergence deve ser >= 0
        assert kl >= 0
        assert isinstance(kl, (int, float))

    def test_combined_loss(self, trained_teacher_model):
        """Testa cálculo de combined loss."""
        kd = KnowledgeDistillation(
            teacher_model=trained_teacher_model,
            alpha=0.5
        )

        y_true = np.array([[1, 0], [0, 1]])
        soft_labels = np.array([[0.8, 0.2], [0.3, 0.7]])
        student_probs = np.array([[0.75, 0.25], [0.35, 0.65]])

        loss = kd._combined_loss(y_true, soft_labels, student_probs)

        # Loss deve ser >= 0
        assert loss >= 0
        assert isinstance(loss, (int, float))


# ============================================================================
# Testes de Edge Cases
# ============================================================================

class TestKnowledgeDistillationEdgeCases:
    """Testes de casos extremos."""

    def test_fit_with_small_dataset(self, trained_teacher_model):
        """Testa fit com dataset pequeno."""
        np.random.seed(42)
        X = np.random.rand(30, 5)
        y = np.random.randint(0, 2, 30)

        kd = KnowledgeDistillation(
            teacher_model=trained_teacher_model,
            n_trials=2,
            validation_split=0.2,
            random_state=42
        )

        kd.fit(X, y, verbose=False)
        assert kd.student_model is not None

    def test_fit_with_different_student_model_types(self, trained_teacher_model, binary_classification_data):
        """Testa fit com diferentes tipos de modelo aluno."""
        X, y = binary_classification_data

        for model_type in [ModelType.LOGISTIC_REGRESSION, ModelType.DECISION_TREE]:
            kd = KnowledgeDistillation(
                teacher_model=trained_teacher_model,
                student_model_type=model_type,
                n_trials=2,
                random_state=42
            )

            kd.fit(X, y, verbose=False)
            predictions = kd.predict(X)
            assert predictions.shape == (len(X),)

    def test_from_probabilities_with_unnormalized_probabilities(self):
        """Testa from_probabilities com probabilidades não normalizadas."""
        # Probabilidades que não somam exatamente 1
        probas = np.array([[0.31, 0.69], [0.41, 0.58], [0.22, 0.77]])

        kd = KnowledgeDistillation.from_probabilities(probas)

        # Deve normalizar
        row_sums = kd.teacher_probabilities.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(3), decimal=5)

    def test_temperature_scaling_effect(self, trained_teacher_model, binary_classification_data):
        """Testa efeito do temperature scaling."""
        X, y = binary_classification_data
        X_sample = X[:5]

        kd_low_temp = KnowledgeDistillation(
            teacher_model=trained_teacher_model,
            temperature=0.5,
            random_state=42
        )

        kd_high_temp = KnowledgeDistillation(
            teacher_model=trained_teacher_model,
            temperature=5.0,
            random_state=42
        )

        soft_labels_low = kd_low_temp._get_teacher_soft_labels(X_sample)
        soft_labels_high = kd_high_temp._get_teacher_soft_labels(X_sample)

        # Com temperatura alta, probabilidades devem ser mais uniformes
        # (menor diferença entre classes)
        diff_low = np.abs(soft_labels_low[:, 0] - soft_labels_low[:, 1])
        diff_high = np.abs(soft_labels_high[:, 0] - soft_labels_high[:, 1])

        # Em média, diferenças devem ser menores com temperatura maior
        assert np.mean(diff_high) < np.mean(diff_low)

    def test_alpha_zero_only_hard_loss(self, trained_teacher_model, binary_classification_data):
        """Testa alpha=0 (apenas hard loss)."""
        X, y = binary_classification_data

        kd = KnowledgeDistillation(
            teacher_model=trained_teacher_model,
            alpha=0.0,
            n_trials=2,
            random_state=42
        )

        kd.fit(X, y, verbose=False)
        assert kd.student_model is not None

    def test_alpha_one_only_soft_loss(self, trained_teacher_model, binary_classification_data):
        """Testa alpha=1 (apenas soft loss)."""
        X, y = binary_classification_data

        kd = KnowledgeDistillation(
            teacher_model=trained_teacher_model,
            alpha=1.0,
            n_trials=2,
            random_state=42
        )

        kd.fit(X, y, verbose=False)
        assert kd.student_model is not None
