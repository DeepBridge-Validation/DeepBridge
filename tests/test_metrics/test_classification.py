"""
Testes para deepbridge.metrics.classification.Classification

Objetivo: Elevar coverage de 0% para 95%+
Foco: calculate_metrics, calculate_kl_divergence, calculate_ks_statistic, calculate_r2_score
"""

import pytest
import numpy as np
import pandas as pd

from deepbridge.metrics.classification import Classification


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def binary_classification_data():
    """Cria dados de classificação binária para testes."""
    np.random.seed(42)
    n_samples = 200

    # True labels (0 or 1)
    y_true = np.random.randint(0, 2, n_samples)

    # Predictions with some errors
    y_pred = y_true.copy()
    errors = np.random.choice(n_samples, size=20, replace=False)
    y_pred[errors] = 1 - y_pred[errors]  # Flip some predictions

    # Probabilities (correlated with y_true)
    y_prob = np.where(y_true == 1,
                      np.random.uniform(0.6, 0.95, n_samples),
                      np.random.uniform(0.05, 0.4, n_samples))

    # Teacher probabilities (slightly better)
    teacher_prob = np.where(y_true == 1,
                           np.random.uniform(0.7, 0.98, n_samples),
                           np.random.uniform(0.02, 0.3, n_samples))

    return y_true, y_pred, y_prob, teacher_prob


@pytest.fixture
def classification_dataframe(binary_classification_data):
    """Cria DataFrame com dados de classificação."""
    y_true, y_pred, y_prob, teacher_prob = binary_classification_data

    return pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'teacher_prob': teacher_prob
    })


# ============================================================================
# Testes de calculate_metrics - Básico
# ============================================================================

class TestClassificationCalculateMetricsBasic:
    """Testes básicos do método calculate_metrics."""

    def test_calculate_metrics_minimal(self, binary_classification_data):
        """Testa cálculo mínimo (apenas y_true e y_pred)."""
        y_true, y_pred, _, _ = binary_classification_data

        metrics = Classification.calculate_metrics(y_true, y_pred)

        # Verificar métricas básicas
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'f1-score' in metrics  # Compatibility alias

        # Verificar tipos
        assert isinstance(metrics['accuracy'], float)
        assert isinstance(metrics['precision'], float)
        assert isinstance(metrics['recall'], float)

        # F1 score deve ter ambos os nomes
        assert metrics['f1_score'] == metrics['f1-score']

    def test_calculate_metrics_with_probabilities(self, binary_classification_data):
        """Testa cálculo com probabilidades."""
        y_true, y_pred, y_prob, _ = binary_classification_data

        metrics = Classification.calculate_metrics(y_true, y_pred, y_prob)

        # Verificar métricas básicas
        assert 'accuracy' in metrics
        assert 'precision' in metrics

        # Verificar métricas que precisam de probabilidades
        assert 'roc_auc' in metrics
        assert 'auc_roc' in metrics  # Compatibility alias
        assert 'auc_pr' in metrics
        assert 'log_loss' in metrics

        # AUC ROC deve ter ambos os nomes
        assert metrics['roc_auc'] == metrics['auc_roc']

        # Verificar ranges
        assert 0 <= metrics['roc_auc'] <= 1
        assert 0 <= metrics['auc_pr'] <= 1
        assert metrics['log_loss'] >= 0

    def test_calculate_metrics_with_teacher(self, binary_classification_data):
        """Testa cálculo com teacher probabilities."""
        y_true, y_pred, y_prob, teacher_prob = binary_classification_data

        metrics = Classification.calculate_metrics(
            y_true, y_pred, y_prob, teacher_prob
        )

        # Verificar métricas de comparação com teacher
        assert 'kl_divergence' in metrics
        assert 'ks_statistic' in metrics
        assert 'ks_pvalue' in metrics
        assert 'r2_score' in metrics

        # Verificar que não são None
        assert metrics['kl_divergence'] is not None
        assert metrics['ks_statistic'] is not None
        assert metrics['r2_score'] is not None

        # Verificar ranges
        assert metrics['kl_divergence'] >= 0
        assert 0 <= metrics['ks_statistic'] <= 1
        assert 0 <= metrics['ks_pvalue'] <= 1

    def test_calculate_metrics_with_pandas_series(self, classification_dataframe):
        """Testa com pandas Series."""
        metrics = Classification.calculate_metrics(
            classification_dataframe['y_true'],
            classification_dataframe['y_pred'],
            classification_dataframe['y_prob'],
            classification_dataframe['teacher_prob']
        )

        assert 'accuracy' in metrics
        assert 'roc_auc' in metrics
        assert 'kl_divergence' in metrics


# ============================================================================
# Testes de calculate_metrics - Edge Cases
# ============================================================================

class TestClassificationCalculateMetricsEdgeCases:
    """Testes de casos extremos do calculate_metrics."""

    def test_calculate_metrics_perfect_predictions(self):
        """Testa com predições perfeitas."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = y_true.copy()
        y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.15, 0.85])

        metrics = Classification.calculate_metrics(y_true, y_pred, y_prob)

        # Accuracy deve ser 1.0
        assert metrics['accuracy'] == 1.0

        # Precision e recall devem ser 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1_score'] == 1.0

    def test_calculate_metrics_single_class_true(self):
        """Testa quando y_true tem apenas uma classe."""
        y_true = np.array([1, 1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 0, 1])
        y_prob = np.array([0.9, 0.8, 0.7, 0.4, 0.85])

        metrics = Classification.calculate_metrics(y_true, y_pred, y_prob)

        # Métricas básicas devem ser calculadas
        assert 'accuracy' in metrics
        assert 'precision' in metrics

        # AUC metrics devem ser None (single class)
        assert metrics['roc_auc'] is None
        assert metrics['auc_pr'] is None
        assert metrics['log_loss'] is None

    def test_calculate_metrics_single_class_pred(self):
        """Testa quando y_pred tem apenas uma classe."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([1, 1, 1, 1, 1, 1])  # All predict 1

        metrics = Classification.calculate_metrics(y_true, y_pred)

        # Accuracy deve ser calculada
        assert 'accuracy' in metrics

        # Precision, recall podem ser 0 ou valores específicos
        assert 'precision' in metrics
        assert 'recall' in metrics

    def test_calculate_metrics_all_wrong(self):
        """Testa quando todas as predições estão erradas."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = 1 - y_true  # Inverted
        y_prob = np.array([0.9, 0.8, 0.1, 0.2, 0.85, 0.15])

        metrics = Classification.calculate_metrics(y_true, y_pred, y_prob)

        # Accuracy deve ser 0.0
        assert metrics['accuracy'] == 0.0

    def test_calculate_metrics_without_probs(self, binary_classification_data):
        """Testa sem y_prob (métricas limitadas)."""
        y_true, y_pred, _, _ = binary_classification_data

        metrics = Classification.calculate_metrics(y_true, y_pred, None, None)

        # Métricas básicas devem estar presentes
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics

        # Métricas que precisam de probabilidades não devem estar
        assert 'roc_auc' not in metrics
        assert 'kl_divergence' not in metrics

    def test_calculate_metrics_with_prob_no_teacher(self, binary_classification_data):
        """Testa com y_prob mas sem teacher_prob."""
        y_true, y_pred, y_prob, _ = binary_classification_data

        metrics = Classification.calculate_metrics(y_true, y_pred, y_prob, None)

        # AUC metrics devem estar presentes
        assert 'roc_auc' in metrics
        assert 'auc_pr' in metrics

        # Teacher comparison metrics não devem estar
        assert 'kl_divergence' not in metrics


# ============================================================================
# Testes de calculate_metrics_from_predictions
# ============================================================================

class TestClassificationCalculateMetricsFromPredictions:
    """Testes do método calculate_metrics_from_predictions."""

    def test_calculate_from_dataframe_minimal(self, classification_dataframe):
        """Testa cálculo mínimo a partir de DataFrame."""
        metrics = Classification.calculate_metrics_from_predictions(
            data=classification_dataframe,
            target_column='y_true',
            pred_column='y_pred'
        )

        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics

    def test_calculate_from_dataframe_with_prob(self, classification_dataframe):
        """Testa com probabilidades."""
        metrics = Classification.calculate_metrics_from_predictions(
            data=classification_dataframe,
            target_column='y_true',
            pred_column='y_pred',
            prob_column='y_prob'
        )

        assert 'accuracy' in metrics
        assert 'roc_auc' in metrics
        assert 'auc_pr' in metrics

    def test_calculate_from_dataframe_with_teacher(self, classification_dataframe):
        """Testa com teacher probabilities."""
        metrics = Classification.calculate_metrics_from_predictions(
            data=classification_dataframe,
            target_column='y_true',
            pred_column='y_pred',
            prob_column='y_prob',
            teacher_prob_column='teacher_prob'
        )

        assert 'accuracy' in metrics
        assert 'roc_auc' in metrics
        assert 'kl_divergence' in metrics
        assert 'ks_statistic' in metrics

    def test_calculate_from_dataframe_custom_names(self):
        """Testa com nomes de colunas customizados."""
        df = pd.DataFrame({
            'ground_truth': [0, 0, 1, 1],
            'prediction': [0, 1, 1, 1],
            'probability': [0.1, 0.6, 0.8, 0.9],
            'teacher': [0.05, 0.4, 0.85, 0.95]
        })

        metrics = Classification.calculate_metrics_from_predictions(
            data=df,
            target_column='ground_truth',
            pred_column='prediction',
            prob_column='probability',
            teacher_prob_column='teacher'
        )

        assert 'accuracy' in metrics
        assert 'roc_auc' in metrics
        assert 'kl_divergence' in metrics


# ============================================================================
# Testes de calculate_kl_divergence
# ============================================================================

class TestClassificationCalculateKLDivergence:
    """Testes do método calculate_kl_divergence."""

    def test_kl_divergence_identical_distributions(self):
        """Testa com distribuições idênticas."""
        p = np.array([0.2, 0.5, 0.8, 0.9])
        q = p.copy()

        kl = Classification.calculate_kl_divergence(p, q)

        # KL divergence deve ser próximo de 0
        assert kl < 0.01

    def test_kl_divergence_different_distributions(self):
        """Testa com distribuições diferentes."""
        p = np.array([0.9, 0.8, 0.7, 0.6])
        q = np.array([0.1, 0.2, 0.3, 0.4])

        kl = Classification.calculate_kl_divergence(p, q)

        # KL divergence deve ser > 0
        assert kl > 0

    def test_kl_divergence_with_pandas_series(self):
        """Testa com pandas Series."""
        p = pd.Series([0.2, 0.5, 0.8, 0.9])
        q = pd.Series([0.25, 0.55, 0.75, 0.85])

        kl = Classification.calculate_kl_divergence(p, q)

        # Deve funcionar e retornar valor
        assert isinstance(kl, float)
        assert kl >= 0

    def test_kl_divergence_2d_format(self):
        """Testa com formato 2D (multi-class)."""
        p = np.array([[0.2, 0.8], [0.6, 0.4], [0.3, 0.7]])
        q = np.array([[0.25, 0.75], [0.55, 0.45], [0.35, 0.65]])

        kl = Classification.calculate_kl_divergence(p, q)

        # Deve calcular KL divergence
        assert isinstance(kl, float)
        assert kl >= 0

    def test_kl_divergence_extreme_values(self):
        """Testa com valores extremos."""
        p = np.array([0.99, 0.98, 0.95, 0.90])
        q = np.array([0.01, 0.02, 0.05, 0.10])

        kl = Classification.calculate_kl_divergence(p, q)

        # KL divergence deve ser grande (distribuições muito diferentes)
        assert kl > 1.0


# ============================================================================
# Testes de calculate_ks_statistic
# ============================================================================

class TestClassificationCalculateKSStatistic:
    """Testes do método calculate_ks_statistic."""

    def test_ks_statistic_identical_distributions(self):
        """Testa com distribuições idênticas."""
        teacher_prob = np.array([0.2, 0.5, 0.8, 0.9, 0.3, 0.6])
        student_prob = teacher_prob.copy()

        ks_stat, p_value = Classification.calculate_ks_statistic(
            teacher_prob, student_prob
        )

        # KS statistic deve ser próximo de 0
        assert ks_stat < 0.2
        # p-value deve ser alto (não rejeita H0 de igualdade)
        assert p_value > 0.05

    def test_ks_statistic_different_distributions(self):
        """Testa com distribuições diferentes."""
        teacher_prob = np.array([0.9, 0.8, 0.7, 0.6, 0.85, 0.95])
        student_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.15, 0.05])

        ks_stat, p_value = Classification.calculate_ks_statistic(
            teacher_prob, student_prob
        )

        # KS statistic deve ser alto (distribuições diferentes)
        assert ks_stat > 0.5

    def test_ks_statistic_2d_format(self):
        """Testa com formato 2D (extrai coluna positiva)."""
        teacher_prob = np.array([[0.3, 0.7], [0.4, 0.6], [0.2, 0.8]])
        student_prob = np.array([[0.35, 0.65], [0.45, 0.55], [0.25, 0.75]])

        ks_stat, p_value = Classification.calculate_ks_statistic(
            teacher_prob, student_prob
        )

        # Deve funcionar e retornar valores válidos
        assert isinstance(ks_stat, float)
        assert isinstance(p_value, float)
        assert 0 <= ks_stat <= 1
        assert 0 <= p_value <= 1

    def test_ks_statistic_with_nans(self):
        """Testa com valores NaN."""
        teacher_prob = np.array([0.5, np.nan, 0.7, 0.8])
        student_prob = np.array([0.45, 0.55, np.nan, 0.75])

        ks_stat, p_value = Classification.calculate_ks_statistic(
            teacher_prob, student_prob
        )

        # Deve remover NaNs e calcular
        assert isinstance(ks_stat, float)
        assert isinstance(p_value, float)

    def test_ks_statistic_empty_after_nan_removal(self):
        """Testa quando todos são NaN."""
        teacher_prob = np.array([np.nan, np.nan, np.nan])
        student_prob = np.array([np.nan, np.nan, np.nan])

        ks_stat, p_value = Classification.calculate_ks_statistic(
            teacher_prob, student_prob
        )

        # Deve retornar valores padrão
        assert ks_stat == 0.0
        assert p_value == 1.0

    def test_ks_statistic_with_pandas_series(self):
        """Testa com pandas Series."""
        teacher_prob = pd.Series([0.2, 0.5, 0.8, 0.9])
        student_prob = pd.Series([0.25, 0.55, 0.75, 0.85])

        ks_stat, p_value = Classification.calculate_ks_statistic(
            teacher_prob, student_prob
        )

        # Deve funcionar
        assert isinstance(ks_stat, float)
        assert isinstance(p_value, float)


# ============================================================================
# Testes de calculate_r2_score
# ============================================================================

class TestClassificationCalculateR2Score:
    """Testes do método calculate_r2_score."""

    def test_r2_score_identical_distributions(self):
        """Testa com distribuições idênticas."""
        teacher_prob = np.array([0.2, 0.5, 0.8, 0.9, 0.3, 0.6])
        student_prob = teacher_prob.copy()

        r2 = Classification.calculate_r2_score(teacher_prob, student_prob)

        # R² deve ser 1.0 (perfect match)
        assert np.isclose(r2, 1.0)

    def test_r2_score_similar_distributions(self):
        """Testa com distribuições similares."""
        teacher_prob = np.array([0.2, 0.5, 0.8, 0.9, 0.3, 0.6])
        student_prob = np.array([0.25, 0.55, 0.75, 0.85, 0.35, 0.65])

        r2 = Classification.calculate_r2_score(teacher_prob, student_prob)

        # R² deve ser alto (similar distributions)
        assert r2 > 0.9

    def test_r2_score_different_distributions(self):
        """Testa com distribuições diferentes."""
        teacher_prob = np.array([0.9, 0.8, 0.7, 0.6, 0.85, 0.95])
        student_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.15, 0.05])

        r2 = Classification.calculate_r2_score(teacher_prob, student_prob)

        # R² pode ser baixo ou até negativo
        assert r2 < 1.0

    def test_r2_score_2d_format(self):
        """Testa com formato 2D."""
        teacher_prob = np.array([[0.3, 0.7], [0.4, 0.6], [0.2, 0.8]])
        student_prob = np.array([[0.35, 0.65], [0.45, 0.55], [0.25, 0.75]])

        r2 = Classification.calculate_r2_score(teacher_prob, student_prob)

        # Deve funcionar
        assert isinstance(r2, float)

    def test_r2_score_with_nans(self):
        """Testa com valores NaN."""
        teacher_prob = np.array([0.5, np.nan, 0.7, 0.8])
        student_prob = np.array([0.45, 0.55, np.nan, 0.75])

        r2 = Classification.calculate_r2_score(teacher_prob, student_prob)

        # Deve remover NaNs e calcular
        assert isinstance(r2, float)

    def test_r2_score_empty_after_nan_removal(self):
        """Testa quando todos são NaN."""
        teacher_prob = np.array([np.nan, np.nan, np.nan])
        student_prob = np.array([np.nan, np.nan, np.nan])

        r2 = Classification.calculate_r2_score(teacher_prob, student_prob)

        # Deve retornar valor padrão
        assert r2 == 0.0

    def test_r2_score_different_lengths(self):
        """Testa com comprimentos diferentes."""
        teacher_prob = np.array([0.2, 0.5, 0.8, 0.9, 0.3])
        student_prob = np.array([0.25, 0.55, 0.75])

        r2 = Classification.calculate_r2_score(teacher_prob, student_prob)

        # Deve truncar para o menor comprimento e calcular
        assert isinstance(r2, float)


# ============================================================================
# Testes de Integração
# ============================================================================

class TestClassificationIntegration:
    """Testes de integração completa."""

    def test_full_pipeline_with_all_metrics(self, binary_classification_data):
        """Testa pipeline completo com todas as métricas."""
        y_true, y_pred, y_prob, teacher_prob = binary_classification_data

        metrics = Classification.calculate_metrics(
            y_true, y_pred, y_prob, teacher_prob
        )

        # Verificar que todas as métricas esperadas estão presentes
        expected_metrics = [
            'accuracy', 'precision', 'recall', 'f1_score', 'f1-score',
            'roc_auc', 'auc_roc', 'auc_pr', 'log_loss',
            'kl_divergence', 'ks_statistic', 'ks_pvalue', 'r2_score'
        ]

        for metric in expected_metrics:
            assert metric in metrics
            # Verificar que não são None (com dados válidos)
            if metric not in ['ks_pvalue']:  # p-value pode ser qualquer valor
                assert metrics[metric] is not None

    def test_metrics_consistency(self, binary_classification_data):
        """Testa consistência entre métricas."""
        y_true, y_pred, y_prob, teacher_prob = binary_classification_data

        metrics = Classification.calculate_metrics(
            y_true, y_pred, y_prob, teacher_prob
        )

        # F1 score deve ser harmonic mean de precision e recall
        if metrics['precision'] > 0 and metrics['recall'] > 0:
            expected_f1 = 2 * (metrics['precision'] * metrics['recall']) / \
                         (metrics['precision'] + metrics['recall'])
            assert np.isclose(metrics['f1_score'], expected_f1, atol=1e-5)

    def test_dataframe_vs_arrays_equivalence(self, binary_classification_data, classification_dataframe):
        """Testa que DataFrame e arrays produzem mesmos resultados."""
        y_true, y_pred, y_prob, teacher_prob = binary_classification_data

        # Calcular com arrays
        metrics_arrays = Classification.calculate_metrics(
            y_true, y_pred, y_prob, teacher_prob
        )

        # Calcular com DataFrame
        metrics_df = Classification.calculate_metrics_from_predictions(
            data=classification_dataframe,
            target_column='y_true',
            pred_column='y_pred',
            prob_column='y_prob',
            teacher_prob_column='teacher_prob'
        )

        # Comparar métricas principais
        for key in ['accuracy', 'precision', 'recall', 'roc_auc', 'kl_divergence']:
            assert np.isclose(metrics_arrays[key], metrics_df[key], rtol=1e-5)


# ============================================================================
# Additional Error Handling Tests for Missing Coverage
# ============================================================================

def test_calculate_metrics_exception_in_precision_recall():
    """Test error handling when precision/recall/f1 calculation fails"""
    from unittest.mock import patch
    
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    
    # Mock precision_score to raise an exception
    with patch('deepbridge.metrics.classification.precision_score', side_effect=Exception("Test error")):
        metrics = Classification.calculate_metrics(y_true, y_pred)
        
        # Should handle error gracefully and set to 0.0
        assert metrics['precision'] == 0.0
        assert metrics['recall'] == 0.0
        assert metrics['f1_score'] == 0.0
        assert metrics['f1-score'] == 0.0


def test_calculate_metrics_auc_value_error():
    """Test handling of ValueError in AUC calculation"""
    from unittest.mock import patch
    
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8])
    
    # Mock roc_auc_score to raise ValueError
    with patch('deepbridge.metrics.classification.roc_auc_score', side_effect=ValueError("Test error")):
        metrics = Classification.calculate_metrics(y_true, y_pred, y_prob)
        
        # Should handle error gracefully and set to None
        assert metrics['roc_auc'] is None
        assert metrics['auc_roc'] is None
        assert metrics['auc_pr'] is None
        assert metrics['log_loss'] is None


def test_calculate_metrics_ks_statistic_exception():
    """Test handling of exception in KS statistic calculation"""
    from unittest.mock import patch
    
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8])
    teacher_prob = np.array([0.1, 0.9, 0.2, 0.8])
    
    # Mock calculate_ks_statistic to raise exception
    with patch.object(Classification, 'calculate_ks_statistic', side_effect=Exception("Test error")):
        metrics = Classification.calculate_metrics(y_true, y_pred, y_prob, teacher_prob)
        
        # Should handle error gracefully
        assert metrics['ks_statistic'] is None
        assert metrics['ks_pvalue'] is None


def test_calculate_metrics_r2_score_exception():
    """Test handling of exception in R² calculation"""
    from unittest.mock import patch
    
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8])
    teacher_prob = np.array([0.1, 0.9, 0.2, 0.8])
    
    # Mock calculate_r2_score to raise exception
    with patch.object(Classification, 'calculate_r2_score', side_effect=Exception("Test error")):
        metrics = Classification.calculate_metrics(y_true, y_pred, y_prob, teacher_prob)
        
        # Should handle error gracefully
        assert metrics['r2_score'] is None


def test_calculate_metrics_teacher_prob_exception():
    """Test handling of exception in teacher probability processing"""
    from unittest.mock import patch
    
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8])
    teacher_prob = np.array([0.1, 0.9, 0.2, 0.8])
    
    # Mock calculate_kl_divergence to raise exception
    with patch.object(Classification, 'calculate_kl_divergence', side_effect=Exception("Test error")):
        metrics = Classification.calculate_metrics(y_true, y_pred, y_prob, teacher_prob)
        
        # Should handle error gracefully and set all teacher-related metrics to None
        assert metrics['kl_divergence'] is None
        assert metrics['ks_statistic'] is None
        assert metrics['ks_pvalue'] is None
        assert metrics['r2_score'] is None


def test_calculate_r2_score_with_list_inputs():
    """Test calculate_r2_score with list inputs (triggers array conversion)."""
    teacher_prob = [0.1, 0.2, 0.3, 0.4, 0.5]  # List, not ndarray
    student_prob = [0.15, 0.25, 0.35, 0.45, 0.55]  # List, not ndarray
    
    r2 = Classification.calculate_r2_score(teacher_prob, student_prob)
    
    assert isinstance(r2, (float, type(None)))
    assert r2 is not None  # Should successfully calculate


def test_calculate_r2_score_with_invalid_input_triggers_exception():
    """Test calculate_r2_score with input that causes exception in r2_score."""
    from unittest.mock import patch
    
    teacher_prob = np.array([0.1, 0.2, 0.3])
    student_prob = np.array([0.1, 0.2, 0.3])
    
    # Mock r2_score to raise an exception
    with patch('deepbridge.metrics.classification.r2_score', side_effect=Exception("Test error")):
        r2 = Classification.calculate_r2_score(teacher_prob, student_prob)
        
        # Should return None when exception occurs
        assert r2 is None


def test_calculate_ks_statistic_with_invalid_input_raises():
    """Test calculate_ks_statistic re-raises exception on invalid input."""
    from unittest.mock import patch
    
    teacher_prob = np.array([0.1, 0.2, 0.3])
    student_prob = np.array([0.1, 0.2, 0.3])
    
    # Mock ks_2samp to raise an exception
    with patch('deepbridge.metrics.classification.stats.ks_2samp', side_effect=ValueError("Test error")):
        with pytest.raises(ValueError, match="Test error"):
            Classification.calculate_ks_statistic(teacher_prob, student_prob)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
