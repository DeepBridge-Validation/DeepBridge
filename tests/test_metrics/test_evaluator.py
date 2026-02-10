"""
Testes para deepbridge.metrics.evaluator.MetricsEvaluator

Objetivo: Elevar coverage de 0% para 95%+
Foco: find_best_model, get_valid_results, get_model_comparison_metrics,
      get_factor_impact, get_available_metrics
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock

from deepbridge.metrics.evaluator import MetricsEvaluator
from deepbridge.utils.model_registry import ModelType


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_config():
    """Cria mock de DistillationConfig."""
    config = Mock()
    config.log_info = Mock()
    return config


@pytest.fixture
def sample_results_df():
    """Cria DataFrame de resultados de exemplo."""
    np.random.seed(42)

    data = {
        'model_type': ['GBM', 'GBM', 'DECISION_TREE', 'DECISION_TREE', 'LOGISTIC_REGRESSION'],
        'temperature': [1.0, 2.0, 1.0, 2.0, 1.0],
        'alpha': [0.5, 0.7, 0.5, 0.7, 0.5],
        'test_accuracy': [0.85, 0.87, 0.82, 0.83, 0.80],
        'test_precision': [0.84, 0.86, 0.81, 0.82, 0.79],
        'test_recall': [0.83, 0.85, 0.80, 0.81, 0.78],
        'test_f1': [0.835, 0.855, 0.805, 0.815, 0.785],
        'test_auc_roc': [0.90, 0.92, 0.88, 0.89, 0.86],
        'test_auc_pr': [0.88, 0.90, 0.86, 0.87, 0.84],
        'test_kl_divergence': [0.15, 0.12, 0.18, 0.16, 0.20],
        'test_ks_statistic': [0.10, 0.08, 0.12, 0.11, 0.14],
        'test_r2_score': [0.92, 0.94, 0.90, 0.91, 0.88],
        'train_accuracy': [0.90, 0.91, 0.87, 0.88, 0.85]
    }

    return pd.DataFrame(data)


@pytest.fixture
def evaluator(sample_results_df, mock_config):
    """Cria MetricsEvaluator com dados de exemplo."""
    return MetricsEvaluator(sample_results_df, mock_config)


# ============================================================================
# Testes de Inicialização
# ============================================================================

class TestMetricsEvaluatorInitialization:
    """Testes de inicialização do MetricsEvaluator."""

    def test_init_with_dataframe_and_config(self, sample_results_df, mock_config):
        """Testa inicialização com DataFrame e config."""
        evaluator = MetricsEvaluator(sample_results_df, mock_config)

        assert evaluator.results_df is sample_results_df
        assert evaluator.config is mock_config

    def test_init_with_empty_dataframe(self, mock_config):
        """Testa inicialização com DataFrame vazio."""
        empty_df = pd.DataFrame()
        evaluator = MetricsEvaluator(empty_df, mock_config)

        assert evaluator.results_df.empty
        assert evaluator.config is mock_config


# ============================================================================
# Testes de find_best_model
# ============================================================================

class TestMetricsEvaluatorFindBestModel:
    """Testes do método find_best_model."""

    def test_find_best_model_maximize(self, evaluator):
        """Testa encontrar melhor modelo (maximize)."""
        best = evaluator.find_best_model(metric='test_accuracy', minimize=False)

        # Deve retornar o modelo com maior accuracy (0.87)
        assert best['test_accuracy'] == 0.87
        assert best['model_type'] == 'GBM'
        assert best['temperature'] == 2.0

    def test_find_best_model_minimize(self, evaluator):
        """Testa encontrar melhor modelo (minimize)."""
        best = evaluator.find_best_model(metric='test_kl_divergence', minimize=True)

        # Deve retornar o modelo com menor KL divergence (0.12)
        assert best['test_kl_divergence'] == 0.12
        assert best['model_type'] == 'GBM'

    def test_find_best_model_default_metric(self, evaluator):
        """Testa com métrica padrão (test_accuracy)."""
        best = evaluator.find_best_model()

        # Deve usar test_accuracy por padrão
        assert 'test_accuracy' in best
        assert best['test_accuracy'] == 0.87

    def test_find_best_model_nonexistent_metric(self, evaluator, mock_config):
        """Testa com métrica inexistente."""
        with pytest.raises(ValueError, match="not found in results"):
            evaluator.find_best_model(metric='nonexistent_metric')

        # Verificar que config.log_info foi chamado
        assert mock_config.log_info.called

    def test_find_best_model_with_nan_values(self, mock_config):
        """Testa com valores NaN."""
        df = pd.DataFrame({
            'model_type': ['GBM', 'DECISION_TREE', 'GBM'],
            'test_accuracy': [0.85, np.nan, 0.90]
        })
        evaluator = MetricsEvaluator(df, mock_config)

        best = evaluator.find_best_model(metric='test_accuracy')

        # Deve ignorar NaN e retornar 0.90
        assert best['test_accuracy'] == 0.90

    def test_find_best_model_all_nan(self, mock_config):
        """Testa quando todos os valores são NaN."""
        df = pd.DataFrame({
            'model_type': ['GBM', 'DECISION_TREE'],
            'test_accuracy': [np.nan, np.nan]
        })
        evaluator = MetricsEvaluator(df, mock_config)

        with pytest.raises(ValueError, match="No valid results"):
            evaluator.find_best_model(metric='test_accuracy')


# ============================================================================
# Testes de get_valid_results
# ============================================================================

class TestMetricsEvaluatorGetValidResults:
    """Testes do método get_valid_results."""

    def test_get_valid_results_with_metric(self, evaluator):
        """Testa obter resultados válidos para uma métrica."""
        valid = evaluator.get_valid_results(metric='test_accuracy')

        # Todos os resultados devem ter test_accuracy não-NaN
        assert not valid['test_accuracy'].isna().any()
        assert len(valid) == 5

    def test_get_valid_results_without_metric(self, evaluator):
        """Testa obter todos os resultados."""
        valid = evaluator.get_valid_results(metric=None)

        # Deve retornar todos os resultados
        assert len(valid) == 5

    def test_get_valid_results_with_nan(self, mock_config):
        """Testa com valores NaN."""
        df = pd.DataFrame({
            'model_type': ['GBM', 'DECISION_TREE', 'GBM'],
            'test_accuracy': [0.85, np.nan, 0.90]
        })
        evaluator = MetricsEvaluator(df, mock_config)

        valid = evaluator.get_valid_results(metric='test_accuracy')

        # Deve retornar apenas 2 resultados (sem NaN)
        assert len(valid) == 2
        assert not valid['test_accuracy'].isna().any()

    def test_get_valid_results_nonexistent_metric(self, evaluator, mock_config):
        """Testa com métrica inexistente."""
        valid = evaluator.get_valid_results(metric='nonexistent')

        # Deve retornar DataFrame vazio
        assert valid.empty
        assert mock_config.log_info.called

    def test_get_valid_results_empty_df(self, mock_config):
        """Testa com DataFrame vazio."""
        evaluator = MetricsEvaluator(pd.DataFrame(), mock_config)

        valid = evaluator.get_valid_results(metric='test_accuracy')

        # Deve retornar DataFrame vazio
        assert valid.empty


# ============================================================================
# Testes de get_model_comparison_metrics
# ============================================================================

class TestMetricsEvaluatorGetModelComparisonMetrics:
    """Testes do método get_model_comparison_metrics."""

    def test_get_model_comparison_metrics(self, evaluator):
        """Testa obter métricas de comparação de modelos."""
        comparison = evaluator.get_model_comparison_metrics()

        # Deve ter uma linha para cada tipo de modelo
        assert len(comparison) == 3  # GBM, DECISION_TREE, LOGISTIC_REGRESSION
        assert 'model' in comparison.columns

        # Verificar tipos de modelo
        models = comparison['model'].tolist()
        assert 'GBM' in models
        assert 'DECISION_TREE' in models
        assert 'LOGISTIC_REGRESSION' in models

    def test_get_model_comparison_metrics_columns(self, evaluator):
        """Testa colunas de métricas de comparação."""
        comparison = evaluator.get_model_comparison_metrics()

        # Deve ter métricas agregadas (avg e max/min)
        assert 'avg_accuracy' in comparison.columns
        assert 'max_accuracy' in comparison.columns
        assert 'avg_kl_div' in comparison.columns
        assert 'min_kl_div' in comparison.columns  # KL div usa min (lower is better)

    def test_get_model_comparison_metrics_values(self, evaluator, sample_results_df):
        """Testa valores das métricas."""
        comparison = evaluator.get_model_comparison_metrics()

        # GBM tem test_accuracy de [0.85, 0.87]
        gbm_row = comparison[comparison['model'] == 'GBM'].iloc[0]
        expected_avg = (0.85 + 0.87) / 2
        assert np.isclose(gbm_row['avg_accuracy'], expected_avg)
        assert gbm_row['max_accuracy'] == 0.87

    def test_get_model_comparison_metrics_kl_divergence(self, evaluator):
        """Testa que KL divergence usa min (lower is better)."""
        comparison = evaluator.get_model_comparison_metrics()

        # Verificar que min_kl_div é menor que avg_kl_div
        gbm_row = comparison[comparison['model'] == 'GBM'].iloc[0]
        assert gbm_row['min_kl_div'] <= gbm_row['avg_kl_div']

    def test_get_model_comparison_metrics_empty_df(self, mock_config):
        """Testa com DataFrame vazio."""
        evaluator = MetricsEvaluator(pd.DataFrame(), mock_config)

        comparison = evaluator.get_model_comparison_metrics()

        # Deve retornar DataFrame vazio
        assert comparison.empty

    def test_get_model_comparison_metrics_missing_columns(self, mock_config):
        """Testa quando métricas não existem."""
        df = pd.DataFrame({
            'model_type': ['GBM', 'DECISION_TREE'],
            'other_column': [1, 2]
        })
        evaluator = MetricsEvaluator(df, mock_config)

        comparison = evaluator.get_model_comparison_metrics()

        # Deve retornar DataFrame vazio (sem métricas disponíveis)
        assert comparison.empty


# ============================================================================
# Testes de get_factor_impact
# ============================================================================

class TestMetricsEvaluatorGetFactorImpact:
    """Testes do método get_factor_impact."""

    def test_get_factor_impact_temperature(self, evaluator):
        """Testa análise de impacto de temperatura."""
        impact = evaluator.get_factor_impact(factor='temperature')

        # Deve ter colunas model_type, temperature e métricas
        assert 'model_type' in impact.columns
        assert 'temperature' in impact.columns
        assert 'test_accuracy' in impact.columns

        # Deve ter pelo menos uma linha por combinação model_type + temperature
        assert len(impact) > 0

    def test_get_factor_impact_alpha(self, evaluator):
        """Testa análise de impacto de alpha."""
        impact = evaluator.get_factor_impact(factor='alpha')

        # Deve ter colunas model_type, alpha e métricas
        assert 'model_type' in impact.columns
        assert 'alpha' in impact.columns
        assert 'test_accuracy' in impact.columns

    def test_get_factor_impact_values(self, evaluator):
        """Testa valores de análise de impacto."""
        impact = evaluator.get_factor_impact(factor='temperature')

        # Deve calcular média para cada combinação
        # GBM + temp=1.0 tem test_accuracy=0.85
        # GBM + temp=2.0 tem test_accuracy=0.87
        gbm_temp1 = impact[
            (impact['model_type'] == 'GBM') &
            (impact['temperature'] == 1.0)
        ]
        if not gbm_temp1.empty:
            assert gbm_temp1['test_accuracy'].iloc[0] == 0.85

    def test_get_factor_impact_nonexistent_factor(self, evaluator, mock_config):
        """Testa com fator inexistente."""
        impact = evaluator.get_factor_impact(factor='nonexistent')

        # Deve retornar DataFrame vazio
        assert impact.empty
        assert mock_config.log_info.called

    def test_get_factor_impact_empty_df(self, mock_config):
        """Testa com DataFrame vazio."""
        evaluator = MetricsEvaluator(pd.DataFrame(), mock_config)

        impact = evaluator.get_factor_impact(factor='temperature')

        # Deve retornar DataFrame vazio
        assert impact.empty

    def test_get_factor_impact_missing_metrics(self, mock_config):
        """Testa quando métricas não existem."""
        df = pd.DataFrame({
            'model_type': ['GBM', 'DECISION_TREE'],
            'temperature': [1.0, 2.0],
            'other_column': [1, 2]
        })
        evaluator = MetricsEvaluator(df, mock_config)

        impact = evaluator.get_factor_impact(factor='temperature')

        # Deve retornar DataFrame vazio (sem métricas disponíveis)
        assert impact.empty


# ============================================================================
# Testes de get_available_metrics
# ============================================================================

class TestMetricsEvaluatorGetAvailableMetrics:
    """Testes do método get_available_metrics."""

    def test_get_available_metrics(self, evaluator):
        """Testa obter métricas disponíveis."""
        metrics = evaluator.get_available_metrics()

        # Deve retornar lista de métricas
        assert isinstance(metrics, list)
        assert len(metrics) > 0

        # Verificar métricas esperadas
        expected = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc',
                   'auc_pr', 'kl_divergence', 'ks_statistic', 'r2_score']
        for metric in expected:
            assert metric in metrics

    def test_get_available_metrics_partial(self, mock_config):
        """Testa com apenas algumas métricas disponíveis."""
        df = pd.DataFrame({
            'model_type': ['GBM'],
            'test_accuracy': [0.85],
            'test_precision': [0.84]
        })
        evaluator = MetricsEvaluator(df, mock_config)

        metrics = evaluator.get_available_metrics()

        # Deve retornar apenas métricas presentes
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' not in metrics

    def test_get_available_metrics_empty_df(self, mock_config):
        """Testa com DataFrame vazio."""
        evaluator = MetricsEvaluator(pd.DataFrame(), mock_config)

        metrics = evaluator.get_available_metrics()

        # Deve retornar lista vazia
        assert metrics == []

    def test_get_available_metrics_all_nan(self, mock_config):
        """Testa quando todas as métricas são NaN."""
        df = pd.DataFrame({
            'model_type': ['GBM'],
            'test_accuracy': [np.nan],
            'test_precision': [np.nan]
        })
        evaluator = MetricsEvaluator(df, mock_config)

        metrics = evaluator.get_available_metrics()

        # Deve retornar lista vazia (todas métricas são NaN)
        assert metrics == []

    def test_get_available_metrics_none_df(self, mock_config):
        """Testa quando results_df é None."""
        evaluator = MetricsEvaluator(None, mock_config)

        metrics = evaluator.get_available_metrics()

        # Deve retornar lista vazia
        assert metrics == []


# ============================================================================
# Testes de Integração
# ============================================================================

class TestMetricsEvaluatorIntegration:
    """Testes de integração completa."""

    def test_full_pipeline(self, evaluator):
        """Testa pipeline completo."""
        # 1. Obter métricas disponíveis
        available = evaluator.get_available_metrics()
        assert len(available) > 0

        # 2. Encontrar melhor modelo
        best = evaluator.find_best_model(metric='test_accuracy')
        assert 'test_accuracy' in best
        assert 'model_type' in best

        # 3. Obter comparação de modelos
        comparison = evaluator.get_model_comparison_metrics()
        assert not comparison.empty
        assert 'model' in comparison.columns

        # 4. Analisar impacto de fator
        impact = evaluator.get_factor_impact(factor='temperature')
        assert not impact.empty
        assert 'temperature' in impact.columns

    def test_consistency_best_model_vs_comparison(self, evaluator):
        """Testa consistência entre find_best_model e comparison."""
        # Encontrar melhor modelo
        best = evaluator.find_best_model(metric='test_accuracy')
        best_model_type = best['model_type']
        best_accuracy = best['test_accuracy']

        # Obter comparação
        comparison = evaluator.get_model_comparison_metrics()
        model_row = comparison[comparison['model'] == best_model_type].iloc[0]

        # max_accuracy deve ser >= best_accuracy
        # (podem ser iguais se houver apenas 1 resultado para esse modelo)
        assert model_row['max_accuracy'] >= best_accuracy

    def test_multiple_evaluators_independence(self, sample_results_df, mock_config):
        """Testa que múltiplos evaluators são independentes."""
        evaluator1 = MetricsEvaluator(sample_results_df, mock_config)

        # Criar segundo DataFrame diferente
        df2 = pd.DataFrame({
            'model_type': ['GBM'],
            'test_accuracy': [0.95]
        })
        evaluator2 = MetricsEvaluator(df2, mock_config)

        # Evaluators devem ter resultados diferentes
        best1 = evaluator1.find_best_model()
        best2 = evaluator2.find_best_model()

        assert best1['test_accuracy'] != best2['test_accuracy']


# ============================================================================
# Testes de Error Handling
# ============================================================================

class TestMetricsEvaluatorErrorHandling:
    """Testes de tratamento de erros."""

    def test_error_handling_in_find_best_model(self, mock_config):
        """Testa error handling em find_best_model."""
        # DataFrame com estrutura problemática
        df = pd.DataFrame({'invalid': [1, 2, 3]})
        evaluator = MetricsEvaluator(df, mock_config)

        with pytest.raises(ValueError):
            evaluator.find_best_model(metric='test_accuracy')

    def test_error_handling_in_get_model_comparison(self, mock_config):
        """Testa error handling em get_model_comparison_metrics."""
        # DataFrame inválido
        df = pd.DataFrame({'invalid': [1, 2, 3]})
        evaluator = MetricsEvaluator(df, mock_config)

        result = evaluator.get_model_comparison_metrics()

        # Deve retornar DataFrame vazio sem crashar
        assert result.empty

    def test_error_handling_in_get_factor_impact(self, mock_config):
        """Testa error handling em get_factor_impact."""
        # DataFrame sem métricas válidas
        df = pd.DataFrame({
            'model_type': ['GBM', 'GBM'],
            'temperature': [1.0, 2.0],
            'other_column': [1, 2]  # Sem test_accuracy
        })
        evaluator = MetricsEvaluator(df, mock_config)

        result = evaluator.get_factor_impact(factor='temperature')

        # Deve retornar DataFrame vazio sem crashar (sem métricas disponíveis)
        assert result.empty

    def test_get_valid_results_exception_handling(self, mock_config):
        """Testa exception handling em get_valid_results."""
        from unittest.mock import patch

        df = pd.DataFrame({'test_accuracy': [0.85]})
        evaluator = MetricsEvaluator(df, mock_config)

        # Force an exception during dropna
        with patch.object(pd.DataFrame, 'dropna', side_effect=Exception("Test error")):
            result = evaluator.get_valid_results(metric='test_accuracy')

        # Should return empty DataFrame
        assert result.empty
        assert mock_config.log_info.called

    def test_get_model_comparison_metrics_no_available_metrics_path(self, mock_config):
        """Testa path quando não há métricas disponíveis."""
        # DataFrame com model_type mas sem métricas test_*
        df = pd.DataFrame({
            'model_type': ['GBM', 'DECISION_TREE'],
            'accuracy': [0.85, 0.83],  # Sem prefixo test_
        })
        evaluator = MetricsEvaluator(df, mock_config)

        result = evaluator.get_model_comparison_metrics()

        # Deve retornar DataFrame vazio e logar
        assert result.empty
        # Verificar que foi logado "No metrics available" (verificar todas as calls)
        assert mock_config.log_info.called

    def test_get_model_comparison_metrics_exception_in_processing(self, mock_config):
        """Testa exception durante processamento de get_model_comparison_metrics."""
        from unittest.mock import patch

        df = pd.DataFrame({
            'model_type': ['GBM'],
            'test_accuracy': [0.85]
        })
        evaluator = MetricsEvaluator(df, mock_config)

        # Force exception during unique() call
        with patch.object(pd.Series, 'unique', side_effect=Exception("Test error")):
            result = evaluator.get_model_comparison_metrics()

        # Should return empty DataFrame and log error
        assert result.empty
        # Verify error was logged
        log_calls = [str(call) for call in mock_config.log_info.call_args_list]
        assert any('Error in get_model_comparison_metrics' in str(call) for call in log_calls)

    def test_get_factor_impact_no_metrics_available(self, mock_config):
        """Testa path quando não há métricas disponíveis em get_factor_impact."""
        df = pd.DataFrame({
            'model_type': ['GBM', 'DECISION_TREE'],
            'temperature': [1.0, 2.0],
            'accuracy': [0.85, 0.83],  # Sem prefixo test_
        })
        evaluator = MetricsEvaluator(df, mock_config)

        result = evaluator.get_factor_impact(factor='temperature')

        # Deve retornar DataFrame vazio
        assert result.empty
        # Verificar que foi logado
        assert mock_config.log_info.called

    def test_get_factor_impact_groupby_exception(self, mock_config):
        """Testa exception durante groupby em get_factor_impact."""
        from unittest.mock import patch

        df = pd.DataFrame({
            'model_type': ['GBM', 'GBM'],
            'temperature': [1.0, 2.0],
            'test_accuracy': [0.85, 0.87]
        })
        evaluator = MetricsEvaluator(df, mock_config)

        # Force exception during groupby
        with patch.object(pd.DataFrame, 'groupby', side_effect=Exception("Groupby error")):
            result = evaluator.get_factor_impact(factor='temperature')

        # Should return empty DataFrame
        assert result.empty
        # Verify error was logged
        log_calls = [str(call) for call in mock_config.log_info.call_args_list]
        assert any('Error in groupby operation' in str(call) for call in log_calls)

    def test_get_factor_impact_outer_exception(self, mock_config):
        """Testa outer exception handler em get_factor_impact."""
        from unittest.mock import patch

        df = pd.DataFrame({
            'model_type': ['GBM'],
            'temperature': [1.0],
            'test_accuracy': [0.85]
        })
        evaluator = MetricsEvaluator(df, mock_config)

        # Force exception in get_valid_results to trigger outer handler
        with patch.object(evaluator, 'get_valid_results', side_effect=Exception("Outer error")):
            result = evaluator.get_factor_impact(factor='temperature')

        # Should return empty DataFrame
        assert result.empty
        # Verify error was logged with traceback
        log_calls = [str(call) for call in mock_config.log_info.call_args_list]
        assert any('Error in get_factor_impact' in str(call) for call in log_calls)

    def test_get_available_metrics_exception_handling(self, mock_config):
        """Testa exception handling em get_available_metrics."""
        from unittest.mock import patch, PropertyMock

        df = pd.DataFrame({'test_accuracy': [0.85]})
        evaluator = MetricsEvaluator(df, mock_config)

        # Force exception by making results_df.columns raise exception
        # Need to patch on the evaluator instance's results_df specifically
        with patch.object(type(evaluator.results_df), 'columns', new_callable=PropertyMock) as mock_cols:
            mock_cols.side_effect = Exception("Test error")
            result = evaluator.get_available_metrics()

        # Should return empty list
        assert result == []
        # Verify error was logged
        assert mock_config.log_info.called



if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

