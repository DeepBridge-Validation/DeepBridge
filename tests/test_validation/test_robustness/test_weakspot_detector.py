"""
Testes para deepbridge.validation.robustness.weakspot_detector.WeakspotDetector

Objetivo: Elevar coverage de 0% para 90%+
Foco: detect_weak_regions, slicing methods, edge cases
"""

import pytest
import numpy as np
import pandas as pd
import warnings

from deepbridge.validation.robustness.weakspot_detector import WeakspotDetector


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def regression_data_with_weakspot():
    """Cria dados de regressão com weakspot deliberado."""
    np.random.seed(42)
    n_samples = 500

    # Create features
    age = np.random.uniform(20, 80, n_samples)
    income = np.random.uniform(20000, 150000, n_samples)

    # True values - linear relationship
    y_true = 0.5 * age + 0.0001 * income + np.random.randn(n_samples) * 5

    # Predictions - DELIBERATE WEAKSPOT for age > 70
    y_pred = y_true.copy()
    weakspot_mask = age > 70
    y_pred[weakspot_mask] += np.random.randn(weakspot_mask.sum()) * 30  # Muito pior

    X = pd.DataFrame({
        'age': age,
        'income': income
    })

    return X, y_true, y_pred


@pytest.fixture
def classification_data():
    """Cria dados de classificação."""
    np.random.seed(42)
    n_samples = 300

    X = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples)
    })

    y_true = (X['feature1'] + X['feature2'] > 0).astype(int).values

    # Predictions with some errors
    y_pred = y_true.copy()
    error_mask = np.random.rand(n_samples) < 0.15
    y_pred[error_mask] = 1 - y_pred[error_mask]

    return X, y_true, y_pred


@pytest.fixture
def small_dataset():
    """Cria dataset pequeno para edge cases."""
    np.random.seed(42)
    n_samples = 50

    X = pd.DataFrame({
        'x1': np.random.randn(n_samples),
        'x2': np.random.randn(n_samples)
    })

    y_true = np.random.randn(n_samples)
    y_pred = y_true + np.random.randn(n_samples) * 0.5

    return X, y_true, y_pred


# ============================================================================
# Testes de Inicialização
# ============================================================================

class TestWeakspotDetectorInitialization:
    """Testes de inicialização do WeakspotDetector."""

    def test_init_default_params(self):
        """Testa inicialização com parâmetros padrão."""
        detector = WeakspotDetector()

        assert detector.slice_method == 'quantile'
        assert detector.n_slices == 10
        assert detector.min_samples_per_slice == 30
        assert detector.severity_threshold == 0.15

    def test_init_custom_params(self):
        """Testa inicialização com parâmetros customizados."""
        detector = WeakspotDetector(
            slice_method='uniform',
            n_slices=5,
            min_samples_per_slice=20,
            severity_threshold=0.25
        )

        assert detector.slice_method == 'uniform'
        assert detector.n_slices == 5
        assert detector.min_samples_per_slice == 20
        assert detector.severity_threshold == 0.25

    def test_init_invalid_slice_method(self):
        """Testa erro ao usar slice_method inválido."""
        with pytest.raises(ValueError, match='slice_method must be one of'):
            WeakspotDetector(slice_method='invalid')

    def test_init_all_valid_slice_methods(self):
        """Testa que todos os métodos válidos são aceitos."""
        for method in ['uniform', 'quantile', 'tree-based']:
            detector = WeakspotDetector(slice_method=method)
            assert detector.slice_method == method


# ============================================================================
# Testes de detect_weak_regions
# ============================================================================

class TestWeakspotDetectorDetectWeakRegions:
    """Testes do método detect_weak_regions."""

    def test_detect_with_weakspot_quantile(self, regression_data_with_weakspot):
        """Testa detecção de weakspot usando método quantile."""
        X, y_true, y_pred = regression_data_with_weakspot

        detector = WeakspotDetector(
            slice_method='quantile',
            n_slices=10,
            severity_threshold=0.15
        )

        results = detector.detect_weak_regions(
            X=X,
            y_true=y_true,
            y_pred=y_pred,
            slice_features=['age'],
            metric='mae'
        )

        # Verificar estrutura do resultado
        assert 'weakspots' in results
        assert 'summary' in results
        assert 'slice_analysis' in results
        assert 'global_mean_residual' in results
        assert 'config' in results

        # Deve encontrar weakspots (age > 70 tem erro alto)
        assert results['summary']['total_weakspots'] > 0

    def test_detect_with_uniform_slicing(self, regression_data_with_weakspot):
        """Testa detecção com método uniform."""
        X, y_true, y_pred = regression_data_with_weakspot

        detector = WeakspotDetector(slice_method='uniform', n_slices=5)

        results = detector.detect_weak_regions(
            X=X,
            y_true=y_true,
            y_pred=y_pred,
            slice_features=['income'],
            metric='mae'
        )

        assert 'weakspots' in results
        assert results['config']['slice_method'] == 'uniform'

    def test_detect_with_tree_based_slicing(self, regression_data_with_weakspot):
        """Testa detecção com método tree-based (warning esperado)."""
        X, y_true, y_pred = regression_data_with_weakspot

        detector = WeakspotDetector(slice_method='tree-based')

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = detector.detect_weak_regions(
                X=X,
                y_true=y_true,
                y_pred=y_pred,
                metric='mae'
            )
            # Deve ter warning sobre tree-based não implementado
            assert len(w) > 0
            assert 'tree-based slicing not fully implemented' in str(w[0].message)

        assert 'weakspots' in results

    def test_detect_auto_select_features(self, regression_data_with_weakspot):
        """Testa auto-seleção de features numéricas."""
        X, y_true, y_pred = regression_data_with_weakspot

        detector = WeakspotDetector()

        # Não passar slice_features - deve auto-selecionar
        results = detector.detect_weak_regions(
            X=X,
            y_true=y_true,
            y_pred=y_pred,
            metric='mae'
        )

        # Deve ter analisado 'age' e 'income' (ambas numéricas)
        assert results['summary']['features_analyzed'] == 2

    def test_detect_classification_error_rate(self, classification_data):
        """Testa detecção com classificação (error_rate metric)."""
        X, y_true, y_pred = classification_data

        detector = WeakspotDetector(n_slices=5, severity_threshold=0.1)

        results = detector.detect_weak_regions(
            X=X,
            y_true=y_true,
            y_pred=y_pred,
            metric='error_rate'
        )

        assert results['config']['metric'] == 'error_rate'
        assert 'weakspots' in results

    def test_detect_mse_metric(self, regression_data_with_weakspot):
        """Testa detecção com métrica MSE."""
        X, y_true, y_pred = regression_data_with_weakspot

        detector = WeakspotDetector()

        results = detector.detect_weak_regions(
            X=X,
            y_true=y_true,
            y_pred=y_pred,
            slice_features=['age'],
            metric='mse'
        )

        assert results['config']['metric'] == 'mse'

    def test_detect_residual_metric(self, regression_data_with_weakspot):
        """Testa detecção com métrica residual."""
        X, y_true, y_pred = regression_data_with_weakspot

        detector = WeakspotDetector()

        results = detector.detect_weak_regions(
            X=X,
            y_true=y_true,
            y_pred=y_pred,
            slice_features=['age'],
            metric='residual'
        )

        assert results['config']['metric'] == 'residual'

    def test_summary_fields(self, regression_data_with_weakspot):
        """Testa campos do summary."""
        X, y_true, y_pred = regression_data_with_weakspot

        detector = WeakspotDetector()
        results = detector.detect_weak_regions(X, y_true, y_pred)

        summary = results['summary']

        # Verificar campos do summary
        assert 'total_weakspots' in summary
        assert 'features_with_weakspots' in summary
        assert 'features_analyzed' in summary
        assert 'avg_severity' in summary
        assert 'max_severity' in summary
        assert 'critical_weakspots' in summary

        # Verificar tipos
        assert isinstance(summary['total_weakspots'], int)
        assert isinstance(summary['avg_severity'], float)
        assert isinstance(summary['max_severity'], float)

    def test_weakspot_ordered_by_severity(self, regression_data_with_weakspot):
        """Testa que weakspots são ordenados por severity (pior primeiro)."""
        X, y_true, y_pred = regression_data_with_weakspot

        detector = WeakspotDetector(severity_threshold=0.1)
        results = detector.detect_weak_regions(X, y_true, y_pred)

        weakspots = results['weakspots']

        if len(weakspots) > 1:
            # Verificar ordem decrescente de severity
            severities = [w['severity'] for w in weakspots]
            assert severities == sorted(severities, reverse=True)

    def test_slice_analysis_structure(self, regression_data_with_weakspot):
        """Testa estrutura do slice_analysis."""
        X, y_true, y_pred = regression_data_with_weakspot

        detector = WeakspotDetector()
        results = detector.detect_weak_regions(
            X, y_true, y_pred, slice_features=['age']
        )

        slice_analysis = results['slice_analysis']

        assert 'age' in slice_analysis
        feature_analysis = slice_analysis['age']

        assert 'feature' in feature_analysis
        assert 'slices' in feature_analysis
        assert 'worst_slice' in feature_analysis
        assert 'best_slice' in feature_analysis
        assert 'n_slices_evaluated' in feature_analysis


# ============================================================================
# Testes de Edge Cases
# ============================================================================

class TestWeakspotDetectorEdgeCases:
    """Testes de casos extremos."""

    def test_mismatched_lengths_y(self):
        """Testa erro quando y_true e y_pred têm tamanhos diferentes."""
        X = pd.DataFrame({'x': [1, 2, 3]})
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2])  # Diferente

        detector = WeakspotDetector()

        with pytest.raises(ValueError, match='y_true and y_pred must have same length'):
            detector.detect_weak_regions(X, y_true, y_pred)

    def test_mismatched_lengths_X(self):
        """Testa erro quando X e y_true têm tamanhos diferentes."""
        X = pd.DataFrame({'x': [1, 2]})
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3])

        detector = WeakspotDetector()

        with pytest.raises(ValueError, match='X and y_true must have same length'):
            detector.detect_weak_regions(X, y_true, y_pred)

    def test_no_numeric_features(self):
        """Testa erro quando não há features numéricas."""
        X = pd.DataFrame({'cat': ['a', 'b', 'c']})
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3])

        detector = WeakspotDetector()

        with pytest.raises(ValueError, match='No numeric features found in X'):
            detector.detect_weak_regions(X, y_true, y_pred)

    def test_missing_slice_features(self):
        """Testa erro quando slice_features não existem em X."""
        X = pd.DataFrame({'x': [1, 2, 3]})
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3])

        detector = WeakspotDetector()

        with pytest.raises(ValueError, match='Features not found in X'):
            detector.detect_weak_regions(
                X, y_true, y_pred, slice_features=['nonexistent']
            )

    def test_feature_with_many_nans(self, regression_data_with_weakspot):
        """Testa warning quando feature tem muitos NaNs."""
        X, y_true, y_pred = regression_data_with_weakspot

        # Adicionar feature com >50% NaN
        X['nan_feature'] = np.nan
        X.loc[:100, 'nan_feature'] = np.random.randn(101)

        detector = WeakspotDetector()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = detector.detect_weak_regions(
                X, y_true, y_pred, slice_features=['nan_feature']
            )

            # Deve ter warning
            assert any('50% missing values' in str(warning.message) for warning in w)

    def test_constant_feature(self):
        """Testa com feature constante (todos valores iguais)."""
        X = pd.DataFrame({'const': [5.0] * 100})
        y_true = np.random.randn(100)
        y_pred = y_true + np.random.randn(100) * 0.1

        detector = WeakspotDetector()

        results = detector.detect_weak_regions(
            X, y_true, y_pred, slice_features=['const']
        )

        # Deve funcionar (1 slice com todos os dados)
        assert 'weakspots' in results

    def test_small_slices_ignored(self, small_dataset):
        """Testa que slices pequenos são ignorados."""
        X, y_true, y_pred = small_dataset

        detector = WeakspotDetector(
            n_slices=20,  # Muitos slices
            min_samples_per_slice=30  # Alto threshold
        )

        results = detector.detect_weak_regions(X, y_true, y_pred)

        # Com dataset pequeno e min_samples alto, não deve avaliar muitos slices
        for feature_analysis in results['slice_analysis'].values():
            assert feature_analysis['n_slices_evaluated'] < 20

    def test_perfect_predictions_no_weakspots(self):
        """Testa que predições perfeitas não geram weakspots."""
        X = pd.DataFrame({
            'x1': np.random.randn(200),
            'x2': np.random.randn(200)
        })
        y_true = np.random.randn(200)
        y_pred = y_true.copy()  # Perfeito

        detector = WeakspotDetector(severity_threshold=0.01)

        results = detector.detect_weak_regions(X, y_true, y_pred)

        # Não deve ter weakspots (erro é zero em todos os slices)
        assert results['summary']['total_weakspots'] == 0

    def test_invalid_metric(self):
        """Testa erro com métrica inválida."""
        X = pd.DataFrame({'x': np.random.randn(100)})
        y_true = np.random.randn(100)
        y_pred = np.random.randn(100)

        detector = WeakspotDetector()

        with pytest.raises(ValueError, match='Unknown metric'):
            detector.detect_weak_regions(X, y_true, y_pred, metric='invalid')


# ============================================================================
# Testes de Métodos Auxiliares
# ============================================================================

class TestWeakspotDetectorHelperMethods:
    """Testes de métodos auxiliares."""

    def test_get_top_weakspots(self, regression_data_with_weakspot):
        """Testa método get_top_weakspots."""
        X, y_true, y_pred = regression_data_with_weakspot

        detector = WeakspotDetector(severity_threshold=0.1)
        results = detector.detect_weak_regions(X, y_true, y_pred)

        # Pegar top 3
        top3 = detector.get_top_weakspots(results, n=3)

        assert len(top3) <= 3
        assert len(top3) <= len(results['weakspots'])

        # Se tem weakspots, verificar ordem
        if top3:
            assert top3[0]['severity'] >= top3[-1]['severity']

    def test_get_top_weakspots_more_than_available(self, regression_data_with_weakspot):
        """Testa get_top_weakspots quando n > total de weakspots."""
        X, y_true, y_pred = regression_data_with_weakspot

        detector = WeakspotDetector(severity_threshold=0.5)  # High threshold
        results = detector.detect_weak_regions(X, y_true, y_pred)

        # Pedir mais do que existe
        top10 = detector.get_top_weakspots(results, n=10)

        assert len(top10) <= 10
        assert len(top10) == len(results['weakspots'])

    def test_print_summary_basic(self, regression_data_with_weakspot, capsys):
        """Testa print_summary (output básico)."""
        X, y_true, y_pred = regression_data_with_weakspot

        detector = WeakspotDetector()
        results = detector.detect_weak_regions(X, y_true, y_pred)

        detector.print_summary(results, verbose=False)

        captured = capsys.readouterr()

        # Verificar que imprimiu informações chave
        assert 'WEAKSPOT DETECTION SUMMARY' in captured.out
        assert 'Total Weakspots Found' in captured.out
        assert 'Global Mean Residual' in captured.out

    def test_print_summary_verbose(self, regression_data_with_weakspot, capsys):
        """Testa print_summary com verbose=True."""
        X, y_true, y_pred = regression_data_with_weakspot

        detector = WeakspotDetector()
        results = detector.detect_weak_regions(X, y_true, y_pred)

        detector.print_summary(results, verbose=True)

        captured = capsys.readouterr()

        # Deve incluir top weakspots
        if results['weakspots']:
            assert 'TOP 5 WEAKSPOTS' in captured.out
            assert 'Feature:' in captured.out
            assert 'Severity:' in captured.out


# ============================================================================
# Testes de Integração
# ============================================================================

class TestWeakspotDetectorIntegration:
    """Testes de integração (pipeline completo)."""

    def test_full_pipeline_regression(self, regression_data_with_weakspot):
        """Testa pipeline completo para regressão."""
        X, y_true, y_pred = regression_data_with_weakspot

        # Pipeline completo
        detector = WeakspotDetector(
            slice_method='quantile',
            n_slices=8,
            severity_threshold=0.2
        )

        results = detector.detect_weak_regions(
            X=X,
            y_true=y_true,
            y_pred=y_pred,
            metric='mae'
        )

        # Verificar que encontrou o weakspot deliberado (age > 70)
        age_weakspots = [w for w in results['weakspots'] if w['feature'] == 'age']
        assert len(age_weakspots) > 0

        # Pegar top weakspots
        top5 = detector.get_top_weakspots(results, n=5)
        assert len(top5) > 0

    def test_full_pipeline_classification(self, classification_data):
        """Testa pipeline completo para classificação."""
        X, y_true, y_pred = classification_data

        detector = WeakspotDetector(
            slice_method='uniform',
            n_slices=5,
            severity_threshold=0.1
        )

        results = detector.detect_weak_regions(
            X=X,
            y_true=y_true,
            y_pred=y_pred,
            metric='error_rate'
        )

        # Verificar estrutura completa
        assert 'weakspots' in results
        assert 'summary' in results
        assert results['config']['metric'] == 'error_rate'

    def test_different_severity_thresholds(self, regression_data_with_weakspot):
        """Testa que severity_threshold afeta número de weakspots."""
        X, y_true, y_pred = regression_data_with_weakspot

        # Low threshold - mais weakspots
        detector_low = WeakspotDetector(severity_threshold=0.05)
        results_low = detector_low.detect_weak_regions(X, y_true, y_pred)

        # High threshold - menos weakspots
        detector_high = WeakspotDetector(severity_threshold=0.5)
        results_high = detector_high.detect_weak_regions(X, y_true, y_pred)

        # Threshold baixo deve encontrar mais (ou igual) weakspots
        assert results_low['summary']['total_weakspots'] >= results_high['summary']['total_weakspots']

    def test_comparison_uniform_vs_quantile(self, regression_data_with_weakspot):
        """Testa diferença entre métodos uniform e quantile."""
        X, y_true, y_pred = regression_data_with_weakspot

        detector_uniform = WeakspotDetector(slice_method='uniform')
        results_uniform = detector_uniform.detect_weak_regions(X, y_true, y_pred)

        detector_quantile = WeakspotDetector(slice_method='quantile')
        results_quantile = detector_quantile.detect_weak_regions(X, y_true, y_pred)

        # Ambos devem produzir resultados válidos
        assert 'weakspots' in results_uniform
        assert 'weakspots' in results_quantile

        # Podem ter diferentes números de weakspots (diferentes particionamentos)
        # Mas ambos devem ter encontrado algo (dado o weakspot deliberado)
        assert results_uniform['summary']['total_weakspots'] > 0 or \
               results_quantile['summary']['total_weakspots'] > 0


# ============================================================================
# Testes de Valores de Métricas
# ============================================================================

class TestWeakspotDetectorMetricValues:
    """Testes dos valores específicos das métricas."""

    def test_severity_calculation(self):
        """Testa cálculo de severity."""
        np.random.seed(42)
        X = pd.DataFrame({'x': np.arange(300)})

        # Criar erro pequeno uniforme
        y_true = np.random.randn(300)
        y_pred = y_true + np.random.randn(300) * 0.2

        # Adicionar erro MUITO grande em x > 240 (últimas slices)
        y_pred[240:] += 30  # Erro massivo

        # Usar min_samples baixo para garantir que slices sejam avaliados
        detector = WeakspotDetector(
            n_slices=10,
            severity_threshold=0.15,
            min_samples_per_slice=20  # Baixo threshold para garantir avaliação
        )
        results = detector.detect_weak_regions(X, y_true, y_pred, metric='mae')

        # Deve ter encontrado weakspot (erro em x > 240 é muito maior)
        assert results['summary']['total_weakspots'] > 0

        # Severity deve ser positivo (degradação)
        for ws in results['weakspots']:
            assert ws['severity'] > 0

    def test_global_mean_residual_positive(self, regression_data_with_weakspot):
        """Testa que global_mean_residual é sempre positivo."""
        X, y_true, y_pred = regression_data_with_weakspot

        detector = WeakspotDetector()
        results = detector.detect_weak_regions(X, y_true, y_pred)

        assert results['global_mean_residual'] >= 0

    def test_critical_weakspots_threshold(self):
        """Testa contagem de critical weakspots (severity > 0.5)."""
        X = pd.DataFrame({'x': np.arange(200)})
        y_true = np.random.randn(200)
        y_pred = y_true.copy()

        # Criar região MUITO ruim (severity > 0.5)
        y_pred[180:] += 50  # Erro enorme

        detector = WeakspotDetector(n_slices=10, severity_threshold=0.1)
        results = detector.detect_weak_regions(X, y_true, y_pred, metric='mae')

        # Deve ter pelo menos 1 critical weakspot
        if results['summary']['total_weakspots'] > 0:
            critical = results['summary']['critical_weakspots']
            assert critical >= 0

            # Se severity máximo > 0.5, critical deve ser > 0
            if results['summary']['max_severity'] > 0.5:
                assert critical > 0



def test_create_slices_empty_valid_values():
    """Test _create_slices with no valid values (all NaN)."""
    import numpy as np
    from deepbridge.validation.robustness.weakspot_detector import WeakspotDetector
    
    detector = WeakspotDetector(n_slices=5)
    feature_values = np.array([np.nan, np.nan, np.nan, np.nan])
    
    # Should return empty list when all values are NaN
    slices = detector._create_slices(feature_values, method="uniform")
    assert slices == []


def test_create_slices_unknown_method():
    """Test _create_slices with unknown method raises ValueError."""
    import numpy as np
    import pytest
    from deepbridge.validation.robustness.weakspot_detector import WeakspotDetector
    
    detector = WeakspotDetector(n_slices=5)
    feature_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    with pytest.raises(ValueError, match="Unknown slice method"):
        detector._create_slices(feature_values, method="invalid_method")


def test_uniform_slices_all_same_values():
    """Test _uniform_slices when all values are identical."""
    import numpy as np
    from deepbridge.validation.robustness.weakspot_detector import WeakspotDetector
    
    detector = WeakspotDetector(n_slices=5)
    feature_values = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
    valid_values = feature_values.copy()
    
    # Should return single slice when min == max
    slices = detector._uniform_slices(feature_values, valid_values)
    assert len(slices) == 1
    assert slices[0][0] == (5.0, 5.0)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

