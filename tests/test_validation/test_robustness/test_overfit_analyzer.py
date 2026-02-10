"""
Testes para deepbridge.validation.robustness.overfit_analyzer.OverfitAnalyzer

Objetivo: Elevar coverage de 0% para 90%+
Foco: compute_gap_by_slice, analyze_multiple_features, slicing methods
"""

import pytest
import numpy as np
import pandas as pd
import warnings
from unittest.mock import Mock

from deepbridge.validation.robustness.overfit_analyzer import OverfitAnalyzer


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_model_with_overfitting():
    """Cria modelo mock com overfitting deliberado."""
    model = Mock()

    def predict_with_overfit(X):
        """Predições que overfittam em valores altos."""
        feature_values = X.iloc[:, 0].values if isinstance(X, pd.DataFrame) else X[:, 0]

        # Base prediction
        predictions = np.random.rand(len(feature_values)) * 0.5 + 0.5

        # OVERFITTING: Perfect on train for high values (>0.7)
        # But poor on test for high values
        # Simulamos isso marcando amostras
        high_value_mask = feature_values > 0.7
        predictions[high_value_mask] = 0.95  # Very high (overfitted)

        return predictions

    model.predict = predict_with_overfit
    return model


@pytest.fixture
def mock_classifier():
    """Cria classificador mock."""
    model = Mock()

    def predict_proba(X):
        """Retorna probabilidades."""
        n_samples = len(X)
        # Binary classification probabilities
        prob_positive = np.random.rand(n_samples) * 0.5 + 0.25
        prob_negative = 1 - prob_positive
        return np.column_stack([prob_negative, prob_positive])

    model.predict_proba = predict_proba
    model.predict = lambda X: (model.predict_proba(X)[:, 1] > 0.5).astype(int)
    return model


@pytest.fixture
def binary_classification_data():
    """Cria dados de classificação binária com overfitting."""
    np.random.seed(42)
    n_train = 300
    n_test = 200

    # Features
    X_train = pd.DataFrame({
        'feature1': np.random.rand(n_train),
        'feature2': np.random.rand(n_train)
    })

    X_test = pd.DataFrame({
        'feature1': np.random.rand(n_test),
        'feature2': np.random.rand(n_test)
    })

    # Labels
    y_train = (X_train['feature1'] + X_train['feature2'] > 1.0).astype(int).values
    y_test = (X_test['feature1'] + X_test['feature2'] > 1.0).astype(int).values

    return X_train, X_test, y_train, y_test


@pytest.fixture
def regression_data():
    """Cria dados de regressão."""
    np.random.seed(42)
    n_train = 400
    n_test = 200

    X_train = pd.DataFrame({
        'x': np.random.rand(n_train)
    })

    X_test = pd.DataFrame({
        'x': np.random.rand(n_test)
    })

    y_train = X_train['x'].values * 2 + np.random.randn(n_train) * 0.1
    y_test = X_test['x'].values * 2 + np.random.randn(n_test) * 0.1

    return X_train, X_test, y_train, y_test


def simple_metric(y_true, y_pred):
    """Métrica simples para testes (accuracy-like)."""
    return np.mean(y_true == (y_pred > 0.5).astype(int))


def mae_metric(y_true, y_pred):
    """MAE metric."""
    return -np.mean(np.abs(y_true - y_pred))  # Negativo para "higher is better"


# ============================================================================
# Testes de Inicialização
# ============================================================================

class TestOverfitAnalyzerInitialization:
    """Testes de inicialização do OverfitAnalyzer."""

    def test_init_default_params(self):
        """Testa inicialização com parâmetros padrão."""
        analyzer = OverfitAnalyzer()

        assert analyzer.n_slices == 10
        assert analyzer.slice_method == 'quantile'
        assert analyzer.gap_threshold == 0.1
        assert analyzer.min_samples_per_slice == 30

    def test_init_custom_params(self):
        """Testa inicialização com parâmetros customizados."""
        analyzer = OverfitAnalyzer(
            n_slices=5,
            slice_method='uniform',
            gap_threshold=0.15,
            min_samples_per_slice=20
        )

        assert analyzer.n_slices == 5
        assert analyzer.slice_method == 'uniform'
        assert analyzer.gap_threshold == 0.15
        assert analyzer.min_samples_per_slice == 20


# ============================================================================
# Testes de compute_gap_by_slice
# ============================================================================

class TestOverfitAnalyzerComputeGapBySlice:
    """Testes do método compute_gap_by_slice."""

    def test_compute_gap_basic(self, binary_classification_data, mock_classifier):
        """Testa cálculo básico de gap."""
        X_train, X_test, y_train, y_test = binary_classification_data

        analyzer = OverfitAnalyzer(n_slices=5, min_samples_per_slice=10)

        results = analyzer.compute_gap_by_slice(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            model=mock_classifier,
            slice_feature='feature1',
            metric_func=simple_metric
        )

        # Verificar estrutura
        assert 'feature' in results
        assert 'slices' in results
        assert 'max_gap' in results
        assert 'avg_gap' in results
        assert 'std_gap' in results
        assert 'overfit_slices' in results
        assert 'summary' in results
        assert 'config' in results

        # Verificar feature
        assert results['feature'] == 'feature1'

    def test_compute_gap_with_quantile_slicing(self, binary_classification_data, mock_classifier):
        """Testa com método quantile."""
        X_train, X_test, y_train, y_test = binary_classification_data

        analyzer = OverfitAnalyzer(slice_method='quantile')

        results = analyzer.compute_gap_by_slice(
            X_train, X_test, y_train, y_test,
            mock_classifier, 'feature1', simple_metric
        )

        assert results['config']['slice_method'] == 'quantile'
        assert 'slices' in results

    def test_compute_gap_with_uniform_slicing(self, binary_classification_data, mock_classifier):
        """Testa com método uniform."""
        X_train, X_test, y_train, y_test = binary_classification_data

        analyzer = OverfitAnalyzer(slice_method='uniform')

        results = analyzer.compute_gap_by_slice(
            X_train, X_test, y_train, y_test,
            mock_classifier, 'feature2', simple_metric
        )

        assert results['config']['slice_method'] == 'uniform'

    def test_compute_gap_summary_fields(self, binary_classification_data, mock_classifier):
        """Testa campos do summary."""
        X_train, X_test, y_train, y_test = binary_classification_data

        analyzer = OverfitAnalyzer()
        results = analyzer.compute_gap_by_slice(
            X_train, X_test, y_train, y_test,
            mock_classifier, 'feature1', simple_metric
        )

        summary = results['summary']
        assert 'total_slices' in summary
        assert 'overfit_slices_count' in summary
        assert 'overfit_percentage' in summary

        assert isinstance(summary['total_slices'], int)
        assert isinstance(summary['overfit_slices_count'], int)
        assert isinstance(summary['overfit_percentage'], float)

    def test_compute_gap_slice_structure(self, binary_classification_data, mock_classifier):
        """Testa estrutura de cada slice."""
        X_train, X_test, y_train, y_test = binary_classification_data

        analyzer = OverfitAnalyzer(min_samples_per_slice=10)
        results = analyzer.compute_gap_by_slice(
            X_train, X_test, y_train, y_test,
            mock_classifier, 'feature1', simple_metric
        )

        if results['slices']:
            slice_info = results['slices'][0]

            assert 'slice_idx' in slice_info
            assert 'train_range' in slice_info
            assert 'test_range' in slice_info
            assert 'range_str' in slice_info
            assert 'train_samples' in slice_info
            assert 'test_samples' in slice_info
            assert 'train_metric' in slice_info
            assert 'test_metric' in slice_info
            assert 'gap' in slice_info
            assert 'gap_percentage' in slice_info
            assert 'is_overfitting' in slice_info

    def test_compute_gap_regression(self, regression_data):
        """Testa com dados de regressão."""
        X_train, X_test, y_train, y_test = regression_data

        # Create simple regressor mock
        model = Mock()
        model.predict = lambda X: X.iloc[:, 0].values * 2  # Linear predictor

        analyzer = OverfitAnalyzer(min_samples_per_slice=20)
        results = analyzer.compute_gap_by_slice(
            X_train, X_test, y_train, y_test,
            model, 'x', mae_metric
        )

        assert 'slices' in results
        assert results['feature'] == 'x'


# ============================================================================
# Testes de Edge Cases
# ============================================================================

class TestOverfitAnalyzerEdgeCases:
    """Testes de casos extremos."""

    def test_feature_not_in_train(self, binary_classification_data, mock_classifier):
        """Testa erro quando feature não existe em X_train."""
        X_train, X_test, y_train, y_test = binary_classification_data

        analyzer = OverfitAnalyzer()

        with pytest.raises(ValueError, match="not found in X_train"):
            analyzer.compute_gap_by_slice(
                X_train, X_test, y_train, y_test,
                mock_classifier, 'nonexistent', simple_metric
            )

    def test_feature_not_in_test(self, binary_classification_data, mock_classifier):
        """Testa erro quando feature não existe em X_test."""
        X_train, X_test, y_train, y_test = binary_classification_data

        # Remover feature de X_test
        X_test_modified = X_test.drop(columns=['feature1'])

        analyzer = OverfitAnalyzer()

        with pytest.raises(ValueError, match="not found in X_test"):
            analyzer.compute_gap_by_slice(
                X_train, X_test_modified, y_train, y_test,
                mock_classifier, 'feature1', simple_metric
            )

    def test_too_few_samples_per_slice(self):
        """Testa que slices com poucas amostras são ignorados."""
        np.random.seed(42)

        # Dataset muito pequeno
        X_train = pd.DataFrame({'x': np.random.rand(50)})
        X_test = pd.DataFrame({'x': np.random.rand(30)})
        y_train = np.random.randint(0, 2, 50)
        y_test = np.random.randint(0, 2, 30)

        model = Mock()
        model.predict = lambda X: np.random.rand(len(X))

        analyzer = OverfitAnalyzer(
            n_slices=20,  # Muitos slices
            min_samples_per_slice=50  # Threshold alto
        )

        results = analyzer.compute_gap_by_slice(
            X_train, X_test, y_train, y_test,
            model, 'x', simple_metric
        )

        # Poucos ou nenhum slice avaliado
        assert results['summary']['total_slices'] < 20

    def test_constant_feature(self, binary_classification_data, mock_classifier):
        """Testa com feature constante."""
        X_train, X_test, y_train, y_test = binary_classification_data

        # Adicionar feature constante
        X_train['const'] = 5.0
        X_test['const'] = 5.0

        analyzer = OverfitAnalyzer()
        results = analyzer.compute_gap_by_slice(
            X_train, X_test, y_train, y_test,
            mock_classifier, 'const', simple_metric
        )

        # Deve funcionar (1 slice com todos os dados)
        assert 'slices' in results

    def test_invalid_slice_method(self):
        """Testa erro com slice_method inválido."""
        X_train = pd.DataFrame({'x': [1, 2, 3]})
        X_test = pd.DataFrame({'x': [1, 2, 3]})
        y_train = np.array([0, 1, 0])
        y_test = np.array([1, 0, 1])

        model = Mock()
        model.predict = lambda X: np.array([0.5] * len(X))

        analyzer = OverfitAnalyzer(slice_method='invalid')

        with pytest.raises(ValueError, match='Unknown slice method'):
            analyzer.compute_gap_by_slice(
                X_train, X_test, y_train, y_test,
                model, 'x', simple_metric
            )

    def test_metric_computation_error(self, binary_classification_data, mock_classifier):
        """Testa warning quando métrica falha."""
        X_train, X_test, y_train, y_test = binary_classification_data

        def failing_metric(y_true, y_pred):
            raise ValueError("Metric failed!")

        analyzer = OverfitAnalyzer(min_samples_per_slice=10)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = analyzer.compute_gap_by_slice(
                X_train, X_test, y_train, y_test,
                mock_classifier, 'feature1', failing_metric
            )

            # Deve ter warnings sobre erros
            assert len(w) > 0

    def test_zero_train_metric_gap_percentage(self):
        """Testa gap_percentage quando train_metric é zero."""
        np.random.seed(42)

        X_train = pd.DataFrame({'x': np.random.rand(100)})
        X_test = pd.DataFrame({'x': np.random.rand(100)})
        y_train = np.zeros(100)
        y_test = np.zeros(100)

        model = Mock()
        model.predict = lambda X: np.zeros(len(X))

        def zero_metric(y_true, y_pred):
            return 0.0  # Always returns 0

        analyzer = OverfitAnalyzer(min_samples_per_slice=5)
        results = analyzer.compute_gap_by_slice(
            X_train, X_test, y_train, y_test,
            model, 'x', zero_metric
        )

        # gap_percentage deve ser 0 quando train_metric é 0
        for slice_info in results['slices']:
            if slice_info['train_metric'] == 0:
                assert slice_info['gap_percentage'] == 0.0


# ============================================================================
# Testes de analyze_multiple_features
# ============================================================================

class TestOverfitAnalyzerMultipleFeatures:
    """Testes do método analyze_multiple_features."""

    def test_analyze_multiple_features_basic(self, binary_classification_data, mock_classifier):
        """Testa análise de múltiplas features."""
        X_train, X_test, y_train, y_test = binary_classification_data

        analyzer = OverfitAnalyzer(min_samples_per_slice=10)

        results = analyzer.analyze_multiple_features(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            model=mock_classifier,
            features=['feature1', 'feature2'],
            metric_func=simple_metric
        )

        # Verificar estrutura
        assert 'features' in results
        assert 'worst_feature' in results
        assert 'summary' in results

        # Verificar que ambas features foram analisadas
        assert 'feature1' in results['features']
        assert 'feature2' in results['features']

    def test_analyze_multiple_features_summary(self, binary_classification_data, mock_classifier):
        """Testa summary de múltiplas features."""
        X_train, X_test, y_train, y_test = binary_classification_data

        analyzer = OverfitAnalyzer(min_samples_per_slice=10)
        results = analyzer.analyze_multiple_features(
            X_train, X_test, y_train, y_test,
            mock_classifier, ['feature1', 'feature2'], simple_metric
        )

        summary = results['summary']
        assert 'total_features' in summary
        assert 'features_with_overfitting' in summary
        assert 'global_max_gap' in summary

        assert summary['total_features'] == 2

    def test_analyze_multiple_features_worst_feature(self, binary_classification_data, mock_classifier):
        """Testa identificação da pior feature."""
        X_train, X_test, y_train, y_test = binary_classification_data

        analyzer = OverfitAnalyzer(gap_threshold=0.05, min_samples_per_slice=10)
        results = analyzer.analyze_multiple_features(
            X_train, X_test, y_train, y_test,
            mock_classifier, ['feature1', 'feature2'], simple_metric
        )

        # worst_feature deve ser uma das features analisadas
        assert results['worst_feature'] in ['feature1', 'feature2']

    def test_analyze_multiple_features_missing_feature_warning(self, binary_classification_data, mock_classifier):
        """Testa warning quando feature não existe."""
        X_train, X_test, y_train, y_test = binary_classification_data

        analyzer = OverfitAnalyzer()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = analyzer.analyze_multiple_features(
                X_train, X_test, y_train, y_test,
                mock_classifier,
                ['feature1', 'nonexistent', 'feature2'],
                simple_metric
            )

            # Deve ter warning sobre feature não encontrada
            assert any('not found' in str(warning.message) for warning in w)

            # Apenas features válidas nos resultados
            assert 'feature1' in results['features']
            assert 'feature2' in results['features']
            assert 'nonexistent' not in results['features']

    def test_analyze_multiple_features_empty_list(self):
        """Testa com lista vazia de features."""
        X_train = pd.DataFrame({'x': [1, 2, 3]})
        X_test = pd.DataFrame({'x': [1, 2, 3]})
        y_train = np.array([0, 1, 0])
        y_test = np.array([1, 0, 1])

        model = Mock()
        model.predict = lambda X: np.array([0.5] * len(X))

        analyzer = OverfitAnalyzer()
        results = analyzer.analyze_multiple_features(
            X_train, X_test, y_train, y_test,
            model, [], simple_metric
        )

        # worst_feature deve ser None
        assert results['worst_feature'] is None
        assert results['summary']['total_features'] == 0
        assert results['summary']['global_max_gap'] == 0.0


# ============================================================================
# Testes de Métodos Auxiliares
# ============================================================================

class TestOverfitAnalyzerHelperMethods:
    """Testes de métodos auxiliares."""

    def test_predict_with_regressor(self):
        """Testa _predict com regressor."""
        analyzer = OverfitAnalyzer()

        # Regressor (sem predict_proba)
        model = Mock(spec=['predict'])  # Only has predict
        model.predict = lambda X: np.array([1.5, 2.5, 3.5])

        X = pd.DataFrame({'x': [1, 2, 3]})
        y = np.array([1, 2, 3])

        predictions = analyzer._predict(model, X, y)

        assert len(predictions) == 3
        np.testing.assert_array_equal(predictions, [1.5, 2.5, 3.5])

    def test_predict_with_binary_classifier(self):
        """Testa _predict com classificador binário."""
        analyzer = OverfitAnalyzer()

        # Binary classifier
        model = Mock()
        model.predict_proba = lambda X: np.array([[0.3, 0.7], [0.6, 0.4], [0.2, 0.8]])

        X = pd.DataFrame({'x': [1, 2, 3]})
        y = np.array([1, 0, 1])

        predictions = analyzer._predict(model, X, y)

        # Deve retornar probabilidade da classe positiva
        np.testing.assert_array_equal(predictions, [0.7, 0.4, 0.8])

    def test_predict_with_multiclass_classifier(self):
        """Testa _predict com classificador multi-classe."""
        analyzer = OverfitAnalyzer()

        # Multi-class classifier
        model = Mock()
        model.predict_proba = lambda X: np.array([
            [0.2, 0.3, 0.5],
            [0.6, 0.2, 0.2],
            [0.1, 0.8, 0.1]
        ])
        model.predict = lambda X: np.array([2, 0, 1])

        X = pd.DataFrame({'x': [1, 2, 3]})
        y = np.array([2, 0, 1])

        predictions = analyzer._predict(model, X, y)

        # Para multi-classe, usa predict
        np.testing.assert_array_equal(predictions, [2, 0, 1])

    def test_print_summary_single_feature(self, binary_classification_data, mock_classifier, capsys):
        """Testa print_summary para single feature."""
        X_train, X_test, y_train, y_test = binary_classification_data

        analyzer = OverfitAnalyzer(gap_threshold=0.05, min_samples_per_slice=10)
        results = analyzer.compute_gap_by_slice(
            X_train, X_test, y_train, y_test,
            mock_classifier, 'feature1', simple_metric
        )

        analyzer.print_summary(results, verbose=False)

        captured = capsys.readouterr()
        assert 'SLICED OVERFITTING ANALYSIS' in captured.out
        assert 'feature1' in captured.out

    def test_print_summary_multiple_features(self, binary_classification_data, mock_classifier, capsys):
        """Testa print_summary para multiple features."""
        X_train, X_test, y_train, y_test = binary_classification_data

        analyzer = OverfitAnalyzer(min_samples_per_slice=10)
        results = analyzer.analyze_multiple_features(
            X_train, X_test, y_train, y_test,
            mock_classifier, ['feature1', 'feature2'], simple_metric
        )

        analyzer.print_summary(results, verbose=True)

        captured = capsys.readouterr()
        assert 'MULTI-FEATURE OVERFITTING ANALYSIS' in captured.out
        assert 'PER-FEATURE SUMMARY' in captured.out


# ============================================================================
# Testes de Integração
# ============================================================================

class TestOverfitAnalyzerIntegration:
    """Testes de integração (pipeline completo)."""

    def test_full_pipeline_single_feature(self, binary_classification_data, mock_classifier):
        """Testa pipeline completo para single feature."""
        X_train, X_test, y_train, y_test = binary_classification_data

        analyzer = OverfitAnalyzer(
            n_slices=5,
            slice_method='quantile',
            gap_threshold=0.1,
            min_samples_per_slice=20
        )

        results = analyzer.compute_gap_by_slice(
            X_train, X_test, y_train, y_test,
            mock_classifier, 'feature1', simple_metric
        )

        # Verificar pipeline completo
        assert 'slices' in results
        assert 'overfit_slices' in results
        assert results['summary']['total_slices'] >= 0

    def test_full_pipeline_multiple_features(self, binary_classification_data, mock_classifier):
        """Testa pipeline completo para múltiplas features."""
        X_train, X_test, y_train, y_test = binary_classification_data

        analyzer = OverfitAnalyzer(n_slices=5, min_samples_per_slice=20)

        results = analyzer.analyze_multiple_features(
            X_train, X_test, y_train, y_test,
            mock_classifier,
            ['feature1', 'feature2'],
            simple_metric
        )

        # Verificar estrutura completa
        assert len(results['features']) == 2
        assert results['worst_feature'] is not None
        assert results['summary']['total_features'] == 2

    def test_different_gap_thresholds(self, binary_classification_data, mock_classifier):
        """Testa que gap_threshold afeta número de overfit slices."""
        X_train, X_test, y_train, y_test = binary_classification_data

        # Low threshold - mais overfit slices
        analyzer_low = OverfitAnalyzer(gap_threshold=0.01, min_samples_per_slice=10)
        results_low = analyzer_low.compute_gap_by_slice(
            X_train, X_test, y_train, y_test,
            mock_classifier, 'feature1', simple_metric
        )

        # High threshold - menos overfit slices
        analyzer_high = OverfitAnalyzer(gap_threshold=0.5, min_samples_per_slice=10)
        results_high = analyzer_high.compute_gap_by_slice(
            X_train, X_test, y_train, y_test,
            mock_classifier, 'feature1', simple_metric
        )

        # Low threshold deve encontrar mais (ou igual) overfit slices
        assert results_low['summary']['overfit_slices_count'] >= \
               results_high['summary']['overfit_slices_count']

    def test_comparison_quantile_vs_uniform(self, binary_classification_data, mock_classifier):
        """Testa diferença entre métodos quantile e uniform."""
        X_train, X_test, y_train, y_test = binary_classification_data

        analyzer_quantile = OverfitAnalyzer(slice_method='quantile', min_samples_per_slice=10)
        results_quantile = analyzer_quantile.compute_gap_by_slice(
            X_train, X_test, y_train, y_test,
            mock_classifier, 'feature1', simple_metric
        )

        analyzer_uniform = OverfitAnalyzer(slice_method='uniform', min_samples_per_slice=10)
        results_uniform = analyzer_uniform.compute_gap_by_slice(
            X_train, X_test, y_train, y_test,
            mock_classifier, 'feature1', simple_metric
        )

        # Ambos devem produzir resultados válidos
        assert 'slices' in results_quantile
        assert 'slices' in results_uniform

    def test_slicing_all_nan_values(self, binary_classification_data, mock_classifier):
        """Test slicing with all NaN feature values."""
        X_train, X_test, y_train, y_test = binary_classification_data

        # Create data with all NaN values for one feature
        X_train_nan = X_train.copy()
        X_train_nan['feature1'] = np.nan

        X_test_nan = X_test.copy()
        X_test_nan['feature1'] = np.nan

        analyzer = OverfitAnalyzer(n_slices=5, min_samples_per_slice=10)
        results = analyzer.compute_gap_by_slice(
            X_train_nan, X_test_nan, y_train, y_test,
            mock_classifier, 'feature1', simple_metric
        )

        # Should return empty slices
        assert results['slices'] == []

    def test_uniform_slicing_constant_values(self, binary_classification_data, mock_classifier):
        """Test uniform slicing when all values are the same."""
        X_train, X_test, y_train, y_test = binary_classification_data

        # Create data with constant values
        X_train_const = X_train.copy()
        X_train_const['feature1'] = 0.5  # All same value

        X_test_const = X_test.copy()
        X_test_const['feature1'] = 0.5

        analyzer = OverfitAnalyzer(slice_method='uniform', n_slices=5, min_samples_per_slice=10)
        results = analyzer.compute_gap_by_slice(
            X_train_const, X_test_const, y_train, y_test,
            mock_classifier, 'feature1', simple_metric
        )

        # Should return single slice when min==max
        assert len(results['slices']) == 1

    def test_print_summary_verbose(self, binary_classification_data, mock_classifier, capsys):
        """Test print_summary with verbose=True showing overfit slices."""
        X_train, X_test, y_train, y_test = binary_classification_data

        analyzer = OverfitAnalyzer(n_slices=5, min_samples_per_slice=10, gap_threshold=0.01)
        results = analyzer.compute_gap_by_slice(
            X_train, X_test, y_train, y_test,
            mock_classifier, 'feature1', simple_metric
        )

        # Print with verbose to trigger lines 476-490
        analyzer.print_summary(results, verbose=True)

        captured = capsys.readouterr()
        # Check that verbose output is present
        if results['overfit_slices']:
            assert 'OVERFIT SLICES' in captured.out
            assert 'Range:' in captured.out or len(results['overfit_slices']) == 0
