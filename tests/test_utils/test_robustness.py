"""
Comprehensive tests for robustness utility functions.

This test suite validates:
1. Robustness testing wrapper function
2. Metric direction utility (higher/lower better)
3. Deprecated visualization functions

Coverage Target: ~90%+
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from deepbridge.utils.robustness import (
    run_robustness_tests,
    is_metric_higher_better,
    plot_robustness_results,
    compare_models_robustness,
    robustness_report_to_html,
)


# ==================== Fixtures ====================


@pytest.fixture
def mock_dataset():
    """Create a mock DBDataset"""
    dataset = Mock()
    dataset.X_train = Mock()
    dataset.y_train = Mock()
    dataset.X_test = Mock()
    dataset.y_test = Mock()
    dataset.model = Mock()
    return dataset


@pytest.fixture
def mock_robustness_results():
    """Mock results from robustness tests"""
    return {
        'base_score': 0.85,
        'avg_overall_impact': 0.12,
        'avg_raw_impact': 0.10,
        'avg_quantile_impact': 0.14,
        'feature_importance': {
            'feature_1': 0.3,
            'feature_2': 0.5,
            'feature_3': 0.2,
        },
        'raw': {
            'by_level': {
                '0.1': {
                    'runs': {'all': [0.84, 0.83, 0.85]},
                    'overall_result': {'all': 0.84},
                },
                '0.2': {
                    'runs': {'all': [0.80, 0.81, 0.82]},
                    'overall_result': {'all': 0.81},
                },
            }
        },
        'quantile': {
            'by_level': {
                '0.1': {
                    'runs': {'all': [0.83, 0.82, 0.84]},
                    'overall_result': {'all': 0.83},
                },
                '0.2': {
                    'runs': {'all': [0.79, 0.80, 0.81]},
                    'overall_result': {'all': 0.80},
                },
            }
        },
    }


@pytest.fixture
def mock_robustness_results_with_subset():
    """Mock results with feature subset"""
    return {
        'base_score': 0.85,
        'avg_overall_impact': 0.12,
        'avg_raw_impact': 0.10,
        'avg_quantile_impact': 0.14,
        'feature_subset': ['feature_1', 'feature_2'],
        'raw': {
            'by_level': {
                '0.1': {
                    'runs': {'feature_subset': [0.84, 0.83]},
                    'overall_result': {'feature_subset': 0.835},
                }
            }
        },
        'quantile': {
            'by_level': {
                '0.1': {
                    'runs': {'feature_subset': [0.82, 0.83]},
                    'overall_result': {'feature_subset': 0.825},
                }
            }
        },
    }


# ==================== is_metric_higher_better Tests ====================


class TestIsMetricHigherBetter:
    """Tests for is_metric_higher_better utility function"""

    def test_accuracy_higher_better(self):
        """Test that accuracy is recognized as higher-better"""
        assert is_metric_higher_better('accuracy') is True
        assert is_metric_higher_better('accuracy_score') is True
        assert is_metric_higher_better('balanced_accuracy') is True

    def test_precision_recall_higher_better(self):
        """Test that precision and recall are higher-better"""
        assert is_metric_higher_better('precision') is True
        assert is_metric_higher_better('recall') is True
        assert is_metric_higher_better('precision_score') is True
        assert is_metric_higher_better('recall_score') is True

    def test_f1_auc_higher_better(self):
        """Test that F1 and AUC are higher-better"""
        assert is_metric_higher_better('f1') is True
        assert is_metric_higher_better('f1_score') is True
        assert is_metric_higher_better('auc') is True
        assert is_metric_higher_better('roc_auc') is True
        assert is_metric_higher_better('average_precision') is True

    def test_r2_higher_better(self):
        """Test that R2 is higher-better"""
        assert is_metric_higher_better('r2') is True
        assert is_metric_higher_better('explained_variance') is True

    def test_error_metrics_lower_better(self):
        """Test that error metrics are lower-better"""
        assert is_metric_higher_better('error') is False
        assert is_metric_higher_better('mae') is False
        assert is_metric_higher_better('mse') is False
        assert is_metric_higher_better('rmse') is False

    def test_mean_errors_lower_better(self):
        """Test that mean error variants are lower-better"""
        assert is_metric_higher_better('mean_squared_error') is False
        assert is_metric_higher_better('mean_absolute_error') is False
        assert is_metric_higher_better('median_absolute_error') is False
        assert is_metric_higher_better('max_error') is False

    def test_loss_metrics_lower_better(self):
        """Test that loss metrics are lower-better"""
        assert is_metric_higher_better('log_loss') is False
        assert is_metric_higher_better('cross_entropy') is False
        assert is_metric_higher_better('hinge_loss') is False

    def test_case_insensitive(self):
        """Test that metric detection is case-insensitive"""
        assert is_metric_higher_better('ACCURACY') is True
        assert is_metric_higher_better('Accuracy') is True
        assert is_metric_higher_better('MSE') is False
        assert is_metric_higher_better('Mse') is False

    def test_partial_match(self):
        """Test that partial matches work"""
        assert is_metric_higher_better('test_accuracy_train') is True
        assert is_metric_higher_better('custom_mse_metric') is False

    def test_unknown_metric_defaults_higher_better(self):
        """Test that unknown metrics default to higher-better"""
        assert is_metric_higher_better('unknown_metric') is True
        assert is_metric_higher_better('custom_score') is True


# ==================== run_robustness_tests Tests ====================


class TestRunRobustnessTests:
    """Tests for run_robustness_tests wrapper function"""

    def test_basic_run_default_iterations(self, mock_dataset, mock_robustness_results):
        """Test basic robustness run with default iterations"""
        with patch('deepbridge.validation.wrappers.robustness_suite.RobustnessSuite.__init__', return_value=None):
            with patch('deepbridge.validation.wrappers.robustness_suite.RobustnessSuite.config') as mock_config:
                mock_config.return_value = MagicMock(run=Mock(return_value=mock_robustness_results))

                results = run_robustness_tests(mock_dataset, config_name='full')

                assert results['n_iterations'] == 10  # full config default
                assert 'metrics' in results
                assert results['metrics']['base_score'] == 0.85

    def test_quick_config_iterations(self, mock_dataset, mock_robustness_results):
        """Test quick config uses 3 iterations"""
        with patch('deepbridge.validation.wrappers.robustness_suite.RobustnessSuite.__init__', return_value=None):
            with patch('deepbridge.validation.wrappers.robustness_suite.RobustnessSuite.config') as mock_config:
                mock_config.return_value = MagicMock(run=Mock(return_value=mock_robustness_results))

                results = run_robustness_tests(mock_dataset, config_name='quick')

                assert results['n_iterations'] == 3

    def test_medium_config_iterations(self, mock_dataset, mock_robustness_results):
        """Test medium config uses 6 iterations"""
        with patch('deepbridge.validation.wrappers.robustness_suite.RobustnessSuite.__init__', return_value=None):
            with patch('deepbridge.validation.wrappers.robustness_suite.RobustnessSuite.config') as mock_config:
                mock_config.return_value = MagicMock(run=Mock(return_value=mock_robustness_results))

                results = run_robustness_tests(mock_dataset, config_name='medium')

                assert results['n_iterations'] == 6

    def test_unknown_config_defaults_to_3_iterations(self, mock_dataset, mock_robustness_results):
        """Test unknown config defaults to 3 iterations"""
        with patch('deepbridge.validation.wrappers.robustness_suite.RobustnessSuite.__init__', return_value=None):
            with patch('deepbridge.validation.wrappers.robustness_suite.RobustnessSuite.config') as mock_config:
                mock_config.return_value = MagicMock(run=Mock(return_value=mock_robustness_results))

                results = run_robustness_tests(mock_dataset, config_name='unknown')

                assert results['n_iterations'] == 3

    def test_explicit_iterations_override(self, mock_dataset, mock_robustness_results):
        """Test explicit n_iterations parameter overrides config default"""
        with patch('deepbridge.validation.wrappers.robustness_suite.RobustnessSuite.__init__', return_value=None):
            with patch('deepbridge.validation.wrappers.robustness_suite.RobustnessSuite.config') as mock_config:
                mock_config.return_value = MagicMock(run=Mock(return_value=mock_robustness_results))

                results = run_robustness_tests(mock_dataset, config_name='quick', n_iterations=20)

                assert results['n_iterations'] == 20

    def test_verbose_output(self, mock_dataset, mock_robustness_results, capsys):
        """Test verbose output printing"""
        with patch('deepbridge.validation.wrappers.robustness_suite.RobustnessSuite.__init__', return_value=None):
            with patch('deepbridge.validation.wrappers.robustness_suite.RobustnessSuite.config') as mock_config:
                mock_config.return_value = MagicMock(run=Mock(return_value=mock_robustness_results))

                results = run_robustness_tests(mock_dataset, verbose=True)

                captured = capsys.readouterr()
                assert 'Robustness Test Summary' in captured.out
                assert 'Overall robustness score' in captured.out
                assert '0.880' in captured.out  # 1.0 - 0.12 = 0.88

    def test_verbose_output_with_model_name(self, mock_dataset, mock_robustness_results, capsys):
        """Test verbose output with model name"""
        with patch('deepbridge.validation.wrappers.robustness_suite.RobustnessSuite.__init__', return_value=None):
            with patch('deepbridge.validation.wrappers.robustness_suite.RobustnessSuite.config') as mock_config:
                mock_config.return_value = MagicMock(run=Mock(return_value=mock_robustness_results))

                results = run_robustness_tests(
                    mock_dataset, verbose=True, model_name='RandomForest'
                )

                captured = capsys.readouterr()
                assert '[RandomForest]' in captured.out
                assert results['model_name'] == 'RandomForest'

    def test_no_verbose_output(self, mock_dataset, mock_robustness_results, capsys):
        """Test no output when verbose=False"""
        with patch('deepbridge.validation.wrappers.robustness_suite.RobustnessSuite.__init__', return_value=None):
            with patch('deepbridge.validation.wrappers.robustness_suite.RobustnessSuite.config') as mock_config:
                mock_config.return_value = MagicMock(run=Mock(return_value=mock_robustness_results))

                results = run_robustness_tests(mock_dataset, verbose=False)

                captured = capsys.readouterr()
                assert 'Robustness Test Summary' not in captured.out

    def test_with_feature_subset(self, mock_dataset, mock_robustness_results_with_subset):
        """Test with feature subset specified"""
        with patch('deepbridge.validation.wrappers.robustness_suite.RobustnessSuite.__init__', return_value=None):
            with patch('deepbridge.validation.wrappers.robustness_suite.RobustnessSuite.config') as mock_config:
                # First call returns all-features results, second call returns subset results
                mock_config.return_value = MagicMock(run=Mock(return_value=mock_robustness_results_with_subset))

                feature_subset = ['feature_1', 'feature_2']
                results = run_robustness_tests(mock_dataset, feature_subset=feature_subset)

                assert 'feature_subset' in results
                assert results['feature_subset'] == feature_subset

    def test_metrics_structure(self, mock_dataset, mock_robustness_results):
        """Test that metrics are properly structured in results"""
        with patch('deepbridge.validation.wrappers.robustness_suite.RobustnessSuite.__init__', return_value=None):
            with patch('deepbridge.validation.wrappers.robustness_suite.RobustnessSuite.config') as mock_config:
                mock_config.return_value = MagicMock(run=Mock(return_value=mock_robustness_results))

                results = run_robustness_tests(mock_dataset)

                assert 'metrics' in results
                assert 'base_score' in results['metrics']
                assert 'roc_auc' in results['metrics']
                assert results['metrics']['base_score'] == 0.85
                assert results['metrics']['roc_auc'] == 0.85

    def test_custom_metric(self, mock_dataset, mock_robustness_results):
        """Test with custom metric parameter"""
        with patch('deepbridge.validation.wrappers.robustness_suite.RobustnessSuite.__init__', return_value=None):
            with patch('deepbridge.validation.wrappers.robustness_suite.RobustnessSuite.config') as mock_config:
                mock_config.return_value = MagicMock(run=Mock(return_value=mock_robustness_results))

                results = run_robustness_tests(mock_dataset, metric='f1')

                # Verify the test ran successfully
                assert 'metrics' in results


# ==================== Deprecated Function Tests ====================


class TestDeprecatedFunctions:
    """Tests for deprecated visualization and reporting functions"""

    def test_plot_robustness_results_raises_not_implemented(self):
        """Test that plot_robustness_results raises NotImplementedError"""
        with pytest.raises(NotImplementedError, match='Visualization'):
            plot_robustness_results({})

    def test_compare_models_robustness_raises_not_implemented(self):
        """Test that compare_models_robustness raises NotImplementedError"""
        with pytest.raises(NotImplementedError, match='Visualization'):
            compare_models_robustness({})

    def test_robustness_report_to_html_raises_not_implemented(self):
        """Test that robustness_report_to_html raises NotImplementedError"""
        with pytest.raises(NotImplementedError, match='Report generation'):
            robustness_report_to_html({})


# ==================== Edge Cases ====================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_missing_base_score_in_results(self, mock_dataset):
        """Test handling when base_score is missing"""
        results_no_base = {
            'avg_overall_impact': 0.10,
            'avg_raw_impact': 0.08,
            'avg_quantile_impact': 0.12,
            'raw': {'by_level': {}},
            'quantile': {'by_level': {}},
        }

        with patch('deepbridge.validation.wrappers.robustness_suite.RobustnessSuite.__init__', return_value=None):
            with patch('deepbridge.validation.wrappers.robustness_suite.RobustnessSuite.config') as mock_config:
                mock_config.return_value = MagicMock(run=Mock(return_value=results_no_base))

                results = run_robustness_tests(mock_dataset)

                # Should not crash, metrics should have base_score of 0
                assert results['metrics']['base_score'] == 0

    def test_missing_impact_metrics_in_verbose(self, mock_dataset, capsys):
        """Test verbose output when impact metrics are missing"""
        results_no_impact = {
            'base_score': 0.85,
            'raw': {'by_level': {}},
            'quantile': {'by_level': {}},
        }

        with patch('deepbridge.validation.wrappers.robustness_suite.RobustnessSuite.__init__', return_value=None):
            with patch('deepbridge.validation.wrappers.robustness_suite.RobustnessSuite.config') as mock_config:
                mock_config.return_value = MagicMock(run=Mock(return_value=results_no_impact))

                results = run_robustness_tests(mock_dataset, verbose=True)

                captured = capsys.readouterr()
                # Should not crash, should use 0 for missing metrics
                assert 'Robustness Test Summary' in captured.out

    def test_feature_subset_verbose_output(self, mock_dataset, mock_robustness_results, capsys):
        """Test verbose output includes feature subset info"""
        with patch('deepbridge.validation.wrappers.robustness_suite.RobustnessSuite.__init__', return_value=None):
            with patch('deepbridge.validation.wrappers.robustness_suite.RobustnessSuite.config') as mock_config:
                # Configure to handle both calls (all features + subset)
                call_count = [0]
                def side_effect_run():
                    call_count[0] += 1
                    result = mock_robustness_results.copy()
                    if call_count[0] == 2:  # Second call is subset
                        result['feature_subset'] = ['feature_1', 'feature_2']
                    return result

                mock_run = Mock(side_effect=side_effect_run)
                mock_config.return_value = MagicMock(run=mock_run)

                feature_subset = ['feature_1', 'feature_2']
                results = run_robustness_tests(
                    mock_dataset, feature_subset=feature_subset, verbose=True
                )

                captured = capsys.readouterr()
                assert 'Features perturbed' in captured.out
                assert 'feature_1' in captured.out
