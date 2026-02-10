"""
Comprehensive tests for validation utility functions.

This test suite validates the utility wrappers for:
1. Hyperparameter importance testing
2. Resilience testing
3. Uncertainty quantification

Coverage Target: ~95%+
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from deepbridge.utils.hyperparameter import (
    run_hyperparameter_tests,
    hyperparameter_report_to_html,
    compare_hyperparameter_importance,
)
from deepbridge.utils.resilience import (
    run_resilience_tests,
    resilience_report_to_html,
    compare_models_resilience,
)
from deepbridge.utils.uncertainty import (
    run_uncertainty_tests,
    plot_uncertainty_results,
    compare_models_uncertainty,
    uncertainty_report_to_html,
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
def mock_hyperparameter_results():
    """Mock results from hyperparameter tests"""
    return {
        'tuning_order': ['learning_rate', 'n_estimators', 'max_depth'],
        'sorted_importance': {
            'learning_rate': 0.45,
            'n_estimators': 0.32,
            'max_depth': 0.23,
        },
        'importance_scores': {
            'learning_rate': 0.45,
            'n_estimators': 0.32,
            'max_depth': 0.23,
            'min_samples_split': 0.12,
        },
    }


@pytest.fixture
def mock_resilience_results():
    """Mock results from resilience tests"""
    return {
        'resilience_score': 0.85,
        'distribution_shift': {
            'by_alpha': {
                0.1: {'avg_performance_gap': 0.05},
                0.5: {'avg_performance_gap': 0.15},
                1.0: {'avg_performance_gap': 0.25},
            },
            'by_distance_metric': {
                'euclidean': {
                    'top_features': {'feature_1': 0.8, 'feature_2': 0.6}
                },
                'manhattan': {
                    'top_features': {'feature_1': 0.7, 'feature_3': 0.5}
                },
            },
        },
    }


@pytest.fixture
def mock_uncertainty_results():
    """Mock results from uncertainty tests"""
    return {
        'uncertainty_quality_score': 0.92,
        'avg_coverage_error': 0.03,
        'avg_normalized_width': 0.25,
        'mean_width': 1.5,
        'threshold_value': 0.7,
        'reliability_analysis': {'reliable_count': 85, 'unreliable_count': 15},
    }


# ==================== Hyperparameter Tests ====================


class TestRunHyperparameterTests:
    """Tests for run_hyperparameter_tests function"""

    def test_basic_run(self, mock_dataset, mock_hyperparameter_results):
        """Test basic hyperparameter test run"""
        # Mock the instance methods, not the class
        with patch('deepbridge.validation.wrappers.hyperparameter_suite.HyperparameterSuite.__init__', return_value=None):
            with patch('deepbridge.validation.wrappers.hyperparameter_suite.HyperparameterSuite.config') as mock_config:
                with patch('deepbridge.validation.wrappers.hyperparameter_suite.HyperparameterSuite.run', return_value=mock_hyperparameter_results):
                    # config should return self for chaining
                    mock_config.return_value = MagicMock(run=Mock(return_value=mock_hyperparameter_results))

                    results = run_hyperparameter_tests(mock_dataset)

                    assert results == mock_hyperparameter_results

    def test_with_custom_config(self, mock_dataset, mock_hyperparameter_results):
        """Test with custom configuration"""
        with patch('deepbridge.validation.wrappers.hyperparameter_suite.HyperparameterSuite.__init__', return_value=None):
            with patch('deepbridge.validation.wrappers.hyperparameter_suite.HyperparameterSuite.config') as mock_config:
                mock_config.return_value = MagicMock(run=Mock(return_value=mock_hyperparameter_results))

                results = run_hyperparameter_tests(
                    mock_dataset, config_name='full', metric='f1'
                )

                # Verify config called with 'full'
                mock_config.assert_called_once_with('full', feature_subset=None)

    def test_with_feature_subset(self, mock_dataset, mock_hyperparameter_results):
        """Test with feature subset"""
        with patch('deepbridge.validation.wrappers.hyperparameter_suite.HyperparameterSuite.__init__', return_value=None):
            with patch('deepbridge.validation.wrappers.hyperparameter_suite.HyperparameterSuite.config') as mock_config:
                mock_config.return_value = MagicMock(run=Mock(return_value=mock_hyperparameter_results))

                feature_subset = ['feature1', 'feature2']
                results = run_hyperparameter_tests(
                    mock_dataset, feature_subset=feature_subset
                )

                mock_config.assert_called_once_with('quick', feature_subset=feature_subset)

    def test_verbose_output(self, mock_dataset, mock_hyperparameter_results, capsys):
        """Test verbose output printing"""
        with patch('deepbridge.validation.wrappers.hyperparameter_suite.HyperparameterSuite.__init__', return_value=None):
            with patch('deepbridge.validation.wrappers.hyperparameter_suite.HyperparameterSuite.config') as mock_config:
                mock_config.return_value = MagicMock(run=Mock(return_value=mock_hyperparameter_results))

                results = run_hyperparameter_tests(mock_dataset, verbose=True)

                # Check that verbose output was printed
                captured = capsys.readouterr()
                assert 'Hyperparameter Importance Summary' in captured.out
                assert 'learning_rate' in captured.out

    def test_no_verbose_output(self, mock_dataset, mock_hyperparameter_results, capsys):
        """Test no output when verbose=False"""
        with patch('deepbridge.validation.wrappers.hyperparameter_suite.HyperparameterSuite.__init__', return_value=None):
            with patch('deepbridge.validation.wrappers.hyperparameter_suite.HyperparameterSuite.config') as mock_config:
                mock_config.return_value = MagicMock(run=Mock(return_value=mock_hyperparameter_results))

                results = run_hyperparameter_tests(mock_dataset, verbose=False)

                # Check that no summary was printed
                captured = capsys.readouterr()
                assert 'Hyperparameter Importance Summary' not in captured.out


class TestHyperparameterReportToHTML:
    """Tests for deprecated hyperparameter_report_to_html function"""

    def test_raises_not_implemented(self):
        """Test that function raises NotImplementedError"""
        with pytest.raises(NotImplementedError, match='Report generation'):
            hyperparameter_report_to_html({})


class TestCompareHyperparameterImportance:
    """Tests for compare_hyperparameter_importance function"""

    def test_basic_comparison(self):
        """Test basic comparison of two models"""
        results_dict = {
            'model1': {
                'importance_scores': {
                    'learning_rate': 0.5,
                    'n_estimators': 0.3,
                    'max_depth': 0.2,
                }
            },
            'model2': {
                'importance_scores': {
                    'learning_rate': 0.4,
                    'n_estimators': 0.35,
                    'max_depth': 0.25,
                }
            },
        }

        comparison = compare_hyperparameter_importance(results_dict)

        assert 'model1' in comparison['model_names']
        assert 'model2' in comparison['model_names']
        assert len(comparison['common_params']) == 3
        assert 'learning_rate' in comparison['param_importance']

    def test_comparison_with_different_params(self):
        """Test comparison when models have different parameters"""
        results_dict = {
            'model1': {'importance_scores': {'learning_rate': 0.5, 'n_estimators': 0.3}},
            'model2': {'importance_scores': {'learning_rate': 0.4, 'max_depth': 0.25}},
        }

        comparison = compare_hyperparameter_importance(results_dict)

        # Only learning_rate is common
        assert len(comparison['common_params']) == 1
        assert 'learning_rate' in comparison['common_params']
        assert 'n_estimators' not in comparison['common_params']

    def test_param_importance_structure(self):
        """Test structure of param_importance output"""
        results_dict = {
            'model1': {'importance_scores': {'learning_rate': 0.5}},
            'model2': {'importance_scores': {'learning_rate': 0.4}},
        }

        comparison = compare_hyperparameter_importance(results_dict)

        # Check structure
        assert 'learning_rate' in comparison['param_importance']
        importance_list = comparison['param_importance']['learning_rate']
        assert len(importance_list) == 2
        assert importance_list[0]['model'] == 'model1'
        assert importance_list[0]['importance'] == 0.5


# ==================== Resilience Tests ====================


class TestRunResilienceTests:
    """Tests for run_resilience_tests function"""

    def test_basic_run(self, mock_dataset, mock_resilience_results):
        """Test basic resilience test run"""
        with patch('deepbridge.validation.wrappers.resilience_suite.ResilienceSuite.__init__', return_value=None):
            with patch('deepbridge.validation.wrappers.resilience_suite.ResilienceSuite.config') as mock_config:
                mock_config.return_value = MagicMock(run=Mock(return_value=mock_resilience_results))

                results = run_resilience_tests(mock_dataset)

                assert results == mock_resilience_results

    def test_with_custom_metric(self, mock_dataset, mock_resilience_results):
        """Test with custom metric"""
        with patch('deepbridge.validation.wrappers.resilience_suite.ResilienceSuite.__init__', return_value=None):
            with patch('deepbridge.validation.wrappers.resilience_suite.ResilienceSuite.config') as mock_config:
                mock_config.return_value = MagicMock(run=Mock(return_value=mock_resilience_results))

                results = run_resilience_tests(mock_dataset, metric='f1')

                mock_config.assert_called_once_with('quick', feature_subset=None)

    def test_verbose_output(self, mock_dataset, mock_resilience_results, capsys):
        """Test verbose output"""
        with patch('deepbridge.validation.wrappers.resilience_suite.ResilienceSuite.__init__', return_value=None):
            with patch('deepbridge.validation.wrappers.resilience_suite.ResilienceSuite.config') as mock_config:
                mock_config.return_value = MagicMock(run=Mock(return_value=mock_resilience_results))

                results = run_resilience_tests(mock_dataset, verbose=True)

                captured = capsys.readouterr()
                assert 'Resilience Test Summary' in captured.out
                assert 'resilience score' in captured.out


class TestResilienceReportToHTML:
    """Tests for deprecated resilience_report_to_html function"""

    def test_raises_not_implemented(self):
        """Test that function raises NotImplementedError"""
        with pytest.raises(NotImplementedError, match='Report generation'):
            resilience_report_to_html({})


class TestCompareModelsResilience:
    """Tests for compare_models_resilience function"""

    def test_basic_comparison(self, mock_resilience_results):
        """Test basic model comparison"""
        results_dict = {
            'model1': mock_resilience_results,
            'model2': mock_resilience_results,
        }

        comparison = compare_models_resilience(results_dict)

        assert len(comparison['model_names']) == 2
        assert len(comparison['resilience_scores']) == 2

    def test_performance_gaps_collection(self, mock_resilience_results):
        """Test that performance gaps are collected correctly"""
        results_dict = {'model1': mock_resilience_results}

        comparison = compare_models_resilience(results_dict)

        assert 0.1 in comparison['performance_gaps']
        assert 0.5 in comparison['performance_gaps']
        assert comparison['performance_gaps'][0.1][0]['gap'] == 0.05

    def test_feature_importance_collection(self, mock_resilience_results):
        """Test that feature importance is collected correctly"""
        results_dict = {'model1': mock_resilience_results}

        comparison = compare_models_resilience(results_dict)

        assert 'euclidean' in comparison['feature_importance']
        assert 'feature_1' in comparison['feature_importance']['euclidean']


# ==================== Uncertainty Tests ====================


class TestRunUncertaintyTests:
    """Tests for run_uncertainty_tests function"""

    def test_with_enhanced_version(self, mock_dataset, mock_uncertainty_results, capsys):
        """Test using enhanced uncertainty version"""
        with patch(
            'deepbridge.validation.wrappers.enhanced_uncertainty_suite.run_enhanced_uncertainty_tests'
        ) as mock_enhanced:
            mock_enhanced.return_value = mock_uncertainty_results

            results = run_uncertainty_tests(mock_dataset, verbose=True)

            mock_enhanced.assert_called_once()
            assert results == mock_uncertainty_results

            captured = capsys.readouterr()
            assert 'enhanced uncertainty analysis' in captured.out

    def test_fallback_to_standard(self, mock_dataset, mock_uncertainty_results, capsys):
        """Test fallback to standard version when enhanced is not available"""
        with patch(
            'deepbridge.validation.wrappers.enhanced_uncertainty_suite.run_enhanced_uncertainty_tests',
            side_effect=ImportError,
        ):
            with patch('deepbridge.validation.wrappers.uncertainty_suite.UncertaintySuite.__init__', return_value=None):
                with patch('deepbridge.validation.wrappers.uncertainty_suite.UncertaintySuite.config') as mock_config:
                    mock_config.return_value = MagicMock(run=Mock(return_value=mock_uncertainty_results))

                    results = run_uncertainty_tests(mock_dataset, verbose=True)

                    assert results == mock_uncertainty_results

                    captured = capsys.readouterr()
                    assert 'standard uncertainty analysis' in captured.out

    def test_verbose_output(self, mock_dataset, mock_uncertainty_results, capsys):
        """Test verbose output with all metrics"""
        with patch(
            'deepbridge.validation.wrappers.enhanced_uncertainty_suite.run_enhanced_uncertainty_tests'
        ) as mock_enhanced:
            mock_enhanced.return_value = mock_uncertainty_results

            results = run_uncertainty_tests(mock_dataset, verbose=True)

            captured = capsys.readouterr()
            assert 'Uncertainty Test Summary' in captured.out
            assert '0.92' in captured.out  # uncertainty quality score
            assert '85/100' in captured.out  # reliable predictions

    def test_with_feature_subset(self, mock_dataset, mock_uncertainty_results):
        """Test with feature subset"""
        with patch(
            'deepbridge.validation.wrappers.enhanced_uncertainty_suite.run_enhanced_uncertainty_tests'
        ) as mock_enhanced:
            mock_enhanced.return_value = mock_uncertainty_results

            feature_subset = ['feature1', 'feature2']
            results = run_uncertainty_tests(
                mock_dataset, feature_subset=feature_subset
            )

            call_args = mock_enhanced.call_args[0]
            assert call_args[3] == feature_subset  # 4th argument


class TestPlotUncertaintyResults:
    """Tests for deprecated plot_uncertainty_results function"""

    def test_raises_not_implemented(self):
        """Test that function raises NotImplementedError"""
        with pytest.raises(NotImplementedError, match='Visualization'):
            plot_uncertainty_results({})


class TestCompareModelsUncertainty:
    """Tests for deprecated compare_models_uncertainty function"""

    def test_raises_not_implemented(self):
        """Test that function raises NotImplementedError"""
        with pytest.raises(NotImplementedError, match='Visualization'):
            compare_models_uncertainty({})


class TestUncertaintyReportToHTML:
    """Tests for deprecated uncertainty_report_to_html function"""

    def test_raises_not_implemented(self):
        """Test that function raises NotImplementedError"""
        with pytest.raises(NotImplementedError, match='Report generation'):
            uncertainty_report_to_html({})


# ==================== Edge Cases ====================


class TestEdgeCases:
    """Tests for edge cases and error handling"""

    def test_hyperparameter_with_empty_results(self, mock_dataset, capsys):
        """Test hyperparameter function with empty results"""
        with patch('deepbridge.validation.wrappers.hyperparameter_suite.HyperparameterSuite.__init__', return_value=None):
            with patch('deepbridge.validation.wrappers.hyperparameter_suite.HyperparameterSuite.config') as mock_config:
                empty_results = {'tuning_order': [], 'sorted_importance': {}}
                mock_config.return_value = MagicMock(run=Mock(return_value=empty_results))

                results = run_hyperparameter_tests(mock_dataset, verbose=True)

                # Should not crash with empty results
                assert results == empty_results

    def test_compare_hyperparameter_with_empty_dict(self):
        """Test compare function with empty dictionary"""
        comparison = compare_hyperparameter_importance({})

        assert comparison['model_names'] == []
        assert len(comparison['common_params']) == 0

    def test_resilience_with_missing_alpha_data(self, mock_dataset):
        """Test resilience with missing alpha data"""
        results_missing_alpha = {
            'resilience_score': 0.8,
            'distribution_shift': {'by_alpha': {}},
        }

        with patch('deepbridge.validation.wrappers.resilience_suite.ResilienceSuite.__init__', return_value=None):
            with patch('deepbridge.validation.wrappers.resilience_suite.ResilienceSuite.config') as mock_config:
                mock_config.return_value = MagicMock(run=Mock(return_value=results_missing_alpha))

                results = run_resilience_tests(mock_dataset, verbose=True)

                # Should not crash with missing alpha data
                assert results == results_missing_alpha

    def test_compare_resilience_with_empty_dict(self):
        """Test compare resilience with empty dictionary"""
        comparison = compare_models_resilience({})

        assert comparison['model_names'] == []
        assert comparison['resilience_scores'] == []
