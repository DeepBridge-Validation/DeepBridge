"""
Comprehensive tests for RobustnessManager.

This test suite validates:
1. run_tests - test execution with different configurations
2. compare_models - model comparison functionality
3. compare_models_robustness - robustness score comparison logic
4. Alternative model testing
5. Edge cases and error handling

Coverage Target: ~90%+
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
import copy

from deepbridge.core.experiment.managers.robustness_manager import RobustnessManager


# ==================== Fixtures ====================


@pytest.fixture
def mock_dataset():
    """Create a mock dataset"""
    dataset = Mock()
    dataset.set_model = Mock()
    return dataset


@pytest.fixture
def mock_alternative_models():
    """Create mock alternative models"""
    return {
        'model_1': Mock(name='Model1'),
        'model_2': Mock(name='Model2')
    }


@pytest.fixture
def manager(mock_dataset):
    """Create RobustnessManager instance"""
    return RobustnessManager(dataset=mock_dataset, verbose=False)


@pytest.fixture
def manager_with_alternatives(mock_dataset, mock_alternative_models):
    """Create RobustnessManager with alternative models"""
    return RobustnessManager(
        dataset=mock_dataset,
        alternative_models=mock_alternative_models,
        verbose=False
    )


@pytest.fixture
def verbose_manager(mock_dataset):
    """Create verbose RobustnessManager"""
    return RobustnessManager(dataset=mock_dataset, verbose=True)


@pytest.fixture
def mock_robustness_suite():
    """Create mock RobustnessSuite class"""
    mock_suite = Mock()
    mock_instance = Mock()
    mock_instance.config = Mock(return_value=mock_instance)
    mock_instance.run = Mock(return_value={
        'robustness_scores': {'overall_score': 0.85},
        'perturbation_results': []
    })
    mock_suite.return_value = mock_instance
    return mock_suite


# ==================== Basic Initialization Tests ====================


class TestInitialization:
    """Tests for manager initialization"""

    def test_initialization_with_dataset(self, mock_dataset):
        """Test basic initialization"""
        manager = RobustnessManager(dataset=mock_dataset)

        assert manager.dataset == mock_dataset
        assert manager.alternative_models == {}
        assert manager.verbose is False

    def test_initialization_with_alternative_models(self, mock_dataset, mock_alternative_models):
        """Test initialization with alternative models"""
        manager = RobustnessManager(
            dataset=mock_dataset,
            alternative_models=mock_alternative_models
        )

        assert manager.alternative_models == mock_alternative_models

    def test_initialization_with_verbose(self, mock_dataset):
        """Test initialization with verbose mode"""
        manager = RobustnessManager(dataset=mock_dataset, verbose=True)

        assert manager.verbose is True


# ==================== run_tests Tests ====================


class TestRunTests:
    """Tests for run_tests method"""

    @patch('deepbridge.validation.wrappers.robustness_suite.RobustnessSuite')
    def test_run_tests_basic(self, mock_suite_class, manager, mock_dataset):
        """Test basic test execution"""
        # Setup mock
        mock_instance = Mock()
        mock_instance.config = Mock(return_value=mock_instance)
        mock_instance.run = Mock(return_value={
            'robustness_scores': {'overall_score': 0.85}
        })
        mock_suite_class.return_value = mock_instance

        result = manager.run_tests(config_name='quick')

        # Verify RobustnessSuite was created correctly
        mock_suite_class.assert_called_once_with(
            dataset=mock_dataset,
            verbose=False,
            n_iterations=1
        )

        # Verify config and run were called
        mock_instance.config.assert_called_once_with('quick')
        mock_instance.run.assert_called_once()

        # Verify result structure
        assert 'main_model' in result
        assert 'alternative_models' in result
        assert 'comparison' in result

    @patch('deepbridge.validation.wrappers.robustness_suite.RobustnessSuite')
    def test_run_tests_with_custom_iterations(self, mock_suite_class, manager):
        """Test with custom n_iterations"""
        mock_instance = Mock()
        mock_instance.config = Mock(return_value=mock_instance)
        mock_instance.run = Mock(return_value={})
        mock_suite_class.return_value = mock_instance

        manager.run_tests(config_name='quick', n_iterations=5)

        # Verify n_iterations was passed correctly
        mock_suite_class.assert_called_once()
        call_kwargs = mock_suite_class.call_args[1]
        assert call_kwargs['n_iterations'] == 5

    @patch('deepbridge.validation.wrappers.robustness_suite.RobustnessSuite')
    def test_run_tests_with_alternative_models(self, mock_suite_class, manager_with_alternatives, mock_dataset):
        """Test execution with alternative models"""
        # Setup mock
        mock_instance = Mock()
        mock_instance.config = Mock(return_value=mock_instance)
        mock_instance.run = Mock(return_value={
            'robustness_scores': {'overall_score': 0.85}
        })
        mock_suite_class.return_value = mock_instance

        with patch('deepbridge.core.experiment.managers.robustness_manager.copy.deepcopy') as mock_deepcopy:
            mock_temp_dataset = Mock()
            mock_deepcopy.return_value = mock_temp_dataset

            result = manager_with_alternatives.run_tests(config_name='medium')

            # Should be called 3 times: once for main, twice for alternatives
            assert mock_suite_class.call_count == 3

            # Verify alternative models were tested
            assert len(result['alternative_models']) == 2
            assert 'model_1' in result['alternative_models']
            assert 'model_2' in result['alternative_models']

            # Verify temp datasets were created and models were set
            assert mock_deepcopy.call_count == 2
            assert mock_temp_dataset.set_model.call_count == 2

    @patch('deepbridge.validation.wrappers.robustness_suite.RobustnessSuite')
    def test_run_tests_with_different_configs(self, mock_suite_class, manager):
        """Test with different configuration profiles"""
        mock_instance = Mock()
        mock_instance.config = Mock(return_value=mock_instance)
        mock_instance.run = Mock(return_value={})
        mock_suite_class.return_value = mock_instance

        for config in ['quick', 'medium', 'full']:
            manager.run_tests(config_name=config)
            mock_instance.config.assert_called_with(config)

    @patch('deepbridge.validation.wrappers.robustness_suite.RobustnessSuite')
    def test_run_tests_calls_compare_models_robustness(self, mock_suite_class, manager):
        """Test that comparison is performed"""
        mock_instance = Mock()
        mock_instance.config = Mock(return_value=mock_instance)
        mock_instance.run = Mock(return_value={
            'robustness_scores': {'overall_score': 0.85}
        })
        mock_suite_class.return_value = mock_instance

        with patch.object(manager, 'compare_models_robustness', return_value={'test': 'comparison'}) as mock_compare:
            result = manager.run_tests()

            mock_compare.assert_called_once()
            assert result['comparison'] == {'test': 'comparison'}


# ==================== compare_models Tests ====================


class TestCompareModels:
    """Tests for compare_models method"""

    @patch('deepbridge.validation.wrappers.robustness_suite.RobustnessSuite')
    def test_compare_models_calls_run_tests(self, mock_suite_class, manager):
        """Test that compare_models calls run_tests"""
        mock_instance = Mock()
        mock_instance.config = Mock(return_value=mock_instance)
        mock_instance.run = Mock(return_value={})
        mock_suite_class.return_value = mock_instance

        result = manager.compare_models(config_name='full')

        # Should have same structure as run_tests
        assert 'main_model' in result
        assert 'alternative_models' in result
        assert 'comparison' in result

    @patch('deepbridge.validation.wrappers.robustness_suite.RobustnessSuite')
    def test_compare_models_passes_kwargs(self, mock_suite_class, manager):
        """Test that kwargs are passed through"""
        mock_instance = Mock()
        mock_instance.config = Mock(return_value=mock_instance)
        mock_instance.run = Mock(return_value={})
        mock_suite_class.return_value = mock_instance

        manager.compare_models(config_name='medium', n_iterations=10)

        # Verify n_iterations was used
        call_kwargs = mock_suite_class.call_args[1]
        assert call_kwargs['n_iterations'] == 10


# ==================== compare_models_robustness Tests ====================


class TestCompareModelsRobustness:
    """Tests for compare_models_robustness method"""

    def test_compare_with_robustness_scores(self, manager):
        """Test comparison with robustness_scores structure"""
        results = {
            'main_model': {
                'robustness_scores': {'overall_score': 0.85}
            },
            'alternative_models': {
                'model_1': {
                    'robustness_scores': {'overall_score': 0.90}
                },
                'model_2': {
                    'robustness_scores': {'overall_score': 0.75}
                }
            }
        }

        comparison = manager.compare_models_robustness(results)

        assert comparison['most_robust_model'] == 'model_1'
        assert comparison['most_robust_score'] == 0.90
        assert comparison['all_scores']['main_model'] == 0.85
        assert comparison['all_scores']['model_1'] == 0.90
        assert comparison['all_scores']['model_2'] == 0.75

    def test_compare_with_robustness_score(self, manager):
        """Test comparison with robustness_score structure"""
        results = {
            'main_model': {
                'robustness_score': 0.80
            },
            'alternative_models': {
                'model_1': {
                    'robustness_score': 0.85
                }
            }
        }

        comparison = manager.compare_models_robustness(results)

        assert comparison['most_robust_model'] == 'model_1'
        assert comparison['most_robust_score'] == 0.85

    def test_compare_with_main_model_as_best(self, manager):
        """Test when main model is most robust"""
        results = {
            'main_model': {
                'robustness_scores': {'overall_score': 0.95}
            },
            'alternative_models': {
                'model_1': {
                    'robustness_scores': {'overall_score': 0.85}
                }
            }
        }

        comparison = manager.compare_models_robustness(results)

        assert comparison['most_robust_model'] == 'main_model'
        assert comparison['most_robust_score'] == 0.95

    def test_compare_with_none_scores(self, manager):
        """Test comparison with None scores"""
        results = {
            'main_model': {},  # No score
            'alternative_models': {
                'model_1': {
                    'robustness_scores': {'overall_score': 0.85}
                }
            }
        }

        comparison = manager.compare_models_robustness(results)

        # Model with actual score should win
        assert comparison['most_robust_model'] == 'model_1'
        assert comparison['all_scores']['main_model'] is None

    def test_compare_with_all_none_scores(self, manager):
        """Test comparison when all scores are None"""
        results = {
            'main_model': {},
            'alternative_models': {
                'model_1': {}
            }
        }

        comparison = manager.compare_models_robustness(results)

        # Should still identify a winner (main_model by default due to dict order)
        assert 'most_robust_model' in comparison
        assert comparison['most_robust_score'] is None

    def test_compare_with_no_alternative_models(self, manager):
        """Test comparison with only main model"""
        results = {
            'main_model': {
                'robustness_scores': {'overall_score': 0.85}
            }
        }

        comparison = manager.compare_models_robustness(results)

        assert comparison['most_robust_model'] == 'main_model'
        assert comparison['most_robust_score'] == 0.85
        assert len(comparison['all_scores']) == 1

    def test_compare_logs_results_when_verbose(self, verbose_manager, capsys):
        """Test that results are logged in verbose mode"""
        results = {
            'main_model': {
                'robustness_scores': {'overall_score': 0.85}
            },
            'alternative_models': {
                'model_1': {
                    'robustness_scores': {'overall_score': 0.90}
                }
            }
        }

        verbose_manager.compare_models_robustness(results)

        captured = capsys.readouterr()
        assert 'Robustness comparison results' in captured.out
        assert 'main_model: 0.8500' in captured.out
        assert 'model_1: 0.9000' in captured.out
        assert 'Most robust model: model_1' in captured.out

    def test_compare_logs_none_scores(self, verbose_manager, capsys):
        """Test logging when scores are None"""
        results = {
            'main_model': {},
            'alternative_models': {}
        }

        verbose_manager.compare_models_robustness(results)

        captured = capsys.readouterr()
        assert 'main_model: None' in captured.out
        assert 'Most robust model' in captured.out


# ==================== Logging Tests ====================


class TestLogging:
    """Tests for logging functionality"""

    def test_log_when_verbose_enabled(self, verbose_manager, capsys):
        """Test that messages are logged when verbose=True"""
        verbose_manager.log('Test message')

        captured = capsys.readouterr()
        assert 'Test message' in captured.out

    def test_log_when_verbose_disabled(self, manager, capsys):
        """Test that messages are not logged when verbose=False"""
        manager.log('Test message')

        captured = capsys.readouterr()
        assert 'Test message' not in captured.out

    @patch('deepbridge.validation.wrappers.robustness_suite.RobustnessSuite')
    def test_run_tests_logs_progress(self, mock_suite_class, verbose_manager, capsys):
        """Test that progress is logged during test execution"""
        mock_instance = Mock()
        mock_instance.config = Mock(return_value=mock_instance)
        mock_instance.run = Mock(return_value={})
        mock_suite_class.return_value = mock_instance

        verbose_manager.run_tests(n_iterations=3)

        captured = capsys.readouterr()
        assert 'Running robustness tests with 3 iterations' in captured.out

    @patch('deepbridge.validation.wrappers.robustness_suite.RobustnessSuite')
    def test_run_tests_logs_alternative_model_testing(self, mock_suite_class, manager_with_alternatives, capsys):
        """Test logging for alternative model testing"""
        # Make manager verbose
        manager_with_alternatives.verbose = True

        mock_instance = Mock()
        mock_instance.config = Mock(return_value=mock_instance)
        mock_instance.run = Mock(return_value={})
        mock_suite_class.return_value = mock_instance

        with patch('deepbridge.core.experiment.managers.robustness_manager.copy.deepcopy', return_value=Mock()):
            manager_with_alternatives.run_tests()

        captured = capsys.readouterr()
        assert 'Testing robustness of alternative model: model_1' in captured.out
        assert 'Testing robustness of alternative model: model_2' in captured.out


# ==================== Integration Tests ====================


class TestIntegration:
    """Integration tests for complete workflows"""

    @patch('deepbridge.validation.wrappers.robustness_suite.RobustnessSuite')
    def test_full_workflow_with_alternatives(self, mock_suite_class, manager_with_alternatives):
        """Test complete workflow with alternative models"""
        mock_instance = Mock()
        mock_instance.config = Mock(return_value=mock_instance)
        mock_instance.run = Mock(side_effect=[
            {'robustness_scores': {'overall_score': 0.85}},  # main model
            {'robustness_scores': {'overall_score': 0.90}},  # model_1
            {'robustness_scores': {'overall_score': 0.80}},  # model_2
        ])
        mock_suite_class.return_value = mock_instance

        with patch('deepbridge.core.experiment.managers.robustness_manager.copy.deepcopy', return_value=Mock()):
            result = manager_with_alternatives.run_tests(config_name='full', n_iterations=10)

        # Verify all components
        assert 'main_model' in result
        assert 'alternative_models' in result
        assert 'comparison' in result

        # Verify alternative models were tested
        assert 'model_1' in result['alternative_models']
        assert 'model_2' in result['alternative_models']

        # Verify comparison identified best model
        assert result['comparison']['most_robust_model'] == 'model_1'
        assert result['comparison']['most_robust_score'] == 0.90


# ==================== Edge Cases ====================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    @patch('deepbridge.validation.wrappers.robustness_suite.RobustnessSuite')
    def test_run_tests_with_empty_alternative_models(self, mock_suite_class, mock_dataset):
        """Test with empty alternative_models dict"""
        manager = RobustnessManager(dataset=mock_dataset, alternative_models={})

        mock_instance = Mock()
        mock_instance.config = Mock(return_value=mock_instance)
        mock_instance.run = Mock(return_value={})
        mock_suite_class.return_value = mock_instance

        result = manager.run_tests()

        # Should still work, just no alternative models tested
        assert len(result['alternative_models']) == 0

    def test_compare_with_missing_main_model(self, manager):
        """Test comparison with missing main_model key"""
        results = {
            'alternative_models': {
                'model_1': {'robustness_scores': {'overall_score': 0.85}}
            }
        }

        comparison = manager.compare_models_robustness(results)

        # Should handle gracefully
        assert 'all_scores' in comparison
        assert 'main_model' in comparison['all_scores']

    def test_compare_with_missing_alternative_models(self, manager):
        """Test comparison with missing alternative_models key"""
        results = {
            'main_model': {'robustness_scores': {'overall_score': 0.85}}
        }

        comparison = manager.compare_models_robustness(results)

        # Should handle gracefully
        assert comparison['most_robust_model'] == 'main_model'
