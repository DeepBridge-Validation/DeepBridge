"""
Comprehensive tests for HyperparameterManager.

This test suite validates:
1. run_tests - running hyperparameter tests on primary model
2. compare_models - comparing hyperparameter importance across models
3. Metric parameter handling
4. Alternative model handling
5. Edge cases and error handling

Coverage Target: ~95%+
"""

import pytest
from unittest.mock import Mock, patch

from deepbridge.core.experiment.managers.hyperparameter_manager import HyperparameterManager


# ==================== Fixtures ====================


@pytest.fixture
def mock_dataset():
    """Create mock dataset"""
    dataset = Mock()
    dataset.model = Mock()
    return dataset


@pytest.fixture
def manager(mock_dataset):
    """Create HyperparameterManager instance"""
    return HyperparameterManager(dataset=mock_dataset)


@pytest.fixture
def verbose_manager(mock_dataset):
    """Create HyperparameterManager with verbose=True"""
    return HyperparameterManager(dataset=mock_dataset, verbose=True)


@pytest.fixture
def manager_with_alternatives(mock_dataset):
    """Create manager with alternative models"""
    alt_models = {
        'model1': Mock(),
        'model2': Mock(),
    }
    return HyperparameterManager(
        dataset=mock_dataset,
        alternative_models=alt_models,
        verbose=False
    )


# ==================== run_tests Tests ====================


class TestRunTests:
    """Tests for run_tests method"""

    @patch('deepbridge.utils.hyperparameter.run_hyperparameter_tests')
    def test_run_tests_with_defaults(self, mock_run_tests, manager, mock_dataset):
        """Test run_tests with default parameters"""
        mock_run_tests.return_value = {'importance': 0.85}

        result = manager.run_tests()

        assert result == {'importance': 0.85}
        mock_run_tests.assert_called_once_with(
            mock_dataset,
            config_name='quick',
            metric='accuracy',
            verbose=False,
        )

    @patch('deepbridge.utils.hyperparameter.run_hyperparameter_tests')
    def test_run_tests_with_custom_metric(self, mock_run_tests, manager, mock_dataset):
        """Test run_tests with custom metric"""
        mock_run_tests.return_value = {'importance': 0.85}

        result = manager.run_tests(metric='f1')

        mock_run_tests.assert_called_once_with(
            mock_dataset,
            config_name='quick',
            metric='f1',
            verbose=False,
        )

    @patch('deepbridge.utils.hyperparameter.run_hyperparameter_tests')
    def test_run_tests_with_quick_config(self, mock_run_tests, manager, mock_dataset):
        """Test run_tests with 'quick' config"""
        mock_run_tests.return_value = {'importance': 0.85}

        manager.run_tests(config_name='quick')

        call_kwargs = mock_run_tests.call_args[1]
        assert call_kwargs['config_name'] == 'quick'

    @patch('deepbridge.utils.hyperparameter.run_hyperparameter_tests')
    def test_run_tests_with_medium_config(self, mock_run_tests, manager, mock_dataset):
        """Test run_tests with 'medium' config"""
        mock_run_tests.return_value = {'importance': 0.85}

        manager.run_tests(config_name='medium')

        call_kwargs = mock_run_tests.call_args[1]
        assert call_kwargs['config_name'] == 'medium'

    @patch('deepbridge.utils.hyperparameter.run_hyperparameter_tests')
    def test_run_tests_with_full_config(self, mock_run_tests, manager, mock_dataset):
        """Test run_tests with 'full' config"""
        mock_run_tests.return_value = {'importance': 0.85}

        manager.run_tests(config_name='full')

        call_kwargs = mock_run_tests.call_args[1]
        assert call_kwargs['config_name'] == 'full'

    @patch('deepbridge.utils.hyperparameter.run_hyperparameter_tests')
    def test_run_tests_with_verbose_true(
        self, mock_run_tests, verbose_manager, mock_dataset
    ):
        """Test run_tests respects verbose flag"""
        mock_run_tests.return_value = {'importance': 0.85}

        verbose_manager.run_tests()

        call_kwargs = mock_run_tests.call_args[1]
        assert call_kwargs['verbose'] is True

    @patch('deepbridge.utils.hyperparameter.run_hyperparameter_tests')
    def test_run_tests_with_both_config_and_metric(
        self, mock_run_tests, manager, mock_dataset
    ):
        """Test run_tests with both config_name and metric"""
        mock_run_tests.return_value = {'importance': 0.85}

        manager.run_tests(config_name='full', metric='precision')

        mock_run_tests.assert_called_once_with(
            mock_dataset,
            config_name='full',
            metric='precision',
            verbose=False,
        )


# ==================== compare_models Tests ====================


class TestCompareModels:
    """Tests for compare_models method"""

    @patch('deepbridge.utils.hyperparameter.run_hyperparameter_tests')
    def test_compare_models_with_no_alternatives(
        self, mock_run_tests, manager, mock_dataset
    ):
        """Test compare_models with no alternative models"""
        mock_run_tests.return_value = {'importance': 0.85}

        result = manager.compare_models()

        assert 'primary_model' in result
        assert 'alternative_models' in result
        assert result['primary_model'] == {'importance': 0.85}
        assert result['alternative_models'] == {}
        mock_run_tests.assert_called_once()

    @patch('deepbridge.utils.dataset_factory.DBDatasetFactory.create_for_alternative_model')
    @patch('deepbridge.utils.hyperparameter.run_hyperparameter_tests')
    def test_compare_models_with_alternatives(
        self, mock_run_tests, mock_factory, manager_with_alternatives, mock_dataset
    ):
        """Test compare_models with alternative models"""
        mock_run_tests.side_effect = [
            {'importance': 0.85},  # primary
            {'importance': 0.80},  # model1
            {'importance': 0.82},  # model2
        ]

        mock_factory.side_effect = [Mock(), Mock()]

        result = manager_with_alternatives.compare_models()

        assert result['primary_model'] == {'importance': 0.85}
        assert len(result['alternative_models']) == 2
        assert result['alternative_models']['model1'] == {'importance': 0.80}
        assert result['alternative_models']['model2'] == {'importance': 0.82}

    @patch('deepbridge.utils.dataset_factory.DBDatasetFactory.create_for_alternative_model')
    @patch('deepbridge.utils.hyperparameter.run_hyperparameter_tests')
    def test_compare_models_with_custom_metric(
        self, mock_run_tests, mock_factory, manager_with_alternatives
    ):
        """Test compare_models with custom metric"""
        mock_run_tests.side_effect = [
            {'importance': 0.85},
            {'importance': 0.80},
            {'importance': 0.82},
        ]

        mock_factory.side_effect = [Mock(), Mock()]

        manager_with_alternatives.compare_models(metric='f1')

        # Check all calls used 'f1' metric
        for call in mock_run_tests.call_args_list:
            assert call[1]['metric'] == 'f1'

    @patch('deepbridge.utils.dataset_factory.DBDatasetFactory.create_for_alternative_model')
    @patch('deepbridge.utils.hyperparameter.run_hyperparameter_tests')
    def test_compare_models_with_config_name(
        self, mock_run_tests, mock_factory, manager_with_alternatives
    ):
        """Test compare_models with config_name"""
        mock_run_tests.side_effect = [
            {'importance': 0.85},
            {'importance': 0.80},
            {'importance': 0.82},
        ]

        mock_factory.side_effect = [Mock(), Mock()]

        manager_with_alternatives.compare_models(config_name='medium')

        # Check all calls used 'medium' config
        for call in mock_run_tests.call_args_list:
            assert call[1]['config_name'] == 'medium'

    @patch('deepbridge.utils.dataset_factory.DBDatasetFactory.create_for_alternative_model')
    @patch('deepbridge.utils.hyperparameter.run_hyperparameter_tests')
    def test_compare_models_creates_alternative_datasets(
        self, mock_run_tests, mock_factory, manager_with_alternatives, mock_dataset
    ):
        """Test compare_models creates datasets for alternative models"""
        mock_run_tests.side_effect = [
            {'importance': 0.85},
            {'importance': 0.80},
            {'importance': 0.82},
        ]

        alt_dataset1 = Mock()
        alt_dataset2 = Mock()
        mock_factory.side_effect = [alt_dataset1, alt_dataset2]

        manager_with_alternatives.compare_models()

        assert mock_factory.call_count == 2


# ==================== Integration Tests ====================


class TestIntegration:
    """Integration tests for complete workflows"""

    @patch('deepbridge.utils.hyperparameter.run_hyperparameter_tests')
    def test_full_workflow_run_then_compare(
        self, mock_run_tests, manager_with_alternatives
    ):
        """Test complete workflow: run_tests then compare_models"""
        mock_run_tests.side_effect = [
            {'importance': 0.85},  # run_tests
            {'importance': 0.85},  # compare primary
            {'importance': 0.80},  # compare model1
            {'importance': 0.82},  # compare model2
        ]

        with patch('deepbridge.utils.dataset_factory.DBDatasetFactory.create_for_alternative_model') as mock_factory:
            mock_factory.side_effect = [Mock(), Mock()]

            result1 = manager_with_alternatives.run_tests()
            assert result1 == {'importance': 0.85}

            result2 = manager_with_alternatives.compare_models()
            assert 'primary_model' in result2
            assert 'alternative_models' in result2


# ==================== Edge Cases ====================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    @patch('deepbridge.utils.hyperparameter.run_hyperparameter_tests')
    def test_run_tests_returns_empty_dict(self, mock_run_tests, manager):
        """Test run_tests when underlying function returns empty dict"""
        mock_run_tests.return_value = {}

        result = manager.run_tests()

        assert result == {}

    @patch('deepbridge.utils.dataset_factory.DBDatasetFactory.create_for_alternative_model')
    @patch('deepbridge.utils.hyperparameter.run_hyperparameter_tests')
    def test_compare_models_with_single_alternative(
        self, mock_run_tests, mock_factory, mock_dataset
    ):
        """Test compare_models with single alternative model"""
        manager = HyperparameterManager(
            dataset=mock_dataset,
            alternative_models={'model1': Mock()}
        )

        mock_run_tests.side_effect = [
            {'importance': 0.85},
            {'importance': 0.80},
        ]

        mock_factory.return_value = Mock()

        result = manager.compare_models()

        assert len(result['alternative_models']) == 1

    def test_manager_with_none_alternatives(self, mock_dataset):
        """Test manager with None alternative_models"""
        manager = HyperparameterManager(
            dataset=mock_dataset,
            alternative_models=None
        )

        # None is converted to empty dict by BaseManager
        assert manager.alternative_models == {}

    @patch('deepbridge.utils.hyperparameter.run_hyperparameter_tests')
    def test_run_tests_with_different_metrics(self, mock_run_tests, manager):
        """Test run_tests with various metrics"""
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']

        for metric in metrics:
            mock_run_tests.return_value = {}
            manager.run_tests(metric=metric)

            call_kwargs = mock_run_tests.call_args[1]
            assert call_kwargs['metric'] == metric

    @patch('deepbridge.utils.dataset_factory.DBDatasetFactory.create_for_alternative_model')
    @patch('deepbridge.utils.hyperparameter.run_hyperparameter_tests')
    def test_compare_models_with_both_config_and_metric(
        self, mock_run_tests, mock_factory, manager_with_alternatives
    ):
        """Test compare_models with both config and metric"""
        mock_run_tests.side_effect = [
            {'importance': 0.85},
            {'importance': 0.80},
            {'importance': 0.82},
        ]

        mock_factory.side_effect = [Mock(), Mock()]

        manager_with_alternatives.compare_models(
            config_name='full',
            metric='precision'
        )

        for call in mock_run_tests.call_args_list:
            assert call[1]['config_name'] == 'full'
            assert call[1]['metric'] == 'precision'
