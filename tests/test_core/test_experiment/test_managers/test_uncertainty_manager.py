"""
Comprehensive tests for UncertaintyManager.

This test suite validates:
1. run_tests - running uncertainty tests on primary model
2. compare_models - comparing uncertainty across models
3. Configuration validation
4. Alternative model handling
5. Edge cases and error handling

Coverage Target: ~95%+
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from deepbridge.core.experiment.managers.uncertainty_manager import UncertaintyManager


# ==================== Fixtures ====================


@pytest.fixture
def mock_dataset():
    """Create mock dataset"""
    dataset = Mock()
    dataset.model = Mock()
    return dataset


@pytest.fixture
def manager(mock_dataset):
    """Create UncertaintyManager instance"""
    return UncertaintyManager(dataset=mock_dataset)


@pytest.fixture
def verbose_manager(mock_dataset):
    """Create UncertaintyManager with verbose=True"""
    return UncertaintyManager(dataset=mock_dataset, verbose=True)


@pytest.fixture
def manager_with_alternatives(mock_dataset):
    """Create manager with alternative models"""
    alt_models = {
        'model1': Mock(),
        'model2': Mock(),
    }
    return UncertaintyManager(
        dataset=mock_dataset,
        alternative_models=alt_models,
        verbose=False
    )


# ==================== run_tests Tests ====================


class TestRunTests:
    """Tests for run_tests method"""

    @patch('deepbridge.utils.uncertainty.run_uncertainty_tests')
    def test_run_tests_with_default_config(self, mock_run_tests, manager, mock_dataset):
        """Test run_tests with default config"""
        mock_run_tests.return_value = {'metric': 0.85}

        result = manager.run_tests()

        assert result == {'metric': 0.85}
        mock_run_tests.assert_called_once_with(
            mock_dataset,
            config_name='quick',
            verbose=False,
        )

    @patch('deepbridge.utils.uncertainty.run_uncertainty_tests')
    def test_run_tests_with_quick_config(self, mock_run_tests, manager, mock_dataset):
        """Test run_tests with 'quick' config"""
        mock_run_tests.return_value = {'metric': 0.85}

        result = manager.run_tests(config_name='quick')

        mock_run_tests.assert_called_once_with(
            mock_dataset,
            config_name='quick',
            verbose=False,
        )

    @patch('deepbridge.utils.uncertainty.run_uncertainty_tests')
    def test_run_tests_with_medium_config(self, mock_run_tests, manager, mock_dataset):
        """Test run_tests with 'medium' config"""
        mock_run_tests.return_value = {'metric': 0.85}

        result = manager.run_tests(config_name='medium')

        mock_run_tests.assert_called_once_with(
            mock_dataset,
            config_name='medium',
            verbose=False,
        )

    @patch('deepbridge.utils.uncertainty.run_uncertainty_tests')
    def test_run_tests_with_full_config(self, mock_run_tests, manager, mock_dataset):
        """Test run_tests with 'full' config"""
        mock_run_tests.return_value = {'metric': 0.85}

        result = manager.run_tests(config_name='full')

        mock_run_tests.assert_called_once_with(
            mock_dataset,
            config_name='full',
            verbose=False,
        )

    @patch('deepbridge.utils.uncertainty.run_uncertainty_tests')
    def test_run_tests_with_invalid_config_falls_back_to_quick(
        self, mock_run_tests, manager, mock_dataset
    ):
        """Test run_tests with invalid config falls back to 'quick'"""
        mock_run_tests.return_value = {'metric': 0.85}

        result = manager.run_tests(config_name='invalid_config')

        # Should fall back to 'quick'
        mock_run_tests.assert_called_once_with(
            mock_dataset,
            config_name='quick',
            verbose=False,
        )

    @patch('deepbridge.utils.uncertainty.run_uncertainty_tests')
    def test_run_tests_with_kwargs(self, mock_run_tests, manager, mock_dataset):
        """Test run_tests passes additional kwargs"""
        mock_run_tests.return_value = {'metric': 0.85}

        manager.run_tests(config_name='quick', custom_param='value', another=123)

        mock_run_tests.assert_called_once_with(
            mock_dataset,
            config_name='quick',
            verbose=False,
            custom_param='value',
            another=123,
        )

    @patch('deepbridge.utils.uncertainty.run_uncertainty_tests')
    def test_run_tests_with_verbose_true(
        self, mock_run_tests, verbose_manager, mock_dataset
    ):
        """Test run_tests respects verbose flag"""
        mock_run_tests.return_value = {'metric': 0.85}

        verbose_manager.run_tests()

        call_kwargs = mock_run_tests.call_args[1]
        assert call_kwargs['verbose'] is True


# ==================== compare_models Tests ====================


class TestCompareModels:
    """Tests for compare_models method"""

    @patch('deepbridge.utils.uncertainty.run_uncertainty_tests')
    def test_compare_models_with_no_alternatives(
        self, mock_run_tests, manager, mock_dataset
    ):
        """Test compare_models with no alternative models"""
        mock_run_tests.return_value = {'metric': 0.85}

        result = manager.compare_models()

        assert 'primary_model' in result
        assert 'alternative_models' in result
        assert result['primary_model'] == {'metric': 0.85}
        assert result['alternative_models'] == {}
        mock_run_tests.assert_called_once()

    @patch('deepbridge.utils.dataset_factory.DBDatasetFactory.create_for_alternative_model')
    @patch('deepbridge.utils.uncertainty.run_uncertainty_tests')
    def test_compare_models_with_alternatives(
        self, mock_run_tests, mock_factory, manager_with_alternatives, mock_dataset
    ):
        """Test compare_models with alternative models"""
        # Setup return values
        mock_run_tests.side_effect = [
            {'metric': 0.85},  # primary
            {'metric': 0.80},  # model1
            {'metric': 0.82},  # model2
        ]

        alt_dataset1 = Mock()
        alt_dataset2 = Mock()
        mock_factory.side_effect = [alt_dataset1, alt_dataset2]

        result = manager_with_alternatives.compare_models()

        # Check structure
        assert 'primary_model' in result
        assert 'alternative_models' in result
        assert result['primary_model'] == {'metric': 0.85}
        assert len(result['alternative_models']) == 2
        assert result['alternative_models']['model1'] == {'metric': 0.80}
        assert result['alternative_models']['model2'] == {'metric': 0.82}

        # Verify calls
        assert mock_run_tests.call_count == 3

    @patch('deepbridge.utils.dataset_factory.DBDatasetFactory.create_for_alternative_model')
    @patch('deepbridge.utils.uncertainty.run_uncertainty_tests')
    def test_compare_models_with_config_name(
        self, mock_run_tests, mock_factory, manager_with_alternatives, mock_dataset
    ):
        """Test compare_models passes config_name correctly"""
        mock_run_tests.side_effect = [
            {'metric': 0.85},
            {'metric': 0.80},
            {'metric': 0.82},
        ]

        mock_factory.side_effect = [Mock(), Mock()]

        manager_with_alternatives.compare_models(config_name='full')

        # Check all calls used 'full' config
        for call in mock_run_tests.call_args_list:
            assert call[1]['config_name'] == 'full'

    @patch('deepbridge.utils.dataset_factory.DBDatasetFactory.create_for_alternative_model')
    @patch('deepbridge.utils.uncertainty.run_uncertainty_tests')
    def test_compare_models_with_invalid_config_falls_back(
        self, mock_run_tests, mock_factory, manager_with_alternatives
    ):
        """Test compare_models with invalid config falls back to 'quick'"""
        mock_run_tests.side_effect = [
            {'metric': 0.85},
            {'metric': 0.80},
            {'metric': 0.82},
        ]

        mock_factory.side_effect = [Mock(), Mock()]

        manager_with_alternatives.compare_models(config_name='invalid')

        # Should fall back to 'quick'
        for call in mock_run_tests.call_args_list:
            assert call[1]['config_name'] == 'quick'

    @patch('deepbridge.utils.dataset_factory.DBDatasetFactory.create_for_alternative_model')
    @patch('deepbridge.utils.uncertainty.run_uncertainty_tests')
    def test_compare_models_creates_alternative_datasets(
        self, mock_run_tests, mock_factory, manager_with_alternatives, mock_dataset
    ):
        """Test compare_models creates datasets for alternative models"""
        mock_run_tests.side_effect = [
            {'metric': 0.85},
            {'metric': 0.80},
            {'metric': 0.82},
        ]

        alt_dataset1 = Mock()
        alt_dataset2 = Mock()
        mock_factory.side_effect = [alt_dataset1, alt_dataset2]

        manager_with_alternatives.compare_models()

        # Verify factory was called for each alternative
        assert mock_factory.call_count == 2

    @patch('deepbridge.utils.dataset_factory.DBDatasetFactory.create_for_alternative_model')
    @patch('deepbridge.utils.uncertainty.run_uncertainty_tests')
    def test_compare_models_with_kwargs(
        self, mock_run_tests, mock_factory, manager_with_alternatives
    ):
        """Test compare_models passes kwargs to all tests"""
        mock_run_tests.side_effect = [
            {'metric': 0.85},
            {'metric': 0.80},
            {'metric': 0.82},
        ]

        mock_factory.side_effect = [Mock(), Mock()]

        manager_with_alternatives.compare_models(
            config_name='quick',
            custom_param='value'
        )

        # Check kwargs passed to all calls
        for call in mock_run_tests.call_args_list:
            assert call[1]['custom_param'] == 'value'


# ==================== Integration Tests ====================


class TestIntegration:
    """Integration tests for complete workflows"""

    @patch('deepbridge.utils.uncertainty.run_uncertainty_tests')
    def test_full_workflow_run_then_compare(
        self, mock_run_tests, manager_with_alternatives
    ):
        """Test complete workflow: run_tests then compare_models"""
        mock_run_tests.side_effect = [
            {'metric': 0.85},  # run_tests
            {'metric': 0.85},  # compare primary
            {'metric': 0.80},  # compare model1
            {'metric': 0.82},  # compare model2
        ]

        with patch('deepbridge.utils.dataset_factory.DBDatasetFactory.create_for_alternative_model') as mock_factory:
            mock_factory.side_effect = [Mock(), Mock()]

            # First run tests
            result1 = manager_with_alternatives.run_tests()
            assert result1 == {'metric': 0.85}

            # Then compare
            result2 = manager_with_alternatives.compare_models()
            assert 'primary_model' in result2
            assert 'alternative_models' in result2


# ==================== Edge Cases ====================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    @patch('deepbridge.utils.uncertainty.run_uncertainty_tests')
    def test_run_tests_with_empty_kwargs(self, mock_run_tests, manager, mock_dataset):
        """Test run_tests with explicitly empty kwargs"""
        mock_run_tests.return_value = {}

        result = manager.run_tests(**{})

        assert result == {}

    @patch('deepbridge.utils.uncertainty.run_uncertainty_tests')
    def test_run_tests_returns_empty_dict(self, mock_run_tests, manager):
        """Test run_tests when underlying function returns empty dict"""
        mock_run_tests.return_value = {}

        result = manager.run_tests()

        assert result == {}

    @patch('deepbridge.utils.dataset_factory.DBDatasetFactory.create_for_alternative_model')
    @patch('deepbridge.utils.uncertainty.run_uncertainty_tests')
    def test_compare_models_with_single_alternative(
        self, mock_run_tests, mock_factory, mock_dataset
    ):
        """Test compare_models with single alternative model"""
        manager = UncertaintyManager(
            dataset=mock_dataset,
            alternative_models={'model1': Mock()}
        )

        mock_run_tests.side_effect = [
            {'metric': 0.85},
            {'metric': 0.80},
        ]

        mock_factory.return_value = Mock()

        result = manager.compare_models()

        assert len(result['alternative_models']) == 1
        assert 'model1' in result['alternative_models']

    def test_manager_with_none_alternatives(self, mock_dataset):
        """Test manager with None alternative_models"""
        manager = UncertaintyManager(
            dataset=mock_dataset,
            alternative_models=None
        )

        # None is converted to empty dict by BaseManager
        assert manager.alternative_models == {}

    @patch('deepbridge.utils.uncertainty.run_uncertainty_tests')
    def test_run_tests_with_none_config_name(self, mock_run_tests, manager):
        """Test run_tests with None config_name uses default"""
        mock_run_tests.return_value = {}

        # Passing None should use default 'quick'
        manager.run_tests(config_name=None)

        # Since None is not valid, should fall back to 'quick'
        call_kwargs = mock_run_tests.call_args[1]
        assert call_kwargs['config_name'] == 'quick'
