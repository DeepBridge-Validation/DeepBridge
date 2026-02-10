"""
Comprehensive tests for ExperimentRunner.

This test suite validates:
1. __init__ - initialization with dataset and config
2. run_experiments - running multiple experiments
3. _run_single_experiment - single experiment execution
4. get_trained_model - getting trained model
5. save_results - saving results to CSV
6. Edge cases and error handling

Coverage Target: ~90%+
"""

import pytest
import pandas as pd
import os
import tempfile
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

from deepbridge.distillation.experiment_runner import ExperimentRunner
from deepbridge.config.settings import DistillationConfig
from deepbridge.core.db_data import DBDataset
from deepbridge.utils.model_registry import ModelType


# ==================== Fixtures ====================


@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame for testing"""
    return pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [2.0, 3.0, 4.0, 5.0, 6.0],
        'target': [0, 1, 0, 1, 0],
        'prob_0': [0.8, 0.2, 0.7, 0.3, 0.9],
        'prob_1': [0.2, 0.8, 0.3, 0.7, 0.1],
    })


@pytest.fixture
def mock_dataset(sample_dataframe):
    """Create mock DBDataset"""
    dataset = Mock(spec=DBDataset)
    dataset.df = sample_dataframe
    dataset.features = sample_dataframe[['feature1', 'feature2']]
    dataset.target = sample_dataframe['target']
    dataset.probabilities = sample_dataframe[['prob_0', 'prob_1']]
    return dataset


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_config(temp_output_dir):
    """Create mock DistillationConfig"""
    config = Mock(spec=DistillationConfig)
    config.output_dir = temp_output_dir
    config.test_size = 0.2
    config.random_state = 42
    config.n_trials = 1
    config.validation_split = 0.2
    config.verbose = False
    config.model_types = [ModelType.LOGISTIC_REGRESSION]
    config.temperatures = [1.0]
    config.alphas = [0.5]
    config.log_info = Mock()
    config.get_total_configurations = Mock(return_value=1)
    return config


@pytest.fixture
def mock_experiment():
    """Create mock Experiment"""
    experiment = Mock()
    experiment.fit = Mock(return_value=experiment)
    experiment._results_data = {
        'train': {
            'accuracy': 0.85,
            'precision': 0.80,
            'recall': 0.75,
            'f1_score': 0.77,
            'auc_roc': 0.90,
            'auc_pr': 0.88,
            'kl_divergence': 0.1,
            'ks_statistic': 0.3,
            'ks_pvalue': 0.01,
            'r2_score': 0.7,
            'best_params': {'param1': 'value1'},
        },
        'test': {
            'accuracy': 0.80,
            'precision': 0.75,
            'recall': 0.70,
            'f1_score': 0.72,
            'auc_roc': 0.85,
            'auc_pr': 0.83,
            'kl_divergence': 0.15,
            'ks_statistic': 0.35,
            'ks_pvalue': 0.02,
            'r2_score': 0.65,
            'best_params': {'param1': 'value1'},
        }
    }
    experiment.distillation_model = Mock()
    return experiment


# ==================== Initialization Tests ====================


class TestInitialization:
    """Tests for __init__ method"""

    @patch('deepbridge.distillation.experiment_runner.Experiment')
    @patch('deepbridge.distillation.experiment_runner.Classification')
    def test_init_creates_output_directory(self, mock_classification, mock_experiment_class, mock_dataset, temp_output_dir):
        """Test initialization creates output directory"""
        config = Mock(spec=DistillationConfig)
        new_dir = os.path.join(temp_output_dir, 'new_output')
        config.output_dir = new_dir
        config.test_size = 0.2
        config.random_state = 42

        runner = ExperimentRunner(mock_dataset, config)

        assert os.path.exists(new_dir)

    @patch('deepbridge.distillation.experiment_runner.Experiment')
    @patch('deepbridge.distillation.experiment_runner.Classification')
    def test_init_stores_dataset_and_config(self, mock_classification, mock_experiment_class, mock_dataset, mock_config):
        """Test initialization stores dataset and config"""
        runner = ExperimentRunner(mock_dataset, mock_config)

        assert runner.dataset is mock_dataset
        assert runner.config is mock_config

    @patch('deepbridge.distillation.experiment_runner.Experiment')
    @patch('deepbridge.distillation.experiment_runner.Classification')
    def test_init_creates_experiment(self, mock_classification, mock_experiment_class, mock_dataset, mock_config):
        """Test initialization creates Experiment instance"""
        runner = ExperimentRunner(mock_dataset, mock_config)

        mock_experiment_class.assert_called_once()
        assert hasattr(runner, 'experiment')

    @patch('deepbridge.distillation.experiment_runner.Experiment')
    @patch('deepbridge.distillation.experiment_runner.Classification')
    def test_init_creates_metrics_calculator(self, mock_classification, mock_experiment_class, mock_dataset, mock_config):
        """Test initialization creates metrics calculator"""
        runner = ExperimentRunner(mock_dataset, mock_config)

        mock_classification.assert_called_once()
        assert hasattr(runner, 'metrics_calculator')

    @patch('deepbridge.distillation.experiment_runner.Experiment')
    @patch('deepbridge.distillation.experiment_runner.Classification')
    def test_init_initializes_empty_results(self, mock_classification, mock_experiment_class, mock_dataset, mock_config):
        """Test initialization creates empty results list"""
        runner = ExperimentRunner(mock_dataset, mock_config)

        assert runner.results == []


# ==================== run_experiments Tests ====================


class TestRunExperiments:
    """Tests for run_experiments method"""

    @patch('deepbridge.distillation.experiment_runner.Experiment')
    @patch('deepbridge.distillation.experiment_runner.Classification')
    def test_run_experiments_returns_dataframe(self, mock_classification, mock_experiment_class, mock_dataset, mock_config, mock_experiment):
        """Test run_experiments returns DataFrame"""
        mock_experiment_class.return_value = mock_experiment
        runner = ExperimentRunner(mock_dataset, mock_config)

        result = runner.run_experiments()

        assert isinstance(result, pd.DataFrame)

    @patch('deepbridge.distillation.experiment_runner.Experiment')
    @patch('deepbridge.distillation.experiment_runner.Classification')
    def test_run_experiments_logs_info(self, mock_classification, mock_experiment_class, mock_dataset, mock_config, mock_experiment):
        """Test run_experiments logs information"""
        mock_experiment_class.return_value = mock_experiment
        runner = ExperimentRunner(mock_dataset, mock_config)

        runner.run_experiments()

        # Should have called log_info multiple times
        assert mock_config.log_info.call_count >= 3

    @patch('deepbridge.distillation.experiment_runner.Experiment')
    @patch('deepbridge.distillation.experiment_runner.Classification')
    def test_run_experiments_with_use_probabilities_true(self, mock_classification, mock_experiment_class, mock_dataset, mock_config, mock_experiment):
        """Test run_experiments with use_probabilities=True"""
        mock_experiment_class.return_value = mock_experiment
        runner = ExperimentRunner(mock_dataset, mock_config)

        result = runner.run_experiments(use_probabilities=True)

        assert len(result) > 0

    @patch('deepbridge.distillation.experiment_runner.Experiment')
    @patch('deepbridge.distillation.experiment_runner.Classification')
    def test_run_experiments_with_use_probabilities_false(self, mock_classification, mock_experiment_class, mock_dataset, mock_config, mock_experiment):
        """Test run_experiments with use_probabilities=False"""
        mock_experiment_class.return_value = mock_experiment
        runner = ExperimentRunner(mock_dataset, mock_config)

        result = runner.run_experiments(use_probabilities=False)

        assert len(result) > 0

    @patch('deepbridge.distillation.experiment_runner.Experiment')
    @patch('deepbridge.distillation.experiment_runner.Classification')
    def test_run_experiments_with_different_distillation_method(self, mock_classification, mock_experiment_class, mock_dataset, mock_config, mock_experiment):
        """Test run_experiments with different distillation method"""
        mock_experiment_class.return_value = mock_experiment
        runner = ExperimentRunner(mock_dataset, mock_config)

        result = runner.run_experiments(distillation_method='surrogate')

        assert len(result) > 0
        assert result.iloc[0]['distillation_method'] == 'surrogate'

    @patch('deepbridge.distillation.experiment_runner.Experiment')
    @patch('deepbridge.distillation.experiment_runner.Classification')
    def test_run_experiments_iterates_all_configurations(self, mock_classification, mock_experiment_class, mock_dataset, mock_config, mock_experiment):
        """Test run_experiments iterates through all configurations"""
        mock_config.model_types = [ModelType.LOGISTIC_REGRESSION, ModelType.RANDOM_FOREST]
        mock_config.temperatures = [1.0, 2.0]
        mock_config.alphas = [0.3, 0.5]
        mock_config.get_total_configurations.return_value = 8
        mock_experiment_class.return_value = mock_experiment

        runner = ExperimentRunner(mock_dataset, mock_config)
        result = runner.run_experiments()

        # Should have 2 * 2 * 2 = 8 experiments
        assert len(result) == 8

    @patch('deepbridge.distillation.experiment_runner.Experiment')
    @patch('deepbridge.distillation.experiment_runner.Classification')
    @patch('deepbridge.distillation.experiment_runner.time')
    def test_run_experiments_measures_time(self, mock_time, mock_classification, mock_experiment_class, mock_dataset, mock_config, mock_experiment):
        """Test run_experiments measures execution time"""
        mock_time.time.side_effect = [100, 150]  # Start and end times
        mock_experiment_class.return_value = mock_experiment

        runner = ExperimentRunner(mock_dataset, mock_config)
        runner.run_experiments()

        # Should log time information
        log_calls = [call[0][0] for call in mock_config.log_info.call_args_list]
        assert any('seconds' in str(call) for call in log_calls)


# ==================== _run_single_experiment Tests ====================


class TestRunSingleExperiment:
    """Tests for _run_single_experiment method"""

    @patch('deepbridge.distillation.experiment_runner.Experiment')
    @patch('deepbridge.distillation.experiment_runner.Classification')
    def test_run_single_experiment_returns_dict(self, mock_classification, mock_experiment_class, mock_dataset, mock_config, mock_experiment):
        """Test _run_single_experiment returns dictionary"""
        mock_experiment_class.return_value = mock_experiment
        runner = ExperimentRunner(mock_dataset, mock_config)

        result = runner._run_single_experiment(
            model_type=ModelType.LOGISTIC_REGRESSION,
            temperature=1.0,
            alpha=0.5,
            use_probabilities=True
        )

        assert isinstance(result, dict)

    @patch('deepbridge.distillation.experiment_runner.Experiment')
    @patch('deepbridge.distillation.experiment_runner.Classification')
    def test_run_single_experiment_includes_all_metrics(self, mock_classification, mock_experiment_class, mock_dataset, mock_config, mock_experiment):
        """Test _run_single_experiment includes all metrics"""
        mock_experiment_class.return_value = mock_experiment
        runner = ExperimentRunner(mock_dataset, mock_config)

        result = runner._run_single_experiment(
            model_type=ModelType.LOGISTIC_REGRESSION,
            temperature=1.0,
            alpha=0.5,
            use_probabilities=True
        )

        # Check for key metrics
        assert 'model_type' in result
        assert 'temperature' in result
        assert 'alpha' in result
        assert 'test_accuracy' in result
        assert 'train_accuracy' in result
        assert 'test_f1_score' in result

    @patch('deepbridge.distillation.experiment_runner.Experiment')
    @patch('deepbridge.distillation.experiment_runner.Classification')
    def test_run_single_experiment_handles_exception(self, mock_classification, mock_experiment_class, mock_dataset, mock_config):
        """Test _run_single_experiment handles exceptions"""
        mock_experiment = Mock()
        mock_experiment.fit.side_effect = Exception("Test error")
        mock_experiment_class.return_value = mock_experiment

        runner = ExperimentRunner(mock_dataset, mock_config)

        result = runner._run_single_experiment(
            model_type=ModelType.LOGISTIC_REGRESSION,
            temperature=1.0,
            alpha=0.5,
            use_probabilities=True
        )

        assert 'error' in result
        assert result['error'] == 'Test error'

    @patch('deepbridge.distillation.experiment_runner.Experiment')
    @patch('deepbridge.distillation.experiment_runner.Classification')
    def test_run_single_experiment_calls_fit_with_correct_params(self, mock_classification, mock_experiment_class, mock_dataset, mock_config, mock_experiment):
        """Test _run_single_experiment calls fit with correct parameters"""
        mock_experiment_class.return_value = mock_experiment
        runner = ExperimentRunner(mock_dataset, mock_config)

        runner._run_single_experiment(
            model_type=ModelType.RANDOM_FOREST,
            temperature=2.0,
            alpha=0.7,
            use_probabilities=False,
            distillation_method='surrogate'
        )

        mock_experiment.fit.assert_called_once()
        call_kwargs = mock_experiment.fit.call_args[1]
        assert call_kwargs['student_model_type'] == ModelType.RANDOM_FOREST
        assert call_kwargs['temperature'] == 2.0
        assert call_kwargs['alpha'] == 0.7
        assert call_kwargs['use_probabilities'] is False
        assert call_kwargs['distillation_method'] == 'surrogate'


# ==================== get_trained_model Tests ====================


class TestGetTrainedModel:
    """Tests for get_trained_model method"""

    @patch('deepbridge.distillation.experiment_runner.Experiment')
    @patch('deepbridge.distillation.experiment_runner.Classification')
    def test_get_trained_model_returns_model(self, mock_classification, mock_experiment_class, mock_dataset, mock_config, mock_experiment):
        """Test get_trained_model returns trained model"""
        mock_experiment_class.return_value = mock_experiment
        runner = ExperimentRunner(mock_dataset, mock_config)

        model = runner.get_trained_model(
            model_type=ModelType.LOGISTIC_REGRESSION,
            temperature=1.0,
            alpha=0.5
        )

        assert model is not None
        assert model == mock_experiment.distillation_model

    @patch('deepbridge.distillation.experiment_runner.Experiment')
    @patch('deepbridge.distillation.experiment_runner.Classification')
    def test_get_trained_model_calls_fit(self, mock_classification, mock_experiment_class, mock_dataset, mock_config, mock_experiment):
        """Test get_trained_model calls experiment.fit"""
        mock_experiment_class.return_value = mock_experiment
        runner = ExperimentRunner(mock_dataset, mock_config)

        runner.get_trained_model(
            model_type=ModelType.LOGISTIC_REGRESSION,
            temperature=1.0,
            alpha=0.5
        )

        mock_experiment.fit.assert_called_once()


# ==================== save_results Tests ====================


class TestSaveResults:
    """Tests for save_results method"""

    @patch('deepbridge.distillation.experiment_runner.Experiment')
    @patch('deepbridge.distillation.experiment_runner.Classification')
    def test_save_results_creates_csv_file(self, mock_classification, mock_experiment_class, mock_dataset, mock_config, mock_experiment):
        """Test save_results creates CSV file"""
        mock_experiment_class.return_value = mock_experiment
        runner = ExperimentRunner(mock_dataset, mock_config)
        runner.run_experiments()

        runner.save_results()

        results_path = os.path.join(mock_config.output_dir, 'distillation_results.csv')
        assert os.path.exists(results_path)

    @patch('deepbridge.distillation.experiment_runner.Experiment')
    @patch('deepbridge.distillation.experiment_runner.Classification')
    def test_save_results_raises_error_without_results(self, mock_classification, mock_experiment_class, mock_dataset, mock_config):
        """Test save_results raises error when no results available"""
        mock_experiment_class.return_value = Mock()
        runner = ExperimentRunner(mock_dataset, mock_config)

        with pytest.raises(ValueError, match='No results available'):
            runner.save_results()

    @patch('deepbridge.distillation.experiment_runner.Experiment')
    @patch('deepbridge.distillation.experiment_runner.Classification')
    def test_save_results_logs_info(self, mock_classification, mock_experiment_class, mock_dataset, mock_config, mock_experiment):
        """Test save_results logs save information"""
        mock_experiment_class.return_value = mock_experiment
        runner = ExperimentRunner(mock_dataset, mock_config)
        runner.run_experiments()

        runner.save_results()

        # Should log save information
        log_calls = [call[0][0] for call in mock_config.log_info.call_args_list]
        assert any('saved' in str(call).lower() for call in log_calls)


# ==================== Integration Tests ====================


class TestIntegration:
    """Integration tests for complete workflows"""

    @patch('deepbridge.distillation.experiment_runner.Experiment')
    @patch('deepbridge.distillation.experiment_runner.Classification')
    def test_full_workflow_run_and_save(self, mock_classification, mock_experiment_class, mock_dataset, mock_config, mock_experiment):
        """Test complete workflow: initialize, run, and save"""
        mock_experiment_class.return_value = mock_experiment
        runner = ExperimentRunner(mock_dataset, mock_config)

        # Run experiments
        results = runner.run_experiments()
        assert len(results) > 0

        # Save results
        runner.save_results()
        assert os.path.exists(os.path.join(mock_config.output_dir, 'distillation_results.csv'))


# ==================== Edge Cases ====================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    @patch('deepbridge.distillation.experiment_runner.Experiment')
    @patch('deepbridge.distillation.experiment_runner.Classification')
    def test_run_experiments_with_empty_model_types(self, mock_classification, mock_experiment_class, mock_dataset, mock_config, mock_experiment):
        """Test run_experiments with empty model_types list"""
        mock_config.model_types = []
        mock_experiment_class.return_value = mock_experiment

        runner = ExperimentRunner(mock_dataset, mock_config)
        result = runner.run_experiments()

        assert len(result) == 0

    @patch('deepbridge.distillation.experiment_runner.Experiment')
    @patch('deepbridge.distillation.experiment_runner.Classification')
    def test_run_single_experiment_with_missing_metrics(self, mock_classification, mock_experiment_class, mock_dataset, mock_config):
        """Test _run_single_experiment with missing metrics in results"""
        mock_experiment = Mock()
        mock_experiment.fit.return_value = mock_experiment
        mock_experiment._results_data = {
            'train': {},  # Empty metrics
            'test': {}
        }
        mock_experiment_class.return_value = mock_experiment

        runner = ExperimentRunner(mock_dataset, mock_config)
        result = runner._run_single_experiment(
            model_type=ModelType.LOGISTIC_REGRESSION,
            temperature=1.0,
            alpha=0.5,
            use_probabilities=True
        )

        # Should return None for missing metrics
        assert result['test_accuracy'] is None
        assert result['train_accuracy'] is None
