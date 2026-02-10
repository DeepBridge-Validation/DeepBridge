"""
Tests for parameter_standards module.

Tests enums and standard parameter names.
"""

import pytest
from deepbridge.core.experiment.parameter_standards import (
    ParameterNames,
    TestType,
    ConfigName,
    ExperimentType
)


class TestParameterNames:
    """Test ParameterNames constants."""

    def test_dataset_parameters(self):
        """Test dataset-related parameter names."""
        assert ParameterNames.DATASET == 'dataset'
        assert ParameterNames.FEATURES == 'features'
        assert ParameterNames.FEATURE_SUBSET == 'feature_subset'
        assert ParameterNames.TARGET == 'target'

    def test_train_test_parameters(self):
        """Test train/test parameter names."""
        assert ParameterNames.X_TRAIN == 'X_train'
        assert ParameterNames.X_TEST == 'X_test'
        assert ParameterNames.Y_TRAIN == 'y_train'
        assert ParameterNames.Y_TEST == 'y_test'

    def test_configuration_parameters(self):
        """Test configuration parameter names."""
        assert ParameterNames.CONFIG_NAME == 'config_name'
        assert ParameterNames.EXPERIMENT_TYPE == 'experiment_type'
        assert ParameterNames.TESTS == 'tests'
        assert ParameterNames.VERBOSE == 'verbose'

    def test_test_specific_parameters(self):
        """Test test-specific parameter names."""
        assert ParameterNames.TEST_TYPE == 'test_type'
        assert ParameterNames.METRIC == 'metric'
        assert ParameterNames.N_TRIALS == 'n_trials'
        assert ParameterNames.N_ITERATIONS == 'n_iterations'

    def test_model_parameters(self):
        """Test model parameter names."""
        assert ParameterNames.MODEL == 'model'
        assert ParameterNames.MODEL_TYPE == 'model_type'
        assert ParameterNames.HYPERPARAMETERS == 'hyperparameters'

    def test_splitting_parameters(self):
        """Test splitting parameter names."""
        assert ParameterNames.TEST_SIZE == 'test_size'
        assert ParameterNames.RANDOM_STATE == 'random_state'

    def test_performance_metric_parameters(self):
        """Test performance metric parameter names."""
        assert ParameterNames.ACCURACY == 'accuracy'
        assert ParameterNames.AUC == 'auc'
        assert ParameterNames.F1 == 'f1'
        assert ParameterNames.PRECISION == 'precision'
        assert ParameterNames.RECALL == 'recall'


class TestTestTypeEnum:
    """Test TestType enum."""

    def test_test_type_values(self):
        """Test that TestType has correct values."""
        assert TestType.ROBUSTNESS.value == 'robustness'
        assert TestType.UNCERTAINTY.value == 'uncertainty'
        assert TestType.RESILIENCE.value == 'resilience'
        assert TestType.HYPERPARAMETERS.value == 'hyperparameters'

    def test_test_type_str(self):
        """Test TestType string representation."""
        assert str(TestType.ROBUSTNESS) == 'robustness'
        assert str(TestType.UNCERTAINTY) == 'uncertainty'
        assert str(TestType.RESILIENCE) == 'resilience'
        assert str(TestType.HYPERPARAMETERS) == 'hyperparameters'

    def test_test_type_iteration(self):
        """Test that we can iterate over TestType values."""
        test_types = list(TestType)
        assert len(test_types) == 4
        assert TestType.ROBUSTNESS in test_types

    def test_test_type_comparison(self):
        """Test TestType comparison."""
        assert TestType.ROBUSTNESS == TestType.ROBUSTNESS
        assert TestType.ROBUSTNESS != TestType.UNCERTAINTY


class TestConfigNameEnum:
    """Test ConfigName enum."""

    def test_config_name_values(self):
        """Test that ConfigName has correct values."""
        assert ConfigName.QUICK.value == 'quick'
        assert ConfigName.MEDIUM.value == 'medium'
        assert ConfigName.FULL.value == 'full'

    def test_config_name_str(self):
        """Test ConfigName string representation."""
        assert str(ConfigName.QUICK) == 'quick'
        assert str(ConfigName.MEDIUM) == 'medium'
        assert str(ConfigName.FULL) == 'full'

    def test_config_name_iteration(self):
        """Test that we can iterate over ConfigName values."""
        config_names = list(ConfigName)
        assert len(config_names) == 3
        assert ConfigName.QUICK in config_names

    def test_config_name_comparison(self):
        """Test ConfigName comparison."""
        assert ConfigName.QUICK == ConfigName.QUICK
        assert ConfigName.QUICK != ConfigName.MEDIUM


class TestExperimentTypeEnum:
    """Test ExperimentType enum."""

    def test_experiment_type_values(self):
        """Test that ExperimentType has correct values."""
        assert ExperimentType.BINARY_CLASSIFICATION.value == 'binary_classification'
        assert ExperimentType.MULTICLASS_CLASSIFICATION.value == 'multiclass_classification'
        assert ExperimentType.REGRESSION.value == 'regression'
        assert ExperimentType.FORECASTING.value == 'forecasting'

    def test_experiment_type_str(self):
        """Test ExperimentType string representation."""
        assert str(ExperimentType.BINARY_CLASSIFICATION) == 'binary_classification'
        assert str(ExperimentType.MULTICLASS_CLASSIFICATION) == 'multiclass_classification'
        assert str(ExperimentType.REGRESSION) == 'regression'
        assert str(ExperimentType.FORECASTING) == 'forecasting'

    def test_experiment_type_iteration(self):
        """Test that we can iterate over ExperimentType values."""
        experiment_types = list(ExperimentType)
        assert len(experiment_types) == 4
        assert ExperimentType.BINARY_CLASSIFICATION in experiment_types

    def test_experiment_type_comparison(self):
        """Test ExperimentType comparison."""
        assert ExperimentType.BINARY_CLASSIFICATION == ExperimentType.BINARY_CLASSIFICATION
        assert ExperimentType.BINARY_CLASSIFICATION != ExperimentType.REGRESSION


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


def test_get_test_config_invalid_test_type():
    """Test get_test_config with invalid test type."""
    from deepbridge.core.experiment.parameter_standards import get_test_config
    
    with pytest.raises(ValueError, match="Invalid test type"):
        get_test_config(test_type="invalid_type", config_name="quick")


def test_get_test_config_invalid_config_name():
    """Test get_test_config with invalid config name."""
    from deepbridge.core.experiment.parameter_standards import get_test_config
    
    with pytest.raises(ValueError, match="Invalid configuration name"):
        get_test_config(test_type="robustness", config_name="invalid_config")


def test_get_test_config_test_type_not_in_configs():
    """Test get_test_config with test type that has no config."""
    from deepbridge.core.experiment.parameter_standards import get_test_config
    from unittest.mock import patch
    
    # Mock TEST_CONFIGS to not include a valid test type
    with patch('deepbridge.core.experiment.parameter_standards.TEST_CONFIGS', {}):
        with pytest.raises(ValueError, match="No configuration template defined"):
            get_test_config(test_type="robustness", config_name="quick")


def test_get_test_config_config_name_not_in_test_config():
    """Test get_test_config with config name not in test config."""
    from deepbridge.core.experiment.parameter_standards import get_test_config
    from unittest.mock import patch
    
    # Mock TEST_CONFIGS with robustness but without 'quick' config
    mock_configs = {
        "robustness": {
            "medium": {"n_trials": 10}
        }
    }
    
    with patch('deepbridge.core.experiment.parameter_standards.TEST_CONFIGS', mock_configs):
        with pytest.raises(ValueError, match="No quick configuration defined"):
            get_test_config(test_type="robustness", config_name="quick")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
