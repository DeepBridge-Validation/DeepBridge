"""
Tests for experiment interfaces and abstract base classes.

Coverage Target: 100%
"""

import pytest
from abc import ABC
from unittest.mock import Mock

from deepbridge.core.experiment.interfaces import (
    TestResult,
    ModelResult,
    ITestRunner,
    IExperiment,
    ParameterNames,
    TestType
)


# Concrete implementations for testing
class ConcreteTestResult(TestResult):
    """Concrete implementation of TestResult for testing"""

    def __init__(self, name_val='test', results_val=None, metadata_val=None):
        self._name = name_val
        self._results = results_val or {}
        self._metadata = metadata_val or {}

    @property
    def name(self):
        return self._name

    @property
    def results(self):
        return self._results

    @property
    def metadata(self):
        return self._metadata

    def to_dict(self):
        return {
            'name': self.name,
            'results': self.results,
            'metadata': self.metadata
        }


class ConcreteModelResult(ModelResult):
    """Concrete implementation of ModelResult for testing"""

    def __init__(self):
        self._model_name = 'test_model'
        self._model_type = 'classifier'
        self._metrics = {'accuracy': 0.95}
        self._hyperparameters = {'n_estimators': 100}
        self._metadata = {'version': '1.0'}

    @property
    def model_name(self):
        return self._model_name

    @property
    def model_type(self):
        return self._model_type

    @property
    def metrics(self):
        return self._metrics

    @property
    def hyperparameters(self):
        return self._hyperparameters

    @property
    def metadata(self):
        return self._metadata

    def to_dict(self):
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'metrics': self.metrics,
            'hyperparameters': self.hyperparameters,
            'metadata': self.metadata
        }

    def get_metric(self, metric_name, default=None):
        return self._metrics.get(metric_name, default)


class ConcreteTestRunner(ITestRunner):
    """Concrete implementation of ITestRunner for testing"""

    def run_tests(self, config_name='quick', **kwargs):
        return {'robustness': {'score': 0.9}}

    def run_test(self, test_type, config_name='quick', **kwargs):
        return ConcreteTestResult(name_val=test_type)

    def get_test_results(self, test_type=None):
        if test_type:
            return ConcreteTestResult(name_val=test_type)
        return {'all': 'results'}

    def get_test_config(self, test_type, config_name='quick'):
        return {'test_type': test_type, 'config_name': config_name}


class ConcreteExperiment(IExperiment):
    """Concrete implementation of IExperiment for testing"""

    def __init__(self):
        self._experiment_type = 'binary_classification'
        self._test_results = {}
        self._model = Mock()

    @property
    def experiment_type(self):
        return self._experiment_type

    @property
    def test_results(self):
        return self._test_results

    @property
    def model(self):
        return self._model

    def run_tests(self, config_name='quick', **kwargs):
        return {'robustness': {'score': 0.9}}

    def run_test(self, test_type, config_name='quick', **kwargs):
        return ConcreteTestResult(name_val=test_type)

    def fit(self, **kwargs):
        return self


class TestTestResultInterface:
    """Tests for TestResult interface"""

    def test_test_result_is_abc(self):
        """Test that TestResult is an abstract base class"""
        assert issubclass(TestResult, ABC)

    def test_test_result_cannot_be_instantiated(self):
        """Test that TestResult cannot be instantiated directly"""
        with pytest.raises(TypeError):
            TestResult()

    def test_concrete_test_result_can_be_instantiated(self):
        """Test that concrete implementation can be instantiated"""
        result = ConcreteTestResult()
        assert result is not None

    def test_concrete_test_result_name_property(self):
        """Test name property of concrete implementation"""
        result = ConcreteTestResult(name_val='robustness')
        assert result.name == 'robustness'

    def test_concrete_test_result_results_property(self):
        """Test results property of concrete implementation"""
        results_dict = {'score': 0.85}
        result = ConcreteTestResult(results_val=results_dict)
        assert result.results == results_dict

    def test_concrete_test_result_metadata_property(self):
        """Test metadata property of concrete implementation"""
        metadata_dict = {'timestamp': '2025-01-01'}
        result = ConcreteTestResult(metadata_val=metadata_dict)
        assert result.metadata == metadata_dict

    def test_concrete_test_result_to_dict(self):
        """Test to_dict method of concrete implementation"""
        result = ConcreteTestResult(
            name_val='test',
            results_val={'score': 0.9},
            metadata_val={'version': '1.0'}
        )
        result_dict = result.to_dict()

        assert result_dict['name'] == 'test'
        assert result_dict['results'] == {'score': 0.9}
        assert result_dict['metadata'] == {'version': '1.0'}


class TestModelResultInterface:
    """Tests for ModelResult interface"""

    def test_model_result_is_abc(self):
        """Test that ModelResult is an abstract base class"""
        assert issubclass(ModelResult, ABC)

    def test_model_result_cannot_be_instantiated(self):
        """Test that ModelResult cannot be instantiated directly"""
        with pytest.raises(TypeError):
            ModelResult()

    def test_concrete_model_result_can_be_instantiated(self):
        """Test that concrete implementation can be instantiated"""
        result = ConcreteModelResult()
        assert result is not None

    def test_concrete_model_result_model_name(self):
        """Test model_name property"""
        result = ConcreteModelResult()
        assert result.model_name == 'test_model'

    def test_concrete_model_result_model_type(self):
        """Test model_type property"""
        result = ConcreteModelResult()
        assert result.model_type == 'classifier'

    def test_concrete_model_result_metrics(self):
        """Test metrics property"""
        result = ConcreteModelResult()
        assert result.metrics == {'accuracy': 0.95}

    def test_concrete_model_result_hyperparameters(self):
        """Test hyperparameters property"""
        result = ConcreteModelResult()
        assert result.hyperparameters == {'n_estimators': 100}

    def test_concrete_model_result_metadata(self):
        """Test metadata property"""
        result = ConcreteModelResult()
        assert result.metadata == {'version': '1.0'}

    def test_concrete_model_result_to_dict(self):
        """Test to_dict method"""
        result = ConcreteModelResult()
        result_dict = result.to_dict()

        assert result_dict['model_name'] == 'test_model'
        assert result_dict['model_type'] == 'classifier'
        assert result_dict['metrics'] == {'accuracy': 0.95}
        assert result_dict['hyperparameters'] == {'n_estimators': 100}
        assert result_dict['metadata'] == {'version': '1.0'}

    def test_concrete_model_result_get_metric_existing(self):
        """Test get_metric for existing metric"""
        result = ConcreteModelResult()
        accuracy = result.get_metric('accuracy')

        assert accuracy == 0.95

    def test_concrete_model_result_get_metric_missing_with_default(self):
        """Test get_metric for missing metric with default"""
        result = ConcreteModelResult()
        value = result.get_metric('precision', default=0.0)

        assert value == 0.0

    def test_concrete_model_result_get_metric_missing_without_default(self):
        """Test get_metric for missing metric without default"""
        result = ConcreteModelResult()
        value = result.get_metric('precision')

        assert value is None


class TestITestRunnerInterface:
    """Tests for ITestRunner interface"""

    def test_itest_runner_is_abc(self):
        """Test that ITestRunner is an abstract base class"""
        assert issubclass(ITestRunner, ABC)

    def test_itest_runner_cannot_be_instantiated(self):
        """Test that ITestRunner cannot be instantiated directly"""
        with pytest.raises(TypeError):
            ITestRunner()

    def test_concrete_test_runner_can_be_instantiated(self):
        """Test that concrete implementation can be instantiated"""
        runner = ConcreteTestRunner()
        assert runner is not None

    def test_concrete_test_runner_run_tests(self):
        """Test run_tests method"""
        runner = ConcreteTestRunner()
        results = runner.run_tests(config_name='quick')

        assert isinstance(results, dict)
        assert 'robustness' in results

    def test_concrete_test_runner_run_test(self):
        """Test run_test method"""
        runner = ConcreteTestRunner()
        result = runner.run_test('robustness', config_name='quick')

        # Check it's a TestResult subclass instance
        assert hasattr(result, 'name')
        assert hasattr(result, 'results')
        assert result.name == 'robustness'

    def test_concrete_test_runner_get_test_results_with_type(self):
        """Test get_test_results with specific test type"""
        runner = ConcreteTestRunner()
        result = runner.get_test_results('robustness')

        # Check it's a TestResult subclass instance
        assert hasattr(result, 'name')
        assert hasattr(result, 'results')

    def test_concrete_test_runner_get_test_results_without_type(self):
        """Test get_test_results for all results"""
        runner = ConcreteTestRunner()
        results = runner.get_test_results(None)

        assert isinstance(results, dict)

    def test_concrete_test_runner_get_test_config(self):
        """Test get_test_config method"""
        runner = ConcreteTestRunner()
        config = runner.get_test_config('robustness', config_name='medium')

        assert config['test_type'] == 'robustness'
        assert config['config_name'] == 'medium'


class TestIExperimentInterface:
    """Tests for IExperiment interface"""

    def test_iexperiment_is_abc(self):
        """Test that IExperiment is an abstract base class"""
        assert issubclass(IExperiment, ABC)

    def test_iexperiment_cannot_be_instantiated(self):
        """Test that IExperiment cannot be instantiated directly"""
        with pytest.raises(TypeError):
            IExperiment()

    def test_concrete_experiment_can_be_instantiated(self):
        """Test that concrete implementation can be instantiated"""
        experiment = ConcreteExperiment()
        assert experiment is not None

    def test_concrete_experiment_experiment_type(self):
        """Test experiment_type property"""
        experiment = ConcreteExperiment()
        assert experiment.experiment_type == 'binary_classification'

    def test_concrete_experiment_test_results(self):
        """Test test_results property"""
        experiment = ConcreteExperiment()
        assert isinstance(experiment.test_results, dict)

    def test_concrete_experiment_model(self):
        """Test model property"""
        experiment = ConcreteExperiment()
        assert experiment.model is not None

    def test_concrete_experiment_run_tests(self):
        """Test run_tests method"""
        experiment = ConcreteExperiment()
        results = experiment.run_tests(config_name='quick')

        assert isinstance(results, dict)
        assert 'robustness' in results

    def test_concrete_experiment_run_test(self):
        """Test run_test method"""
        experiment = ConcreteExperiment()
        result = experiment.run_test('robustness')

        # Check it's a TestResult subclass instance
        assert hasattr(result, 'name')
        assert hasattr(result, 'results')

    def test_concrete_experiment_fit(self):
        """Test fit method returns self for chaining"""
        experiment = ConcreteExperiment()
        result = experiment.fit()

        assert result is experiment


class TestParameterNames:
    """Tests for ParameterNames placeholder"""

    def test_parameter_names_has_dataset(self):
        """Test DATASET constant"""
        assert hasattr(ParameterNames, 'DATASET')
        assert ParameterNames.DATASET == 'dataset'

    def test_parameter_names_has_feature_subset(self):
        """Test FEATURE_SUBSET constant"""
        assert hasattr(ParameterNames, 'FEATURE_SUBSET')
        assert ParameterNames.FEATURE_SUBSET == 'feature_subset'

    def test_parameter_names_has_config_name(self):
        """Test CONFIG_NAME constant"""
        assert hasattr(ParameterNames, 'CONFIG_NAME')
        assert ParameterNames.CONFIG_NAME == 'config_name'

    def test_parameter_names_has_verbose(self):
        """Test VERBOSE constant"""
        assert hasattr(ParameterNames, 'VERBOSE')
        assert ParameterNames.VERBOSE == 'verbose'

    def test_parameter_names_has_metric(self):
        """Test METRIC constant"""
        assert hasattr(ParameterNames, 'METRIC')
        assert ParameterNames.METRIC == 'metric'


class TestTestTypeEnum:
    """Tests for TestType enum/placeholder"""

    def test_test_type_has_robustness(self):
        """Test ROBUSTNESS constant"""
        assert hasattr(TestType, 'ROBUSTNESS')
        # May be enum or string depending on whether parameter_standards is available
        assert TestType.ROBUSTNESS is not None

    def test_test_type_has_uncertainty(self):
        """Test UNCERTAINTY constant"""
        assert hasattr(TestType, 'UNCERTAINTY')
        assert TestType.UNCERTAINTY is not None

    def test_test_type_has_resilience(self):
        """Test RESILIENCE constant"""
        assert hasattr(TestType, 'RESILIENCE')
        assert TestType.RESILIENCE is not None

    def test_test_type_has_hyperparameters(self):
        """Test HYPERPARAMETERS constant"""
        assert hasattr(TestType, 'HYPERPARAMETERS')
        assert TestType.HYPERPARAMETERS is not None
