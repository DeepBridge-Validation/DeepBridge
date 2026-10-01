"""
Tests for parameter_standards module.

Coverage Target: Cover missing lines in enum __str__ methods and helper functions
"""

import pytest

from deepbridge.core.experiment.parameter_standards import (
    TestType,
    ValidationTestType,
    ConfigName,
    ExperimentType,
    standardize_feature_names,
    get_test_types,
    get_config_names,
    get_experiment_types,
    is_valid_test_type,
    is_valid_config_name,
    is_valid_experiment_type,
)


class TestEnumStrMethods:
    """Tests for enum __str__ methods"""

    def test_test_type_str(self):
        """Test ValidationTestType __str__ method"""
        assert str(ValidationTestType.ROBUSTNESS) == 'robustness'
        assert str(ValidationTestType.UNCERTAINTY) == 'uncertainty'
        assert str(ValidationTestType.RESILIENCE) == 'resilience'
        assert str(ValidationTestType.HYPERPARAMETERS) == 'hyperparameters'

    def test_test_type_legacy_alias(self):
        """The deprecated TestType alias must stay bound to the same enum.

        ``TestType`` is the pre-2.0 public name of ``ValidationTestType`` and
        is kept for the whole 2.x line, so user code importing it keeps
        working. Importing the alias here also guards the ``__test__ = False``
        marker on the class: without it, this very import makes pytest emit a
        PytestCollectionWarning.
        """
        assert TestType is ValidationTestType
        assert TestType.ROBUSTNESS is ValidationTestType.ROBUSTNESS

    def test_config_name_str(self):
        """Test ConfigName __str__ method"""
        assert str(ConfigName.QUICK) == 'quick'
        assert str(ConfigName.MEDIUM) == 'medium'
        assert str(ConfigName.FULL) == 'full'

    def test_experiment_type_str(self):
        """Test ExperimentType __str__ method"""
        assert str(ExperimentType.BINARY_CLASSIFICATION) == 'binary_classification'
        assert str(ExperimentType.MULTICLASS_CLASSIFICATION) == 'multiclass_classification'
        assert str(ExperimentType.REGRESSION) == 'regression'
        assert str(ExperimentType.FORECASTING) == 'forecasting'


class TestStandardizeFunctions:
    """Tests for standardization functions"""

    def test_standardize_feature_names_basic(self):
        """Test basic feature name standardization"""
        features = ['Feature One', 'Feature Two', 'Feature Three']
        result = standardize_feature_names(features)

        assert result == ['feature_one', 'feature_two', 'feature_three']

    def test_standardize_feature_names_with_spaces(self):
        """Test standardization with multiple spaces"""
        features = ['My Feature Name', 'Another  Feature']
        result = standardize_feature_names(features)

        assert 'my_feature_name' in result
        assert 'another__feature' in result  # Double space becomes double underscore

    def test_standardize_feature_names_empty_list(self):
        """Test standardization with empty list"""
        result = standardize_feature_names([])
        assert result == []


class TestGetterFunctions:
    """Tests for getter functions"""

    def test_get_test_types(self):
        """Test getting all test types"""
        test_types = get_test_types()

        assert 'robustness' in test_types
        assert 'uncertainty' in test_types
        assert 'resilience' in test_types
        assert 'hyperparameters' in test_types
        assert len(test_types) == 4

    def test_get_config_names(self):
        """Test getting all config names"""
        config_names = get_config_names()

        assert 'quick' in config_names
        assert 'medium' in config_names
        assert 'full' in config_names
        assert len(config_names) == 3

    def test_get_experiment_types(self):
        """Test getting all experiment types"""
        exp_types = get_experiment_types()

        assert 'binary_classification' in exp_types
        assert 'multiclass_classification' in exp_types
        assert 'regression' in exp_types
        assert 'forecasting' in exp_types
        assert len(exp_types) == 4


class TestValidationFunctions:
    """Tests for validation functions"""

    def test_is_valid_test_type_valid(self):
        """Test validation with valid test types"""
        assert is_valid_test_type('robustness') is True
        assert is_valid_test_type('uncertainty') is True
        assert is_valid_test_type('resilience') is True
        assert is_valid_test_type('hyperparameters') is True

    def test_is_valid_test_type_invalid(self):
        """Test validation with invalid test type"""
        assert is_valid_test_type('invalid_test') is False
        assert is_valid_test_type('') is False

    def test_is_valid_config_name_valid(self):
        """Test validation with valid config names"""
        assert is_valid_config_name('quick') is True
        assert is_valid_config_name('medium') is True
        assert is_valid_config_name('full') is True

    def test_is_valid_config_name_invalid(self):
        """Test validation with invalid config name"""
        assert is_valid_config_name('invalid_config') is False
        assert is_valid_config_name('') is False

    def test_is_valid_experiment_type_valid(self):
        """Test validation with valid experiment types"""
        assert is_valid_experiment_type('binary_classification') is True
        assert is_valid_experiment_type('multiclass_classification') is True
        assert is_valid_experiment_type('regression') is True
        assert is_valid_experiment_type('forecasting') is True

    def test_is_valid_experiment_type_invalid(self):
        """Test validation with invalid experiment type"""
        assert is_valid_experiment_type('invalid_experiment') is False
        assert is_valid_experiment_type('') is False
