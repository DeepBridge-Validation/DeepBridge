"""
Comprehensive tests for model result classes.

This test suite validates:
1. BaseModelResult - core functionality and properties
2. ClassificationModelResult - classification-specific features
3. RegressionModelResult - regression-specific features
4. create_model_result - factory function
5. compare_with - model comparison logic

Coverage Target: ~95%+
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from deepbridge.core.experiment.model_result import (
    BaseModelResult,
    ClassificationModelResult,
    RegressionModelResult,
    create_model_result,
)


# ==================== Fixtures ====================


@pytest.fixture
def base_metrics():
    """Sample metrics for testing"""
    return {
        'accuracy': 0.95,
        'f1': 0.92,
        'precision': 0.93,
        'recall': 0.91
    }


@pytest.fixture
def base_hyperparameters():
    """Sample hyperparameters"""
    return {
        'max_depth': 5,
        'n_estimators': 100,
        'learning_rate': 0.01
    }


@pytest.fixture
def base_predictions():
    """Sample predictions"""
    return {
        'y_pred': [0, 1, 1, 0],
        'y_true': [0, 1, 0, 0],
        'probabilities': [0.1, 0.9, 0.8, 0.2]
    }


@pytest.fixture
def base_metadata():
    """Sample metadata"""
    return {
        'timestamp': '2024-01-01 12:00:00',
        'dataset_name': 'test_dataset',
        'n_samples': 1000
    }


@pytest.fixture
def base_model_result(base_metrics, base_hyperparameters):
    """Create BaseModelResult instance"""
    return BaseModelResult(
        model_name='TestModel',
        model_type='RandomForest',
        metrics=base_metrics,
        hyperparameters=base_hyperparameters
    )


# ==================== BaseModelResult Tests ====================


class TestBaseModelResultInitialization:
    """Tests for BaseModelResult initialization"""

    def test_init_with_all_parameters(self, base_metrics, base_hyperparameters, base_predictions, base_metadata):
        """Test initialization with all parameters"""
        result = BaseModelResult(
            model_name='TestModel',
            model_type='RandomForest',
            metrics=base_metrics,
            hyperparameters=base_hyperparameters,
            predictions=base_predictions,
            metadata=base_metadata
        )

        assert result.model_name == 'TestModel'
        assert result.model_type == 'RandomForest'
        assert result.metrics == base_metrics
        assert result.hyperparameters == base_hyperparameters
        assert result.predictions == base_predictions
        assert result.metadata == base_metadata

    def test_init_with_minimal_parameters(self, base_metrics):
        """Test initialization with minimal required parameters"""
        result = BaseModelResult(
            model_name='MinimalModel',
            model_type='LogisticRegression',
            metrics=base_metrics
        )

        assert result.model_name == 'MinimalModel'
        assert result.model_type == 'LogisticRegression'
        assert result.metrics == base_metrics
        assert result.hyperparameters == {}
        assert result.predictions == {}
        assert result.metadata == {}

    def test_init_with_none_metrics(self):
        """Test initialization with None metrics defaults to empty dict"""
        result = BaseModelResult(
            model_name='Test',
            model_type='Model',
            metrics=None
        )

        assert result.metrics == {}

    def test_init_with_none_optionals(self, base_metrics):
        """Test that None optionals default to empty dicts"""
        result = BaseModelResult(
            model_name='Test',
            model_type='Model',
            metrics=base_metrics,
            hyperparameters=None,
            predictions=None,
            metadata=None
        )

        assert result.hyperparameters == {}
        assert result.predictions == {}
        assert result.metadata == {}


class TestBaseModelResultProperties:
    """Tests for BaseModelResult properties"""

    def test_model_name_property(self, base_model_result):
        """Test model_name property"""
        assert base_model_result.model_name == 'TestModel'

    def test_model_type_property(self, base_model_result):
        """Test model_type property"""
        assert base_model_result.model_type == 'RandomForest'

    def test_metrics_property(self, base_model_result, base_metrics):
        """Test metrics property"""
        assert base_model_result.metrics == base_metrics

    def test_hyperparameters_property(self, base_model_result, base_hyperparameters):
        """Test hyperparameters property"""
        assert base_model_result.hyperparameters == base_hyperparameters

    def test_predictions_property(self, base_metrics):
        """Test predictions property"""
        result = BaseModelResult(
            model_name='Test',
            model_type='Model',
            metrics=base_metrics,
            predictions={'test': 'data'}
        )
        assert result.predictions == {'test': 'data'}

    def test_metadata_property(self, base_metrics):
        """Test metadata property"""
        result = BaseModelResult(
            model_name='Test',
            model_type='Model',
            metrics=base_metrics,
            metadata={'key': 'value'}
        )
        assert result.metadata == {'key': 'value'}


class TestGetMetric:
    """Tests for get_metric method"""

    def test_get_existing_metric(self, base_model_result):
        """Test getting an existing metric"""
        assert base_model_result.get_metric('accuracy') == 0.95

    def test_get_missing_metric_returns_none(self, base_model_result):
        """Test getting missing metric returns None"""
        assert base_model_result.get_metric('nonexistent') is None

    def test_get_missing_metric_with_default(self, base_model_result):
        """Test getting missing metric with custom default"""
        assert base_model_result.get_metric('nonexistent', default=0.0) == 0.0

    def test_get_metric_with_custom_default(self, base_model_result):
        """Test that default is only used when metric is missing"""
        assert base_model_result.get_metric('accuracy', default=0.0) == 0.95


class TestGetHyperparameter:
    """Tests for get_hyperparameter method"""

    def test_get_existing_hyperparameter(self, base_model_result):
        """Test getting an existing hyperparameter"""
        assert base_model_result.get_hyperparameter('max_depth') == 5

    def test_get_missing_hyperparameter_returns_none(self, base_model_result):
        """Test getting missing hyperparameter returns None"""
        assert base_model_result.get_hyperparameter('nonexistent') is None

    def test_get_missing_hyperparameter_with_default(self, base_model_result):
        """Test getting missing hyperparameter with custom default"""
        assert base_model_result.get_hyperparameter('nonexistent', default='default') == 'default'

    def test_get_hyperparameter_with_custom_default(self, base_model_result):
        """Test that default is only used when hyperparameter is missing"""
        assert base_model_result.get_hyperparameter('max_depth', default=0) == 5


class TestToDict:
    """Tests for to_dict method"""

    def test_to_dict_structure(self, base_model_result, base_metrics, base_hyperparameters):
        """Test to_dict returns correct structure"""
        result_dict = base_model_result.to_dict()

        assert 'name' in result_dict
        assert 'type' in result_dict
        assert 'metrics' in result_dict
        assert 'hyperparameters' in result_dict
        assert 'metadata' in result_dict

    def test_to_dict_values(self, base_model_result, base_metrics, base_hyperparameters):
        """Test to_dict returns correct values"""
        result_dict = base_model_result.to_dict()

        assert result_dict['name'] == 'TestModel'
        assert result_dict['type'] == 'RandomForest'
        assert result_dict['metrics'] == base_metrics
        assert result_dict['hyperparameters'] == base_hyperparameters
        assert result_dict['metadata'] == {}

    def test_to_dict_with_metadata(self, base_metrics, base_metadata):
        """Test to_dict includes metadata"""
        result = BaseModelResult(
            model_name='Test',
            model_type='Model',
            metrics=base_metrics,
            metadata=base_metadata
        )

        result_dict = result.to_dict()
        assert result_dict['metadata'] == base_metadata

    def test_to_dict_excludes_predictions(self, base_metrics, base_predictions):
        """Test to_dict does not include predictions"""
        result = BaseModelResult(
            model_name='Test',
            model_type='Model',
            metrics=base_metrics,
            predictions=base_predictions
        )

        result_dict = result.to_dict()
        assert 'predictions' not in result_dict


class TestCompareWith:
    """Tests for compare_with method"""

    def test_compare_with_all_common_metrics(self):
        """Test comparing two models with all common metrics"""
        model1 = BaseModelResult(
            model_name='Model1',
            model_type='RF',
            metrics={'accuracy': 0.9, 'f1': 0.85}
        )
        model2 = BaseModelResult(
            model_name='Model2',
            model_type='LR',
            metrics={'accuracy': 0.95, 'f1': 0.88}
        )

        comparison = model1.compare_with(model2)

        assert comparison['model1'] == 'Model1'
        assert comparison['model2'] == 'Model2'
        assert 'accuracy' in comparison['metrics_compared']
        assert 'f1' in comparison['metrics_compared']

    def test_compare_with_specific_metrics(self):
        """Test comparing with specific metrics list"""
        model1 = BaseModelResult(
            model_name='Model1',
            model_type='RF',
            metrics={'accuracy': 0.9, 'f1': 0.85, 'precision': 0.82}
        )
        model2 = BaseModelResult(
            model_name='Model2',
            model_type='LR',
            metrics={'accuracy': 0.95, 'f1': 0.88, 'precision': 0.90}
        )

        comparison = model1.compare_with(model2, metrics=['accuracy'])

        assert 'accuracy' in comparison['metrics_compared']
        assert 'f1' not in comparison['metrics_compared']
        assert 'precision' not in comparison['metrics_compared']

    def test_compare_with_calculates_difference(self):
        """Test that comparison calculates difference correctly"""
        model1 = BaseModelResult(
            model_name='Model1',
            model_type='RF',
            metrics={'accuracy': 0.9}
        )
        model2 = BaseModelResult(
            model_name='Model2',
            model_type='LR',
            metrics={'accuracy': 0.95}
        )

        comparison = model1.compare_with(model2)

        assert comparison['metrics_compared']['accuracy']['model1_value'] == 0.9
        assert comparison['metrics_compared']['accuracy']['model2_value'] == 0.95
        assert comparison['metrics_compared']['accuracy']['difference'] == pytest.approx(0.05)

    def test_compare_with_calculates_percent_change(self):
        """Test that comparison calculates percent change correctly"""
        model1 = BaseModelResult(
            model_name='Model1',
            model_type='RF',
            metrics={'accuracy': 0.8}
        )
        model2 = BaseModelResult(
            model_name='Model2',
            model_type='LR',
            metrics={'accuracy': 0.9}
        )

        comparison = model1.compare_with(model2)

        # (0.9 - 0.8) / 0.8 * 100 = 12.5%
        assert comparison['metrics_compared']['accuracy']['percent_change'] == pytest.approx(12.5)

    def test_compare_with_zero_base_value_positive_diff(self):
        """Test comparison when base value is 0 and diff is positive"""
        model1 = BaseModelResult(
            model_name='Model1',
            model_type='RF',
            metrics={'accuracy': 0.0}
        )
        model2 = BaseModelResult(
            model_name='Model2',
            model_type='LR',
            metrics={'accuracy': 0.5}
        )

        comparison = model1.compare_with(model2)

        assert comparison['metrics_compared']['accuracy']['percent_change'] == float('inf')

    def test_compare_with_zero_base_value_negative_diff(self):
        """Test comparison when base value is 0 and diff is negative"""
        model1 = BaseModelResult(
            model_name='Model1',
            model_type='RF',
            metrics={'accuracy': 0.0}
        )
        model2 = BaseModelResult(
            model_name='Model2',
            model_type='LR',
            metrics={'accuracy': -0.5}
        )

        comparison = model1.compare_with(model2)

        assert comparison['metrics_compared']['accuracy']['percent_change'] == float('-inf')

    def test_compare_with_zero_base_value_zero_diff(self):
        """Test comparison when base value is 0 and diff is 0"""
        model1 = BaseModelResult(
            model_name='Model1',
            model_type='RF',
            metrics={'accuracy': 0.0}
        )
        model2 = BaseModelResult(
            model_name='Model2',
            model_type='LR',
            metrics={'accuracy': 0.0}
        )

        comparison = model1.compare_with(model2)

        assert comparison['metrics_compared']['accuracy']['percent_change'] == 0

    def test_compare_with_none_values_skipped(self):
        """Test that metrics with None values are skipped"""
        model1 = BaseModelResult(
            model_name='Model1',
            model_type='RF',
            metrics={'accuracy': None, 'f1': 0.8}
        )
        model2 = BaseModelResult(
            model_name='Model2',
            model_type='LR',
            metrics={'accuracy': 0.9, 'f1': 0.85}
        )

        comparison = model1.compare_with(model2)

        # accuracy should be skipped (None in model1)
        assert 'accuracy' not in comparison['metrics_compared']
        # f1 should be included
        assert 'f1' in comparison['metrics_compared']

    def test_compare_with_no_common_metrics(self):
        """Test comparison with no common metrics"""
        model1 = BaseModelResult(
            model_name='Model1',
            model_type='RF',
            metrics={'accuracy': 0.9}
        )
        model2 = BaseModelResult(
            model_name='Model2',
            model_type='LR',
            metrics={'f1': 0.85}
        )

        comparison = model1.compare_with(model2)

        assert comparison['metrics_compared'] == {}


# ==================== ClassificationModelResult Tests ====================


class TestClassificationModelResultInitialization:
    """Tests for ClassificationModelResult initialization"""

    def test_init_with_all_parameters(self, base_metrics, base_hyperparameters):
        """Test initialization with all classification-specific parameters"""
        confusion_matrix = [[10, 2], [1, 15]]
        class_names = ['Class0', 'Class1']
        auc_curve = ([0.0, 0.1, 1.0], [0.0, 0.9, 1.0])

        result = ClassificationModelResult(
            model_name='ClassModel',
            model_type='RandomForest',
            metrics=base_metrics,
            hyperparameters=base_hyperparameters,
            confusion_matrix=confusion_matrix,
            class_names=class_names,
            auc_curve=auc_curve
        )

        assert result.model_name == 'ClassModel'
        assert result.confusion_matrix == confusion_matrix
        assert result.class_names == class_names
        assert result.auc_curve == auc_curve

    def test_init_with_minimal_parameters(self, base_metrics):
        """Test initialization with minimal parameters"""
        result = ClassificationModelResult(
            model_name='ClassModel',
            model_type='LogisticRegression',
            metrics=base_metrics
        )

        assert result.model_name == 'ClassModel'
        assert result.confusion_matrix is None
        assert result.class_names == []
        assert result.auc_curve is None

    def test_init_inherits_from_base(self, base_metrics, base_hyperparameters):
        """Test that ClassificationModelResult inherits BaseModelResult functionality"""
        result = ClassificationModelResult(
            model_name='ClassModel',
            model_type='RF',
            metrics=base_metrics,
            hyperparameters=base_hyperparameters
        )

        # Should have base properties
        assert result.model_name == 'ClassModel'
        assert result.metrics == base_metrics
        assert result.hyperparameters == base_hyperparameters


class TestClassificationModelResultProperties:
    """Tests for ClassificationModelResult properties"""

    def test_confusion_matrix_property(self, base_metrics):
        """Test confusion_matrix property"""
        cm = [[10, 2], [1, 15]]
        result = ClassificationModelResult(
            model_name='Test',
            model_type='RF',
            metrics=base_metrics,
            confusion_matrix=cm
        )

        assert result.confusion_matrix == cm

    def test_class_names_property(self, base_metrics):
        """Test class_names property"""
        names = ['A', 'B', 'C']
        result = ClassificationModelResult(
            model_name='Test',
            model_type='RF',
            metrics=base_metrics,
            class_names=names
        )

        assert result.class_names == names

    def test_class_names_default_empty_list(self, base_metrics):
        """Test class_names defaults to empty list"""
        result = ClassificationModelResult(
            model_name='Test',
            model_type='RF',
            metrics=base_metrics
        )

        assert result.class_names == []

    def test_auc_curve_property(self, base_metrics):
        """Test auc_curve property"""
        auc_curve = ([0.0, 0.5, 1.0], [0.0, 0.8, 1.0])
        result = ClassificationModelResult(
            model_name='Test',
            model_type='RF',
            metrics=base_metrics,
            auc_curve=auc_curve
        )

        assert result.auc_curve == auc_curve


# ==================== RegressionModelResult Tests ====================


class TestRegressionModelResultInitialization:
    """Tests for RegressionModelResult initialization"""

    def test_init_with_all_parameters(self, base_metrics, base_hyperparameters):
        """Test initialization with all regression-specific parameters"""
        residuals = [0.1, -0.2, 0.05, -0.15]
        feature_importances = {'feature1': 0.5, 'feature2': 0.3, 'feature3': 0.2}

        result = RegressionModelResult(
            model_name='RegModel',
            model_type='LinearRegression',
            metrics=base_metrics,
            hyperparameters=base_hyperparameters,
            residuals=residuals,
            feature_importances=feature_importances
        )

        assert result.model_name == 'RegModel'
        assert result.residuals == residuals
        assert result.feature_importances == feature_importances

    def test_init_with_minimal_parameters(self, base_metrics):
        """Test initialization with minimal parameters"""
        result = RegressionModelResult(
            model_name='RegModel',
            model_type='Ridge',
            metrics=base_metrics
        )

        assert result.model_name == 'RegModel'
        assert result.residuals is None
        assert result.feature_importances == {}

    def test_init_inherits_from_base(self, base_metrics, base_hyperparameters):
        """Test that RegressionModelResult inherits BaseModelResult functionality"""
        result = RegressionModelResult(
            model_name='RegModel',
            model_type='Ridge',
            metrics=base_metrics,
            hyperparameters=base_hyperparameters
        )

        # Should have base properties
        assert result.model_name == 'RegModel'
        assert result.metrics == base_metrics
        assert result.hyperparameters == base_hyperparameters


class TestRegressionModelResultProperties:
    """Tests for RegressionModelResult properties"""

    def test_residuals_property(self, base_metrics):
        """Test residuals property"""
        residuals = [0.1, -0.2, 0.3]
        result = RegressionModelResult(
            model_name='Test',
            model_type='Ridge',
            metrics=base_metrics,
            residuals=residuals
        )

        assert result.residuals == residuals

    def test_residuals_default_none(self, base_metrics):
        """Test residuals defaults to None"""
        result = RegressionModelResult(
            model_name='Test',
            model_type='Ridge',
            metrics=base_metrics
        )

        assert result.residuals is None

    def test_feature_importances_property(self, base_metrics):
        """Test feature_importances property"""
        importances = {'f1': 0.5, 'f2': 0.3}
        result = RegressionModelResult(
            model_name='Test',
            model_type='Ridge',
            metrics=base_metrics,
            feature_importances=importances
        )

        assert result.feature_importances == importances

    def test_feature_importances_default_empty_dict(self, base_metrics):
        """Test feature_importances defaults to empty dict"""
        result = RegressionModelResult(
            model_name='Test',
            model_type='Ridge',
            metrics=base_metrics
        )

        assert result.feature_importances == {}


# ==================== create_model_result Factory Tests ====================


class TestCreateModelResult:
    """Tests for create_model_result factory function"""

    def test_create_classification_result(self, base_metrics):
        """Test creating classification model result"""
        result = create_model_result(
            model_name='ClassModel',
            model_type='RF',
            metrics=base_metrics,
            problem_type='classification'
        )

        assert isinstance(result, ClassificationModelResult)
        assert result.model_name == 'ClassModel'

    def test_create_classification_with_extra_params(self, base_metrics):
        """Test creating classification result with classification-specific params"""
        cm = [[10, 2], [1, 15]]
        result = create_model_result(
            model_name='ClassModel',
            model_type='RF',
            metrics=base_metrics,
            problem_type='classification',
            confusion_matrix=cm
        )

        assert isinstance(result, ClassificationModelResult)
        assert result.confusion_matrix == cm

    def test_create_regression_result(self, base_metrics):
        """Test creating regression model result"""
        result = create_model_result(
            model_name='RegModel',
            model_type='Ridge',
            metrics=base_metrics,
            problem_type='regression'
        )

        assert isinstance(result, RegressionModelResult)
        assert result.model_name == 'RegModel'

    def test_create_regression_with_extra_params(self, base_metrics):
        """Test creating regression result with regression-specific params"""
        residuals = [0.1, -0.2]
        result = create_model_result(
            model_name='RegModel',
            model_type='Ridge',
            metrics=base_metrics,
            problem_type='regression',
            residuals=residuals
        )

        assert isinstance(result, RegressionModelResult)
        assert result.residuals == residuals

    def test_create_forecasting_result(self, base_metrics):
        """Test creating forecasting model result (maps to regression)"""
        result = create_model_result(
            model_name='ForecastModel',
            model_type='ARIMA',
            metrics=base_metrics,
            problem_type='forecasting'
        )

        assert isinstance(result, RegressionModelResult)
        assert result.model_name == 'ForecastModel'

    def test_create_unknown_type_returns_base(self, base_metrics):
        """Test creating with unknown problem_type returns BaseModelResult"""
        result = create_model_result(
            model_name='UnknownModel',
            model_type='CustomModel',
            metrics=base_metrics,
            problem_type='unknown'
        )

        assert isinstance(result, BaseModelResult)
        # Should not be a subclass
        assert type(result) == BaseModelResult

    def test_create_case_insensitive_problem_type(self, base_metrics):
        """Test that problem_type is case-insensitive"""
        result1 = create_model_result(
            model_name='Model',
            model_type='RF',
            metrics=base_metrics,
            problem_type='CLASSIFICATION'
        )
        result2 = create_model_result(
            model_name='Model',
            model_type='RF',
            metrics=base_metrics,
            problem_type='Classification'
        )

        assert isinstance(result1, ClassificationModelResult)
        assert isinstance(result2, ClassificationModelResult)

    def test_create_with_hyperparameters(self, base_metrics, base_hyperparameters):
        """Test creating result with hyperparameters"""
        result = create_model_result(
            model_name='Model',
            model_type='RF',
            metrics=base_metrics,
            problem_type='classification',
            hyperparameters=base_hyperparameters
        )

        assert result.hyperparameters == base_hyperparameters


# ==================== Integration Tests ====================


class TestIntegration:
    """Integration tests for complete workflows"""

    def test_full_classification_workflow(self):
        """Test complete workflow for classification model"""
        # Create classification result
        result = create_model_result(
            model_name='RandomForest',
            model_type='RandomForestClassifier',
            metrics={'accuracy': 0.95, 'f1': 0.93},
            problem_type='classification',
            hyperparameters={'n_estimators': 100},
            confusion_matrix=[[50, 5], [3, 42]],
            class_names=['Negative', 'Positive']
        )

        # Verify properties
        assert result.model_name == 'RandomForest'
        assert result.get_metric('accuracy') == 0.95
        assert result.get_hyperparameter('n_estimators') == 100
        assert result.confusion_matrix == [[50, 5], [3, 42]]
        assert result.class_names == ['Negative', 'Positive']

        # Convert to dict
        result_dict = result.to_dict()
        assert result_dict['name'] == 'RandomForest'
        assert result_dict['metrics']['accuracy'] == 0.95

    def test_full_regression_workflow(self):
        """Test complete workflow for regression model"""
        # Create regression result
        result = create_model_result(
            model_name='Ridge',
            model_type='RidgeRegression',
            metrics={'mse': 0.05, 'r2': 0.92},
            problem_type='regression',
            hyperparameters={'alpha': 1.0},
            residuals=[0.1, -0.2, 0.05],
            feature_importances={'f1': 0.6, 'f2': 0.4}
        )

        # Verify properties
        assert result.model_name == 'Ridge'
        assert result.get_metric('mse') == 0.05
        assert result.get_hyperparameter('alpha') == 1.0
        assert result.residuals == [0.1, -0.2, 0.05]
        assert result.feature_importances == {'f1': 0.6, 'f2': 0.4}

        # Convert to dict
        result_dict = result.to_dict()
        assert result_dict['type'] == 'RidgeRegression'

    def test_model_comparison_workflow(self):
        """Test comparing two models"""
        model1 = create_model_result(
            model_name='Model1',
            model_type='RF',
            metrics={'accuracy': 0.90, 'f1': 0.88},
            problem_type='classification'
        )

        model2 = create_model_result(
            model_name='Model2',
            model_type='LR',
            metrics={'accuracy': 0.92, 'f1': 0.90},
            problem_type='classification'
        )

        comparison = model1.compare_with(model2, metrics=['accuracy'])

        assert comparison['model1'] == 'Model1'
        assert comparison['model2'] == 'Model2'
        assert comparison['metrics_compared']['accuracy']['model1_value'] == 0.90
        assert comparison['metrics_compared']['accuracy']['model2_value'] == 0.92
        assert comparison['metrics_compared']['accuracy']['difference'] == pytest.approx(0.02)


# ==================== Edge Cases ====================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_empty_metrics_dict(self):
        """Test with empty metrics"""
        result = BaseModelResult(
            model_name='Test',
            model_type='Model',
            metrics={}
        )

        assert result.metrics == {}
        assert result.get_metric('any_metric') is None

    def test_very_long_model_name(self):
        """Test with very long model name"""
        long_name = 'A' * 1000
        result = BaseModelResult(
            model_name=long_name,
            model_type='Model',
            metrics={'accuracy': 0.9}
        )

        assert result.model_name == long_name

    def test_special_characters_in_names(self):
        """Test with special characters in names"""
        result = BaseModelResult(
            model_name='Model-v1.2 (final)',
            model_type='RF@2024',
            metrics={'accuracy': 0.9}
        )

        assert result.model_name == 'Model-v1.2 (final)'
        assert result.model_type == 'RF@2024'

    def test_negative_metrics(self):
        """Test with negative metric values"""
        result = BaseModelResult(
            model_name='Test',
            model_type='Model',
            metrics={'loss': -0.5, 'error': -1.2}
        )

        assert result.get_metric('loss') == -0.5

    def test_compare_with_itself(self):
        """Test comparing a model with itself"""
        result = BaseModelResult(
            model_name='SelfModel',
            model_type='RF',
            metrics={'accuracy': 0.9}
        )

        comparison = result.compare_with(result)

        assert comparison['metrics_compared']['accuracy']['difference'] == 0
        assert comparison['metrics_compared']['accuracy']['percent_change'] == 0

    def test_large_confusion_matrix(self):
        """Test with large confusion matrix (multiclass)"""
        cm = [[10, 2, 1], [1, 15, 3], [0, 2, 20]]
        result = ClassificationModelResult(
            model_name='MultiClass',
            model_type='RF',
            metrics={'accuracy': 0.9},
            confusion_matrix=cm
        )

        assert result.confusion_matrix == cm

    def test_many_feature_importances(self):
        """Test with many feature importances"""
        importances = {f'feature_{i}': 0.01 for i in range(100)}
        result = RegressionModelResult(
            model_name='Test',
            model_type='Ridge',
            metrics={'mse': 0.1},
            feature_importances=importances
        )

        assert len(result.feature_importances) == 100
