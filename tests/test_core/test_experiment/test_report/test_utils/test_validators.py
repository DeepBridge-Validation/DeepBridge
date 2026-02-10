"""
Comprehensive tests for DataValidator.

This test suite validates:
1. validate_robustness_data - robustness data validation
2. validate_uncertainty_data - uncertainty data validation
3. validate_resilience_data - resilience data validation
4. validate_hyperparameter_data - hyperparameter data validation
5. Default value setting for missing fields
6. Edge cases and error handling

Coverage Target: ~95%+
"""

import pytest
from unittest.mock import patch

from deepbridge.core.experiment.report.utils.validators import DataValidator


# ==================== validate_robustness_data Tests ====================


class TestValidateRobustnessData:
    """Tests for validate_robustness_data method"""

    def test_validate_complete_data(self):
        """Test validation with complete data"""
        data = {
            'model_name': 'TestModel',
            'model_type': 'RandomForest',
            'metric': 'accuracy',
            'base_score': 0.85,
            'robustness_score': 0.80,
            'raw_impact': 0.05,
            'quantile_impact': 0.10,
            'feature_subset': ['feature1', 'feature2'],
            'feature_subset_display': 'feature1, feature2',
            'feature_importance': {'feature1': 0.6},
            'model_feature_importance': {'feature1': 0.7},
            'raw': {'data': 'value'},
            'quantile': {'data': 'value'}
        }

        result = DataValidator.validate_robustness_data(data)

        assert result['model_name'] == 'TestModel'
        assert result['model_type'] == 'RandomForest'
        assert result['metric'] == 'accuracy'
        assert result['base_score'] == 0.85

    def test_validate_missing_required_fields(self):
        """Test validation adds defaults for missing fields"""
        data = {}

        with patch('deepbridge.core.experiment.report.utils.validators.logger') as mock_logger:
            result = DataValidator.validate_robustness_data(data)

            assert result['model_name'] == 'Model'
            assert result['model_type'] == 'Unknown Model'
            assert result['metric'] == 'score'
            assert result['base_score'] == 0.0
            assert result['robustness_score'] == 0.0
            assert result['raw_impact'] == 0.0
            assert result['quantile_impact'] == 0.0
            assert result['feature_subset'] == []
            assert result['feature_subset_display'] == 'All Features'
            assert result['feature_importance'] == {}
            assert result['model_feature_importance'] == {}

            # Should have logged warnings
            assert mock_logger.warning.call_count > 0

    def test_validate_none_values(self):
        """Test validation replaces None values with defaults"""
        data = {
            'model_name': None,
            'base_score': None,
            'feature_subset': None
        }

        with patch('deepbridge.core.experiment.report.utils.validators.logger'):
            result = DataValidator.validate_robustness_data(data)

            assert result['model_name'] == 'Model'
            assert result['base_score'] == 0.0
            assert result['feature_subset'] == []

    def test_validate_adds_raw_dict(self):
        """Test that raw dict is added if missing"""
        data = {'model_name': 'Test'}

        with patch('deepbridge.core.experiment.report.utils.validators.logger'):
            result = DataValidator.validate_robustness_data(data)

            assert 'raw' in result
            assert isinstance(result['raw'], dict)

    def test_validate_adds_quantile_dict(self):
        """Test that quantile dict is added if missing"""
        data = {'model_name': 'Test'}

        with patch('deepbridge.core.experiment.report.utils.validators.logger'):
            result = DataValidator.validate_robustness_data(data)

            assert 'quantile' in result
            assert isinstance(result['quantile'], dict)

    def test_validate_converts_string_feature_subset_to_list(self):
        """Test converting feature_subset from string to list"""
        data = {'feature_subset': 'single_feature'}

        with patch('deepbridge.core.experiment.report.utils.validators.logger'):
            result = DataValidator.validate_robustness_data(data)

            assert result['feature_subset'] == ['single_feature']

    def test_validate_sets_feature_subset_display_from_list(self):
        """Test setting feature_subset_display from list"""
        data = {'feature_subset': ['feat1', 'feat2', 'feat3']}

        with patch('deepbridge.core.experiment.report.utils.validators.logger'):
            result = DataValidator.validate_robustness_data(data)

            assert result['feature_subset_display'] == 'feat1, feat2, feat3'

    def test_validate_sets_all_features_when_empty_subset(self):
        """Test feature_subset_display defaults to 'All Features'"""
        data = {'feature_subset': []}

        with patch('deepbridge.core.experiment.report.utils.validators.logger'):
            result = DataValidator.validate_robustness_data(data)

            assert result['feature_subset_display'] == 'All Features'

    def test_validate_preserves_original_data(self):
        """Test that original data is not modified"""
        original_data = {'model_name': 'Original'}

        with patch('deepbridge.core.experiment.report.utils.validators.logger'):
            result = DataValidator.validate_robustness_data(original_data)

        # Result should be a copy
        assert result is not original_data
        result['model_name'] = 'Modified'
        assert original_data['model_name'] == 'Original'


# ==================== validate_uncertainty_data Tests ====================


class TestValidateUncertaintyData:
    """Tests for validate_uncertainty_data method"""

    def test_validate_complete_data(self):
        """Test validation with complete data"""
        data = {
            'model_name': 'TestModel',
            'model_type': 'NeuralNet',
            'metric': 'mse',
            'method': 'crqr',
            'uncertainty_score': 0.75,
            'avg_coverage': 0.90,
            'avg_width': 0.25,
            'alpha_levels': [0.1, 0.2, 0.3],
            'metrics': {'score': 0.85}
        }

        result = DataValidator.validate_uncertainty_data(data)

        assert result['model_name'] == 'TestModel'
        assert result['model_type'] == 'NeuralNet'
        assert result['method'] == 'crqr'
        assert result['uncertainty_score'] == 0.75

    def test_validate_missing_required_fields(self):
        """Test validation adds defaults for missing fields"""
        data = {}

        with patch('deepbridge.core.experiment.report.utils.validators.logger') as mock_logger:
            result = DataValidator.validate_uncertainty_data(data)

            assert result['model_name'] == 'Model'
            assert result['model_type'] == 'Unknown Model'
            assert result['metric'] == 'score'
            assert result['method'] == 'crqr'
            assert result['uncertainty_score'] == 0.5
            assert result['avg_coverage'] == 0.0
            assert result['avg_width'] == 0.0
            assert result['alpha_levels'] == []

            # Should have logged warnings
            assert mock_logger.warning.call_count > 0

    def test_validate_none_values(self):
        """Test validation replaces None values with defaults"""
        data = {
            'model_name': None,
            'uncertainty_score': None,
            'alpha_levels': None
        }

        with patch('deepbridge.core.experiment.report.utils.validators.logger'):
            result = DataValidator.validate_uncertainty_data(data)

            assert result['model_name'] == 'Model'
            assert result['uncertainty_score'] == 0.5
            assert result['alpha_levels'] == []

    def test_validate_adds_metrics_dict(self):
        """Test that metrics dict is added if missing"""
        data = {'model_name': 'Test'}

        with patch('deepbridge.core.experiment.report.utils.validators.logger'):
            result = DataValidator.validate_uncertainty_data(data)

            assert 'metrics' in result
            assert isinstance(result['metrics'], dict)

    def test_validate_replaces_invalid_metrics(self):
        """Test that invalid metrics structure is replaced"""
        data = {'metrics': 'not a dict'}

        with patch('deepbridge.core.experiment.report.utils.validators.logger'):
            result = DataValidator.validate_uncertainty_data(data)

            assert result['metrics'] == {}

    def test_validate_preserves_original_data(self):
        """Test that original data is not modified"""
        original_data = {'model_name': 'Original'}

        with patch('deepbridge.core.experiment.report.utils.validators.logger'):
            result = DataValidator.validate_uncertainty_data(original_data)

        # Result should be a copy
        assert result is not original_data


# ==================== validate_resilience_data Tests ====================


class TestValidateResilienceData:
    """Tests for validate_resilience_data method"""

    def test_validate_complete_data(self):
        """Test validation with complete data"""
        data = {
            'model_name': 'TestModel',
            'model_type': 'SVM',
            'metric': 'f1',
            'resilience_score': 0.88,
            'avg_performance_gap': 0.12,
            'distribution_shift_results': [{'shift': 'temporal'}],
            'distance_metrics': ['PSI', 'KS'],
            'alphas': [0.1, 0.2],
            'metrics': {'accuracy': 0.90}
        }

        result = DataValidator.validate_resilience_data(data)

        assert result['model_name'] == 'TestModel'
        assert result['model_type'] == 'SVM'
        assert result['resilience_score'] == 0.88
        assert result['avg_performance_gap'] == 0.12

    def test_validate_missing_required_fields(self):
        """Test validation adds defaults for missing fields"""
        data = {}

        with patch('deepbridge.core.experiment.report.utils.validators.logger') as mock_logger:
            result = DataValidator.validate_resilience_data(data)

            assert result['model_name'] == 'Model'
            assert result['model_type'] == 'Unknown Model'
            assert result['metric'] == 'score'
            assert result['resilience_score'] == 0.0
            assert result['avg_performance_gap'] == 0.0
            assert result['distribution_shift_results'] == []
            assert result['distance_metrics'] == ['PSI', 'KS', 'WD1']
            assert result['alphas'] == [0.1, 0.2, 0.3]

            # Should have logged warnings
            assert mock_logger.warning.call_count > 0

    def test_validate_none_values(self):
        """Test validation replaces None values with defaults"""
        data = {
            'model_name': None,
            'resilience_score': None,
            'distribution_shift_results': None
        }

        with patch('deepbridge.core.experiment.report.utils.validators.logger'):
            result = DataValidator.validate_resilience_data(data)

            assert result['model_name'] == 'Model'
            assert result['resilience_score'] == 0.0
            assert result['distribution_shift_results'] == []

    def test_validate_adds_metrics_dict(self):
        """Test that metrics dict is added if missing"""
        data = {'model_name': 'Test'}

        with patch('deepbridge.core.experiment.report.utils.validators.logger'):
            result = DataValidator.validate_resilience_data(data)

            assert 'metrics' in result
            assert isinstance(result['metrics'], dict)

    def test_validate_preserves_original_data(self):
        """Test that original data is not modified"""
        original_data = {'model_name': 'Original'}

        with patch('deepbridge.core.experiment.report.utils.validators.logger'):
            result = DataValidator.validate_resilience_data(original_data)

        # Result should be a copy
        assert result is not original_data


# ==================== validate_hyperparameter_data Tests ====================


class TestValidateHyperparameterData:
    """Tests for validate_hyperparameter_data method"""

    def test_validate_complete_data(self):
        """Test validation with complete data"""
        data = {
            'model_name': 'TestModel',
            'model_type': 'XGBoost',
            'metric': 'rmse',
            'base_score': 0.92,
            'importance_scores': {'lr': 0.5, 'depth': 0.3},
            'tuning_order': ['lr', 'depth'],
            'importance_results': [{'param': 'lr'}],
            'optimization_results': [{'iteration': 1}],
            'metrics': {'score': 0.92}
        }

        result = DataValidator.validate_hyperparameter_data(data)

        assert result['model_name'] == 'TestModel'
        assert result['model_type'] == 'XGBoost'
        assert result['base_score'] == 0.92
        assert len(result['importance_scores']) == 2

    def test_validate_missing_required_fields(self):
        """Test validation adds defaults for missing fields"""
        data = {}

        with patch('deepbridge.core.experiment.report.utils.validators.logger') as mock_logger:
            result = DataValidator.validate_hyperparameter_data(data)

            assert result['model_name'] == 'Model'
            assert result['model_type'] == 'Unknown Model'
            assert result['metric'] == 'score'
            assert result['base_score'] == 0.0
            assert result['importance_scores'] == {}
            assert result['tuning_order'] == []
            assert result['importance_results'] == []
            assert result['optimization_results'] == []

            # Should have logged warnings
            assert mock_logger.warning.call_count > 0

    def test_validate_none_values(self):
        """Test validation replaces None values with defaults"""
        data = {
            'model_name': None,
            'base_score': None,
            'importance_scores': None
        }

        with patch('deepbridge.core.experiment.report.utils.validators.logger'):
            result = DataValidator.validate_hyperparameter_data(data)

            assert result['model_name'] == 'Model'
            assert result['base_score'] == 0.0
            assert result['importance_scores'] == {}

    def test_validate_adds_metrics_dict(self):
        """Test that metrics dict is added if missing"""
        data = {'model_name': 'Test'}

        with patch('deepbridge.core.experiment.report.utils.validators.logger'):
            result = DataValidator.validate_hyperparameter_data(data)

            assert 'metrics' in result
            assert isinstance(result['metrics'], dict)

    def test_validate_preserves_original_data(self):
        """Test that original data is not modified"""
        original_data = {'model_name': 'Original'}

        with patch('deepbridge.core.experiment.report.utils.validators.logger'):
            result = DataValidator.validate_hyperparameter_data(original_data)

        # Result should be a copy
        assert result is not original_data


# ==================== Integration Tests ====================


class TestIntegration:
    """Integration tests for complete workflows"""

    def test_validate_multiple_data_types(self):
        """Test validating multiple data types in sequence"""
        # Start with minimal data
        robustness_data = {'model_name': 'Model1'}
        uncertainty_data = {'model_name': 'Model2'}
        resilience_data = {'model_name': 'Model3'}
        hyperparameter_data = {'model_name': 'Model4'}

        with patch('deepbridge.core.experiment.report.utils.validators.logger'):
            result1 = DataValidator.validate_robustness_data(robustness_data)
            result2 = DataValidator.validate_uncertainty_data(uncertainty_data)
            result3 = DataValidator.validate_resilience_data(resilience_data)
            result4 = DataValidator.validate_hyperparameter_data(hyperparameter_data)

            # Each should have their type-specific defaults
            assert result1['robustness_score'] == 0.0
            assert result2['uncertainty_score'] == 0.5
            assert result3['resilience_score'] == 0.0
            assert result4['base_score'] == 0.0


# ==================== Edge Cases ====================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_validate_robustness_with_empty_string_features(self):
        """Test validation with empty string feature_subset"""
        data = {'feature_subset': ''}

        with patch('deepbridge.core.experiment.report.utils.validators.logger'):
            result = DataValidator.validate_robustness_data(data)

            assert result['feature_subset'] == ['']
            assert result['feature_subset_display'] == ''

    def test_validate_with_extra_fields(self):
        """Test that extra fields are preserved"""
        data = {
            'model_name': 'Test',
            'custom_field': 'custom_value',
            'extra_data': [1, 2, 3]
        }

        with patch('deepbridge.core.experiment.report.utils.validators.logger'):
            result = DataValidator.validate_uncertainty_data(data)

            assert result['custom_field'] == 'custom_value'
            assert result['extra_data'] == [1, 2, 3]

    def test_validate_with_numeric_strings(self):
        """Test validation with numeric values as strings"""
        data = {
            'base_score': '0.85',
            'robustness_score': '0.80'
        }

        with patch('deepbridge.core.experiment.report.utils.validators.logger'):
            result = DataValidator.validate_robustness_data(data)

            # Strings should be preserved (not converted)
            assert result['base_score'] == '0.85'
            assert result['robustness_score'] == '0.80'

    def test_validate_with_nested_structures(self):
        """Test validation preserves nested data structures"""
        data = {
            'model_name': 'Test',
            'nested': {
                'level1': {
                    'level2': {
                        'value': 42
                    }
                }
            }
        }

        with patch('deepbridge.core.experiment.report.utils.validators.logger'):
            result = DataValidator.validate_robustness_data(data)

            assert result['nested']['level1']['level2']['value'] == 42
