"""
Comprehensive tests for InitialResultsTransformer.

This test suite validates:
1. transform - main transformation logic
2. _extract_config - configuration extraction
3. _extract_models - models data extraction
4. _normalize_metrics - metrics normalization

Coverage Target: ~90%+
"""

import pytest
from unittest.mock import patch

from deepbridge.core.experiment.report.transformers.initial_results import InitialResultsTransformer


# ==================== Fixtures ====================


@pytest.fixture
def transformer():
    """Create InitialResultsTransformer instance"""
    return InitialResultsTransformer()


@pytest.fixture
def sample_data():
    """Sample initial results data"""
    return {
        'config': {
            'dataset_info': {
                'n_samples': 1000,
                'n_features': 20,
                'test_size': 0.2
            },
            'tests': ['robustness', 'fairness'],
            'verbose': True
        },
        'models': {
            'model1': {
                'name': 'RandomForest',
                'type': 'RandomForestClassifier',
                'metrics': {
                    'accuracy': 0.95,
                    'roc_auc': 0.98,
                    'f1': 0.94
                }
            }
        },
        'test_configs': {
            'robustness': {'iterations': 10}
        }
    }


# ==================== transform Tests ====================


class TestTransform:
    """Tests for transform method"""

    def test_transform_valid_data(self, transformer, sample_data):
        """Test transforming valid data"""
        result = transformer.transform(sample_data)

        assert 'config' in result
        assert 'models' in result
        assert 'test_configs' in result
        assert result['test_configs'] == sample_data['test_configs']

    def test_transform_empty_data(self, transformer):
        """Test transforming empty/None data"""
        result = transformer.transform(None)

        # Should return minimal valid structure
        assert 'config' in result
        assert 'models' in result
        assert result['models']['primary_model']['name'] == 'Primary Model'

    def test_transform_empty_dict(self, transformer):
        """Test transforming empty dict"""
        result = transformer.transform({})

        # Should return minimal valid structure
        assert 'config' in result
        assert 'models' in result

    def test_transform_logs_info(self, transformer, sample_data):
        """Test that transform logs info messages"""
        with patch('deepbridge.core.experiment.report.transformers.initial_results.logger') as mock_logger:
            transformer.transform(sample_data)

            # Should log transformation
            assert mock_logger.info.call_count >= 1

    def test_transform_logs_warning_on_empty(self, transformer):
        """Test that transform logs warning on empty data"""
        with patch('deepbridge.core.experiment.report.transformers.initial_results.logger') as mock_logger:
            transformer.transform(None)

            mock_logger.warning.assert_called_once()

    def test_transform_handles_exception(self, transformer):
        """Test transform handles exceptions gracefully"""
        # Create data that will cause exception
        bad_data = {'config': {'dataset_info': None}}

        with patch.object(transformer, '_extract_config', side_effect=Exception("Test error")):
            result = transformer.transform(bad_data)

            # Should return minimal structure
            assert result == {'config': {}, 'models': {}, 'test_configs': {}}

    def test_transform_multiple_models(self, transformer):
        """Test transforming data with multiple models"""
        data = {
            'config': {},
            'models': {
                'model1': {'name': 'Model 1', 'metrics': {'accuracy': 0.9}},
                'model2': {'name': 'Model 2', 'metrics': {'accuracy': 0.8}}
            }
        }

        result = transformer.transform(data)

        assert len(result['models']) == 2
        assert 'model1' in result['models']
        assert 'model2' in result['models']


# ==================== _extract_config Tests ====================


class TestExtractConfig:
    """Tests for _extract_config method"""

    def test_extract_config_full_data(self, transformer):
        """Test extracting complete config data"""
        config_data = {
            'dataset_info': {'n_samples': 1000},
            'tests': ['robustness'],
            'verbose': True
        }

        result = transformer._extract_config(config_data)

        assert result['dataset_info'] == {'n_samples': 1000}
        assert result['tests'] == ['robustness']
        assert result['verbose'] is True

    def test_extract_config_empty(self, transformer):
        """Test extracting from empty config"""
        result = transformer._extract_config({})

        assert result == {}

    def test_extract_config_none(self, transformer):
        """Test extracting from None"""
        result = transformer._extract_config(None)

        assert result == {}

    def test_extract_config_partial(self, transformer):
        """Test extracting partial config data"""
        config_data = {'tests': ['fairness']}

        result = transformer._extract_config(config_data)

        assert result['tests'] == ['fairness']
        assert result['verbose'] is False
        assert result['dataset_info'] == {}

    def test_extract_config_defaults(self, transformer):
        """Test that defaults are applied"""
        config_data = {}

        result = transformer._extract_config(config_data)

        # Empty dict is returned, not defaults
        assert result == {}


# ==================== _extract_models Tests ====================


class TestExtractModels:
    """Tests for _extract_models method"""

    def test_extract_models_single(self, transformer):
        """Test extracting single model"""
        models_data = {
            'model1': {
                'name': 'Test Model',
                'type': 'RandomForest',
                'metrics': {'accuracy': 0.9}
            }
        }

        result = transformer._extract_models(models_data)

        assert len(result) == 1
        assert result['model1']['name'] == 'Test Model'
        assert result['model1']['type'] == 'RandomForest'

    def test_extract_models_empty(self, transformer):
        """Test extracting from empty models dict"""
        result = transformer._extract_models({})

        assert result == {}

    def test_extract_models_none(self, transformer):
        """Test extracting from None"""
        result = transformer._extract_models(None)

        assert result == {}

    def test_extract_models_skips_none_data(self, transformer):
        """Test that None model data is skipped"""
        models_data = {
            'model1': {'name': 'Valid'},
            'model2': None,
            'model3': {}
        }

        result = transformer._extract_models(models_data)

        assert 'model1' in result
        assert 'model2' not in result  # Skipped
        assert 'model3' not in result  # Skipped (empty)

    def test_extract_models_uses_id_as_name_fallback(self, transformer):
        """Test that model_id is used as name when name not provided"""
        models_data = {
            'my_model': {
                'type': 'LogisticRegression',
                'metrics': {}
            }
        }

        result = transformer._extract_models(models_data)

        assert result['my_model']['name'] == 'my_model'

    def test_extract_models_default_type(self, transformer):
        """Test default type is 'Unknown' when not provided"""
        models_data = {
            'model1': {'name': 'Test', 'metrics': {}}
        }

        result = transformer._extract_models(models_data)

        assert result['model1']['type'] == 'Unknown'

    def test_extract_models_includes_optional_fields(self, transformer):
        """Test that optional fields are extracted"""
        models_data = {
            'model1': {
                'name': 'Test',
                'metrics': {},
                'hyperparameters': {'max_depth': 5},
                'feature_importance': {'feature1': 0.8}
            }
        }

        result = transformer._extract_models(models_data)

        assert result['model1']['hyperparameters'] == {'max_depth': 5}
        assert result['model1']['feature_importance'] == {'feature1': 0.8}


# ==================== _normalize_metrics Tests ====================


class TestNormalizeMetrics:
    """Tests for _normalize_metrics method"""

    def test_normalize_metrics_standard(self, transformer):
        """Test normalizing standard metrics"""
        metrics = {
            'accuracy': 0.95,
            'roc_auc': 0.98,
            'f1': 0.94
        }

        result = transformer._normalize_metrics(metrics)

        assert result['accuracy'] == 0.95
        assert result['roc_auc'] == 0.98
        assert result['f1'] == 0.94

    def test_normalize_metrics_fills_missing_standard(self, transformer):
        """Test that missing standard metrics are filled with 0.0"""
        metrics = {'accuracy': 0.9}

        result = transformer._normalize_metrics(metrics)

        assert result['accuracy'] == 0.9
        assert result['precision'] == 0.0
        assert result['recall'] == 0.0
        assert result['f1'] == 0.0
        assert result['roc_auc'] == 0.0

    def test_normalize_metrics_none_values(self, transformer):
        """Test handling None values"""
        metrics = {'accuracy': None, 'f1': 0.8}

        result = transformer._normalize_metrics(metrics)

        assert result['accuracy'] == 0.0
        assert result['f1'] == 0.8

    def test_normalize_metrics_string_numbers(self, transformer):
        """Test converting string numbers"""
        metrics = {
            'accuracy': '0.95',
            'f1': '85%'
        }

        result = transformer._normalize_metrics(metrics)

        assert result['accuracy'] == 0.95
        assert result['f1'] == 85.0  # % stripped

    def test_normalize_metrics_skips_error_strings(self, transformer):
        """Test that error metric with string is skipped"""
        metrics = {
            'accuracy': 0.9,
            'error': 'Some error message'
        }

        result = transformer._normalize_metrics(metrics)

        assert 'error' not in result or result.get('error') != 'Some error message'

    def test_normalize_metrics_handles_invalid_conversions(self, transformer):
        """Test handling values that can't be converted"""
        metrics = {
            'accuracy': 'invalid',
            'f1': [1, 2, 3]  # Can't convert list
        }

        with patch('deepbridge.core.experiment.report.transformers.initial_results.logger'):
            result = transformer._normalize_metrics(metrics)

            # Standard metrics should have defaults
            assert result['accuracy'] == 0.0

    def test_normalize_metrics_integer_values(self, transformer):
        """Test converting integer values to float"""
        metrics = {'accuracy': 1, 'f1': 0}

        result = transformer._normalize_metrics(metrics)

        assert isinstance(result['accuracy'], float)
        assert result['accuracy'] == 1.0


# ==================== Integration Tests ====================


class TestIntegration:
    """Integration tests for complete workflows"""

    def test_full_transformation_pipeline(self, transformer):
        """Test complete transformation from raw data to output"""
        raw_data = {
            'config': {
                'dataset_info': {'n_samples': 500},
                'tests': ['robustness', 'fairness'],
                'verbose': False
            },
            'models': {
                'rf_model': {
                    'name': 'RandomForest Classifier',
                    'type': 'RandomForestClassifier',
                    'metrics': {
                        'accuracy': '0.92',
                        'precision': 0.90,
                        'recall': None,
                        'f1': 0.91
                    },
                    'hyperparameters': {'n_estimators': 100}
                },
                'lr_model': {
                    'name': 'LogReg',
                    'type': 'LogisticRegression',
                    'metrics': {'accuracy': 0.88}
                }
            },
            'test_configs': {'config1': 'value1'}
        }

        result = transformer.transform(raw_data)

        # Verify structure
        assert 'config' in result
        assert 'models' in result
        assert 'test_configs' in result

        # Verify config
        assert result['config']['tests'] == ['robustness', 'fairness']
        assert result['config']['verbose'] is False

        # Verify models
        assert len(result['models']) == 2
        assert result['models']['rf_model']['metrics']['accuracy'] == 0.92
        assert result['models']['rf_model']['metrics']['recall'] == 0.0

    def test_transformation_with_errors(self, transformer):
        """Test transformation with various error conditions"""
        problematic_data = {
            'config': {'tests': 'not a list'},  # Wrong type
            'models': {
                'model1': {
                    'metrics': {
                        'accuracy': 'bad_value',
                        'error': 'Error occurred'
                    }
                }
            }
        }

        # Should not crash
        result = transformer.transform(problematic_data)

        assert 'models' in result


# ==================== Edge Cases ====================


class TestEdgeCases:
    """Tests for edge cases"""

    def test_empty_everything(self, transformer):
        """Test with all empty data"""
        data = {'config': {}, 'models': {}, 'test_configs': {}}

        result = transformer.transform(data)

        assert result['config'] == {}
        assert result['models'] == {}

    def test_large_number_of_models(self, transformer):
        """Test with many models"""
        models_data = {
            f'model_{i}': {
                'name': f'Model {i}',
                'metrics': {'accuracy': 0.5 + i * 0.01}
            }
            for i in range(100)
        }

        data = {'models': models_data}
        result = transformer.transform(data)

        assert len(result['models']) == 100

    def test_special_characters_in_names(self, transformer):
        """Test model names with special characters"""
        data = {
            'models': {
                'model-1': {'name': 'Model #1 (v2.0)'},
                'model_2': {'name': 'Model @2'}
            }
        }

        result = transformer.transform(data)

        assert 'model-1' in result['models']
        assert result['models']['model-1']['name'] == 'Model #1 (v2.0)'
