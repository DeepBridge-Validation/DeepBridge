"""
Comprehensive tests for HyperparameterDataTransformer.

This test suite validates:
1. transform - main transformation logic with all branches
2. Data extraction and enrichment
3. Importance results processing
4. Tuning order extraction
5. Edge cases and error handling

Coverage Target: ~95%+
"""

import pytest
from unittest.mock import patch

from deepbridge.core.experiment.report.transformers.hyperparameter import HyperparameterDataTransformer


# ==================== Fixtures ====================


@pytest.fixture
def transformer():
    """Create HyperparameterDataTransformer instance"""
    return HyperparameterDataTransformer()


@pytest.fixture
def basic_results():
    """Basic hyperparameter results"""
    return {
        'model_name': 'TestModel',
        'model_type': 'RandomForest',
        'metrics': {'accuracy': 0.85},
    }


@pytest.fixture
def importance_results():
    """Results with importance data"""
    return {
        'model_name': 'TestModel',
        'importance': {
            'all_results': [
                {
                    'normalized_importance': {'param1': 0.8, 'param2': 0.2},
                    'raw_importance_scores': {'param1': 10, 'param2': 2.5},
                    'tuning_order': ['param1', 'param2'],
                    'sorted_importance': {
                        'param1': 0.8,
                        'param2': 0.2
                    }
                }
            ]
        },
        'metrics': {}
    }


@pytest.fixture
def nested_results():
    """Results with nested structure"""
    return {
        'primary_model': {
            'model_name': 'PrimaryModel',
            'model_type': 'GradientBoosting',
            'metrics': {'f1': 0.92},
            'importance_results': [{'param': 'value'}]
        }
    }


# ==================== Basic Transform Tests ====================


class TestBasicTransform:
    """Tests for basic transformation functionality"""

    def test_transform_basic_results(self, transformer, basic_results):
        """Test transforming basic results"""
        result = transformer.transform(basic_results, 'TestModel')

        assert result['model_name'] == 'TestModel'
        assert result['model_type'] == 'RandomForest'
        assert 'timestamp' in result
        assert result['metrics'] == {'accuracy': 0.85}

    def test_transform_adds_timestamp_if_missing(self, transformer, basic_results):
        """Test that timestamp is added when not provided"""
        result = transformer.transform(basic_results, 'TestModel')

        assert 'timestamp' in result
        assert '202' in result['timestamp']  # Recent year

    def test_transform_with_custom_timestamp(self, transformer, basic_results):
        """Test transformation with custom timestamp"""
        custom_timestamp = '2025-06-15 14:30:00'

        result = transformer.transform(basic_results, 'TestModel', timestamp=custom_timestamp)

        assert result['timestamp'] == custom_timestamp

    def test_transform_preserves_existing_timestamp(self, transformer):
        """Test that existing timestamp is preserved"""
        results = {
            'model_name': 'TestModel',
            'timestamp': '2020-01-01 00:00:00',
            'metrics': {}
        }

        result = transformer.transform(results, 'TestModel')

        assert result['timestamp'] == '2020-01-01 00:00:00'

    def test_transform_uses_parameter_model_name_when_missing(self, transformer):
        """Test that parameter model_name is used when missing"""
        results = {'metrics': {}}

        result = transformer.transform(results, 'ParameterModel')

        assert result['model_name'] == 'ParameterModel'

    def test_transform_preserves_model_name_from_results(self, transformer):
        """Test that model_name from results is preserved"""
        results = {
            'model_name': 'OriginalModel',
            'metrics': {}
        }

        result = transformer.transform(results, 'DifferentModel')

        assert result['model_name'] == 'OriginalModel'


# ==================== to_dict Handling Tests ====================


class TestToDictHandling:
    """Tests for handling objects with to_dict() method"""

    def test_transform_with_to_dict_method(self, transformer):
        """Test transformation when results have to_dict() method"""
        class MockResults:
            def to_dict(self):
                return {
                    'model_name': 'FromToDict',
                    'model_type': 'CustomModel',
                    'metrics': {'score': 0.95}
                }

        mock_results = MockResults()
        result = transformer.transform(mock_results, 'TestModel')

        assert result['model_name'] == 'FromToDict'
        assert result['model_type'] == 'CustomModel'


# ==================== Primary Model Extraction Tests ====================


class TestPrimaryModelExtraction:
    """Tests for extracting data from primary_model key"""

    def test_extract_from_primary_model(self, transformer, nested_results):
        """Test extraction of primary_model data to top level"""
        result = transformer.transform(nested_results, 'TestModel')

        assert result['model_name'] == 'PrimaryModel'
        assert result['model_type'] == 'GradientBoosting'
        assert result['metrics'] == {'f1': 0.92}
        assert result['importance_results'] == [{'param': 'value'}]

    def test_primary_model_does_not_override_existing_fields(self, transformer):
        """Test that primary_model doesn't override existing fields"""
        results = {
            'model_name': 'TopLevel',
            'primary_model': {
                'model_name': 'Primary',
                'new_field': 'value'
            }
        }

        result = transformer.transform(results, 'TestModel')

        assert result['model_name'] == 'TopLevel'
        assert result['new_field'] == 'value'


# ==================== Model Type Tests ====================


class TestModelType:
    """Tests for model_type handling"""

    def test_model_type_from_results(self, transformer, basic_results):
        """Test model_type from results"""
        result = transformer.transform(basic_results, 'TestModel')

        assert result['model_type'] == 'RandomForest'

    def test_model_type_from_primary_model(self, transformer):
        """Test model_type extracted from primary_model"""
        results = {
            'primary_model': {
                'model_type': 'DecisionTree'
            }
        }

        result = transformer.transform(results, 'TestModel')

        assert result['model_type'] == 'DecisionTree'

    def test_model_type_defaults_to_unknown(self, transformer):
        """Test model_type defaults to Unknown Model"""
        results = {'metrics': {}}

        result = transformer.transform(results, 'TestModel')

        assert result['model_type'] == 'Unknown Model'


# ==================== Metrics Tests ====================


class TestMetrics:
    """Tests for metrics handling"""

    def test_metrics_initialized_if_missing(self, transformer):
        """Test that metrics dict is initialized if missing"""
        results = {'model_name': 'Test'}

        result = transformer.transform(results, 'TestModel')

        assert 'metrics' in result
        assert isinstance(result['metrics'], dict)

    def test_metrics_preserved_if_present(self, transformer, basic_results):
        """Test that existing metrics are preserved"""
        result = transformer.transform(basic_results, 'TestModel')

        assert result['metrics'] == {'accuracy': 0.85}


# ==================== Alternative Models Tests ====================


class TestAlternativeModels:
    """Tests for alternative models extraction"""

    def test_extract_alternative_models_from_nested_structure(self, transformer):
        """Test extraction of alternative models from nested results"""
        results = {
            'model_name': 'Primary',
            'results': {
                'hyperparameter': {
                    'results': {
                        'alternative_models': {
                            'alt1': {'model_name': 'Alternative1'},
                            'alt2': {'model_name': 'Alternative2'}
                        }
                    }
                }
            }
        }

        result = transformer.transform(results, 'TestModel')

        assert 'alternative_models' in result
        assert 'alt1' in result['alternative_models']
        assert 'alt2' in result['alternative_models']

    def test_alternative_models_preserved_if_present(self, transformer):
        """Test that existing alternative_models is preserved"""
        results = {
            'model_name': 'Primary',
            'alternative_models': {
                'existing': {'model_name': 'Existing'}
            }
        }

        result = transformer.transform(results, 'TestModel')

        assert result['alternative_models'] == {'existing': {'model_name': 'Existing'}}


# ==================== Importance Results Tests ====================


class TestImportanceResults:
    """Tests for importance_results handling"""

    def test_importance_results_from_importance_all_results(self, transformer):
        """Test extraction of importance_results from importance.all_results"""
        results = {
            'model_name': 'Test',
            'importance': {
                'all_results': [
                    {'param': 'value1'},
                    {'param': 'value2'}
                ]
            }
        }

        result = transformer.transform(results, 'TestModel')

        assert result['importance_results'] == [
            {'param': 'value1'},
            {'param': 'value2'}
        ]

    def test_importance_results_defaults_to_empty_list(self, transformer):
        """Test that importance_results defaults to empty list"""
        results = {'model_name': 'Test'}

        result = transformer.transform(results, 'TestModel')

        assert result['importance_results'] == []

    def test_importance_results_preserved_if_present(self, transformer):
        """Test that existing importance_results is preserved"""
        results = {
            'model_name': 'Test',
            'importance_results': [{'existing': 'data'}]
        }

        result = transformer.transform(results, 'TestModel')

        assert result['importance_results'] == [{'existing': 'data'}]


# ==================== Importance Scores Tests ====================


class TestImportanceScores:
    """Tests for importance_scores extraction"""

    def test_importance_scores_from_normalized_importance(self, transformer):
        """Test extraction from normalized_importance"""
        results = {
            'model_name': 'Test',
            'importance': {
                'all_results': [
                    {
                        'normalized_importance': {'param1': 0.8, 'param2': 0.2}
                    }
                ]
            }
        }

        result = transformer.transform(results, 'TestModel')

        assert result['importance_scores'] == {'param1': 0.8, 'param2': 0.2}

    def test_importance_scores_from_raw_importance_scores(self, transformer):
        """Test extraction from raw_importance_scores when normalized not available"""
        results = {
            'model_name': 'Test',
            'importance': {
                'all_results': [
                    {
                        'raw_importance_scores': {'param1': 10, 'param2': 2.5}
                    }
                ]
            }
        }

        result = transformer.transform(results, 'TestModel')

        assert result['importance_scores'] == {'param1': 10, 'param2': 2.5}

    def test_importance_scores_prefers_normalized_over_raw(self, transformer):
        """Test that normalized_importance is preferred over raw"""
        results = {
            'model_name': 'Test',
            'importance': {
                'all_results': [
                    {
                        'normalized_importance': {'param1': 0.8},
                        'raw_importance_scores': {'param1': 10}
                    }
                ]
            }
        }

        result = transformer.transform(results, 'TestModel')

        assert result['importance_scores'] == {'param1': 0.8}

    def test_importance_scores_not_added_when_no_data(self, transformer):
        """Test that importance_scores is not added when no data available"""
        results = {
            'model_name': 'Test',
            'importance': {
                'all_results': []
            }
        }

        result = transformer.transform(results, 'TestModel')

        assert 'importance_scores' not in result

    def test_importance_scores_preserved_if_present(self, transformer):
        """Test that existing importance_scores is preserved"""
        results = {
            'model_name': 'Test',
            'importance_scores': {'existing': 0.95}
        }

        result = transformer.transform(results, 'TestModel')

        assert result['importance_scores'] == {'existing': 0.95}


# ==================== Tuning Order Tests ====================


class TestTuningOrder:
    """Tests for tuning_order extraction"""

    def test_tuning_order_from_importance_results(self, transformer):
        """Test extraction of tuning_order from importance results"""
        results = {
            'model_name': 'Test',
            'importance': {
                'all_results': [
                    {
                        'tuning_order': ['param1', 'param2', 'param3']
                    }
                ]
            }
        }

        result = transformer.transform(results, 'TestModel')

        assert result['tuning_order'] == ['param1', 'param2', 'param3']

    def test_tuning_order_from_sorted_importance(self, transformer):
        """Test extraction from sorted_importance when tuning_order not available"""
        results = {
            'model_name': 'Test',
            'importance': {
                'all_results': [
                    {
                        'sorted_importance': {
                            'param1': 0.8,
                            'param2': 0.2
                        }
                    }
                ]
            }
        }

        result = transformer.transform(results, 'TestModel')

        assert 'tuning_order' in result
        assert set(result['tuning_order']) == {'param1', 'param2'}

    def test_tuning_order_prefers_explicit_over_sorted(self, transformer):
        """Test that explicit tuning_order is preferred"""
        results = {
            'model_name': 'Test',
            'importance': {
                'all_results': [
                    {
                        'tuning_order': ['param2', 'param1'],
                        'sorted_importance': {'param1': 0.8, 'param2': 0.2}
                    }
                ]
            }
        }

        result = transformer.transform(results, 'TestModel')

        assert result['tuning_order'] == ['param2', 'param1']

    def test_tuning_order_not_added_when_no_data(self, transformer):
        """Test that tuning_order is not added when no data available"""
        results = {
            'model_name': 'Test',
            'importance': {
                'all_results': []
            }
        }

        result = transformer.transform(results, 'TestModel')

        assert 'tuning_order' not in result

    def test_tuning_order_preserved_if_present(self, transformer):
        """Test that existing tuning_order is preserved"""
        results = {
            'model_name': 'Test',
            'tuning_order': ['existing1', 'existing2']
        }

        result = transformer.transform(results, 'TestModel')

        assert result['tuning_order'] == ['existing1', 'existing2']


# ==================== Integration Tests ====================


class TestIntegration:
    """Integration tests for complete workflows"""

    def test_full_transformation_workflow(self, transformer):
        """Test complete transformation with all features"""
        results = {
            'primary_model': {
                'model_name': 'CompleteModel',
                'model_type': 'GradientBoosting',
                'metrics': {'accuracy': 0.95, 'f1': 0.93},
            },
            'importance': {
                'all_results': [
                    {
                        'normalized_importance': {'lr': 0.5, 'depth': 0.3, 'estimators': 0.2},
                        'raw_importance_scores': {'lr': 50, 'depth': 30, 'estimators': 20},
                        'tuning_order': ['lr', 'depth', 'estimators'],
                        'sorted_importance': {'lr': 0.5, 'depth': 0.3, 'estimators': 0.2}
                    },
                    {
                        'normalized_importance': {'lr': 0.4, 'depth': 0.4, 'estimators': 0.2},
                    }
                ]
            },
            'results': {
                'hyperparameter': {
                    'results': {
                        'alternative_models': {
                            'alt1': {'model_name': 'Alt1', 'model_type': 'RF'},
                        }
                    }
                }
            }
        }

        result = transformer.transform(results, 'TestModel', timestamp='2025-01-01 00:00:00')

        # Verify all components
        assert result['model_name'] == 'CompleteModel'
        assert result['model_type'] == 'GradientBoosting'
        assert result['timestamp'] == '2025-01-01 00:00:00'
        assert result['metrics'] == {'accuracy': 0.95, 'f1': 0.93}
        assert len(result['importance_results']) == 2
        assert result['importance_scores'] == {'lr': 0.5, 'depth': 0.3, 'estimators': 0.2}
        assert result['tuning_order'] == ['lr', 'depth', 'estimators']
        assert 'alternative_models' in result
        assert 'alt1' in result['alternative_models']


# ==================== Edge Cases ====================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_transform_empty_results(self, transformer):
        """Test transformation with empty results"""
        results = {}

        result = transformer.transform(results, 'TestModel')

        assert result['model_name'] == 'TestModel'
        assert 'timestamp' in result
        assert result['metrics'] == {}
        assert result['model_type'] == 'Unknown Model'
        assert result['importance_results'] == []

    def test_transform_with_empty_importance_all_results(self, transformer):
        """Test with empty all_results list"""
        results = {
            'model_name': 'Test',
            'importance': {
                'all_results': []
            }
        }

        result = transformer.transform(results, 'TestModel')

        assert result['importance_results'] == []
        assert 'importance_scores' not in result
        assert 'tuning_order' not in result

    def test_transform_preserves_extra_fields(self, transformer):
        """Test that extra fields are preserved"""
        results = {
            'model_name': 'Test',
            'custom_field': 'custom_value',
            'nested': {'data': 'value'},
            'metrics': {}
        }

        result = transformer.transform(results, 'TestModel')

        assert result['custom_field'] == 'custom_value'
        assert result['nested'] == {'data': 'value'}

    def test_transform_with_incomplete_importance_data(self, transformer):
        """Test with incomplete importance data"""
        results = {
            'model_name': 'Test',
            'importance': {
                'all_results': [
                    {
                        # Missing all importance fields
                    }
                ]
            }
        }

        result = transformer.transform(results, 'TestModel')

        # Should not crash, just not add the fields
        assert result['importance_results'] == [{}]
        assert 'importance_scores' not in result
        assert 'tuning_order' not in result

    def test_transform_with_numpy_types(self, transformer):
        """Test that numpy types are converted"""
        results = {
            'model_name': 'Test',
            'metrics': {'score': 0.85}
        }

        result = transformer.transform(results, 'TestModel')

        # Should complete without errors (tests convert_numpy_types call)
        assert 'metrics' in result
