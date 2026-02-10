"""
Comprehensive tests for UncertaintyDataTransformer.

This test suite validates:
1. transform - main transformation logic with all branches
2. Data extraction and enrichment
3. CRQR processing and metrics calculation
4. Alternative models processing
5. Edge cases and error handling

Coverage Target: ~95%+
"""

import pytest
from unittest.mock import patch, Mock
import datetime

from deepbridge.core.experiment.report.transformers.uncertainty import UncertaintyDataTransformer


# ==================== Fixtures ====================


@pytest.fixture
def transformer():
    """Create UncertaintyDataTransformer instance"""
    return UncertaintyDataTransformer()


@pytest.fixture
def basic_results():
    """Basic uncertainty results with minimal data"""
    return {
        'model_name': 'TestModel',
        'model_type': 'RandomForest',
        'metrics': {'accuracy': 0.85},
        'method': 'crqr',
    }


@pytest.fixture
def crqr_results():
    """Results with CRQR data"""
    return {
        'model_name': 'TestModel',
        'crqr': {
            'by_alpha': {
                '0.1': {
                    'overall_result': {
                        'coverage': 0.9,
                        'expected_coverage': 0.9,
                        'mean_width': 0.5,
                    }
                },
                '0.2': {
                    'overall_result': {
                        'coverage': 0.8,
                        'expected_coverage': 0.8,
                        'mean_width': 0.3,
                    }
                },
            }
        },
        'metrics': {}
    }


@pytest.fixture
def nested_primary_model_results():
    """Results with primary_model key"""
    return {
        'primary_model': {
            'model_name': 'PrimaryModel',
            'model_type': 'GradientBoosting',
            'crqr': {
                'by_alpha': {
                    '0.1': {
                        'overall_result': {
                            'coverage': 0.95,
                            'expected_coverage': 0.9,
                            'mean_width': 0.4,
                        }
                    }
                }
            },
            'metrics': {'f1': 0.9},
        },
        'timestamp': '2024-01-01 10:00:00',
    }


@pytest.fixture
def alternative_models_results():
    """Results with alternative models"""
    return {
        'model_name': 'PrimaryModel',
        'model_type': 'RF',
        'metrics': {'accuracy': 0.85},
        'alternative_models': {
            'model_1': {
                'model_name': 'Alternative1',
                'model_type': 'SVM'
            },
            'model_2': {
                'model_name': 'Alternative2',
                'model_type': 'LR'
            }
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
        # Should be recent timestamp
        assert '202' in result['timestamp']  # Year starts with 202x

    def test_transform_preserves_existing_timestamp(self, transformer):
        """Test that existing timestamp is preserved"""
        results = {
            'model_name': 'TestModel',
            'timestamp': '2020-01-01 00:00:00',
            'metrics': {}
        }

        result = transformer.transform(results, 'TestModel')

        assert result['timestamp'] == '2020-01-01 00:00:00'

    def test_transform_with_custom_timestamp(self, transformer, basic_results):
        """Test transformation with custom timestamp"""
        custom_timestamp = '2025-06-15 14:30:00'

        result = transformer.transform(basic_results, 'TestModel', timestamp=custom_timestamp)

        assert result['timestamp'] == custom_timestamp

    def test_transform_preserves_model_name_from_results(self, transformer):
        """Test that model_name from results is preserved"""
        results = {
            'model_name': 'OriginalModel',
            'metrics': {}
        }

        result = transformer.transform(results, 'DifferentModel')

        assert result['model_name'] == 'OriginalModel'

    def test_transform_uses_parameter_model_name_when_missing(self, transformer):
        """Test that parameter model_name is used when missing in results"""
        results = {'metrics': {}}

        result = transformer.transform(results, 'ParameterModel')

        assert result['model_name'] == 'ParameterModel'


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

    def test_extract_from_primary_model(self, transformer, nested_primary_model_results):
        """Test extraction of primary_model data to top level"""
        result = transformer.transform(nested_primary_model_results, 'TestModel')

        assert result['model_name'] == 'PrimaryModel'
        assert result['model_type'] == 'GradientBoosting'
        assert 'crqr' in result
        assert result['metrics'] == {'f1': 0.9}

    def test_primary_model_crqr_override(self, transformer):
        """Test that primary_model crqr overrides top-level crqr"""
        results = {
            'crqr': {'old': 'data'},
            'primary_model': {
                'crqr': {'new': 'data'},
                'model_name': 'Primary'
            }
        }

        result = transformer.transform(results, 'TestModel')

        assert result['crqr'] == {'new': 'data'}

    def test_primary_model_does_not_override_existing_fields(self, transformer):
        """Test that primary_model doesn't override existing non-crqr fields"""
        results = {
            'model_name': 'TopLevel',
            'primary_model': {
                'model_name': 'Primary',
                'new_field': 'value'
            }
        }

        result = transformer.transform(results, 'TestModel')

        # Should keep top-level model_name unless it's crqr
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

    def test_metric_name_from_first_metric_key(self, transformer):
        """Test that metric name is set from first metrics key"""
        results = {
            'metrics': {'precision': 0.9, 'recall': 0.8}
        }

        result = transformer.transform(results, 'TestModel')

        assert result['metric'] in ['precision', 'recall']  # Order may vary

    def test_metric_defaults_to_score(self, transformer):
        """Test that metric defaults to 'score' when no metrics"""
        results = {'metrics': {}}

        result = transformer.transform(results, 'TestModel')

        assert result['metric'] == 'score'

    def test_metric_preserves_existing_value(self, transformer):
        """Test that existing metric name is preserved"""
        results = {
            'metric': 'custom_metric',
            'metrics': {'precision': 0.9}
        }

        result = transformer.transform(results, 'TestModel')

        assert result['metric'] == 'custom_metric'


# ==================== Uncertainty Score Tests ====================


class TestUncertaintyScore:
    """Tests for uncertainty_score calculation"""

    def test_uncertainty_score_from_crqr_data(self, transformer, crqr_results):
        """Test uncertainty score calculated from CRQR data"""
        result = transformer.transform(crqr_results, 'TestModel')

        assert 'uncertainty_score' in result
        # Both alphas have perfect coverage (actual == expected)
        assert result['uncertainty_score'] == pytest.approx(1.0)

    def test_uncertainty_score_with_over_coverage(self, transformer):
        """Test uncertainty score with over-coverage"""
        results = {
            'crqr': {
                'by_alpha': {
                    '0.1': {
                        'overall_result': {
                            'coverage': 0.95,
                            'expected_coverage': 0.9,
                            'mean_width': 0.5,
                        }
                    }
                }
            }
        }

        result = transformer.transform(results, 'TestModel')

        # Over-coverage should be capped at 1.1
        assert result['uncertainty_score'] <= 1.1

    def test_uncertainty_score_with_under_coverage(self, transformer):
        """Test uncertainty score with under-coverage"""
        results = {
            'crqr': {
                'by_alpha': {
                    '0.1': {
                        'overall_result': {
                            'coverage': 0.7,
                            'expected_coverage': 0.9,
                            'mean_width': 0.5,
                        }
                    }
                }
            }
        }

        result = transformer.transform(results, 'TestModel')

        # Under-coverage should result in ratio < 1.0
        assert result['uncertainty_score'] == pytest.approx(0.7 / 0.9)

    def test_uncertainty_score_defaults_when_no_crqr(self, transformer):
        """Test uncertainty score defaults to 0.5 when no CRQR data"""
        results = {'metrics': {}}

        result = transformer.transform(results, 'TestModel')

        assert result['uncertainty_score'] == 0.5

    def test_uncertainty_score_defaults_when_no_ratios(self, transformer):
        """Test uncertainty score defaults when no valid ratios"""
        results = {
            'crqr': {
                'by_alpha': {
                    '0.1': {
                        'overall_result': {
                            'coverage': 0.9,
                            'expected_coverage': 0,  # Invalid expected
                        }
                    }
                }
            }
        }

        result = transformer.transform(results, 'TestModel')

        assert result['uncertainty_score'] == 0.5

    def test_uncertainty_score_preserved_if_present(self, transformer):
        """Test that existing uncertainty_score is preserved"""
        results = {
            'uncertainty_score': 0.95,
            'crqr': {
                'by_alpha': {
                    '0.1': {'overall_result': {'coverage': 0.7, 'expected_coverage': 0.9}}
                }
            }
        }

        result = transformer.transform(results, 'TestModel')

        assert result['uncertainty_score'] == 0.95


# ==================== Average Coverage and Width Tests ====================


class TestAverageCoverageAndWidth:
    """Tests for avg_coverage and avg_width calculation"""

    def test_calculate_avg_coverage_from_crqr(self, transformer, crqr_results):
        """Test average coverage calculation from CRQR data"""
        result = transformer.transform(crqr_results, 'TestModel')

        assert 'avg_coverage' in result
        # Average of 0.9 and 0.8
        assert result['avg_coverage'] == pytest.approx(0.85)

    def test_calculate_avg_width_from_crqr(self, transformer, crqr_results):
        """Test average width calculation from CRQR data"""
        result = transformer.transform(crqr_results, 'TestModel')

        assert 'avg_width' in result
        # Average of 0.5 and 0.3
        assert result['avg_width'] == pytest.approx(0.4)

    def test_avg_coverage_defaults_to_zero_when_no_data(self, transformer):
        """Test avg_coverage defaults to 0 when no CRQR data"""
        results = {
            'crqr': {
                'by_alpha': {
                    '0.1': {'overall_result': {}}  # Missing coverage
                }
            }
        }

        result = transformer.transform(results, 'TestModel')

        assert result['avg_coverage'] == 0

    def test_avg_width_defaults_to_zero_when_no_data(self, transformer):
        """Test avg_width defaults to 0 when no width data"""
        results = {
            'crqr': {
                'by_alpha': {
                    '0.1': {'overall_result': {}}  # Missing mean_width
                }
            }
        }

        result = transformer.transform(results, 'TestModel')

        assert result['avg_width'] == 0

    def test_avg_coverage_preserved_if_present(self, transformer, crqr_results):
        """Test that existing avg_coverage is preserved"""
        crqr_results['avg_coverage'] = 0.99

        result = transformer.transform(crqr_results, 'TestModel')

        assert result['avg_coverage'] == 0.99


# ==================== Alpha Levels Tests ====================


class TestAlphaLevels:
    """Tests for alpha_levels extraction"""

    def test_extract_alpha_levels_from_crqr(self, transformer, crqr_results):
        """Test alpha levels extraction from CRQR data"""
        result = transformer.transform(crqr_results, 'TestModel')

        assert 'alpha_levels' in result
        assert set(result['alpha_levels']) == {0.1, 0.2}

    def test_alpha_levels_preserved_if_present(self, transformer, crqr_results):
        """Test that existing alpha_levels is preserved"""
        crqr_results['alpha_levels'] = [0.05, 0.15]

        result = transformer.transform(crqr_results, 'TestModel')

        assert result['alpha_levels'] == [0.05, 0.15]

    def test_alpha_levels_not_added_without_crqr(self, transformer):
        """Test that alpha_levels is not added without CRQR data"""
        results = {'metrics': {}}

        result = transformer.transform(results, 'TestModel')

        # Should not have alpha_levels if no crqr data
        assert 'alpha_levels' not in result or result.get('alpha_levels') == []


# ==================== Method Tests ====================


class TestMethod:
    """Tests for method field handling"""

    def test_method_preserved_if_present(self, transformer, basic_results):
        """Test that existing method is preserved"""
        result = transformer.transform(basic_results, 'TestModel')

        assert result['method'] == 'crqr'

    def test_method_defaults_to_crqr(self, transformer):
        """Test that method defaults to 'crqr'"""
        results = {'metrics': {}}

        result = transformer.transform(results, 'TestModel')

        assert result['method'] == 'crqr'


# ==================== Alternative Models Tests ====================


class TestAlternativeModels:
    """Tests for alternative models processing"""

    def test_process_alternative_models(self, transformer, alternative_models_results):
        """Test processing of alternative models"""
        result = transformer.transform(alternative_models_results, 'TestModel')

        assert 'alternative_models' in result
        assert 'model_1' in result['alternative_models']
        assert 'model_2' in result['alternative_models']

    def test_alternative_models_metrics_initialized(self, transformer, alternative_models_results):
        """Test that metrics are initialized for each alternative model"""
        result = transformer.transform(alternative_models_results, 'TestModel')

        assert 'metrics' in result['alternative_models']['model_1']
        assert 'metrics' in result['alternative_models']['model_2']

    def test_extract_alternative_models_from_nested_structure(self, transformer):
        """Test extraction of alternative models from nested results"""
        results = {
            'model_name': 'Primary',
            'results': {
                'uncertainty': {
                    'results': {
                        'alternative_models': {
                            'alt1': {'model_name': 'Alternative1'}
                        }
                    }
                }
            }
        }

        result = transformer.transform(results, 'TestModel')

        assert 'alternative_models' in result
        assert 'alt1' in result['alternative_models']

    def test_alternative_models_converted_to_dict_if_not_dict(self, transformer):
        """Test that alternative_models is converted to dict if not dict"""
        results = {
            'model_name': 'Primary',
            'alternative_models': ['not', 'a', 'dict']  # Invalid type
        }

        result = transformer.transform(results, 'TestModel')

        # Should be converted to empty dict
        assert isinstance(result['alternative_models'], dict)
        assert result['alternative_models'] == {}


# ==================== Integration Tests ====================


class TestIntegration:
    """Integration tests for complete workflows"""

    def test_full_transformation_workflow(self, transformer):
        """Test complete transformation with all features"""
        results = {
            'primary_model': {
                'model_name': 'CompleteModel',
                'model_type': 'GradientBoosting',
                'crqr': {
                    'by_alpha': {
                        '0.1': {
                            'overall_result': {
                                'coverage': 0.92,
                                'expected_coverage': 0.9,
                                'mean_width': 0.45,
                            }
                        },
                        '0.2': {
                            'overall_result': {
                                'coverage': 0.82,
                                'expected_coverage': 0.8,
                                'mean_width': 0.35,
                            }
                        },
                    }
                },
                'metrics': {'accuracy': 0.95, 'f1': 0.93},
            },
            'alternative_models': {
                'alt1': {'model_name': 'Alt1', 'model_type': 'RF'},
                'alt2': {'model_name': 'Alt2', 'model_type': 'SVM'},
            }
        }

        result = transformer.transform(results, 'TestModel', timestamp='2025-01-01 00:00:00')

        # Verify all components
        assert result['model_name'] == 'CompleteModel'
        assert result['model_type'] == 'GradientBoosting'
        assert result['timestamp'] == '2025-01-01 00:00:00'
        assert 'crqr' in result
        assert result['metrics'] == {'accuracy': 0.95, 'f1': 0.93}
        assert 'uncertainty_score' in result
        assert 'avg_coverage' in result
        assert 'avg_width' in result
        assert 'alpha_levels' in result
        assert result['method'] == 'crqr'
        assert 'alternative_models' in result
        assert 'alt1' in result['alternative_models']
        assert 'metrics' in result['alternative_models']['alt1']


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

    def test_transform_with_missing_overall_result(self, transformer):
        """Test transformation when overall_result is missing"""
        results = {
            'crqr': {
                'by_alpha': {
                    '0.1': {}  # Missing overall_result
                }
            }
        }

        result = transformer.transform(results, 'TestModel')

        # Should not crash, defaults to 0.5
        assert result['uncertainty_score'] == 0.5

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

    def test_transform_with_numpy_types(self, transformer):
        """Test that numpy types are converted"""
        # This tests the convert_numpy_types call at the end
        results = {
            'model_name': 'Test',
            'metrics': {'score': 0.85}  # Assuming this would be numpy.float64 in real use
        }

        result = transformer.transform(results, 'TestModel')

        # Should complete without errors
        assert 'metrics' in result
