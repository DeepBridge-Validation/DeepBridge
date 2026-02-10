"""
Comprehensive tests for UncertaintyPipeline.

This test suite validates:
1. UncertaintyValidator - data validation
2. UncertaintyTransformer - data transformation
3. UncertaintyEnricher - data enrichment
4. create_uncertainty_pipeline - pipeline factory
5. Integration tests for complete workflows
6. Edge cases and error handling

Coverage Target: ~95%+
"""

import pytest
from unittest.mock import patch

from deepbridge.core.experiment.report.transformers.uncertainty_pipeline import (
    UncertaintyValidator,
    UncertaintyTransformer,
    UncertaintyEnricher,
    create_uncertainty_pipeline,
)


# ==================== Fixtures ====================


@pytest.fixture
def valid_uncertainty_data():
    """Create valid uncertainty data for testing"""
    return {
        'model_name': 'XGBoost',
        'test_results': {
            'primary_model': {
                'model_type': 'XGBRegressor',
                'crqr': {
                    'alphas': {
                        '0.1': {
                            'coverage': 0.91,
                            'expected_coverage': 0.90,
                            'avg_width': 2.34,
                        },
                        '0.2': {
                            'coverage': 0.81,
                            'expected_coverage': 0.80,
                            'avg_width': 1.87,
                        },
                        '0.3': {
                            'coverage': 0.71,
                            'expected_coverage': 0.70,
                            'avg_width': 1.45,
                        },
                    }
                },
            }
        },
        'initial_model_evaluation': {
            'feature_importance': {
                'feature1': 0.45,
                'feature2': 0.32,
                'feature3': 0.15,
                'feature4': 0.08,
            }
        },
    }


@pytest.fixture
def validator():
    """Create UncertaintyValidator instance"""
    return UncertaintyValidator()


@pytest.fixture
def transformer():
    """Create UncertaintyTransformer instance"""
    return UncertaintyTransformer()


@pytest.fixture
def enricher():
    """Create UncertaintyEnricher instance"""
    return UncertaintyEnricher()


# ==================== UncertaintyValidator Tests ====================


class TestUncertaintyValidator:
    """Tests for UncertaintyValidator class"""

    def test_validate_valid_data(self, validator, valid_uncertainty_data):
        """Test validation with valid data"""
        errors = validator.validate(valid_uncertainty_data)
        assert errors == []

    def test_validate_missing_test_results(self, validator):
        """Test validation fails when test_results is missing"""
        data = {'initial_model_evaluation': {'feature_importance': {}}}
        errors = validator.validate(data)

        assert len(errors) > 0
        assert any('test_results' in err for err in errors)

    def test_validate_missing_primary_model(self, validator):
        """Test validation fails when primary_model is missing"""
        data = {
            'test_results': {},
            'initial_model_evaluation': {'feature_importance': {}},
        }
        errors = validator.validate(data)

        assert len(errors) > 0
        assert any('primary_model' in err for err in errors)

    def test_validate_missing_crqr(self, validator):
        """Test validation fails when crqr is missing"""
        data = {
            'test_results': {'primary_model': {}},
            'initial_model_evaluation': {'feature_importance': {}},
        }
        errors = validator.validate(data)

        assert len(errors) > 0
        assert any('crqr' in err for err in errors)

    def test_validate_missing_alphas(self, validator):
        """Test validation fails when alphas is missing"""
        data = {
            'test_results': {'primary_model': {'crqr': {}}},
            'initial_model_evaluation': {'feature_importance': {}},
        }
        errors = validator.validate(data)

        assert len(errors) > 0
        assert any('alphas' in err for err in errors)

    def test_validate_empty_alphas(self, validator):
        """Test validation fails when alphas is empty"""
        data = {
            'test_results': {'primary_model': {'crqr': {'alphas': {}}}},
            'initial_model_evaluation': {'feature_importance': {}},
        }
        errors = validator.validate(data)

        assert len(errors) > 0
        assert any('alphas' in err for err in errors)

    def test_validate_missing_initial_model_evaluation(self, validator):
        """Test validation fails when initial_model_evaluation is missing"""
        data = {
            'test_results': {
                'primary_model': {'crqr': {'alphas': {'0.1': {}}}}
            }
        }
        errors = validator.validate(data)

        assert len(errors) > 0
        assert any('initial_model_evaluation' in err for err in errors)

    def test_validate_missing_feature_importance(self, validator):
        """Test validation fails when feature_importance is missing"""
        data = {
            'test_results': {
                'primary_model': {'crqr': {'alphas': {'0.1': {}}}}
            },
            'initial_model_evaluation': {},
        }
        errors = validator.validate(data)

        assert len(errors) > 0
        assert any('feature_importance' in err for err in errors)

    def test_validate_multiple_errors(self, validator):
        """Test validation returns multiple errors"""
        data = {}
        errors = validator.validate(data)

        # Should have errors for both test_results and initial_model_evaluation
        assert len(errors) >= 2


# ==================== UncertaintyTransformer Tests ====================


class TestUncertaintyTransformer:
    """Tests for UncertaintyTransformer class"""

    def test_transform_valid_data(self, transformer, valid_uncertainty_data):
        """Test transformation with valid data"""
        result = transformer.transform(valid_uncertainty_data)

        assert result['model_name'] == 'XGBoost'
        assert result['model_type'] == 'XGBRegressor'
        assert len(result['alphas']) == 3
        assert 'feature_importance' in result
        assert 'metadata' in result

    def test_transform_alphas_sorted(self, transformer, valid_uncertainty_data):
        """Test that alphas are sorted by alpha value"""
        result = transformer.transform(valid_uncertainty_data)

        alphas = result['alphas']
        assert alphas[0]['alpha'] == 0.1
        assert alphas[1]['alpha'] == 0.2
        assert alphas[2]['alpha'] == 0.3

    def test_transform_alpha_data_structure(
        self, transformer, valid_uncertainty_data
    ):
        """Test alpha data structure is correct"""
        result = transformer.transform(valid_uncertainty_data)

        alpha_data = result['alphas'][0]
        assert 'alpha' in alpha_data
        assert 'coverage' in alpha_data
        assert 'expected_coverage' in alpha_data
        assert 'avg_width' in alpha_data
        assert 'calibration_error' in alpha_data

    def test_transform_calibration_error_calculation(
        self, transformer, valid_uncertainty_data
    ):
        """Test calibration error is calculated correctly"""
        result = transformer.transform(valid_uncertainty_data)

        alpha_data = result['alphas'][0]
        # coverage=0.91, expected=0.90, error should be 0.01
        assert alpha_data['calibration_error'] == abs(0.91 - 0.90)

    def test_transform_missing_optional_fields(self, transformer):
        """Test transformation with missing optional fields"""
        data = {
            'test_results': {
                'primary_model': {
                    'crqr': {
                        'alphas': {
                            '0.1': {}  # Empty alpha data
                        }
                    }
                }
            },
            'initial_model_evaluation': {},
        }

        result = transformer.transform(data)

        assert result['model_name'] == 'Model'  # Default value
        assert result['model_type'] == 'Unknown'  # Default value
        assert len(result['alphas']) == 1
        assert result['alphas'][0]['coverage'] == 0.0
        assert result['alphas'][0]['avg_width'] == 0.0

    def test_transform_feature_importance_extraction(
        self, transformer, valid_uncertainty_data
    ):
        """Test feature importance is extracted correctly"""
        result = transformer.transform(valid_uncertainty_data)

        assert len(result['feature_importance']) == 4
        assert result['feature_importance']['feature1'] == 0.45
        assert result['feature_importance']['feature2'] == 0.32

    def test_transform_metadata(self, transformer, valid_uncertainty_data):
        """Test metadata is calculated correctly"""
        result = transformer.transform(valid_uncertainty_data)

        assert result['metadata']['num_alphas'] == 3
        assert result['metadata']['num_features'] == 4

    def test_transform_non_dict_alpha_values_skipped(self, transformer):
        """Test that non-dict alpha values are skipped"""
        data = {
            'test_results': {
                'primary_model': {
                    'crqr': {
                        'alphas': {
                            '0.1': {'coverage': 0.9},
                            '0.2': 'not_a_dict',  # Should be skipped
                            '0.3': {'coverage': 0.7},
                        }
                    }
                }
            },
            'initial_model_evaluation': {'feature_importance': {}},
        }

        result = transformer.transform(data)

        # Only 2 alphas should be processed (0.1 and 0.3)
        assert len(result['alphas']) == 2


# ==================== UncertaintyEnricher Tests ====================


class TestUncertaintyEnricher:
    """Tests for UncertaintyEnricher class"""

    def test_enrich_adds_summary(self, enricher):
        """Test enrichment adds summary field"""
        data = {
            'alphas': [
                {
                    'alpha': 0.1,
                    'coverage': 0.91,
                    'calibration_error': 0.01,
                    'avg_width': 2.0,
                },
                {
                    'alpha': 0.2,
                    'coverage': 0.81,
                    'calibration_error': 0.01,
                    'avg_width': 1.5,
                },
            ],
            'feature_importance': {'feat1': 0.5, 'feat2': 0.3},
        }

        result = enricher.enrich(data)

        assert 'summary' in result
        assert 'uncertainty_score' in result['summary']
        assert 'total_alphas' in result['summary']
        assert 'avg_coverage' in result['summary']
        assert 'avg_calibration_error' in result['summary']
        assert 'avg_width' in result['summary']
        assert 'is_well_calibrated' in result['summary']

    def test_enrich_summary_calculations(self, enricher):
        """Test summary calculations are correct"""
        data = {
            'alphas': [
                {
                    'coverage': 0.90,
                    'calibration_error': 0.02,
                    'avg_width': 2.0,
                },
                {
                    'coverage': 0.80,
                    'calibration_error': 0.04,
                    'avg_width': 1.0,
                },
            ],
            'feature_importance': {},
        }

        result = enricher.enrich(data)
        summary = result['summary']

        # avg_coverage = (0.90 + 0.80) / 2 = 0.85
        assert summary['avg_coverage'] == 0.85
        # avg_calibration_error = (0.02 + 0.04) / 2 = 0.03
        assert summary['avg_calibration_error'] == 0.03
        # avg_width = (2.0 + 1.0) / 2 = 1.5
        assert summary['avg_width'] == 1.5
        # total_alphas = 2
        assert summary['total_alphas'] == 2

    def test_enrich_uncertainty_score_calculation(self, enricher):
        """Test uncertainty score calculation"""
        data = {
            'alphas': [
                {'coverage': 0.90, 'calibration_error': 0.02, 'avg_width': 1.0}
            ],
            'feature_importance': {},
        }

        result = enricher.enrich(data)

        # uncertainty_score = max(0, 1 - (0.02 * 2)) = 0.96
        assert result['summary']['uncertainty_score'] == 0.96

    def test_enrich_is_well_calibrated_true(self, enricher):
        """Test is_well_calibrated is True when error < 0.05"""
        data = {
            'alphas': [
                {'coverage': 0.90, 'calibration_error': 0.03, 'avg_width': 1.0}
            ],
            'feature_importance': {},
        }

        result = enricher.enrich(data)

        assert result['summary']['is_well_calibrated'] is True

    def test_enrich_is_well_calibrated_false(self, enricher):
        """Test is_well_calibrated is False when error >= 0.05"""
        data = {
            'alphas': [
                {'coverage': 0.90, 'calibration_error': 0.10, 'avg_width': 1.0}
            ],
            'feature_importance': {},
        }

        result = enricher.enrich(data)

        assert result['summary']['is_well_calibrated'] is False

    def test_enrich_empty_alphas(self, enricher):
        """Test enrichment with empty alphas list"""
        data = {'alphas': [], 'feature_importance': {}}

        result = enricher.enrich(data)
        summary = result['summary']

        assert summary['total_alphas'] == 0
        assert summary['avg_coverage'] == 0.0
        assert summary['avg_calibration_error'] == 0.0
        assert summary['avg_width'] == 0.0
        assert summary['uncertainty_score'] == 0.0

    def test_enrich_top_features(self, enricher):
        """Test top features extraction"""
        data = {
            'alphas': [],
            'feature_importance': {
                'feat1': 0.5,
                'feat2': -0.4,  # Negative importance
                'feat3': 0.3,
                'feat4': 0.1,
            },
        }

        result = enricher.enrich(data)

        # Should be sorted by absolute importance
        top_features = result['top_features']
        assert len(top_features) <= 10
        assert top_features[0][0] == 'feat1'  # 0.5 (highest)
        assert top_features[1][0] == 'feat2'  # -0.4 (second highest absolute)
        assert top_features[2][0] == 'feat3'  # 0.3

    def test_enrich_top_features_limit_10(self, enricher):
        """Test that top features are limited to 10"""
        data = {
            'alphas': [],
            'feature_importance': {f'feat{i}': 0.1 * i for i in range(20)},
        }

        result = enricher.enrich(data)

        assert len(result['top_features']) == 10

    def test_enrich_empty_feature_importance(self, enricher):
        """Test enrichment with empty feature importance"""
        data = {'alphas': [], 'feature_importance': {}}

        result = enricher.enrich(data)

        assert result['top_features'] == []
        assert result['features']['total'] == 0
        assert result['features']['top_10'] == []

    def test_enrich_adds_features_field(self, enricher):
        """Test enrichment adds features field"""
        data = {
            'alphas': [],
            'feature_importance': {'feat1': 0.5, 'feat2': 0.3},
        }

        result = enricher.enrich(data)

        assert 'features' in result
        assert result['features']['total'] == 2
        assert 'top_10' in result['features']

    @patch('deepbridge.core.experiment.report.transformers.uncertainty_pipeline.logger')
    def test_enrich_logs_debug_info(self, mock_logger, enricher):
        """Test enrichment logs debug information"""
        data = {
            'alphas': [
                {'coverage': 0.9, 'calibration_error': 0.02, 'avg_width': 1.0}
            ],
            'feature_importance': {'feat1': 0.5},
        }

        enricher.enrich(data)

        mock_logger.debug.assert_called_once()


# ==================== create_uncertainty_pipeline Tests ====================


class TestCreateUncertaintyPipeline:
    """Tests for create_uncertainty_pipeline factory function"""

    def test_creates_pipeline_instance(self):
        """Test factory creates TransformPipeline instance"""
        pipeline = create_uncertainty_pipeline()

        assert pipeline is not None
        # Check it has the expected type name
        assert 'TransformPipeline' in str(type(pipeline))

    def test_pipeline_has_correct_stages(self):
        """Test pipeline has all required stages"""
        pipeline = create_uncertainty_pipeline()

        # Pipeline should have 3 stages
        stages_repr = repr(pipeline)
        assert 'UncertaintyValidator' in stages_repr
        assert 'UncertaintyTransformer' in stages_repr
        assert 'UncertaintyEnricher' in stages_repr

    def test_pipeline_execute_valid_data(self, valid_uncertainty_data):
        """Test pipeline execution with valid data"""
        pipeline = create_uncertainty_pipeline()

        result = pipeline.execute(valid_uncertainty_data)

        # Should have all fields from transformation and enrichment
        assert 'model_name' in result
        assert 'alphas' in result
        assert 'summary' in result
        assert 'top_features' in result

    def test_pipeline_execute_invalid_data_raises_error(self):
        """Test pipeline execution with invalid data raises ValueError"""
        pipeline = create_uncertainty_pipeline()

        with pytest.raises(ValueError):
            pipeline.execute({})  # Empty data should fail validation


# ==================== Integration Tests ====================


class TestIntegration:
    """Integration tests for complete workflows"""

    def test_full_pipeline_execution(self, valid_uncertainty_data):
        """Test complete pipeline execution from start to finish"""
        pipeline = create_uncertainty_pipeline()

        result = pipeline.execute(valid_uncertainty_data)

        # Verify transformation
        assert result['model_name'] == 'XGBoost'
        assert result['model_type'] == 'XGBRegressor'
        assert len(result['alphas']) == 3

        # Verify enrichment
        assert 'summary' in result
        assert result['summary']['total_alphas'] == 3
        assert 'uncertainty_score' in result['summary']
        assert len(result['top_features']) > 0

    def test_pipeline_with_minimal_data(self):
        """Test pipeline with minimal valid data"""
        minimal_data = {
            'test_results': {
                'primary_model': {
                    'crqr': {'alphas': {'0.1': {'coverage': 0.9}}}
                }
            },
            'initial_model_evaluation': {'feature_importance': {'f1': 0.5}},
        }

        pipeline = create_uncertainty_pipeline()
        result = pipeline.execute(minimal_data)

        assert result is not None
        assert 'summary' in result

    def test_pipeline_stages_execute_in_order(self, valid_uncertainty_data):
        """Test that pipeline stages execute in correct order"""
        pipeline = create_uncertainty_pipeline()

        # Execute pipeline
        result = pipeline.execute(valid_uncertainty_data)

        # Enrichment should have added summary (last stage)
        assert 'summary' in result
        # Transformation should have structured alphas (middle stage)
        assert isinstance(result['alphas'], list)
        # Validation passed (first stage, no error raised)
        assert True  # If we got here, validation passed


# ==================== Edge Cases ====================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_validator_with_nested_none_values(self, validator):
        """Test validator with None values in nested structures"""
        data = {
            'test_results': None,
            'initial_model_evaluation': None,
        }

        # Validator doesn't handle None gracefully, raises TypeError
        with pytest.raises(TypeError):
            validator.validate(data)

    def test_transformer_with_large_number_of_alphas(self, transformer):
        """Test transformer with many alpha levels"""
        alphas_dict = {
            f'0.{i:02d}': {'coverage': 0.9, 'expected_coverage': 0.9}
            for i in range(1, 100)
        }

        data = {
            'test_results': {'primary_model': {'crqr': {'alphas': alphas_dict}}},
            'initial_model_evaluation': {'feature_importance': {}},
        }

        result = transformer.transform(data)

        assert len(result['alphas']) == 99
        # Should be sorted
        assert result['alphas'][0]['alpha'] < result['alphas'][-1]['alpha']

    def test_enricher_with_negative_coverage_values(self, enricher):
        """Test enricher handles negative values gracefully"""
        data = {
            'alphas': [
                {
                    'coverage': -0.1,
                    'calibration_error': 0.5,
                    'avg_width': -1.0,
                }
            ],
            'feature_importance': {},
        }

        result = enricher.enrich(data)

        # Should calculate without errors
        assert 'summary' in result
        assert result['summary']['avg_coverage'] == -0.1

    def test_pipeline_with_unicode_feature_names(self):
        """Test pipeline with unicode characters in feature names"""
        data = {
            'test_results': {
                'primary_model': {'crqr': {'alphas': {'0.1': {}}}}
            },
            'initial_model_evaluation': {
                'feature_importance': {
                    'José': 0.5,
                    '北京': 0.3,
                    'café': 0.2,
                }
            },
        }

        pipeline = create_uncertainty_pipeline()
        result = pipeline.execute(data)

        assert len(result['feature_importance']) == 3
