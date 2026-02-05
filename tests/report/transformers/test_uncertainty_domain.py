"""
Tests for UncertaintyDomainTransformer (Phase 3 Sprint 10.2).

Demonstrates benefits of domain models vs Dict[str, Any].
"""

import pytest

from deepbridge.core.experiment.report.domain import UncertaintyReportData
from deepbridge.core.experiment.report.transformers.uncertainty_domain import (
    UncertaintyDomainTransformer,
)

# Sample test data
SAMPLE_UNCERTAINTY_RESULTS = {
    'primary_model': {
        'model_type': 'RandomForest',
        'timestamp': '2025-11-05T10:00:00',
        'dataset_size': 1000,
        'crqr': {
            'by_alpha': {
                'alpha_0.1': {
                    'overall_result': {
                        'coverage': 0.92,
                        'expected_coverage': 0.90,
                        'mean_width': 0.15
                    }
                },
                'alpha_0.2': {
                    'overall_result': {
                        'coverage': 0.82,
                        'expected_coverage': 0.80,
                        'mean_width': 0.12
                    }
                },
                'alpha_0.3': {
                    'overall_result': {
                        'coverage': 0.72,
                        'expected_coverage': 0.70,
                        'mean_width': 0.10
                    }
                },
            }
        }
    },
    'initial_model_evaluation': {
        'feature_importance': {
            'feature1': 0.8,
            'feature2': 0.6,
            'feature3': 0.4,
            'feature4': 0.2,
        }
    }
}


class TestUncertaintyDomainTransformer:
    """Tests for domain-model based transformer."""

    def test_transform_to_model_returns_report_data(self):
        """Test that transform_to_model returns UncertaintyReportData."""
        transformer = UncertaintyDomainTransformer()

        report = transformer.transform_to_model(
            SAMPLE_UNCERTAINTY_RESULTS,
            "TestModel"
        )

        # Verify type
        assert isinstance(report, UncertaintyReportData)

        # Verify basic fields
        assert report.model_name == "TestModel"
        assert report.model_type == "RandomForest"

    def test_transform_to_model_calculates_metrics(self):
        """Test that metrics are calculated correctly."""
        transformer = UncertaintyDomainTransformer()

        report = transformer.transform_to_model(
            SAMPLE_UNCERTAINTY_RESULTS,
            "TestModel"
        )

        # Verify metrics exist
        assert report.metrics.coverage > 0.0
        assert report.metrics.mean_width > 0.0
        assert report.metrics.uncertainty_score > 0.0

        # Calibration error should be auto-computed
        assert report.metrics.calibration_error >= 0.0

    def test_transform_to_model_creates_calibration_results(self):
        """Test that calibration results are created."""
        transformer = UncertaintyDomainTransformer()

        report = transformer.transform_to_model(
            SAMPLE_UNCERTAINTY_RESULTS,
            "TestModel"
        )

        # Should have calibration results
        assert report.has_calibration_results
        assert report.calibration_results is not None
        assert report.calibration_results.num_alpha_levels == 3

        # Check data
        assert len(report.calibration_results.alpha_values) == 3
        assert len(report.calibration_results.coverage_values) == 3

    def test_transform_to_model_extracts_features(self):
        """Test that features are extracted."""
        transformer = UncertaintyDomainTransformer()

        report = transformer.transform_to_model(
            SAMPLE_UNCERTAINTY_RESULTS,
            "TestModel"
        )

        # Should have features
        assert report.has_feature_importance
        assert len(report.features) == 4
        assert len(report.feature_importance) == 4

        # Check top features
        top_features = report.top_features
        assert len(top_features) == 4  # All 4 in this case
        assert top_features[0][0] == 'feature1'  # Highest importance
        assert top_features[0][1] == 0.8

    def test_transform_backward_compatible(self):
        """Test that transform() returns Dict for backward compatibility."""
        transformer = UncertaintyDomainTransformer()

        result = transformer.transform(
            SAMPLE_UNCERTAINTY_RESULTS,
            "TestModel"
        )

        # Verify it's a dict
        assert isinstance(result, dict)

        # Verify old structure is maintained
        assert 'model_name' in result
        assert 'summary' in result
        assert 'alphas' in result
        assert 'features' in result
        assert 'metadata' in result

        # Verify old .get() patterns still work
        score = result.get('uncertainty_score', 0.0)
        assert score > 0.0


class TestDomainModelBenefits:
    """Tests demonstrating benefits of domain models."""

    def test_type_safety_vs_dict(self):
        """Demonstrate type-safe access vs Dict."""
        transformer = UncertaintyDomainTransformer()

        # NEW WAY: Type-safe model
        report = transformer.transform_to_model(
            SAMPLE_UNCERTAINTY_RESULTS,
            "TestModel"
        )

        # Type-safe access (IDE autocomplete works!)
        score = report.metrics.uncertainty_score
        coverage = report.metrics.coverage
        has_calibration = report.has_calibration_results
        well_calibrated = report.is_well_calibrated

        # All values exist (no None checks needed)
        assert isinstance(score, float)
        assert isinstance(coverage, float)
        assert isinstance(has_calibration, bool)
        assert isinstance(well_calibrated, bool)

        # OLD WAY: Dict with .get() calls
        result_dict = transformer.transform(
            SAMPLE_UNCERTAINTY_RESULTS,
            "TestModel"
        )

        # Verbose .get() calls
        score_old = result_dict.get('uncertainty_score', 0.0)
        coverage_old = result_dict['summary'].get('avg_coverage', 0.0)

        # Same values, but with more boilerplate
        assert score == score_old
        assert coverage == coverage_old

    def test_eliminates_get_calls(self):
        """Demonstrate elimination of .get() calls."""
        transformer = UncertaintyDomainTransformer()

        report = transformer.transform_to_model(
            SAMPLE_UNCERTAINTY_RESULTS,
            "TestModel"
        )

        # Before (Dict): Multiple .get() calls
        # score = data.get('uncertainty_score', 0.0)
        # coverage = data.get('avg_coverage', 0.0)
        # width = data.get('avg_width', 0.0)
        # has_features = 'features' in data and isinstance(data['features'], dict)

        # After (Model): Direct access
        score = report.metrics.uncertainty_score
        coverage = report.metrics.coverage
        width = report.metrics.mean_width
        has_features = report.has_feature_importance

        # All values guaranteed to exist
        assert score is not None
        assert coverage is not None
        assert width is not None
        assert isinstance(has_features, bool)

    def test_property_convenience(self):
        """Demonstrate convenience properties."""
        transformer = UncertaintyDomainTransformer()

        report = transformer.transform_to_model(
            SAMPLE_UNCERTAINTY_RESULTS,
            "TestModel"
        )

        # Convenient properties instead of manual checks
        assert report.has_calibration_results  # vs checking if calibration_results is not None
        assert report.has_feature_importance  # vs checking if feature_importance dict is not empty
        assert isinstance(report.top_features, list)  # Already sorted and limited
        assert isinstance(report.get_summary_stats(), dict)  # Quick overview

        # is_well_calibrated property
        is_calibrated = report.is_well_calibrated
        assert isinstance(is_calibrated, bool)

        # Manual calculation would be:
        # is_calibrated_manual = report.metrics.calibration_error < 0.05
        # assert is_calibrated == is_calibrated_manual

    def test_validation_catches_errors(self):
        """Demonstrate automatic validation."""
        transformer = UncertaintyDomainTransformer()

        # Valid data works
        report = transformer.transform_to_model(
            SAMPLE_UNCERTAINTY_RESULTS,
            "TestModel"
        )
        assert report.metrics.coverage >= 0.0
        assert report.metrics.coverage <= 1.0

        # Domain model ensures valid ranges
        # (can't set coverage > 1.0 due to Field validation)

    def test_str_representation(self):
        """Test human-readable string representation."""
        transformer = UncertaintyDomainTransformer()

        report = transformer.transform_to_model(
            SAMPLE_UNCERTAINTY_RESULTS,
            "TestModel"
        )

        # Get string representation
        str_repr = str(report)

        # Should include key info
        assert "TestModel" in str_repr
        assert "score=" in str_repr
        assert "coverage=" in str_repr


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_results(self):
        """Test handling of empty results."""
        transformer = UncertaintyDomainTransformer()

        # Empty results
        empty_results = {'primary_model': {}, 'initial_model_evaluation': {}}

        report = transformer.transform_to_model(empty_results, "EmptyModel")

        # Should create valid model with defaults
        assert report.model_name == "EmptyModel"
        # Empty data gives score of 1.0 (no error because no data)
        assert report.metrics.uncertainty_score == 1.0
        assert not report.has_calibration_results

    def test_missing_feature_importance(self):
        """Test handling when feature importance is missing."""
        transformer = UncertaintyDomainTransformer()

        results_no_features = {
            'primary_model': SAMPLE_UNCERTAINTY_RESULTS['primary_model'],
            'initial_model_evaluation': {}  # No features
        }

        report = transformer.transform_to_model(
            results_no_features,
            "NoFeaturesModel"
        )

        # Should handle gracefully
        assert not report.has_feature_importance
        assert len(report.features) == 0
        assert len(report.feature_importance) == 0

    def test_alternative_result_format(self):
        """Test handling of alternative result format (with test_results wrapper)."""
        transformer = UncertaintyDomainTransformer()

        # Wrapped format
        wrapped_results = {
            'test_results': SAMPLE_UNCERTAINTY_RESULTS
        }

        report = transformer.transform_to_model(
            wrapped_results,
            "WrappedModel"
        )

        # Should work the same
        assert report.model_name == "WrappedModel"
        assert report.has_calibration_results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
