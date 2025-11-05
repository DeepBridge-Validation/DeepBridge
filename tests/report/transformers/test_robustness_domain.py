"""
Tests for RobustnessDomainTransformer (Phase 3 Sprint 10.3).

Demonstrates benefits of domain models vs Dict[str, Any].
"""

import pytest
from deepbridge.core.experiment.report.transformers.robustness_domain import (
    RobustnessDomainTransformer
)
from deepbridge.core.experiment.report.domain import RobustnessReportData


# Sample test data
SAMPLE_ROBUSTNESS_RESULTS = {
    'primary_model': {
        'model_type': 'RandomForest',
        'base_score': 0.85,
        'robustness_score': 0.75,
        'avg_raw_impact': 0.10,
        'avg_quantile_impact': 0.12,
        'metric': 'AUC',
        'n_iterations': 20,
        'feature_importance': {
            'feature1': 0.3,
            'feature2': 0.2,
            'feature3': 0.1,
        },
        'raw': {
            'by_level': {
                '0.1': {
                    'overall_result': {
                        'all_features': {
                            'mean_score': 0.80,
                            'std_score': 0.03,
                            'impact': 0.05,
                            'worst_score': 0.75
                        }
                    }
                },
                '0.5': {
                    'overall_result': {
                        'all_features': {
                            'mean_score': 0.70,
                            'std_score': 0.05,
                            'impact': 0.15,
                            'worst_score': 0.60
                        }
                    }
                },
                '1.0': {
                    'overall_result': {
                        'all_features': {
                            'mean_score': 0.60,
                            'std_score': 0.08,
                            'impact': 0.25,
                            'worst_score': 0.45
                        }
                    }
                },
            }
        }
    },
    'initial_model_evaluation': {
        'models': {
            'primary_model': {
                'feature_importance': {
                    'feature1': 0.8,
                    'feature2': 0.6,
                    'feature3': 0.4,
                    'feature4': 0.2,
                }
            }
        }
    }
}


class TestRobustnessDomainTransformer:
    """Tests for domain-model based transformer."""

    def test_transform_to_model_returns_report_data(self):
        """Test that transform_to_model returns RobustnessReportData."""
        transformer = RobustnessDomainTransformer()

        report = transformer.transform_to_model(
            SAMPLE_ROBUSTNESS_RESULTS,
            "TestModel"
        )

        # Verify type
        assert isinstance(report, RobustnessReportData)

        # Verify basic fields
        assert report.model_name == "TestModel"
        assert report.model_type == "RandomForest"

    def test_transform_to_model_calculates_metrics(self):
        """Test that metrics are calculated correctly."""
        transformer = RobustnessDomainTransformer()

        report = transformer.transform_to_model(
            SAMPLE_ROBUSTNESS_RESULTS,
            "TestModel"
        )

        # Verify metrics exist
        assert report.metrics.base_score == 0.85
        assert report.metrics.robustness_score == 0.75
        assert report.metrics.avg_raw_impact == 0.10
        assert report.metrics.avg_quantile_impact == 0.12
        assert report.metrics.avg_overall_impact > 0.0
        assert report.metrics.metric == "AUC"

    def test_transform_to_model_creates_perturbation_levels(self):
        """Test that perturbation levels are created."""
        transformer = RobustnessDomainTransformer()

        report = transformer.transform_to_model(
            SAMPLE_ROBUSTNESS_RESULTS,
            "TestModel"
        )

        # Should have 3 perturbation levels
        assert report.has_perturbation_data
        assert report.num_perturbation_levels == 3

        # Check data
        levels = report.perturbation_levels
        assert levels[0].level == 0.1
        assert levels[1].level == 0.5
        assert levels[2].level == 1.0

        # Check impacts increase with level
        assert levels[0].impact < levels[1].impact < levels[2].impact

    def test_transform_to_model_extracts_features(self):
        """Test that features are extracted."""
        transformer = RobustnessDomainTransformer()

        report = transformer.transform_to_model(
            SAMPLE_ROBUSTNESS_RESULTS,
            "TestModel"
        )

        # Should have features
        assert report.has_feature_data
        assert report.num_features == 4

        # Check features are sorted by importance
        features = report.features
        assert features[0].name == 'feature1'  # Highest importance (0.8)
        assert features[0].importance == 0.8
        assert features[0].robustness_impact == 0.3  # From primary_model

    def test_transform_backward_compatible(self):
        """Test that transform() returns Dict for backward compatibility."""
        transformer = RobustnessDomainTransformer()

        result = transformer.transform(
            SAMPLE_ROBUSTNESS_RESULTS,
            "TestModel"
        )

        # Verify it's a dict
        assert isinstance(result, dict)

        # Verify old structure is maintained
        assert 'model_name' in result
        assert 'summary' in result
        assert 'levels' in result
        assert 'features' in result
        assert 'metadata' in result

        # Verify old .get() patterns still work
        score = result.get('robustness_score', 0.0)
        assert score == 0.75

    def test_properties_work(self):
        """Test that properties provide useful information."""
        transformer = RobustnessDomainTransformer()

        report = transformer.transform_to_model(
            SAMPLE_ROBUSTNESS_RESULTS,
            "TestModel"
        )

        # Test is_robust property
        assert report.metrics.is_robust is True  # Score 0.75 > 0.7

        # Test worst_perturbation_level
        worst = report.worst_perturbation_level
        assert worst is not None
        assert worst.level == 1.0  # Highest impact

        # Test top_features
        top = report.top_features
        assert len(top) == 4  # All 4 in this case
        assert top[0].name == 'feature1'

        # Test most_sensitive_features
        sensitive = report.most_sensitive_features
        assert len(sensitive) > 0


class TestDomainModelBenefits:
    """Tests demonstrating benefits of domain models."""

    def test_type_safety_vs_dict(self):
        """Demonstrate type-safe access vs Dict."""
        transformer = RobustnessDomainTransformer()

        # NEW WAY: Type-safe model
        report = transformer.transform_to_model(
            SAMPLE_ROBUSTNESS_RESULTS,
            "TestModel"
        )

        # Type-safe access (IDE autocomplete works!)
        score = report.metrics.robustness_score
        base = report.metrics.base_score
        is_robust = report.metrics.is_robust
        has_data = report.has_perturbation_data

        # All values exist (no None checks needed)
        assert isinstance(score, float)
        assert isinstance(base, float)
        assert isinstance(is_robust, bool)
        assert isinstance(has_data, bool)

        # OLD WAY: Dict with .get() calls
        result_dict = transformer.transform(
            SAMPLE_ROBUSTNESS_RESULTS,
            "TestModel"
        )

        # Verbose .get() calls
        score_old = result_dict.get('robustness_score', 0.0)
        base_old = result_dict['summary'].get('base_score', 0.0)

        # Same values, but with more boilerplate
        assert score == score_old
        assert base == base_old

    def test_eliminates_get_calls(self):
        """Demonstrate elimination of .get() calls."""
        transformer = RobustnessDomainTransformer()

        report = transformer.transform_to_model(
            SAMPLE_ROBUSTNESS_RESULTS,
            "TestModel"
        )

        # Before (Dict): Multiple .get() calls
        # score = data.get('robustness_score', 0.0)
        # base = data.get('base_score', 0.0)
        # impact = data.get('avg_overall_impact', 0.0)
        # has_levels = 'levels' in data and isinstance(data['levels'], list)

        # After (Model): Direct access
        score = report.metrics.robustness_score
        base = report.metrics.base_score
        impact = report.metrics.avg_overall_impact
        has_levels = report.has_perturbation_data

        # All values guaranteed to exist
        assert score is not None
        assert base is not None
        assert impact is not None
        assert isinstance(has_levels, bool)

    def test_property_convenience(self):
        """Demonstrate convenience properties."""
        transformer = RobustnessDomainTransformer()

        report = transformer.transform_to_model(
            SAMPLE_ROBUSTNESS_RESULTS,
            "TestModel"
        )

        # Convenient properties instead of manual checks
        assert report.has_perturbation_data
        assert report.has_feature_data
        assert isinstance(report.top_features, list)
        assert isinstance(report.get_summary_stats(), dict)

        # is_robust property
        is_robust = report.metrics.is_robust
        assert isinstance(is_robust, bool)

        # worst_perturbation_level
        worst = report.worst_perturbation_level
        assert worst is not None

    def test_str_representation(self):
        """Test human-readable string representation."""
        transformer = RobustnessDomainTransformer()

        report = transformer.transform_to_model(
            SAMPLE_ROBUSTNESS_RESULTS,
            "TestModel"
        )

        # Get string representation
        str_repr = str(report)

        # Should include key info
        assert "TestModel" in str_repr
        assert "score=" in str_repr
        assert "levels=" in str_repr


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_results(self):
        """Test handling of empty results."""
        transformer = RobustnessDomainTransformer()

        # Empty results
        empty_results = {'primary_model': {}, 'initial_model_evaluation': {}}

        report = transformer.transform_to_model(empty_results, "EmptyModel")

        # Should create valid model with defaults
        assert report.model_name == "EmptyModel"
        assert report.metrics.robustness_score == 0.0
        assert not report.has_perturbation_data

    def test_missing_feature_importance(self):
        """Test handling when feature importance is missing."""
        transformer = RobustnessDomainTransformer()

        results_no_features = {
            'primary_model': SAMPLE_ROBUSTNESS_RESULTS['primary_model'].copy(),
            'initial_model_evaluation': {}  # No features
        }
        results_no_features['primary_model']['feature_importance'] = {}

        report = transformer.transform_to_model(
            results_no_features,
            "NoFeaturesModel"
        )

        # Should handle gracefully
        assert not report.has_feature_data
        assert report.num_features == 0

    def test_alternative_result_format(self):
        """Test handling of alternative result format (with test_results wrapper)."""
        transformer = RobustnessDomainTransformer()

        # Wrapped format
        wrapped_results = {
            'test_results': SAMPLE_ROBUSTNESS_RESULTS
        }

        report = transformer.transform_to_model(
            wrapped_results,
            "WrappedModel"
        )

        # Should work the same
        assert report.model_name == "WrappedModel"
        assert report.has_perturbation_data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
