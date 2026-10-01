"""
Tests for ResilienceDomainTransformer (Phase 3 Sprint 10.4).

Demonstrates benefits of domain models for complex multi-test resilience data.
"""

import numpy as np
import pytest

from deepbridge.core.experiment.report.domain import ResilienceReportData
from deepbridge.core.experiment.report.transformers.resilience_domain import (
    ResilienceDomainTransformer,
)

# Sample test data with multiple test types
SAMPLE_RESILIENCE_RESULTS = {
    'primary_model': {
        'model_type': 'RandomForest',
        'resilience_score': 0.85,
        'metrics': {'accuracy': 0.90},
        'test_scores': {
            'distribution_shift': 0.88,
            'worst_sample': 0.82,
            'worst_cluster': 0.85,
        },
        'distance_metrics': ['euclidean', 'manhattan'],
        'alphas': [0.1, 0.2, 0.3],
        # Distribution shift results
        'distribution_shift': {
            'all_results': [
                {
                    'name': 'Scenario 1',
                    'alpha': 0.1,
                    'distance_metric': 'euclidean',
                    'metric': 'accuracy',
                    'performance_gap': 0.10,
                    'worst_metric': 0.80,
                    'remaining_metric': 0.90,
                },
                {
                    'name': 'Scenario 2',
                    'alpha': 0.2,
                    'distance_metric': 'manhattan',
                    'metric': 'accuracy',
                    'performance_gap': 0.15,
                    'worst_metric': 0.75,
                    'remaining_metric': 0.90,
                },
            ]
        },
        # Worst sample results
        'worst_sample': {
            'all_results': [
                {
                    'alpha': 0.2,
                    'ranking_method': 'loss',
                    'metric': 'accuracy',
                    'performance_gap': 0.12,
                    'worst_metric': 0.75,
                    'remaining_metric': 0.87,
                    'n_worst_samples': 100,
                    'n_remaining_samples': 400,
                }
            ]
        },
        # Worst cluster results
        'worst_cluster': {
            'all_results': [
                {
                    'n_clusters': 5,
                    'worst_cluster_id': 2,
                    'metric': 'accuracy',
                    'performance_gap': 0.18,
                    'worst_cluster_metric': 0.70,
                    'remaining_metric': 0.88,
                    'worst_cluster_size': 50,
                    'remaining_size': 450,
                    'feature_contributions': {
                        'feature1': 0.5,
                        'feature2': 0.3,
                        'feature3': 0.2,
                    },
                }
            ]
        },
        # Outer sample results
        'outer_sample': {
            'all_results': [
                {
                    'alpha': 0.1,
                    'outlier_method': 'isolation_forest',
                    'metric': 'accuracy',
                    'performance_gap': 0.20,
                    'outer_metric': 0.65,
                    'inner_metric': 0.85,
                    'n_outer_samples': 50,
                    'n_inner_samples': 450,
                }
            ]
        },
        # Hard sample results
        'hard_sample': {
            'all_results': [
                {
                    'disagreement_threshold': 0.3,
                    'metric': 'accuracy',
                    'performance_gap': 0.15,
                    'hard_metric': 0.70,
                    'easy_metric': 0.85,
                    'n_hard_samples': 80,
                    'n_easy_samples': 420,
                    'model_disagreements': {
                        'model_A-model_B': 0.4,
                        'model_A-model_C': 0.35,
                    },
                }
            ]
        },
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
    },
}


class TestResilienceDomainTransformer:
    """Tests for domain-model based transformer."""

    def test_transform_to_model_returns_report_data(self):
        """Test that transform_to_model returns ResilienceReportData."""
        transformer = ResilienceDomainTransformer()

        report = transformer.transform_to_model(
            SAMPLE_RESILIENCE_RESULTS, 'TestModel'
        )

        # Verify type
        assert isinstance(report, ResilienceReportData)

        # Verify basic fields
        assert report.model_name == 'TestModel'
        assert report.model_type == 'RandomForest'

    def test_transform_to_model_calculates_metrics(self):
        """Test that metrics are calculated correctly."""
        transformer = ResilienceDomainTransformer()

        report = transformer.transform_to_model(
            SAMPLE_RESILIENCE_RESULTS, 'TestModel'
        )

        # Verify metrics exist
        assert report.metrics.resilience_score == 0.85
        assert report.metrics.total_scenarios > 0
        assert report.metrics.valid_scenarios > 0
        assert report.metrics.avg_performance_gap > 0.0

    def test_transform_to_model_creates_distribution_shift(self):
        """Test that distribution shift scenarios are created."""
        transformer = ResilienceDomainTransformer()

        report = transformer.transform_to_model(
            SAMPLE_RESILIENCE_RESULTS, 'TestModel'
        )

        # Should have distribution shift data
        assert report.has_distribution_shift
        assert len(report.distribution_shift_scenarios) == 2

        # Check first scenario
        scenario = report.distribution_shift_scenarios[0]
        assert scenario.name == 'Scenario 1'
        assert scenario.alpha == 0.1
        assert scenario.distance_metric == 'euclidean'
        assert scenario.performance_gap == 0.10

    def test_transform_to_model_creates_worst_sample(self):
        """Test that worst-sample tests are created."""
        transformer = ResilienceDomainTransformer()

        report = transformer.transform_to_model(
            SAMPLE_RESILIENCE_RESULTS, 'TestModel'
        )

        # Should have worst-sample data
        assert report.has_worst_sample
        assert len(report.worst_sample_tests) == 1

        # Check test data
        test = report.worst_sample_tests[0]
        assert test.alpha == 0.2
        assert test.ranking_method == 'loss'
        assert test.performance_gap == 0.12

    def test_transform_to_model_creates_worst_cluster(self):
        """Test that worst-cluster tests are created."""
        transformer = ResilienceDomainTransformer()

        report = transformer.transform_to_model(
            SAMPLE_RESILIENCE_RESULTS, 'TestModel'
        )

        # Should have worst-cluster data
        assert report.has_worst_cluster
        assert len(report.worst_cluster_tests) == 1

        # Check test data
        test = report.worst_cluster_tests[0]
        assert test.n_clusters == 5
        assert test.worst_cluster_id == 2
        assert test.performance_gap == 0.18
        assert len(test.top_features) == 3

    def test_transform_to_model_creates_outer_sample(self):
        """Test that outer-sample tests are created."""
        transformer = ResilienceDomainTransformer()

        report = transformer.transform_to_model(
            SAMPLE_RESILIENCE_RESULTS, 'TestModel'
        )

        # Should have outer-sample data
        assert report.has_outer_sample
        assert len(report.outer_sample_tests) == 1

        # Check test data
        test = report.outer_sample_tests[0]
        assert test.alpha == 0.1
        assert test.outlier_method == 'isolation_forest'
        assert test.performance_gap == 0.20

    def test_transform_to_model_creates_hard_sample(self):
        """Test that hard-sample tests are created."""
        transformer = ResilienceDomainTransformer()

        report = transformer.transform_to_model(
            SAMPLE_RESILIENCE_RESULTS, 'TestModel'
        )

        # Should have hard-sample data
        assert report.has_hard_sample
        assert len(report.hard_sample_tests) == 1

        # Check test data
        test = report.hard_sample_tests[0]
        assert test.skipped is False
        assert test.disagreement_threshold == 0.3
        assert test.performance_gap == 0.15
        assert len(test.model_disagreements) == 2

    def test_transform_to_model_extracts_features(self):
        """Test that features are extracted."""
        transformer = ResilienceDomainTransformer()

        report = transformer.transform_to_model(
            SAMPLE_RESILIENCE_RESULTS, 'TestModel'
        )

        # Should have features
        assert report.has_feature_importance
        assert len(report.features) == 4

        # Check top features
        top_features = report.top_features
        assert len(top_features) == 4  # All 4 in this case
        assert top_features[0][0] == 'feature1'  # Highest importance
        assert top_features[0][1] == 0.8

    def test_transform_backward_compatible(self):
        """Test that transform() returns Dict for backward compatibility."""
        transformer = ResilienceDomainTransformer()

        result = transformer.transform(SAMPLE_RESILIENCE_RESULTS, 'TestModel')

        # Verify it's a dict
        assert isinstance(result, dict)

        # Verify old structure is maintained
        assert 'model_name' in result
        assert 'summary' in result
        assert 'distribution_shift' in result
        assert 'worst_sample' in result
        assert 'features' in result
        assert 'metadata' in result

        # Verify old .get() patterns still work
        score = result.get('resilience_score', 1.0)
        assert score == 0.85

    def test_properties_work(self):
        """Test that properties provide useful information."""
        transformer = ResilienceDomainTransformer()

        report = transformer.transform_to_model(
            SAMPLE_RESILIENCE_RESULTS, 'TestModel'
        )

        # Test is_resilient property
        assert report.metrics.is_resilient is True

        # Test available_test_types
        types = report.available_test_types
        assert 'distribution_shift' in types
        assert 'worst_sample' in types
        assert len(types) == 5  # All 5 test types

        # Test worst/best test types
        assert report.worst_test_type is not None
        assert report.best_test_type is not None


class TestDomainModelBenefits:
    """Tests demonstrating benefits of domain models."""

    def test_type_safety_vs_dict(self):
        """Demonstrate type-safe access vs Dict."""
        transformer = ResilienceDomainTransformer()

        # NEW WAY: Type-safe model
        report = transformer.transform_to_model(
            SAMPLE_RESILIENCE_RESULTS, 'TestModel'
        )

        # Type-safe access (IDE autocomplete works!)
        score = report.metrics.resilience_score
        gap = report.metrics.avg_performance_gap
        is_resilient = report.metrics.is_resilient
        has_tests = report.has_distribution_shift

        # All values exist (no None checks needed)
        assert isinstance(score, float)
        assert isinstance(gap, float)
        assert isinstance(is_resilient, bool)
        assert isinstance(has_tests, bool)

        # OLD WAY: Dict with .get() calls
        result_dict = transformer.transform(
            SAMPLE_RESILIENCE_RESULTS, 'TestModel'
        )

        # Verbose .get() calls
        score_old = result_dict.get('resilience_score', 1.0)
        gap_old = result_dict['summary'].get('avg_performance_gap', 0.0)

        # Same values, but with more boilerplate
        assert score == score_old
        assert gap == gap_old

    def test_eliminates_get_calls(self):
        """Demonstrate elimination of .get() calls."""
        transformer = ResilienceDomainTransformer()

        report = transformer.transform_to_model(
            SAMPLE_RESILIENCE_RESULTS, 'TestModel'
        )

        # Before (Dict): Multiple .get() calls with nested dicts
        # score = data.get('resilience_score', 1.0)
        # gap = data.get('summary', {}).get('avg_performance_gap', 0.0)
        # scenarios = data.get('distribution_shift', [])
        # has_tests = 'distribution_shift' in data and len(data['distribution_shift']) > 0

        # After (Model): Direct access
        score = report.metrics.resilience_score
        gap = report.metrics.avg_performance_gap
        scenarios = report.distribution_shift_scenarios
        has_tests = report.has_distribution_shift

        # All values guaranteed to exist
        assert score is not None
        assert gap is not None
        assert scenarios is not None
        assert isinstance(has_tests, bool)

    def test_property_convenience(self):
        """Demonstrate convenience properties."""
        transformer = ResilienceDomainTransformer()

        report = transformer.transform_to_model(
            SAMPLE_RESILIENCE_RESULTS, 'TestModel'
        )

        # Convenient properties instead of manual checks
        assert report.has_distribution_shift
        assert report.has_feature_importance
        assert isinstance(report.top_features, list)
        assert isinstance(report.get_summary_stats(), dict)
        assert isinstance(report.available_test_types, list)

        # Test-specific summaries
        summary = report.get_test_type_summary('distribution_shift')
        assert summary.total_tests > 0
        assert isinstance(summary, object)  # Type-safe result

    def test_complex_nested_access(self):
        """Demonstrate simplified access to complex nested data."""
        transformer = ResilienceDomainTransformer()

        report = transformer.transform_to_model(
            SAMPLE_RESILIENCE_RESULTS, 'TestModel'
        )

        # Direct access to nested data (no .get() chains!)
        first_scenario = report.distribution_shift_scenarios[0]
        assert first_scenario.name == 'Scenario 1'
        assert first_scenario.has_significant_gap is False  # 0.10 < 0.2

        # Access worst-cluster top features
        cluster_test = report.worst_cluster_tests[0]
        top_features = cluster_test.top_features
        assert len(top_features) == 3
        assert top_features[0]['name'] == 'feature1'

    def test_str_representation(self):
        """Test human-readable string representation."""
        transformer = ResilienceDomainTransformer()

        report = transformer.transform_to_model(
            SAMPLE_RESILIENCE_RESULTS, 'TestModel'
        )

        # Get string representation
        str_repr = str(report)

        # Should include key info
        assert 'TestModel' in str_repr
        assert 'score=' in str_repr
        assert 'tests=' in str_repr


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_results(self):
        """Test handling of empty results."""
        transformer = ResilienceDomainTransformer()

        # Empty results
        empty_results = {'primary_model': {}, 'initial_model_evaluation': {}}

        report = transformer.transform_to_model(empty_results, 'EmptyModel')

        # Should create valid model with defaults
        assert report.model_name == 'EmptyModel'
        assert report.metrics.resilience_score == 1.0
        assert not report.has_distribution_shift

    def test_missing_feature_importance(self):
        """Test handling when feature importance is missing."""
        transformer = ResilienceDomainTransformer()

        results_no_features = {
            'primary_model': SAMPLE_RESILIENCE_RESULTS['primary_model'],
            'initial_model_evaluation': {},  # No features
        }

        report = transformer.transform_to_model(
            results_no_features, 'NoFeaturesModel'
        )

        # Should handle gracefully
        assert not report.has_feature_importance
        assert len(report.features) == 0

    def test_nan_values_handled(self):
        """Test that NaN values are handled correctly."""
        transformer = ResilienceDomainTransformer()

        results_with_nan = {
            'primary_model': {
                'model_type': 'RF',
                'resilience_score': 0.85,
                'distribution_shift': {
                    'all_results': [
                        {
                            'name': 'Scenario 1',
                            'alpha': 0.1,
                            'distance_metric': 'euclidean',
                            'metric': 'accuracy',
                            'performance_gap': float('nan'),  # NaN!
                            'worst_metric': 0.80,
                            'remaining_metric': 0.90,
                        }
                    ]
                },
            },
            'initial_model_evaluation': {},
        }

        report = transformer.transform_to_model(results_with_nan, 'NaNModel')

        # Should handle NaN gracefully
        scenario = report.distribution_shift_scenarios[0]
        assert scenario.performance_gap is None  # Converted to None
        assert scenario.is_valid is False

    def test_skipped_hard_sample_test(self):
        """Test handling of skipped hard-sample test."""
        transformer = ResilienceDomainTransformer()

        results_skipped = {
            'primary_model': {
                'model_type': 'RF',
                'resilience_score': 0.85,
                'hard_sample': {
                    'all_results': [
                        {
                            # No hard_metric/easy_metric = skipped
                            'hard_metric': None,
                            'easy_metric': None,
                        }
                    ]
                },
            },
            'initial_model_evaluation': {},
        }

        report = transformer.transform_to_model(
            results_skipped, 'SkippedModel'
        )

        # Should handle skipped test
        assert report.has_hard_sample
        test = report.hard_sample_tests[0]
        assert test.skipped is True
        assert test.reason == 'No alternative models available'

    def test_alternative_result_format(self):
        """Test handling of alternative result format (with test_results wrapper)."""
        transformer = ResilienceDomainTransformer()

        # Wrapped format
        wrapped_results = {'test_results': SAMPLE_RESILIENCE_RESULTS}

        report = transformer.transform_to_model(
            wrapped_results, 'WrappedModel'
        )

        # Should work the same
        assert report.model_name == 'WrappedModel'
        assert report.has_distribution_shift


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
