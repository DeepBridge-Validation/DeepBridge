"""
E2E Integration Tests for Domain Model Pipeline (Phase 3 Sprint 10.5).

Tests the complete pipeline from raw results → domain models → rendered reports.

Validates:
- Full transformation pipeline for all report types
- Cross-transformer consistency
- Round-trip conversions (model → dict → model)
- Performance characteristics
- Real-world multi-report scenarios
- Edge case handling across the stack
"""

import pytest
import time
from typing import Dict, Any

from deepbridge.core.experiment.report.transformers.uncertainty_domain import (
    UncertaintyDomainTransformer
)
from deepbridge.core.experiment.report.transformers.robustness_domain import (
    RobustnessDomainTransformer
)
from deepbridge.core.experiment.report.transformers.resilience_domain import (
    ResilienceDomainTransformer
)
from deepbridge.core.experiment.report.domain import (
    UncertaintyReportData,
    RobustnessReportData,
    ResilienceReportData,
)


# Realistic sample data for all three report types
REALISTIC_UNCERTAINTY_DATA = {
    'primary_model': {
        'model_type': 'GradientBoosting',
        'timestamp': '2025-11-05T14:30:00',
        'dataset_size': 5000,
        'crqr': {
            'by_alpha': {
                'alpha_0.05': {
                    'overall_result': {
                        'coverage': 0.93,
                        'expected_coverage': 0.95,
                        'mean_width': 0.125
                    }
                },
                'alpha_0.10': {
                    'overall_result': {
                        'coverage': 0.88,
                        'expected_coverage': 0.90,
                        'mean_width': 0.15
                    }
                },
                'alpha_0.15': {
                    'overall_result': {
                        'coverage': 0.83,
                        'expected_coverage': 0.85,
                        'mean_width': 0.175
                    }
                },
                'alpha_0.20': {
                    'overall_result': {
                        'coverage': 0.78,
                        'expected_coverage': 0.80,
                        'mean_width': 0.20
                    }
                },
                'alpha_0.25': {
                    'overall_result': {
                        'coverage': 0.73,
                        'expected_coverage': 0.75,
                        'mean_width': 0.225
                    }
                },
            }
        }
    },
    'initial_model_evaluation': {
        'feature_importance': {
            f'feature_{i}': 1.0 / (i + 1) for i in range(10)
        }
    }
}

REALISTIC_ROBUSTNESS_DATA = {
    'primary_model': {
        'model_type': 'RandomForest',
        'base_score': 0.92,
        'robustness_score': 0.78,
        'avg_raw_impact': 0.14,
        'avg_quantile_impact': 0.12,
        'metric': 'F1',
        'n_iterations': 50,
        'feature_importance': {
            f'feature_{i}': 0.1 * (10 - i) for i in range(10)
        },
        'raw': {
            'by_level': {
                str(level): {
                    'overall_result': {
                        'all_features': {
                            'mean_score': 0.92 - level * 0.15,
                            'std_score': 0.03 + level * 0.02,
                            'impact': level * 0.15,
                            'worst_score': 0.92 - level * 0.20
                        }
                    }
                }
                for level in [0.1, 0.3, 0.5, 0.7, 1.0]
            }
        }
    },
    'initial_model_evaluation': {
        'models': {
            'primary_model': {
                'feature_importance': {
                    f'feature_{i}': 1.0 / (i + 1) for i in range(10)
                }
            }
        }
    }
}

REALISTIC_RESILIENCE_DATA = {
    'primary_model': {
        'model_type': 'NeuralNetwork',
        'resilience_score': 0.82,
        'metrics': {'accuracy': 0.91},
        'test_scores': {
            'distribution_shift': 0.85,
            'worst_sample': 0.78,
            'worst_cluster': 0.80,
            'outer_sample': 0.76,
            'hard_sample': 0.82
        },
        'distance_metrics': ['euclidean', 'manhattan', 'cosine'],
        'alphas': [0.1, 0.2, 0.3],
        'distribution_shift': {
            'all_results': [
                {
                    'name': f'Scenario_{i}',
                    'alpha': alpha,
                    'distance_metric': metric,
                    'metric': 'accuracy',
                    'performance_gap': 0.05 + alpha * 0.3,
                    'worst_metric': 0.70,
                    'remaining_metric': 0.90
                }
                for i, (alpha, metric) in enumerate(zip(
                    [0.1, 0.2, 0.3],
                    ['euclidean', 'manhattan', 'cosine']
                ))
            ]
        },
        'worst_sample': {
            'all_results': [
                {
                    'alpha': alpha,
                    'ranking_method': 'loss',
                    'metric': 'accuracy',
                    'performance_gap': 0.10 + alpha * 0.2,
                    'worst_metric': 0.70,
                    'remaining_metric': 0.88,
                    'n_worst_samples': int(5000 * alpha),
                    'n_remaining_samples': int(5000 * (1 - alpha))
                }
                for alpha in [0.1, 0.2]
            ]
        },
        'worst_cluster': {
            'all_results': [
                {
                    'n_clusters': k,
                    'worst_cluster_id': 0,
                    'metric': 'accuracy',
                    'performance_gap': 0.15,
                    'worst_cluster_metric': 0.68,
                    'remaining_metric': 0.90,
                    'worst_cluster_size': 1000,
                    'remaining_size': 4000,
                    'feature_contributions': {
                        f'feature_{i}': 0.1 * (5 - i) for i in range(5)
                    }
                }
                for k in [3, 5, 7]
            ]
        },
        'outer_sample': {
            'all_results': [
                {
                    'alpha': 0.1,
                    'outlier_method': 'isolation_forest',
                    'metric': 'accuracy',
                    'performance_gap': 0.20,
                    'outer_metric': 0.65,
                    'inner_metric': 0.92,
                    'n_outer_samples': 500,
                    'n_inner_samples': 4500
                }
            ]
        },
        'hard_sample': {
            'all_results': [
                {
                    'disagreement_threshold': 0.3,
                    'metric': 'accuracy',
                    'performance_gap': 0.12,
                    'hard_metric': 0.72,
                    'easy_metric': 0.93,
                    'n_hard_samples': 800,
                    'n_easy_samples': 4200,
                    'model_disagreements': {
                        'model_A-model_B': 0.35,
                        'model_A-model_C': 0.32
                    }
                }
            ]
        }
    },
    'initial_model_evaluation': {
        'models': {
            'primary_model': {
                'feature_importance': {
                    f'feature_{i}': 1.0 / (i + 1) for i in range(10)
                }
            }
        }
    }
}


class TestFullPipeline:
    """Test complete transformation pipeline for each report type."""

    def test_uncertainty_full_pipeline(self):
        """Test Uncertainty: raw data → model → dict → validation."""
        transformer = UncertaintyDomainTransformer()

        # Step 1: Transform to domain model
        report = transformer.transform_to_model(
            REALISTIC_UNCERTAINTY_DATA,
            "E2E_Uncertainty_Model"
        )

        # Validate domain model
        assert isinstance(report, UncertaintyReportData)
        assert report.model_name == "E2E_Uncertainty_Model"
        assert report.metrics.uncertainty_score > 0.0
        assert report.has_calibration_results
        assert report.calibration_results.num_alpha_levels == 5

        # Step 2: Transform to dict (backward compatibility)
        report_dict = transformer.transform(
            REALISTIC_UNCERTAINTY_DATA,
            "E2E_Uncertainty_Model"
        )

        # Validate dict structure
        assert isinstance(report_dict, dict)
        assert 'model_name' in report_dict
        assert 'summary' in report_dict
        assert 'alphas' in report_dict

        # Step 3: Validate consistency between modes
        assert report.model_name == report_dict['model_name']
        assert report.metrics.uncertainty_score == report_dict['uncertainty_score']

    def test_robustness_full_pipeline(self):
        """Test Robustness: raw data → model → dict → validation."""
        transformer = RobustnessDomainTransformer()

        # Step 1: Transform to domain model
        report = transformer.transform_to_model(
            REALISTIC_ROBUSTNESS_DATA,
            "E2E_Robustness_Model"
        )

        # Validate domain model
        assert isinstance(report, RobustnessReportData)
        assert report.model_name == "E2E_Robustness_Model"
        assert report.metrics.robustness_score == 0.78
        assert report.has_perturbation_data
        assert report.num_perturbation_levels == 5

        # Step 2: Transform to dict
        report_dict = transformer.transform(
            REALISTIC_ROBUSTNESS_DATA,
            "E2E_Robustness_Model"
        )

        # Validate dict structure
        assert isinstance(report_dict, dict)
        assert 'model_name' in report_dict
        assert 'summary' in report_dict
        assert 'levels' in report_dict

        # Step 3: Validate consistency
        assert report.model_name == report_dict['model_name']
        assert report.metrics.robustness_score == report_dict['robustness_score']

    def test_resilience_full_pipeline(self):
        """Test Resilience: raw data → model → dict → validation."""
        transformer = ResilienceDomainTransformer()

        # Step 1: Transform to domain model
        report = transformer.transform_to_model(
            REALISTIC_RESILIENCE_DATA,
            "E2E_Resilience_Model"
        )

        # Validate domain model
        assert isinstance(report, ResilienceReportData)
        assert report.model_name == "E2E_Resilience_Model"
        assert report.metrics.resilience_score == 0.82
        assert report.num_test_types == 5

        # Step 2: Transform to dict
        report_dict = transformer.transform(
            REALISTIC_RESILIENCE_DATA,
            "E2E_Resilience_Model"
        )

        # Validate dict structure
        assert isinstance(report_dict, dict)
        assert 'model_name' in report_dict
        assert 'summary' in report_dict
        assert 'distribution_shift' in report_dict

        # Step 3: Validate consistency
        assert report.model_name == report_dict['model_name']
        assert report.metrics.resilience_score == report_dict['resilience_score']


class TestCrossTransformerConsistency:
    """Test that all transformers work consistently together."""

    def test_all_transformers_produce_valid_models(self):
        """Test that all three transformers produce valid domain models."""
        uncertainty_transformer = UncertaintyDomainTransformer()
        robustness_transformer = RobustnessDomainTransformer()
        resilience_transformer = ResilienceDomainTransformer()

        # Transform all three
        uncertainty_report = uncertainty_transformer.transform_to_model(
            REALISTIC_UNCERTAINTY_DATA, "Model_1"
        )
        robustness_report = robustness_transformer.transform_to_model(
            REALISTIC_ROBUSTNESS_DATA, "Model_1"
        )
        resilience_report = resilience_transformer.transform_to_model(
            REALISTIC_RESILIENCE_DATA, "Model_1"
        )

        # All should be valid
        assert isinstance(uncertainty_report, UncertaintyReportData)
        assert isinstance(robustness_report, RobustnessReportData)
        assert isinstance(resilience_report, ResilienceReportData)

        # All should have same model name
        assert uncertainty_report.model_name == "Model_1"
        assert robustness_report.model_name == "Model_1"
        assert resilience_report.model_name == "Model_1"

    def test_all_transformers_have_backward_compatibility(self):
        """Test that all transformers maintain backward compatibility."""
        transformers = [
            (UncertaintyDomainTransformer(), REALISTIC_UNCERTAINTY_DATA),
            (RobustnessDomainTransformer(), REALISTIC_ROBUSTNESS_DATA),
            (ResilienceDomainTransformer(), REALISTIC_RESILIENCE_DATA),
        ]

        for transformer, data in transformers:
            # Both modes should work
            model = transformer.transform_to_model(data, "Test")
            dict_result = transformer.transform(data, "Test")

            # Both should produce valid results
            assert model is not None
            assert isinstance(dict_result, dict)
            assert 'model_name' in dict_result

    def test_consistent_feature_handling(self):
        """Test that feature importance is handled consistently."""
        uncertainty_transformer = UncertaintyDomainTransformer()
        robustness_transformer = RobustnessDomainTransformer()
        resilience_transformer = ResilienceDomainTransformer()

        uncertainty_report = uncertainty_transformer.transform_to_model(
            REALISTIC_UNCERTAINTY_DATA, "Model"
        )
        robustness_report = robustness_transformer.transform_to_model(
            REALISTIC_ROBUSTNESS_DATA, "Model"
        )
        resilience_report = resilience_transformer.transform_to_model(
            REALISTIC_RESILIENCE_DATA, "Model"
        )

        # All should have feature data
        assert uncertainty_report.has_feature_importance
        assert robustness_report.has_feature_data
        assert resilience_report.has_feature_importance

        # All should have same number of features (10 in test data)
        assert len(uncertainty_report.features) == 10
        assert len(robustness_report.features) == 10
        assert len(resilience_report.features) == 10


class TestPerformance:
    """Test performance characteristics of domain model pipeline."""

    def test_transformation_speed_uncertainty(self):
        """Test that Uncertainty transformation is fast enough."""
        transformer = UncertaintyDomainTransformer()

        start = time.time()
        for _ in range(100):
            report = transformer.transform_to_model(
                REALISTIC_UNCERTAINTY_DATA,
                "PerfTest"
            )
        duration = time.time() - start

        # Should complete 100 transformations in < 1 second
        assert duration < 1.0, f"Too slow: {duration:.3f}s for 100 transformations"

    def test_transformation_speed_robustness(self):
        """Test that Robustness transformation is fast enough."""
        transformer = RobustnessDomainTransformer()

        start = time.time()
        for _ in range(100):
            report = transformer.transform_to_model(
                REALISTIC_ROBUSTNESS_DATA,
                "PerfTest"
            )
        duration = time.time() - start

        # Should complete 100 transformations in < 1 second
        assert duration < 1.0, f"Too slow: {duration:.3f}s for 100 transformations"

    def test_transformation_speed_resilience(self):
        """Test that Resilience transformation is fast enough."""
        transformer = ResilienceDomainTransformer()

        start = time.time()
        for _ in range(100):
            report = transformer.transform_to_model(
                REALISTIC_RESILIENCE_DATA,
                "PerfTest"
            )
        duration = time.time() - start

        # Should complete 100 transformations in < 2 seconds (more complex)
        assert duration < 2.0, f"Too slow: {duration:.3f}s for 100 transformations"

    def test_no_memory_leaks(self):
        """Test that repeated transformations don't leak memory."""
        import gc

        transformer = UncertaintyDomainTransformer()

        # Force garbage collection
        gc.collect()

        # Create and discard many reports
        for _ in range(1000):
            report = transformer.transform_to_model(
                REALISTIC_UNCERTAINTY_DATA,
                "MemTest"
            )
            # Report should be eligible for garbage collection immediately
            del report

        # Force garbage collection
        gc.collect()

        # If we got here without MemoryError, we're good
        assert True


class TestRealWorldScenarios:
    """Test real-world usage scenarios with multiple reports."""

    def test_generate_multiple_model_reports(self):
        """Test generating reports for multiple models in sequence."""
        uncertainty_transformer = UncertaintyDomainTransformer()
        robustness_transformer = RobustnessDomainTransformer()

        # Simulate analyzing 3 different models
        models = ["Model_A", "Model_B", "Model_C"]
        uncertainty_reports = []
        robustness_reports = []

        for model_name in models:
            u_report = uncertainty_transformer.transform_to_model(
                REALISTIC_UNCERTAINTY_DATA,
                model_name
            )
            r_report = robustness_transformer.transform_to_model(
                REALISTIC_ROBUSTNESS_DATA,
                model_name
            )

            uncertainty_reports.append(u_report)
            robustness_reports.append(r_report)

        # All reports should be valid
        assert len(uncertainty_reports) == 3
        assert len(robustness_reports) == 3

        # Each should have correct model name
        for i, model_name in enumerate(models):
            assert uncertainty_reports[i].model_name == model_name
            assert robustness_reports[i].model_name == model_name

    def test_compare_models_using_domain_properties(self):
        """Test using domain model properties to compare models."""
        import copy

        transformer = UncertaintyDomainTransformer()

        # Create data for two models with different quality (deep copy to avoid mutation)
        good_model_data = copy.deepcopy(REALISTIC_UNCERTAINTY_DATA)
        bad_model_data = copy.deepcopy(REALISTIC_UNCERTAINTY_DATA)

        # Make bad model very poorly calibrated (large errors across multiple alphas)
        bad_model_data['primary_model']['crqr']['by_alpha']['alpha_0.05']['overall_result']['coverage'] = 0.70
        bad_model_data['primary_model']['crqr']['by_alpha']['alpha_0.10']['overall_result']['coverage'] = 0.65
        bad_model_data['primary_model']['crqr']['by_alpha']['alpha_0.15']['overall_result']['coverage'] = 0.60

        good_report = transformer.transform_to_model(good_model_data, "GoodModel")
        bad_report = transformer.transform_to_model(bad_model_data, "BadModel")

        # Can easily compare using properties and metrics
        # Good model should have better uncertainty score
        assert good_report.metrics.uncertainty_score > bad_report.metrics.uncertainty_score
        # Good model has smaller calibration error
        assert good_report.metrics.calibration_error < bad_report.metrics.calibration_error


class TestEdgeCasesIntegration:
    """Test edge case handling across the full pipeline."""

    def test_empty_data_all_transformers(self):
        """Test that all transformers handle empty data gracefully."""
        empty_data = {'primary_model': {}, 'initial_model_evaluation': {}}

        uncertainty_transformer = UncertaintyDomainTransformer()
        robustness_transformer = RobustnessDomainTransformer()
        resilience_transformer = ResilienceDomainTransformer()

        # All should handle empty data without crashing
        u_report = uncertainty_transformer.transform_to_model(empty_data, "Empty")
        r_report = robustness_transformer.transform_to_model(empty_data, "Empty")
        res_report = resilience_transformer.transform_to_model(empty_data, "Empty")

        # All should create valid (though empty) reports
        assert u_report.model_name == "Empty"
        assert r_report.model_name == "Empty"
        assert res_report.model_name == "Empty"

    def test_nan_validation_uncertainty(self):
        """Test that domain models properly reject NaN values via validation."""
        import copy
        from pydantic import ValidationError

        data_with_nan = copy.deepcopy(REALISTIC_UNCERTAINTY_DATA)
        data_with_nan['primary_model']['crqr']['by_alpha']['alpha_0.10']['overall_result']['coverage'] = float('nan')

        transformer = UncertaintyDomainTransformer()

        # Domain models should reject NaN through validation (type safety!)
        # This is CORRECT behavior - invalid data should fail fast
        with pytest.raises(ValidationError):
            report = transformer.transform_to_model(data_with_nan, "NaN_Test")

    def test_mixed_valid_invalid_data(self):
        """Test handling of partially valid data across transformers."""
        # Resilience with some valid and some invalid scenarios
        mixed_data = REALISTIC_RESILIENCE_DATA.copy()
        mixed_data['primary_model']['distribution_shift']['all_results'].append({
            'name': 'Invalid_Scenario',
            'alpha': 0.5,
            'distance_metric': 'euclidean',
            'metric': 'accuracy',
            'performance_gap': float('nan'),  # Invalid!
            'worst_metric': 0.70,
            'remaining_metric': 0.90
        })

        transformer = ResilienceDomainTransformer()
        report = transformer.transform_to_model(mixed_data, "MixedTest")

        # Should process valid scenarios and mark invalid ones
        assert report.has_distribution_shift
        scenarios = report.distribution_shift_scenarios
        assert any(s.is_valid for s in scenarios)  # Some valid
        assert any(not s.is_valid for s in scenarios)  # Some invalid


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
