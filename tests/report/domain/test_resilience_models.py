"""
Tests for Resilience domain models (Phase 3 Sprint 10.4).

Tests type safety, validation, and properties of complex multi-test Pydantic models.
"""

import pytest
from pydantic import ValidationError

from deepbridge.core.experiment.report.domain import (
    HardSampleTestData,
    OuterSampleTestData,
    ResilienceMetrics,
    ResilienceReportData,
    ScenarioData,
    TestTypeSummary,
    WorstClusterTestData,
    WorstSampleTestData,
)


class TestResilienceMetrics:
    """Tests for ResilienceMetrics model."""

    def test_create_with_valid_data(self):
        """Test creating metrics with valid data."""
        metrics = ResilienceMetrics(
            resilience_score=0.85,
            total_scenarios=100,
            valid_scenarios=95,
            avg_performance_gap=0.15,
            max_performance_gap=0.35,
            min_performance_gap=0.05,
            base_performance=0.90
        )

        assert metrics.resilience_score == 0.85
        assert metrics.total_scenarios == 100
        assert metrics.avg_performance_gap == 0.15

    def test_defaults_applied(self):
        """Test that defaults are applied."""
        metrics = ResilienceMetrics()

        assert metrics.resilience_score == 1.0
        assert metrics.total_scenarios == 0
        assert metrics.avg_performance_gap == 0.0

    def test_is_resilient_property(self):
        """Test is_resilient property."""
        resilient = ResilienceMetrics(resilience_score=0.75)
        not_resilient = ResilienceMetrics(resilience_score=0.65)

        assert resilient.is_resilient is True
        assert not_resilient.is_resilient is False

    def test_has_critical_gaps_property(self):
        """Test has_critical_gaps property."""
        critical = ResilienceMetrics(max_performance_gap=0.35)
        not_critical = ResilienceMetrics(max_performance_gap=0.25)

        assert critical.has_critical_gaps is True
        assert not_critical.has_critical_gaps is False


class TestScenarioData:
    """Tests for ScenarioData model."""

    def test_create_scenario(self):
        """Test creating scenario data."""
        scenario = ScenarioData(
            id=1,
            name="Test Scenario",
            alpha=0.1,
            distance_metric="euclidean",
            metric="accuracy",
            performance_gap=0.15,
            baseline_performance=0.75,
            target_performance=0.90,
            is_valid=True
        )

        assert scenario.id == 1
        assert scenario.name == "Test Scenario"
        assert scenario.performance_gap == 0.15

    def test_has_significant_gap_property(self):
        """Test has_significant_gap property."""
        significant = ScenarioData(
            id=1,
            name="Test",
            performance_gap=0.25,
            is_valid=True
        )
        not_significant = ScenarioData(
            id=2,
            name="Test",
            performance_gap=0.15,
            is_valid=True
        )

        assert significant.has_significant_gap is True
        assert not_significant.has_significant_gap is False


class TestWorstSampleTestData:
    """Tests for WorstSampleTestData model."""

    def test_create_worst_sample_test(self):
        """Test creating worst-sample test data."""
        test = WorstSampleTestData(
            id=1,
            alpha=0.2,
            ranking_method="loss",
            metric="accuracy",
            performance_gap=0.12,
            worst_metric=0.75,
            remaining_metric=0.87,
            n_worst_samples=100,
            n_remaining_samples=400,
            is_valid=True
        )

        assert test.id == 1
        assert test.alpha == 0.2
        assert test.ranking_method == "loss"
        assert test.performance_gap == 0.12


class TestWorstClusterTestData:
    """Tests for WorstClusterTestData model."""

    def test_create_worst_cluster_test(self):
        """Test creating worst-cluster test data."""
        test = WorstClusterTestData(
            id=1,
            n_clusters=5,
            worst_cluster_id=2,
            metric="accuracy",
            performance_gap=0.18,
            worst_cluster_metric=0.70,
            remaining_metric=0.88,
            worst_cluster_size=50,
            remaining_size=450,
            top_features=[
                {'name': 'f1', 'contribution': 0.5},
                {'name': 'f2', 'contribution': 0.3}
            ],
            is_valid=True
        )

        assert test.id == 1
        assert test.n_clusters == 5
        assert len(test.top_features) == 2


class TestOuterSampleTestData:
    """Tests for OuterSampleTestData model."""

    def test_create_outer_sample_test(self):
        """Test creating outer-sample test data."""
        test = OuterSampleTestData(
            id=1,
            alpha=0.1,
            outlier_method="isolation_forest",
            metric="accuracy",
            performance_gap=0.20,
            outer_metric=0.65,
            inner_metric=0.85,
            n_outer_samples=50,
            n_inner_samples=450,
            is_valid=True
        )

        assert test.id == 1
        assert test.outlier_method == "isolation_forest"
        assert test.performance_gap == 0.20


class TestHardSampleTestData:
    """Tests for HardSampleTestData model."""

    def test_create_hard_sample_test(self):
        """Test creating hard-sample test data."""
        test = HardSampleTestData(
            id=1,
            skipped=False,
            disagreement_threshold=0.3,
            metric="accuracy",
            performance_gap=0.15,
            hard_metric=0.70,
            easy_metric=0.85,
            n_hard_samples=80,
            n_easy_samples=420,
            model_disagreements=[
                {'model_pair': 'A-B', 'disagreement': 0.4}
            ],
            is_valid=True
        )

        assert test.id == 1
        assert test.skipped is False
        assert test.disagreement_threshold == 0.3

    def test_create_skipped_test(self):
        """Test creating skipped test."""
        test = HardSampleTestData(
            id=1,
            skipped=True,
            reason="No alternative models available",
            is_valid=False
        )

        assert test.skipped is True
        assert test.reason == "No alternative models available"


class TestTestTypeSummary:
    """Tests for TestTypeSummary model."""

    def test_create_test_type_summary(self):
        """Test creating test type summary."""
        summary = TestTypeSummary(
            test_type="distribution_shift",
            total_tests=20,
            valid_tests=18,
            avg_performance_gap=0.12,
            has_results=True
        )

        assert summary.test_type == "distribution_shift"
        assert summary.total_tests == 20
        assert summary.valid_tests == 18


class TestResilienceReportData:
    """Tests for ResilienceReportData model."""

    def test_create_minimal_report(self):
        """Test creating report with minimal data."""
        report = ResilienceReportData(
            model_name="TestModel"
        )

        assert report.model_name == "TestModel"
        assert report.model_type == "Unknown"
        assert report.metrics.resilience_score == 1.0

    def test_create_complete_report(self):
        """Test creating report with complete data."""
        report = ResilienceReportData(
            model_name="CompleteModel",
            model_type="RandomForest",
            metrics=ResilienceMetrics(
                resilience_score=0.85,
                total_scenarios=50,
                valid_scenarios=48,
                avg_performance_gap=0.12
            ),
            distribution_shift_scenarios=[
                ScenarioData(
                    id=1,
                    name="Scenario 1",
                    alpha=0.1,
                    performance_gap=0.10,
                    is_valid=True
                )
            ],
            worst_sample_tests=[
                WorstSampleTestData(
                    id=1,
                    alpha=0.2,
                    ranking_method="loss",
                    performance_gap=0.15,
                    is_valid=True
                )
            ],
            test_scores={
                'distribution_shift': 0.90,
                'worst_sample': 0.85
            },
            feature_importance={'f1': 0.8, 'f2': 0.6},
            features=['f1', 'f2']
        )

        assert report.model_name == "CompleteModel"
        assert report.metrics.resilience_score == 0.85
        assert report.has_distribution_shift
        assert report.has_worst_sample
        assert report.has_feature_importance

    def test_property_has_test_types(self):
        """Test has_* properties for test types."""
        report = ResilienceReportData(
            model_name="Test",
            distribution_shift_scenarios=[
                ScenarioData(id=1, name="S1", is_valid=True)
            ],
            worst_sample_tests=[
                WorstSampleTestData(id=1, is_valid=True)
            ]
        )

        assert report.has_distribution_shift is True
        assert report.has_worst_sample is True
        assert report.has_worst_cluster is False
        assert report.has_outer_sample is False
        assert report.has_hard_sample is False

    def test_property_available_test_types(self):
        """Test available_test_types property."""
        report = ResilienceReportData(
            model_name="Test",
            distribution_shift_scenarios=[ScenarioData(id=1, name="S1")],
            worst_sample_tests=[WorstSampleTestData(id=1)]
        )

        types = report.available_test_types
        assert 'distribution_shift' in types
        assert 'worst_sample' in types
        assert len(types) == 2

    def test_property_num_test_types(self):
        """Test num_test_types property."""
        report = ResilienceReportData(
            model_name="Test",
            distribution_shift_scenarios=[ScenarioData(id=1, name="S1")],
            worst_sample_tests=[WorstSampleTestData(id=1)],
            outer_sample_tests=[OuterSampleTestData(id=1)]
        )

        assert report.num_test_types == 3

    def test_property_top_features(self):
        """Test top_features property."""
        report = ResilienceReportData(
            model_name="Test",
            feature_importance={
                'f1': 0.9,
                'f2': -0.8,  # Negative importance
                'f3': 0.7,
                'f4': 0.6,
                'f5': 0.5,
                'f6': 0.4,
                'f7': 0.3,
                'f8': 0.2,
                'f9': 0.1,
                'f10': 0.05,
                'f11': 0.01  # Should not appear in top 10
            }
        )

        top = report.top_features
        assert len(top) == 10
        # Should be sorted by absolute value
        assert top[0][0] == 'f1'  # 0.9
        assert top[1][0] == 'f2'  # |-0.8| = 0.8

    def test_property_worst_test_type(self):
        """Test worst_test_type property."""
        report = ResilienceReportData(
            model_name="Test",
            test_scores={
                'distribution_shift': 0.90,
                'worst_sample': 0.70,  # Worst
                'worst_cluster': 0.85
            }
        )

        worst = report.worst_test_type
        assert worst == 'worst_sample'

    def test_property_best_test_type(self):
        """Test best_test_type property."""
        report = ResilienceReportData(
            model_name="Test",
            test_scores={
                'distribution_shift': 0.95,  # Best
                'worst_sample': 0.70,
                'worst_cluster': 0.85
            }
        )

        best = report.best_test_type
        assert best == 'distribution_shift'

    def test_get_test_type_summary(self):
        """Test get_test_type_summary method."""
        report = ResilienceReportData(
            model_name="Test",
            distribution_shift_scenarios=[
                ScenarioData(id=1, name="S1", performance_gap=0.10, is_valid=True),
                ScenarioData(id=2, name="S2", performance_gap=0.20, is_valid=True),
                ScenarioData(id=3, name="S3", performance_gap=None, is_valid=False),
            ]
        )

        summary = report.get_test_type_summary('distribution_shift')

        assert summary.test_type == 'distribution_shift'
        assert summary.total_tests == 3
        assert summary.valid_tests == 2
        assert summary.avg_performance_gap == pytest.approx(0.15, abs=1e-6)
        assert summary.has_results is True

    def test_get_summary_stats(self):
        """Test get_summary_stats method."""
        report = ResilienceReportData(
            model_name="Test",
            model_type="RF",
            metrics=ResilienceMetrics(
                resilience_score=0.85,
                total_scenarios=50,
                valid_scenarios=48,
                avg_performance_gap=0.12,
                max_performance_gap=0.30
            ),
            distribution_shift_scenarios=[ScenarioData(id=1, name="S1")],
            worst_sample_tests=[WorstSampleTestData(id=1)],
            test_scores={
                'distribution_shift': 0.90,
                'worst_sample': 0.80
            }
        )

        stats = report.get_summary_stats()

        assert stats['model_name'] == "Test"
        assert stats['resilience_score'] == 0.85
        assert stats['is_resilient'] is True
        assert stats['num_test_types'] == 2
        assert stats['worst_test_type'] == 'worst_sample'
        assert stats['best_test_type'] == 'distribution_shift'

    def test_str_representation(self):
        """Test string representation."""
        report = ResilienceReportData(
            model_name="TestModel",
            metrics=ResilienceMetrics(
                resilience_score=0.85,
                avg_performance_gap=0.12
            ),
            distribution_shift_scenarios=[ScenarioData(id=1, name="S1")],
            worst_sample_tests=[WorstSampleTestData(id=1)]
        )

        str_repr = str(report)

        assert "TestModel" in str_repr
        assert "0.850" in str_repr or "0.85" in str_repr
        assert "tests=2" in str_repr


class TestDomainModelBenefits:
    """Tests demonstrating benefits of domain models."""

    def test_type_safety_no_get_calls(self):
        """Demonstrate that domain models eliminate .get() calls."""
        report = ResilienceReportData(
            model_name="Test",
            metrics=ResilienceMetrics(
                resilience_score=0.85,
                avg_performance_gap=0.12
            )
        )

        # Before (Dict): report.get('resilience_score', 1.0)
        # After (Model): report.metrics.resilience_score
        score = report.metrics.resilience_score
        gap = report.metrics.avg_performance_gap

        assert score == 0.85
        assert gap == 0.12

    def test_property_convenience(self):
        """Demonstrate convenience properties."""
        report = ResilienceReportData(
            model_name="Test",
            metrics=ResilienceMetrics(resilience_score=0.75),
            distribution_shift_scenarios=[ScenarioData(id=1, name="S1")],
            feature_importance={'f1': 0.8}
        )

        # Convenient properties
        assert report.has_distribution_shift
        assert report.has_feature_importance
        assert isinstance(report.top_features, list)
        assert isinstance(report.get_summary_stats(), dict)
        assert report.metrics.is_resilient

    def test_complex_nested_access(self):
        """Demonstrate simplified access to nested data."""
        report = ResilienceReportData(
            model_name="Test",
            distribution_shift_scenarios=[
                ScenarioData(
                    id=1,
                    name="Critical Scenario",
                    performance_gap=0.35,
                    is_valid=True
                )
            ],
            test_scores={
                'distribution_shift': 0.80,
                'worst_sample': 0.75
            }
        )

        # Direct access to nested data (no .get() chains!)
        first_scenario = report.distribution_shift_scenarios[0]
        assert first_scenario.name == "Critical Scenario"
        assert first_scenario.has_significant_gap is True

        # Properties work on nested objects
        worst_type = report.worst_test_type
        assert worst_type == 'worst_sample'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
