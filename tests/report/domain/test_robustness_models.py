"""
Tests for Robustness domain models (Phase 3 Sprint 10.3).

Tests type safety, validation, and properties of Pydantic models.
"""

import pytest
from pydantic import ValidationError

from deepbridge.core.experiment.report.domain import (
    FeatureRobustnessData,
    PerturbationLevelData,
    RobustnessMetrics,
    RobustnessReportData,
)


class TestRobustnessMetrics:
    """Tests for RobustnessMetrics model."""

    def test_create_with_valid_data(self):
        """Test creating metrics with valid data."""
        metrics = RobustnessMetrics(
            base_score=0.85,
            robustness_score=0.75,
            avg_raw_impact=0.10,
            avg_quantile_impact=0.12,
            avg_overall_impact=0.11,
            metric="AUC"
        )

        assert metrics.base_score == 0.85
        assert metrics.robustness_score == 0.75
        assert metrics.avg_raw_impact == 0.10
        assert metrics.metric == "AUC"

    def test_defaults_applied(self):
        """Test that defaults are applied for missing fields."""
        metrics = RobustnessMetrics()

        assert metrics.base_score == 0.0
        assert metrics.robustness_score == 0.0
        assert metrics.metric == "AUC"

    def test_validation_score_range(self):
        """Test that scores are validated to [0, 1]."""
        # Valid
        RobustnessMetrics(base_score=0.0)
        RobustnessMetrics(base_score=1.0)
        RobustnessMetrics(robustness_score=0.5)

        # Invalid
        with pytest.raises(ValidationError):
            RobustnessMetrics(base_score=-0.1)

        with pytest.raises(ValidationError):
            RobustnessMetrics(robustness_score=1.1)

    def test_is_robust_property(self):
        """Test is_robust property."""
        robust = RobustnessMetrics(robustness_score=0.75)
        not_robust = RobustnessMetrics(robustness_score=0.65)

        assert robust.is_robust is True
        assert not_robust.is_robust is False

    def test_degradation_rate_property(self):
        """Test degradation_rate property."""
        metrics = RobustnessMetrics(
            base_score=0.80,
            avg_overall_impact=0.16
        )

        assert metrics.degradation_rate == pytest.approx(0.20, abs=1e-6)


class TestPerturbationLevelData:
    """Tests for PerturbationLevelData model."""

    def test_create_level_data(self):
        """Test creating perturbation level data."""
        level = PerturbationLevelData(
            level=0.1,
            level_display="0.1",
            mean_score=0.75,
            std_score=0.05,
            impact=0.10,
            worst_score=0.65
        )

        assert level.level == 0.1
        assert level.mean_score == 0.75
        assert level.impact == 0.10

    def test_has_significant_impact_property(self):
        """Test has_significant_impact property."""
        significant = PerturbationLevelData(level=0.1, impact=0.15)
        not_significant = PerturbationLevelData(level=0.1, impact=0.05)

        assert significant.has_significant_impact is True
        assert not_significant.has_significant_impact is False


class TestFeatureRobustnessData:
    """Tests for FeatureRobustnessData model."""

    def test_create_feature_data(self):
        """Test creating feature robustness data."""
        feature = FeatureRobustnessData(
            name="feature1",
            importance=0.8,
            robustness_impact=0.3
        )

        assert feature.name == "feature1"
        assert feature.importance == 0.8
        assert feature.robustness_impact == 0.3

    def test_is_sensitive_property(self):
        """Test is_sensitive property."""
        sensitive = FeatureRobustnessData(
            name="f1",
            robustness_impact=0.25
        )
        not_sensitive = FeatureRobustnessData(
            name="f2",
            robustness_impact=0.15
        )

        assert sensitive.is_sensitive is True
        assert not_sensitive.is_sensitive is False


class TestRobustnessReportData:
    """Tests for RobustnessReportData model."""

    def test_create_minimal_report(self):
        """Test creating report with minimal data."""
        report = RobustnessReportData(
            model_name="TestModel"
        )

        assert report.model_name == "TestModel"
        assert report.model_type == "Unknown"
        assert report.metrics.robustness_score == 0.0

    def test_create_complete_report(self):
        """Test creating report with complete data."""
        report = RobustnessReportData(
            model_name="CompleteModel",
            model_type="RandomForest",
            metrics=RobustnessMetrics(
                base_score=0.85,
                robustness_score=0.75,
                avg_overall_impact=0.10
            ),
            perturbation_levels=[
                PerturbationLevelData(level=0.1, mean_score=0.80, impact=0.05),
                PerturbationLevelData(level=0.5, mean_score=0.70, impact=0.15),
            ],
            features=[
                FeatureRobustnessData(name="f1", importance=0.8, robustness_impact=0.2),
                FeatureRobustnessData(name="f2", importance=0.6, robustness_impact=0.3),
            ],
            n_iterations=20
        )

        assert report.model_name == "CompleteModel"
        assert report.metrics.robustness_score == 0.75
        assert report.has_perturbation_data
        assert report.has_feature_data
        assert report.num_perturbation_levels == 2
        assert report.num_features == 2
        assert report.n_iterations == 20

    def test_property_has_perturbation_data(self):
        """Test has_perturbation_data property."""
        empty_report = RobustnessReportData(model_name="Test")
        assert not empty_report.has_perturbation_data

        with_data = RobustnessReportData(
            model_name="Test",
            perturbation_levels=[
                PerturbationLevelData(level=0.1)
            ]
        )
        assert with_data.has_perturbation_data

    def test_property_has_feature_data(self):
        """Test has_feature_data property."""
        empty_report = RobustnessReportData(model_name="Test")
        assert not empty_report.has_feature_data

        with_data = RobustnessReportData(
            model_name="Test",
            features=[
                FeatureRobustnessData(name="f1")
            ]
        )
        assert with_data.has_feature_data

    def test_property_top_features(self):
        """Test top_features property."""
        report = RobustnessReportData(
            model_name="Test",
            features=[
                FeatureRobustnessData(name="f1", importance=0.9),
                FeatureRobustnessData(name="f2", importance=0.8),
                FeatureRobustnessData(name="f3", importance=0.7),
                FeatureRobustnessData(name="f4", importance=0.6),
                FeatureRobustnessData(name="f5", importance=0.5),
                FeatureRobustnessData(name="f6", importance=0.4),  # Should not appear
            ]
        )

        top = report.top_features
        assert len(top) == 5
        assert top[0].name == "f1"
        assert top[4].name == "f5"

    def test_property_most_sensitive_features(self):
        """Test most_sensitive_features property."""
        report = RobustnessReportData(
            model_name="Test",
            features=[
                FeatureRobustnessData(name="f1", importance=0.9, robustness_impact=0.1),
                FeatureRobustnessData(name="f2", importance=0.8, robustness_impact=0.5),  # Most sensitive
                FeatureRobustnessData(name="f3", importance=0.7, robustness_impact=0.3),
            ]
        )

        sensitive = report.most_sensitive_features
        assert len(sensitive) == 3
        assert sensitive[0].name == "f2"  # Highest impact

    def test_property_worst_perturbation_level(self):
        """Test worst_perturbation_level property."""
        report = RobustnessReportData(
            model_name="Test",
            perturbation_levels=[
                PerturbationLevelData(level=0.1, impact=0.05),
                PerturbationLevelData(level=0.5, impact=0.25),  # Worst
                PerturbationLevelData(level=1.0, impact=0.15),
            ]
        )

        worst = report.worst_perturbation_level
        assert worst is not None
        assert worst.level == 0.5
        assert worst.impact == 0.25

    def test_get_summary_stats(self):
        """Test get_summary_stats method."""
        report = RobustnessReportData(
            model_name="Test",
            model_type="RF",
            metrics=RobustnessMetrics(
                base_score=0.85,
                robustness_score=0.75,
                avg_overall_impact=0.10,
                metric="AUC"
            ),
            perturbation_levels=[PerturbationLevelData(level=0.1)],
            features=[FeatureRobustnessData(name="f1")]
        )

        stats = report.get_summary_stats()

        assert stats['model_name'] == "Test"
        assert stats['base_score'] == 0.85
        assert stats['robustness_score'] == 0.75
        assert stats['is_robust'] is True
        assert stats['num_levels'] == 1
        assert stats['num_features'] == 1
        assert stats['metric'] == "AUC"

    def test_str_representation(self):
        """Test string representation."""
        report = RobustnessReportData(
            model_name="TestModel",
            metrics=RobustnessMetrics(
                base_score=0.85,
                robustness_score=0.75,
                avg_overall_impact=0.10
            ),
            perturbation_levels=[
                PerturbationLevelData(level=0.1),
                PerturbationLevelData(level=0.5)
            ]
        )

        str_repr = str(report)

        assert "TestModel" in str_repr
        assert "0.750" in str_repr or "0.75" in str_repr
        assert "levels=2" in str_repr


class TestDomainModelBenefits:
    """Tests demonstrating benefits of domain models."""

    def test_type_safety_no_get_calls(self):
        """Demonstrate that domain models eliminate .get() calls."""
        report = RobustnessReportData(
            model_name="Test",
            metrics=RobustnessMetrics(
                robustness_score=0.75,
                base_score=0.85
            )
        )

        # Before (Dict): report.get('robustness_score', 0.0)
        # After (Model): report.metrics.robustness_score
        score = report.metrics.robustness_score
        base = report.metrics.base_score

        assert score == 0.75
        assert base == 0.85

    def test_property_convenience(self):
        """Demonstrate convenience properties."""
        report = RobustnessReportData(
            model_name="Test",
            metrics=RobustnessMetrics(robustness_score=0.75),
            perturbation_levels=[PerturbationLevelData(level=0.1)],
            features=[FeatureRobustnessData(name="f1")]
        )

        # Convenient properties
        assert report.has_perturbation_data
        assert report.has_feature_data
        assert isinstance(report.top_features, list)
        assert isinstance(report.get_summary_stats(), dict)
        assert report.metrics.is_robust


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
