"""
Tests for Uncertainty domain models (Phase 3 Sprint 10).

Tests type safety, validation, and properties of Pydantic models.
"""

import pytest
from pydantic import ValidationError

from deepbridge.core.experiment.report.domain import (
    UncertaintyMetrics,
    CalibrationResults,
    AlternativeModelData,
    UncertaintyReportData,
)


class TestUncertaintyMetrics:
    """Tests for UncertaintyMetrics model."""

    def test_create_with_valid_data(self):
        """Test creating metrics with valid data."""
        metrics = UncertaintyMetrics(
            uncertainty_score=0.85,
            coverage=0.90,
            mean_width=0.15,
            expected_coverage=0.90
        )

        assert metrics.uncertainty_score == 0.85
        assert metrics.coverage == 0.90
        assert metrics.mean_width == 0.15
        assert metrics.expected_coverage == 0.90

    def test_defaults_applied(self):
        """Test that defaults are applied for missing fields."""
        metrics = UncertaintyMetrics()

        assert metrics.uncertainty_score == 0.0
        assert metrics.coverage == 0.0
        assert metrics.mean_width == 0.0

    def test_none_coerced_to_default(self):
        """Test that None values are coerced to defaults."""
        metrics = UncertaintyMetrics(
            uncertainty_score=None,
            coverage=None
        )

        assert metrics.uncertainty_score == 0.0
        assert metrics.coverage == 0.0

    def test_validation_uncertainty_score_range(self):
        """Test that uncertainty_score is validated to [0, 1]."""
        # Valid
        UncertaintyMetrics(uncertainty_score=0.0)
        UncertaintyMetrics(uncertainty_score=1.0)
        UncertaintyMetrics(uncertainty_score=0.5)

        # Invalid - too low
        with pytest.raises(ValidationError):
            UncertaintyMetrics(uncertainty_score=-0.1)

        # Invalid - too high
        with pytest.raises(ValidationError):
            UncertaintyMetrics(uncertainty_score=1.1)

    def test_validation_coverage_range(self):
        """Test that coverage is validated to [0, 1]."""
        with pytest.raises(ValidationError):
            UncertaintyMetrics(coverage=1.5)

    def test_calibration_error_auto_computed(self):
        """Test that calibration_error is auto-computed."""
        metrics = UncertaintyMetrics(
            coverage=0.85,
            expected_coverage=0.90
        )

        assert metrics.calibration_error == pytest.approx(0.05, abs=1e-6)

    def test_float_rounding(self):
        """Test that floats are rounded to 4 decimals."""
        metrics = UncertaintyMetrics(
            uncertainty_score=0.123456789
        )

        assert metrics.uncertainty_score == 0.1235


class TestCalibrationResults:
    """Tests for CalibrationResults model."""

    def test_create_with_valid_data(self):
        """Test creating calibration results."""
        results = CalibrationResults(
            alpha_values=[0.1, 0.2, 0.3],
            coverage_values=[0.9, 0.8, 0.7],
            expected_coverages=[0.9, 0.8, 0.7],
            width_values=[0.1, 0.15, 0.2]
        )

        assert len(results.alpha_values) == 3
        assert results.has_calibration_data
        assert results.num_alpha_levels == 3

    def test_empty_calibration(self):
        """Test empty calibration results."""
        results = CalibrationResults()

        assert not results.has_calibration_data
        assert results.num_alpha_levels == 0

    def test_validation_length_mismatch(self):
        """Test that array lengths must match."""
        with pytest.raises(ValidationError) as exc_info:
            CalibrationResults(
                alpha_values=[0.1, 0.2, 0.3],
                coverage_values=[0.9, 0.8]  # Too short!
            )

        assert "Length mismatch" in str(exc_info.value)

    def test_properties(self):
        """Test convenience properties."""
        results = CalibrationResults(
            alpha_values=[0.1, 0.2],
            coverage_values=[0.9, 0.8],
            expected_coverages=[0.9, 0.8],
            width_values=[0.1, 0.15]
        )

        assert results.has_calibration_data is True
        assert results.num_alpha_levels == 2


class TestAlternativeModelData:
    """Tests for AlternativeModelData model."""

    def test_create_alternative_model(self):
        """Test creating alternative model data."""
        model = AlternativeModelData(
            name="Monte Carlo Dropout",
            uncertainty_score=0.80,
            coverage=0.88,
            mean_width=0.20,
            metrics={'dropout_rate': 0.5}
        )

        assert model.name == "Monte Carlo Dropout"
        assert model.uncertainty_score == 0.80
        assert model.metrics['dropout_rate'] == 0.5

    def test_is_better_than_property(self):
        """Test is_better_than heuristic."""
        good_model = AlternativeModelData(
            name="Good",
            uncertainty_score=0.75
        )

        bad_model = AlternativeModelData(
            name="Bad",
            uncertainty_score=0.30
        )

        assert good_model.is_better_than is True
        assert bad_model.is_better_than is False


class TestUncertaintyReportData:
    """Tests for UncertaintyReportData model."""

    def test_create_minimal_report(self):
        """Test creating report with minimal data."""
        report = UncertaintyReportData(
            model_name="TestModel",
            timestamp="2025-11-05T10:00:00"
        )

        assert report.model_name == "TestModel"
        assert report.model_type == "Unknown"  # Default
        assert report.metrics.uncertainty_score == 0.0  # Default

    def test_create_complete_report(self):
        """Test creating report with complete data."""
        report = UncertaintyReportData(
            model_name="CompleteModel",
            model_type="Neural Network",
            timestamp="2025-11-05T10:00:00",
            metrics=UncertaintyMetrics(
                uncertainty_score=0.85,
                coverage=0.90,
                mean_width=0.15
            ),
            calibration_results=CalibrationResults(
                alpha_values=[0.1, 0.2],
                coverage_values=[0.9, 0.8],
                expected_coverages=[0.9, 0.8],
                width_values=[0.1, 0.15]
            ),
            feature_importance={'feature1': 0.8, 'feature2': 0.6},
            features=['feature1', 'feature2'],
            alternative_models={
                'mc_dropout': AlternativeModelData(
                    name="MC Dropout",
                    uncertainty_score=0.78
                )
            },
            dataset_size=1000
        )

        assert report.model_name == "CompleteModel"
        assert report.metrics.uncertainty_score == 0.85
        assert report.has_calibration_results
        assert report.has_feature_importance
        assert report.has_alternative_models
        assert report.num_alternative_models == 1

    def test_property_has_alternative_models(self):
        """Test has_alternative_models property."""
        report = UncertaintyReportData(
            model_name="Test",
            timestamp="2025-11-05"
        )

        assert not report.has_alternative_models

        report.alternative_models['method1'] = AlternativeModelData(name="Method1")

        assert report.has_alternative_models
        assert report.num_alternative_models == 1

    def test_property_has_calibration_results(self):
        """Test has_calibration_results property."""
        report = UncertaintyReportData(
            model_name="Test",
            timestamp="2025-11-05"
        )

        assert not report.has_calibration_results

        report.calibration_results = CalibrationResults(
            alpha_values=[0.1],
            coverage_values=[0.9],
            expected_coverages=[0.9],
            width_values=[0.1]
        )

        assert report.has_calibration_results

    def test_property_top_features(self):
        """Test top_features property."""
        report = UncertaintyReportData(
            model_name="Test",
            timestamp="2025-11-05",
            feature_importance={
                'f1': 0.9,
                'f2': 0.8,
                'f3': 0.7,
                'f4': 0.6,
                'f5': 0.5,
                'f6': 0.4  # Should not appear in top 5
            }
        )

        top = report.top_features

        assert len(top) == 5
        assert top[0] == ('f1', 0.9)
        assert top[4] == ('f5', 0.5)
        assert 'f6' not in [f[0] for f in top]

    def test_property_is_well_calibrated(self):
        """Test is_well_calibrated property."""
        well_calibrated = UncertaintyReportData(
            model_name="Good",
            timestamp="2025-11-05",
            metrics=UncertaintyMetrics(
                coverage=0.90,
                expected_coverage=0.92
                # calibration_error = 0.02 < 0.05
            )
        )

        poorly_calibrated = UncertaintyReportData(
            model_name="Bad",
            timestamp="2025-11-05",
            metrics=UncertaintyMetrics(
                coverage=0.80,
                expected_coverage=0.90
                # calibration_error = 0.10 > 0.05
            )
        )

        assert well_calibrated.is_well_calibrated is True
        assert poorly_calibrated.is_well_calibrated is False

    def test_get_summary_stats(self):
        """Test get_summary_stats method."""
        report = UncertaintyReportData(
            model_name="Test",
            timestamp="2025-11-05",
            metrics=UncertaintyMetrics(
                uncertainty_score=0.85,
                coverage=0.90,
                mean_width=0.15,
                calibration_error=0.05
            ),
            features=['f1', 'f2', 'f3']
        )

        stats = report.get_summary_stats()

        assert stats['uncertainty_score'] == 0.85
        assert stats['coverage'] == 0.90
        assert stats['mean_width'] == 0.15
        assert stats['calibration_error'] == 0.05
        assert stats['num_features'] == 3
        assert stats['num_alternative_models'] == 0

    def test_str_representation(self):
        """Test string representation."""
        report = UncertaintyReportData(
            model_name="TestModel",
            timestamp="2025-11-05",
            metrics=UncertaintyMetrics(
                uncertainty_score=0.85,
                coverage=0.90,
                calibration_error=0.05
            )
        )

        str_repr = str(report)

        assert "TestModel" in str_repr
        assert "0.850" in str_repr or "0.85" in str_repr
        assert "0.900" in str_repr or "0.9" in str_repr

    def test_type_safety_no_get_calls(self):
        """
        Demonstrate that domain models eliminate .get() calls.

        This is the key benefit - type-safe access!
        """
        report = UncertaintyReportData(
            model_name="Test",
            timestamp="2025-11-05",
            metrics=UncertaintyMetrics(
                uncertainty_score=0.85,
                coverage=0.90
            )
        )

        # Before (Dict): report_data.get('uncertainty_score', 0.0)
        # After (Model): report.metrics.uncertainty_score
        # Type-safe, IDE autocomplete, no .get() needed!

        score = report.metrics.uncertainty_score  # Direct access!
        coverage = report.metrics.coverage  # No .get()!

        assert score == 0.85
        assert coverage == 0.90

    def test_validation_on_assignment(self):
        """Test that validation occurs on attribute assignment."""
        report = UncertaintyReportData(
            model_name="Test",
            timestamp="2025-11-05"
        )

        # Should validate on assignment
        report.metrics = UncertaintyMetrics(uncertainty_score=0.5)
        assert report.metrics.uncertainty_score == 0.5

        # Invalid value should raise error
        with pytest.raises(ValidationError):
            report.metrics = UncertaintyMetrics(uncertainty_score=1.5)


class TestDomainModelBenefits:
    """Tests demonstrating benefits of domain models."""

    def test_eliminates_get_with_defaults(self):
        """Demonstrate elimination of .get(key, default) pattern."""
        # Old way with Dict
        old_data = {'uncertainty_score': 0.85}
        score = old_data.get('uncertainty_score', 0.0)  # Verbose
        coverage = old_data.get('coverage', 0.0)  # Missing key needs default

        # New way with domain model
        report = UncertaintyReportData(
            model_name="Test",
            timestamp="2025-11-05",
            metrics=UncertaintyMetrics(uncertainty_score=0.85)
        )

        score_new = report.metrics.uncertainty_score  # Clean!
        coverage_new = report.metrics.coverage  # Default applied automatically!

        assert score == score_new
        assert coverage == coverage_new

    def test_eliminates_isinstance_checks(self):
        """Demonstrate elimination of isinstance checks."""
        # Old way with Dict - need to validate
        old_data = {'alternative_models': {'method1': {'name': 'Test'}}}

        has_alternatives_old = (
            'alternative_models' in old_data and
            isinstance(old_data['alternative_models'], dict)
        )

        # New way with domain model - type guaranteed!
        report = UncertaintyReportData(
            model_name="Test",
            timestamp="2025-11-05",
            alternative_models={
                'method1': AlternativeModelData(name="Test")
            }
        )

        has_alternatives_new = report.has_alternative_models  # Property!

        assert has_alternatives_old == has_alternatives_new

    def test_provides_ide_autocomplete(self):
        """
        Demonstrate IDE autocomplete capability.

        With domain models, IDEs can provide:
        - Autocomplete for all fields
        - Type hints in tooltips
        - Jump to definition
        - Refactoring support
        """
        report = UncertaintyReportData(
            model_name="Test",
            timestamp="2025-11-05"
        )

        # All of these have full IDE support!
        _ = report.model_name
        _ = report.metrics.uncertainty_score
        _ = report.has_calibration_results
        _ = report.top_features
        _ = report.get_summary_stats()

        # Type checker knows the types!
        assert isinstance(report.model_name, str)
        assert isinstance(report.metrics.uncertainty_score, float)
        assert isinstance(report.has_calibration_results, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
