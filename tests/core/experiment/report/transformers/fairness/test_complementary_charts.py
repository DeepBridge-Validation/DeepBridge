"""
Tests for complementary fairness charts.
"""

import pytest

from deepbridge.core.experiment.report.transformers.fairness.charts.complementary_charts import (
    ComplementaryMetricsRadarChart,
    PrecisionAccuracyComparisonChart,
    TreatmentEqualityScatterChart,
)


class TestPrecisionAccuracyComparisonChart:
    """Tests for PrecisionAccuracyComparisonChart."""

    def test_create_with_valid_data(self, sample_confusion_matrix, sample_protected_attrs, plotly_validator):
        """Test chart creation with valid data."""
        chart = PrecisionAccuracyComparisonChart()
        result = chart.create({
            'confusion_matrix': sample_confusion_matrix,
            'protected_attrs': sample_protected_attrs
        })

        assert plotly_validator(result), "Should return valid Plotly JSON"

    def test_create_with_empty_confusion_matrix(self, sample_protected_attrs):
        """Test chart creation with empty confusion matrix."""
        chart = PrecisionAccuracyComparisonChart()
        result = chart.create({
            'confusion_matrix': {},
            'protected_attrs': sample_protected_attrs
        })

        assert result == '{}', "Should return empty JSON for empty confusion matrix"

    def test_calculates_precision_and_accuracy(self, sample_protected_attrs, plotly_validator):
        """Test that chart correctly calculates precision and accuracy."""
        chart = PrecisionAccuracyComparisonChart()

        # Perfect predictions
        cm_perfect = {
            'gender': {
                'Male': {'TP': 50, 'TN': 50, 'FP': 0, 'FN': 0}
            }
        }

        result = chart.create({
            'confusion_matrix': cm_perfect,
            'protected_attrs': sample_protected_attrs
        })

        assert plotly_validator(result), "Should calculate metrics correctly"


class TestTreatmentEqualityScatterChart:
    """Tests for TreatmentEqualityScatterChart."""

    def test_create_with_valid_data(self, sample_confusion_matrix, sample_protected_attrs, plotly_validator):
        """Test chart creation with valid data."""
        chart = TreatmentEqualityScatterChart()
        result = chart.create({
            'confusion_matrix': sample_confusion_matrix,
            'protected_attrs': sample_protected_attrs
        })

        assert plotly_validator(result), "Should return valid Plotly JSON"

    def test_create_with_empty_confusion_matrix(self, sample_protected_attrs):
        """Test chart creation with empty confusion matrix."""
        chart = TreatmentEqualityScatterChart()
        result = chart.create({
            'confusion_matrix': {},
            'protected_attrs': sample_protected_attrs
        })

        assert result == '{}', "Should return empty JSON for empty confusion matrix"

    def test_plots_fn_vs_fp_rates(self, sample_protected_attrs, plotly_validator):
        """Test that chart plots FN vs FP rates."""
        chart = TreatmentEqualityScatterChart()

        cm_data = {
            'gender': {
                'Male': {'TP': 40, 'TN': 35, 'FP': 10, 'FN': 15},
                'Female': {'TP': 38, 'TN': 32, 'FP': 13, 'FN': 17}
            }
        }

        result = chart.create({
            'confusion_matrix': cm_data,
            'protected_attrs': sample_protected_attrs
        })

        assert plotly_validator(result), "Should plot FN vs FP rates"


class TestComplementaryMetricsRadarChart:
    """Tests for ComplementaryMetricsRadarChart."""

    def test_create_with_valid_data(self, sample_posttrain_metrics, sample_protected_attrs, plotly_validator):
        """Test chart creation with valid data."""
        chart = ComplementaryMetricsRadarChart()
        result = chart.create({
            'posttrain_metrics': sample_posttrain_metrics,
            'protected_attrs': sample_protected_attrs
        })

        assert plotly_validator(result), "Should return valid Plotly JSON"

    def test_create_with_empty_metrics(self, sample_protected_attrs):
        """Test chart creation with empty metrics."""
        chart = ComplementaryMetricsRadarChart()
        result = chart.create({
            'posttrain_metrics': {},
            'protected_attrs': sample_protected_attrs
        })

        assert result == '{}', "Should return empty JSON for empty metrics"

    def test_includes_complementary_metrics(self, sample_protected_attrs, plotly_validator):
        """Test that chart includes complementary metrics."""
        chart = ComplementaryMetricsRadarChart()

        metrics = {
            'gender': {
                'conditional_acceptance': {'value': 0.02, 'interpretation': '✓ GOOD'},
                'conditional_rejection': {'value': 0.03, 'interpretation': '✓ GOOD'},
                'precision_difference': {'disparity': 0.04, 'interpretation': '✓ GOOD'},
                'accuracy_difference': {'disparity': 0.01, 'interpretation': '✓ GOOD'},
                'treatment_equality': {'value': 0.05, 'interpretation': '⚠ WARNING'},
                'entropy_index': {'value': 0.02, 'interpretation': '✓ GOOD'}
            }
        }

        result = chart.create({
            'posttrain_metrics': metrics,
            'protected_attrs': sample_protected_attrs
        })

        assert plotly_validator(result), "Should include complementary metrics"
