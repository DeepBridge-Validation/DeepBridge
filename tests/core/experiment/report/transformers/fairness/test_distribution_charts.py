"""
Tests for distribution charts.
"""

import pytest

from deepbridge.core.experiment.report.transformers.fairness.charts.distribution_charts import (
    ProtectedAttributesDistributionChart,
    TargetDistributionChart,
)


class TestProtectedAttributesDistributionChart:
    """Tests for ProtectedAttributesDistributionChart."""

    def test_create_with_valid_data(self, sample_protected_attrs_distribution, sample_protected_attrs, plotly_validator):
        """Test chart creation with valid data."""
        chart = ProtectedAttributesDistributionChart()
        result = chart.create({
            'protected_attrs_distribution': sample_protected_attrs_distribution,
            'protected_attrs': sample_protected_attrs
        })

        assert plotly_validator(result), "Should return valid Plotly JSON"

    def test_create_with_empty_distribution(self, sample_protected_attrs):
        """Test chart creation with empty distribution."""
        chart = ProtectedAttributesDistributionChart()
        result = chart.create({
            'protected_attrs_distribution': {},
            'protected_attrs': sample_protected_attrs
        })

        assert result == '{}', "Should return empty JSON for empty distribution"

    def test_displays_percentages_on_bars(self, sample_protected_attrs, plotly_validator):
        """Test that percentages are displayed on bars."""
        chart = ProtectedAttributesDistributionChart()

        distribution = {
            'gender': {
                'distribution': {
                    'Male': {'count': 700, 'percentage': 70.0},
                    'Female': {'count': 300, 'percentage': 30.0}
                }
            }
        }

        result = chart.create({
            'protected_attrs_distribution': distribution,
            'protected_attrs': sample_protected_attrs
        })

        assert plotly_validator(result), "Should display percentages on bars"


class TestTargetDistributionChart:
    """Tests for TargetDistributionChart."""

    def test_create_with_valid_data(self, sample_target_distribution, plotly_validator):
        """Test chart creation with valid data."""
        chart = TargetDistributionChart()
        result = chart.create({
            'target_distribution': sample_target_distribution
        })

        assert plotly_validator(result), "Should return valid Plotly JSON"

    def test_create_with_empty_distribution(self):
        """Test chart creation with empty distribution."""
        chart = TargetDistributionChart()
        result = chart.create({
            'target_distribution': {}
        })

        assert result == '{}', "Should return empty JSON for empty distribution"

    def test_creates_pie_chart(self, plotly_validator):
        """Test that a pie chart is created."""
        chart = TargetDistributionChart()

        distribution = {
            '0': {'count': 450, 'percentage': 45.0},
            '1': {'count': 550, 'percentage': 55.0}
        }

        result = chart.create({
            'target_distribution': distribution
        })

        assert plotly_validator(result), "Should create pie chart"
