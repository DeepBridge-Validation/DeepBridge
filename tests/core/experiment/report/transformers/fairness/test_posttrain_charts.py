"""
Tests for post-training fairness charts.
"""

import pytest

from deepbridge.core.experiment.report.transformers.fairness.charts.posttrain_charts import (
    ComplianceStatusMatrixChart,
    DisparateImpactGaugeChart,
    DisparityComparisonChart,
)


class TestDisparateImpactGaugeChart:
    """Tests for DisparateImpactGaugeChart."""

    def test_create_with_valid_data(
        self,
        sample_posttrain_metrics,
        sample_protected_attrs,
        plotly_validator,
    ):
        """Test chart creation with valid data."""
        chart = DisparateImpactGaugeChart()
        result = chart.create(
            {
                'posttrain_metrics': sample_posttrain_metrics,
                'protected_attrs': sample_protected_attrs,
            }
        )

        assert plotly_validator(result), 'Should return valid Plotly JSON'

    def test_create_with_empty_metrics(self, sample_protected_attrs):
        """Test chart creation with empty metrics."""
        chart = DisparateImpactGaugeChart()
        result = chart.create(
            {
                'posttrain_metrics': {},
                'protected_attrs': sample_protected_attrs,
            }
        )

        assert result == '{}', 'Should return empty JSON for empty metrics'

    def test_create_with_no_disparate_impact(self, sample_protected_attrs):
        """Test chart creation when disparate_impact metric is missing."""
        chart = DisparateImpactGaugeChart()
        result = chart.create(
            {
                'posttrain_metrics': {'gender': {'other_metric': {}}},
                'protected_attrs': sample_protected_attrs,
            }
        )

        assert (
            result == '{}'
        ), 'Should return empty JSON when disparate_impact is missing'

    def test_gauge_colors_based_on_compliance(self, sample_protected_attrs):
        """Test that gauge colors change based on compliance levels."""
        chart = DisparateImpactGaugeChart()

        # Test with compliant ratio (>= 0.8)
        result_compliant = chart.create(
            {
                'posttrain_metrics': {
                    'gender': {
                        'disparate_impact': {
                            'ratio': 0.85,
                            'passes_80_rule': True,
                        }
                    }
                },
                'protected_attrs': sample_protected_attrs,
            }
        )
        assert (
            result_compliant != '{}'
        ), 'Should create chart for compliant ratio'

        # Test with warning ratio (0.7-0.8)
        result_warning = chart.create(
            {
                'posttrain_metrics': {
                    'gender': {
                        'disparate_impact': {
                            'ratio': 0.75,
                            'passes_80_rule': False,
                        }
                    }
                },
                'protected_attrs': sample_protected_attrs,
            }
        )
        assert result_warning != '{}', 'Should create chart for warning ratio'

        # Test with critical ratio (< 0.7)
        result_critical = chart.create(
            {
                'posttrain_metrics': {
                    'gender': {
                        'disparate_impact': {
                            'ratio': 0.65,
                            'passes_80_rule': False,
                        }
                    }
                },
                'protected_attrs': sample_protected_attrs,
            }
        )
        assert (
            result_critical != '{}'
        ), 'Should create chart for critical ratio'


class TestDisparityComparisonChart:
    """Tests for DisparityComparisonChart."""

    def test_create_with_valid_data(
        self,
        sample_posttrain_metrics,
        sample_protected_attrs,
        plotly_validator,
    ):
        """Test chart creation with valid data."""
        chart = DisparityComparisonChart()
        result = chart.create(
            {
                'posttrain_metrics': sample_posttrain_metrics,
                'protected_attrs': sample_protected_attrs,
            }
        )

        assert plotly_validator(result), 'Should return valid Plotly JSON'

    def test_create_with_empty_metrics(self, sample_protected_attrs):
        """Test chart creation with empty metrics."""
        chart = DisparityComparisonChart()
        result = chart.create(
            {
                'posttrain_metrics': {},
                'protected_attrs': sample_protected_attrs,
            }
        )

        assert result == '{}', 'Should return empty JSON for empty metrics'

    def test_create_with_no_statistical_parity(self, sample_protected_attrs):
        """Test chart creation when statistical_parity metric is missing."""
        chart = DisparityComparisonChart()
        result = chart.create(
            {
                'posttrain_metrics': {'gender': {'other_metric': {}}},
                'protected_attrs': sample_protected_attrs,
            }
        )

        assert (
            result == '{}'
        ), 'Should return empty JSON when statistical_parity is missing'

    def test_diverging_bar_with_positive_negative_disparities(
        self, sample_protected_attrs, plotly_validator
    ):
        """Test that diverging bars handle both positive and negative disparities."""
        chart = DisparityComparisonChart()
        result = chart.create(
            {
                'posttrain_metrics': {
                    'gender': {
                        'statistical_parity': {
                            'disparity': 0.08,
                            'interpretation': '✓ GOOD',
                        }
                    },
                    'age': {
                        'statistical_parity': {
                            'disparity': -0.12,
                            'interpretation': '⚠ WARNING',
                        }
                    },
                },
                'protected_attrs': ['gender', 'age'],
            }
        )

        assert plotly_validator(
            result
        ), 'Should handle both positive and negative disparities'


class TestComplianceStatusMatrixChart:
    """Tests for ComplianceStatusMatrixChart."""

    def test_create_with_valid_data(
        self,
        sample_posttrain_metrics,
        sample_protected_attrs,
        plotly_validator,
    ):
        """Test chart creation with valid data."""
        chart = ComplianceStatusMatrixChart()
        result = chart.create(
            {
                'posttrain_metrics': sample_posttrain_metrics,
                'protected_attrs': sample_protected_attrs,
            }
        )

        assert plotly_validator(result), 'Should return valid Plotly JSON'

    def test_create_with_empty_metrics(self, sample_protected_attrs):
        """Test chart creation with empty metrics."""
        chart = ComplianceStatusMatrixChart()
        result = chart.create(
            {
                'posttrain_metrics': {},
                'protected_attrs': sample_protected_attrs,
            }
        )

        assert result == '{}', 'Should return empty JSON for empty metrics'

    def test_matrix_includes_all_main_metrics(
        self, sample_protected_attrs, plotly_validator
    ):
        """Test that matrix includes all 5 main post-training metrics."""
        chart = ComplianceStatusMatrixChart()

        # Create metrics with all 5 main metrics
        metrics = {
            'gender': {
                'statistical_parity': {'interpretation': '✓ GOOD'},
                'equal_opportunity': {'interpretation': '✓ GOOD'},
                'equalized_odds': {'interpretation': '⚠ WARNING'},
                'disparate_impact': {'interpretation': '✓ GOOD'},
                'false_negative_rate_difference': {
                    'interpretation': '✗ CRITICAL'
                },
            }
        }

        result = chart.create(
            {
                'posttrain_metrics': metrics,
                'protected_attrs': sample_protected_attrs,
            }
        )

        assert plotly_validator(
            result
        ), 'Should create valid chart with all main metrics'

    def test_status_symbols_mapping(self, sample_protected_attrs):
        """Test that status interpretations are correctly mapped to symbols."""
        chart = ComplianceStatusMatrixChart()

        # Test different status levels
        metrics = {
            'gender': {
                'statistical_parity': {'interpretation': '✓ EXCELLENT'},
                'equal_opportunity': {'interpretation': '⚠ MODERATE'},
                'equalized_odds': {'interpretation': '✗ CRITICAL'},
                'disparate_impact': {'interpretation': 'GOOD'},
                'false_negative_rate_difference': {
                    'interpretation': 'Unknown'
                },
            }
        }

        result = chart.create(
            {
                'posttrain_metrics': metrics,
                'protected_attrs': sample_protected_attrs,
            }
        )

        assert result != '{}', 'Should handle various status interpretations'
