"""
Tests for pre-training fairness charts.
"""

import pytest

from deepbridge.core.experiment.report.transformers.fairness.charts.pretrain_charts import (
    ConceptBalanceChart,
    GroupSizesChart,
    PretrainMetricsOverviewChart,
)


class TestPretrainMetricsOverviewChart:
    """Tests for PretrainMetricsOverviewChart."""

    def test_create_with_valid_data(
        self, sample_pretrain_metrics, sample_protected_attrs, plotly_validator
    ):
        """Test chart creation with valid data."""
        chart = PretrainMetricsOverviewChart()
        result = chart.create(
            {
                'pretrain_metrics': sample_pretrain_metrics,
                'protected_attrs': sample_protected_attrs,
            }
        )

        assert plotly_validator(result), 'Should return valid Plotly JSON'

    def test_create_with_empty_metrics(self, sample_protected_attrs):
        """Test chart creation with empty metrics."""
        chart = PretrainMetricsOverviewChart()
        result = chart.create(
            {'pretrain_metrics': {}, 'protected_attrs': sample_protected_attrs}
        )

        assert result == '{}', 'Should return empty JSON for empty metrics'

    def test_includes_all_four_pretrain_metrics(
        self, sample_protected_attrs, plotly_validator
    ):
        """Test that chart includes all 4 pre-training metrics."""
        chart = PretrainMetricsOverviewChart()

        metrics = {
            'gender': {
                'class_balance': {'value': 0.02, 'interpretation': '✓ GOOD'},
                'concept_balance': {'value': 0.03, 'interpretation': '✓ GOOD'},
                'kl_divergence': {'value': 0.015, 'interpretation': '✓ GOOD'},
                'js_divergence': {'value': 0.008, 'interpretation': '✓ GOOD'},
            }
        }

        result = chart.create(
            {
                'pretrain_metrics': metrics,
                'protected_attrs': sample_protected_attrs,
            }
        )

        assert plotly_validator(
            result
        ), 'Should create chart with all 4 metrics'

    def test_handles_multiple_attributes(self, plotly_validator):
        """Test chart with multiple protected attributes."""
        chart = PretrainMetricsOverviewChart()

        metrics = {
            'gender': {
                'class_balance': {'value': 0.02, 'interpretation': '✓ GOOD'}
            },
            'age': {
                'class_balance': {'value': 0.05, 'interpretation': '⚠ WARNING'}
            },
            'race': {
                'class_balance': {
                    'value': 0.08,
                    'interpretation': '✗ CRITICAL',
                }
            },
        }

        result = chart.create(
            {
                'pretrain_metrics': metrics,
                'protected_attrs': ['gender', 'age', 'race'],
            }
        )

        assert plotly_validator(result), 'Should handle multiple attributes'


class TestGroupSizesChart:
    """Tests for GroupSizesChart."""

    def test_create_with_valid_data(
        self,
        sample_protected_attrs_distribution,
        sample_protected_attrs,
        plotly_validator,
    ):
        """Test chart creation with valid data."""
        chart = GroupSizesChart()
        result = chart.create(
            {
                'protected_attrs_distribution': sample_protected_attrs_distribution,
                'protected_attrs': sample_protected_attrs,
            }
        )

        assert plotly_validator(result), 'Should return valid Plotly JSON'

    def test_create_with_empty_distribution(self, sample_protected_attrs):
        """Test chart creation with empty distribution."""
        chart = GroupSizesChart()
        result = chart.create(
            {
                'protected_attrs_distribution': {},
                'protected_attrs': sample_protected_attrs,
            }
        )

        assert (
            result == '{}'
        ), 'Should return empty JSON for empty distribution'

    def test_displays_counts_and_percentages(
        self, sample_protected_attrs, plotly_validator
    ):
        """Test that chart displays both counts and percentages."""
        chart = GroupSizesChart()

        distribution = {
            'gender': {
                'distribution': {
                    'Male': {'count': 600, 'percentage': 60.0},
                    'Female': {'count': 400, 'percentage': 40.0},
                },
                'total_samples': 1000,
            }
        }

        result = chart.create(
            {
                'protected_attrs_distribution': distribution,
                'protected_attrs': sample_protected_attrs,
            }
        )

        assert plotly_validator(
            result
        ), 'Should display counts and percentages'

    def test_handles_multiple_groups(self, plotly_validator):
        """Test chart with multiple groups per attribute."""
        chart = GroupSizesChart()

        distribution = {
            'race': {
                'distribution': {
                    'White': {'count': 400, 'percentage': 40.0},
                    'Black': {'count': 300, 'percentage': 30.0},
                    'Hispanic': {'count': 200, 'percentage': 20.0},
                    'Asian': {'count': 100, 'percentage': 10.0},
                },
                'total_samples': 1000,
            }
        }

        result = chart.create(
            {
                'protected_attrs_distribution': distribution,
                'protected_attrs': ['race'],
            }
        )

        assert plotly_validator(result), 'Should handle multiple groups'


class TestConceptBalanceChart:
    """Tests for ConceptBalanceChart."""

    def test_create_with_valid_data(
        self, sample_pretrain_metrics, sample_protected_attrs, plotly_validator
    ):
        """Test chart creation with valid data."""
        chart = ConceptBalanceChart()
        result = chart.create(
            {
                'pretrain_metrics': sample_pretrain_metrics,
                'protected_attrs': sample_protected_attrs,
            }
        )

        assert plotly_validator(result), 'Should return valid Plotly JSON'

    def test_create_with_empty_metrics(self, sample_protected_attrs):
        """Test chart creation with empty metrics."""
        chart = ConceptBalanceChart()
        result = chart.create(
            {'pretrain_metrics': {}, 'protected_attrs': sample_protected_attrs}
        )

        assert result == '{}', 'Should return empty JSON for empty metrics'

    def test_create_without_concept_balance_metric(
        self, sample_protected_attrs
    ):
        """Test chart creation when concept_balance metric is missing."""
        chart = ConceptBalanceChart()
        result = chart.create(
            {
                'pretrain_metrics': {
                    'gender': {'class_balance': {'value': 0.02}}
                },
                'protected_attrs': sample_protected_attrs,
            }
        )

        assert (
            result == '{}'
        ), 'Should return empty JSON when concept_balance is missing'

    def test_compares_group_positive_rates(
        self, sample_protected_attrs, plotly_validator
    ):
        """Test that chart compares positive rates between groups."""
        chart = ConceptBalanceChart()

        metrics = {
            'gender': {
                'concept_balance': {
                    'value': 0.05,
                    'group_a': 'Male',
                    'group_b': 'Female',
                    'group_a_positive_rate': 0.55,
                    'group_b_positive_rate': 0.50,
                    'interpretation': '✓ GOOD',
                }
            }
        }

        result = chart.create(
            {
                'pretrain_metrics': metrics,
                'protected_attrs': sample_protected_attrs,
            }
        )

        assert plotly_validator(result), 'Should compare group positive rates'

    def test_handles_multiple_attributes_grouped(self, plotly_validator):
        """Test chart with multiple attributes showing grouped bars."""
        chart = ConceptBalanceChart()

        metrics = {
            'gender': {
                'concept_balance': {
                    'group_a': 'Male',
                    'group_b': 'Female',
                    'group_a_positive_rate': 0.55,
                    'group_b_positive_rate': 0.50,
                }
            },
            'age': {
                'concept_balance': {
                    'group_a': 'Young',
                    'group_b': 'Old',
                    'group_a_positive_rate': 0.60,
                    'group_b_positive_rate': 0.52,
                }
            },
        }

        result = chart.create(
            {'pretrain_metrics': metrics, 'protected_attrs': ['gender', 'age']}
        )

        assert plotly_validator(result), 'Should group bars by attribute'
