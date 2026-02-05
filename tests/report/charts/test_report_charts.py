"""
Tests for report-specific chart generators (Phase 3 Sprint 9).

Tests all chart types for uncertainty, robustness, and resilience reports.
"""

import pytest

from deepbridge.core.experiment.report.charts import ChartRegistry, ChartResult

# ==================================================================================
# Test Data Fixtures
# ==================================================================================


@pytest.fixture
def uncertainty_coverage_data():
    """Sample data for coverage charts."""
    return {
        'alphas': [0.1, 0.2, 0.3, 0.4, 0.5],
        'coverage': [0.91, 0.81, 0.72, 0.61, 0.51],
        'expected': [0.90, 0.80, 0.70, 0.60, 0.50],
    }


@pytest.fixture
def width_coverage_data():
    """Sample data for width vs coverage."""
    return {
        'coverage': [0.91, 0.81, 0.71, 0.61, 0.51],
        'width': [2.3, 1.8, 1.5, 1.2, 0.9],
    }


@pytest.fixture
def calibration_error_data():
    """Sample data for calibration errors."""
    return {
        'alphas': [0.1, 0.2, 0.3, 0.4, 0.5],
        'calibration_errors': [0.01, 0.01, 0.02, 0.01, 0.01],
    }


@pytest.fixture
def alternative_methods_data():
    """Sample data for alternative UQ methods."""
    return {
        'methods': ['CRQR', 'CQR', 'CHR', 'QRF'],
        'scores': [0.92, 0.88, 0.85, 0.81],
    }


@pytest.fixture
def perturbation_data():
    """Sample data for perturbation impact."""
    return {
        'perturbation_levels': [0.01, 0.05, 0.10, 0.15, 0.20],
        'mean_scores': [0.95, 0.92, 0.88, 0.83, 0.78],
        'std_scores': [0.02, 0.03, 0.04, 0.05, 0.06],
    }


@pytest.fixture
def feature_robustness_data():
    """Sample data for feature robustness."""
    return {
        'features': [
            'feature_1',
            'feature_2',
            'feature_3',
            'feature_4',
            'feature_5',
        ],
        'robustness_scores': [0.85, 0.78, 0.92, 0.65, 0.88],
    }


@pytest.fixture
def test_type_data():
    """Sample data for test type comparison."""
    return {
        'test_types': [
            'worst_sample',
            'worst_cluster',
            'outer_sample',
            'hard_sample',
        ],
        'scores': [0.85, 0.82, 0.88, 0.79],
    }


@pytest.fixture
def scenario_data():
    """Sample data for scenario degradation."""
    return {
        'scenarios': ['base', 'scenario_1', 'scenario_2', 'scenario_3'],
        'psi_values': [0.0, 0.15, 0.25, 0.35],
        'performance': [0.95, 0.90, 0.85, 0.80],
    }


@pytest.fixture
def model_comparison_data():
    """Sample data for model comparison."""
    return {
        'models': ['Model A', 'Model B', 'Model C'],
        'metrics': {
            'accuracy': [0.85, 0.88, 0.82],
            'robustness': [0.78, 0.82, 0.85],
            'uncertainty': [0.92, 0.88, 0.90],
        },
    }


@pytest.fixture
def interval_boxplot_data():
    """Sample data for interval boxplot."""
    return {
        'categories': ['Cat A', 'Cat B', 'Cat C'],
        'intervals': [
            {'lower': [1.0, 2.0, 3.0], 'upper': [4.0, 5.0, 6.0]},
            {'lower': [2.0, 3.0, 4.0], 'upper': [5.0, 6.0, 7.0]},
            {'lower': [1.5, 2.5, 3.5], 'upper': [4.5, 5.5, 6.5]},
        ],
    }


# ==================================================================================
# Chart Registration Tests
# ==================================================================================


class TestChartRegistration:
    """Test that all report charts are properly registered."""

    def test_all_uncertainty_charts_registered(self):
        """Test uncertainty-specific charts are registered."""
        uncertainty_charts = [
            'coverage_chart',  # From Phase 2
            'width_vs_coverage',
            'calibration_error',
            'alternative_methods_comparison',
        ]

        for chart_name in uncertainty_charts:
            assert ChartRegistry.is_registered(
                chart_name
            ), f"Chart '{chart_name}' not registered"

    def test_all_robustness_charts_registered(self):
        """Test robustness-specific charts are registered."""
        robustness_charts = ['perturbation_impact', 'feature_robustness']

        for chart_name in robustness_charts:
            assert ChartRegistry.is_registered(
                chart_name
            ), f"Chart '{chart_name}' not registered"

    def test_all_resilience_charts_registered(self):
        """Test resilience-specific charts are registered."""
        resilience_charts = ['test_type_comparison', 'scenario_degradation']

        for chart_name in resilience_charts:
            assert ChartRegistry.is_registered(
                chart_name
            ), f"Chart '{chart_name}' not registered"

    def test_all_general_charts_registered(self):
        """Test general-purpose charts are registered."""
        general_charts = ['model_comparison', 'interval_boxplot']

        for chart_name in general_charts:
            assert ChartRegistry.is_registered(
                chart_name
            ), f"Chart '{chart_name}' not registered"

    def test_static_versions_registered(self):
        """Test static image versions are registered."""
        static_charts = [
            'width_vs_coverage_static',
            'perturbation_impact_static',
        ]

        for chart_name in static_charts:
            assert ChartRegistry.is_registered(
                chart_name
            ), f"Static chart '{chart_name}' not registered"

    def test_minimum_chart_count(self):
        """Test that we have at least 15 charts (Phase 2 + Phase 3)."""
        chart_count = ChartRegistry.count()
        assert (
            chart_count >= 15
        ), f'Expected at least 15 charts, found {chart_count}'


# ==================================================================================
# Uncertainty Charts Tests
# ==================================================================================


class TestUncertaintyCharts:
    """Test uncertainty report charts."""

    def test_width_vs_coverage_generation(self, width_coverage_data):
        """Test width vs coverage chart generation."""
        result = ChartRegistry.generate(
            'width_vs_coverage', width_coverage_data
        )

        assert isinstance(result, ChartResult)
        assert result.is_success
        assert result.format == 'plotly'
        assert len(result.content) > 0

    def test_calibration_error_generation(self, calibration_error_data):
        """Test calibration error chart generation."""
        result = ChartRegistry.generate(
            'calibration_error', calibration_error_data
        )

        assert isinstance(result, ChartResult)
        assert result.is_success
        assert result.format == 'plotly'
        assert len(result.content) > 0

    def test_calibration_error_with_threshold(self, calibration_error_data):
        """Test calibration error with custom threshold."""
        result = ChartRegistry.generate(
            'calibration_error', calibration_error_data, threshold=0.03
        )

        assert result.is_success

    def test_alternative_methods_comparison(self, alternative_methods_data):
        """Test alternative methods comparison chart."""
        result = ChartRegistry.generate(
            'alternative_methods_comparison', alternative_methods_data
        )

        assert isinstance(result, ChartResult)
        assert result.is_success
        assert result.format == 'plotly'

    def test_width_vs_coverage_with_title(self, width_coverage_data):
        """Test chart with custom title."""
        result = ChartRegistry.generate(
            'width_vs_coverage', width_coverage_data, title='Custom Title'
        )

        assert result.is_success
        assert (
            'Custom Title' in result.content
            or result.metadata.get('title') == 'Custom Title'
        )


# ==================================================================================
# Robustness Charts Tests
# ==================================================================================


class TestRobustnessCharts:
    """Test robustness report charts."""

    def test_perturbation_impact_generation(self, perturbation_data):
        """Test perturbation impact chart generation."""
        result = ChartRegistry.generate(
            'perturbation_impact', perturbation_data
        )

        assert isinstance(result, ChartResult)
        assert result.is_success
        assert result.format == 'plotly'

    def test_perturbation_impact_without_std(self):
        """Test perturbation impact without std deviation."""
        data = {
            'perturbation_levels': [0.01, 0.05, 0.10],
            'mean_scores': [0.95, 0.92, 0.88],
        }

        result = ChartRegistry.generate('perturbation_impact', data)
        assert result.is_success

    def test_feature_robustness_generation(self, feature_robustness_data):
        """Test feature robustness chart generation."""
        result = ChartRegistry.generate(
            'feature_robustness', feature_robustness_data
        )

        assert isinstance(result, ChartResult)
        assert result.is_success
        assert result.format == 'plotly'

    def test_feature_robustness_top_n(self, feature_robustness_data):
        """Test feature robustness with top_n parameter."""
        result = ChartRegistry.generate(
            'feature_robustness', feature_robustness_data, top_n=3
        )

        assert result.is_success

    def test_feature_robustness_many_features(self):
        """Test feature robustness with many features."""
        data = {
            'features': [f'feature_{i}' for i in range(20)],
            'robustness_scores': [0.8 + (i * 0.01) for i in range(20)],
        }

        result = ChartRegistry.generate('feature_robustness', data, top_n=10)
        assert result.is_success


# ==================================================================================
# Resilience Charts Tests
# ==================================================================================


class TestResilienceCharts:
    """Test resilience report charts."""

    def test_test_type_comparison_generation(self, test_type_data):
        """Test test type comparison (radar) chart generation."""
        result = ChartRegistry.generate('test_type_comparison', test_type_data)

        assert isinstance(result, ChartResult)
        assert result.is_success
        assert result.format == 'plotly'

    def test_scenario_degradation_generation(self, scenario_data):
        """Test scenario degradation chart generation."""
        result = ChartRegistry.generate('scenario_degradation', scenario_data)

        assert isinstance(result, ChartResult)
        assert result.is_success
        assert result.format == 'plotly'

    def test_test_type_comparison_with_title(self, test_type_data):
        """Test radar chart with custom title."""
        result = ChartRegistry.generate(
            'test_type_comparison', test_type_data, title='Resilience Analysis'
        )

        assert result.is_success


# ==================================================================================
# General Purpose Charts Tests
# ==================================================================================


class TestGeneralPurposeCharts:
    """Test general-purpose charts."""

    def test_model_comparison_generation(self, model_comparison_data):
        """Test model comparison chart generation."""
        result = ChartRegistry.generate(
            'model_comparison', model_comparison_data
        )

        assert isinstance(result, ChartResult)
        assert result.is_success
        assert result.format == 'plotly'

    def test_interval_boxplot_generation(self, interval_boxplot_data):
        """Test interval boxplot chart generation."""
        result = ChartRegistry.generate(
            'interval_boxplot', interval_boxplot_data
        )

        assert isinstance(result, ChartResult)
        assert result.is_success
        assert result.format == 'plotly'

    def test_model_comparison_single_metric(self):
        """Test model comparison with single metric."""
        data = {
            'models': ['Model A', 'Model B'],
            'metrics': {'accuracy': [0.85, 0.88]},
        }

        result = ChartRegistry.generate('model_comparison', data)
        assert result.is_success


# ==================================================================================
# Static Image Charts Tests
# ==================================================================================


class TestStaticImageCharts:
    """Test static image (matplotlib) chart generators."""

    def test_width_vs_coverage_static_generation(self, width_coverage_data):
        """Test static width vs coverage chart."""
        result = ChartRegistry.generate(
            'width_vs_coverage_static', width_coverage_data
        )

        assert isinstance(result, ChartResult)
        assert result.is_success
        assert result.format == 'png'
        assert result.is_base64

    def test_perturbation_impact_static_generation(self, perturbation_data):
        """Test static perturbation impact chart."""
        result = ChartRegistry.generate(
            'perturbation_impact_static', perturbation_data
        )

        assert isinstance(result, ChartResult)
        assert result.is_success
        assert result.format == 'png'
        assert result.is_base64

    def test_static_chart_with_custom_figsize(self, width_coverage_data):
        """Test static chart with custom figure size."""
        result = ChartRegistry.generate(
            'width_vs_coverage_static', width_coverage_data, figsize=(8, 5)
        )

        assert result.is_success


# ==================================================================================
# Error Handling Tests
# ==================================================================================


class TestChartErrorHandling:
    """Test error handling in chart generation."""

    def test_missing_required_field(self):
        """Test error when required field is missing."""
        data = {'coverage': [0.9, 0.8]}  # Missing 'width'

        result = ChartRegistry.generate('width_vs_coverage', data)

        # Should return error result instead of raising
        assert isinstance(result, ChartResult)
        assert not result.is_success
        assert result.error is not None
        assert 'Missing required keys' in result.error

    def test_empty_data(self):
        """Test error with empty data."""
        data = {'coverage': [], 'width': []}

        # Should not raise, but may produce empty chart
        result = ChartRegistry.generate('width_vs_coverage', data)
        assert isinstance(result, ChartResult)

    def test_mismatched_array_lengths(self):
        """Test with mismatched array lengths."""
        data = {
            'coverage': [0.9, 0.8, 0.7],
            'width': [2.0, 1.5],  # Different length
        }

        # Plotly should handle this gracefully
        result = ChartRegistry.generate('width_vs_coverage', data)
        assert isinstance(result, ChartResult)


# ==================================================================================
# Integration Tests
# ==================================================================================


class TestChartIntegration:
    """Integration tests for chart system."""

    def test_generate_all_uncertainty_charts(
        self,
        uncertainty_coverage_data,
        width_coverage_data,
        calibration_error_data,
        alternative_methods_data,
    ):
        """Test generating all uncertainty charts in sequence."""
        charts = {
            'coverage_chart': uncertainty_coverage_data,
            'width_vs_coverage': width_coverage_data,
            'calibration_error': calibration_error_data,
            'alternative_methods_comparison': alternative_methods_data,
        }

        results = {}
        for chart_name, data in charts.items():
            result = ChartRegistry.generate(chart_name, data)
            assert result.is_success, f"Chart '{chart_name}' generation failed"
            results[chart_name] = result

        assert len(results) == 4

    def test_generate_all_robustness_charts(
        self, perturbation_data, feature_robustness_data
    ):
        """Test generating all robustness charts in sequence."""
        charts = {
            'perturbation_impact': perturbation_data,
            'feature_robustness': feature_robustness_data,
        }

        results = {}
        for chart_name, data in charts.items():
            result = ChartRegistry.generate(chart_name, data)
            assert result.is_success, f"Chart '{chart_name}' generation failed"
            results[chart_name] = result

        assert len(results) == 2

    def test_generate_all_resilience_charts(
        self, test_type_data, scenario_data
    ):
        """Test generating all resilience charts in sequence."""
        charts = {
            'test_type_comparison': test_type_data,
            'scenario_degradation': scenario_data,
        }

        results = {}
        for chart_name, data in charts.items():
            result = ChartRegistry.generate(chart_name, data)
            assert result.is_success, f"Chart '{chart_name}' generation failed"
            results[chart_name] = result

        assert len(results) == 2

    def test_mixed_plotly_and_static(self, width_coverage_data):
        """Test generating both Plotly and static versions of same chart."""
        plotly_result = ChartRegistry.generate(
            'width_vs_coverage', width_coverage_data
        )
        static_result = ChartRegistry.generate(
            'width_vs_coverage_static', width_coverage_data
        )

        assert plotly_result.is_success
        assert static_result.is_success
        assert plotly_result.format == 'plotly'
        assert static_result.format == 'png'
        assert plotly_result.is_interactive
        assert static_result.is_base64


# ==================================================================================
# Performance Tests
# ==================================================================================


class TestChartPerformance:
    """Test chart generation performance."""

    def test_multiple_chart_generations(self, width_coverage_data):
        """Test generating same chart multiple times."""
        for _ in range(10):
            result = ChartRegistry.generate(
                'width_vs_coverage', width_coverage_data
            )
            assert result.is_success

    def test_large_dataset(self):
        """Test chart generation with large dataset."""
        data = {
            'coverage': [0.9 - (i * 0.001) for i in range(100)],
            'width': [2.0 - (i * 0.01) for i in range(100)],
        }

        result = ChartRegistry.generate('width_vs_coverage', data)
        assert result.is_success
