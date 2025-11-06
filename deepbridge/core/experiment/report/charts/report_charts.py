"""
Report-specific chart generators for Phase 3.

Provides complete set of chart generators for uncertainty, robustness,
and resilience reports with both Plotly (interactive) and Matplotlib
(static) implementations.

Created in Phase 3 Sprint 9 (TAREFA 9.1).
"""

from typing import Dict, Any, List, Optional
import logging
from .base import PlotlyChartGenerator, StaticImageGenerator, ChartResult
from .registry import ChartRegistry, register_chart

logger = logging.getLogger("deepbridge.reports")


# ==================================================================================
# Uncertainty Report Charts (Plotly)
# ==================================================================================

@register_chart('width_vs_coverage')
class WidthVsCoverageChart(PlotlyChartGenerator):
    """
    Width vs Coverage chart for uncertainty reports.

    Data structure:
        {
            'coverage': [0.91, 0.81, 0.71, ...],
            'width': [2.3, 1.8, 1.5, ...]
        }

    Shows trade-off between prediction interval width and coverage.
    """

    def _create_plotly_figure(self, data: Dict[str, Any], **kwargs) -> Dict:
        """Create width vs coverage scatter plot."""
        self._validate_data(data, ['coverage', 'width'])

        figure = {
            'data': [{
                'x': data['coverage'],
                'y': data['width'],
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Width-Coverage Trade-off',
                'line': {'color': '#2ca02c', 'width': 2},
                'marker': {'size': 8, 'color': '#2ca02c'}
            }],
            'layout': {
                'title': kwargs.get('title', 'Width vs Coverage Trade-off'),
                'xaxis': {
                    'title': 'Coverage',
                    'tickformat': '.2f',
                    'range': [0, 1.05]
                },
                'yaxis': {
                    'title': 'Average Width',
                    'tickformat': '.2f'
                },
                'template': 'plotly_white',
                'hovermode': 'closest',
                'showlegend': False
            }
        }

        return figure


@register_chart('calibration_error')
class CalibrationErrorChart(PlotlyChartGenerator):
    """
    Calibration error by alpha level for uncertainty reports.

    Data structure:
        {
            'alphas': [0.1, 0.2, 0.3, ...],
            'calibration_errors': [0.01, 0.01, 0.02, ...]
        }

    Shows how well calibrated the model is at each alpha level.
    """

    def _create_plotly_figure(self, data: Dict[str, Any], **kwargs) -> Dict:
        """Create calibration error bar chart."""
        self._validate_data(data, ['alphas', 'calibration_errors'])

        # Color bars based on threshold (default 0.05)
        threshold = kwargs.get('threshold', 0.05)
        colors = [
            '#d62728' if err > threshold else '#2ca02c'
            for err in data['calibration_errors']
        ]

        figure = {
            'data': [{
                'x': [f"{a:.1f}" for a in data['alphas']],
                'y': data['calibration_errors'],
                'type': 'bar',
                'marker': {'color': colors},
                'text': [f"{e:.3f}" for e in data['calibration_errors']],
                'textposition': 'auto'
            }],
            'layout': {
                'title': kwargs.get('title', 'Calibration Error by Alpha Level'),
                'xaxis': {'title': 'Alpha Level'},
                'yaxis': {
                    'title': 'Calibration Error',
                    'tickformat': '.3f'
                },
                'template': 'plotly_white',
                'showlegend': False,
                'shapes': [{
                    'type': 'line',
                    'x0': -0.5,
                    'x1': len(data['alphas']) - 0.5,
                    'y0': threshold,
                    'y1': threshold,
                    'line': {'color': 'red', 'dash': 'dash', 'width': 2}
                }]
            }
        }

        return figure


@register_chart('alternative_methods_comparison')
class AlternativeMethodsChart(PlotlyChartGenerator):
    """
    Comparison of alternative UQ methods.

    Data structure:
        {
            'methods': ['CRQR', 'CQR', 'CHR', 'QRF'],
            'scores': [0.92, 0.88, 0.85, 0.81]
        }

    Compares different uncertainty quantification methods.
    """

    def _create_plotly_figure(self, data: Dict[str, Any], **kwargs) -> Dict:
        """Create horizontal bar chart of methods."""
        self._validate_data(data, ['methods', 'scores'])

        figure = {
            'data': [{
                'x': data['scores'],
                'y': data['methods'],
                'type': 'bar',
                'orientation': 'h',
                'marker': {
                    'color': data['scores'],
                    'colorscale': 'Viridis',
                    'showscale': False
                },
                'text': [f"{s:.3f}" for s in data['scores']],
                'textposition': 'auto'
            }],
            'layout': {
                'title': kwargs.get('title', 'Alternative UQ Methods Comparison'),
                'xaxis': {
                    'title': 'Uncertainty Score',
                    'tickformat': '.2f',
                    'range': [0, 1]
                },
                'yaxis': {'title': ''},
                'template': 'plotly_white',
                'showlegend': False
            }
        }

        return figure


# ==================================================================================
# Robustness Report Charts (Plotly)
# ==================================================================================

@register_chart('perturbation_impact')
class PerturbationImpactChart(PlotlyChartGenerator):
    """
    Performance degradation by perturbation level.

    Data structure:
        {
            'perturbation_levels': [0.01, 0.05, 0.10, 0.15, 0.20],
            'mean_scores': [0.95, 0.92, 0.88, 0.83, 0.78],
            'std_scores': [0.02, 0.03, 0.04, 0.05, 0.06]
        }

    Shows how performance degrades with increasing perturbation.
    """

    def _create_plotly_figure(self, data: Dict[str, Any], **kwargs) -> Dict:
        """Create perturbation impact chart with error bars."""
        self._validate_data(data, ['perturbation_levels', 'mean_scores'])

        std_scores = data.get('std_scores', [0] * len(data['mean_scores']))

        figure = {
            'data': [{
                'x': data['perturbation_levels'],
                'y': data['mean_scores'],
                'error_y': {
                    'type': 'data',
                    'array': std_scores,
                    'visible': True
                },
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Mean Performance',
                'line': {'color': '#d62728', 'width': 2},
                'marker': {'size': 10}
            }],
            'layout': {
                'title': kwargs.get('title', 'Performance vs Perturbation Level'),
                'xaxis': {
                    'title': 'Perturbation Level',
                    'tickformat': '.2f'
                },
                'yaxis': {
                    'title': 'Performance Score',
                    'tickformat': '.2f',
                    'range': [0, 1.05]
                },
                'template': 'plotly_white',
                'hovermode': 'x unified'
            }
        }

        return figure


@register_chart('feature_robustness')
class FeatureRobustnessChart(PlotlyChartGenerator):
    """
    Feature importance under perturbation.

    Data structure:
        {
            'features': ['feature_1', 'feature_2', ...],
            'robustness_scores': [0.85, 0.78, 0.92, ...],
            'impacts': [0.15, 0.22, 0.08, ...]  # Optional
        }

    Shows which features are most robust to perturbation.
    """

    def _create_plotly_figure(self, data: Dict[str, Any], **kwargs) -> Dict:
        """Create horizontal bar chart of feature robustness."""
        self._validate_data(data, ['features', 'robustness_scores'])

        # Sort by robustness score
        sorted_data = sorted(
            zip(data['features'], data['robustness_scores']),
            key=lambda x: x[1],
            reverse=True
        )

        # Limit to top N features
        top_n = kwargs.get('top_n', 10)
        features, scores = zip(*sorted_data[:top_n])

        # Color by robustness (green = robust, red = sensitive)
        colors = [
            '#2ca02c' if s > 0.8 else '#ff7f0e' if s > 0.6 else '#d62728'
            for s in scores
        ]

        figure = {
            'data': [{
                'x': list(scores),
                'y': list(features),
                'type': 'bar',
                'orientation': 'h',
                'marker': {'color': colors},
                'text': [f"{s:.2f}" for s in scores],
                'textposition': 'auto'
            }],
            'layout': {
                'title': kwargs.get('title', f'Top {top_n} Feature Robustness'),
                'xaxis': {
                    'title': 'Robustness Score',
                    'tickformat': '.2f',
                    'range': [0, 1]
                },
                'yaxis': {'title': ''},
                'template': 'plotly_white',
                'showlegend': False,
                'height': max(400, len(features) * 30)
            }
        }

        return figure


# ==================================================================================
# Resilience Report Charts (Plotly)
# ==================================================================================

@register_chart('test_type_comparison')
class TestTypeComparisonChart(PlotlyChartGenerator):
    """
    Comparison across different resilience test types.

    Data structure:
        {
            'test_types': ['worst_sample', 'worst_cluster', 'outer_sample', ...],
            'scores': [0.85, 0.82, 0.88, ...]
        }

    Shows performance across different resilience tests.
    """

    def _create_plotly_figure(self, data: Dict[str, Any], **kwargs) -> Dict:
        """Create radar chart for test type comparison."""
        self._validate_data(data, ['test_types', 'scores'])

        # Close the radar chart by repeating first point
        test_types = data['test_types'] + [data['test_types'][0]]
        scores = data['scores'] + [data['scores'][0]]

        figure = {
            'data': [{
                'type': 'scatterpolar',
                'r': scores,
                'theta': test_types,
                'fill': 'toself',
                'fillcolor': 'rgba(31, 119, 180, 0.2)',
                'line': {'color': '#1f77b4', 'width': 2},
                'marker': {'size': 8}
            }],
            'layout': {
                'title': kwargs.get('title', 'Resilience Across Test Types'),
                'polar': {
                    'radialaxis': {
                        'visible': True,
                        'range': [0, 1],
                        'tickformat': '.2f'
                    }
                },
                'template': 'plotly_white',
                'showlegend': False
            }
        }

        return figure


@register_chart('scenario_degradation')
class ScenarioDegradationChart(PlotlyChartGenerator):
    """
    Performance degradation across scenarios.

    Data structure:
        {
            'scenarios': ['base', 'scenario_1', 'scenario_2', ...],
            'psi_values': [0.0, 0.15, 0.25, ...],
            'performance': [0.95, 0.90, 0.85, ...]
        }

    Shows how performance degrades with distribution shift (PSI).
    """

    def _create_plotly_figure(self, data: Dict[str, Any], **kwargs) -> Dict:
        """Create scatter plot of PSI vs performance."""
        self._validate_data(data, ['scenarios', 'psi_values', 'performance'])

        figure = {
            'data': [{
                'x': data['psi_values'],
                'y': data['performance'],
                'type': 'scatter',
                'mode': 'markers+text',
                'marker': {
                    'size': 12,
                    'color': data['psi_values'],
                    'colorscale': 'RdYlGn_r',
                    'showscale': True,
                    'colorbar': {'title': 'PSI Value'}
                },
                'text': data['scenarios'],
                'textposition': 'top center'
            }],
            'layout': {
                'title': kwargs.get('title', 'Performance vs Distribution Shift'),
                'xaxis': {
                    'title': 'PSI (Population Stability Index)',
                    'tickformat': '.2f'
                },
                'yaxis': {
                    'title': 'Performance',
                    'tickformat': '.2f',
                    'range': [0, 1.05]
                },
                'template': 'plotly_white',
                'hovermode': 'closest'
            }
        }

        return figure


# ==================================================================================
# General Purpose Charts
# ==================================================================================

@register_chart('model_comparison')
class ModelComparisonChart(PlotlyChartGenerator):
    """
    Multi-metric model comparison chart.

    Data structure:
        {
            'models': ['Model A', 'Model B', 'Model C'],
            'metrics': {
                'accuracy': [0.85, 0.88, 0.82],
                'robustness': [0.78, 0.82, 0.85],
                'uncertainty': [0.92, 0.88, 0.90]
            }
        }

    Compares multiple models across multiple metrics.
    """

    def _create_plotly_figure(self, data: Dict[str, Any], **kwargs) -> Dict:
        """Create grouped bar chart for model comparison."""
        self._validate_data(data, ['models', 'metrics'])

        traces = []
        for metric_name, values in data['metrics'].items():
            traces.append({
                'x': data['models'],
                'y': values,
                'type': 'bar',
                'name': metric_name.capitalize(),
                'text': [f"{v:.2f}" for v in values],
                'textposition': 'auto'
            })

        figure = {
            'data': traces,
            'layout': {
                'title': kwargs.get('title', 'Model Comparison'),
                'xaxis': {'title': 'Model'},
                'yaxis': {
                    'title': 'Score',
                    'tickformat': '.2f',
                    'range': [0, 1.05]
                },
                'template': 'plotly_white',
                'barmode': 'group',
                'hovermode': 'x unified'
            }
        }

        return figure


@register_chart('interval_boxplot')
class IntervalBoxplotChart(PlotlyChartGenerator):
    """
    Boxplot of prediction intervals by category.

    Data structure:
        {
            'categories': ['Cat A', 'Cat B', 'Cat C'],
            'intervals': [
                {'lower': [1, 2, 3, ...], 'upper': [4, 5, 6, ...]},
                {'lower': [2, 3, 4, ...], 'upper': [5, 6, 7, ...]},
                ...
            ]
        }

    Shows distribution of prediction interval widths by category.
    """

    def _create_plotly_figure(self, data: Dict[str, Any], **kwargs) -> Dict:
        """Create boxplot of interval widths."""
        self._validate_data(data, ['categories', 'intervals'])

        traces = []
        for i, category in enumerate(data['categories']):
            interval_data = data['intervals'][i]
            widths = [
                u - l for u, l in zip(interval_data['upper'], interval_data['lower'])
            ]

            traces.append({
                'y': widths,
                'type': 'box',
                'name': category,
                'boxmean': 'sd'
            })

        figure = {
            'data': traces,
            'layout': {
                'title': kwargs.get('title', 'Prediction Interval Widths by Category'),
                'xaxis': {'title': 'Category'},
                'yaxis': {
                    'title': 'Interval Width',
                    'tickformat': '.2f'
                },
                'template': 'plotly_white',
                'showlegend': False
            }
        }

        return figure


# ==================================================================================
# Matplotlib Static Versions
# ==================================================================================

class WidthVsCoverageStatic(StaticImageGenerator):
    """Static PNG version of width vs coverage chart."""

    def _create_image(self, data: Dict[str, Any], **kwargs) -> tuple[str, str]:
        """Create static width vs coverage chart."""
        self._validate_data(data, ['coverage', 'width'])

        try:
            import matplotlib.pyplot as plt
            import io
            import base64

            fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))

            ax.plot(data['coverage'], data['width'], 'o-', color='#2ca02c',
                   linewidth=2, markersize=8)

            ax.set_xlabel('Coverage', fontsize=12)
            ax.set_ylabel('Average Width', fontsize=12)
            ax.set_title(kwargs.get('title', 'Width vs Coverage Trade-off'),
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1.05)

            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close(fig)

            return img_base64, 'png'

        except ImportError:
            raise ValueError("Matplotlib required for static images")


class PerturbationImpactStatic(StaticImageGenerator):
    """Static PNG version of perturbation impact chart."""

    def _create_image(self, data: Dict[str, Any], **kwargs) -> tuple[str, str]:
        """Create static perturbation impact chart."""
        self._validate_data(data, ['perturbation_levels', 'mean_scores'])

        try:
            import matplotlib.pyplot as plt
            import io
            import base64
            import numpy as np

            fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))

            std_scores = data.get('std_scores', [0] * len(data['mean_scores']))

            ax.errorbar(data['perturbation_levels'], data['mean_scores'],
                       yerr=std_scores, fmt='o-', color='#d62728',
                       linewidth=2, markersize=8, capsize=5)

            ax.set_xlabel('Perturbation Level', fontsize=12)
            ax.set_ylabel('Performance Score', fontsize=12)
            ax.set_title(kwargs.get('title', 'Performance vs Perturbation Level'),
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.05)

            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close(fig)

            return img_base64, 'png'

        except ImportError:
            raise ValueError("Matplotlib required for static images")


# ==================================================================================
# Bulk Registration
# ==================================================================================

def register_report_charts():
    """
    Register all report-specific charts including static versions.

    This function is called automatically when the module is imported,
    but can also be called manually if needed.
    """
    # Plotly charts are already registered via @register_chart decorator

    # Register static versions
    static_charts = {
        'width_vs_coverage_static': WidthVsCoverageStatic(),
        'perturbation_impact_static': PerturbationImpactStatic()
    }

    for name, generator in static_charts.items():
        if not ChartRegistry.is_registered(name):
            ChartRegistry.register(name, generator)
            logger.info(f"Registered static chart: {name}")


# Auto-register on import
register_report_charts()


# ==================================================================================
# Main - Example Usage
# ==================================================================================

if __name__ == "__main__":
    """Demonstrate report-specific charts."""
    print("=" * 80)
    print("Report-Specific Chart Generators")
    print("=" * 80)

    print(f"\nTotal registered charts: {ChartRegistry.count()}")
    print(f"Available charts: {ChartRegistry.list_charts()}")

    # Example: Width vs Coverage
    print("\n" + "-" * 80)
    print("Example 1: Width vs Coverage")
    print("-" * 80)

    result = ChartRegistry.generate(
        'width_vs_coverage',
        data={
            'coverage': [0.91, 0.81, 0.71, 0.61, 0.51],
            'width': [2.3, 1.8, 1.5, 1.2, 0.9]
        },
        title='Width-Coverage Trade-off'
    )

    print(f"Result: {result}")
    print(f"Success: {result.is_success}")

    # Example: Test Type Comparison
    print("\n" + "-" * 80)
    print("Example 2: Test Type Comparison (Radar)")
    print("-" * 80)

    result = ChartRegistry.generate(
        'test_type_comparison',
        data={
            'test_types': ['worst_sample', 'worst_cluster', 'outer_sample', 'hard_sample'],
            'scores': [0.85, 0.82, 0.88, 0.79]
        }
    )

    print(f"Result: {result}")
    print(f"Success: {result.is_success}")

    print("\n" + "=" * 80)
    print(f"✅ {ChartRegistry.count()} charts registered and ready!")
    print("=" * 80)
