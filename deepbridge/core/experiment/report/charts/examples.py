"""
Example chart generators for Phase 2.

Provides concrete implementations demonstrating chart generation patterns.
These serve as templates for Phase 3 expansion.

Created in Phase 2 Sprint 7-8 (TAREFA 7.2).
"""

from typing import Dict, Any
import logging
from .base import PlotlyChartGenerator, StaticImageGenerator, ChartResult
from .registry import ChartRegistry, register_chart

logger = logging.getLogger("deepbridge.reports")


# ==================================================================================
# Example Plotly Charts
# ==================================================================================

@register_chart('line_chart')
class LineChartGenerator(PlotlyChartGenerator):
    """
    Simple line chart generator.

    Data structure:
        {
            'x': [1, 2, 3, ...],
            'y': [4, 5, 6, ...]
        }

    Example:
        >>> generator = LineChartGenerator()
        >>> result = generator.generate(
        ...     data={'x': [1, 2, 3], 'y': [4, 5, 6]},
        ...     title='My Line Chart'
        ... )
    """

    def _create_plotly_figure(self, data: Dict[str, Any], **kwargs) -> Dict:
        """Create Plotly line chart figure."""
        self._validate_data(data, ['x', 'y'])

        figure = {
            'data': [{
                'x': data['x'],
                'y': data['y'],
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': kwargs.get('series_name', 'Series'),
                'line': {'color': kwargs.get('color', '#1f77b4'), 'width': 2},
                'marker': {'size': 6}
            }],
            'layout': {
                'title': kwargs.get('title', 'Line Chart'),
                'xaxis': {'title': kwargs.get('xaxis_title', 'X')},
                'yaxis': {'title': kwargs.get('yaxis_title', 'Y')},
                'template': 'plotly_white',
                'hovermode': 'x unified'
            }
        }

        return figure


@register_chart('bar_chart')
class BarChartGenerator(PlotlyChartGenerator):
    """
    Simple bar chart generator.

    Data structure:
        {
            'labels': ['A', 'B', 'C', ...],
            'values': [10, 20, 15, ...]
        }

    Example:
        >>> generator = BarChartGenerator()
        >>> result = generator.generate(
        ...     data={'labels': ['A', 'B', 'C'], 'values': [10, 20, 15]},
        ...     title='My Bar Chart'
        ... )
    """

    def _create_plotly_figure(self, data: Dict[str, Any], **kwargs) -> Dict:
        """Create Plotly bar chart figure."""
        self._validate_data(data, ['labels', 'values'])

        figure = {
            'data': [{
                'x': data['labels'],
                'y': data['values'],
                'type': 'bar',
                'name': kwargs.get('series_name', 'Values'),
                'marker': {
                    'color': kwargs.get('color', '#2ca02c'),
                    'line': {'width': 1, 'color': 'white'}
                }
            }],
            'layout': {
                'title': kwargs.get('title', 'Bar Chart'),
                'xaxis': {'title': kwargs.get('xaxis_title', 'Category')},
                'yaxis': {'title': kwargs.get('yaxis_title', 'Value')},
                'template': 'plotly_white',
                'hovermode': 'x'
            }
        }

        return figure


@register_chart('coverage_chart')
class CoverageChartGenerator(PlotlyChartGenerator):
    """
    Coverage vs Expected chart for uncertainty reports.

    Data structure:
        {
            'alphas': [0.1, 0.2, 0.3, ...],
            'coverage': [0.91, 0.81, 0.71, ...],
            'expected': [0.90, 0.80, 0.70, ...]
        }

    This is a practical example for Phase 3 uncertainty reports.
    """

    def _create_plotly_figure(self, data: Dict[str, Any], **kwargs) -> Dict:
        """Create coverage vs expected chart."""
        self._validate_data(data, ['alphas', 'coverage', 'expected'])

        figure = {
            'data': [
                {
                    'x': data['alphas'],
                    'y': data['coverage'],
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Actual Coverage',
                    'line': {'color': '#1f77b4', 'width': 2},
                    'marker': {'size': 8}
                },
                {
                    'x': data['alphas'],
                    'y': data['expected'],
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': 'Expected Coverage',
                    'line': {
                        'color': '#ff7f0e',
                        'width': 2,
                        'dash': 'dash'
                    }
                }
            ],
            'layout': {
                'title': kwargs.get('title', 'Coverage vs Expected'),
                'xaxis': {
                    'title': 'Alpha Level',
                    'tickformat': '.1f'
                },
                'yaxis': {
                    'title': 'Coverage',
                    'tickformat': '.2f',
                    'range': [0, 1.05]
                },
                'template': 'plotly_white',
                'hovermode': 'x unified',
                'legend': {'x': 0.02, 'y': 0.98}
            }
        }

        return figure


# ==================================================================================
# Example Static Image Chart
# ==================================================================================

class SimpleBarImageGenerator(StaticImageGenerator):
    """
    Simple bar chart as static PNG image.

    Demonstrates how to create static images (for PDF reports in Phase 4).
    Note: Requires matplotlib.
    """

    def _create_image(self, data: Dict[str, Any], **kwargs) -> tuple[str, str]:
        """Create bar chart as base64 PNG."""
        self._validate_data(data, ['labels', 'values'])

        try:
            import matplotlib.pyplot as plt
            import io
            import base64

            # Create figure
            fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))

            # Plot
            ax.bar(data['labels'], data['values'], color=kwargs.get('color', '#2ca02c'))

            # Styling
            ax.set_title(kwargs.get('title', 'Bar Chart'))
            ax.set_xlabel(kwargs.get('xlabel', 'Category'))
            ax.set_ylabel(kwargs.get('ylabel', 'Value'))
            ax.grid(True, alpha=0.3, axis='y')

            # Convert to base64
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close(fig)

            return img_base64, 'png'

        except ImportError:
            logger.warning("Matplotlib not available, returning error")
            raise ValueError("Matplotlib required for static images")


# ==================================================================================
# Bulk Registration Helper
# ==================================================================================

def register_example_charts():
    """
    Register all example charts.

    This function is called automatically when the module is imported,
    but can also be called manually if needed.

    Example:
        >>> from deepbridge.core.experiment.report.charts.examples import register_example_charts
        >>> register_example_charts()
        >>> print(ChartRegistry.list_charts())
    """
    # Charts are already registered via @register_chart decorator
    # This function exists for explicit re-registration if needed

    # Register static image chart (not decorated)
    if not ChartRegistry.is_registered('bar_image'):
        ChartRegistry.register('bar_image', SimpleBarImageGenerator())
        logger.info("Registered static bar image chart")


# Auto-register on import
register_example_charts()


# ==================================================================================
# Main - Example Usage
# ==================================================================================

if __name__ == "__main__":
    """
    Demonstrate example chart generators.
    """
    print("=" * 80)
    print("Example Chart Generators")
    print("=" * 80)

    # Show registered charts
    print(f"\nRegistered charts: {ChartRegistry.list_charts()}")
    print(f"Total: {ChartRegistry.count()}")

    # Example 1: Line chart
    print("\n" + "-" * 80)
    print("Example 1: Line Chart")
    print("-" * 80)

    line_data = {
        'x': [1, 2, 3, 4, 5],
        'y': [2.1, 4.3, 3.8, 5.2, 6.1]
    }

    result = ChartRegistry.generate(
        'line_chart',
        data=line_data,
        title='Performance Over Time',
        xaxis_title='Iteration',
        yaxis_title='Score'
    )

    print(f"Result: {result}")
    print(f"Success: {result.is_success}")
    print(f"Format: {result.format}")

    # Example 2: Bar chart
    print("\n" + "-" * 80)
    print("Example 2: Bar Chart")
    print("-" * 80)

    bar_data = {
        'labels': ['Model A', 'Model B', 'Model C', 'Model D'],
        'values': [0.85, 0.92, 0.78, 0.88]
    }

    result = ChartRegistry.generate(
        'bar_chart',
        data=bar_data,
        title='Model Comparison',
        yaxis_title='Accuracy'
    )

    print(f"Result: {result}")
    print(f"Success: {result.is_success}")

    # Example 3: Coverage chart (uncertainty-specific)
    print("\n" + "-" * 80)
    print("Example 3: Coverage Chart")
    print("-" * 80)

    coverage_data = {
        'alphas': [0.1, 0.2, 0.3, 0.4, 0.5],
        'coverage': [0.91, 0.81, 0.72, 0.61, 0.51],
        'expected': [0.90, 0.80, 0.70, 0.60, 0.50]
    }

    result = ChartRegistry.generate(
        'coverage_chart',
        data=coverage_data,
        title='Calibration Analysis'
    )

    print(f"Result: {result}")
    print(f"Success: {result.is_success}")

    print("\n" + "=" * 80)
    print("Example Chart Generators Complete")
    print("=" * 80)
