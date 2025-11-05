"""
Chart generation system for reports.

Provides base classes and registry for managing chart generators
(Phase 2 Sprint 7-8).

Example Usage:
    >>> from deepbridge.core.experiment.report.charts import (
    ...     ChartRegistry,
    ...     ChartGenerator,
    ...     ChartResult
    ... )
    >>>
    >>> # Register a chart
    >>> ChartRegistry.register('my_chart', MyChartGenerator())
    >>>
    >>> # Generate chart
    >>> result = ChartRegistry.generate('my_chart', data={'x': [1,2], 'y': [3,4]})
    >>>
    >>> # Check result
    >>> if result.is_success:
    ...     print(f"Chart content: {result.content}")
"""

from .base import (
    ChartResult,
    ChartGenerator,
    PlotlyChartGenerator,
    StaticImageGenerator
)

from .registry import (
    ChartRegistry,
    register_chart
)

__all__ = [
    # Base classes
    'ChartResult',
    'ChartGenerator',
    'PlotlyChartGenerator',
    'StaticImageGenerator',
    # Registry
    'ChartRegistry',
    'register_chart',
]
