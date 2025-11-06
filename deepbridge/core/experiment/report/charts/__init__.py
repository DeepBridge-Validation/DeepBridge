"""
Chart generation system for reports.

Provides base classes and registry for managing chart generators.
Phase 2 Sprint 7-8 (basic infrastructure).
Phase 3 Sprint 9 (complete chart library).

Example Usage:
    >>> from deepbridge.core.experiment.report.charts import (
    ...     ChartRegistry,
    ...     ChartGenerator,
    ...     ChartResult
    ... )
    >>>
    >>> # Generate chart
    >>> result = ChartRegistry.generate(
    ...     'width_vs_coverage',
    ...     data={'coverage': [0.9, 0.8], 'width': [2.0, 1.5]}
    ... )
    >>>
    >>> # Check result
    >>> if result.is_success:
    ...     print(f"Chart generated: {result.format}")
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

# Auto-register all chart generators
from . import examples  # Phase 2 examples
from . import report_charts  # Phase 3 complete library

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
