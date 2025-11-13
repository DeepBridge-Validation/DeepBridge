"""
Legacy fairness charts - DEPRECATED.

This module provides backward compatibility by re-exporting deprecated chart classes.
For new code, use the specialized chart modules instead:
- posttrain_charts.py
- pretrain_charts.py
- complementary_charts.py
- distribution_charts.py

These legacy charts will be removed in a future version.
"""

import warnings


def _deprecated_chart_warning(chart_name: str):
    """Issue deprecation warning when legacy chart is instantiated."""
    warnings.warn(
        f"{chart_name} is deprecated and will be removed in a future version. "
        f"Use the specialized chart classes from fairness.charts instead.",
        DeprecationWarning,
        stacklevel=3
    )


# Import from deprecated module
from ..deprecated.legacy_charts import (
    MetricsComparisonChart as _MetricsComparisonChart,
    FairnessRadarChart as _FairnessRadarChart,
    ConfusionMatricesChart as _ConfusionMatricesChart,
    ThresholdAnalysisChart as _ThresholdAnalysisChart
)


# Wrapper classes that emit deprecation warnings
class MetricsComparisonChart(_MetricsComparisonChart):
    """DEPRECATED: Use specialized posttrain charts instead."""

    def __init__(self):
        _deprecated_chart_warning('MetricsComparisonChart')
        super().__init__()


class FairnessRadarChart(_FairnessRadarChart):
    """DEPRECATED: Use specialized radar charts instead."""

    def __init__(self):
        _deprecated_chart_warning('FairnessRadarChart')
        super().__init__()


class ConfusionMatricesChart(_ConfusionMatricesChart):
    """DEPRECATED: Use complementary charts instead."""

    def __init__(self):
        _deprecated_chart_warning('ConfusionMatricesChart')
        super().__init__()


class ThresholdAnalysisChart(_ThresholdAnalysisChart):
    """DEPRECATED: Consider using posttrain charts for threshold analysis."""

    def __init__(self):
        _deprecated_chart_warning('ThresholdAnalysisChart')
        super().__init__()


__all__ = [
    'MetricsComparisonChart',
    'FairnessRadarChart',
    'ConfusionMatricesChart',
    'ThresholdAnalysisChart'
]
