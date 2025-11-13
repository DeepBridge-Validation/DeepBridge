"""
Deprecated fairness chart modules.

These charts are maintained for backward compatibility but are discouraged for new code.
They will be removed in a future version.

Prefer using the newer, specialized chart modules from:
- fairness.charts.posttrain_charts
- fairness.charts.pretrain_charts
- fairness.charts.complementary_charts
- fairness.charts.distribution_charts
"""

import warnings


def _emit_deprecation_warning(chart_name: str):
    """Emit a deprecation warning for a legacy chart."""
    warnings.warn(
        f"{chart_name} is deprecated and will be removed in a future version. "
        f"Please use the specialized chart classes from fairness.charts instead.",
        DeprecationWarning,
        stacklevel=3
    )


from .legacy_charts import (
    MetricsComparisonChart,
    FairnessRadarChart,
    ConfusionMatricesChart,
    ThresholdAnalysisChart
)

__all__ = [
    'MetricsComparisonChart',
    'FairnessRadarChart',
    'ConfusionMatricesChart',
    'ThresholdAnalysisChart'
]
