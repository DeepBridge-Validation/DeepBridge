"""
Static renderers package for generating non-interactive HTML reports with Seaborn.
"""

from .base_static_renderer import BaseStaticRenderer
from .static_resilience_renderer import StaticResilienceRenderer
from .static_robustness_renderer import StaticRobustnessRenderer
from .static_uncertainty_renderer import StaticUncertaintyRenderer

# NOTE: the old report.utils.resilience_charts module was removed. The chart
# generator now lives in deepbridge.templates.report_types.resilience.static
# .charts and is imported directly by StaticResilienceRenderer, so there is no
# optional re-export here any more.

__all__ = [
    'BaseStaticRenderer',
    'StaticRobustnessRenderer',
    'StaticUncertaintyRenderer',
    'StaticResilienceRenderer',
]
