"""
Static renderers package for generating non-interactive HTML reports with Seaborn.
"""

from .base_static_renderer import BaseStaticRenderer
from .static_robustness_renderer import StaticRobustnessRenderer
from .static_uncertainty_renderer import StaticUncertaintyRenderer
from .static_resilience_renderer import StaticResilienceRenderer

# Make sure we expose ResilienceChartGenerator if available
try:
    from ...utils.resilience_charts import ResilienceChartGenerator
    has_resilience_charts = True
except ImportError:
    has_resilience_charts = False

__all__ = [
    'BaseStaticRenderer',
    'StaticRobustnessRenderer',
    'StaticUncertaintyRenderer',
    'StaticResilienceRenderer',
]

# Add ResilienceChartGenerator to __all__ if available
if has_resilience_charts:
    __all__.append('ResilienceChartGenerator')