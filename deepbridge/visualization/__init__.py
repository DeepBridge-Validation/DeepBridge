"""
Visualization module for DeepBridge.

This module provides visualization tools for analyzing model performance,
distillation results, distribution similarities, and robustness testing.
"""

from deepbridge.visualization.base_visualizer import BaseVisualizer
from deepbridge.visualization.distribution import DistributionVisualizer, DistributionPlots
from deepbridge.visualization.metrics import MetricsVisualizer
from deepbridge.visualization.auto_visualizer import AutoVisualizer

# Abstract interfaces
from deepbridge.visualization.robustness.base_robustness_visualizer import BaseRobustnessVisualizer
from deepbridge.visualization.uncertainty.base_uncertainty_visualizer import BaseUncertaintyVisualizer
from deepbridge.visualization.synthetic.base_synthetic_visualizer import BaseSyntheticVisualizer

# Concrete implementations
from deepbridge.visualization.synthetic.distribution_viz import DistributionViz
from deepbridge.visualization.synthetic.performance_viz import PerformanceViz

__all__ = [
    # Abstract interfaces
    "BaseVisualizer",
    "BaseRobustnessVisualizer",
    "BaseUncertaintyVisualizer",
    "BaseSyntheticVisualizer",
    
    # Concrete implementations
    "DistributionViz", 
    "PerformanceViz",
    
    # Legacy visualizers
    "DistributionVisualizer",
    "DistributionPlots",
    "MetricsVisualizer",
    "AutoVisualizer"
]