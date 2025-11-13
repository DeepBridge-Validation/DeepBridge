"""
Fairness report transformer module.

Provides data transformation and visualization for fairness analysis reports.
"""

from .data_transformer import FairnessDataTransformer
from .chart_factory import ChartFactory

__all__ = ['FairnessDataTransformer', 'ChartFactory']
