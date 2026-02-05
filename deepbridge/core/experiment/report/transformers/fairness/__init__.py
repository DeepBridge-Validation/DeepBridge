"""
Fairness report transformer module.

Provides data transformation and visualization for fairness analysis reports.
"""

from .chart_factory import ChartFactory
from .data_transformer import FairnessDataTransformer

__all__ = ['FairnessDataTransformer', 'ChartFactory']
