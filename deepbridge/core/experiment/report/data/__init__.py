"""
Data layer for report generation.

This package contains data classes and transformers for converting raw
experiment results into typed data structures suitable for rendering.
"""

from .base import ReportData, DataTransformer

__all__ = [
    'ReportData',
    'DataTransformer',
]
