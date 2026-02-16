"""
Template engine for report generation.

This package contains the template engine and related utilities for
rendering HTML reports using Jinja2.
"""

from .engine import TemplateEngine
from .filters import register_custom_filters

__all__ = [
    'TemplateEngine',
    'register_custom_filters',
]
