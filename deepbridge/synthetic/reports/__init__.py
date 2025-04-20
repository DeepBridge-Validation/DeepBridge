"""
Report generation for synthetic data quality.

This module provides tools for creating detailed quality reports
comparing real and synthetic data, with visualizations and metrics.

Usage:
    from synthetic.reports.report_generator import generate_quality_report
    report_path = generate_quality_report(real_data, synthetic_data, metrics)
"""

from .report_generator import generate_quality_report