"""
Utility module for mapping chart names properly between different components.

This module ensures consistent chart naming between transformer, renderer and template.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger("deepbridge.reports")

def ensure_chart_mappings(charts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure chart mappings are consistent by duplicating charts under all aliases.
    
    Args:
        charts: Dictionary of chart name to chart content
        
    Returns:
        Updated charts dictionary with all mappings
    """
    chart_mappings = {
        # Bidirectional mappings for reliability charts
        'reliability_distribution': ['feature_reliability', 'reliability_analysis'],  
        'feature_reliability': ['reliability_distribution', 'reliability_analysis'],
        'reliability_analysis': ['feature_reliability', 'reliability_distribution'],
        
        # Bidirectional mappings for bandwidth charts
        'marginal_bandwidth': ['interval_widths_comparison', 'width_distribution', 'interval_widths_boxplot'],
        'interval_widths_comparison': ['marginal_bandwidth', 'width_distribution', 'interval_widths_boxplot'],
        'width_distribution': ['interval_widths_comparison', 'marginal_bandwidth', 'interval_widths_boxplot'],
        'interval_widths_boxplot': ['interval_widths_comparison', 'marginal_bandwidth', 'width_distribution'],
        
        # Bidirectional mappings for model comparison charts
        'model_comparison': ['model_metrics_comparison', 'model_comparison_chart', 'model_metrics'],
        'model_metrics_comparison': ['model_comparison', 'model_comparison_chart', 'model_metrics'],
        'model_comparison_chart': ['model_comparison', 'model_metrics_comparison', 'model_metrics'],
        'model_metrics': ['model_comparison', 'model_metrics_comparison', 'model_comparison_chart'],
        
        # Other charts
        'performance_gap_by_alpha': [],
        'coverage_vs_expected': [],
        'width_vs_coverage': [],
        'uncertainty_metrics': []
    }
    
    # Create a new dictionary to hold all the mappings
    mapped_charts = charts.copy()
    
    # Add all the mappings
    for chart_name, chart_content in charts.items():
        if chart_name in chart_mappings:
            # Add all aliases
            for alias in chart_mappings[chart_name]:
                if alias not in mapped_charts:
                    mapped_charts[alias] = chart_content
                    logger.info(f"Added alias '{alias}' for chart '{chart_name}'")
    
    # For debugging, log all chart names
    if mapped_charts:
        logger.info(f"Chart names after mapping: {list(mapped_charts.keys())}")
        
    return mapped_charts