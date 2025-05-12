"""
Performance gap by alpha chart generator for uncertainty visualization.
"""

import logging
import matplotlib.pyplot as plt
import numpy as np
import base64
import io
from typing import Dict, Any, List, Optional, Union, Tuple

from .base_chart import BaseChartGenerator

# Configure logger
logger = logging.getLogger("deepbridge.reports")


class PerformanceGapByAlphaChart(BaseChartGenerator):
    """
    Generator for performance gap by alpha charts in uncertainty reports.
    """
    
    def _validate_data(self, models_data: Dict[str, Any]) -> bool:
        """
        Validate data for performance gap by alpha chart.
        
        Parameters:
        -----------
        models_data : Dict[str, Any]
            Data to validate
        
        Returns:
        --------
        bool : Whether the data is valid
        """
        # Basic validation from BaseChartGenerator
        if not super()._validate_data(models_data):
            logger.warning("Basic validation failed for performance gap by alpha chart")
            return False
            
        # Add detailed logging
        logger.info(f"Validating performance gap by alpha data with: type={type(models_data)}")
        
        if isinstance(models_data, dict):
            logger.info(f"models_data keys: {list(models_data.keys())}")
            
            # Check for calibration_results
            if 'calibration_results' in models_data:
                cal_results = models_data['calibration_results']
                logger.info(f"calibration_results type: {type(cal_results)}")
                
                if isinstance(cal_results, dict):
                    logger.info(f"calibration_results keys: {list(cal_results.keys())}")
                    
                    # Check for alpha values and coverage data
                    has_alpha = 'alpha_values' in cal_results
                    has_coverage = 'coverage_values' in cal_results
                    has_expected = 'expected_coverages' in cal_results
                    
                    logger.info(f"Has required data: alpha_values={has_alpha}, coverage_values={has_coverage}, expected_coverages={has_expected}")
                    
                    # If we have the minimum required data, it's valid
                    if has_alpha and (has_coverage and has_expected):
                        return True
            
            # Check alternative data paths
            has_alpha = 'alpha_levels' in models_data or 'alphas' in models_data
            has_perf_gaps = 'performance_gaps' in models_data
            
            if has_alpha and has_perf_gaps:
                logger.info("Found alternative data paths with alpha values and performance gaps")
                return True
                
            # Check for primary_model with plot_data
            if 'primary_model' in models_data and isinstance(models_data['primary_model'], dict):
                primary_model = models_data['primary_model']
                
                if 'plot_data' in primary_model and isinstance(primary_model['plot_data'], dict):
                    plot_data = primary_model['plot_data']
                    
                    if 'alpha_comparison' in plot_data and isinstance(plot_data['alpha_comparison'], dict):
                        alpha_data = plot_data['alpha_comparison']
                        logger.info(f"alpha_comparison keys: {list(alpha_data.keys())}")
                        
                        # Check for required data
                        has_alpha = 'alphas' in alpha_data
                        has_coverage = 'coverages' in alpha_data
                        has_expected = 'expected_coverages' in alpha_data
                        
                        logger.info(f"alpha_comparison has required data: alphas={has_alpha}, coverages={has_coverage}, expected_coverages={has_expected}")
                        
                        if has_alpha and (has_coverage and has_expected):
                            return True
        
        logger.warning("No suitable data found for performance gap by alpha chart")
        return False

    def generate(self, models_data: Dict[str, Any],
                 title: str = "Performance Gap by Alpha Level",
                 add_annotations: bool = True) -> Optional[str]:
        """
        Generate a chart showing performance gap by alpha level.

        Parameters:
        -----------
        models_data : Dict[str, Any]
            Data containing alpha levels and performance gaps
        title : str, optional
            Title for the chart
        add_annotations : bool, optional
            Whether to add annotations to the chart

        Returns:
        --------
        Optional[str] : Base64 encoded image or None if generation fails
        """
        try:
            # Validate data
            if not self._validate_data(models_data):
                logger.warning("Invalid data provided for performance gap by alpha chart")
                return None

            # Extract alpha values and performance gaps
            alpha_values = None
            performance_gaps = None
            
            # Try to find data in different formats
            if 'calibration_results' in models_data and isinstance(models_data['calibration_results'], dict):
                if 'alpha_values' in models_data['calibration_results']:
                    alpha_values = models_data['calibration_results']['alpha_values']
                
                if 'performance_gaps' in models_data['calibration_results']:
                    performance_gaps = models_data['calibration_results']['performance_gaps']
                elif 'coverage_values' in models_data['calibration_results'] and 'expected_coverages' in models_data['calibration_results']:
                    # Calculate performance gaps as the difference between actual and expected coverage
                    actual = models_data['calibration_results']['coverage_values']
                    expected = models_data['calibration_results']['expected_coverages']
                    if len(actual) == len(expected):
                        performance_gaps = [a - e for a, e in zip(actual, expected)]
            
            # Try alternative data paths
            if alpha_values is None and 'alphas' in models_data:
                alpha_values = models_data['alphas']
            if alpha_values is None and 'alpha_levels' in models_data:
                alpha_values = models_data['alpha_levels']
                
            if performance_gaps is None and 'performance_gaps' in models_data:
                performance_gaps = models_data['performance_gaps']
                
            # If primary model has the data
            if (alpha_values is None or performance_gaps is None) and 'primary_model' in models_data:
                primary_model = models_data['primary_model']
                
                if 'plot_data' in primary_model and 'alpha_comparison' in primary_model['plot_data']:
                    alpha_data = primary_model['plot_data']['alpha_comparison']
                    
                    if alpha_values is None and 'alphas' in alpha_data:
                        alpha_values = alpha_data['alphas']
                        
                    if performance_gaps is None:
                        if 'performance_gaps' in alpha_data:
                            performance_gaps = alpha_data['performance_gaps']
                        elif 'coverages' in alpha_data and 'expected_coverages' in alpha_data:
                            actual = alpha_data['coverages']
                            expected = alpha_data['expected_coverages']
                            if len(actual) == len(expected):
                                performance_gaps = [a - e for a, e in zip(actual, expected)]
            
            # If data is not available, return None
            if alpha_values is None or performance_gaps is None:
                logger.warning("Alpha values or performance gaps not found in data")
                return None
                
            # Ensure alpha_values and performance_gaps are the same length
            if len(alpha_values) != len(performance_gaps):
                logger.warning(f"Alpha values ({len(alpha_values)}) and performance gaps ({len(performance_gaps)}) have different lengths")
                # Use the shorter length
                min_len = min(len(alpha_values), len(performance_gaps))
                alpha_values = alpha_values[:min_len]
                performance_gaps = performance_gaps[:min_len]
                
            # Create plot
            plt.figure(figsize=(10, 6))
            
            # Plot performance gaps
            plt.bar(alpha_values, performance_gaps, color='steelblue', alpha=0.7)
            
            # Add a horizontal line at y=0
            plt.axhline(y=0, color='grey', linestyle='-', linewidth=1)
            
            # Add annotations if requested
            if add_annotations:
                for i, gap in enumerate(performance_gaps):
                    color = 'green' if abs(gap) < 0.05 else ('orange' if abs(gap) < 0.1 else 'red')
                    plt.annotate(f'{gap:.3f}', 
                                xy=(alpha_values[i], gap), 
                                xytext=(0, 5 if gap >= 0 else -15),
                                textcoords='offset points',
                                ha='center', 
                                va='center' if gap >= 0 else 'top',
                                color=color,
                                fontweight='bold')
            
            # Set up plot appearance
            plt.xlabel('Alpha Level')
            plt.ylabel('Coverage Gap (Actual - Expected)')
            plt.title(title)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add a text explanation
            plt.figtext(0.5, 0.01, 
                       "Bars above zero indicate over-coverage (conservative intervals).\n"
                       "Bars below zero indicate under-coverage (intervals too narrow).",
                       ha='center', 
                       fontsize=9, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout(rect=[0, 0.05, 1, 1])  # Adjust layout to make room for the text
            
            # Convert plot to base64
            return self._save_figure_to_base64(plt.gcf())
            
        except Exception as e:
            logger.error(f"Error generating performance gap by alpha chart: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None