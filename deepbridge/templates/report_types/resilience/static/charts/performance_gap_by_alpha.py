"""
Module for generating performance gap by alpha charts.
"""

import logging
from typing import Dict, List, Optional

from .base_chart import BaseChartGenerator

# Configure logger
logger = logging.getLogger("deepbridge.reports")

class PerformanceGapByAlphaChart(BaseChartGenerator):
    """
    Generate charts showing performance gaps across alpha levels for different models.
    """
    
    def generate(self,
                alpha_levels: List[float],
                models_data: Dict[str, Dict[str, List[float]]],
                title: str = "Performance Gap by Alpha Level",
                y_label: str = "Performance Gap") -> str:
        """
        Generate a line chart showing performance gaps across alpha levels for different models.

        Parameters:
        -----------
        alpha_levels : List[float]
            List of alpha (perturbation intensity) levels
        models_data : Dict[str, Dict[str, List[float]]]
            Dictionary with model names as keys and dictionaries containing 'worst' and 'remaining' performance lists
        title : str, optional
            Chart title
        y_label : str, optional
            Label for y-axis

        Returns:
        --------
        str : Base64 encoded image or empty string if data invalid
        """
        self._validate_chart_generator()

        # Validate input data
        if not self._validate_data(alpha_levels) or not self._validate_data(models_data):
            logger.warning("Invalid alpha levels or models data for performance gap chart")
            return ""

        # Clean alpha levels
        clean_alphas = []
        for alpha in alpha_levels:
            try:
                clean_alphas.append(float(alpha))
            except (ValueError, TypeError):
                continue

        if not clean_alphas:
            logger.warning("No valid numeric alpha levels for performance gap chart")
            return ""

        # Process and clean model data
        clean_models_data = {}

        for model_name, model_data in models_data.items():
            if not isinstance(model_data, dict):
                continue

            # Check if model has both 'worst' and 'remaining' data
            if 'worst' not in model_data or 'remaining' not in model_data:
                continue

            worst_list = model_data.get('worst', [])
            remaining_list = model_data.get('remaining', [])

            # Both lists should be present and have at least one element
            if not worst_list or not remaining_list:
                continue

            # Clean the lists and calculate performance gaps
            clean_worst = []
            clean_remaining = []

            # Process the shorter of the two lists
            n_points = min(len(worst_list), len(remaining_list))

            for i in range(n_points):
                try:
                    worst_val = float(worst_list[i])
                    remaining_val = float(remaining_list[i])
                    clean_worst.append(worst_val)
                    clean_remaining.append(remaining_val)
                except (ValueError, TypeError):
                    continue

            # Calculate performance gaps (absolute difference between worst and remaining)
            if clean_worst and clean_remaining:
                performance_gaps = [abs(remaining_val - worst_val) for worst_val, remaining_val
                                  in zip(clean_worst, clean_remaining)]

                clean_models_data[model_name] = {
                    'gaps': performance_gaps,
                    'worst': clean_worst,
                    'remaining': clean_remaining
                }

        if not clean_models_data:
            logger.warning("No valid model data for performance gap chart")
            return ""

        # If using existing chart generator
        if self.chart_generator and hasattr(self.chart_generator, 'line_chart'):
            try:
                # Prepare data for chart generator
                chart_data = {
                    'x': clean_alphas[:min(len(clean_alphas), max(len(model_data['gaps']) for model_data in clean_models_data.values()))],
                    'x_label': 'Alpha (Perturbation Intensity)',
                    'y_label': y_label,
                    'series': []
                }

                for model_name, model_data in clean_models_data.items():
                    gaps = model_data['gaps']

                    # Handle potential length mismatch
                    if len(gaps) < len(chart_data['x']):
                        # Pad with NaN for shorter series
                        padded_gaps = gaps + [float('nan')] * (len(chart_data['x']) - len(gaps))
                        chart_data['series'].append({
                            'name': model_name,
                            'values': padded_gaps
                        })
                    else:
                        # Truncate longer series
                        chart_data['series'].append({
                            'name': model_name,
                            'values': gaps[:len(chart_data['x'])]
                        })

                return self.chart_generator.line_chart(
                    data=chart_data,
                    title=title
                )
            except Exception as e:
                logger.error(f"Error using chart generator for performance gap chart: {str(e)}")

        # If no chart generator or line_chart method not available, return empty string
        logger.error("No suitable chart generator available for performance gap chart")
        return ""