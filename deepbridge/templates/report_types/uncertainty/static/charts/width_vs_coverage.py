"""
Module for generating width vs coverage charts.
"""

import logging
from typing import Dict, List, Any, Optional

from .base_chart import BaseChartGenerator

# Configure logger
logger = logging.getLogger("deepbridge.reports")

class WidthVsCoverageChart(BaseChartGenerator):
    """
    Generate charts showing the relationship between interval width and coverage.
    """
    
    def generate(self,
                models_data: Dict[str, Dict[str, Any]],
                title: str = "Interval Width vs Coverage") -> str:
        """
        Generate a line chart showing the relationship between interval width and coverage.

        Parameters:
        -----------
        models_data : Dict[str, Dict[str, Any]]
            Dictionary with model names as keys and dictionaries containing width and coverage data
            Expected structure:
            {
                "model_name": {
                    "coverages": [coverage1, coverage2, ...],
                    "widths": [width1, width2, ...]
                },
                ...
            }
        title : str, optional
            Chart title

        Returns:
        --------
        str : Base64 encoded image or empty string if data invalid
        """
        self._validate_chart_generator()

        # Validate input data
        if not self._validate_data(models_data):
            logger.warning("Invalid models data for width vs coverage chart")
            return ""

        # If using existing chart generator
        if self.chart_generator and hasattr(self.chart_generator, 'line_chart'):
            try:
                # Prepare data for chart generator
                # This depends on the expected format for the chart generator
                # You'll need to adjust this based on your chart generator's API
                return ""
            except Exception as e:
                logger.error(f"Error using chart generator for width vs coverage: {str(e)}")

        # Fallback - implement direct charting
        try:
            # Create figure
            fig, ax = self.plt.subplots(figsize=(12, 8))
            
            # Define colors for different models
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
            
            # Plot each model's data
            for i, (model_name, model_data) in enumerate(models_data.items()):
                # Check for required data
                if not all(key in model_data for key in ['coverages', 'widths']):
                    continue
                
                coverages = model_data['coverages']
                widths = model_data['widths']
                
                # Ensure both lists have the same length
                min_len = min(len(coverages), len(widths))
                
                if min_len == 0:
                    continue
                    
                coverages = coverages[:min_len]
                widths = widths[:min_len]
                
                # Plot the width vs coverage
                color = colors[i % len(colors)]
                ax.plot(coverages, widths, 'o-', linewidth=2, markersize=8, 
                       label=model_name, color=color)
            
            # Set labels and title
            ax.set_xlabel('Coverage')
            ax.set_ylabel('Interval Width')
            ax.set_title(title)
            
            # Add grid and legend
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(loc='upper left')
            
            # Add explanation text
            textstr = '\n'.join((
                'Trade-off between coverage and width:',
                '- Higher coverage usually requires wider intervals',
                '- Narrower intervals with same coverage = better model',
                '- Optimal point depends on application needs'
            ))
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
            
            # Save the figure to base64
            return self._save_figure_to_base64(fig)
        except Exception as e:
            logger.error(f"Error generating width vs coverage chart: {str(e)}")
            return ""