"""
Module for generating uncertainty metrics comparison charts.
"""

import logging
from typing import Dict, List, Any, Optional
import numpy as np

from .base_chart import BaseChartGenerator

# Configure logger
logger = logging.getLogger("deepbridge.reports")

class UncertaintyMetricsChart(BaseChartGenerator):
    """
    Generate charts comparing uncertainty metrics across models.
    """
    
    def generate(self,
                models_data: Dict[str, Dict[str, float]],
                title: str = "Uncertainty Metrics Comparison") -> str:
        """
        Generate a bar chart comparing different uncertainty metrics across models.

        Parameters:
        -----------
        models_data : Dict[str, Dict[str, float]]
            Dictionary with model names as keys and dictionaries containing metrics
            Expected structure:
            {
                "model_name": {
                    "avg_coverage_error": float,
                    "uncertainty_quality_score": float,
                    "avg_norm_width": float,
                    ...
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
            logger.warning("Invalid models data for uncertainty metrics chart")
            return ""

        # If using existing chart generator
        if self.chart_generator and hasattr(self.chart_generator, 'bar_chart'):
            try:
                # Prepare data for chart generator
                # This depends on the expected format for the chart generator
                # You'll need to adjust this based on your chart generator's API
                return ""
            except Exception as e:
                logger.error(f"Error using chart generator for uncertainty metrics: {str(e)}")

        # Fallback - implement direct charting
        try:
            # Create figure
            fig, ax = self.plt.subplots(figsize=(14, 8))
            
            # Identify common metrics across all models
            metric_keys = set()
            for model_data in models_data.values():
                metric_keys.update(model_data.keys())
            
            # Filter out non-numeric metrics and limit to common uncertainty metrics
            common_metrics = [
                'avg_coverage_error', 'uncertainty_quality_score', 'avg_norm_width',
                'coverage_error', 'quality_score', 'norm_width'
            ]
            
            metrics_to_use = [m for m in metric_keys if m in common_metrics]
            
            if not metrics_to_use:
                logger.warning("No common uncertainty metrics found")
                return ""
            
            # Set up bar chart dimensions
            model_names = list(models_data.keys())
            n_models = len(model_names)
            n_metrics = len(metrics_to_use)
            
            # Set width of bars
            bar_width = 0.8 / n_metrics
            
            # Set positions of bars on x-axis
            indices = np.arange(n_models)
            
            # Plot bars for each metric
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            
            for i, metric in enumerate(metrics_to_use):
                # Collect the metric values for each model
                values = []
                for model_name in model_names:
                    if metric in models_data[model_name]:
                        values.append(models_data[model_name][metric])
                    else:
                        values.append(0)  # Default value if metric not present
                
                # Display friendly metric name
                display_name = metric.replace('_', ' ').title()
                
                # Plot the bars for this metric
                pos = indices - 0.4 + (i * bar_width)
                bars = ax.bar(pos, values, bar_width, label=display_name, color=colors[i % len(colors)])
                
                # Add the values on top of the bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8, rotation=0)
            
            # Set labels and title
            ax.set_xlabel('Model')
            ax.set_ylabel('Value')
            ax.set_title(title)
            
            # Set x-tick positions and labels
            ax.set_xticks(indices)
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            
            # Add legend
            ax.legend()
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7, axis='y')
            
            # Adjust layout for better visibility
            fig.tight_layout()
            
            # Add explanation text
            textstr = '\n'.join((
                'Metrics Explained:',
                '- Coverage Error: Lower is better (closer to expected coverage)',
                '- Quality Score: Higher is better (better calibration)',
                '- Norm Width: Lower is better (narrower intervals)'
            ))
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
            
            # Save the figure to base64
            return self._save_figure_to_base64(fig)
        except Exception as e:
            logger.error(f"Error generating uncertainty metrics chart: {str(e)}")
            return ""