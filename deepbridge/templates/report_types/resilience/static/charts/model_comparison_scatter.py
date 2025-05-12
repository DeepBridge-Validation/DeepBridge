"""
Module for generating model comparison scatter charts.
"""

import logging
from typing import Dict, Optional

from .base_chart import BaseChartGenerator

# Configure logger
logger = logging.getLogger("deepbridge.reports")

class ModelComparisonScatterChart(BaseChartGenerator):
    """
    Generate scatter charts comparing accuracy and resilience scores across models.
    """
    
    def generate(self,
                models_data: Dict[str, Dict[str, float]],
                title: str = "Accuracy vs Resilience Score",
                x_label: str = "Accuracy",
                y_label: str = "Resilience Score") -> str:
        """
        Generate a scatter plot comparing accuracy and resilience scores across models.

        Parameters:
        -----------
        models_data : Dict[str, Dict[str, float]]
            Dictionary with model names as keys and dictionaries containing 'accuracy' and 'resilience_score' as values
        title : str, optional
            Chart title
        x_label : str, optional
            Label for x-axis (typically accuracy metric)
        y_label : str, optional
            Label for y-axis (typically resilience score)

        Returns:
        --------
        str : Base64 encoded image or empty string if data invalid
        """
        self._validate_chart_generator()

        # Validate models data
        if not self._validate_data(models_data):
            logger.warning("Invalid or empty models data for comparison scatter plot")
            return ""

        # Clean and prepare model data
        clean_models_data = {}
        for model_name, model_info in models_data.items():
            if not isinstance(model_info, dict):
                continue

            # Check for required fields
            if 'accuracy' not in model_info or 'resilience_score' not in model_info:
                continue

            try:
                accuracy = float(model_info['accuracy'])
                resilience = float(model_info['resilience_score'])
                clean_models_data[model_name] = {
                    'accuracy': accuracy,
                    'resilience_score': resilience
                }
            except (ValueError, TypeError):
                continue

        if not clean_models_data:
            logger.warning("No valid model data for comparison scatter plot")
            return ""

        # If using existing chart generator
        if self.chart_generator and hasattr(self.chart_generator, 'scatter_plot'):
            try:
                # Prepare data for existing chart generator
                scatter_data = {
                    'x': [],
                    'y': [],
                    'labels': [],
                    'x_label': x_label,
                    'y_label': y_label
                }

                for model_name, model_info in clean_models_data.items():
                    scatter_data['x'].append(model_info['accuracy'])
                    scatter_data['y'].append(model_info['resilience_score'])
                    scatter_data['labels'].append(model_name)

                return self.chart_generator.scatter_plot(
                    data=scatter_data,
                    title=title
                )
            except Exception as e:
                logger.error(f"Error using chart generator for model comparison scatter: {str(e)}")

        # Fallback - implement direct charting
        try:
            # Extract data
            model_names = list(clean_models_data.keys())
            accuracy_values = [model_info['accuracy'] for model_info in clean_models_data.values()]
            resilience_values = [model_info['resilience_score'] for model_info in clean_models_data.values()]

            # Create figure
            fig, ax = self.plt.subplots(figsize=(10, 8))

            # Create a palette with distinct colors
            colors = ['#1b78de', '#ff9f1c', '#2ec4b6', '#e71d36', '#8338ec', '#06d6a0', '#ef476f', '#ffd166']
            scatter_colors = [colors[i % len(colors)] for i in range(len(model_names))]

            # Create scatter plot
            scatter = ax.scatter(
                accuracy_values,
                resilience_values,
                c=scatter_colors,
                s=100,  # Marker size
                alpha=0.7
            )

            # Add model names as annotations
            for i, model_name in enumerate(model_names):
                ax.annotate(
                    model_name,
                    (accuracy_values[i], resilience_values[i]),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center',
                    fontsize=10,
                    fontweight='bold'
                )

            # Add diagonal line for reference (higher is better for both metrics)
            if len(accuracy_values) > 0 and len(resilience_values) > 0:
                min_x = min(accuracy_values)
                max_x = max(accuracy_values)
                min_y = min(resilience_values)
                max_y = max(resilience_values)

                # Normalize the values for the diagonal
                x_range = max_x - min_x if max_x > min_x else 1.0
                y_range = max_y - min_y if max_y > min_y else 1.0

                diag_x = [min_x, max_x]
                diag_y = [min_y, max_y]

                ax.plot(diag_x, diag_y, 'k--', alpha=0.3)

            # Labels and title
            ax.set_title(title, fontsize=16)
            ax.set_xlabel(x_label, fontsize=14)
            ax.set_ylabel(y_label, fontsize=14)

            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)

            # Add annotation explaining the chart
            fig.text(0.5, 0.01,
                    "Models in the top-right corner perform well and are more resilient",
                    ha='center', fontsize=10, fontstyle='italic')

            # Tight layout
            self.plt.tight_layout(rect=[0, 0.03, 1, 1])

            return self._save_figure_to_base64(fig)
        except Exception as e:
            logger.error(f"Error generating model comparison scatter plot: {str(e)}")
            return ""