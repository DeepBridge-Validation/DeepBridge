"""
Module for generating model resilience scores charts.
"""

import logging
from typing import Dict, Optional

from .base_chart import BaseChartGenerator

# Configure logger
logger = logging.getLogger("deepbridge.reports")

class ModelResilienceScoresChart(BaseChartGenerator):
    """
    Generate horizontal bar charts showing resilience scores for different models.
    """
    
    def generate(self,
                models_data: Dict[str, float],
                title: str = "Resilience Scores by Model",
                sort_by: str = "score",  # "score" or "name"
                ascending: bool = False) -> str:
        """
        Generate a horizontal bar chart showing resilience scores for different models.

        Parameters:
        -----------
        models_data : Dict[str, float]
            Dictionary with model names as keys and resilience scores as values
        title : str, optional
            Chart title
        sort_by : str, optional
            Sort method ('score' or 'name')
        ascending : bool, optional
            Whether to sort in ascending order

        Returns:
        --------
        str : Base64 encoded image or empty string if data invalid
        """
        self._validate_chart_generator()

        # Validate input data
        if not self._validate_data(models_data):
            logger.warning("Invalid or empty models data for resilience scores chart")
            return ""

        # Clean and process model scores
        clean_data = {}
        for model_name, score in models_data.items():
            try:
                clean_data[model_name] = float(score)
            except (ValueError, TypeError):
                continue

        if not clean_data:
            logger.warning("No valid numeric resilience scores for chart")
            return ""

        # Sort the data
        if sort_by and sort_by.lower() == "score":
            # Sort by score
            sorted_data = sorted(clean_data.items(), key=lambda x: x[1], reverse=not ascending)
        else:
            # Sort by model name
            sorted_data = sorted(clean_data.items(), key=lambda x: x[0], reverse=not ascending)

        # Extract sorted model names and scores
        model_names = [item[0] for item in sorted_data]
        scores = [item[1] for item in sorted_data]

        # If using existing chart generator
        if self.chart_generator and hasattr(self.chart_generator, 'bar_chart'):
            try:
                # Prepare data for chart generator
                bar_data = {
                    'x': model_names,
                    'y': scores,
                    'x_label': 'Model',
                    'y_label': 'Resilience Score',
                    'horizontal': False  # Indicates a vertical bar chart
                }

                return self.chart_generator.bar_chart(
                    data=bar_data,
                    title=title
                )
            except Exception as e:
                logger.error(f"Error using chart generator for resilience scores chart: {str(e)}")

        # Fallback - implement direct charting
        try:
            # Create figure
            fig_height = max(5, min(15, len(model_names) * 0.5 + 2))
            fig, ax = self.plt.subplots(figsize=(10, fig_height))

            # Create a colormap to color bars by score
            norm = self.plt.Normalize(min(scores), max(scores))
            colors = self.plt.cm.viridis(norm(scores))

            # Create horizontal bar chart
            bars = ax.barh(model_names, scores, color=colors)

            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                       f"{width:.4f}", va='center', fontweight='bold')

            # Labels and title
            ax.set_title(title, fontsize=16)
            ax.set_xlabel("Resilience Score", fontsize=14)
            ax.set_ylabel("Model", fontsize=14)

            # Add grid for better readability
            ax.grid(True, axis='x', linestyle='--', alpha=0.7)

            # Add annotation explaining the chart
            fig.text(0.5, 0.01,
                    "Higher scores indicate better model resilience to distribution shifts",
                    ha='center', fontsize=10, fontstyle='italic')

            # Tight layout
            self.plt.tight_layout(rect=[0, 0.05, 1, 1])

            return self._save_figure_to_base64(fig)
        except Exception as e:
            logger.error(f"Error generating resilience scores chart: {str(e)}")
            return ""