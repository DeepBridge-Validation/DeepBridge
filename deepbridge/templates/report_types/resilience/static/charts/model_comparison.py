"""
Module for generating model comparison charts across perturbation levels.
"""

import logging
from typing import Dict, List, Optional

from .base_chart import BaseChartGenerator

# Configure logger
logger = logging.getLogger("deepbridge.reports")

class ModelComparisonChart(BaseChartGenerator):
    """
    Generate charts comparing multiple models across perturbation levels.
    """
    
    def generate(self,
                perturbation_levels: List[float],
                models_data: Dict[str, Dict],
                title: str = "Model Resilience Comparison",
                metric_name: str = "Score") -> str:
        """
        Generate a chart comparing multiple models across perturbation levels.

        Parameters:
        -----------
        perturbation_levels : List[float]
            List of perturbation intensity levels
        models_data : Dict[str, Dict]
            Dictionary with model data (scores and base_score)
        title : str, optional
            Chart title
        metric_name : str, optional
            Name of the metric being displayed

        Returns:
        --------
        str : Base64 encoded image or empty string if data invalid
        """
        self._validate_chart_generator()

        # Check if perturbation levels and models data are valid
        if not self._validate_data(perturbation_levels) or not self._validate_data(models_data):
            logger.warning("Invalid perturbation levels or models data for comparison chart")
            return ""

        # Clean perturbation levels
        clean_perturbations = []
        for level in perturbation_levels:
            try:
                clean_perturbations.append(float(level))
            except (ValueError, TypeError):
                continue

        if not clean_perturbations:
            logger.warning("No valid numeric perturbation levels")
            return ""

        # Clean and prepare model data
        clean_models_data = {}
        for model_name, model_info in models_data.items():
            # Skip invalid model data
            if not isinstance(model_info, dict):
                continue

            clean_model = {}

            # Try to extract scores
            scores = model_info.get('scores', [])
            clean_scores = []

            # Convert scores to float and filter invalid values
            for score in scores:
                try:
                    clean_scores.append(float(score))
                except (ValueError, TypeError):
                    continue

            # Only include model if it has at least one valid score
            if clean_scores:
                clean_model['scores'] = clean_scores

                # Try to extract base_score if available
                if 'base_score' in model_info:
                    try:
                        clean_model['base_score'] = float(model_info['base_score'])
                    except (ValueError, TypeError):
                        pass

                # Add to clean models data
                clean_models_data[model_name] = clean_model

        if not clean_models_data:
            logger.warning("No valid model data for comparison chart")
            return ""

        # If using existing chart generator
        if self.chart_generator:
            try:
                # Try model_comparison_chart if available
                if hasattr(self.chart_generator, 'model_comparison_chart'):
                    # Handle potential mismatch in score lengths
                    for model_name, model_info in clean_models_data.items():
                        if len(model_info.get('scores', [])) != len(clean_perturbations):
                            # If scores are shorter than perturbation levels, pad with None
                            if len(model_info.get('scores', [])) < len(clean_perturbations):
                                scores = model_info.get('scores', [])
                                padding = [None] * (len(clean_perturbations) - len(scores))
                                model_info['scores'] = scores + padding
                            # If scores are longer, truncate
                            else:
                                model_info['scores'] = model_info.get('scores', [])[:len(clean_perturbations)]

                    return self.chart_generator.model_comparison_chart(
                        perturbation_levels=clean_perturbations,
                        models_data=clean_models_data,
                        title=title,
                        metric_name=metric_name
                    )
            except Exception as e:
                logger.error(f"Error using chart generator for model comparison: {str(e)}")

        # Fallback - implement direct charting if needed
        try:
            # Create figure
            fig, ax = self.plt.subplots(figsize=(12, 8))

            # Create a palette with distinct colors
            colors = ['#1b78de', '#ff9f1c', '#2ec4b6', '#e71d36', '#8338ec', '#06d6a0', '#ef476f', '#ffd166']

            # Plot each model
            for i, (model_name, model_info) in enumerate(clean_models_data.items()):
                scores = model_info.get('scores', [])
                base_score = model_info.get('base_score', None)

                # Handle potential mismatch in score lengths
                if len(scores) != len(clean_perturbations):
                    # If there are more scores than perturbation levels, truncate scores
                    if len(scores) > len(clean_perturbations):
                        scores = scores[:len(clean_perturbations)]
                    # If there are fewer scores, plot with available data
                    else:
                        # Create a limited set of perturbation levels to match score length
                        plot_perturbations = clean_perturbations[:len(scores)]
                        if plot_perturbations:
                            color = colors[i % len(colors)]
                            ax.plot(plot_perturbations, scores, marker='o', linestyle='-',
                                   linewidth=2, markersize=8, label=model_name, color=color)

                        # If base score is available, plot as horizontal line
                        if base_score is not None:
                            ax.axhline(y=base_score, color=color, linestyle='--', alpha=0.5)

                        continue

                # Plot line
                color = colors[i % len(colors)]
                ax.plot(clean_perturbations, scores, marker='o', linestyle='-',
                       linewidth=2, markersize=8, label=model_name, color=color)

                # Add base score as horizontal line if available
                if base_score is not None:
                    ax.axhline(y=base_score, color=color, linestyle='--', alpha=0.5)

            # Labels and title
            ax.set_title(title, fontsize=16)
            ax.set_xlabel("Perturbation Intensity", fontsize=14)
            ax.set_ylabel(metric_name, fontsize=14)

            # Add legend
            ax.legend()

            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)

            # Tight layout
            self.plt.tight_layout()

            return self._save_figure_to_base64(fig)
        except Exception as e:
            logger.error(f"Error generating model comparison chart: {str(e)}")
            return ""