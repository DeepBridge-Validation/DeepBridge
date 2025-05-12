"""
Module for generating distance metrics comparison charts.
"""

import logging
from typing import Dict, List, Optional

from .base_chart import BaseChartGenerator

# Configure logger
logger = logging.getLogger("deepbridge.reports")

class DistanceMetricsComparisonChart(BaseChartGenerator):
    """
    Generate charts comparing different distance metrics across alpha levels.
    """
    
    def generate(self,
                alpha_levels: List[float],
                metrics_data: Dict[str, List[float]],
                title: str = "Distance Metrics Comparison by Alpha",
                y_label: str = "Distance Value") -> str:
        """
        Generate a line chart comparing different distance metrics (PSI, WD1, KS, etc.) across alpha levels.

        Parameters:
        -----------
        alpha_levels : List[float]
            List of alpha (perturbation intensity) levels
        metrics_data : Dict[str, List[float]]
            Dictionary with metric names as keys and lists of values as values
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
        if not self._validate_data(alpha_levels) or not self._validate_data(metrics_data):
            logger.warning("Invalid alpha levels or metrics data for distance metrics comparison")
            return ""

        # Clean alpha levels
        clean_alphas = []
        for alpha in alpha_levels:
            try:
                clean_alphas.append(float(alpha))
            except (ValueError, TypeError):
                continue

        if not clean_alphas:
            logger.warning("No valid numeric alpha levels for distance metrics comparison")
            return ""

        # Clean metrics data
        clean_metrics = {}
        for metric_name, values in metrics_data.items():
            if not isinstance(values, list):
                continue

            clean_values = []
            for value in values:
                try:
                    clean_values.append(float(value))
                except (ValueError, TypeError):
                    continue

            # Only include metrics with valid numeric values
            if clean_values:
                clean_metrics[metric_name] = clean_values

        if not clean_metrics:
            logger.warning("No valid metrics data for distance metrics comparison")
            return ""

        # If using existing chart generator
        if self.chart_generator and hasattr(self.chart_generator, 'line_chart'):
            try:
                # Prepare data for chart generator
                chart_data = {
                    'x': clean_alphas,
                    'x_label': 'Alpha (Perturbation Intensity)',
                    'y_label': y_label,
                    'series': []
                }

                for metric_name, values in clean_metrics.items():
                    if len(values) < len(clean_alphas):
                        # Pad shorter series with NaN
                        padded_values = values + [float('nan')] * (len(clean_alphas) - len(values))
                        chart_data['series'].append({
                            'name': metric_name,
                            'values': padded_values
                        })
                    elif len(values) > len(clean_alphas):
                        # Truncate longer series
                        chart_data['series'].append({
                            'name': metric_name,
                            'values': values[:len(clean_alphas)]
                        })
                    else:
                        chart_data['series'].append({
                            'name': metric_name,
                            'values': values
                        })

                return self.chart_generator.line_chart(
                    data=chart_data,
                    title=title
                )
            except Exception as e:
                logger.error(f"Error using chart generator for distance metrics comparison: {str(e)}")

        # Fallback - implement direct charting
        try:
            # Create figure
            fig, ax = self.plt.subplots(figsize=(12, 8))

            # Create a palette with distinct colors
            colors = ['#1b78de', '#ff9f1c', '#2ec4b6', '#e71d36', '#8338ec', '#06d6a0', '#ef476f', '#ffd166']

            # Plot each metric as a line series
            for i, (metric_name, values) in enumerate(clean_metrics.items()):
                # Handle series length mismatch
                if len(values) < len(clean_alphas):
                    # Use only available data points
                    plot_alphas = clean_alphas[:len(values)]
                    ax.plot(plot_alphas, values, marker='o', linestyle='-',
                           linewidth=2, markersize=8, label=metric_name,
                           color=colors[i % len(colors)])
                elif len(values) > len(clean_alphas):
                    # Truncate to match alpha levels
                    ax.plot(clean_alphas, values[:len(clean_alphas)], marker='o', linestyle='-',
                           linewidth=2, markersize=8, label=metric_name,
                           color=colors[i % len(colors)])
                else:
                    # Perfect match
                    ax.plot(clean_alphas, values, marker='o', linestyle='-',
                           linewidth=2, markersize=8, label=metric_name,
                           color=colors[i % len(colors)])

            # Labels and title
            ax.set_title(title, fontsize=16)
            ax.set_xlabel("Alpha (Perturbation Intensity)", fontsize=14)
            ax.set_ylabel(y_label, fontsize=14)

            # Format x-axis ticks to show alpha values cleanly
            ax.xaxis.set_major_formatter(self.plt.FuncFormatter(lambda x, _: f"{x:.2f}"))

            # Add legend
            ax.legend(fontsize=12)

            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)

            # Add annotation explaining the chart
            fig.text(0.5, 0.01,
                    "Higher values indicate more distribution shift between original and perturbed data",
                    ha='center', fontsize=10, fontstyle='italic')

            # Tight layout
            self.plt.tight_layout(rect=[0, 0.03, 1, 1])

            return self._save_figure_to_base64(fig)
        except Exception as e:
            logger.error(f"Error generating distance metrics comparison chart: {str(e)}")
            return ""