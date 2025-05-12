"""
Module for generating feature distance heatmap charts.
"""

import logging
from typing import Dict, Optional

from .base_chart import BaseChartGenerator

# Configure logger
logger = logging.getLogger("deepbridge.reports")

class FeatureDistanceHeatmapChart(BaseChartGenerator):
    """
    Generate heatmap charts showing feature distances across multiple models or metrics.
    """
    
    def generate(self,
                feature_distances: Dict[str, Dict[str, float]],
                title: str = "Feature Distance Heatmap",
                top_n: int = 15,
                cmap: str = "viridis") -> str:
        """
        Generate a heatmap showing feature distances across multiple models or metrics.

        Parameters:
        -----------
        feature_distances : Dict[str, Dict[str, float]]
            Dictionary with metrics/models as keys and dictionaries of {feature: distance} as values
        title : str, optional
            Chart title
        top_n : int, optional
            Number of top features to display
        cmap : str, optional
            Colormap to use for heatmap

        Returns:
        --------
        str : Base64 encoded image or empty string if data invalid
        """
        self._validate_chart_generator()

        # Validate input data
        if not self._validate_data(feature_distances):
            logger.warning("Invalid or empty feature distances data for heatmap")
            return ""

        # Clean and process feature distances
        clean_data = {}
        all_features = set()

        for metric_name, feature_dict in feature_distances.items():
            if not isinstance(feature_dict, dict):
                continue

            clean_features = {}
            for feature, distance in feature_dict.items():
                try:
                    clean_features[feature] = float(distance)
                    all_features.add(feature)
                except (ValueError, TypeError):
                    continue

            if clean_features:
                clean_data[metric_name] = clean_features

        if not clean_data:
            logger.warning("No valid numeric feature distances for heatmap")
            return ""

        # Find the top N features across all metrics/models
        combined_features = {}
        for feature in all_features:
            # Calculate average distance across all metrics/models
            total = 0.0
            count = 0
            for metric_data in clean_data.values():
                if feature in metric_data:
                    total += metric_data[feature]
                    count += 1

            if count > 0:
                combined_features[feature] = total / count

        # Sort features by average distance (descending) and take top N
        sorted_features = sorted(combined_features.items(), key=lambda x: x[1], reverse=True)
        top_features = [feature for feature, _ in sorted_features[:min(top_n, len(sorted_features))]]

        if not top_features:
            logger.warning("No features available for heatmap after processing")
            return ""

        # If using existing chart generator
        if self.chart_generator and hasattr(self.chart_generator, 'heatmap_chart'):
            try:
                # Prepare data for heatmap
                matrix = []
                metrics = list(clean_data.keys())

                for feature in top_features:
                    feature_row = []
                    for metric in metrics:
                        metric_data = clean_data[metric]
                        value = metric_data.get(feature, 0.0)
                        feature_row.append(value)

                    matrix.append(feature_row)

                return self.chart_generator.heatmap_chart(
                    matrix=matrix,
                    x_labels=metrics,
                    y_labels=top_features,
                    title=title
                )
            except Exception as e:
                logger.error(f"Error using chart generator for feature distance heatmap: {str(e)}")

        # Fallback - implement direct charting
        try:
            # Prepare the data for heatmap
            metrics = list(clean_data.keys())

            # Create DataFrame for the heatmap
            data = []
            for metric in metrics:
                metric_data = clean_data[metric]
                for feature in top_features:
                    value = metric_data.get(feature, 0.0)
                    data.append({
                        'Metric': metric,
                        'Feature': feature,
                        'Distance': value
                    })

            # Convert to DataFrame
            df = self.pd.DataFrame(data)
            pivot_df = df.pivot(index='Feature', columns='Metric', values='Distance')

            # Create figure
            fig_width = max(10, min(20, len(metrics) * 1.5))
            fig_height = max(8, min(20, len(top_features) * 0.6))
            fig, ax = self.plt.subplots(figsize=(fig_width, fig_height))

            # Create heatmap
            heatmap = self.sns.heatmap(
                pivot_df,
                cmap=cmap,
                annot=True,  # Show values in cells
                fmt='.3f',
                linewidths=.5,
                ax=ax,
                cbar_kws={'label': 'Distance Value'}
            )

            # Set title and labels
            ax.set_title(title, fontsize=16)

            # Rotate column labels for better readability if needed
            if len(metrics) > 3:
                self.plt.xticks(rotation=45, ha='right')

            # Tight layout
            self.plt.tight_layout()

            return self._save_figure_to_base64(fig)
        except Exception as e:
            logger.error(f"Error generating feature distance heatmap: {str(e)}")
            return ""