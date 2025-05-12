"""
Module for generating feature distribution shift charts.
"""

import logging
from typing import Dict, Optional

from .base_chart import BaseChartGenerator

# Configure logger
logger = logging.getLogger("deepbridge.reports")

class FeatureDistributionShiftChart(BaseChartGenerator):
    """
    Generate charts showing the distribution shift for different features.
    """
    
    def generate(self,
                feature_distances: Dict[str, float],
                title: str = "Feature Distribution Shift",
                top_n: int = 10) -> str:
        """
        Generate a chart showing the distribution shift for different features.

        Parameters:
        -----------
        feature_distances : Dict[str, float]
            Dictionary with feature names as keys and distance values as values
        title : str, optional
            Chart title
        top_n : int, optional
            Number of top features to display

        Returns:
        --------
        str : Base64 encoded image or empty string if data invalid
        """
        self._validate_chart_generator()

        # Try to extract meaningful data even if it's incomplete
        if not self._validate_data(feature_distances):
            logger.warning("Invalid or empty feature distances data for chart")
            return ""
        
        # Sort features by distance (highest first)
        # Filter out non-numeric values first
        clean_distances = {}
        for feature, value in feature_distances.items():
            try:
                clean_distances[feature] = float(value)
            except (ValueError, TypeError):
                continue
                
        if not clean_distances:
            logger.warning("No valid numeric distance values found")
            return ""
            
        sorted_features = sorted(clean_distances.items(), key=lambda x: x[1], reverse=True)
        top_features = dict(sorted_features[:min(top_n, len(sorted_features))])
        
        # If using existing chart generator
        if self.chart_generator:
            feature_data = {
                'Feature': list(top_features.keys()),
                'Distance': list(top_features.values())
            }
            
            try:
                # Try feature_psi_chart first if available
                if hasattr(self.chart_generator, 'feature_psi_chart'):
                    return self.chart_generator.feature_psi_chart(
                        psi_data=feature_data,
                        title=title
                    )
                # Fall back to bar_chart
                elif hasattr(self.chart_generator, 'bar_chart'):
                    dist_data = {
                        'x': list(top_features.keys()),
                        'y': list(top_features.values()),
                        'x_label': 'Features',
                        'y_label': 'Distance'
                    }
                    return self.chart_generator.bar_chart(
                        data=dist_data,
                        title=title
                    )
            except Exception as e:
                logger.error(f"Error using chart generator for feature distribution shift: {str(e)}")
        
        # Fallback - implement direct charting if needed
        try:
            # Create figure
            fig, ax = self.plt.subplots(figsize=(14, 8))
            
            # Prepare data
            features = list(top_features.keys())
            values = list(top_features.values())
            
            # Create DataFrame
            df = self.pd.DataFrame({'Feature': features, 'Distance': values})
            
            # Create bar chart
            bars = self.sns.barplot(x='Feature', y='Distance', hue='Feature', data=df, ax=ax, palette='viridis', legend=False)
            
            # Add values on top of bars
            for i, v in enumerate(values):
                ax.text(i, v + 0.02 * max(values), f"{v:.3f}", ha='center', fontsize=10)
            
            # Labels and title
            ax.set_title(title, fontsize=16)
            ax.set_xlabel("Features", fontsize=14)
            ax.set_ylabel("Distribution Shift", fontsize=14)
            
            # Rotate x-axis labels for better readability
            self.plt.xticks(rotation=45, ha='right')

            # Add grid for better readability
            ax.grid(True, axis='y', linestyle='--', alpha=0.7)

            # Tight layout
            self.plt.tight_layout()
            
            return self._save_figure_to_base64(fig)
        except Exception as e:
            logger.error(f"Error generating feature distribution shift chart: {str(e)}")
            return ""