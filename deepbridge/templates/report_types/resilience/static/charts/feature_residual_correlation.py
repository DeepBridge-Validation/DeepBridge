"""
Module for generating feature-residual correlation charts.
"""

import logging
from typing import Dict, Optional

from .base_chart import BaseChartGenerator

# Configure logger
logger = logging.getLogger("deepbridge.reports")

class FeatureResidualCorrelationChart(BaseChartGenerator):
    """
    Generate charts showing correlation between features and residuals.
    """
    
    def generate(self,
                feature_correlations: Dict[str, float],
                title: str = "Feature-Residual Correlation",
                top_n: int = 8) -> str:
        """
        Generate a chart showing correlation between features and residuals.

        Parameters:
        -----------
        feature_correlations : Dict[str, float]
            Dictionary with feature names and correlation values
        title : str, optional
            Chart title
        top_n : int, optional
            Number of top features to display

        Returns:
        --------
        str : Base64 encoded image or empty string if data invalid
        """
        self._validate_chart_generator()

        # Try to extract meaningful correlation data
        if not self._validate_data(feature_correlations):
            logger.warning("Invalid or empty feature correlations data for chart")
            return ""
        
        # Clean and convert correlation values to float
        clean_correlations = {}
        for feature, corr_value in feature_correlations.items():
            try:
                clean_correlations[feature] = float(corr_value)
            except (ValueError, TypeError):
                continue
                
        if not clean_correlations:
            logger.warning("No valid numeric correlation values")
            return ""
        
        # Sort by absolute correlation
        sorted_correlations = sorted(
            clean_correlations.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )
        
        # Take top N
        top_correlations = sorted_correlations[:min(top_n, len(sorted_correlations))]
        
        # If using existing chart generator
        if self.chart_generator:
            # Create correlation matrix for heatmap
            correlation_matrix = []
            feature_names = []
            
            for feature, corr_value in top_correlations:
                feature_names.append(feature)
                correlation_matrix.append([round(corr_value, 3)])
            
            if correlation_matrix and feature_names:
                try:
                    # Try heatmap_chart if available
                    if hasattr(self.chart_generator, 'heatmap_chart'):
                        return self.chart_generator.heatmap_chart(
                            matrix=correlation_matrix,
                            x_labels=['Correlation'],
                            y_labels=feature_names,
                            title=title
                        )
                    # Fall back to bar_chart if available
                    elif hasattr(self.chart_generator, 'bar_chart'):
                        bar_data = {
                            'x': feature_names,
                            'y': [row[0] for row in correlation_matrix],
                            'x_label': 'Features',
                            'y_label': 'Correlation'
                        }
                        return self.chart_generator.bar_chart(
                            data=bar_data,
                            title=title
                        )
                except Exception as e:
                    logger.error(f"Error using chart generator for correlation chart: {str(e)}")
        
        # Fallback - implement direct charting if needed
        try:
            # Extract features and correlations
            features = []
            correlations = []
            
            for feature, corr_value in top_correlations:
                features.append(feature)
                correlations.append(corr_value)
            
            if not features:
                logger.warning("No valid correlations found for correlation chart")
                return ""
            
            # Create figure
            fig, ax = self.plt.subplots(figsize=(10, 8))
            
            # Create horizontal bar chart
            bars = self.sns.barplot(y=features, x=correlations, palette='coolwarm', ax=ax)
            
            # Add value labels
            for i, v in enumerate(correlations):
                ax.text(v + 0.01 if v >= 0 else v - 0.06, i, f"{v:.3f}", va='center', fontsize=10)
            
            # Labels and title
            ax.set_title(title, fontsize=16)
            ax.set_xlabel("Correlation Coefficient", fontsize=14)
            ax.set_ylabel("", fontsize=14)
            
            # Add vertical line at zero
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            
            # Add grid
            ax.grid(True, axis='x', linestyle='--', alpha=0.7)
            
            # Tight layout
            self.plt.tight_layout()
            
            return self._save_figure_to_base64(fig)
        except Exception as e:
            logger.error(f"Error generating feature-residual correlation chart: {str(e)}")
            return ""