"""
Module for generating critical feature distributions charts.
"""

import logging
from typing import Dict, List, Optional

from .base_chart import BaseChartGenerator

# Configure logger
logger = logging.getLogger("deepbridge.reports")

class CriticalFeatureDistributionsChart(BaseChartGenerator):
    """
    Generate charts showing distributions of critical features.
    """
    
    def generate(self,
                worst_samples: Dict[str, List[float]],
                remaining_samples: Dict[str, List[float]],
                top_features: List[str],
                title: str = "Critical Feature Distributions") -> str:
        """
        Generate a chart showing distributions of critical features.

        Parameters:
        -----------
        worst_samples : Dict[str, List[float]]
            Dictionary with feature values for worst samples
        remaining_samples : Dict[str, List[float]]
            Dictionary with feature values for remaining samples
        top_features : List[str]
            List of top feature names to display
        title : str, optional
            Chart title

        Returns:
        --------
        str : Base64 encoded image or empty string if data invalid
        """
        self._validate_chart_generator()

        # Try to work with any available sample data
        if not self._validate_data(worst_samples) and not self._validate_data(remaining_samples):
            logger.warning("No valid sample data for critical feature distributions chart")
            return ""
            
        if not self._validate_data(top_features):
            # If top_features not provided, try to extract feature names from samples
            top_features = []
            if self._validate_data(worst_samples):
                top_features.extend(list(worst_samples.keys()))
            if self._validate_data(remaining_samples):
                top_features.extend(list(remaining_samples.keys()))
                
            # Remove duplicates
            top_features = list(set(top_features))
            
            # Limit to first 5
            top_features = top_features[:5]
            
            if not top_features:
                logger.warning("No feature names available for distributions chart")
                return ""
        
        # Limit to max 5 features
        features_to_plot = top_features[:min(5, len(top_features))]
        
        # If using existing chart generator
        if self.chart_generator:
            # Create model data structure for boxplot
            boxplot_data = []
            
            for feature in features_to_plot:
                # Handle worst samples if available
                if self._validate_data(worst_samples) and feature in worst_samples and self._validate_data(worst_samples[feature]):
                    try:
                        # Convert to list of floats if possible
                        values = worst_samples[feature]
                        if not isinstance(values, list):
                            values = [float(values)]
                        else:
                            values = [float(v) for v in values if v is not None]
                            
                        if values:  # Only add if we have valid values
                            boxplot_data.append({
                                'name': f"{feature} (Worst)",
                                'scores': values,
                                'baseScore': None
                            })
                    except (ValueError, TypeError):
                        pass
                
                # Handle remaining samples if available
                if self._validate_data(remaining_samples) and feature in remaining_samples and self._validate_data(remaining_samples[feature]):
                    try:
                        # Convert to list of floats if possible
                        values = remaining_samples[feature]
                        if not isinstance(values, list):
                            values = [float(values)]
                        else:
                            values = [float(v) for v in values if v is not None]
                            
                        if values:  # Only add if we have valid values
                            boxplot_data.append({
                                'name': f"{feature} (Remaining)",
                                'scores': values,
                                'baseScore': None
                            })
                    except (ValueError, TypeError):
                        pass
            
            if boxplot_data:
                try:
                    # Try boxplot_chart if available
                    if hasattr(self.chart_generator, 'boxplot_chart'):
                        return self.chart_generator.boxplot_chart(
                            models_data=boxplot_data,
                            title=title,
                            metric_name="Feature Value"
                        )
                except Exception as e:
                    logger.error(f"Error using chart generator for critical features: {str(e)}")
        
        # Fallback - implement direct charting if needed
        try:
            # Set up the figure with multiple subplots (one per feature)
            n_features = sum(1 for f in features_to_plot if 
                           (f in worst_samples and self._validate_data(worst_samples[f])) or 
                           (f in remaining_samples and self._validate_data(remaining_samples[f])))
                           
            if n_features == 0:
                logger.warning("No valid feature data found for distributions chart")
                return ""
            
            fig, axes = self.plt.subplots(1, n_features, figsize=(16, 6))
            
            # Handle single feature case
            if n_features == 1:
                axes = [axes]
            
            # Plot each feature
            i = 0
            for feature in features_to_plot:
                worst_data = None
                remaining_data = None
                
                # Get worst samples data if available
                if self._validate_data(worst_samples) and feature in worst_samples and self._validate_data(worst_samples[feature]):
                    try:
                        values = worst_samples[feature]
                        if not isinstance(values, list):
                            values = [float(values)]
                        else:
                            values = [float(v) for v in values if v is not None]
                            
                        if values:
                            worst_data = values
                    except (ValueError, TypeError):
                        pass
                
                # Get remaining samples data if available
                if self._validate_data(remaining_samples) and feature in remaining_samples and self._validate_data(remaining_samples[feature]):
                    try:
                        values = remaining_samples[feature]
                        if not isinstance(values, list):
                            values = [float(values)]
                        else:
                            values = [float(v) for v in values if v is not None]
                            
                        if values:
                            remaining_data = values
                    except (ValueError, TypeError):
                        pass
                
                # Only plot if we have at least one valid dataset
                if worst_data or remaining_data:
                    ax = axes[i]
                    i += 1
                    
                    # Plot available datasets
                    data_to_plot = []
                    if worst_data:
                        data_to_plot.append(worst_data)
                    if remaining_data:
                        data_to_plot.append(remaining_data)
                    
                    # Plot violin with boxplot inside
                    self.sns.violinplot(
                        data=data_to_plot,
                        ax=ax,
                        palette=['red', 'blue'][:len(data_to_plot)],
                        inner='box'
                    )
                    
                    # Set labels
                    ax.set_title(feature.replace('_', ' ').title(), fontsize=12)
                    ax.set_ylabel("Value", fontsize=10)
                    labels = []
                    if worst_data:
                        labels.append('Worst')
                    if remaining_data:
                        labels.append('Remaining')
                    ax.set_xticklabels(labels)
                    
                    # Add grid
                    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
            
            # Set overall title
            self.plt.suptitle(title, fontsize=16)
            
            # Tight layout
            self.plt.tight_layout()
            self.plt.subplots_adjust(top=0.85)
            
            return self._save_figure_to_base64(fig)
        except Exception as e:
            logger.error(f"Error generating critical feature distributions chart: {str(e)}")
            return ""