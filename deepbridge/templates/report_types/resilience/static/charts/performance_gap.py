"""
Module for generating performance gap charts.
"""

import logging
from typing import Dict, List, Optional

from .base_chart import BaseChartGenerator

# Configure logger
logger = logging.getLogger("deepbridge.reports")

class PerformanceGapChart(BaseChartGenerator):
    """
    Generate charts comparing performance between worst and remaining samples.
    """
    
    def generate(self,
                performance_metrics: Dict[str, float],
                title: str = "Performance Comparison: Worst vs Remaining Samples",
                task_type: str = "classification") -> str:
        """
        Generate a chart comparing performance between worst and remaining samples.

        Parameters:
        -----------
        performance_metrics : Dict[str, float]
            Dictionary with performance metrics
        title : str, optional
            Chart title
        task_type : str, optional
            Task type ('classification' or 'regression')

        Returns:
        --------
        str : Base64 encoded image or empty string if data invalid
        """
        self._validate_chart_generator()

        # Try to extract meaningful metrics even with partial data
        if not self._validate_data(performance_metrics):
            logger.warning("Invalid or empty performance metrics data for chart")
            return ""
        
        # Select metrics based on task type
        if task_type == "classification":
            metric_keys = ['auc', 'f1', 'precision', 'recall', 'accuracy']
        else:  # regression
            metric_keys = ['mse', 'mae', 'r2', 'smape']
        
        # Build metric comparison data
        metric_names = []
        worst_values = []
        remaining_values = []
        
        # Try to find any paired metrics (worst and remaining)
        for metric in metric_keys:
            worst_key = f'worst_{metric}'
            remaining_key = f'remaining_{metric}'
            
            if worst_key in performance_metrics and remaining_key in performance_metrics:
                try:
                    worst_val = float(performance_metrics[worst_key])
                    remaining_val = float(performance_metrics[remaining_key])
                    metric_names.append(metric.upper())
                    worst_values.append(worst_val)
                    remaining_values.append(remaining_val)
                except (ValueError, TypeError):
                    # Skip if values can't be converted to float
                    continue
        
        # If no paired metrics, look for any individual metrics to display
        if not metric_names:
            for key, value in performance_metrics.items():
                if 'worst' in key or 'remaining' in key:
                    try:
                        # Extract metric name by removing 'worst_' or 'remaining_' prefix
                        if key.startswith('worst_'):
                            metric = key[6:].upper()
                            metric_names.append(f"{metric} (Worst)")
                            worst_values.append(float(value))
                        elif key.startswith('remaining_'):
                            metric = key[10:].upper()
                            metric_names.append(f"{metric} (Remaining)")
                            worst_values.append(float(value))
                    except (ValueError, TypeError):
                        continue
        
        # If using existing chart generator
        if self.chart_generator and (metric_names and worst_values):
            # If we have paired metrics
            if len(remaining_values) > 0:
                # Create data for performance comparison chart
                performance_data = {
                    'models': ['Worst Samples', 'Remaining Samples'],
                    'metrics': metric_names
                }
                
                # Add metrics data
                for i, metric in enumerate(metric_names):
                    performance_data[metric] = [worst_values[i], remaining_values[i]]
                
                try:
                    # Try model_metrics_heatmap if available
                    if hasattr(self.chart_generator, 'model_metrics_heatmap'):
                        return self.chart_generator.model_metrics_heatmap(
                            results_df=performance_data,
                            title=title
                        )
                    # Fall back to bar_chart for first metric if available
                    elif hasattr(self.chart_generator, 'bar_chart') and len(metric_names) > 0:
                        bar_data = {
                            'x': ['Worst Samples', 'Remaining Samples'],
                            'y': [worst_values[0], remaining_values[0]],
                            'x_label': 'Sample Group',
                            'y_label': metric_names[0]
                        }
                        return self.chart_generator.bar_chart(
                            data=bar_data,
                            title=f"Performance Gap: {metric_names[0]}"
                        )
                except Exception as e:
                    logger.error(f"Error using chart generator for performance gap: {str(e)}")
            # If we only have individual metrics
            elif len(worst_values) > 0:
                try:
                    # Create a simple bar chart with available metrics
                    bar_data = {
                        'x': metric_names,
                        'y': worst_values,
                        'x_label': 'Metric',
                        'y_label': 'Value'
                    }
                    return self.chart_generator.bar_chart(
                        data=bar_data,
                        title="Performance Metrics"
                    )
                except Exception as e:
                    logger.error(f"Error using chart generator for performance metrics: {str(e)}")
        
        # Fallback - implement direct charting if needed
        try:
            if not metric_names:
                logger.warning("No matching metrics found for performance gap chart")
                return ""
            
            # Create figure
            fig, ax = self.plt.subplots(figsize=(12, 8))
            
            # If we have paired metrics
            if len(remaining_values) > 0:
                # Create DataFrame for the plot
                df = self.pd.DataFrame({
                    'Metric': metric_names,
                    'Worst': worst_values,
                    'Remaining': remaining_values
                })
                
                # Melt the DataFrame for seaborn
                df_melted = self.pd.melt(df, id_vars=['Metric'], value_vars=['Worst', 'Remaining'],
                                      var_name='Sample Group', value_name='Value')
                
                # Create grouped bar chart
                bars = self.sns.barplot(x='Metric', y='Value', hue='Sample Group', data=df_melted, 
                                    ax=ax, palette=['#ff9999', '#66b3ff'])
            else:
                # Create a simple bar chart with available metrics
                df = self.pd.DataFrame({
                    'Metric': metric_names,
                    'Value': worst_values
                })
                
                # Create bar chart
                bars = self.sns.barplot(x='Metric', y='Value', data=df, ax=ax)
            
            # Labels and title
            ax.set_title(title, fontsize=16)
            ax.set_xlabel("", fontsize=14)
            ax.set_ylabel("Score", fontsize=14)
            
            # Add grid for better readability
            ax.grid(True, axis='y', linestyle='--', alpha=0.7)
            
            # Tight layout
            self.plt.tight_layout()
            
            return self._save_figure_to_base64(fig)
        except Exception as e:
            logger.error(f"Error generating performance gap chart: {str(e)}")
            return ""