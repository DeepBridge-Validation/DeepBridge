"""
Fixed method for enhanced_charts.py to resolve the 'numpy.ndarray' is not callable error.

This module provides a corrected version of the generate_model_metrics_comparison method
that can be patched into the EnhancedUncertaintyCharts class.
"""

import logging
import base64
import io
import traceback
import numpy as np
from typing import Dict, Any

logger = logging.getLogger("deepbridge.reports")

def generate_model_metrics_comparison_fixed(self, data: Dict[str, Any]):
    """
    Generate comprehensive comparison of model metrics.
    
    Parameters:
    -----------
    data : Dict[str, Any]
        Transformed uncertainty data
    
    Returns:
    --------
    str : Base64-encoded image data
    """
    if not self.has_visualization_libs():
        logger.error("Visualization libraries not available")
        return None
        
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        
        # Get model metrics
        metrics = {}
        
        # Get primary model metrics - convert values to float to avoid numpy array issues
        metrics['Primary Model'] = {
            'Uncertainty Score': float(data.get('uncertainty_score', 0)),
            'Coverage': float(data.get('coverage', 0)),
            'Mean Width': float(data.get('mean_width', 0)),
        }
        
        # Add MSE and MAE if available
        if 'mse' in data:
            metrics['Primary Model']['MSE'] = float(data['mse'])
        if 'mae' in data:
            metrics['Primary Model']['MAE'] = float(data['mae'])
        
        # Get alternative models if available
        alt_models = data.get('alternative_models', {})
        for model_name, model_data in alt_models.items():
            model_metrics = {
                'Uncertainty Score': float(model_data.get('uncertainty_score', 0)),
                'Coverage': float(model_data.get('coverage', 0)),
                'Mean Width': float(model_data.get('mean_width', 0)),
            }
            
            # Add MSE and MAE if available
            if 'mse' in model_data:
                model_metrics['MSE'] = float(model_data['mse'])
            if 'mae' in model_data:
                model_metrics['MAE'] = float(model_data['mae'])
                
            metrics[model_name] = model_metrics
            
        # Convert to DataFrame for the bar chart
        df = pd.DataFrame.from_dict(metrics, orient='index')
        
        # Try radar chart if available
        if self.base_generator and hasattr(self.base_generator, 'metrics_radar_chart'):
            try:
                # Check parameter names using inspect
                import inspect
                sig = inspect.signature(self.base_generator.metrics_radar_chart)
                param_names = list(sig.parameters.keys())
                logger.info(f"metrics_radar_chart parameters: {param_names}")
                
                # Try calling with different parameter names
                if 'models_metrics' in param_names:
                    logger.info("Calling radar chart with models_metrics parameter")
                    return self.base_generator.metrics_radar_chart(models_metrics=metrics)
                elif 'metrics_data' in param_names:
                    logger.info("Calling radar chart with metrics_data parameter")
                    return self.base_generator.metrics_radar_chart(metrics_data=metrics)
                elif 'data' in param_names:
                    logger.info("Calling radar chart with data parameter")
                    return self.base_generator.metrics_radar_chart(data=metrics)
                else:
                    # No matching parameter name, try as positional argument
                    logger.info("Calling radar chart with positional parameter")
                    return self.base_generator.metrics_radar_chart(metrics)
            except Exception as e:
                logger.error(f"Error calling metrics_radar_chart: {str(e)}")
                logger.error(traceback.format_exc())
                # Continue to fallback implementation
        
        # Fallback to bar chart
        logger.info("Fallback to bar chart implementation")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # Create figure with subplots for each metric
        num_metrics = len(df.columns)
        if num_metrics > 0:
            fig, axes = plt.subplots(1, num_metrics, figsize=(4 * num_metrics, 5))
            
            # Handle case with only one metric
            if num_metrics == 1:
                axes = [axes]
                
            # Create a bar chart for each metric
            for i, metric in enumerate(df.columns):
                if i < len(axes):  # Ensure we don't go out of bounds
                    ax = axes[i]
                    df[metric].plot(kind='bar', ax=ax)
                    ax.set_title(metric)
                    ax.set_ylim(bottom=0)  # Start from zero
                    
                    # Add values on top of bars
                    for j, val in enumerate(df[metric]):
                        ax.text(j, val, f"{val:.3f}", ha='center', va='bottom')
            
            # Adjust layout
            plt.tight_layout()
            
            # Return base64 image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            
            return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
        else:
            logger.warning("No metrics available for chart")
            return None
        
    except Exception as e:
        logger.error(f"Error generating model metrics comparison chart: {str(e)}")
        logger.error(traceback.format_exc())
        return None

# Function to patch the EnhancedUncertaintyCharts class
def apply_patch():
    """Apply the fixed method to the EnhancedUncertaintyCharts class."""
    try:
        from deepbridge.templates.report_types.uncertainty.static.charts.enhanced_charts import EnhancedUncertaintyCharts
        
        # Replace the method
        EnhancedUncertaintyCharts.generate_model_metrics_comparison = generate_model_metrics_comparison_fixed
        
        # For method to be bound properly, it needs correct 'self' parameter
        # This should work as-is since we defined the fixed method with 'self' as first parameter
        
        logger.info("Successfully patched EnhancedUncertaintyCharts.generate_model_metrics_comparison")
        return True
    except Exception as e:
        logger.error(f"Error applying patch to EnhancedUncertaintyCharts: {str(e)}")
        logger.error(traceback.format_exc())
        return False