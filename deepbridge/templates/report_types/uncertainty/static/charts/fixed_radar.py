"""
Fix for the radar chart error in enhanced_charts.py

This module contains a replacement for the generate_model_metrics_comparison
method that correctly formats the data for the radar chart without trying to
call a numpy ndarray.
"""

import base64
import io
import logging
import numpy as np
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

logger = logging.getLogger("deepbridge.reports")

def generate_radar_chart(data, base_generator):
    """
    Generate a radar chart comparing model metrics.
    
    Args:
        data: The report data dictionary
        base_generator: The chart generator with metrics_radar_chart method
        
    Returns:
        str: Base64 encoded image if successful, None otherwise
    """
    try:
        # Get model metrics
        metrics = {}
        
        # Get primary model metrics
        metrics['Primary Model'] = {
            'Uncertainty Score': float(data.get('uncertainty_score', 0)),
            'Coverage': float(data.get('coverage', 0)),
            'Mean Width': float(data.get('mean_width', 0)),
            'MSE': float(data.get('mse', 0) if 'mse' in data else 0),
            'MAE': float(data.get('mae', 0) if 'mae' in data else 0)
        }
        
        # Get alternative models if available
        alt_models = data.get('alternative_models', {})
        for model_name, model_data in alt_models.items():
            metrics[model_name] = {
                'Uncertainty Score': float(model_data.get('uncertainty_score', 0)),
                'Coverage': float(model_data.get('coverage', 0)),
                'Mean Width': float(model_data.get('mean_width', 0)),
                'MSE': float(model_data.get('mse', 0) if 'mse' in model_data else 0),
                'MAE': float(model_data.get('mae', 0) if 'mae' in model_data else 0)
            }
        
        logger.info(f"Prepared metrics dictionary with {len(metrics)} models")
        
        # Try to use the metrics_radar_chart method with correct parameters
        try:
            if hasattr(base_generator, 'metrics_radar_chart'):
                import inspect
                try:
                    # Get parameter signature
                    sig = inspect.signature(base_generator.metrics_radar_chart)
                    param_names = list(sig.parameters.keys())
                    logger.info(f"Found metrics_radar_chart method with parameters: {param_names}")
                    
                    # Try calling with models_metrics parameter
                    if 'models_metrics' in param_names:
                        logger.info("Calling metrics_radar_chart with models_metrics parameter")
                        chart = base_generator.metrics_radar_chart(models_metrics=metrics)
                        return chart
                    
                    # Try with different parameter names that might be used
                    elif 'metrics_data' in param_names:
                        logger.info("Calling metrics_radar_chart with metrics_data parameter")
                        chart = base_generator.metrics_radar_chart(metrics_data=metrics)
                        return chart
                    
                    elif 'data' in param_names:
                        logger.info("Calling metrics_radar_chart with data parameter")
                        chart = base_generator.metrics_radar_chart(data=metrics)
                        return chart
                    
                    # Use positional parameter as last resort
                    else:
                        logger.info("Calling metrics_radar_chart with positional parameter")
                        chart = base_generator.metrics_radar_chart(metrics)
                        return chart
                except Exception as e:
                    logger.error(f"Error calling metrics_radar_chart: {str(e)}")
                    logger.error(traceback.format_exc())
            else:
                logger.error("Base generator does not have metrics_radar_chart method")
        except Exception as e:
            logger.error(f"Error during metrics_radar_chart method check: {str(e)}")
            logger.error(traceback.format_exc())
        
        # Fall back to creating a bar chart
        logger.info("Falling back to bar chart implementation")
        
        # Create pandas DataFrame for the chart
        try:
            import pandas as pd
            # Convert metrics dict to DataFrame
            df = pd.DataFrame.from_dict(metrics, orient='index')
            
            # Set style
            plt.style.use('seaborn-v0_8')
            
            # Create figure with 5 subplots in a row
            fig, axes = plt.subplots(1, 5, figsize=(20, 5))
            
            # Create a bar chart for each metric
            metrics_list = list(df.columns)
            
            # Handle case with less than 5 metrics
            for i in range(min(len(metrics_list), len(axes))):
                metric = metrics_list[i]
                ax = axes[i]
                df[metric].plot(kind='bar', ax=ax)
                ax.set_title(metric)
                ax.set_ylim(bottom=0)  # Start from zero
                
                # Add values on top of bars
                for j, val in enumerate(df[metric]):
                    ax.text(j, val, f"{val:.3f}", ha='center', va='bottom')
            
            # Hide unused axes
            for i in range(len(metrics_list), len(axes)):
                axes[i].set_visible(False)
            
            # Adjust layout
            plt.tight_layout()
            
            # Return base64 image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            
            return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
            
        except Exception as e:
            logger.error(f"Error generating bar chart fallback: {str(e)}")
            logger.error(traceback.format_exc())
            return None
            
    except Exception as e:
        logger.error(f"Error in generate_radar_chart: {str(e)}")
        logger.error(traceback.format_exc())
        return None

# Patch function to update the EnhancedUncertaintyCharts class
def patch_enhanced_charts():
    """
    Patch the EnhancedUncertaintyCharts class to fix the radar chart error.
    """
    try:
        from deepbridge.templates.report_types.uncertainty.static.charts.enhanced_charts import EnhancedUncertaintyCharts
        
        # Replace the generate_model_metrics_comparison method
        def new_generate_model_metrics_comparison(self, data: Dict[str, Any]):
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
                
            return generate_radar_chart(data, self.base_generator)
        
        # Replace the method in the class
        EnhancedUncertaintyCharts.generate_model_metrics_comparison = new_generate_model_metrics_comparison
        logger.info("Successfully patched EnhancedUncertaintyCharts.generate_model_metrics_comparison")
        return True
    except Exception as e:
        logger.error(f"Error patching EnhancedUncertaintyCharts: {str(e)}")
        logger.error(traceback.format_exc())
        return False