"""
Enhanced charts for uncertainty visualization.

This module adds additional chart types specifically for visualizing
reliability analysis and feature distributions from enhanced uncertainty tests.
"""

import base64
import io
import logging
import numpy as np
import traceback
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger("deepbridge.reports")

class EnhancedUncertaintyCharts:
    """Class providing additional chart generators for enhanced uncertainty visualization."""
    
    def __init__(self, base_chart_generator=None):
        """
        Initialize the enhanced chart generator.
        
        Parameters:
        -----------
        base_chart_generator : object, optional
            Base chart generator to delegate to when needed
        """
        self.base_generator = base_chart_generator
        
    def has_visualization_libs(self):
        """Check if required visualization libraries are available."""
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            import seaborn as sns
            return True
        except ImportError:
            return False
            
    def generate_reliability_distribution(self, data: Dict[str, Any], feature: Optional[str] = None):
        """
        Generate chart showing distributions of feature values for reliable vs unreliable predictions.
        
        Parameters:
        -----------
        data : Dict[str, Any]
            Transformed uncertainty data
        feature : str, optional
            Specific feature to visualize (default: use first feature found)
        
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
            
            # Get reliability analysis data with detailed logging
            logger.info(f"[DISTRIBUTION_DEBUG] generate_reliability_distribution called with data keys: {list(data.keys())}")
            
            if 'reliability_analysis' not in data:
                logger.error("[DISTRIBUTION_DEBUG] No reliability_analysis data found in input data")
                # Log some of the top-level keys to help debug
                for key in list(data.keys())[:5]:  # First 5 keys
                    if isinstance(data[key], dict):
                        logger.info(f"[DISTRIBUTION_DEBUG] data['{key}'] is a dict with keys: {list(data[key].keys())}")
                    else:
                        logger.info(f"[DISTRIBUTION_DEBUG] data['{key}'] type: {type(data[key])}")
                return None
            
            # Log reliability_analysis structure
            reliability = data['reliability_analysis']
            logger.info(f"[DISTRIBUTION_DEBUG] reliability_analysis keys: {list(reliability.keys())}")
            
            if 'feature_distributions' not in reliability:
                logger.error("[DISTRIBUTION_DEBUG] No feature_distributions found in reliability_analysis")
                return None
            
            # Log feature_distributions structure
            feature_distributions = reliability['feature_distributions']
            logger.info(f"[DISTRIBUTION_DEBUG] feature_distributions keys: {list(feature_distributions.keys())}")
            
            # Check if there are any distribution types with data
            has_data = False
            for dist_type, features in feature_distributions.items():
                if features and isinstance(features, dict):
                    logger.info(f"[DISTRIBUTION_DEBUG] Distribution type '{dist_type}' has {len(features)} features")
                    has_data = True
                    # Log a few feature names as example
                    feature_list = list(features.keys())
                    logger.info(f"[DISTRIBUTION_DEBUG] Example features: {feature_list[:3]}")
                    # Check if the feature values are valid lists with data
                    for fname in feature_list[:2]:  # Check first 2 features
                        feature_values = features[fname]
                        if isinstance(feature_values, (list, tuple)):
                            logger.info(f"[DISTRIBUTION_DEBUG] Feature '{fname}' has {len(feature_values)} values")
                            if len(feature_values) > 0:
                                logger.info(f"[DISTRIBUTION_DEBUG] Feature '{fname}' sample values: {feature_values[:5]}")
                        else:
                            logger.info(f"[DISTRIBUTION_DEBUG] Feature '{fname}' has non-list data: {type(feature_values)}")
            
            if not has_data:
                logger.error("[DISTRIBUTION_DEBUG] feature_distributions exists but contains no valid data")
                return None
                
            feature_distributions = reliability['feature_distributions']
            
            # If no specific feature, use the first one available
            if feature is None:
                available_features = list(feature_distributions.get('reliable', {}).keys())
                if not available_features:
                    logger.error("No features found in distributions")
                    return None
                feature = available_features[0]
                
            # Check if feature exists
            if feature not in feature_distributions.get('reliable', {}) or feature not in feature_distributions.get('unreliable', {}):
                logger.error(f"Feature {feature} not found in distributions")
                return None
                
            # Get data for the feature
            reliable_values = feature_distributions['reliable'][feature]
            unreliable_values = feature_distributions['unreliable'][feature]
            
            # Set style
            plt.style.use('seaborn-v0_8')
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot distributions
            sns.kdeplot(reliable_values, ax=ax, label='Reliable Predictions', color='green', fill=True, alpha=0.4)
            sns.kdeplot(unreliable_values, ax=ax, label='Unreliable Predictions', color='red', fill=True, alpha=0.4)
            
            # Add a rug plot to show individual data points
            sns.rugplot(reliable_values, ax=ax, color='green', alpha=0.3)
            sns.rugplot(unreliable_values, ax=ax, color='red', alpha=0.3)
            
            # Add labels and legend
            ax.set_title(f'Distribution of {feature} for Reliable vs Unreliable Predictions')
            ax.set_xlabel(feature)
            ax.set_ylabel('Density')
            ax.legend()
            
            # Add informative text
            reliable_count = len(reliable_values)
            unreliable_count = len(unreliable_values)
            total = reliable_count + unreliable_count
            
            # Calculate PSI if available
            psi_value = reliability.get('psi_values', {}).get(feature, None)
            psi_text = f"PSI: {psi_value:.4f}" if psi_value is not None else "PSI: Not available"
            
            text = (f"Feature: {feature}\n"
                   f"Reliable predictions: {reliable_count} ({100*reliable_count/total:.1f}%)\n"
                   f"Unreliable predictions: {unreliable_count} ({100*unreliable_count/total:.1f}%)\n"
                   f"{psi_text}")
            
            plt.figtext(0.02, 0.02, text, fontsize=9, ha='left')
            
            # Return base64 image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            
            return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
            
        except Exception as e:
            logger.error(f"Error generating reliability distribution chart: {str(e)}")
            logger.error(traceback.format_exc())
            return None
            
    def generate_marginal_bandwidth_chart(self, data: Dict[str, Any], feature: Optional[str] = None):
        """
        Generate chart showing how prediction interval width varies with feature values.
        
        Parameters:
        -----------
        data : Dict[str, Any]
            Transformed uncertainty data
        feature : str, optional
            Specific feature to visualize (default: use first feature found)
        
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
            
            # Get marginal bandwidth data with detailed logging
            logger.info(f"[BANDWIDTH_DEBUG] generate_marginal_bandwidth_chart called with data keys: {list(data.keys())}")
            logger.info(f"[BANDWIDTH_DEBUG] Feature parameter: {feature}")
            
            if 'marginal_bandwidth' not in data:
                logger.error("[BANDWIDTH_DEBUG] No marginal_bandwidth data found in input data")
                # Log some of the top-level keys to help debug
                for key in list(data.keys())[:5]:  # First 5 keys
                    if isinstance(data[key], dict):
                        logger.info(f"[BANDWIDTH_DEBUG] data['{key}'] is a dict with keys: {list(data[key].keys())}")
                    else:
                        logger.info(f"[BANDWIDTH_DEBUG] data['{key}'] type: {type(data[key])}")
                return None
            
            # Log marginal_bandwidth structure
            marginal_bandwidth = data['marginal_bandwidth']
            logger.info(f"[BANDWIDTH_DEBUG] marginal_bandwidth type: {type(marginal_bandwidth)}")
            
            if not isinstance(marginal_bandwidth, dict):
                logger.error(f"[BANDWIDTH_DEBUG] marginal_bandwidth is not a dictionary: {type(marginal_bandwidth)}")
                return None
                
            logger.info(f"[BANDWIDTH_DEBUG] marginal_bandwidth keys: {list(marginal_bandwidth.keys())}")
            
            # If no specific feature, use the first one available
            if feature is None:
                available_features = list(marginal_bandwidth.keys())
                if not available_features:
                    logger.error("[BANDWIDTH_DEBUG] No features found in marginal bandwidth data")
                    return None
                feature = available_features[0]
                logger.info(f"[BANDWIDTH_DEBUG] Selected first available feature: {feature}")
            
            # Check if feature exists
            if feature not in marginal_bandwidth:
                logger.error(f"[BANDWIDTH_DEBUG] Feature {feature} not found in marginal bandwidth data")
                return None
            
            # Log feature data structure
            feature_data = marginal_bandwidth[feature]
            logger.info(f"[BANDWIDTH_DEBUG] Feature data keys: {list(feature_data.keys())}")
            
            # Check required components
            required_keys = ['bin_centers', 'avg_widths', 'counts_below_threshold', 'counts_above_threshold', 'threshold']
            missing_keys = [key for key in required_keys if key not in feature_data]
            
            if missing_keys:
                logger.error(f"[BANDWIDTH_DEBUG] Missing required data for feature {feature}: {missing_keys}")
                return None
            
            # Log data length and type
            for key in required_keys:
                if key in feature_data:
                    value = feature_data[key]
                    if isinstance(value, (list, tuple, np.ndarray)):
                        logger.info(f"[BANDWIDTH_DEBUG] {key} has {len(value)} values")
                        if len(value) > 0:
                            logger.info(f"[BANDWIDTH_DEBUG] {key} sample values: {value[:3]}")
                    else:
                        logger.info(f"[BANDWIDTH_DEBUG] {key} = {value} (type: {type(value)})")
                
            # Get data for the feature
            feature_data = marginal_bandwidth[feature]
            bin_centers = feature_data['bin_centers']
            avg_widths = feature_data['avg_widths']
            counts_below = feature_data['counts_below_threshold']
            counts_above = feature_data['counts_above_threshold']
            threshold = feature_data['threshold']
            
            # Set style
            plt.style.use('seaborn-v0_8')
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), height_ratios=[3, 1], sharex=True)
            
            # Plot average widths on top subplot
            ax1.plot(bin_centers, avg_widths, marker='o', linestyle='-', color='blue')
            
            # Add threshold line
            ax1.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.3f}')
            
            # Add labels and legend
            ax1.set_title(f'Interval Width vs {feature}')
            ax1.set_ylabel('Average Interval Width')
            ax1.legend()
            
            # Plot counts on bottom subplot
            width = min(np.diff(bin_centers).min() * 0.8, 0.8) if len(bin_centers) > 1 else 0.8
            bar_positions = np.arange(len(bin_centers))
            
            # Create stacked bar chart
            ax2.bar(bar_positions, counts_below, width, label='Reliable (Below Threshold)', color='green', alpha=0.7)
            ax2.bar(bar_positions, counts_above, width, bottom=counts_below, 
                   label='Unreliable (Above Threshold)', color='red', alpha=0.7)
            
            # Set x-ticks to bin centers
            ax2.set_xticks(bar_positions)
            ax2.set_xticklabels([f"{x:.2f}" for x in bin_centers], rotation=45)
            
            # Add labels and legend
            ax2.set_xlabel(feature)
            ax2.set_ylabel('Count')
            ax2.legend()
            
            # Adjust layout
            plt.tight_layout()
            
            # Return base64 image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            
            return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
            
        except Exception as e:
            logger.error(f"Error generating marginal bandwidth chart: {str(e)}")
            logger.error(traceback.format_exc())
            return None
            
    def generate_interval_widths_boxplot(self, data: Dict[str, Any]):
        """
        Generate boxplot showing distribution of prediction interval widths.
        
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
            
            # Get interval widths data with detailed logging
            logger.info(f"[INTERVAL_DEBUG] generate_interval_widths_boxplot called with data keys: {list(data.keys())}")
            
            if 'interval_widths' not in data:
                logger.error("[INTERVAL_DEBUG] No interval_widths data found in input data")
                # Log some of the top-level keys to help debug
                for key in list(data.keys())[:5]:  # First 5 keys
                    if isinstance(data[key], dict):
                        logger.info(f"[INTERVAL_DEBUG] data['{key}'] is a dict with keys: {list(data[key].keys())}")
                    else:
                        logger.info(f"[INTERVAL_DEBUG] data['{key}'] type: {type(data[key])}")
                return None
            
            # Log interval_widths structure
            interval_widths = data['interval_widths']
            logger.info(f"[INTERVAL_DEBUG] interval_widths type: {type(interval_widths)}")
            
            if not interval_widths:
                logger.error("[INTERVAL_DEBUG] Interval widths data is empty")
                return None
                
            # Log interval_widths details based on its type
            if isinstance(interval_widths, dict):
                logger.info(f"[INTERVAL_DEBUG] interval_widths is a dictionary with keys: {list(interval_widths.keys())}")
                # Sample check of values
                for model, widths in list(interval_widths.items())[:2]:  # First 2 models
                    if isinstance(widths, (list, tuple, np.ndarray)):
                        logger.info(f"[INTERVAL_DEBUG] Model '{model}' has {len(widths)} width values")
                        if len(widths) > 0:
                            logger.info(f"[INTERVAL_DEBUG] Model '{model}' sample values: {widths[:5]}")
                    else:
                        logger.info(f"[INTERVAL_DEBUG] Model '{model}' has non-list data: {type(widths)}")
            elif isinstance(interval_widths, list):
                logger.info(f"[INTERVAL_DEBUG] interval_widths is a list with {len(interval_widths)} elements")
                # Check what's inside the list
                if len(interval_widths) > 0:
                    first_item = interval_widths[0]
                    logger.info(f"[INTERVAL_DEBUG] First element type: {type(first_item)}")
                    
                    if isinstance(first_item, dict):
                        logger.info(f"[INTERVAL_DEBUG] First element is a dict with keys: {list(first_item.keys())}")
                        # Check if it's alpha-based data
                        if 'alpha' in first_item and 'widths' in first_item:
                            logger.info(f"[INTERVAL_DEBUG] Alpha-based data detected")
                            logger.info(f"[INTERVAL_DEBUG] First element: alpha={first_item['alpha']}, widths has {len(first_item['widths'])} values")
                    elif isinstance(first_item, (list, tuple, np.ndarray)):
                        logger.info(f"[INTERVAL_DEBUG] First element is a list with {len(first_item)} values")
                        if len(first_item) > 0:
                            logger.info(f"[INTERVAL_DEBUG] Sample values: {first_item[:5]}")
            else:
                logger.error(f"[INTERVAL_DEBUG] interval_widths has unexpected type: {type(interval_widths)}")
                
            # Format data for boxplot
            boxplot_data = []
            labels = []
            
            # Handle different possible formats of interval_widths
            if isinstance(interval_widths, dict):
                # Format: {'model_name': [...widths...], ...}
                for model_name, widths in interval_widths.items():
                    boxplot_data.append(widths)
                    labels.append(model_name)
            elif isinstance(interval_widths, list):
                if all(isinstance(item, dict) for item in interval_widths):
                    # Format: [{'alpha': x, 'widths': [...], ...}]
                    for item in interval_widths:
                        if 'alpha' in item and 'widths' in item:
                            boxplot_data.append(item['widths'])
                            labels.append(f"Alpha: {item['alpha']}")
                elif len(interval_widths) > 0:
                    # Just a list of widths
                    boxplot_data = [interval_widths]
                    labels = ["Interval Widths"]
            
            if not boxplot_data:
                logger.error("Could not format interval widths data for boxplot")
                return None
                
            # Set style
            plt.style.use('seaborn-v0_8')
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create boxplot
            sns.boxplot(data=boxplot_data, ax=ax)
            
            # Calculate threshold if available
            threshold = data.get('threshold_value', None)
            if threshold is None and 'reliability_analysis' in data:
                threshold = data['reliability_analysis'].get('threshold_value', None)
                
            # Add threshold line if available
            if threshold is not None:
                ax.axhline(y=threshold, color='red', linestyle='--', 
                          label=f'Reliability Threshold: {threshold:.3f}')
                ax.legend()
                
            # Add labels
            ax.set_title('Distribution of Prediction Interval Widths')
            ax.set_xlabel('Model / Configuration')
            ax.set_ylabel('Interval Width')
            ax.set_xticklabels(labels)
            
            # Add statistics
            mean_width = data.get('mean_width', None)
            median_width = data.get('median_width', None)
            
            if mean_width is not None or median_width is not None:
                stats_text = "Statistics:\n"
                if mean_width is not None:
                    stats_text += f"Mean width: {mean_width:.3f}\n"
                if median_width is not None:
                    stats_text += f"Median width: {median_width:.3f}\n"
                plt.figtext(0.02, 0.02, stats_text, fontsize=9, ha='left')
            
            # Return base64 image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            
            return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
            
        except Exception as e:
            logger.error(f"Error generating interval widths boxplot: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def generate_model_metrics_comparison(self, data: Dict[str, Any]):
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
            
        # Use the fixed implementation from fixed_radar.py if available
        try:
            # First check if we have the fixed implementation
            from . import HAS_RADAR_FIX
            if HAS_RADAR_FIX:
                from .fixed_radar import generate_radar_chart
                logger.info("Using fixed radar chart implementation")
                return generate_radar_chart(data, self.base_generator)
        except ImportError:
            logger.warning("Fixed radar chart implementation not available, using fallback")
        except Exception as e:
            logger.error(f"Error using fixed radar chart: {str(e)}")
            
        # Fallback implementation if the fixed version is not available
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
                
            # Skip the pandas DataFrame conversion that was causing issues
            # Instead, directly use the metrics dictionary
            logger.info(f"Using metrics dictionary with {len(metrics)} models for radar chart")
            
            # Create radar chart
            if self.base_generator and hasattr(self.base_generator, 'metrics_radar_chart'):
                # Use base generator's radar chart if available
                # Convert DataFrame to the format expected by metrics_radar_chart
                models_metrics = {}
                for model_name in df.index:
                    models_metrics[model_name] = {
                        'metrics': {
                            metric: float(df.loc[model_name, metric]) for metric in df.columns 
                        }
                    }
                
                # Now call metrics_radar_chart with the properly formatted data
                try:
                    return self.base_generator.metrics_radar_chart(
                        models_metrics=models_metrics,
                        title="Model Metrics Comparison"
                    )
                except Exception as radar_err:
                    logger.error(f"Error generating radar chart: {str(radar_err)}")
                    logger.error(traceback.format_exc())
                    # Continue to fallback implementation
            
            # Fallback to bar chart if radar chart not available
            # Set style
            plt.style.use('seaborn-v0_8')
            
            # Create figure with 5 subplots in a row
            fig, axes = plt.subplots(1, 5, figsize=(20, 5))
            
            # Create a bar chart for each metric
            metrics_list = ['Uncertainty Score', 'Coverage', 'Mean Width', 'MSE', 'MAE']
            for i, metric in enumerate(metrics_list):
                if metric in df.columns:
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
            
        except Exception as e:
            logger.error(f"Error generating model metrics comparison chart: {str(e)}")
            logger.error(traceback.format_exc())
            return None