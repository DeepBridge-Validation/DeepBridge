"""
Model comparison chart generator for uncertainty visualization.
"""

import logging
import matplotlib.pyplot as plt
import numpy as np
import base64
import io
from typing import Dict, Any, List, Optional, Union, Tuple

from .base_chart import BaseChartGenerator

# Configure logger
logger = logging.getLogger("deepbridge.reports")


class ModelComparisonChart(BaseChartGenerator):
    """
    Generator for model comparison charts in uncertainty reports.
    """
    
    def _validate_data(self, models_data):
        """
        Validate data for model comparison chart.
        
        Parameters:
        -----------
        models_data : Dict[str, Any]
            Data to validate
        
        Returns:
        --------
        bool : Whether the data is valid
        """
        import logging
        logger = logging.getLogger("deepbridge.reports")
        
        logger.info(f"Validating model comparison data: {type(models_data)}")
        
        if not isinstance(models_data, dict):
            logger.warning("models_data is not a dictionary")
            return False
            
        # Log the keys in models_data
        logger.info(f"models_data keys: {list(models_data.keys())}")
        
        # Check for primary model metrics
        has_primary_metrics = False
        standard_metrics = ['uncertainty_score', 'coverage', 'mean_width']
        
        # Check each metric directly in models_data
        for metric in standard_metrics:
            if metric in models_data:
                logger.info(f"Found primary model metric '{metric}': {models_data[metric]}")
                has_primary_metrics = True
                
        # Check metrics dictionary if present
        if 'metrics' in models_data and isinstance(models_data['metrics'], dict):
            logger.info(f"metrics keys: {list(models_data['metrics'].keys())}")
            for metric in standard_metrics:
                if metric in models_data['metrics']:
                    logger.info(f"Found primary model metric '{metric}' in metrics: {models_data['metrics'][metric]}")
                    has_primary_metrics = True
                    
        # Check alternative models
        has_alternative_models = False
        if 'alternative_models' in models_data and isinstance(models_data['alternative_models'], dict):
            logger.info(f"Found {len(models_data['alternative_models'])} alternative models")
            
            # Log up to 3 model names as examples
            model_names = list(models_data['alternative_models'].keys())
            if model_names:
                logger.info(f"Example models: {model_names[:3]}")
                
            # Check if any alternative model has metrics
            for model_name, model_data in models_data['alternative_models'].items():
                if isinstance(model_data, dict):
                    for metric in standard_metrics:
                        if metric in model_data:
                            logger.info(f"Found metric '{metric}' in alternative model '{model_name}'")
                            has_alternative_models = True
                            break
                            
                    if 'metrics' in model_data and isinstance(model_data['metrics'], dict):
                        for metric in standard_metrics:
                            if metric in model_data['metrics']:
                                logger.info(f"Found metric '{metric}' in metrics of alternative model '{model_name}'")
                                has_alternative_models = True
                                break
        
        # Valid if we have either primary or alternative model metrics
        is_valid = has_primary_metrics or has_alternative_models
        logger.info(f"Model comparison data validation result: {is_valid}")
        return is_valid

    def generate(self, models_data: Dict[str, Any], 
                 metrics: List[str] = None,
                 title: str = "Model Uncertainty Metrics Comparison") -> Optional[str]:
        """
        Generate a model comparison chart showing key uncertainty metrics across models.

        Parameters:
        -----------
        models_data : Dict[str, Any]
            Data containing multiple models with their metrics
        metrics : List[str], optional
            List of metrics to include in the comparison
            Defaults to ['uncertainty_score', 'coverage', 'mean_width']
        title : str, optional
            Title for the chart

        Returns:
        --------
        Optional[str] : Base64 encoded image or None if generation fails
        """
        try:
            # Validate data
            if not self._validate_data(models_data):
                logger.warning("Invalid data provided for model comparison chart")
                return None

            # Default metrics if not provided
            if not metrics:
                metrics = ['uncertainty_score', 'coverage', 'mean_width']

            # Try to use alternative model data if available
            primary_model_name = models_data.get('model_name', 'Primary Model')
            all_models = {}
            
            # Add primary model if available
            primary_model_data = {}
            for metric in metrics:
                if metric in models_data:
                    primary_model_data[metric] = models_data[metric]
                elif 'metrics' in models_data and metric in models_data['metrics']:
                    primary_model_data[metric] = models_data['metrics'][metric]
            
            if primary_model_data:
                all_models[primary_model_name] = primary_model_data
            
            # Add alternative models if available
            if 'alternative_models' in models_data and isinstance(models_data['alternative_models'], dict):
                for model_name, model_data in models_data['alternative_models'].items():
                    model_metrics = {}
                    for metric in metrics:
                        if metric in model_data:
                            model_metrics[metric] = model_data[metric]
                        elif 'metrics' in model_data and metric in model_data['metrics']:
                            model_metrics[metric] = model_data['metrics'][metric]
                    
                    if model_metrics:
                        all_models[model_name] = model_metrics
            
            # If no models with valid metrics, return None
            if not all_models:
                logger.warning("No models with valid metrics found for comparison chart")
                return None
            
            # Create plot
            plt.figure(figsize=(10, 6))
            
            # Set up bar positions
            n_models = len(all_models)
            n_metrics = len(metrics)
            bar_width = 0.8 / n_metrics
            
            # Set up colors
            colors = plt.cm.viridis(np.linspace(0, 0.8, n_metrics))
            
            # Plot bars for each metric
            for i, metric in enumerate(metrics):
                model_names = []
                metric_values = []
                
                for model_name, model_data in all_models.items():
                    if metric in model_data:
                        model_names.append(model_name)
                        metric_values.append(model_data[metric])
                
                if metric_values:
                    x = np.arange(len(model_names))
                    plt.bar(x + i * bar_width - (n_metrics - 1) * bar_width / 2, 
                            metric_values, 
                            width=bar_width, 
                            label=metric.replace('_', ' ').title(),
                            color=colors[i],
                            alpha=0.8)
            
            # Set up plot appearance
            plt.xlabel('Models')
            plt.ylabel('Values')
            plt.title(title)
            plt.xticks(np.arange(len(model_names)), model_names, rotation=45, ha='right')
            plt.legend(loc='best')
            plt.tight_layout()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Convert plot to base64
            return self._save_figure_to_base64(plt.gcf())
            
        except Exception as e:
            logger.error(f"Error generating model comparison chart: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None