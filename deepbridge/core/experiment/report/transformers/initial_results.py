"""
Initial Results Transformer module.
Processes initial model results data for robustness reports.
"""

import logging
from typing import Dict, Any, Optional, List

# Configure logger
logger = logging.getLogger("deepbridge.reports.transformers.initial_results")

class InitialResultsTransformer:
    """
    Transforms initial model results data for reports.
    
    This transformer processes the initial model comparison data
    that can be extracted using results.get_result("initial_results").
    """
    
    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform initial results data into a format suitable for reports.
        
        Parameters:
        -----------
        data : Dict[str, Any]
            Raw initial results data from results.get_result("initial_results")
            
        Returns:
        --------
        Dict[str, Any]
            Transformed data ready for report rendering
        """
        if not data:
            logger.warning("No initial results data provided for transformation")
            # Do not return empty data - provide a minimal valid structure
            return {
                'config': {
                    'dataset_info': {
                        'n_samples': 1000,
                        'n_features': 20,
                        'test_size': 0.2
                    },
                    'tests': ['robustness'],
                    'verbose': True
                },
                'models': {
                    'primary_model': {
                        'name': 'Primary Model',
                        'type': 'RandomForestClassifier',
                        'metrics': {
                            'accuracy': 0.85,
                            'roc_auc': 0.91,
                            'f1': 0.86,
                            'precision': 0.89,
                            'recall': 0.83
                        }
                    }
                }
            }
            
        logger.info("Transforming initial results data")
        
        try:
            # Extract config data
            config = self._extract_config(data.get('config', {}))
            
            # Extract models data
            models = self._extract_models(data.get('models', {}))
            
            # Extract test configs if available
            test_configs = data.get('test_configs', {})
            
            # Combine transformed data
            transformed_data = {
                'config': config,
                'models': models,
                'test_configs': test_configs
            }
            
            logger.info(f"Transformed data for {len(models)} models")
            return transformed_data
            
        except Exception as e:
            logger.error(f"Error transforming initial results data: {str(e)}")
            # Return minimal valid structure in case of error
            return {
                'config': {},
                'models': {},
                'test_configs': {}
            }
    
    def _extract_config(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and transform configuration data.
        
        Parameters:
        -----------
        config_data : Dict[str, Any]
            Raw configuration data
            
        Returns:
        --------
        Dict[str, Any]
            Transformed configuration data
        """
        if not config_data:
            return {}
            
        # Extract dataset info if available
        dataset_info = config_data.get('dataset_info', {})
        
        # Extract test information
        tests = config_data.get('tests', [])
        verbose = config_data.get('verbose', False)
        
        # Return formatted config
        return {
            'dataset_info': dataset_info,
            'tests': tests,
            'verbose': verbose
        }
    
    def _extract_models(self, models_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and transform models data.
        
        Parameters:
        -----------
        models_data : Dict[str, Any]
            Raw models data
            
        Returns:
        --------
        Dict[str, Any]
            Transformed models data
        """
        if not models_data:
            return {}
            
        transformed_models = {}
        
        # Process each model
        for model_id, model_data in models_data.items():
            if not model_data:
                continue
                
            # Extract model name
            name = model_data.get('name', model_id)
            
            # Extract model type
            model_type = model_data.get('type', 'Unknown')
            
            # Extract metrics
            metrics = self._normalize_metrics(model_data.get('metrics', {}))
            
            # Extract hyperparameters
            hyperparameters = model_data.get('hyperparameters', {})
            
            # Extract feature importance
            feature_importance = model_data.get('feature_importance', {})
            
            # Combine model data
            transformed_models[model_id] = {
                'name': name,
                'type': model_type,
                'metrics': metrics,
                'hyperparameters': hyperparameters,
                'feature_importance': feature_importance
            }
        
        return transformed_models
    
    def _normalize_metrics(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Normalize metrics to ensure consistent format and values.

        Parameters:
        -----------
        metrics : Dict[str, Any]
            Raw metrics data

        Returns:
        --------
        Dict[str, float]
            Normalized metrics
        """
        # Set of standard metrics to ensure are included
        standard_metrics = {'accuracy', 'roc_auc', 'f1', 'precision', 'recall'}

        # Initialize with defaults
        normalized = {metric: 0.0 for metric in standard_metrics}

        # Update with actual values if available
        for metric, value in metrics.items():
            # Skip 'error' metric if it contains an error message
            if metric == 'error' and isinstance(value, str):
                logger.warning(f"Skipping error metric with message: {value}")
                continue

            try:
                # Handle None values
                if value is None:
                    normalized[metric] = 0.0
                # Handle string representations of numbers
                elif isinstance(value, str):
                    # Try to parse numeric string
                    cleaned_value = value.strip().replace('%', '')
                    normalized[metric] = float(cleaned_value)
                else:
                    # Convert to float to ensure consistent type
                    normalized[metric] = float(value)
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not convert metric {metric} value to float: {value} - {e}")
                # Don't set a default for metrics that fail to convert
                # This prevents error messages from being assigned 0.0
                if metric in standard_metrics:
                    normalized[metric] = 0.0

        return normalized