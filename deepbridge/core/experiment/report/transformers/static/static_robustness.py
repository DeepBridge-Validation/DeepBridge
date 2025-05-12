"""
Data transformer for static robustness reports with Seaborn visualizations.
"""

import logging
from typing import Dict, List, Any, Optional, Union

# Configure logger
logger = logging.getLogger("deepbridge.reports")

class StaticRobustnessTransformer:
    """
    Transformer for preparing robustness data for static reports.
    """
    
    def __init__(self):
        """
        Initialize the transformer.
        """
        # Import the standard robustness transformer
        from ..robustness import RobustnessDataTransformer
        self.base_transformer = RobustnessDataTransformer()
    
    def transform(self, results: Dict[str, Any], model_name: str = "Model") -> Dict[str, Any]:
        """
        Transform the robustness results data for static report rendering.
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Raw robustness test results
        model_name : str, optional
            Name of the model for display in the report
            
        Returns:
        --------
        Dict[str, Any] : Transformed data for static report rendering
        """
        # First, use the base transformer to get the standard transformations
        transformed_data = self.base_transformer.transform(results, model_name)
        
        # Now perform additional transformations specific to static reports
        static_data = self._enhance_for_static_report(transformed_data)
        
        return static_data
    
    def _enhance_for_static_report(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance the transformed data with additional information needed for static reports.
        
        Parameters:
        -----------
        data : Dict[str, Any]
            Standard transformed data
            
        Returns:
        --------
        Dict[str, Any] : Enhanced data for static reports
        """
        # Create a copy to avoid modifying the original
        enhanced = dict(data)
        
        # Extract perturbation levels and iteration data for easier chart generation
        perturbation_levels = []
        iteration_data = {}
        
        if 'raw' in enhanced and 'by_level' in enhanced['raw']:
            raw_data = enhanced['raw']['by_level']
            
            # Get all perturbation levels
            perturbation_levels = sorted([float(level) for level in raw_data.keys()])
            
            # Extract iteration data for each level
            for level in perturbation_levels:
                level_str = str(level)
                
                if level_str in raw_data:
                    level_data = raw_data[level_str]
                    
                    # Extract all iteration scores for this level
                    level_scores = []
                    
                    if 'runs' in level_data and 'all_features' in level_data['runs']:
                        for run in level_data['runs']['all_features']:
                            if 'iterations' in run and 'scores' in run['iterations']:
                                level_scores.extend(run['iterations']['scores'])
                    
                    iteration_data[level] = level_scores
        
        enhanced['perturbation_levels'] = perturbation_levels
        enhanced['iterations_by_level'] = iteration_data
        
        # Process alternative models in the same way
        if 'alternative_models' in enhanced:
            alt_iterations = {}
            
            for model_name, model_data in enhanced['alternative_models'].items():
                model_iterations = {}
                
                if 'raw' in model_data and 'by_level' in model_data['raw']:
                    alt_raw_data = model_data['raw']['by_level']
                    
                    # Extract iteration data for each level
                    for level in perturbation_levels:
                        level_str = str(level)
                        
                        if level_str in alt_raw_data:
                            level_data = alt_raw_data[level_str]
                            
                            # Extract all iteration scores for this level
                            level_scores = []
                            
                            if 'runs' in level_data and 'all_features' in level_data['runs']:
                                for run in level_data['runs']['all_features']:
                                    if 'iterations' in run and 'scores' in run['iterations']:
                                        level_scores.extend(run['iterations']['scores'])
                            
                            model_iterations[level] = level_scores
                
                alt_iterations[model_name] = model_iterations
            
            enhanced['alt_iterations_by_level'] = alt_iterations
        
        # Prepare data for boxplot visualization
        boxplot_data = self._prepare_boxplot_data(enhanced)
        if boxplot_data:
            enhanced['boxplot_data'] = boxplot_data
        
        # Ensure feature importance exists and is sorted
        if 'feature_importance' in enhanced and enhanced['feature_importance']:
            # Ensure all values are numbers
            clean_importance = {}
            for feature, value in enhanced['feature_importance'].items():
                try:
                    clean_importance[feature] = float(value)
                except (ValueError, TypeError):
                    logger.warning(f"Non-numeric feature importance for {feature}: {value}, setting to 0")
                    clean_importance[feature] = 0.0

            # Sort feature importance for better visualization
            sorted_importance = dict(sorted(
                clean_importance.items(),
                key=lambda x: x[1],
                reverse=True
            ))
            enhanced['feature_importance'] = sorted_importance

        # Same for model feature importance
        if 'model_feature_importance' in enhanced and enhanced['model_feature_importance']:
            # Ensure all values are numbers
            clean_importance = {}
            for feature, value in enhanced['model_feature_importance'].items():
                try:
                    clean_importance[feature] = float(value)
                except (ValueError, TypeError):
                    logger.warning(f"Non-numeric model feature importance for {feature}: {value}, setting to 0")
                    clean_importance[feature] = 0.0

            enhanced['model_feature_importance'] = clean_importance
        
        # Add additional information that might be needed for static visualizations
        enhanced['visualization_type'] = 'static'
        enhanced['has_iterations'] = bool(iteration_data and any(iteration_data.values()))
        enhanced['n_iterations'] = self._count_iterations(enhanced)

        # Garantir que a lista de features esteja disponível
        if 'features' not in enhanced or not enhanced['features']:
            # Tentar extrair da feature_importance
            if 'feature_importance' in enhanced and enhanced['feature_importance']:
                enhanced['features'] = list(enhanced['feature_importance'].keys())
                logger.info(f"Extracted {len(enhanced['features'])} features from feature_importance")
            # Ou de primary_model se disponível
            elif 'primary_model' in enhanced:
                if 'features' in enhanced['primary_model'] and enhanced['primary_model']['features']:
                    enhanced['features'] = enhanced['primary_model']['features']
                    logger.info(f"Extracted {len(enhanced['features'])} features from primary_model.features")
                elif 'feature_importance' in enhanced['primary_model'] and enhanced['primary_model']['feature_importance']:
                    enhanced['features'] = list(enhanced['primary_model']['feature_importance'].keys())
                    logger.info(f"Extracted {len(enhanced['features'])} features from primary_model.feature_importance")
                else:
                    enhanced['features'] = []
            else:
                enhanced['features'] = []

            logger.info(f"Final features count: {len(enhanced['features'])}")

        return enhanced
    
    def _prepare_boxplot_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare data for boxplot visualization.
        
        Parameters:
        -----------
        data : Dict[str, Any]
            Enhanced data
            
        Returns:
        --------
        Dict[str, Any] : Boxplot data structure
        """
        # Extract data for primary model
        primary_model = {
            'name': data.get('model_name', 'Primary Model'),
            'modelType': data.get('model_type', 'Unknown'),
            'baseScore': data.get('base_score', 0),
            'scores': []
        }
        
        # Add scores from iterations_by_level
        if 'iterations_by_level' in data:
            for level_scores in data['iterations_by_level'].values():
                if level_scores:
                    primary_model['scores'].extend(level_scores)
        
        # Initialize boxplot models with primary model
        models = [primary_model]
        
        # Add alternative models
        if 'alternative_models' in data and 'alt_iterations_by_level' in data:
            for model_name, model_data in data['alternative_models'].items():
                alt_model = {
                    'name': model_name,
                    'modelType': model_data.get('model_type', 'Unknown'),
                    'baseScore': model_data.get('base_score', 0),
                    'scores': []
                }
                
                # Add scores from alt_iterations_by_level
                if model_name in data['alt_iterations_by_level']:
                    for level_scores in data['alt_iterations_by_level'][model_name].values():
                        if level_scores:
                            alt_model['scores'].extend(level_scores)
                
                # Only add if we have scores
                if alt_model['scores']:
                    models.append(alt_model)
        
        # Only return if we have valid data
        if any(model['scores'] for model in models):
            return {'models': models}
        
        return {}
    
    def _count_iterations(self, data: Dict[str, Any]) -> int:
        """
        Count the number of iterations per perturbation in the test.
        
        Parameters:
        -----------
        data : Dict[str, Any]
            Enhanced data
            
        Returns:
        --------
        int : Number of iterations
        """
        # Check if there's an explicit iterations count
        if 'n_iterations' in data:
            return data['n_iterations']
        
        # Try to infer from the raw data
        max_iterations = 0
        
        if 'raw' in data and 'by_level' in data['raw']:
            for level_data in data['raw']['by_level'].values():
                if 'runs' in level_data and 'all_features' in level_data['runs']:
                    for run in level_data['runs']['all_features']:
                        if 'iterations' in run and 'scores' in run['iterations']:
                            iteration_count = len(run['iterations']['scores'])
                            max_iterations = max(max_iterations, iteration_count)
        
        # If we still don't have a count, assume 1
        return max_iterations if max_iterations > 0 else 1