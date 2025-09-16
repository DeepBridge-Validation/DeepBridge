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

            # Also prepare individual_feature_impacts for the new chart
            # The impacts show how much performance changes when each feature is perturbed
            # Positive values = performance drops (feature is important for robustness)
            # Negative values = performance improves (unusual but possible)
            individual_impacts = {}
            for feature, impact in clean_importance.items():
                # The impact is already calculated correctly in RobustnessEvaluator
                # Positive impact means the feature is sensitive (performance drops when perturbed)
                individual_impacts[feature] = -impact  # Invert to show drop as negative

            enhanced['individual_feature_impacts'] = individual_impacts
            logger.info(f"Prepared individual_feature_impacts with {len(individual_impacts)} features")

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
        
        # Prepare method comparison data if both raw and quantile results exist
        method_comparison_data = self._prepare_method_comparison_data(enhanced)
        if method_comparison_data:
            enhanced['method_comparison'] = method_comparison_data
            logger.info("Prepared method comparison data for raw vs quantile visualization")

        # Prepare selected features comparison data if both all_features and feature_subset results exist
        selected_features_comparison = self._prepare_selected_features_comparison_data(enhanced)
        if selected_features_comparison:
            enhanced['selected_features_comparison'] = selected_features_comparison
            logger.info("Prepared selected features comparison data for all vs selected visualization")

        # Prepare detailed distribution data for enhanced boxplot
        detailed_distribution_data = self._prepare_detailed_distribution_data(enhanced)
        if detailed_distribution_data:
            enhanced['detailed_distribution'] = detailed_distribution_data
            logger.info("Prepared detailed distribution data for enhanced boxplot visualization")

        # Prepare distribution grid data for comprehensive matrix visualization
        distribution_grid_data = self._prepare_distribution_grid_data(enhanced)
        if distribution_grid_data:
            enhanced['distribution_grid'] = distribution_grid_data
            logger.info(f"Prepared distribution grid data for {distribution_grid_data['n_models']} models across {distribution_grid_data['n_levels']} levels")

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

    def _prepare_method_comparison_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare data for method comparison chart (raw vs quantile).

        Parameters:
        -----------
        data : Dict[str, Any]
            Enhanced data containing both raw and quantile results

        Returns:
        --------
        Dict[str, Any] : Method comparison data or empty dict if not available
        """
        # Check if we have both raw and quantile data
        if ('raw' not in data or 'by_level' not in data['raw'] or
            'quantile' not in data or 'by_level' not in data['quantile']):
            return {}

        raw_data = data['raw']['by_level']
        quantile_data = data['quantile']['by_level']

        # Get common perturbation levels
        raw_levels = set(raw_data.keys())
        quantile_levels = set(quantile_data.keys())
        common_levels = raw_levels & quantile_levels

        if not common_levels:
            logger.warning("No common perturbation levels found between raw and quantile methods")
            return {}

        # Sort levels numerically
        sorted_levels = sorted([float(level) for level in common_levels])

        # Extract scores for each method
        raw_scores = []
        quantile_scores = []
        raw_worst_scores = []
        quantile_worst_scores = []

        for level in sorted_levels:
            level_str = str(level)

            # Raw method scores
            if (level_str in raw_data and 'overall_result' in raw_data[level_str] and
                'all_features' in raw_data[level_str]['overall_result']):
                raw_result = raw_data[level_str]['overall_result']['all_features']
                raw_scores.append(raw_result.get('mean_score', 0))
                raw_worst_scores.append(raw_result.get('worst_score', raw_result.get('min_score', 0)))
            else:
                raw_scores.append(0)
                raw_worst_scores.append(0)

            # Quantile method scores
            if (level_str in quantile_data and 'overall_result' in quantile_data[level_str] and
                'all_features' in quantile_data[level_str]['overall_result']):
                quantile_result = quantile_data[level_str]['overall_result']['all_features']
                quantile_scores.append(quantile_result.get('mean_score', 0))
                quantile_worst_scores.append(quantile_result.get('worst_score', quantile_result.get('min_score', 0)))
            else:
                quantile_scores.append(0)
                quantile_worst_scores.append(0)

        # Only return data if we have valid scores
        if not any(raw_scores) or not any(quantile_scores):
            logger.warning("No valid scores found for method comparison")
            return {}

        logger.info(f"Prepared method comparison data for {len(sorted_levels)} perturbation levels")

        return {
            'perturbation_levels': sorted_levels,
            'raw_scores': raw_scores,
            'quantile_scores': quantile_scores,
            'raw_worst_scores': raw_worst_scores,
            'quantile_worst_scores': quantile_worst_scores,
            'base_score': data.get('base_score', 0),
            'metric': data.get('metric', 'Score')
        }

    def _prepare_selected_features_comparison_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare data for selected features comparison chart (all features vs selected features).

        Parameters:
        -----------
        data : Dict[str, Any]
            Enhanced data containing both all_features and feature_subset results

        Returns:
        --------
        Dict[str, Any] : Selected features comparison data or empty dict if not available
        """
        # Check if we have feature_subset information
        if 'feature_subset' not in data or not data['feature_subset']:
            return {}

        selected_features = data['feature_subset']

        # Try to find data with both all_features and feature_subset results
        # First check raw data
        comparison_data = None
        method_name = None

        if ('raw' in data and 'by_level' in data['raw']):
            raw_data = data['raw']['by_level']
            # Check if any level has both all_features and feature_subset
            for level_key, level_data in raw_data.items():
                if ('overall_result' in level_data and
                    'all_features' in level_data['overall_result'] and
                    'feature_subset' in level_data['overall_result']):
                    comparison_data = raw_data
                    method_name = 'raw'
                    break

        # If not found in raw, try quantile
        if not comparison_data and ('quantile' in data and 'by_level' in data['quantile']):
            quantile_data = data['quantile']['by_level']
            # Check if any level has both all_features and feature_subset
            for level_key, level_data in quantile_data.items():
                if ('overall_result' in level_data and
                    'all_features' in level_data['overall_result'] and
                    'feature_subset' in level_data['overall_result']):
                    comparison_data = quantile_data
                    method_name = 'quantile'
                    break

        if not comparison_data:
            logger.warning("No comparison data found with both all_features and feature_subset results")
            return {}

        # Extract perturbation levels
        levels_with_both = []
        for level_key, level_data in comparison_data.items():
            if ('overall_result' in level_data and
                'all_features' in level_data['overall_result'] and
                'feature_subset' in level_data['overall_result']):
                levels_with_both.append(float(level_key))

        if not levels_with_both:
            logger.warning("No perturbation levels found with both all_features and feature_subset data")
            return {}

        # Sort levels numerically
        sorted_levels = sorted(levels_with_both)

        # Extract scores for each approach
        all_features_scores = []
        selected_features_scores = []
        all_features_worst = []
        selected_features_worst = []

        for level in sorted_levels:
            level_str = str(level)
            level_data = comparison_data[level_str]['overall_result']

            # All features scores
            all_features_result = level_data['all_features']
            all_features_scores.append(all_features_result.get('mean_score', 0))
            all_features_worst.append(all_features_result.get('worst_score', all_features_result.get('min_score', 0)))

            # Selected features scores
            selected_features_result = level_data['feature_subset']
            selected_features_scores.append(selected_features_result.get('mean_score', 0))
            selected_features_worst.append(selected_features_result.get('worst_score', selected_features_result.get('min_score', 0)))

        # Only return data if we have valid scores
        if not any(all_features_scores) or not any(selected_features_scores):
            logger.warning("No valid scores found for selected features comparison")
            return {}

        logger.info(f"Prepared selected features comparison data for {len(sorted_levels)} perturbation levels using {method_name} method")

        return {
            'perturbation_levels': sorted_levels,
            'all_features_scores': all_features_scores,
            'selected_features_scores': selected_features_scores,
            'selected_features': selected_features,
            'all_features_worst': all_features_worst,
            'selected_features_worst': selected_features_worst,
            'base_score': data.get('base_score', 0),
            'metric': data.get('metric', 'Score'),
            'method_used': method_name
        }

    def _prepare_detailed_distribution_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare detailed distribution data for enhanced boxplot visualization.

        Parameters:
        -----------
        data : Dict[str, Any]
            Enhanced data containing perturbation results with iterations

        Returns:
        --------
        Dict[str, Any] : PiML distribution data or empty dict if not available
        """
        # Try to find raw data first (preferred), then quantile
        source_data = None
        method_name = None

        if 'raw' in data and 'by_level' in data['raw']:
            source_data = data['raw']['by_level']
            method_name = 'raw'
        elif 'quantile' in data and 'by_level' in data['quantile']:
            source_data = data['quantile']['by_level']
            method_name = 'quantile'

        if not source_data:
            logger.warning("No perturbation data available for detailed distribution")
            return {}

        # Extract detailed score distributions for each perturbation level
        perturbation_data = {}

        for level_key, level_data in source_data.items():
            level_float = float(level_key)

            # Collect all individual scores for this level
            level_scores = []

            # Check runs data structure
            if 'runs' in level_data:
                runs_data = level_data['runs']

                # Try all_features first
                if 'all_features' in runs_data:
                    for run in runs_data['all_features']:
                        if 'iterations' in run and 'scores' in run['iterations']:
                            level_scores.extend(run['iterations']['scores'])

                # If no all_features scores, try feature_subset
                if not level_scores and 'feature_subset' in runs_data:
                    for run in runs_data['feature_subset']:
                        if 'iterations' in run and 'scores' in run['iterations']:
                            level_scores.extend(run['iterations']['scores'])

            # If we have scores for this level, add to perturbation_data
            if level_scores:
                perturbation_data[level_float] = level_scores
                logger.debug(f"Collected {len(level_scores)} scores for level {level_float}")

        if not perturbation_data:
            logger.warning("No iteration scores found for detailed distribution")
            return {}

        # Verify we have enough data points for meaningful distribution
        total_scores = sum(len(scores) for scores in perturbation_data.values())
        if total_scores < 10:  # Minimum threshold for meaningful distribution
            logger.warning(f"Insufficient data points ({total_scores}) for detailed distribution")
            return {}

        logger.info(f"Prepared detailed distribution data for {len(perturbation_data)} perturbation levels "
                   f"with {total_scores} total scores using {method_name} method")

        return {
            'perturbation_data': perturbation_data,
            'base_score': data.get('base_score', 0),
            'metric': data.get('metric', 'Score'),
            'method_used': method_name,
            'total_scores': total_scores,
            'n_levels': len(perturbation_data)
        }

    def _prepare_distribution_grid_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare distribution grid data for comprehensive matrix visualization
        across multiple models and perturbation levels.

        Parameters:
        -----------
        data : Dict[str, Any]
            Combined data containing multiple model results

        Returns:
        --------
        Dict[str, Any] : Distribution grid data or empty dict if not available
        """
        models_data = {}
        baseline_scores = {}

        # Process primary model data - check both 'primary_model' and direct data structure
        primary_data = None
        model_name = data.get('model_name', 'Primary Model')

        if 'primary_model' in data:
            primary_data = data['primary_model']
        elif 'raw' in data or 'quantile' in data:
            # Data is directly in the root structure
            primary_data = data

        if primary_data:
            # Extract distribution data for primary model
            model_distributions = self._extract_model_distributions(primary_data)
            if model_distributions:
                models_data[model_name] = model_distributions
                baseline_scores[model_name] = primary_data.get('base_score', 0)

        # Process alternative models if available
        if 'alternative_models' in data:
            for alt_model_name, alt_model_data in data['alternative_models'].items():
                model_distributions = self._extract_model_distributions(alt_model_data)
                if model_distributions:
                    models_data[alt_model_name] = model_distributions
                    baseline_scores[alt_model_name] = alt_model_data.get('base_score', 0)

        # Also check for other model patterns
        for key, value in data.items():
            if key.startswith('alt_model_') or (key == 'alternative_model' and key not in ['alternative_models']):
                if isinstance(value, dict) and 'raw' in value:
                    # Extract model name from key or use a default
                    model_name = key.replace('alt_model_', 'Model ').replace('alternative_model', 'Alternative Model')

                    model_distributions = self._extract_model_distributions(value)
                    if model_distributions:
                        models_data[model_name] = model_distributions
                        baseline_scores[model_name] = value.get('base_score', 0)

        if not models_data:
            logger.warning("No valid model data found for distribution grid")
            return {}

        # Calculate some summary statistics
        total_models = len(models_data)
        all_levels = set()
        total_distributions = 0

        for model_data in models_data.values():
            all_levels.update(model_data.keys())
            total_distributions += len(model_data)

        logger.info(f"Prepared distribution grid data for {total_models} models "
                   f"across {len(all_levels)} perturbation levels "
                   f"({total_distributions} total distributions)")

        return {
            'models_data': models_data,
            'baseline_scores': baseline_scores,
            'n_models': total_models,
            'n_levels': len(all_levels),
            'perturbation_levels': sorted(list(all_levels)),
            'metric': data.get('metric', data.get('primary_model', {}).get('metric', 'Score'))
        }

    def _extract_model_distributions(self, model_data: Dict[str, Any]) -> Dict[float, List[float]]:
        """
        Extract score distributions for a single model across perturbation levels.

        Parameters:
        -----------
        model_data : Dict[str, Any]
            Model data containing perturbation results

        Returns:
        --------
        Dict[float, List[float]] : Perturbation level -> list of scores
        """
        distributions = {}

        # Try to find raw data first (preferred), then quantile
        source_data = None
        if 'raw' in model_data and 'by_level' in model_data['raw']:
            source_data = model_data['raw']['by_level']
        elif 'quantile' in model_data and 'by_level' in model_data['quantile']:
            source_data = model_data['quantile']['by_level']

        if not source_data:
            return distributions

        for level_key, level_data in source_data.items():
            level_float = float(level_key)
            level_scores = []

            # Extract scores from runs data
            if 'runs' in level_data:
                runs_data = level_data['runs']

                # Try all_features first
                if 'all_features' in runs_data:
                    for run in runs_data['all_features']:
                        if 'iterations' in run and 'scores' in run['iterations']:
                            level_scores.extend(run['iterations']['scores'])

                # If no all_features scores, try feature_subset
                if not level_scores and 'feature_subset' in runs_data:
                    for run in runs_data['feature_subset']:
                        if 'iterations' in run and 'scores' in run['iterations']:
                            level_scores.extend(run['iterations']['scores'])

            # Only include if we have meaningful data
            if len(level_scores) >= 2:  # Minimum for distribution
                distributions[level_float] = level_scores

        return distributions