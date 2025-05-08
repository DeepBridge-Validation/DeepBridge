"""
Data transformation module for robustness reports.
"""

import logging
import datetime
from typing import Dict, Any, Optional

from ..base import DataTransformer

# Configure logger
logger = logging.getLogger("deepbridge.reports")

class RobustnessDataTransformer(DataTransformer):
    """
    Transforms robustness test results data for templates.
    """
    
    def transform(self, results: Dict[str, Any], model_name: str, timestamp: Optional[str] = None) -> Dict[str, Any]:
        """
        Transform robustness results data for template rendering.
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Raw robustness test results
        model_name : str
            Name of the model
        timestamp : str, optional
            Timestamp for the report
            
        Returns:
        --------
        Dict[str, Any] : Transformed data for templates
        
        Raises:
        -------
        ValueError: If required data is missing
        """
        logger.info("Transforming robustness data structure...")
        
        if timestamp is None:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Create a deep copy of the results
        report_data = self._deep_copy(results)
        
        # Handle to_dict() method if available
        if hasattr(report_data, 'to_dict'):
            report_data = report_data.to_dict()
            logger.info("Used to_dict() method to convert results")
        
        # Handle case where results are nested under 'primary_model' key
        if 'primary_model' in report_data:
            logger.info("Found 'primary_model' key, extracting data...")
            primary_data = report_data['primary_model']
            
            # Extract feature importance data
            if 'feature_importance' in primary_data:
                logger.info(f"Found feature_importance in primary_model with {len(primary_data['feature_importance'])} features")
                report_data['feature_importance'] = primary_data['feature_importance']
                
            if 'model_feature_importance' in primary_data:
                logger.info(f"Found model_feature_importance in primary_model with {len(primary_data['model_feature_importance'])} features")
                report_data['model_feature_importance'] = primary_data['model_feature_importance']
            
            # Copy fields from primary_model to the top level
            for key, value in primary_data.items():
                if key not in report_data or key == 'raw' or key == 'quantile':
                    report_data[key] = value
            
            # If raw, quantile exists at the top level, don't overwrite
            if 'raw' not in report_data and 'raw' in primary_data:
                report_data['raw'] = primary_data['raw']
            if 'quantile' not in report_data and 'quantile' in primary_data:
                report_data['quantile'] = primary_data['quantile']
        
        # Add metadata for display
        report_data['model_name'] = report_data.get('model_name', model_name)
        report_data['timestamp'] = report_data.get('timestamp', timestamp)
        
        # Set model_type
        if 'model_type' in report_data:
            # Already has model_type, keep it
            pass
        # Use model_type from primary_model if available
        elif 'primary_model' in report_data and 'model_type' in report_data['primary_model']:
            report_data['model_type'] = report_data['primary_model']['model_type']
        # Use type from initial_results if available
        elif ('initial_results' in report_data and 'models' in report_data['initial_results'] and 
              'primary_model' in report_data['initial_results']['models'] and 
              'type' in report_data['initial_results']['models']['primary_model']):
            report_data['model_type'] = report_data['initial_results']['models']['primary_model']['type']
        else:
            logger.warning("model_type not found in results")
            report_data['model_type'] = "Unknown Model"
        
        # Check if we need to get feature importance data from nested structure
        if 'results' in report_data:
            logger.info("Checking for feature importance in nested results structure")
            if 'robustness' in report_data['results']:
                rob_results = report_data['results']['robustness']
                logger.info(f"Found robustness key with keys: {list(rob_results.keys())}")
                
                # Check in direct robustness object
                if 'feature_importance' in rob_results and rob_results['feature_importance']:
                    logger.info(f"Found feature_importance directly in results.robustness with {len(rob_results['feature_importance'])} features")
                    report_data['feature_importance'] = rob_results['feature_importance']
                
                if 'model_feature_importance' in rob_results and rob_results['model_feature_importance']:
                    logger.info(f"Found model_feature_importance directly in results.robustness with {len(rob_results['model_feature_importance'])} features")
                    report_data['model_feature_importance'] = rob_results['model_feature_importance']
                
                # Check in nested results
                if 'results' in rob_results:
                    nested_results = rob_results['results'] 
                    logger.info(f"Found nested results with keys: {list(nested_results.keys())}")
                    
                    # Check for alternative models in nested structure
                    if 'alternative_models' in nested_results and 'alternative_models' not in report_data:
                        logger.info("Found alternative_models in nested structure")
                        report_data['alternative_models'] = nested_results['alternative_models']
                    
                    if 'primary_model' in nested_results:
                        primary_model = nested_results['primary_model']
                        logger.info("Found primary_model in nested results.robustness.results")
                        
                        if 'feature_importance' in primary_model and primary_model['feature_importance']:
                            logger.info(f"Found feature_importance in nested results with {len(primary_model['feature_importance'])} features")
                            report_data['feature_importance'] = primary_model['feature_importance']
                            
                        if 'model_feature_importance' in primary_model and primary_model['model_feature_importance']:
                            logger.info(f"Found model_feature_importance in nested results with {len(primary_model['model_feature_importance'])} features")
                            report_data['model_feature_importance'] = primary_model['model_feature_importance']
        
        # Ensure we have 'metric' and 'base_score'
        if 'metric' not in report_data:
            logger.warning("metric not found in results")
            report_data['metric'] = 'score'
            
        if 'base_score' not in report_data:
            logger.warning("base_score not found in results")
            report_data['base_score'] = 0.0
        
        # Ensure we have 'robustness_score'
        if 'robustness_score' not in report_data:
            if 'avg_overall_impact' in report_data:
                report_data['robustness_score'] = float(1.0 - report_data['avg_overall_impact'])
                logger.info("Calculated robustness_score from avg_overall_impact")
            else:
                logger.warning("robustness_score not found in results and cannot be calculated")
                report_data['robustness_score'] = 0.0
        
        # Set impact values for display
        if 'avg_raw_impact' not in report_data:
            if 'raw' in report_data and 'overall' in report_data['raw']:
                report_data['avg_raw_impact'] = report_data['raw'].get('overall', {}).get('avg_impact', 0.0)
            else:
                logger.warning("avg_raw_impact not found in results")
                report_data['avg_raw_impact'] = 0.0
        
        if 'avg_quantile_impact' not in report_data:
            if 'quantile' in report_data and 'overall' in report_data['quantile']:
                report_data['avg_quantile_impact'] = report_data['quantile'].get('overall', {}).get('avg_impact', 0.0)
            else:
                logger.warning("avg_quantile_impact not found in results")
                report_data['avg_quantile_impact'] = 0.0
        
        # Set display-friendly alias properties
        report_data['raw_impact'] = report_data['avg_raw_impact']
        report_data['quantile_impact'] = report_data['avg_quantile_impact']

        # Preparar dados para gráficos de perturbação
        if 'raw' in report_data and 'by_level' in report_data['raw']:
            levels = sorted([float(level) for level in report_data['raw']['by_level'].keys()])
            scores = []
            worst_scores = []
            
            for level in levels:
                level_str = str(level)
                level_data = report_data['raw']['by_level'].get(level_str, {})
                
                # Buscar score principal
                score = None
                worst_score = None
                
                # Buscar em overall_result se disponível
                if 'overall_result' in level_data:
                    overall_result = level_data['overall_result']
                    if 'all_features' in overall_result:
                        all_features_result = overall_result['all_features']
                        score = all_features_result.get('mean_score')  # CORREÇÃO: Usar mean_score ao invés de perturbed_score
                        worst_score = all_features_result.get('worst_score')
                    else:
                        score = overall_result.get('mean_score')  # CORREÇÃO: Usar mean_score ao invés de perturbed_score
                        worst_score = overall_result.get('worst_score')
                
                # Se não encontrou, buscar diretamente nas corridas
                if score is None and 'runs' in level_data:
                    if 'all_features' in level_data['runs']:
                        all_features_runs = level_data['runs']['all_features']
                        if all_features_runs and len(all_features_runs) > 0:
                            first_run = all_features_runs[0]
                            score = first_run.get('perturbed_score')
                            worst_score = first_run.get('worst_score')
                
                # Se ainda não encontrou, buscar diretamente nos atributos de nível 
                if score is None:
                    for key, value in level_data.items():
                        if isinstance(value, (int, float)) and 'score' in key.lower():
                            score = value
                            break
                
                scores.append(score if score is not None else 0.0)
                worst_scores.append(worst_score if worst_score is not None else 0.0)
            
            # Garantir que temos dados de perturbação
            report_data['perturbation_chart_data'] = {
                'modelName': report_data.get('model_name', 'Primary Model'),
                'levels': levels,
                'scores': scores,
                'worstScores': worst_scores,
                'baseScore': report_data.get('base_score', 0.0),
                'metric': report_data.get('metric', 'Score')
            }
            
            # Processar modelos alternativos
            if 'alternative_models' in report_data:
                alt_models_data = {}
                for alt_name, alt_data in report_data['alternative_models'].items():
                    if 'raw' in alt_data and 'by_level' in alt_data['raw']:
                        alt_scores = []
                        alt_worst_scores = []
                        
                        for level in levels:
                            level_str = str(level)
                            level_data = alt_data['raw']['by_level'].get(level_str, {})
                            
                            # Extrair scores
                            alt_score = None
                            alt_worst = None
                            
                            # Buscar em overall_result se disponível
                            if 'overall_result' in level_data:
                                overall_result = level_data['overall_result']
                                if 'all_features' in overall_result:
                                    all_features_result = overall_result['all_features']
                                    alt_score = all_features_result.get('mean_score')  # CORREÇÃO: Usar mean_score ao invés de perturbed_score
                                    alt_worst = all_features_result.get('worst_score')
                                else:
                                    alt_score = overall_result.get('mean_score')  # CORREÇÃO: Usar mean_score ao invés de perturbed_score
                                    alt_worst = overall_result.get('worst_score')
                            
                            # Se não encontrou, buscar nas corridas
                            if alt_score is None and 'runs' in level_data:
                                if 'all_features' in level_data['runs']:
                                    all_features_runs = level_data['runs']['all_features']
                                    if all_features_runs and len(all_features_runs) > 0:
                                        first_run = all_features_runs[0]
                                        alt_score = first_run.get('perturbed_score')
                                        alt_worst = first_run.get('worst_score')
                            
                            alt_scores.append(alt_score if alt_score is not None else 0.0)
                            alt_worst_scores.append(alt_worst if alt_worst is not None else 0.0)
                        
                        alt_models_data[alt_name] = {
                            'baseScore': alt_data.get('base_score', 0.0),
                            'scores': alt_scores,
                            'worstScores': alt_worst_scores
                        }
                
                report_data['perturbation_chart_data']['alternativeModels'] = alt_models_data
        
        # Feature subset formatting
        if 'feature_subset' in report_data and report_data['feature_subset']:
            if isinstance(report_data['feature_subset'], list):
                # Already a list, keep as is
                pass
            elif isinstance(report_data['feature_subset'], str):
                # Convert string to list
                report_data['feature_subset'] = [report_data['feature_subset']]
            else:
                logger.warning("feature_subset has unexpected type")
                report_data['feature_subset'] = []
        else:
            report_data['feature_subset'] = []
        
        # Convert feature subset to display string
        if report_data['feature_subset']:
            report_data['feature_subset_display'] = ", ".join(report_data['feature_subset'])
        else:
            report_data['feature_subset_display'] = "All Features"
        
        # Process alternative models if present
        if 'alternative_models' in report_data:
            logger.info("Processing alternative models data...")
            
            # Initialize alternative models dict if needed
            if not isinstance(report_data['alternative_models'], dict):
                logger.warning("alternative_models is not a dictionary")
                report_data['alternative_models'] = {}
            
            # Process each alternative model
            for alt_model_name, model_data in report_data['alternative_models'].items():
                logger.info(f"Processing alternative model: {alt_model_name}")
                
                # Set defaults for missing fields
                if 'base_score' not in model_data:
                    logger.warning(f"base_score not found in alternative model {alt_model_name}")
                    model_data['base_score'] = 0.0
                
                # Process raw and quantile impact
                if 'raw' in model_data and isinstance(model_data['raw'], dict):
                    if 'avg_raw_impact' not in model_data and 'overall' in model_data['raw']:
                        model_data['avg_raw_impact'] = model_data['raw'].get('overall', {}).get('avg_impact', 0.0)
                else:
                    logger.warning(f"raw data not found in alternative model {alt_model_name}")
                    model_data['raw'] = {}
                    model_data['avg_raw_impact'] = 0.0
                
                if 'quantile' in model_data and isinstance(model_data['quantile'], dict):
                    if 'avg_quantile_impact' not in model_data and 'overall' in model_data['quantile']:
                        model_data['avg_quantile_impact'] = model_data['quantile'].get('overall', {}).get('avg_impact', 0.0)
                else:
                    logger.warning(f"quantile data not found in alternative model {alt_model_name}")
                    model_data['quantile'] = {}
                    model_data['avg_quantile_impact'] = 0.0
                
                # Calculate robustness score if missing
                if 'robustness_score' not in model_data:
                    if 'avg_overall_impact' in model_data:
                        model_data['robustness_score'] = float(1.0 - model_data.get('avg_overall_impact', 0.0))
                    else:
                        logger.warning(f"robustness_score not found in alternative model {alt_model_name}")
                        model_data['robustness_score'] = 0.0
                
                # Update the model data in the report
                report_data['alternative_models'][alt_model_name] = model_data
        
        # Convert all numpy types to Python native types
        return self.convert_numpy_types(report_data)