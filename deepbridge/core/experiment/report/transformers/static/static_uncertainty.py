"""
Static uncertainty data transformer for static uncertainty reports.
"""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger("deepbridge.reports")

class StaticUncertaintyTransformer:
    """
    Transforms uncertainty data for static reports.
    """

    def transform(self, data: Dict[str, Any], model_name: str = "Model") -> Dict[str, Any]:
        """
        Transform uncertainty data for static reports.

        Parameters:
        -----------
        data : Dict[str, Any]
            Raw uncertainty test results
        model_name : str, optional
            Name of the model

        Returns:
        --------
        Dict[str, Any] : Transformed data for report
        """
        logger.info("Transforming uncertainty data for static report - Using improved transformer")

        # Add debug logging
        try:
            import json
            logger.info(f"Data keys: {list(data.keys())}")
            if 'primary_model' in data:
                logger.info(f"Primary model keys: {list(data['primary_model'].keys())}")

            # Try to serialize the data to check for any non-serializable objects
            try:
                json.dumps(data, default=str)
                logger.info("Data is serializable")
            except Exception as e:
                logger.warning(f"Data contains non-serializable objects: {str(e)}")
        except Exception as e:
            logger.error(f"Error during debug logging: {str(e)}")

        # Create an output dictionary with only essential info, not defaults
        output = {
            'model_name': model_name,
            'test_type': 'uncertainty',
            'timestamp': data.get('timestamp'),
            # Initialize with empty structures to avoid KeyErrors
            'metrics': {},  # Fix for "metrics não está disponível nos dados"
            'feature_importance': {}
        }

        # Extract model type if available
        if 'model_type' in data:
            output['model_type'] = data['model_type']

        # Extract features if available
        if 'features' in data and isinstance(data['features'], list):
            output['features'] = data['features']

        # Extract metrics if available
        if 'metrics' in data and isinstance(data['metrics'], dict):
            output['metrics'] = data['metrics']

        # Extract alpha_levels - critical for chart generation
        if 'alphas' in data:
            output['alpha_levels'] = data['alphas']
        elif 'alpha_levels' in data:
            output['alpha_levels'] = data['alpha_levels']

        # Extract metrics from top level - required for uncertainty metrics chart
        if 'uncertainty_score' in data:
            output['uncertainty_score'] = data['uncertainty_score']
        elif 'uncertainty_quality_score' in data:
            output['uncertainty_score'] = data['uncertainty_quality_score']

        if 'avg_coverage' in data:
            output['coverage'] = data['avg_coverage']
        elif 'coverage' in data:
            output['coverage'] = data['coverage']

        if 'avg_width' in data:
            output['mean_width'] = data['avg_width']
        elif 'mean_width' in data:
            output['mean_width'] = data['mean_width']
        elif 'avg_normalized_width' in data:
            output['mean_width'] = data['avg_normalized_width']

        # Add calibration size if available
        if 'cal_size' in data:
            output['cal_size'] = data['cal_size']

        # Process primary model data if available
        if 'primary_model' in data and isinstance(data['primary_model'], dict):
            primary_model = data['primary_model']
            logger.info(f"[TRANSFORMER] primary_model keys: {list(primary_model.keys())[:15]}")

            # Extract key metrics from primary_model if available
            if 'uncertainty_quality_score' in primary_model:
                output['uncertainty_score'] = primary_model['uncertainty_quality_score']
            elif 'uncertainty_score' in primary_model:
                output['uncertainty_score'] = primary_model['uncertainty_score']

            # Extract coverage and width if not already found
            if 'avg_coverage' in primary_model and 'coverage' not in output:
                output['coverage'] = primary_model['avg_coverage']
            elif 'coverage' in primary_model and 'coverage' not in output:
                output['coverage'] = primary_model['coverage']

            if 'avg_width' in primary_model and 'mean_width' not in output:
                output['mean_width'] = primary_model['avg_width']
            elif 'mean_width' in primary_model and 'mean_width' not in output:
                output['mean_width'] = primary_model['mean_width']
            elif 'avg_normalized_width' in primary_model and 'mean_width' not in output:
                output['mean_width'] = primary_model['avg_normalized_width']

            # Extract alphas if not already set
            if 'alphas' in primary_model and 'alpha_levels' not in output:
                output['alpha_levels'] = primary_model['alphas']

            # Extract plot data - critical for chart generation
            if 'plot_data' in primary_model:
                plot_data = primary_model['plot_data']

                # Extract alpha comparison data
                if 'alpha_comparison' in plot_data:
                    alpha_data = plot_data['alpha_comparison']
                    
                    # Log what's in alpha_data
                    logger.info(f"alpha_comparison keys: {list(alpha_data.keys()) if isinstance(alpha_data, dict) else 'not a dict'}")
                    
                    # Make sure we always have lists, not numpy arrays
                    alpha_values = alpha_data.get('alphas', [])
                    coverage_values = alpha_data.get('coverages', [])
                    expected_coverages = alpha_data.get('expected_coverages', [])
                    width_values = alpha_data.get('mean_widths', [])
                    
                    # Convert numpy arrays to lists if needed
                    if hasattr(alpha_values, 'tolist') and callable(getattr(alpha_values, 'tolist')):
                        alpha_values = alpha_values.tolist()
                    elif not isinstance(alpha_values, list):
                        try:
                            alpha_values = list(alpha_values)
                        except:
                            alpha_values = []
                            
                    if hasattr(coverage_values, 'tolist') and callable(getattr(coverage_values, 'tolist')):
                        coverage_values = coverage_values.tolist()
                    elif not isinstance(coverage_values, list):
                        try:
                            coverage_values = list(coverage_values)
                        except:
                            coverage_values = []
                            
                    if hasattr(expected_coverages, 'tolist') and callable(getattr(expected_coverages, 'tolist')):
                        expected_coverages = expected_coverages.tolist()
                    elif not isinstance(expected_coverages, list):
                        try:
                            expected_coverages = list(expected_coverages)
                        except:
                            expected_coverages = []
                            
                    if hasattr(width_values, 'tolist') and callable(getattr(width_values, 'tolist')):
                        width_values = width_values.tolist()
                    elif not isinstance(width_values, list):
                        try:
                            width_values = list(width_values)
                        except:
                            width_values = []
                    
                    output['calibration_results'] = {
                        'alpha_values': alpha_values,
                        'coverage_values': coverage_values,
                        'expected_coverages': expected_coverages,
                        'width_values': width_values
                    }
                    
                    # Log transformed data
                    logger.info(f"Transformed calibration_results: {output['calibration_results']}")

                    # If alphas not already set, use from alpha_comparison
                    if 'alpha_levels' not in output and alpha_values:
                        output['alpha_levels'] = alpha_values

                # Extract feature importance with proper format
                if 'feature_importance' in plot_data and isinstance(plot_data['feature_importance'], list):
                    feature_data = plot_data['feature_importance']
                    # Convert list format to dict for easier template access
                    feature_importance = {}
                    for item in feature_data:
                        if 'feature' in item and 'importance' in item:
                            feature_importance[item['feature']] = item['importance']
                    output['feature_importance'] = feature_importance

                    # Also keep original format for charts
                    output['feature_importance_data'] = feature_data

                # Extract width distribution data
                if 'width_distribution' in plot_data:
                    width_dist = plot_data['width_distribution']
                    if isinstance(width_dist, list) and len(width_dist) > 0:
                        all_widths = []
                        for alpha_widths in width_dist:
                            if 'widths' in alpha_widths and hasattr(alpha_widths['widths'], 'tolist'):
                                all_widths.append(alpha_widths['widths'].tolist())
                        if all_widths:
                            output['interval_widths'] = all_widths

                # Extract coverage vs width data
                if 'coverage_vs_width' in plot_data:
                    output['coverage_vs_width'] = plot_data['coverage_vs_width']
                    logger.debug(f"[TRANSFORM_DEBUG] Found coverage_vs_width in plot_data: {plot_data['coverage_vs_width'].keys() if isinstance(plot_data['coverage_vs_width'], dict) else type(plot_data['coverage_vs_width'])}")
                else:
                    logger.warning("[TRANSFORM_DEBUG] coverage_vs_width NOT found in plot_data")

            # Extract feature reliability if available
            if 'feature_reliability' in primary_model:
                output['feature_reliability'] = primary_model['feature_reliability']
                
            # Extract enhanced reliability analysis if available with detailed logging
            if 'reliability_analysis' in primary_model:
                logger.info("[TRANSFORM_DEBUG] Found reliability_analysis in primary_model")
                logger.info(f"[TRANSFORM_DEBUG] reliability_analysis keys: {list(primary_model['reliability_analysis'].keys())}")
                
                # Check if feature_distributions is present
                if 'feature_distributions' in primary_model['reliability_analysis']:
                    ra_fd = primary_model['reliability_analysis']['feature_distributions']
                    logger.info(f"[TRANSFORM_DEBUG] feature_distributions types: {list(ra_fd.keys())}")
                    
                    # Check each distribution type
                    for dist_type, features in ra_fd.items():
                        logger.info(f"[TRANSFORM_DEBUG] '{dist_type}' has {len(features)} features")
                        feature_names = list(features.keys())
                        if feature_names:
                            logger.info(f"[TRANSFORM_DEBUG] '{dist_type}' example features: {feature_names[:3]}")
                            
                            # Check feature value arrays
                            for feature in feature_names[:2]:
                                values = features[feature]
                                if isinstance(values, list):
                                    logger.info(f"[TRANSFORM_DEBUG] '{dist_type}' feature '{feature}' has {len(values)} values")
                                else:
                                    logger.info(f"[TRANSFORM_DEBUG] '{dist_type}' feature '{feature}' has non-list data: {type(values)}")
                
                # Copy the data to output
                output['reliability_analysis'] = primary_model['reliability_analysis'].copy()
                logger.info("[TRANSFORM_DEBUG] Copied reliability_analysis to output")
                
            # Extract marginal bandwidth data if available
            if 'marginal_bandwidth' in primary_model:
                logger.info("[TRANSFORM_DEBUG] Found marginal_bandwidth in primary_model")
                logger.info(f"[TRANSFORM_DEBUG] marginal_bandwidth has {len(primary_model['marginal_bandwidth'])} features")
                
                feature_names = list(primary_model['marginal_bandwidth'].keys())
                if feature_names:
                    logger.info(f"[TRANSFORM_DEBUG] marginal_bandwidth features: {feature_names}")
                    
                    # Check each feature's data
                    for feature in feature_names[:2]:  # Look at first 2 features
                        feature_data = primary_model['marginal_bandwidth'][feature]
                        logger.info(f"[TRANSFORM_DEBUG] Feature '{feature}' data keys: {list(feature_data.keys())}")
                        
                        # Check arrays
                        for key, value in feature_data.items():
                            if isinstance(value, list):
                                logger.info(f"[TRANSFORM_DEBUG] '{feature}' {key} has {len(value)} values")
                            else:
                                logger.info(f"[TRANSFORM_DEBUG] '{feature}' {key} = {value}")
                
                # Copy the data to output
                output['marginal_bandwidth'] = primary_model['marginal_bandwidth'].copy()
                logger.info("[TRANSFORM_DEBUG] Copied marginal_bandwidth to output")
                
            # Extract interval widths for boxplots/violinplots
            if 'interval_widths' in primary_model:
                logger.info("[TRANSFORM_DEBUG] Found interval_widths in primary_model")

                # Log details based on type
                if isinstance(primary_model['interval_widths'], dict):
                    logger.info(f"[TRANSFORM_DEBUG] interval_widths is a dictionary with keys: {list(primary_model['interval_widths'].keys())}")

                    # Check values
                    for model, widths in primary_model['interval_widths'].items():
                        if isinstance(widths, list):
                            logger.info(f"[TRANSFORM_DEBUG] Model '{model}' has {len(widths)} width values")
                        else:
                            logger.info(f"[TRANSFORM_DEBUG] Model '{model}' has non-list data: {type(widths)}")

                elif isinstance(primary_model['interval_widths'], list):
                    logger.info(f"[TRANSFORM_DEBUG] interval_widths is a list with {len(primary_model['interval_widths'])} elements")

                    # Check first element
                    if primary_model['interval_widths']:
                        first_item = primary_model['interval_widths'][0]
                        logger.info(f"[TRANSFORM_DEBUG] First element type: {type(first_item)}")

                        if isinstance(first_item, dict):
                            logger.info(f"[TRANSFORM_DEBUG] First element keys: {list(first_item.keys())}")
                        elif isinstance(first_item, list):
                            logger.info(f"[TRANSFORM_DEBUG] First element is a list with {len(first_item)} values")
                else:
                    logger.info(f"[TRANSFORM_DEBUG] interval_widths has unexpected type: {type(primary_model['interval_widths'])}")

                # Copy the data to output
                output['interval_widths'] = primary_model['interval_widths']
                logger.info("[TRANSFORM_DEBUG] Copied interval_widths to output")

            # Extract test predictions and labels for reliability charts
            if 'test_predictions' in primary_model:
                logger.info(f"[TRANSFORM_DEBUG] Found test_predictions in primary_model with shape: {primary_model['test_predictions'].shape if hasattr(primary_model['test_predictions'], 'shape') else 'N/A'}")
                output['test_predictions'] = primary_model['test_predictions']

            if 'test_labels' in primary_model:
                logger.info(f"[TRANSFORM_DEBUG] Found test_labels in primary_model with shape: {primary_model['test_labels'].shape if hasattr(primary_model['test_labels'], 'shape') else 'N/A'}")
                output['test_labels'] = primary_model['test_labels']

        # If feature_importance not in plot_data, try getting from top level
        if 'feature_importance' not in output and 'feature_importance' in data:
            output['feature_importance'] = data['feature_importance']

        # Check for enhanced data at top level
        if 'reliability_analysis' not in output and 'reliability_analysis' in data:
            output['reliability_analysis'] = data['reliability_analysis']

        if 'marginal_bandwidth' not in output and 'marginal_bandwidth' in data:
            output['marginal_bandwidth'] = data['marginal_bandwidth']

        if 'interval_widths' not in output and 'interval_widths' in data:
            output['interval_widths'] = data['interval_widths']

        # Check for test predictions and labels at top level (from uncertainty suite)
        if 'test_predictions' not in output and 'test_predictions' in data:
            output['test_predictions'] = data['test_predictions']
            logger.info(f"[TRANSFORM_DEBUG] Found test_predictions at top level with shape: {data['test_predictions'].shape if hasattr(data['test_predictions'], 'shape') else 'N/A'}")

        if 'test_labels' not in output and 'test_labels' in data:
            output['test_labels'] = data['test_labels']
            logger.info(f"[TRANSFORM_DEBUG] Found test_labels at top level with shape: {data['test_labels'].shape if hasattr(data['test_labels'], 'shape') else 'N/A'}")
            
        # Additional metrics from top level
        if 'mse' in data:
            output['mse'] = data['mse']
        if 'mae' in data:
            output['mae'] = data['mae']
        if 'predictions' in data:
            output['predictions'] = data['predictions']
        if 'dataset' in data:
            output['dataset'] = data['dataset']

            # Format for charts if needed
            if 'feature_importance_data' not in output:
                feature_importance_data = []
                for feature, importance in data['feature_importance'].items():
                    feature_importance_data.append({'feature': feature, 'importance': importance})
                output['feature_importance_data'] = feature_importance_data

        # Process alternative models data for charts
        if 'alternative_models' in data and isinstance(data['alternative_models'], dict):
            processed_alt_models = {}
            models_data_for_charts = []  # For performance_gap_by_alpha chart

            # Add primary model to models_data
            if 'uncertainty_score' in output and ('coverage' in output or 'mean_width' in output):
                primary_model_data = {
                    'name': model_name,
                    'uncertainty_score': output.get('uncertainty_score', 0),
                    'coverage': output.get('coverage', 0),
                    'mean_width': output.get('mean_width', 0)
                }
                models_data_for_charts.append(primary_model_data)

            for alt_name, model_data in data['alternative_models'].items():
                model_info = {'model_type': model_data.get('model_type', 'Unknown')}

                # Extract key metrics with proper error handling
                # Uncertainty score
                if 'uncertainty_quality_score' in model_data:
                    model_info['uncertainty_score'] = model_data['uncertainty_quality_score']
                elif 'uncertainty_score' in model_data:
                    model_info['uncertainty_score'] = model_data['uncertainty_score']
                else:
                    model_info['uncertainty_score'] = 0

                # Coverage
                if 'avg_coverage' in model_data:
                    model_info['coverage'] = model_data['avg_coverage']
                elif 'coverage' in model_data:
                    model_info['coverage'] = model_data['coverage']
                else:
                    model_info['coverage'] = 0

                # Mean width
                if 'avg_width' in model_data:
                    model_info['mean_width'] = model_data['avg_width']
                elif 'mean_width' in model_data:
                    model_info['mean_width'] = model_data['mean_width']
                elif 'avg_normalized_width' in model_data:
                    model_info['mean_width'] = model_data['avg_normalized_width']
                else:
                    model_info['mean_width'] = 0

                # Extract more data for charts if available
                if 'plot_data' in model_data:
                    if 'alpha_comparison' in model_data['plot_data']:
                        alpha_data = model_data['plot_data']['alpha_comparison']
                        model_info['calibration_results'] = {
                            'alpha_values': alpha_data.get('alphas', []),
                            'coverage_values': alpha_data.get('coverages', []),
                            'expected_coverages': alpha_data.get('expected_coverages', []),
                            'width_values': alpha_data.get('mean_widths', [])
                        }

                # Extract metrics if available
                if 'metrics' in model_data:
                    model_info['metrics'] = model_data['metrics']

                # If feature importance available in model data
                if 'feature_importance' in model_data:
                    model_info['feature_importance'] = model_data['feature_importance']

                processed_alt_models[alt_name] = model_info

                # Add to models_data for charts if it has required metrics
                if all(k in model_info for k in ['uncertainty_score', 'coverage', 'mean_width']):
                    models_data_for_charts.append({
                        'name': alt_name,
                        'uncertainty_score': model_info['uncertainty_score'],
                        'coverage': model_info['coverage'],
                        'mean_width': model_info['mean_width']
                    })

            output['alternative_models'] = processed_alt_models

            # Add models_data for performance_gap_by_alpha chart
            if models_data_for_charts:
                output['models_data'] = models_data_for_charts

        # Process for feature subset display
        feature_subset = data.get('feature_subset', [])
        output['feature_subset'] = feature_subset

        # Create a readable string version for display
        if feature_subset:
            if len(feature_subset) > 5:
                subset_display = f"{', '.join(feature_subset[:5])} + {len(feature_subset) - 5} more"
            else:
                subset_display = ", ".join(feature_subset)
            output['feature_subset_display'] = subset_display
        else:
            output['feature_subset_display'] = "None"

        # Extract interval widths if not already set
        if 'interval_widths' not in output and 'interval_widths' in data:
            if isinstance(data['interval_widths'], list):
                output['interval_widths'] = data['interval_widths']

        # Extract PSI scores
        if 'psi_scores' in data and isinstance(data['psi_scores'], dict):
            output['psi_scores'] = data['psi_scores']

        # Ensure default values for key metrics
        if 'uncertainty_score' not in output:
            output['uncertainty_score'] = 0
        if 'coverage' not in output:
            output['coverage'] = 0
        if 'mean_width' not in output:
            output['mean_width'] = 0

        # Log the results with safe dictionary access
        logger.info(f"Transformed uncertainty data: uncertainty_score={output.get('uncertainty_score', 'N/A')}, coverage={output.get('coverage', 'N/A')}, mean_width={output.get('mean_width', 'N/A')}")
        return output