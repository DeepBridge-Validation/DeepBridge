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
            'timestamp': data.get('timestamp')
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
                    output['calibration_results'] = {
                        'alpha_values': alpha_data.get('alphas', []),
                        'coverage_values': alpha_data.get('coverages', []),
                        'expected_coverages': alpha_data.get('expected_coverages', []),
                        'width_values': alpha_data.get('mean_widths', [])
                    }

                    # If alphas not already set, use from alpha_comparison
                    if 'alpha_levels' not in output and 'alphas' in alpha_data:
                        output['alpha_levels'] = alpha_data['alphas']

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

            # Extract feature reliability if available
            if 'feature_reliability' in primary_model:
                output['feature_reliability'] = primary_model['feature_reliability']

        # If feature_importance not in plot_data, try getting from top level
        if 'feature_importance' not in output and 'feature_importance' in data:
            output['feature_importance'] = data['feature_importance']

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