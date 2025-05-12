"""
Static resilience data transformer for static resilience reports.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import datetime
import traceback
import numpy as np
import base64
import io
import math
from collections import defaultdict

# Try to import visualization libraries
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from scipy import stats
    HAS_VISUALIZATION_LIBS = True
except ImportError as e:
    HAS_VISUALIZATION_LIBS = False

# Import our modular chart system
try:
    # Attempt to import using relative path from project root
    from deepbridge.templates.report_types.resilience.static.charts import ResilienceChartGenerator
    HAS_CHART_GENERATOR = True
    logging.getLogger("deepbridge.reports").info(f"Successfully imported ResilienceChartGenerator")
except ImportError:
    try:
        # Alternative import path - using absolute path from system
        import sys
        import os
        # Get the project root directory
        project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))))
        charts_path = os.path.join(project_root, "templates", "report_types", "resilience", "static", "charts")
        if charts_path not in sys.path:
            sys.path.append(charts_path)
        # Try import again
        from __init__ import ResilienceChartGenerator
        HAS_CHART_GENERATOR = True
        logging.getLogger("deepbridge.reports").info(f"Successfully imported ResilienceChartGenerator using path: {charts_path}")
    except ImportError as e:
        HAS_CHART_GENERATOR = False
        logging.getLogger("deepbridge.reports").error(f"Error importing ResilienceChartGenerator: {str(e)}")
        logging.getLogger("deepbridge.reports").error(f"Charts path attempted: templates/report_types/resilience/static/charts")

logger = logging.getLogger("deepbridge.reports")

class StaticResilienceTransformer:
    """
    Transforms resilience data for static reports.
    """

    def __init__(self):
        """
        Initialize the transformer.
        """
        # Import the standard resilience transformer
        from ..resilience import ResilienceDataTransformer
        self.base_transformer = ResilienceDataTransformer()

        # Set up visualization defaults
        if HAS_VISUALIZATION_LIBS:
            # Set default style for all charts
            sns.set_theme(style="whitegrid")

            # Use a palette that works well for most charts
            sns.set_palette("deep")

            # Improve font scaling for better readability
            plt.rcParams.update({
                'font.size': 12,
                'axes.labelsize': 14,
                'axes.titlesize': 16,
                'xtick.labelsize': 12,
                'ytick.labelsize': 12,
                'legend.fontsize': 12,
                'figure.titlesize': 18
            })
            
        # Initialize chart generator
        if HAS_CHART_GENERATOR:
            self.chart_generator = ResilienceChartGenerator()
        else:
            self.chart_generator = None

    def transform(self, data: Dict[str, Any], model_name: str = "Model") -> Dict[str, Any]:
        """
        Transform resilience data for static reports.

        Parameters:
        -----------
        data : Dict[str, Any]
            Raw resilience test results
        model_name : str, optional
            Name of the model

        Returns:
        --------
        Dict[str, Any] : Transformed data for report
        """
        logger.info("Transforming resilience data for static report")
        
        # First, use the base transformer to get standard transformations
        try:
            transformed_data = self.base_transformer.transform(data, model_name)
            logger.info(f"Base transformer produced data with keys: {list(transformed_data.keys() if isinstance(transformed_data, dict) else [])}")
        except Exception as e:
            logger.error(f"Error during base transformation: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Create a basic container if the base transformer fails
            transformed_data = {
                'model_name': model_name,
                'test_type': 'resilience',
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        
        # Create an output dictionary with default values
        output = {
            'model_name': model_name,
            'test_type': 'resilience',
            'timestamp': transformed_data.get('timestamp', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
            'model_type': transformed_data.get('model_type', data.get('model_type', 'Classification')),
            'features': transformed_data.get('features', data.get('features', []))
        }
        
        # Extract resilience score from different possible sources
        if 'resilience_score' in transformed_data:
            output['resilience_score'] = transformed_data['resilience_score']
        elif 'resilience_score' in data:
            output['resilience_score'] = data['resilience_score']
        
        # Performance gap can be stored under different names
        if 'avg_performance_gap' in transformed_data:
            output['avg_performance_gap'] = transformed_data['avg_performance_gap']
        elif 'performance_gap' in transformed_data:
            output['avg_performance_gap'] = transformed_data['performance_gap']
        elif 'avg_performance_gap' in data:
            output['avg_performance_gap'] = data['avg_performance_gap']
        elif 'performance_gap' in data:
            output['avg_performance_gap'] = data['performance_gap']
        
        # Extract metric name if available
        if 'metric' in transformed_data:
            output['metric'] = transformed_data['metric']
        elif 'metric' in data:
            output['metric'] = data['metric']
        else:
            output['metric'] = 'Score'  # Default value if not found
        
        # Extract feature importance if available
        if 'feature_importance' in transformed_data and isinstance(transformed_data['feature_importance'], dict):
            output['feature_importance'] = transformed_data['feature_importance']
        elif 'feature_importance' in data and isinstance(data['feature_importance'], dict):
            output['feature_importance'] = data['feature_importance']
        
        # Extract feature distances for distribution shift analysis
        if 'feature_distances' in transformed_data and isinstance(transformed_data['feature_distances'], dict):
            output['feature_distances'] = transformed_data['feature_distances']
        elif 'feature_distances' in data and isinstance(data['feature_distances'], dict):
            output['feature_distances'] = data['feature_distances']
        
        # Extract distribution shift metrics
        dist_shift = {}
        for source in [transformed_data, data]:
            if 'distribution_shift' in source and isinstance(source['distribution_shift'], dict):
                dist_shift = source['distribution_shift']
                break
                
        if dist_shift:
            output['distribution_shift'] = dist_shift
            
            # Extract average distance metrics from distribution shift data
            if 'avg_dist_shift' in dist_shift:
                output['avg_dist_shift'] = dist_shift['avg_dist_shift']
            elif 'avg_distance' in dist_shift:
                output['avg_dist_shift'] = dist_shift['avg_distance']
        
        # Extract performance metrics for worst vs remaining samples
        performance_metrics = {}
        for source in [transformed_data, data]:
            if 'performance_metrics' in source and isinstance(source['performance_metrics'], dict):
                performance_metrics = source['performance_metrics']
                break
        
        if performance_metrics:
            output['performance_metrics'] = performance_metrics
        
        # Extract worst and remaining samples for feature distribution analysis
        for sample_type in ['worst_samples', 'remaining_samples']:
            if sample_type in transformed_data and isinstance(transformed_data[sample_type], dict):
                output[sample_type] = transformed_data[sample_type]
            elif sample_type in data and isinstance(data[sample_type], dict):
                output[sample_type] = data[sample_type]
        
        # Extract residuals for error analysis
        for residual_type in ['residuals', 'worst_residuals', 'remaining_residuals']:
            if residual_type in transformed_data:
                output[residual_type] = transformed_data[residual_type]
            elif residual_type in data:
                output[residual_type] = data[residual_type]
        
        # Extract feature correlations
        if 'feature_correlations' in transformed_data and isinstance(transformed_data['feature_correlations'], dict):
            output['feature_correlations'] = transformed_data['feature_correlations']
        elif 'feature_correlations' in data and isinstance(data['feature_correlations'], dict):
            output['feature_correlations'] = data['feature_correlations']
        
        # Extract alpha-level results for model comparison
        alpha_results = {}
        for key in ['by_alpha', 'alpha_results', 'perturbation_levels']:
            if key in transformed_data and isinstance(transformed_data[key], dict):
                alpha_results = transformed_data[key]
                output[key] = alpha_results
                break
            elif key in data and isinstance(data[key], dict):
                alpha_results = data[key]
                output[key] = alpha_results
                break
        
        # Extract alternative models for comparison
        if 'alternative_models' in transformed_data and isinstance(transformed_data['alternative_models'], dict):
            output['alternative_models'] = transformed_data['alternative_models']
        elif 'alternative_models' in data and isinstance(data['alternative_models'], dict):
            output['alternative_models'] = data['alternative_models']
        
        # Extract model feature importance if available (for comparison)
        if 'model_feature_importance' in transformed_data and isinstance(transformed_data['model_feature_importance'], dict):
            output['model_feature_importance'] = transformed_data['model_feature_importance']
        elif 'model_feature_importance' in data and isinstance(data['model_feature_importance'], dict):
            output['model_feature_importance'] = data['model_feature_importance']
        
        # Extract sensitive features if available
        if 'sensitive_features' in transformed_data:
            output['sensitive_features'] = transformed_data['sensitive_features']
        elif 'sensitive_features' in data:
            output['sensitive_features'] = data['sensitive_features']
        
        # Extract most affected scenario if available
        if 'most_affected_scenario' in transformed_data:
            output['most_affected_scenario'] = transformed_data['most_affected_scenario']
        elif 'most_affected_scenario' in data:
            output['most_affected_scenario'] = data['most_affected_scenario']

        # Include raw results for additional chart generation
        if 'raw_results' in transformed_data:
            output['raw_results'] = transformed_data['raw_results']
        elif 'raw_results' in data:
            output['raw_results'] = data['raw_results']

        # Try to extract features from feature distances or feature importance
        if not output.get('features') and 'feature_distances' in output and isinstance(output['feature_distances'], dict):
            output['features'] = list(output['feature_distances'].keys())
        elif not output.get('features') and 'feature_importance' in output and isinstance(output['feature_importance'], dict):
            output['features'] = list(output['feature_importance'].keys())
        elif not output.get('features') and 'distribution_shift' in output and isinstance(output['distribution_shift'], dict):
            dist_shift = output['distribution_shift']
            if 'by_distance_metric' in dist_shift and isinstance(dist_shift['by_distance_metric'], dict):
                for metric_name, metric_data in dist_shift['by_distance_metric'].items():
                    if isinstance(metric_data, dict):
                        if 'avg_feature_distances' in metric_data and isinstance(metric_data['avg_feature_distances'], dict):
                            output['features'] = list(metric_data['avg_feature_distances'].keys())
                            break
                        elif 'top_features' in metric_data and isinstance(metric_data['top_features'], dict):
                            output['features'] = list(metric_data['top_features'].keys())
                            break
        
        logger.info(f"Transformed resilience data for static report: resilience_score={output.get('resilience_score', 'N/A')}, avg_performance_gap={output.get('avg_performance_gap', 'N/A')}")

        # Generate and add visualizations to the output data
        if HAS_VISUALIZATION_LIBS and HAS_CHART_GENERATOR and self.chart_generator:
            logger.info(f"Starting chart generation with modular chart generator")
            try:
                logger.info(f"Input data for charts has keys: {list(output.keys())}")

                # Log important data fields for debugging
                if 'resilience_score' in output:
                    logger.info(f"Resilience score: {output['resilience_score']}")
                if 'avg_performance_gap' in output:
                    logger.info(f"Average performance gap: {output['avg_performance_gap']}")
                if 'feature_importance' in output:
                    logger.info(f"Feature importance available with {len(output['feature_importance'])} features")
                if 'feature_distances' in output:
                    logger.info(f"Feature distances available with {len(output['feature_distances'])} features")
                if 'performance_metrics' in output:
                    logger.info(f"Performance metrics available with keys: {list(output['performance_metrics'].keys())}")

                output['charts'] = self._generate_charts(output)
                logger.info(f"Generated {len(output['charts'])} charts for resilience visualization")

                # Log which charts were successfully generated
                for chart_name in output['charts']:
                    chart_data = output['charts'][chart_name]
                    chart_size = len(chart_data) if chart_data else 0
                    logger.info(f"Chart '{chart_name}' generated with data size: {chart_size} bytes")
            except Exception as e:
                logger.error(f"Error generating charts: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                output['charts'] = {}
        else:
            if not HAS_VISUALIZATION_LIBS:
                logger.error("Cannot generate charts: visualization libraries not available")
                logger.error("Required libraries: matplotlib, seaborn, pandas")
            if not HAS_CHART_GENERATOR:
                logger.error("Cannot generate charts: chart generator not available")
                logger.error("Required module: templates.report_types.resilience.static.charts.ResilienceChartGenerator")
            output['charts'] = {}

        return output

    def _generate_charts(self, report_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate all charts needed for the resilience report using the modular chart system.

        Parameters:
        -----------
        report_data : Dict[str, Any]
            Transformed report data

        Returns:
        --------
        Dict[str, str] : Dictionary of chart names and their base64 encoded images
        """
        charts = {}

        # Log all available data keys for debugging
        logger.info(f"Running _generate_charts in transformer with data keys: {list(report_data.keys())}")

        # Check if we have a valid chart generator
        if not HAS_CHART_GENERATOR or not self.chart_generator:
            logger.error("No chart generator available for visualization")
            return charts

        # 1. Generate resilience score chart
        logger.info("Generating resilience score chart")
        try:
            resilience_score = report_data.get('resilience_score')
            performance_gap = report_data.get('avg_performance_gap')

            if resilience_score is not None or performance_gap is not None:
                # Convert to chart input format
                metrics_data = {}
                if resilience_score is not None:
                    try:
                        metrics_data['Resilience Score'] = float(resilience_score)
                    except (ValueError, TypeError):
                        pass
                        
                if performance_gap is not None:
                    try:
                        metrics_data['Performance Gap'] = float(performance_gap)
                    except (ValueError, TypeError):
                        pass
                
                if metrics_data:
                    charts['resilience_score_chart'] = self.chart_generator.generate_model_resilience_scores(
                        models_data=metrics_data,
                        title="Resilience Metrics Overview",
                        sort_by="score",
                        ascending=False
                    )
                    logger.info("Generated resilience score chart")
        except Exception as e:
            logger.error(f"Error generating resilience score chart: {str(e)}")

        # 2. Generate feature distribution shift chart
        logger.info("Generating feature distribution shift chart")
        try:
            feature_distances = report_data.get('feature_distances')
            logger.info(f"Feature distances data present: {feature_distances is not None}")

            # Look for feature distances in distribution_shift if not directly available
            if not feature_distances:
                distribution_shift = report_data.get('distribution_shift')
                if distribution_shift and isinstance(distribution_shift, dict):
                    logger.info(f"Distribution shift data keys: {list(distribution_shift.keys())}")

                    # Check by_distance_metric
                    if 'by_distance_metric' in distribution_shift and isinstance(distribution_shift['by_distance_metric'], dict):
                        by_distance_metric = distribution_shift['by_distance_metric']
                        logger.info(f"Found by_distance_metric with keys: {list(by_distance_metric.keys())}")

                        # Extract feature distances from any distance metric available
                        for metric_name, metric_data in by_distance_metric.items():
                            if isinstance(metric_data, dict) and 'avg_feature_distances' in metric_data:
                                feature_distances = metric_data['avg_feature_distances']
                                logger.info(f"Extracted avg_feature_distances from {metric_name}: {feature_distances is not None}")
                                if feature_distances:
                                    break

                            # Try to find 'top_features' if avg_feature_distances is not available
                            if not feature_distances and isinstance(metric_data, dict) and 'top_features' in metric_data:
                                feature_distances = metric_data['top_features']
                                logger.info(f"Extracted top_features from {metric_name}: {feature_distances is not None}")
                                if feature_distances:
                                    break

                            # Try to examine individual results if still not found
                            if not feature_distances and isinstance(metric_data, dict) and 'results' in metric_data:
                                results = metric_data.get('results', [])
                                if results and isinstance(results, list) and len(results) > 0:
                                    # Get the first result
                                    first_result = results[0]
                                    if isinstance(first_result, dict) and 'feature_distances' in first_result:
                                        fd = first_result['feature_distances']
                                        if isinstance(fd, dict):
                                            if 'all_feature_distances' in fd:
                                                feature_distances = fd['all_feature_distances']
                                            elif 'top_features' in fd:
                                                feature_distances = fd['top_features']

                                            if feature_distances:
                                                logger.info(f"Extracted feature distances from {metric_name} results: {feature_distances is not None}")
                                                break

                    # Try the original method if the specific method fails
                    if not feature_distances:
                        # Try to extract feature distances from distribution_shift
                        if 'feature_distances' in distribution_shift:
                            feature_distances = distribution_shift['feature_distances']
                            logger.info(f"Found feature_distances in distribution_shift: {feature_distances is not None}")
                        # Try known field names
                        elif 'features' in distribution_shift:
                            feature_distances = distribution_shift['features']
                            logger.info("Using 'features' field from distribution_shift")
                        elif 'distances' in distribution_shift:
                            feature_distances = distribution_shift['distances']
                            logger.info("Using 'distances' field from distribution_shift")
                        # Try by pattern
                        else:
                            for key in distribution_shift.keys():
                                if ('feature' in key.lower() or 'distance' in key.lower()) and isinstance(distribution_shift[key], dict):
                                    feature_distances = distribution_shift[key]
                                    logger.info(f"Using '{key}' from distribution_shift")
                                    break

            # Look in other likely places
            if not feature_distances:
                for field in ['distance_metrics', 'metrics', 'distribution_shift_results']:
                    if field in report_data and isinstance(report_data[field], dict):
                        logger.info(f"Checking '{field}' for feature distances")
                        potential_source = report_data[field]
                        # Look for dictionary fields that could contain feature distances
                        for key, value in potential_source.items():
                            if isinstance(value, dict) and len(value) > 0:
                                # Check if it looks like feature distances (keys are feature names)
                                first_key = next(iter(value))
                                if not first_key.startswith('_') and not first_key.startswith('__'):
                                    feature_distances = value
                                    logger.info(f"Using '{field}.{key}' as feature distances")
                                    break

            if feature_distances and isinstance(feature_distances, dict):
                logger.info(f"Feature distances data has {len(feature_distances)} features")

                # Show a sample of the data
                sample_items = list(feature_distances.items())[:3]
                logger.info(f"Feature distances sample: {sample_items}")

                try:
                    logger.info("Calling generate_feature_distribution_shift")
                    chart_data = self.chart_generator.generate_feature_distribution_shift(feature_distances)
                    if chart_data:
                        charts['feature_distribution_shift'] = chart_data
                        logger.info(f"Generated feature distribution shift chart: {len(chart_data)} bytes")
                    else:
                        logger.error("generate_feature_distribution_shift returned empty result")
                except Exception as inner_e:
                    logger.error(f"Error in generate_feature_distribution_shift: {str(inner_e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
            else:
                logger.warning("No valid feature distances data found for chart")
        except Exception as e:
            logger.error(f"Error generating feature distribution shift chart: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")

        # 3. Generate performance gap chart
        logger.info("Generating performance gap chart")
        try:
            performance_metrics = report_data.get('performance_metrics')
            logger.info(f"Performance metrics present: {performance_metrics is not None}")

            # If performance_metrics not found, look for them in other fields
            if not performance_metrics:
                # Check data in distribution_shift.by_distance_metric structure
                if 'distribution_shift' in report_data and isinstance(report_data['distribution_shift'], dict):
                    dist_shift = report_data['distribution_shift']

                    if 'by_distance_metric' in dist_shift and isinstance(dist_shift['by_distance_metric'], dict):
                        logger.info("Checking distribution_shift.by_distance_metric for performance metrics")

                        # Extract relevant data from the first distance metric
                        for metric_name, metric_data in dist_shift['by_distance_metric'].items():
                            if isinstance(metric_data, dict) and 'results' in metric_data:
                                results = metric_data.get('results', [])
                                # Get information from the most recent result
                                if results and isinstance(results, list) and len(results) > 0:
                                    last_result = results[-1]  # Last result (highest alpha)

                                    # Build performance metrics from results
                                    if isinstance(last_result, dict):
                                        constructed_metrics = {}

                                        # Add relevant metrics
                                        if 'worst_metric' in last_result:
                                            constructed_metrics['worst_accuracy'] = last_result['worst_metric']

                                        if 'remaining_metric' in last_result:
                                            constructed_metrics['remaining_accuracy'] = last_result['remaining_metric']

                                        # Add other available metrics
                                        for key, value in last_result.items():
                                            if key.startswith('worst_') or key.startswith('remaining_'):
                                                constructed_metrics[key] = value

                                        # See if we have enough metrics
                                        worst_keys = [k for k in constructed_metrics.keys() if k.startswith('worst_')]
                                        remaining_keys = [k for k in constructed_metrics.keys() if k.startswith('remaining_')]

                                        if worst_keys and remaining_keys:
                                            performance_metrics = constructed_metrics
                                            logger.info(f"Constructed performance metrics from {metric_name} results with {len(performance_metrics)} keys")
                                            break

                # If still not found, try the original method
                if not performance_metrics:
                    # Look in likely places for performance metrics
                    for field in ['metrics', 'results', 'evaluation', 'primary_model']:
                        if field in report_data and isinstance(report_data[field], dict):
                            logger.info(f"Checking '{field}' for performance metrics")

                            # Check if this field has worst_ prefix keys that might indicate performance metrics
                            metrics_data = report_data[field]
                            worst_keys = [k for k in metrics_data.keys() if k.startswith('worst_')]
                            remaining_keys = [k for k in metrics_data.keys() if k.startswith('remaining_')]

                            if worst_keys or remaining_keys:
                                logger.info(f"Found potential performance metrics in '{field}'")
                                performance_metrics = metrics_data
                                break

                            # Look one level deeper
                            for subfield, value in metrics_data.items():
                                if isinstance(value, dict):
                                    worst_keys = [k for k in value.keys() if k.startswith('worst_')]
                                    remaining_keys = [k for k in value.keys() if k.startswith('remaining_')]

                                    if worst_keys or remaining_keys:
                                        logger.info(f"Found potential performance metrics in '{field}.{subfield}'")
                                        performance_metrics = value
                                        break

            if performance_metrics and isinstance(performance_metrics, dict):
                logger.info(f"Performance metrics keys: {list(performance_metrics.keys())}")

                # Check for paired metrics (worst_ and remaining_)
                paired_metrics = []
                for key in performance_metrics.keys():
                    if key.startswith('worst_'):
                        base_metric = key[6:]  # Remove 'worst_' prefix
                        remaining_key = f'remaining_{base_metric}'
                        if remaining_key in performance_metrics:
                            paired_metrics.append((key, remaining_key, base_metric))

                logger.info(f"Found {len(paired_metrics)} paired metrics: {[m[2] for m in paired_metrics]}")

                if paired_metrics:
                    # Determine task type
                    task_type = "classification"
                    if any(key in performance_metrics for key in ['worst_mse', 'worst_mae', 'worst_r2']):
                        task_type = "regression"
                    logger.info(f"Detected task type: {task_type}")

                    try:
                        logger.info("Calling generate_performance_gap")
                        chart_data = self.chart_generator.generate_performance_gap(performance_metrics, task_type=task_type)
                        if chart_data:
                            charts['performance_gap_chart'] = chart_data
                            logger.info(f"Generated performance gap chart: {len(chart_data)} bytes")
                        else:
                            logger.error("generate_performance_gap returned empty result")
                    except Exception as inner_e:
                        logger.error(f"Error in generate_performance_gap: {str(inner_e)}")
                        logger.error(f"Traceback: {traceback.format_exc()}")
                else:
                    logger.warning("No paired metrics (worst_X and remaining_X) found")
            else:
                logger.warning("No valid performance metrics found")
        except Exception as e:
            logger.error(f"Error generating performance gap chart: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")

        # 4. Generate critical feature distributions chart
        logger.info("Generating critical feature distributions chart")
        try:
            feature_importance = report_data.get('feature_importance')
            worst_samples = report_data.get('worst_samples')
            remaining_samples = report_data.get('remaining_samples')

            if feature_importance and (worst_samples or remaining_samples):
                feature_list = list(feature_importance.keys())[:5]  # Get up to 5 top features

                charts['critical_feature_distributions'] = self.chart_generator.generate_critical_feature_distributions(
                    worst_samples=worst_samples,
                    remaining_samples=remaining_samples,
                    top_features=feature_list
                )
                logger.info("Generated critical feature distributions chart")
        except Exception as e:
            logger.error(f"Error generating critical feature distributions chart: {str(e)}")

        # 5. Generate feature-residual correlation chart
        logger.info("Generating feature-residual correlation chart")
        try:
            feature_correlations = report_data.get('feature_correlations')

            if feature_correlations and isinstance(feature_correlations, dict):
                charts['feature_residual_correlation'] = self.chart_generator.generate_feature_residual_correlation(feature_correlations)
                logger.info("Generated feature-residual correlation chart")
        except Exception as e:
            logger.error(f"Error generating feature-residual correlation chart: {str(e)}")

        # 6. Generate residual distribution chart
        logger.info("Generating residual distribution chart")
        try:
            residuals = report_data.get('residuals')
            worst_residuals = report_data.get('worst_residuals')
            remaining_residuals = report_data.get('remaining_residuals')

            if residuals or worst_residuals or remaining_residuals:
                charts['residual_distribution'] = self.chart_generator.generate_residual_distribution(
                    worst_residuals=worst_residuals,
                    remaining_residuals=remaining_residuals,
                    all_residuals=residuals
                )
                logger.info("Generated residual distribution chart")
        except Exception as e:
            logger.error(f"Error generating residual distribution chart: {str(e)}")

        # 7. Generate feature importance chart
        logger.info("Generating feature importance chart")
        try:
            feature_importance = report_data.get('feature_importance')
            logger.info(f"Feature importance present: {feature_importance is not None}")

            # Look for feature importance in other places if not directly available
            if not feature_importance:
                # Check in distribution_shift.by_distance_metric structure
                if 'distribution_shift' in report_data and isinstance(report_data['distribution_shift'], dict):
                    dist_shift = report_data['distribution_shift']

                    if 'by_distance_metric' in dist_shift and isinstance(dist_shift['by_distance_metric'], dict):
                        logger.info("Checking distribution_shift.by_distance_metric for feature importance")

                        # Check avg_feature_distances or top_features in any metric
                        for metric_name, metric_data in dist_shift['by_distance_metric'].items():
                            if isinstance(metric_data, dict):
                                # Check top_features directly
                                if 'top_features' in metric_data and isinstance(metric_data['top_features'], dict):
                                    feature_importance = metric_data['top_features']
                                    logger.info(f"Using top_features from {metric_name} as feature importance")
                                    break

                                # Check avg_feature_distances
                                if not feature_importance and 'avg_feature_distances' in metric_data and isinstance(metric_data['avg_feature_distances'], dict):
                                    feature_importance = metric_data['avg_feature_distances']
                                    logger.info(f"Using avg_feature_distances from {metric_name} as feature importance")
                                    break

                                # Check in individual results
                                if not feature_importance and 'results' in metric_data and isinstance(metric_data['results'], list) and len(metric_data['results']) > 0:
                                    for result in metric_data['results']:
                                        if isinstance(result, dict) and 'feature_distances' in result:
                                            fd = result['feature_distances']
                                            if isinstance(fd, dict):
                                                if 'top_features' in fd and isinstance(fd['top_features'], dict):
                                                    feature_importance = fd['top_features']
                                                    logger.info(f"Using feature_distances.top_features from {metric_name} results as feature importance")
                                                    break
                                                elif 'all_feature_distances' in fd and isinstance(fd['all_feature_distances'], dict):
                                                    feature_importance = fd['all_feature_distances']
                                                    logger.info(f"Using feature_distances.all_feature_distances from {metric_name} results as feature importance")
                                                    break

                                    if feature_importance:
                                        break

                # Check known locations if still not found
                if not feature_importance:
                    # Check known locations
                    for field in ['primary_model', 'metrics', 'results', 'model_data']:
                        if field in report_data and isinstance(report_data[field], dict):
                            field_data = report_data[field]
                            logger.info(f"Checking '{field}' for feature importance")

                            # Look for direct 'feature_importance' key
                            if 'feature_importance' in field_data and isinstance(field_data['feature_importance'], dict):
                                feature_importance = field_data['feature_importance']
                                logger.info(f"Found feature importance in '{field}.feature_importance'")
                                break

                            # Look for similarly named keys
                            for key in field_data.keys():
                                if 'feature' in key.lower() and 'import' in key.lower() and isinstance(field_data[key], dict):
                                    feature_importance = field_data[key]
                                    logger.info(f"Found feature importance in '{field}.{key}'")
                                    break

                            # Look one level deeper
                            if not feature_importance:
                                for subfield, subvalue in field_data.items():
                                    if isinstance(subvalue, dict):
                                        # Check for feature importance data
                                        if 'feature_importance' in subvalue and isinstance(subvalue['feature_importance'], dict):
                                            feature_importance = subvalue['feature_importance']
                                            logger.info(f"Found feature importance in '{field}.{subfield}.feature_importance'")
                                            break
                                        # Check for fields that look like feature importances directly
                                        if len(subvalue) > 3 and not any(k.startswith('_') for k in subvalue.keys()):
                                            # This might be feature importance data - check values are numeric
                                            numeric = True
                                            for k, v in list(subvalue.items())[:3]:  # Check first few items
                                                try:
                                                    float(v)
                                                except (ValueError, TypeError):
                                                    numeric = False
                                                    break
                                            if numeric:
                                                feature_importance = subvalue
                                                logger.info(f"Found potential feature importance in '{field}.{subfield}'")
                                                break

            if feature_importance and isinstance(feature_importance, dict):
                logger.info(f"Feature importance has {len(feature_importance)} features")

                # Show a sample
                sample_items = list(feature_importance.items())[:3]
                logger.info(f"Feature importance sample: {sample_items}")

                # Use the modular chart system
                try:
                    logger.info("Calling generate_feature_importance_chart")
                    # Convert feature importance to input format expected by the chart generator
                    chart_data = self.chart_generator.generate_feature_distribution_shift(
                        feature_distances=feature_importance,
                        title="Feature Importance for Resilience"
                    )
                    if chart_data:
                        charts['feature_importance_chart'] = chart_data
                        logger.info(f"Generated feature importance chart: {len(chart_data)} bytes")
                    else:
                        logger.error("generate_feature_importance_chart returned empty result")
                except Exception as inner_e:
                    logger.error(f"Error in generate_feature_importance_chart: {str(inner_e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")

                # If there's also model feature importance, create comparison chart
                model_feature_importance = report_data.get('model_feature_importance')
                logger.info(f"Model feature importance present: {model_feature_importance is not None}")

                if model_feature_importance and isinstance(model_feature_importance, dict):
                    try:
                        # Create comparison data for both importance metrics
                        comparison_data = {}
                        for feature in set(feature_importance.keys()) | set(model_feature_importance.keys()):
                            if feature in feature_importance or feature in model_feature_importance:
                                comparison_data[feature] = {
                                    'Model Importance': float(model_feature_importance.get(feature, 0)),
                                    'Resilience Impact': float(feature_importance.get(feature, 0))
                                }
                                
                        if comparison_data:
                            # Use model comparison scatter as a workaround since we don't have a direct equivalent
                            comparison_chart = self.chart_generator.generate_model_comparison_scatter(
                                models_data=comparison_data,
                                title="Feature Importance: Model vs Resilience Analysis",
                                x_label="Model Importance",
                                y_label="Resilience Impact"
                            )
                            if comparison_chart:
                                charts['feature_comparison_chart'] = comparison_chart
                                logger.info(f"Generated feature comparison chart: {len(comparison_chart)} bytes")
                            else:
                                logger.error("Feature comparison chart generation returned empty result")
                    except Exception as inner_e:
                        logger.error(f"Error generating feature comparison chart: {str(inner_e)}")
                        logger.error(f"Traceback: {traceback.format_exc()}")
            else:
                logger.warning("No valid feature importance data found")
        except Exception as e:
            logger.error(f"Error generating feature importance chart: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")

        # 8. Generate model comparison chart
        logger.info("Generating model comparison chart")
        try:
            alternative_models = report_data.get('alternative_models')
            by_alpha = report_data.get('by_alpha')

            logger.info(f"Alternative models present: {alternative_models is not None}")
            logger.info(f"Alpha data present: {by_alpha is not None}")

            # Explore alternative_models structure
            if alternative_models and isinstance(alternative_models, dict):
                logger.info(f"Alternative models has {len(alternative_models)} entries")
                for model_name in alternative_models.keys():
                    logger.info(f"  Found alternative model: {model_name}")

            # Explore by_alpha structure
            if by_alpha and isinstance(by_alpha, dict):
                logger.info(f"Alpha data has {len(by_alpha)} levels")
                # Show sample alpha levels
                sample_alphas = list(by_alpha.keys())[:3]
                logger.info(f"  Alpha levels sample: {sample_alphas}")

                # Check if alpha data has scores
                if sample_alphas:
                    first_alpha_data = by_alpha[sample_alphas[0]]
                    if isinstance(first_alpha_data, dict):
                        logger.info(f"  Alpha data contains keys: {list(first_alpha_data.keys())}")
                        if 'score' in first_alpha_data:
                            logger.info(f"  Alpha data contains score field: {first_alpha_data['score']}")

            # Try to check alphas in alternative locations
            if not by_alpha:
                # Check in distribution_shift.by_alpha
                if 'distribution_shift' in report_data and isinstance(report_data['distribution_shift'], dict):
                    dist_shift = report_data['distribution_shift']

                    if 'by_alpha' in dist_shift and isinstance(dist_shift['by_alpha'], dict):
                        by_alpha = dist_shift['by_alpha']
                        logger.info(f"Found by_alpha in distribution_shift with {len(by_alpha)} entries")

                # If still not found, look in other possible locations
                if not by_alpha:
                    for field in ['alphas', 'perturbation_levels', 'primary_model']:
                        if field in report_data and isinstance(report_data[field], dict):
                            # Check if this looks like alpha data (keys are alpha levels)
                            try:
                                # Try to convert a few keys to float
                                keys = list(report_data[field].keys())[:3]
                                all_float = all(float(k) for k in keys)
                                if all_float:
                                    by_alpha = report_data[field]
                                    logger.info(f"Found alpha data in '{field}'")
                                    break
                            except (ValueError, TypeError):
                                # Not all keys are convertible to float
                                pass

                # If still not found, look in distribution_shift.by_distance_metric structure
                if not by_alpha and 'distribution_shift' in report_data and isinstance(report_data['distribution_shift'], dict):
                    dist_shift = report_data['distribution_shift']

                    if 'by_distance_metric' in dist_shift and isinstance(dist_shift['by_distance_metric'], dict):
                        # Extract data from the results of the first by_distance_metric
                        for metric_name, metric_data in dist_shift['by_distance_metric'].items():
                            if isinstance(metric_data, dict) and 'results' in metric_data:
                                # Group results by alpha
                                alpha_dict = {}
                                for result in metric_data.get('results', []):
                                    if isinstance(result, dict) and 'alpha' in result:
                                        alpha = result['alpha']
                                        alpha_key = str(alpha)
                                        if alpha_key not in alpha_dict:
                                            alpha_dict[alpha_key] = {}

                                        # Convert any metric value to 'score'
                                        if 'worst_metric' in result and 'remaining_metric' in result:
                                            alpha_dict[alpha_key]['score'] = result['remaining_metric']

                                if alpha_dict:
                                    by_alpha = alpha_dict
                                    logger.info(f"Constructed by_alpha from distribution_shift.by_distance_metric.{metric_name} with {len(by_alpha)} entries")
                                    break

            if alternative_models and by_alpha and isinstance(alternative_models, dict) and isinstance(by_alpha, dict):
                try:
                    # Extract primary model data
                    primary_alpha_levels = []
                    primary_scores = []

                    for alpha_str, alpha_data in sorted(by_alpha.items()):
                        try:
                            alpha = float(alpha_str)
                            score = alpha_data.get('score')
                            if score is not None:
                                try:
                                    score = float(score)
                                    primary_alpha_levels.append(alpha)
                                    primary_scores.append(score)
                                except (ValueError, TypeError):
                                    continue
                        except (ValueError, TypeError):
                            continue

                    # Create models data dictionary for our chart generator
                    models_data = {}
                    
                    # Add primary model
                    primary_model_name = report_data.get('model_name', 'Primary Model')
                    primary_resilience_score = report_data.get('resilience_score')
                    models_data[primary_model_name] = {
                        'perturbation_levels': primary_alpha_levels,
                        'scores': primary_scores,
                        'resilience_score': primary_resilience_score
                    }
                    
                    # Add alternative models
                    for alt_name, alt_data in alternative_models.items():
                        if 'by_alpha' in alt_data and isinstance(alt_data['by_alpha'], dict):
                            alt_alpha_levels = []
                            alt_scores = []
                            
                            for alpha_str, alpha_data in sorted(alt_data['by_alpha'].items()):
                                try:
                                    alpha = float(alpha_str)
                                    score = alpha_data.get('score')
                                    if score is not None:
                                        try:
                                            score = float(score)
                                            alt_alpha_levels.append(alpha)
                                            alt_scores.append(score)
                                        except (ValueError, TypeError):
                                            continue
                                except (ValueError, TypeError):
                                    continue
                                    
                            if alt_alpha_levels:
                                alt_resilience_score = None
                                if 'resilience_score' in alt_data:
                                    try:
                                        alt_resilience_score = float(alt_data['resilience_score'])
                                    except (ValueError, TypeError):
                                        pass
                                        
                                models_data[alt_name] = {
                                    'perturbation_levels': alt_alpha_levels,
                                    'scores': alt_scores,
                                    'resilience_score': alt_resilience_score
                                }
                                
                    # Generate chart using our resilience chart generator
                    if len(models_data) > 1:  # We need at least two models to compare
                        logger.info("Calling generate_model_comparison")
                        perturbation_levels = primary_alpha_levels
                        metric_name = report_data.get('metric', 'Score')
                        chart_data = self.chart_generator.generate_model_comparison(
                            perturbation_levels=perturbation_levels,
                            models_data=models_data,
                            title="Model Resilience Comparison",
                            metric_name=metric_name
                        )
                        if chart_data:
                            charts['model_comparison_chart'] = chart_data
                            logger.info(f"Generated model comparison chart: {len(chart_data)} bytes")
                        else:
                            logger.error("generate_model_comparison returned empty result")
                except Exception as inner_e:
                    logger.error(f"Error in generate_model_comparison: {str(inner_e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
            else:
                logger.warning("Missing required data for model comparison chart")
        except Exception as e:
            logger.error(f"Error generating model comparison chart: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")

        # 9. Generate model comparison scatter plot (Accuracy vs Resilience Score)
        logger.info("Generating model comparison scatter plot")
        try:
            # Prepare data for the scatter plot
            models_accuracy_resilience = {}

            # Get the primary model data
            primary_model_name = report_data.get('model_name', 'Primary Model')
            primary_resilience_score = report_data.get('resilience_score')
            primary_accuracy = None

            # Try to find primary model accuracy
            if 'performance_metrics' in report_data and isinstance(report_data['performance_metrics'], dict):
                metrics = report_data['performance_metrics']
                if 'accuracy' in metrics:
                    primary_accuracy = metrics['accuracy']
                elif 'remaining_accuracy' in metrics:
                    primary_accuracy = metrics['remaining_accuracy']

            # Add primary model if we have both values
            if primary_resilience_score is not None and primary_accuracy is not None:
                models_accuracy_resilience[primary_model_name] = {
                    'x': primary_accuracy,
                    'y': primary_resilience_score
                }

            # Add alternative models if available
            if alternative_models and isinstance(alternative_models, dict):
                for model_name, model_data in alternative_models.items():
                    if not isinstance(model_data, dict):
                        continue

                    alt_resilience_score = model_data.get('resilience_score')
                    alt_accuracy = None

                    # Try to find alternative model accuracy
                    if 'performance_metrics' in model_data and isinstance(model_data['performance_metrics'], dict):
                        alt_metrics = model_data['performance_metrics']
                        if 'accuracy' in alt_metrics:
                            alt_accuracy = alt_metrics['accuracy']
                        elif 'remaining_accuracy' in alt_metrics:
                            alt_accuracy = alt_metrics['remaining_accuracy']

                    # Add alternative model if we have both values
                    if alt_resilience_score is not None and alt_accuracy is not None:
                        models_accuracy_resilience[model_name] = {
                            'x': alt_accuracy,
                            'y': alt_resilience_score
                        }

            # Generate the scatter plot if we have at least two models to compare
            if len(models_accuracy_resilience) >= 2:
                try:
                    logger.info("Calling generate_model_comparison_scatter")
                    scatter_chart = self.chart_generator.generate_model_comparison_scatter(
                        models_data=models_accuracy_resilience,
                        title="Accuracy vs Resilience Score",
                        x_label="Accuracy",
                        y_label="Resilience Score"
                    )

                    if scatter_chart:
                        charts['model_comparison_scatter'] = scatter_chart
                        logger.info(f"Generated model comparison scatter plot: {len(scatter_chart)} bytes")
                    else:
                        logger.error("generate_model_comparison_scatter returned empty result")
                except Exception as inner_e:
                    logger.error(f"Error in generate_model_comparison_scatter: {str(inner_e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
            else:
                logger.warning("Not enough models with both accuracy and resilience scores for scatter plot")
        except Exception as e:
            logger.error(f"Error generating model comparison scatter plot: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")

        # 10. Generate distance metrics comparison across alpha levels
        logger.info("Generating distance metrics comparison chart")
        try:
            # Check for distribution_shift.by_distance_metric data
            if 'distribution_shift' in report_data and isinstance(report_data['distribution_shift'], dict):
                dist_shift = report_data['distribution_shift']

                if 'by_distance_metric' in dist_shift and isinstance(dist_shift['by_distance_metric'], dict):
                    by_distance_metric = dist_shift['by_distance_metric']

                    # Get alpha levels from any metric
                    alpha_levels = []
                    for metric_name, metric_data in by_distance_metric.items():
                        if isinstance(metric_data, dict) and 'results' in metric_data:
                            results = metric_data.get('results', [])
                            if results and isinstance(results, list):
                                # Extract alpha values from results
                                alpha_values = []
                                for result in results:
                                    if isinstance(result, dict) and 'alpha' in result:
                                        alpha_values.append(result['alpha'])

                                if alpha_values:
                                    alpha_levels = sorted(alpha_values)
                                    break

                    if alpha_levels:
                        # Prepare data for the metrics comparison chart
                        metrics_data = {}

                        # Extract metrics for each distance metric type
                        for metric_name, metric_data in by_distance_metric.items():
                            if isinstance(metric_data, dict) and 'results' in metric_data:
                                results = metric_data.get('results', [])
                                if results and isinstance(results, list):
                                    # Extract values for each alpha level
                                    metric_values = []
                                    for result in sorted(results, key=lambda x: x.get('alpha', 0)):
                                        if 'avg_distance' in result:
                                            metric_values.append(result['avg_distance'])
                                        elif 'distance' in result:
                                            metric_values.append(result['distance'])
                                        elif 'metric_value' in result:
                                            metric_values.append(result['metric_value'])

                                    if metric_values:
                                        metrics_data[metric_name] = metric_values

                        if metrics_data and len(metrics_data) > 1:
                            try:
                                logger.info("Calling generate_distance_metrics_comparison")
                                metrics_chart = self.chart_generator.generate_distance_metrics_comparison(
                                    alpha_levels=alpha_levels,
                                    metrics_data=metrics_data,
                                    title="Distance Metrics Comparison by Alpha",
                                    y_label="Distance Value"
                                )

                                if metrics_chart:
                                    charts['distance_metrics_comparison'] = metrics_chart
                                    logger.info(f"Generated distance metrics comparison chart: {len(metrics_chart)} bytes")
                                else:
                                    logger.error("generate_distance_metrics_comparison returned empty result")
                            except Exception as inner_e:
                                logger.error(f"Error in generate_distance_metrics_comparison: {str(inner_e)}")
                                logger.error(f"Traceback: {traceback.format_exc()}")
                        else:
                            logger.warning("Not enough metrics data for distance metrics comparison chart")
                    else:
                        logger.warning("No valid alpha levels found for distance metrics comparison")
                else:
                    logger.warning("No by_distance_metric data available for distance metrics comparison")
            else:
                logger.warning("No distribution_shift data available for distance metrics comparison")
        except Exception as e:
            logger.error(f"Error generating distance metrics comparison chart: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")

        # 11. Generate feature distance heatmap
        logger.info("Generating feature distance heatmap")
        try:
            # Check for multiple distance metrics with feature distances
            if 'distribution_shift' in report_data and isinstance(report_data['distribution_shift'], dict):
                dist_shift = report_data['distribution_shift']

                if 'by_distance_metric' in dist_shift and isinstance(dist_shift['by_distance_metric'], dict):
                    by_distance_metric = dist_shift['by_distance_metric']

                    # Prepare data for the feature distance heatmap
                    feature_distances_by_metric = {}

                    # Extract feature distances for each metric
                    for metric_name, metric_data in by_distance_metric.items():
                        if isinstance(metric_data, dict):
                            # Check direct feature distances
                            if 'avg_feature_distances' in metric_data and isinstance(metric_data['avg_feature_distances'], dict):
                                feature_distances_by_metric[metric_name] = metric_data['avg_feature_distances']
                            elif 'top_features' in metric_data and isinstance(metric_data['top_features'], dict):
                                feature_distances_by_metric[metric_name] = metric_data['top_features']
                            # Check in results
                            elif 'results' in metric_data and isinstance(metric_data['results'], list) and len(metric_data['results']) > 0:
                                # Take the last result (highest alpha)
                                last_result = metric_data['results'][-1]
                                if isinstance(last_result, dict) and 'feature_distances' in last_result:
                                    fd = last_result['feature_distances']
                                    if isinstance(fd, dict):
                                        if 'all_feature_distances' in fd and isinstance(fd['all_feature_distances'], dict):
                                            feature_distances_by_metric[metric_name] = fd['all_feature_distances']
                                        elif 'top_features' in fd and isinstance(fd['top_features'], dict):
                                            feature_distances_by_metric[metric_name] = fd['top_features']

                    if feature_distances_by_metric and len(feature_distances_by_metric) > 1:
                        try:
                            logger.info("Calling generate_feature_distance_heatmap")
                            heatmap_chart = self.chart_generator.generate_feature_distance_heatmap(
                                feature_distances=feature_distances_by_metric,
                                title="Feature Distance Heatmap by Metric",
                                top_n=15,
                                cmap="viridis"
                            )

                            if heatmap_chart:
                                charts['feature_distance_heatmap'] = heatmap_chart
                                logger.info(f"Generated feature distance heatmap: {len(heatmap_chart)} bytes")
                            else:
                                logger.error("generate_feature_distance_heatmap returned empty result")
                        except Exception as inner_e:
                            logger.error(f"Error in generate_feature_distance_heatmap: {str(inner_e)}")
                            logger.error(f"Traceback: {traceback.format_exc()}")
                    else:
                        logger.warning("Not enough metrics with feature distances for heatmap")
                else:
                    logger.warning("No by_distance_metric data available for feature distance heatmap")
            else:
                logger.warning("No distribution_shift data available for feature distance heatmap")
        except Exception as e:
            logger.error(f"Error generating feature distance heatmap: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")

        # 12. Generate model resilience scores bar chart
        logger.info("Generating model resilience scores bar chart")
        try:
            # Prepare data for the resilience scores bar chart
            models_resilience = {}

            # Get the primary model data
            primary_model_name = report_data.get('model_name', 'Primary Model')
            primary_resilience_score = report_data.get('resilience_score')

            # Add primary model
            if primary_resilience_score is not None:
                models_resilience[primary_model_name] = primary_resilience_score

            # Add alternative models if available
            if alternative_models and isinstance(alternative_models, dict):
                for model_name, model_data in alternative_models.items():
                    if not isinstance(model_data, dict):
                        continue

                    alt_resilience_score = model_data.get('resilience_score')

                    if alt_resilience_score is not None:
                        models_resilience[model_name] = alt_resilience_score

            # Generate the bar chart if we have at least one model
            if models_resilience:
                try:
                    logger.info("Calling generate_model_resilience_scores")
                    bar_chart = self.chart_generator.generate_model_resilience_scores(
                        models_data=models_resilience,
                        title="Resilience Scores by Model",
                        sort_by="score",
                        ascending=False
                    )

                    if bar_chart:
                        charts['model_resilience_scores'] = bar_chart
                        logger.info(f"Generated model resilience scores bar chart: {len(bar_chart)} bytes")
                    else:
                        logger.error("generate_model_resilience_scores returned empty result")
                except Exception as inner_e:
                    logger.error(f"Error in generate_model_resilience_scores: {str(inner_e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
            else:
                logger.warning("No models with resilience scores available for bar chart")
        except Exception as e:
            logger.error(f"Error generating model resilience scores bar chart: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")

        # 13. Generate performance gap by alpha chart
        logger.info("Generating performance gap by alpha chart")
        try:
            # Extract performance gaps from distribution_shift.by_alpha based on provided example
            alpha_levels = []
            models_perf_data = {}

            # Function to extract performance gaps from a model's data at each alpha level
            def extract_performance_gaps(model_data):
                performance_gaps = []

                # Check if model has distribution_shift.by_alpha structure
                if not isinstance(model_data, dict) or 'distribution_shift' not in model_data:
                    return None

                dist_shift = model_data['distribution_shift']
                if not isinstance(dist_shift, dict) or 'by_alpha' not in dist_shift:
                    return None

                by_alpha = dist_shift['by_alpha']
                if not isinstance(by_alpha, dict):
                    return None

                # Get all alpha values and sort them
                alpha_keys = []
                try:
                    alpha_keys = sorted([float(k) for k in by_alpha.keys()])
                except (ValueError, TypeError):
                    alpha_keys = sorted(by_alpha.keys())

                # Extract performance gaps from each alpha level
                worst_values = []
                remaining_values = []

                for alpha in alpha_keys:
                    alpha_key = str(alpha) if isinstance(alpha, float) else alpha
                    if alpha_key not in by_alpha:
                        continue

                    alpha_data = by_alpha[alpha_key]

                    # Try to extract performance gap directly
                    if 'performance_gap' in alpha_data:
                        performance_gaps.append(alpha_data['performance_gap'])

                    # Try to extract from results
                    elif 'results' in alpha_data and isinstance(alpha_data['results'], list) and len(alpha_data['results']) > 0:
                        results = alpha_data['results'][0]  # Take first result
                        if isinstance(results, dict):
                            # Direct performance gap
                            if 'performance_gap' in results:
                                performance_gaps.append(results['performance_gap'])
                            # Worst and remaining metrics
                            elif 'worst_metric' in results and 'remaining_metric' in results:
                                worst = results.get('worst_metric')
                                remaining = results.get('remaining_metric')
                                try:
                                    worst_values.append(float(worst))
                                    remaining_values.append(float(remaining))
                                except (ValueError, TypeError):
                                    pass

                # If we have worst and remaining values but no direct performance gaps
                if not performance_gaps and worst_values and remaining_values:
                    performance_gaps = [abs(r - w) for w, r in zip(worst_values, remaining_values)]

                # Return both the alpha keys and performance gaps
                if performance_gaps:
                    return {
                        'alphas': alpha_keys[:len(performance_gaps)],
                        'gaps': performance_gaps,
                        'worst': worst_values,
                        'remaining': remaining_values
                    }

                return None

            # Get primary model data
            primary_model_name = report_data.get('model_name', 'Primary Model')
            primary_data = {}

            # Try to get primary model data directly
            if 'distribution_shift' in report_data and isinstance(report_data['distribution_shift'], dict):
                primary_data = {'distribution_shift': report_data['distribution_shift']}
            # Or from primary_model field
            elif 'primary_model' in report_data and isinstance(report_data['primary_model'], dict):
                primary_data = report_data['primary_model']

            # Extract performance gaps for primary model
            primary_perf_data = extract_performance_gaps(primary_data)
            if primary_perf_data and 'alphas' in primary_perf_data:
                alpha_levels = primary_perf_data['alphas']
                models_perf_data[primary_model_name] = primary_perf_data

            # Extract performance gaps for alternative models
            if alternative_models and isinstance(alternative_models, dict):
                for model_name, model_data in alternative_models.items():
                    if not isinstance(model_data, dict):
                        continue

                    model_perf_data = extract_performance_gaps(model_data)
                    if model_perf_data:
                        models_perf_data[model_name] = model_perf_data
                        # If we still don't have alpha levels, get them from this model
                        if not alpha_levels and 'alphas' in model_perf_data:
                            alpha_levels = model_perf_data['alphas']

            # Generate the chart if we have data
            if alpha_levels and models_perf_data:
                try:
                    logger.info(f"Starting performance_gap_by_alpha chart with {len(models_perf_data)} models and {len(alpha_levels)} alpha levels")

                    # Convert data to DataFrame format for plotting using integrated matplotlib
                    try:
                        import pandas as pd
                        import matplotlib.pyplot as plt
                        import seaborn as sns

                        # Prepare data for DataFrame
                        all_alphas = []
                        all_gaps = []
                        all_models = []

                        for model_name, model_data in models_perf_data.items():
                            if 'gaps' in model_data and model_data['gaps']:
                                # Use either model's own alphas or the global alpha levels
                                model_alphas = model_data.get('alphas', alpha_levels[:len(model_data['gaps'])])

                                # Ensure alphas and gaps have the same length
                                n_points = min(len(model_alphas), len(model_data['gaps']))
                                all_alphas.extend(model_alphas[:n_points])
                                all_gaps.extend(model_data['gaps'][:n_points])
                                all_models.extend([model_name] * n_points)

                        if all_alphas and all_gaps and all_models:
                            # Create DataFrame
                            df_gaps = pd.DataFrame({
                                'Alpha': all_alphas,
                                'Performance_Gap': all_gaps,
                                'Model': all_models
                            })

                            # Create plot
                            fig, ax = plt.subplots(figsize=(12, 7))
                            sns.lineplot(data=df_gaps, x='Alpha', y='Performance_Gap', hue='Model',
                                        marker='o', linewidth=2.5, palette='viridis', ax=ax)

                            ax.set_title('Performance Gap by Alpha Level', fontsize=16)
                            ax.set_xlabel('Alpha (Perturbation Intensity)', fontsize=14)
                            ax.set_ylabel('Performance Gap', fontsize=14)
                            ax.legend(title='Model', fontsize=12, title_fontsize=13)
                            ax.grid(True, linestyle='--', alpha=0.7)
                            plt.tight_layout()

                            # Convert to base64
                            buf = io.BytesIO()
                            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                            buf.seek(0)
                            img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                            plt.close(fig)

                            charts['performance_gap_by_alpha'] = f"data:image/png;base64,{img_base64}"
                            logger.info(f"Generated performance gap by alpha chart: {len(charts['performance_gap_by_alpha'])} bytes")
                        else:
                            logger.warning("Failed to prepare data for performance gap chart")
                    except ImportError as imp_err:
                        logger.error(f"Cannot generate chart - required libraries missing: {str(imp_err)}")

                        # Try to use chart generator as fallback
                        if self.chart_generator and hasattr(self.chart_generator, 'generate_performance_gap_by_alpha'):
                            logger.info("Trying chart generator as fallback")
                            chart_data = self.chart_generator.generate_performance_gap_by_alpha(
                                alpha_levels=alpha_levels,
                                models_data=models_perf_data,
                                title="Performance Gap by Alpha Level"
                            )
                            if chart_data:
                                charts['performance_gap_by_alpha'] = chart_data
                                logger.info(f"Generated performance gap chart using chart generator: {len(chart_data)} bytes")
                except Exception as inner_e:
                    logger.error(f"Error generating performance gap chart: {str(inner_e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
            else:
                logger.warning("Not enough data for performance gap by alpha chart")
        except Exception as e:
            logger.error(f"Error generating performance gap by alpha chart: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")

        logger.info(f"Generated {len(charts)} charts for resilience visualization")
        return charts