"""
Static resilience report renderer that uses Seaborn for visualizations.
"""

import os
import sys
import logging
import datetime
import traceback
from typing import Dict, List, Any, Optional

# Configure logger
logger = logging.getLogger("deepbridge.reports")
# Ensure logger has proper handlers
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

class StaticResilienceRenderer:
    """
    Renderer for static resilience test reports using Seaborn charts.
    """
    
    def __init__(self, template_manager, asset_manager):
        """
        Initialize the static resilience renderer.

        Parameters:
        -----------
        template_manager : TemplateManager
            Manager for templates
        asset_manager : AssetManager
            Manager for assets (CSS, JS, images)
        """
        from .base_static_renderer import BaseStaticRenderer
        self.base_renderer = BaseStaticRenderer(template_manager, asset_manager)
        self.template_manager = template_manager
        self.asset_manager = asset_manager

        # Import transformers
        from ...transformers.resilience import ResilienceDataTransformer
        from ...transformers.initial_results import InitialResultsTransformer
        self.data_transformer = ResilienceDataTransformer()
        self.initial_results_transformer = InitialResultsTransformer()

        # Import Seaborn chart utilities
        try:
            from ...utils.seaborn_utils import SeabornChartGenerator
            self.chart_generator = SeabornChartGenerator()
            logger.info("Successfully imported and initialized SeabornChartGenerator")

            # Check if the chart generator has visualization libraries
            if hasattr(self.chart_generator, 'has_visualization_libs'):
                logger.info(f"SeabornChartGenerator has_visualization_libs: {self.chart_generator.has_visualization_libs}")
            else:
                logger.warning("SeabornChartGenerator does not have has_visualization_libs attribute")

            # Check available methods
            chart_methods = [method for method in dir(self.chart_generator)
                            if callable(getattr(self.chart_generator, method)) and not method.startswith('_')]
            logger.info(f"Available chart methods: {chart_methods}")
        except ImportError as e:
            logger.error(f"Could not import SeabornChartGenerator: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.chart_generator = None

        # Import necessary libraries for chart generation
        try:
            import numpy as np
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import seaborn as sns
            import pandas as pd
            self.np = np
            self.sns = sns
            self.pd = pd
            logger.info("Successfully imported visualization libraries (numpy, matplotlib, seaborn, pandas)")
        except ImportError as e:
            logger.error(f"Could not import visualization libraries: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.np = None

        # Import the new modular Resilience chart utilities
        try:
            # Try different possible import paths for the resilience chart generator
            resilience_chart_module = None
            try:
                # Try absolute import
                from deepbridge.templates.report_types.resilience.static.charts import ResilienceChartGenerator
                resilience_chart_module = "deepbridge.templates.report_types.resilience.static.charts"
            except ImportError:
                try:
                    # Try relative import based on project structure
                    import sys
                    import os
                    # Get the project root directory (assuming deepbridge is the root package)
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    project_root = os.path.abspath(os.path.join(current_dir, "../../../../../.."))
                    charts_path = os.path.join(project_root, "templates", "report_types", "resilience", "static", "charts")

                    if charts_path not in sys.path:
                        sys.path.append(charts_path)
                    from __init__ import ResilienceChartGenerator
                    resilience_chart_module = charts_path
                except ImportError:
                    # Last try - use original path but just in case it's still available somewhere
                    from ...utils.resilience_charts import ResilienceChartGenerator
                    resilience_chart_module = "...utils.resilience_charts"

            self.resilience_chart_generator = ResilienceChartGenerator(self.chart_generator)
            logger.info(f"Successfully loaded resilience-specific chart generator from {resilience_chart_module}")

            # Check resilience chart methods
            if self.resilience_chart_generator:
                resilience_methods = [method for method in dir(self.resilience_chart_generator)
                                    if callable(getattr(self.resilience_chart_generator, method))
                                    and not method.startswith('_')]
                logger.info(f"Available resilience chart methods: {resilience_methods}")

            # Check if resilience_chart_generator has visualization libs
            if hasattr(self.resilience_chart_generator, '_validate_chart_generator'):
                logger.info("ResilienceChartGenerator has validation method")
                try:
                    self.resilience_chart_generator._validate_chart_generator()
                    logger.info("ResilienceChartGenerator validation successful")
                except Exception as e:
                    logger.error(f"ResilienceChartGenerator validation failed: {str(e)}")
        except ImportError as e:
            logger.error(f"Resilience chart generator not available: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.resilience_chart_generator = None

            # Try one last approach - import directly from the old module location if still available
            try:
                import importlib.util
                import os

                # Try to find the resilience_charts.py file
                chart_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                         "utils", "resilience_charts.py")

                if os.path.exists(chart_file):
                    logger.info(f"Found old resilience_charts.py file at {chart_file}")

                    # Load the module
                    spec = importlib.util.spec_from_file_location("resilience_charts", chart_file)
                    resilience_charts = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(resilience_charts)

                    # Get the class
                    if hasattr(resilience_charts, "ResilienceChartGenerator"):
                        self.resilience_chart_generator = resilience_charts.ResilienceChartGenerator(self.chart_generator)
                        logger.info("Loaded ResilienceChartGenerator from old module file")
            except Exception as e2:
                logger.error(f"Failed to load chart generator from file: {str(e2)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
    
    def render(self, results: Dict[str, Any], file_path: str, model_name: str = "Model", report_type: str = "static", save_chart: bool = False) -> str:
        """
        Render static resilience report from results data.

        Parameters:
        -----------
        results : Dict[str, Any]
            Resilience test results
        file_path : str
            Path where the HTML report will be saved
        model_name : str, optional
            Name of the model for display in the report
        report_type : str, optional
            Type of report to generate ('interactive' or 'static')
        save_chart : bool, optional
            Whether to save charts as separate PNG files (default: False)

        Returns:
        --------
        str : Path to the generated report

        Raises:
        -------
        FileNotFoundError: If template or assets not found
        ValueError: If required data missing
        """
        logger.info(f"Generating static resilience report to: {file_path}")

        try:
            # Find the static resilience template
            template_paths = self.template_manager.get_template_paths("resilience", "static")
            try:
                template_path = self.template_manager.find_template(template_paths)
                logger.info(f"Found resilience template: {template_path}")
            except Exception as e:
                raise FileNotFoundError(f"No static template found for resilience report: {str(e)}")

            logger.info(f"Using static template: {template_path}")

            # Get CSS content using CSSManager (via base_renderer)
            css_content = self.base_renderer._load_static_css_content('resilience')

            # Load the template
            template = self.template_manager.load_template(template_path)

            # Transform the resilience data
            # First use the standard transformer
            logger.info("Transforming resilience data with standard transformer")
            report_data = self.data_transformer.transform(results, model_name)
            logger.info(f"Standard transformation complete, top-level keys: {list(report_data.keys() if isinstance(report_data, dict) else [])}")

            # Then apply additional transformations for static reports using the dedicated transformer
            from ...transformers.static.static_resilience import StaticResilienceTransformer
            static_transformer = StaticResilienceTransformer()
            logger.info("Using StaticResilienceTransformer")

            # Apply the static transformer to get specialized data for static reports
            transformed_data = static_transformer.transform(results, model_name)
            report_data = transformed_data
            logger.info("Applied static transformations to report data")
            logger.info(f"Static transformation result keys: {list(report_data.keys() if isinstance(report_data, dict) else [])}")

            # Transform initial results data if available
            if isinstance(results, dict) and 'initial_results' in results:
                logger.info("Transforming initial results data")
                initial_results = self.initial_results_transformer.transform(results.get('initial_results', {}))
                report_data['initial_results'] = initial_results

            # Check if the transformer already generated charts
            if 'charts' in report_data and isinstance(report_data['charts'], dict) and report_data['charts']:
                # Use charts from transformer
                logger.info(f"Using {len(report_data['charts'])} charts from transformer")
                charts = report_data['charts']
            else:
                # Generate charts with real data from test results
                logger.info("Generating charts using real data from test results")
                charts = self._generate_charts(report_data)

            # Save charts as PNG files if requested
            if save_chart and charts:
                self.base_renderer.save_charts_as_png(charts, file_path)

            # Create the context for the template
            context = self.base_renderer._create_static_context(report_data, "resilience", css_content)

            # Add charts to context
            context['charts'] = charts

            # Add key context data
            context['test_type'] = 'resilience'
            context['report_type'] = 'resilience'
            context['model_name'] = model_name
            context['current_year'] = datetime.datetime.now().year
            context['resilience_score'] = report_data.get('resilience_score')
            context['avg_performance_gap'] = report_data.get('avg_performance_gap')
            context['metric'] = report_data.get('metric', 'Score')
            context['timestamp'] = report_data.get('timestamp', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            context['model_type'] = report_data.get('model_type', 'Classification')
            context['features'] = report_data.get('features', [])

            # Add sensitive features if available
            if 'sensitive_features' in report_data:
                context['sensitive_features'] = report_data['sensitive_features']

            # Add distribution metrics if available
            if 'distribution_shift' in report_data and isinstance(report_data['distribution_shift'], dict):
                dist_shift_data = report_data['distribution_shift']
                if 'avg_dist_shift' in dist_shift_data:
                    context['dist_shift'] = dist_shift_data['avg_dist_shift']
                elif 'avg_distance' in dist_shift_data:
                    context['dist_shift'] = dist_shift_data['avg_distance']

            # Add most affected scenario if available
            if 'most_affected_scenario' in report_data:
                context['most_affected_scenario'] = report_data['most_affected_scenario']

            # Add report_data_json for JavaScript
            import json
            report_data_json = {
                'model_name': model_name,
                'resilience_score': context['resilience_score'],
                'avg_performance_gap': context['avg_performance_gap'],
                'charts': {k: v for k, v in charts.items()},
                'timestamp': context['timestamp']
            }
            context['report_data_json'] = json.dumps(report_data_json)

            # Create empty JS content if not present
            context['js_content'] = context.get('js_content', '')

            # Log key template variables for debugging
            logger.info(f"Template variables: resilience_score={context.get('resilience_score')}, avg_performance_gap={context.get('avg_performance_gap')}, metric={context.get('metric')}")

            # Render the template
            rendered_html = self.template_manager.render_template(template, context)

            # Write the report to the file
            return self.base_renderer._write_report(rendered_html, file_path)

        except Exception as e:
            logger.error(f"Error generating static resilience report: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def _save_charts_as_png(self, charts: Dict[str, str], file_path: str) -> None:
        """
        Save charts as PNG files in the same directory as the HTML report.

        Parameters:
        -----------
        charts : Dict[str, str]
            Dictionary of chart names and their base64 encoded images
        file_path : str
            Path to the HTML report file
        """
        import os
        import base64

        # Get the directory of the HTML report
        output_dir = os.path.dirname(os.path.abspath(file_path))

        # Get the filename (without extension) to use as a prefix
        file_basename = os.path.splitext(os.path.basename(file_path))[0]

        # Create the charts directory if it doesn't exist
        charts_dir = os.path.join(output_dir, f"{file_basename}_charts")
        os.makedirs(charts_dir, exist_ok=True)

        logger.info(f"Saving charts to directory: {charts_dir}")

        # Save each chart as a PNG file
        for chart_name, chart_data in charts.items():
            try:
                # Extract the base64 encoded image data
                if chart_data and isinstance(chart_data, str) and chart_data.startswith('data:image/png;base64,'):
                    # Remove the data URL prefix
                    base64_data = chart_data.replace('data:image/png;base64,', '')

                    # Decode the base64 data
                    image_data = base64.b64decode(base64_data)

                    # Generate a filename for the chart
                    chart_filename = f"{chart_name}.png"
                    chart_path = os.path.join(charts_dir, chart_filename)

                    # Save the image data to a file
                    with open(chart_path, 'wb') as f:
                        f.write(image_data)

                    logger.info(f"Saved chart to: {chart_path}")
                else:
                    logger.warning(f"Chart '{chart_name}' does not contain valid PNG data, skipping")
            except Exception as e:
                logger.error(f"Error saving chart '{chart_name}' to PNG: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
    

    def _generate_charts(self, report_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate all charts needed for the static resilience report.

        Parameters:
        -----------
        report_data : Dict[str, Any]
            Transformed report data

        Returns:
        --------
        Dict[str, str] : Dictionary of chart names and their base64 encoded images
        """
        charts = {}
        logger.info("Generating charts for resilience report")
        logger.info(f"Input data for chart generation has keys: {list(report_data.keys())}")

        # Log presence of important data fields
        for key in ['resilience_score', 'avg_performance_gap', 'feature_distances',
                    'performance_metrics', 'feature_importance', 'residuals',
                    'feature_correlations', 'alternative_models']:
            if key in report_data:
                logger.info(f"Data field '{key}' is present")
                if isinstance(report_data[key], dict):
                    logger.info(f"  '{key}' is a dictionary with {len(report_data[key])} entries")
                elif isinstance(report_data[key], list):
                    logger.info(f"  '{key}' is a list with {len(report_data[key])} items")
            else:
                logger.warning(f"Data field '{key}' is MISSING")

        # 1. Resilience score chart
        logger.info("Generating resilience score chart")
        try:
            resilience_score = report_data.get('resilience_score')
            performance_gap = report_data.get('avg_performance_gap')

            logger.info(f"Resilience score data: score={resilience_score}, performance_gap={performance_gap}")

            if resilience_score is not None or performance_gap is not None:
                x_values = []
                y_values = []

                if resilience_score is not None:
                    try:
                        x_values.append('Resilience Score')
                        y_values.append(float(resilience_score))
                        logger.info(f"Added resilience score to chart: {float(resilience_score)}")
                    except (ValueError, TypeError) as e:
                        logger.error(f"Could not convert resilience score to float: {e}")

                if performance_gap is not None:
                    try:
                        x_values.append('Performance Gap')
                        y_values.append(float(performance_gap))
                        logger.info(f"Added performance gap to chart: {float(performance_gap)}")
                    except (ValueError, TypeError) as e:
                        logger.error(f"Could not convert performance gap to float: {e}")

                metrics_data = {
                    'x': x_values,
                    'y': y_values,
                    'x_label': 'Metric',
                    'y_label': 'Value'
                }

                if not x_values or not y_values:
                    logger.error("No valid data for resilience score chart after conversion")
                else:
                    logger.info(f"Calling bar_chart with data: {metrics_data}")
                    try:
                        result = self.chart_generator.bar_chart(
                            data=metrics_data,
                            title="Resilience Metrics Overview"
                        )
                        charts['resilience_score_chart'] = result
                        if result:
                            chart_size = len(result)
                            logger.info(f"Generated resilience score chart, size: {chart_size} bytes")
                        else:
                            logger.error("Chart generated but result is empty")
                    except Exception as e:
                        logger.error(f"Error during chart_generator.bar_chart call: {str(e)}")
                        logger.error(f"Traceback: {traceback.format_exc()}")
            else:
                logger.warning("No resilience score or performance gap data available for chart")
        except Exception as e:
            logger.error(f"Error generating resilience chart: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")

        # 2. Feature Distribution Shift Chart
        logger.info("Generating feature distribution shift chart")
        try:
            feature_distances = report_data.get('feature_distances')
            logger.info(f"Feature distances data present: {feature_distances is not None}")

            # Check 'distribution_shift' field as alternative data source
            distribution_shift = report_data.get('distribution_shift')
            logger.info(f"Distribution shift data present: {distribution_shift is not None}")
            if distribution_shift and isinstance(distribution_shift, dict):
                logger.info(f"Distribution shift data keys: {list(distribution_shift.keys())}")

                if not feature_distances:
                    # Try to extract feature distances from distribution_shift data
                    if isinstance(distribution_shift, dict):
                        # Look for feature distances directly
                        if 'feature_distances' in distribution_shift:
                            feature_distances = distribution_shift['feature_distances']
                            logger.info(f"Found feature_distances in distribution_shift")
                        # Try to find distances by key pattern
                        for key in distribution_shift.keys():
                            if 'feature' in key.lower() and 'distance' in key.lower() and isinstance(distribution_shift[key], dict):
                                feature_distances = distribution_shift[key]
                                logger.info(f"Using {key} as feature_distances")
                                break

            if feature_distances and isinstance(feature_distances, dict):
                logger.info(f"Feature distances data has {len(feature_distances)} entries")
                # Log a sample of feature distances
                sample = {k: v for i, (k, v) in enumerate(feature_distances.items()) if i < 3}
                logger.info(f"Feature distances sample: {sample}")

                if self.resilience_chart_generator:
                    try:
                        logger.info("Calling resilience_chart_generator.generate_feature_distribution_shift")
                        result = self.resilience_chart_generator.generate_feature_distribution_shift(
                            feature_distances=feature_distances,
                            title="Feature Distribution Shift",
                            top_n=10
                        )

                        if result:
                            charts['feature_distribution_shift'] = result
                            logger.info(f"Successfully generated feature distribution shift chart, size: {len(result)} bytes")
                        else:
                            logger.error("generate_feature_distribution_shift returned empty result")
                    except Exception as inner_e:
                        logger.error(f"Error in generate_feature_distribution_shift: {str(inner_e)}")
                        logger.error(f"Traceback: {traceback.format_exc()}")
                else:
                    logger.error("resilience_chart_generator is not available")
            else:
                logger.warning("No valid feature distances data for feature distribution shift chart")
        except Exception as e:
            logger.error(f"Error generating feature distribution shift chart: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")

        # 3. Performance Gap Chart
        logger.info("Generating performance gap chart")
        try:
            performance_metrics = report_data.get('performance_metrics')
            logger.info(f"Performance metrics present: {performance_metrics is not None}")

            # Try to look for performance metrics elsewhere in the data
            if not performance_metrics:
                for key in ['metrics', 'evaluation_metrics', 'results']:
                    if key in report_data and isinstance(report_data[key], dict):
                        logger.info(f"Checking '{key}' for performance metrics")
                        metrics_data = report_data[key]
                        # Look for worst_ and remaining_ prefixes in keys
                        has_worst = any(k.startswith('worst_') for k in metrics_data.keys())
                        has_remaining = any(k.startswith('remaining_') for k in metrics_data.keys())
                        if has_worst or has_remaining:
                            performance_metrics = metrics_data
                            logger.info(f"Found performance metrics in '{key}'")
                            break

            if performance_metrics and isinstance(performance_metrics, dict):
                logger.info(f"Performance metrics keys: {list(performance_metrics.keys())}")

                # Determine task type (classification or regression)
                task_type = "classification"
                if any(key in performance_metrics for key in ['worst_mse', 'worst_mae', 'worst_r2']):
                    task_type = "regression"
                logger.info(f"Detected task type: {task_type}")

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
                    if self.resilience_chart_generator:
                        try:
                            logger.info("Calling resilience_chart_generator.generate_performance_gap")
                            result = self.resilience_chart_generator.generate_performance_gap(
                                performance_metrics=performance_metrics,
                                title="Performance Comparison: Worst vs Remaining Samples",
                                task_type=task_type
                            )

                            if result:
                                charts['performance_gap_chart'] = result
                                logger.info(f"Successfully generated performance gap chart, size: {len(result)} bytes")
                            else:
                                logger.error("generate_performance_gap returned empty result")
                        except Exception as inner_e:
                            logger.error(f"Error in generate_performance_gap: {str(inner_e)}")
                            logger.error(f"Traceback: {traceback.format_exc()}")
                    else:
                        logger.error("resilience_chart_generator is not available")
                else:
                    logger.warning("No paired performance metrics found (worst_X and remaining_X)")
            else:
                logger.warning("No valid performance metrics data available")
        except Exception as e:
            logger.error(f"Error generating performance gap chart: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")

        # 4. Critical Feature Distributions Chart
        logger.info("Generating critical feature distributions chart")
        try:
            feature_importance = report_data.get('feature_importance')
            worst_samples = report_data.get('worst_samples')
            remaining_samples = report_data.get('remaining_samples')

            if feature_importance and (worst_samples or remaining_samples):
                feature_list = list(feature_importance.keys())[:5]  # Get up to 5 top features

                if self.resilience_chart_generator:
                    charts['critical_feature_distributions'] = self.resilience_chart_generator.generate_critical_feature_distributions(
                        worst_samples=worst_samples,
                        remaining_samples=remaining_samples,
                        top_features=feature_list,
                        title="Critical Feature Distributions"
                    )
                    logger.info("Generated critical feature distributions chart")
        except Exception as e:
            logger.error(f"Error generating critical feature distributions chart: {str(e)}")

        # 5. Feature-Residual Correlation Chart
        logger.info("Generating feature-residual correlation chart")
        try:
            feature_correlations = report_data.get('feature_correlations')

            if feature_correlations and isinstance(feature_correlations, dict):
                if self.resilience_chart_generator:
                    charts['feature_residual_correlation'] = self.resilience_chart_generator.generate_feature_residual_correlation(
                        feature_correlations=feature_correlations,
                        title="Feature-Residual Correlation",
                        top_n=8
                    )
                    logger.info("Generated feature-residual correlation heatmap")
        except Exception as e:
            logger.error(f"Error generating feature-residual correlation chart: {str(e)}")

        # 6. Residual Distribution Chart
        logger.info("Generating residual distribution chart")
        try:
            residuals = report_data.get('residuals')
            worst_residuals = report_data.get('worst_residuals')
            remaining_residuals = report_data.get('remaining_residuals')

            if residuals or worst_residuals or remaining_residuals:
                if self.resilience_chart_generator:
                    charts['residual_distribution'] = self.resilience_chart_generator.generate_residual_distribution(
                        worst_residuals=worst_residuals,
                        remaining_residuals=remaining_residuals,
                        all_residuals=residuals,
                        title="Model Residual Distribution"
                    )
                    logger.info("Generated residual distribution chart")
        except Exception as e:
            logger.error(f"Error generating residual distribution chart: {str(e)}")

        # 7. Feature Importance Chart
        logger.info("Generating feature importance chart")
        try:
            feature_importance = report_data.get('feature_importance')

            if feature_importance and isinstance(feature_importance, dict):
                charts['feature_importance_chart'] = self.chart_generator.feature_importance_chart(
                    features=feature_importance,
                    title="Feature Importance for Resilience"
                )
                logger.info("Generated feature importance chart")

                # If there's also model feature importance, create comparison chart
                model_feature_importance = report_data.get('model_feature_importance')

                if model_feature_importance and isinstance(model_feature_importance, dict):
                    charts['feature_comparison_chart'] = self.chart_generator.feature_comparison_chart(
                        model_importance=model_feature_importance,
                        robustness_importance=feature_importance,
                        title="Feature Importance: Model vs Resilience Analysis"
                    )
                    logger.info("Generated feature comparison chart")
        except Exception as e:
            logger.error(f"Error generating feature importance chart: {str(e)}")

        # 8. Model Comparison Chart
        logger.info("Generating model comparison chart")
        try:
            alternative_models = report_data.get('alternative_models')
            by_alpha = report_data.get('by_alpha')

            if alternative_models and by_alpha and isinstance(alternative_models, dict) and isinstance(by_alpha, dict):
                models_data = {}
                primary_model_name = report_data.get('model_name', 'Primary Model')
                resilience_score = report_data.get('resilience_score')

                # Extract primary model data
                primary_alpha_levels = []
                primary_scores = []

                for alpha_str, alpha_data in sorted(by_alpha.items()):
                    try:
                        alpha = float(alpha_str)
                        score = alpha_data.get('score')
                        if score is not None:
                            primary_alpha_levels.append(alpha)
                            primary_scores.append(score)
                    except (ValueError, TypeError):
                        continue

                if primary_alpha_levels and primary_scores:
                    models_data[primary_model_name] = {
                        'scores': primary_scores,
                        'base_score': resilience_score
                    }

                    # Process alternative models
                    for model_name, model_data in alternative_models.items():
                        if 'by_alpha' in model_data and isinstance(model_data['by_alpha'], dict):
                            model_alpha_levels = []
                            model_scores = []

                            for alpha_str, alpha_data in sorted(model_data['by_alpha'].items()):
                                try:
                                    alpha = float(alpha_str)
                                    score = alpha_data.get('score')
                                    if score is not None:
                                        model_alpha_levels.append(alpha)
                                        model_scores.append(score)
                                except (ValueError, TypeError):
                                    continue

                            if model_alpha_levels and model_scores:
                                models_data[model_name] = {
                                    'scores': model_scores,
                                    'base_score': model_data.get('resilience_score')
                                }

                    if len(models_data) > 1:
                        if self.resilience_chart_generator:
                            charts['model_comparison_chart'] = self.resilience_chart_generator.generate_model_comparison(
                                perturbation_levels=primary_alpha_levels,
                                models_data=models_data,
                                title="Model Resilience Comparison",
                                metric_name=report_data.get('metric', 'Score')
                            )
                            logger.info("Generated model comparison chart")
        except Exception as e:
            logger.error(f"Error generating model comparison chart: {str(e)}")

        # 9. Model Comparison Scatter Plot (Accuracy vs Resilience Score)
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
                    'accuracy': primary_accuracy,
                    'resilience_score': primary_resilience_score
                }

            # Add alternative models if available
            alternative_models = report_data.get('alternative_models')
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
                            'accuracy': alt_accuracy,
                            'resilience_score': alt_resilience_score
                        }

            # Generate the scatter plot if we have at least two models to compare
            if len(models_accuracy_resilience) >= 2:
                if self.resilience_chart_generator:
                    try:
                        logger.info("Calling generate_model_comparison_scatter")
                        scatter_chart = self.resilience_chart_generator.generate_model_comparison_scatter(
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
                    logger.error("resilience_chart_generator is not available")
            else:
                logger.warning("Not enough models with both accuracy and resilience scores for scatter plot")
        except Exception as e:
            logger.error(f"Error generating model comparison scatter plot: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")

        # 10. Distance Metrics Comparison Chart
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
                            if self.resilience_chart_generator:
                                try:
                                    logger.info("Calling generate_distance_metrics_comparison")
                                    metrics_chart = self.resilience_chart_generator.generate_distance_metrics_comparison(
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
                                logger.error("resilience_chart_generator is not available")
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

        # 11. Feature Distance Heatmap
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
                        if self.resilience_chart_generator:
                            try:
                                logger.info("Calling generate_feature_distance_heatmap")
                                heatmap_chart = self.resilience_chart_generator.generate_feature_distance_heatmap(
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
                            logger.error("resilience_chart_generator is not available")
                    else:
                        logger.warning("Not enough metrics with feature distances for heatmap")
                else:
                    logger.warning("No by_distance_metric data available for feature distance heatmap")
            else:
                logger.warning("No distribution_shift data available for feature distance heatmap")
        except Exception as e:
            logger.error(f"Error generating feature distance heatmap: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")

        # 12. Model Resilience Scores Bar Chart
        logger.info("Generating model resilience scores bar chart")
        try:
            # Prepare data for the resilience scores bar chart
            models_resilience = {}

            # Get the primary model data
            primary_model_name = report_data.get('model_name', 'Primary Model')
            primary_resilience_score = report_data.get('resilience_score')

            # Add primary model
            if primary_resilience_score is not None:
                try:
                    models_resilience[primary_model_name] = float(primary_resilience_score)
                except (ValueError, TypeError):
                    pass

            # Add alternative models if available
            alternative_models = report_data.get('alternative_models')
            if alternative_models and isinstance(alternative_models, dict):
                for model_name, model_data in alternative_models.items():
                    if not isinstance(model_data, dict):
                        continue

                    alt_resilience_score = model_data.get('resilience_score')

                    if alt_resilience_score is not None:
                        try:
                            models_resilience[model_name] = float(alt_resilience_score)
                        except (ValueError, TypeError):
                            continue

            # Generate the bar chart if we have at least one model
            if models_resilience:
                if self.resilience_chart_generator:
                    try:
                        logger.info("Calling generate_model_resilience_scores")
                        bar_chart = self.resilience_chart_generator.generate_model_resilience_scores(
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
                    logger.error("resilience_chart_generator is not available")
            else:
                logger.warning("No models with resilience scores available for bar chart")
        except Exception as e:
            logger.error(f"Error generating model resilience scores bar chart: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")

        # 13. Performance Gap by Alpha Chart
        logger.info("Generating performance gap by alpha chart")
        try:
            # Check if we have alpha levels
            alpha_levels = []
            by_alpha = report_data.get('by_alpha')
            if by_alpha and isinstance(by_alpha, dict):
                try:
                    alpha_levels = [float(alpha) for alpha in by_alpha.keys()]
                    alpha_levels.sort()
                except (ValueError, TypeError):
                    pass

            # If we don't have alpha levels yet, try to find them in distribution_shift.by_distance_metric
            if not alpha_levels and 'distribution_shift' in report_data and isinstance(report_data['distribution_shift'], dict):
                dist_shift = report_data['distribution_shift']

                if 'by_distance_metric' in dist_shift and isinstance(dist_shift['by_distance_metric'], dict):
                    for metric_name, metric_data in dist_shift['by_distance_metric'].items():
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
                # Prepare data for the performance gap chart
                models_perf_data = {}

                # Function to extract performance data from results
                def extract_performance_data(results):
                    worst_perf = []
                    remaining_perf = []

                    for result in sorted(results, key=lambda x: x.get('alpha', 0)):
                        if 'worst_metric' in result and 'remaining_metric' in result:
                            worst_perf.append(result['worst_metric'])
                            remaining_perf.append(result['remaining_metric'])

                    if worst_perf and remaining_perf:
                        return {'worst': worst_perf, 'remaining': remaining_perf}
                    return None

                # Get primary model performance data
                primary_model_name = report_data.get('model_name', 'Primary Model')
                primary_perf_data = None

                # Look in distribution_shift for primary model
                if 'distribution_shift' in report_data and isinstance(report_data['distribution_shift'], dict):
                    dist_shift = report_data['distribution_shift']

                    if 'by_distance_metric' in dist_shift and isinstance(dist_shift['by_distance_metric'], dict):
                        # Use the first metric with results
                        for metric_data in dist_shift['by_distance_metric'].values():
                            if isinstance(metric_data, dict) and 'results' in metric_data:
                                results = metric_data.get('results', [])
                                if results and isinstance(results, list):
                                    primary_perf_data = extract_performance_data(results)
                                    if primary_perf_data:
                                        break

                # Add primary model if we have performance data
                if primary_perf_data:
                    models_perf_data[primary_model_name] = primary_perf_data

                # Add alternative models if available
                alternative_models = report_data.get('alternative_models')
                if alternative_models and isinstance(alternative_models, dict):
                    for model_name, model_data in alternative_models.items():
                        if not isinstance(model_data, dict):
                            continue

                        # Look for performance data in model's distribution_shift
                        if 'distribution_shift' in model_data and isinstance(model_data['distribution_shift'], dict):
                            alt_dist_shift = model_data['distribution_shift']

                            if 'by_distance_metric' in alt_dist_shift and isinstance(alt_dist_shift['by_distance_metric'], dict):
                                # Use the first metric with results
                                for metric_data in alt_dist_shift['by_distance_metric'].values():
                                    if isinstance(metric_data, dict) and 'results' in metric_data:
                                        results = metric_data.get('results', [])
                                        if results and isinstance(results, list):
                                            alt_perf_data = extract_performance_data(results)
                                            if alt_perf_data:
                                                models_perf_data[model_name] = alt_perf_data
                                                break

                # Generate the performance gap chart if we have at least one model with data
                if models_perf_data:
                    if self.resilience_chart_generator:
                        try:
                            logger.info("Calling generate_performance_gap_by_alpha")
                            gap_chart = self.resilience_chart_generator.generate_performance_gap_by_alpha(
                                alpha_levels=alpha_levels,
                                models_data=models_perf_data,
                                title="Performance Gap by Alpha Level",
                                y_label="Performance Gap"
                            )

                            if gap_chart:
                                charts['performance_gap_by_alpha'] = gap_chart
                                logger.info(f"Generated performance gap by alpha chart: {len(gap_chart)} bytes")
                            else:
                                logger.error("generate_performance_gap_by_alpha returned empty result")
                        except Exception as inner_e:
                            logger.error(f"Error in generate_performance_gap_by_alpha: {str(inner_e)}")
                            logger.error(f"Traceback: {traceback.format_exc()}")
                    else:
                        logger.error("resilience_chart_generator is not available")
                else:
                    logger.warning("No models with performance data available for performance gap chart")
            else:
                logger.warning("No alpha levels available for performance gap chart")
        except Exception as e:
            logger.error(f"Error generating performance gap by alpha chart: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")

        # 12. Model Resilience Scores Chart (Added to fix missing chart)
        logger.info("Generating model resilience scores chart")
        try:
            # Prepare models data for resilience scores
            models_scores_data = {}

            # Get primary model data
            primary_model_name = report_data.get('model_name', 'Primary Model')
            primary_resilience_score = report_data.get('resilience_score')

            if primary_resilience_score is not None:
                models_scores_data[primary_model_name] = {
                    'score': primary_resilience_score,
                    'performance_gap': report_data.get('avg_performance_gap', 0)
                }

            # Add alternative models
            alternative_models = report_data.get('alternative_models', {})
            if alternative_models and isinstance(alternative_models, dict):
                for model_name, model_data in alternative_models.items():
                    if isinstance(model_data, dict):
                        model_score = model_data.get('resilience_score')
                        if model_score is not None:
                            models_scores_data[model_name] = {
                                'score': model_score,
                                'performance_gap': model_data.get('avg_performance_gap', 0)
                            }

            # Generate the chart if we have data
            if models_scores_data and self.resilience_chart_generator:
                try:
                    logger.info(f"Calling generate_model_resilience_scores with {len(models_scores_data)} models")
                    scores_chart = self.resilience_chart_generator.generate_model_resilience_scores(
                        models_data=models_scores_data,
                        title="Resilience Scores by Model",
                        sort_by="score",
                        ascending=False
                    )

                    if scores_chart:
                        charts['model_resilience_scores'] = scores_chart
                        logger.info(f"Generated model resilience scores chart: {len(scores_chart)} bytes")
                    else:
                        logger.error("generate_model_resilience_scores returned empty result")
                except Exception as inner_e:
                    logger.error(f"Error in generate_model_resilience_scores: {str(inner_e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
            else:
                if not models_scores_data:
                    logger.warning("No models data available for resilience scores chart")
                if not self.resilience_chart_generator:
                    logger.error("resilience_chart_generator is not available")
        except Exception as e:
            logger.error(f"Error generating model resilience scores chart: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")

        # 13. Performance Gap by Alpha Chart (ensure it's generated properly)
        logger.info("Ensuring Performance Gap by Alpha chart is generated")
        try:
            # Check if we have by_alpha data
            by_alpha = report_data.get('by_alpha')
            if by_alpha and isinstance(by_alpha, dict) and 'performance_gap_by_alpha' not in charts:
                # Prepare data for the chart
                alpha_levels = []
                performance_gaps = []

                for alpha_str in sorted(by_alpha.keys()):
                    alpha_data = by_alpha[alpha_str]
                    if isinstance(alpha_data, dict):
                        try:
                            alpha_levels.append(float(alpha_str))
                            gap = alpha_data.get('performance_gap',
                                                alpha_data.get('remaining_score', 0) - alpha_data.get('worst_score', 0))
                            performance_gaps.append(gap)
                        except (ValueError, TypeError):
                            continue

                if alpha_levels and performance_gaps and self.resilience_chart_generator:
                    # Create models_data with single model
                    models_perf_data = {
                        report_data.get('model_name', 'Model'): {
                            'gaps': performance_gaps
                        }
                    }

                    try:
                        gap_chart = self.resilience_chart_generator.generate_performance_gap_by_alpha(
                            alpha_levels=alpha_levels,
                            models_data=models_perf_data,
                            title="Performance Gap by Alpha Level",
                            y_label="Performance Gap"
                        )

                        if gap_chart:
                            charts['performance_gap_by_alpha'] = gap_chart
                            logger.info(f"Generated performance gap by alpha chart: {len(gap_chart)} bytes")
                    except Exception as inner_e:
                        logger.error(f"Error in generate_performance_gap_by_alpha: {str(inner_e)}")
        except Exception as e:
            logger.error(f"Error ensuring performance gap by alpha chart: {str(e)}")

        logger.info(f"Generated {len(charts)} charts for resilience report")
        return charts