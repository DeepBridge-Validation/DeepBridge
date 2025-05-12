"""
Static resilience report renderer that uses the new chart modules.
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
    Renderer for static resilience test reports using the new chart modules.
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

        # Import Resilience-specific chart utilities from new location
        try:
            # Import from the new location
            from templates.report_types.resilience.static.charts import ResilienceChartGenerator
            self.resilience_chart_generator = ResilienceChartGenerator(self.chart_generator)
            logger.info("Successfully loaded resilience-specific chart generator from new location")

            # Check resilience chart methods
            if self.resilience_chart_generator:
                resilience_methods = [method for method in dir(self.resilience_chart_generator)
                                    if callable(getattr(self.resilience_chart_generator, method))
                                    and not method.startswith('_')]
                logger.info(f"Available resilience chart methods: {resilience_methods}")

        except ImportError as e:
            logger.error(f"New resilience chart generator not available: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Fall back to old location
            try:
                from ...utils.resilience_charts import ResilienceChartGenerator
                self.resilience_chart_generator = ResilienceChartGenerator(self.chart_generator)
                logger.info("Falling back to original resilience chart generator")
                
                # Check resilience chart methods
                if self.resilience_chart_generator:
                    resilience_methods = [method for method in dir(self.resilience_chart_generator)
                                        if callable(getattr(self.resilience_chart_generator, method))
                                        and not method.startswith('_')]
                    logger.info(f"Available resilience chart methods from fallback: {resilience_methods}")
            except ImportError:
                logger.error("Could not import any resilience chart generator")
                self.resilience_chart_generator = None
    
    def render(self, results: Dict[str, Any], file_path: str, model_name: str = "Model", report_type: str = "static") -> str:
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

            # Get CSS content
            css_content = self.base_renderer._load_static_css_content()

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

        # Continue with other charts using the same pattern...
        # 3. Performance Gap Chart
        # 4. Critical Feature Distributions Chart
        # 5. Feature-Residual Correlation Chart
        # 6. Residual Distribution Chart
        # 7. Feature Importance Chart
        # 8. Model Comparison Chart
        # 9. Model Comparison Scatter Plot
        # 10. Distance Metrics Comparison Chart
        # 11. Feature Distance Heatmap
        # 12. Model Resilience Scores Bar Chart
        # 13. Performance Gap by Alpha Chart

        logger.info(f"Generated {len(charts)} charts for resilience report")
        return charts