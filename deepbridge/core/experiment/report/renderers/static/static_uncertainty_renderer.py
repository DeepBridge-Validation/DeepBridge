"""
Static uncertainty report renderer that uses Seaborn for visualizations.
"""

import os
import sys
import logging
import datetime
import traceback
import numpy as np
from typing import Dict, List, Any, Optional

# Configure logger
logger = logging.getLogger("deepbridge.reports")
# Ensure logger has a proper handler
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

class StaticUncertaintyRenderer:
    """
    Renderer for static uncertainty test reports using Seaborn charts.
    """
    
    def __init__(self, template_manager, asset_manager):
        """
        Initialize the static uncertainty renderer.
        
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
        from ...transformers.uncertainty import UncertaintyDataTransformer
        from ...transformers.initial_results import InitialResultsTransformer
        from ...transformers.static.static_uncertainty import StaticUncertaintyTransformer
        self.data_transformer = UncertaintyDataTransformer()
        self.static_transformer = StaticUncertaintyTransformer()
        self.initial_results_transformer = InitialResultsTransformer()
        
        # Import Seaborn chart utilities
        from ...utils.seaborn_utils import SeabornChartGenerator
        self.chart_generator = SeabornChartGenerator()

        # Import UncertaintyChartGenerator for specific uncertainty charts
        try:
            from deepbridge.templates.report_types.uncertainty.static.charts import UncertaintyChartGenerator
            self.uncertainty_chart_generator = UncertaintyChartGenerator()
            logger.info("Successfully loaded UncertaintyChartGenerator")
        except ImportError as e:
            logger.warning(f"Could not load UncertaintyChartGenerator: {e}")
            self.uncertainty_chart_generator = None
    
    def render(self, results: Dict[str, Any], file_path: str, model_name: str = "Model",
              report_type: str = "static", save_chart: bool = False) -> str:
        """
        If save_chart is False (default), charts will be embedded as base64 data directly in the HTML.
        If True, charts will be saved as separate files and referenced with relative URLs.
        """
        """
        Render static uncertainty report from results data.

        Parameters:
        -----------
        results : Dict[str, Any]
            Uncertainty test results
        file_path : str
            Path where the HTML report will be saved
        model_name : str, optional
            Name of the model for display in the report
        report_type : str, optional
            Type of report to generate ('interactive' or 'static')
        save_chart : bool, optional
            Whether to save charts as separate files (True) or embed them directly (False)

        Returns:
        --------
        str : Path to the generated report

        Raises:
        -------
        FileNotFoundError: If template or assets not found
        ValueError: If required data missing
        """
        logger.info(f"Generating static uncertainty report to: {file_path}")
        # Store the report file path for use in chart generation
        self.report_file_path = file_path
        
        try:
            # Find template through standard search paths
            template_paths = self.template_manager.get_template_paths("uncertainty", "static")
            template_path = self.template_manager.find_template(template_paths)
            
            if not template_path:
                raise FileNotFoundError(f"No suitable template found for uncertainty report")
            
            logger.info(f"Using static template: {template_path}")
            
            # Get CSS content
            css_content = self.base_renderer._load_static_css_content()
            
            # Load the template
            template = self.template_manager.load_template(template_path)
            
            # Transform the uncertainty data
            # First use the standard transformer
            logger.info("Starting data transformation with standard transformer")
            report_data = self.data_transformer.transform(results, model_name)
            logger.info(f"Standard transformer produced data with keys: {list(report_data.keys() if isinstance(report_data, dict) else [])}")

            # Then apply additional transformations for static reports
            report_data = self.static_transformer.transform(results, model_name)
            logger.info(f"Static transformer produced data with keys: {list(report_data.keys() if isinstance(report_data, dict) else [])}")
            
            # Transform initial results data if available
            if 'initial_results' in results:
                logger.info("Found initial_results in results, transforming...")
                initial_results = self.initial_results_transformer.transform(results.get('initial_results', {}))
                report_data['initial_results'] = initial_results
            
            # Create the context for the template
            context = self.base_renderer._create_static_context(report_data, "uncertainty", css_content)

            # Ensure report_data is included in the context
            context['report_data'] = report_data

            # For the static template, ensure correct report type values
            context['report_type'] = 'static'
            context['test_type'] = 'uncertainty'

            # Generate charts for the static report
            try:
                charts = self._generate_charts(report_data, save_chart)
                
                # Try to import the chart mapper to improve mappings
                try:
                    from ....chart_mapper import ensure_chart_mappings
                    logger.info("Using chart mapper to ensure all chart names are properly mapped")
                    charts = ensure_chart_mappings(charts)
                except ImportError:
                    logger.info("Chart mapper not available, using explicit mappings")
                    # Make sure we explicitly create mappings for the two chart types that need it
                    if 'marginal_bandwidth' in charts and 'interval_widths_comparison' not in charts:
                        logger.info("Adding marginal_bandwidth → interval_widths_comparison mapping")
                        charts['interval_widths_comparison'] = charts['marginal_bandwidth']
                    if 'interval_widths_comparison' in charts and 'marginal_bandwidth' not in charts:
                        logger.info("Adding interval_widths_comparison → marginal_bandwidth mapping")
                        charts['marginal_bandwidth'] = charts['interval_widths_comparison']
                
                context['charts'] = charts
                logger.info(f"Generated {len(charts)} charts: {list(charts.keys())}")
            except Exception as e:
                logger.error(f"Error generating charts: {str(e)}")
                context['charts'] = {}
            
            # Extract features list, metrics, and feature subset
            features = self._extract_feature_list(report_data)
            metrics, metrics_details = self._extract_metrics(report_data)
            feature_subset, feature_subset_display = self._extract_feature_subset(report_data)
            
            # Add uncertainty-specific context with default values for None
            context.update({
                # Core metrics
                'uncertainty_score': report_data.get('uncertainty_score', 0),
                'coverage': report_data.get('coverage', 0),
                'mean_width': report_data.get('mean_width', 0),

                # Feature importance data
                'feature_importance': report_data.get('feature_importance', {}),
                'model_feature_importance': report_data.get('model_feature_importance', {}),
                'has_feature_importance': bool(report_data.get('feature_importance')),
                'has_model_feature_importance': bool(report_data.get('model_feature_importance')),

                # Test metadata
                'cal_size': report_data.get('cal_size'),
                'model_type': report_data.get('model_type', 'Unknown'),
                'timestamp': report_data.get('timestamp', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                'current_year': datetime.datetime.now().year,

                # Additional context
                'features': features,
                'metrics': metrics,
                'metrics_details': metrics_details,
                'feature_subset': feature_subset,
                'feature_subset_display': feature_subset_display,

                # Title for the report
                'block_title': f"Uncertainty Analysis: {model_name}"
            })

            # Ensure alternative_models data has all required attributes
            if 'alternative_models' in report_data and isinstance(report_data['alternative_models'], dict):
                for model_name, model_data in report_data['alternative_models'].items():
                    # Ensure coverage and mean_width exist in each model
                    if not isinstance(model_data, dict):
                        report_data['alternative_models'][model_name] = {
                            'uncertainty_score': 0,
                            'coverage': 0,
                            'mean_width': 0
                        }
                    else:
                        if 'coverage' not in model_data:
                            report_data['alternative_models'][model_name]['coverage'] = 0
                        if 'mean_width' not in model_data:
                            report_data['alternative_models'][model_name]['mean_width'] = 0
                        if 'uncertainty_score' not in model_data:
                            report_data['alternative_models'][model_name]['uncertainty_score'] = 0
            
            # Map chart names to match the template's expected names
            chart_name_mapping = {
                # This maps the chart names generated by our code to the names expected by the template
                'model_comparison': 'model_comparison',
                'performance_gap_by_alpha': 'performance_gap_by_alpha',
                'reliability_distribution': 'feature_reliability',  # Maps to feature_reliability in template
                'marginal_bandwidth': 'interval_widths_comparison',  # Maps to interval_widths_comparison in template
                'interval_widths_boxplot': 'interval_widths_comparison',
                'model_metrics_comparison': 'model_comparison',  # Maps to model_comparison in template

                # Bidirectional mappings for reliability charts
                'feature_reliability': 'reliability_distribution',
                'reliability_analysis': 'feature_reliability',

                # Bidirectional mappings for bandwidth charts
                'interval_widths_comparison': 'marginal_bandwidth',
                'width_distribution': 'interval_widths_comparison',

                # Bidirectional mappings for model comparison
                'model_comparison_chart': 'model_comparison',
                'model_metrics': 'model_comparison',

                # Special handling for template-specific chart names
                'residual_distribution': 'residual_distribution',
                'feature_residual_correlation': 'feature_residual_correlation',
                'distance_metrics_comparison': 'distance_metrics_comparison',
                'feature_distance_heatmap': 'feature_distance_heatmap',
                'model_resilience_scores': 'model_resilience_scores',
                'coverage_vs_expected': 'coverage_vs_expected',
                'width_vs_coverage': 'width_vs_coverage',
                'uncertainty_metrics': 'uncertainty_metrics',
                'feature_importance': 'feature_importance',

                # New reliability charts
                'reliability_bandwidth': 'reliability_bandwidth',
                'reliability_performance': 'reliability_performance'
            }
            
            # Add aliases for chart names to match template expectations
            if 'charts' in context and context['charts']:
                # Clone the charts dictionary
                mapped_charts = context['charts'].copy()
                
                # Add aliases for chart names to match template expectations
                for old_name, new_name in chart_name_mapping.items():
                    if old_name in context['charts']:
                        # Map from original chart name to template expected name
                        mapped_charts[new_name] = context['charts'][old_name]
                        logger.info(f"Mapped chart '{old_name}' to '{new_name}' for template compatibility")
                        
                        # Also add the original chart directly if it's not already present
                        if old_name not in mapped_charts:
                            mapped_charts[old_name] = context['charts'][old_name]
                            
                # Additional loop to ensure bidirectional mappings work properly
                # This handles cases where a chart was generated using the template name directly
                for old_name, new_name in chart_name_mapping.items():
                    # Check if the new_name exists in original charts but old_name doesn't exist in mapped charts
                    if new_name in context['charts'] and old_name not in mapped_charts:
                        mapped_charts[old_name] = context['charts'][new_name]
                        logger.info(f"Created reverse mapping from '{new_name}' to '{old_name}'")
                
                # Ensure all template-expected chart names are included if we have them
                template_expected_charts = [
                    # Core charts that should always be present
                    'model_comparison', 'coverage_vs_expected', 'width_vs_coverage',
                    'performance_gap_by_alpha', 'uncertainty_metrics', 'feature_importance',

                    # Reliability and bandwidth charts (with all possible names)
                    'feature_reliability', 'reliability_distribution', 'reliability_analysis',
                    'interval_widths_comparison', 'marginal_bandwidth', 'width_distribution',

                    # New reliability charts from PiML
                    'reliability_bandwidth', 'reliability_performance',

                    # Model comparison charts (with all possible names)
                    'model_comparison_chart', 'model_metrics', 'model_metrics_comparison',

                    # Additional charts for special use cases
                    'residual_distribution', 'feature_residual_correlation',
                    'distance_metrics_comparison', 'feature_distance_heatmap',
                    'model_resilience_scores', 'interval_widths_boxplot'
                ]
                
                # Log which expected charts are missing and check for mismatches between naming conventions
                missing_charts = [chart for chart in template_expected_charts if chart not in mapped_charts]
                if missing_charts:
                    logger.info(f"Missing charts needed by template: {missing_charts}")
                
                # Print detailed chart information to help with debugging
                logger.info("=== DETAILED CHART AVAILABILITY ===")
                for chart_name in template_expected_charts:
                    logger.info(f"Chart '{chart_name}': {'Available' if chart_name in mapped_charts else 'MISSING'}")
                logger.info("===================================")
                
                # Special case for marginal_bandwidth / interval_widths_comparison
                if 'marginal_bandwidth' in context['charts'] and 'interval_widths_comparison' not in mapped_charts:
                    logger.info("Found 'marginal_bandwidth' but not mapped to 'interval_widths_comparison'. Adding mapping.")
                    mapped_charts['interval_widths_comparison'] = context['charts']['marginal_bandwidth']
                elif 'interval_widths_comparison' in context['charts'] and 'marginal_bandwidth' not in mapped_charts:
                    logger.info("Found 'interval_widths_comparison' but not mapped to 'marginal_bandwidth'. Adding mapping.")
                    mapped_charts['marginal_bandwidth'] = context['charts']['interval_widths_comparison']
                
                # Replace the charts with the mapped version
                context['charts'] = mapped_charts
                
                # List all charts that were generated
                logger.info("----- Charts Generated For Report -----")
                for chart_name in context['charts'].keys():
                    logger.info(f"✓ {chart_name}")
                logger.info("------------------------------------")
                
                # Look for which template chart variables might be used
                try:
                    template_content = template
                    chart_patterns = []
                    import re
                    
                    # Find all {{charts.NAME}} patterns in the template
                    chart_references = re.findall(r'{{[\s]*charts\.([\w\d_]+)[\s]*}}', template_content)
                    
                    # Find all {% if charts.NAME %} patterns in the template 
                    chart_conditions = re.findall(r'{%[\s]*if[\s]*charts\.([\w\d_]+)[\s]*%}', template_content)
                    
                    # Combine and deduplicate
                    all_chart_refs = list(set(chart_references + chart_conditions))
                    
                    logger.info(f"Found {len(all_chart_refs)} chart references in the template: {all_chart_refs}")
                    
                    # Check which charts are missing from our generated charts
                    missing_template_refs = [ref for ref in all_chart_refs if ref not in mapped_charts]
                    if missing_template_refs:
                        logger.warning(f"Template contains references to charts that were not generated: {missing_template_refs}")
                    
                    # Check which generated charts are not referenced in the template
                    unused_charts = [chart for chart in mapped_charts.keys() if chart not in all_chart_refs]
                    if unused_charts:
                        logger.info(f"Generated charts not referenced in the template: {unused_charts}")
                except Exception as e:
                    logger.error(f"Error analyzing template chart references: {str(e)}")
            else:
                logger.warning("No charts were generated for the report")
                
            # Render the template
            rendered_html = self.template_manager.render_template(template, context)

            # Write the report to the file
            report_path = self.base_renderer._write_report(rendered_html, file_path)
            
            logger.info(f"Uncertainty report generated at: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Error generating static uncertainty report: {str(e)}")
            logger.error(f"Error traceback: {traceback.format_exc()}")
            raise ValueError(f"Failed to generate static uncertainty report: {str(e)}")
    
    def _extract_feature_list(self, report_data: Dict[str, Any]) -> List[str]:
        """
        Extract the list of features from the report data.
        
        Parameters:
        -----------
        report_data : Dict[str, Any]
            Transformed report data
            
        Returns:
        --------
        List[str] : List of features
        """
        # Get features directly from report_data if available
        if 'features' in report_data and isinstance(report_data['features'], list):
            return report_data['features']
            
        # If features not found but feature_importance is available, use those keys
        elif 'feature_importance' in report_data and isinstance(report_data['feature_importance'], dict):
            return list(report_data['feature_importance'].keys())
            
        # Return empty list if no features found
        return []
    
    def _extract_metrics(self, report_data: Dict[str, Any]) -> tuple:
        """
        Extract metrics and metrics details from the report data.
        
        Parameters:
        -----------
        report_data : Dict[str, Any]
            Transformed report data
            
        Returns:
        --------
        tuple : (metrics, metrics_details)
        """
        # Simply extract metrics and metrics_details directly from report_data
        metrics = report_data.get('metrics', {})
        metrics_details = report_data.get('metrics_details', {})
        
        return metrics, metrics_details
    
    def _extract_feature_subset(self, report_data: Dict[str, Any]) -> tuple:
        """
        Extract the feature subset and its display string.

        Parameters:
        -----------
        report_data : Dict[str, Any]
            Transformed report data

        Returns:
        --------
        tuple : (feature_subset, feature_subset_display)
        """
        # Get feature subset and display string directly from report data
        feature_subset = report_data.get('feature_subset', [])
        feature_subset_display = report_data.get('feature_subset_display', 'All Features')

        return feature_subset, feature_subset_display

    def _generate_placeholder_chart(self, title, output_path, chart_type="scatter"):
        """
        Generate a placeholder chart when real data is missing or invalid.

        Parameters:
        -----------
        title : str
            Title for the placeholder chart
        output_path : str
            Path where to save the chart
        chart_type : str, optional
            Type of chart to generate (scatter, bar, line)

        Returns:
        --------
        str : Base64 encoded image data
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            import base64
            import io

            # Set style
            plt.style.use('seaborn-v0_8')

            # Create figure
            plt.figure(figsize=(10, 6))

            # Generate some example data
            np.random.seed(42)  # For reproducibility
            x = np.linspace(0, 10, 30)
            y = np.sin(x) + np.random.normal(0, 0.2, size=30)

            # Draw different types of charts
            if chart_type == "bar":
                categories = ['A', 'B', 'C', 'D', 'E']
                values = np.random.rand(5) * 10
                plt.bar(categories, values, color='skyblue')
                plt.axhline(y=np.mean(values), color='r', linestyle='--', label='Average')
                plt.legend()
            elif chart_type == "line":
                plt.plot(x, y, 'o-', label='Data')
                plt.plot(x, np.sin(x), 'r--', label='True Function')
                plt.fill_between(x, np.sin(x)-0.2, np.sin(x)+0.2, color='r', alpha=0.2, label='Uncertainty')
                plt.legend()
            else:  # Default: scatter
                plt.scatter(x, y, alpha=0.7, label='Data Points')
                plt.plot(x, np.sin(x), 'r-', label='Trend')
                plt.legend()

            # Labels and title
            plt.title(f"Example Chart: {title}")
            plt.xlabel("X Axis (Example Values)")
            plt.ylabel("Y Axis (Example Values)")
            plt.grid(True, alpha=0.3)

            # Add watermark
            plt.figtext(0.5, 0.01, "Example data - Generated for demonstration purposes",
                       ha="center", fontsize=8, color="gray")

            # Save to file if path is provided
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved placeholder chart to {output_path}")

            # Also save to base64 for embedding
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            plt.close()

            return f"data:image/png;base64,{base64.b64encode(buffer.read()).decode('utf-8')}"
        except Exception as e:
            logger.error(f"Error generating placeholder chart: {str(e)}")
            return None
    
    def _save_base64_to_file(self, base64_data: str, chart_name: str, charts_dir: str, charts_subdir: str) -> str:
        """
        Converts a base64 image string to a file and returns the relative URL.

        Parameters:
        -----------
        base64_data : str
            Base64 encoded image data (including the data:image/png;base64, prefix)
        chart_name : str
            Name of the chart (used for the filename)
        charts_dir : str
            Directory path where the chart should be saved
        charts_subdir : str
            Subdirectory name for the chart (used in the relative URL)

        Returns:
        --------
        str : Relative URL path to the saved file
        """
        import base64
        import os
        
        try:
            # Extract the base64 part
            if base64_data.startswith('data:image/png;base64,'):
                img_data = base64_data.split('data:image/png;base64,')[1]
                # Generate filename
                chart_filename = f"{chart_name}.png"
                chart_path = os.path.join(charts_dir, chart_filename)
                
                # Save to file
                with open(chart_path, 'wb') as f:
                    f.write(base64.b64decode(img_data))
                logger.info(f"Saved {chart_filename} to {charts_dir}")
                
                # Return relative URL path to the saved file
                return f"./{charts_subdir}/{chart_filename}"
            else:
                logger.error(f"Invalid base64 image format for chart {chart_name}")
                return base64_data
        except Exception as e:
            logger.error(f"Error saving {chart_name} chart as PNG: {str(e)}")
            return base64_data
            
    def _generate_charts(self, report_data: Dict[str, Any], save_chart: bool = True) -> Dict[str, str]:
        """
        Generate all charts needed for the static report.

        Parameters:
        -----------
        report_data : Dict[str, Any]
            Transformed report data
        save_chart : bool, optional
            Whether to save charts as separate files (True) or embed them directly (False)

        Returns:
        --------
        Dict[str, str] : Dictionary of chart names and either their base64 encoded images or file paths
        """
        # Initialize empty charts dictionary
        charts = {}

        # Create a charts directory if save_chart is True
        if save_chart:
            import os
            # Get the directory of the report file
            report_dir = os.path.dirname(os.path.abspath(self.report_file_path))
            # Create a charts subdirectory
            charts_subdir = "uncertainty_charts"
            charts_dir = os.path.join(report_dir, charts_subdir)
            os.makedirs(charts_dir, exist_ok=True)
            logger.info(f"Created chart directory at: {charts_dir}")
        else:
            charts_dir = None
            charts_subdir = None

        try:
            # Use the new modular chart generator system
            try:
                from deepbridge.templates.report_types.uncertainty.static.charts import UncertaintyChartGenerator
                
                # Try to import enhanced charts - log more details about any error
                try:
                    import importlib
                    module_name = 'deepbridge.templates.report_types.uncertainty.static.charts.enhanced_charts'
                    logger.info(f"Attempting to import {module_name}")
                    
                    # Try to import the module directly first (this helps with debugging)
                    try:
                        enhanced_module = importlib.import_module(module_name)
                        logger.info(f"Successfully imported module: {module_name}")
                        logger.info(f"Module contents: {dir(enhanced_module)}")
                    except Exception as module_error:
                        logger.error(f"Error importing module {module_name}: {str(module_error)}")
                    
                    # Now try to import and instantiate the chart class
                    from deepbridge.templates.report_types.uncertainty.static.charts.enhanced_charts import EnhancedUncertaintyCharts
                    enhanced_charts = EnhancedUncertaintyCharts(self.chart_generator)
                    
                    # Verify that the charts object has the expected methods
                    expected_methods = [
                        'generate_reliability_distribution',
                        'generate_marginal_bandwidth_chart',
                        'generate_interval_widths_boxplot',
                        'generate_model_metrics_comparison'
                    ]
                    
                    available_methods = [method for method in expected_methods if hasattr(enhanced_charts, method)]
                    logger.info(f"Successfully loaded enhanced uncertainty charts with methods: {available_methods}")
                    
                    if set(available_methods) != set(expected_methods):
                        missing_methods = set(expected_methods) - set(available_methods)
                        logger.warning(f"Enhanced charts is missing expected methods: {missing_methods}")
                except ImportError as e:
                    enhanced_charts = None
                    logger.error(f"Enhanced uncertainty charts not available: {str(e)}")
                except Exception as e:
                    enhanced_charts = None
                    logger.error(f"Error loading enhanced uncertainty charts: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())

                # Create chart generator with seaborn fallback
                chart_generator = UncertaintyChartGenerator(self.chart_generator)
                logger.info("Using new modular chart generator system for uncertainty visualization")

                # Generate coverage vs expected coverage chart
                try:
                    # Log data needed for this chart
                    logger.info("DADOS PARA COVERAGE VS EXPECTED CHART:")
                    if 'calibration_results' in report_data:
                        logger.info(f"  - calibration_results disponível: {report_data['calibration_results'].keys() if isinstance(report_data['calibration_results'], dict) else 'não é um dicionário'}")
                        if isinstance(report_data['calibration_results'], dict):
                            for key, value in report_data['calibration_results'].items():
                                if isinstance(value, (list, tuple)):
                                    logger.info(f"  - {key}: {len(value)} valores")
                                else:
                                    logger.info(f"  - {key}: {type(value)}")
                    else:
                        logger.error("  - ERRO: calibration_results não está disponível nos dados")

                    # Check alpha values
                    if 'alpha_levels' in report_data:
                        logger.info(f"  - alpha_levels: {report_data['alpha_levels']}")
                    else:
                        logger.error("  - ERRO: alpha_levels não está disponível nos dados")

                    # Add safeguards and better logging for coverage chart
                    try:
                        # Validate data exists in correct format
                        if ('calibration_results' in report_data and 
                            isinstance(report_data['calibration_results'], dict) and
                            'alpha_values' in report_data['calibration_results'] and 
                            'coverage_values' in report_data['calibration_results'] and
                            'expected_coverages' in report_data['calibration_results']):
                            
                            # Log data for debugging
                            alpha_values = report_data['calibration_results']['alpha_values']
                            coverage_values = report_data['calibration_results']['coverage_values']
                            expected_coverages = report_data['calibration_results']['expected_coverages']
                            
                            logger.info(f"Alpha values: {alpha_values} (type: {type(alpha_values)}, len: {len(alpha_values) if hasattr(alpha_values, '__len__') else 'NA'})")
                            logger.info(f"Coverage values: {coverage_values} (type: {type(coverage_values)}, len: {len(coverage_values) if hasattr(coverage_values, '__len__') else 'NA'})")
                            logger.info(f"Expected coverages: {expected_coverages} (type: {type(expected_coverages)}, len: {len(expected_coverages) if hasattr(expected_coverages, '__len__') else 'NA'})")
                            
                            # Make sure data is lists, not numpy arrays
                            if hasattr(alpha_values, 'tolist') and callable(getattr(alpha_values, 'tolist')):
                                report_data['calibration_results']['alpha_values'] = alpha_values.tolist()
                            if hasattr(coverage_values, 'tolist') and callable(getattr(coverage_values, 'tolist')):
                                report_data['calibration_results']['coverage_values'] = coverage_values.tolist()
                            if hasattr(expected_coverages, 'tolist') and callable(getattr(expected_coverages, 'tolist')):
                                report_data['calibration_results']['expected_coverages'] = expected_coverages.tolist()
                            
                            # Now generate chart
                            coverage_chart = chart_generator.generate_coverage_vs_expected(report_data)
                            if coverage_chart:
                                charts['coverage_vs_expected'] = coverage_chart
                                logger.info("Generated coverage vs expected coverage chart")

                                # Save chart to PNG if requested
                                if save_chart and charts_dir:
                                    charts['coverage_vs_expected'] = self._save_base64_to_file(coverage_chart, 'coverage_vs_expected', charts_dir, charts_subdir)
                        else:
                            logger.warning("Missing required data for coverage vs expected chart")
                    except Exception as e:
                        logger.error(f"Error in coverage chart generation: {str(e)}", exc_info=True)
                except Exception as e:
                    logger.error(f"Error generating coverage vs expected chart: {str(e)}")

                # Generate width vs coverage chart
                try:
                    # Log data needed for this chart
                    logger.info("DADOS PARA WIDTH VS COVERAGE CHART:")
                    if 'calibration_results' in report_data:
                        logger.info(f"  - calibration_results disponível: {report_data['calibration_results'].keys() if isinstance(report_data['calibration_results'], dict) else 'não é um dicionário'}")
                        if isinstance(report_data['calibration_results'], dict):
                            # Check for width values
                            if 'width_values' in report_data['calibration_results']:
                                logger.info(f"  - width_values: {len(report_data['calibration_results']['width_values'])} valores")
                            else:
                                logger.error("  - ERRO: width_values não está disponível em calibration_results")

                            # Check for coverage values
                            if 'coverage_values' in report_data['calibration_results']:
                                logger.info(f"  - coverage_values: {len(report_data['calibration_results']['coverage_values'])} valores")
                            else:
                                logger.error("  - ERRO: coverage_values não está disponível em calibration_results")
                    else:
                        logger.error("  - ERRO: calibration_results não está disponível nos dados")

                    # Add safeguards and better logging for width vs coverage chart
                    try:
                        # Validate data exists in correct format
                        if ('calibration_results' in report_data and 
                            isinstance(report_data['calibration_results'], dict) and
                            'width_values' in report_data['calibration_results'] and 
                            'coverage_values' in report_data['calibration_results']):
                            
                            # Log data for debugging
                            width_values = report_data['calibration_results']['width_values']
                            coverage_values = report_data['calibration_results']['coverage_values']
                            
                            logger.info(f"Width values: {width_values} (type: {type(width_values)}, len: {len(width_values) if hasattr(width_values, '__len__') else 'NA'})")
                            logger.info(f"Coverage values: {coverage_values} (type: {type(coverage_values)}, len: {len(coverage_values) if hasattr(coverage_values, '__len__') else 'NA'})")
                            
                            # Make sure data is lists, not numpy arrays
                            if hasattr(width_values, 'tolist') and callable(getattr(width_values, 'tolist')):
                                report_data['calibration_results']['width_values'] = width_values.tolist()
                            if hasattr(coverage_values, 'tolist') and callable(getattr(coverage_values, 'tolist')):
                                report_data['calibration_results']['coverage_values'] = coverage_values.tolist()
                            
                            # Now generate chart
                            width_chart = chart_generator.generate_width_vs_coverage(report_data)
                            if width_chart:
                                charts['width_vs_coverage'] = width_chart
                                logger.info("Generated width vs coverage chart")

                                # Save chart to PNG if requested
                                if save_chart and charts_dir:
                                    charts['width_vs_coverage'] = self._save_base64_to_file(width_chart, 'width_vs_coverage', charts_dir, charts_subdir)
                        else:
                            logger.warning("Missing required data for width vs coverage chart")
                    except Exception as e:
                        logger.error(f"Error in width vs coverage chart generation: {str(e)}", exc_info=True)
                except Exception as e:
                    logger.error(f"Error generating width vs coverage chart: {str(e)}")

                # Generate uncertainty metrics chart
                try:
                    # Log data needed for this chart
                    logger.info("DADOS PARA UNCERTAINTY METRICS CHART:")
                    # Check uncertainty metrics
                    if 'uncertainty_score' in report_data:
                        logger.info(f"  - uncertainty_score: {report_data['uncertainty_score']}")
                    else:
                        logger.error("  - ERRO: uncertainty_score não está disponível nos dados")

                    if 'coverage' in report_data:
                        logger.info(f"  - coverage: {report_data['coverage']}")
                    else:
                        logger.error("  - ERRO: coverage não está disponível nos dados")

                    if 'mean_width' in report_data:
                        logger.info(f"  - mean_width: {report_data['mean_width']}")
                    else:
                        logger.error("  - ERRO: mean_width não está disponível nos dados")

                    # Check metrics dictionary
                    if 'metrics' in report_data:
                        logger.info(f"  - metrics disponível: {list(report_data['metrics'].keys()) if isinstance(report_data['metrics'], dict) else 'não é um dicionário'}")
                    else:
                        logger.error("  - ERRO: metrics não está disponível nos dados")

                    # Use UncertaintyChartGenerator if available, otherwise skip
                    if self.uncertainty_chart_generator:
                        metrics_chart = self.uncertainty_chart_generator.generate_uncertainty_metrics(
                            report_data,
                            title="Uncertainty Metrics"
                        )
                    else:
                        logger.warning("UncertaintyChartGenerator not available, skipping uncertainty_metrics chart")
                        metrics_chart = None
                    if metrics_chart:
                        charts['uncertainty_metrics'] = metrics_chart
                        logger.info("Generated uncertainty metrics chart")

                        # Save chart to PNG if requested
                        if save_chart and charts_dir:
                            charts['uncertainty_metrics'] = self._save_base64_to_file(metrics_chart, 'uncertainty_metrics', charts_dir, charts_subdir)
                except Exception as e:
                    logger.error(f"Error generating uncertainty metrics chart: {str(e)}")

                # Generate feature importance chart
                logger.info("DADOS PARA FEATURE IMPORTANCE CHART:")
                if 'feature_importance' in report_data:
                    # Log feature importance structure
                    logger.info(f"  - feature_importance type: {type(report_data['feature_importance'])}")
                    if isinstance(report_data['feature_importance'], dict):
                        logger.info(f"  - feature_importance tem {len(report_data['feature_importance'])} características")
                        # Log some features as examples
                        for i, (feature, importance) in enumerate(report_data['feature_importance'].items()):
                            if i < 5:  # Show first 5 features only
                                logger.info(f"  - Exemplo: {feature}: {importance}")
                            else:
                                break
                    else:
                        logger.error(f"  - ERRO: feature_importance não é um dicionário: {type(report_data['feature_importance'])}")

                    try:
                        # Use UncertaintyChartGenerator if available, otherwise use SeabornChartGenerator
                        if self.uncertainty_chart_generator:
                            importance_chart = self.uncertainty_chart_generator.generate_feature_importance(
                                report_data.get('feature_importance', {}),
                                title="Feature Importance"
                            )
                        else:
                            importance_chart = chart_generator.generate_feature_importance(report_data)
                        if importance_chart:
                            charts['feature_importance'] = importance_chart
                            logger.info("Generated feature importance chart")

                            # Save chart to PNG if requested
                            if save_chart and charts_dir:
                                charts['feature_importance'] = self._save_base64_to_file(importance_chart, 'feature_importance', charts_dir, charts_subdir)
                    except Exception as e:
                        logger.error(f"Error generating feature importance chart: {str(e)}")
                else:
                    logger.error("  - ERRO: feature_importance não está disponível nos dados")

                # Generate reliability bandwidth (calibration) chart
                try:
                    logger.info("DADOS PARA RELIABILITY BANDWIDTH CHART:")
                    # Prepare data for reliability bandwidth
                    reliability_data = {}

                    # Try to extract prediction data from the results
                    # Check multiple possible locations for the data
                    y_true = None
                    y_prob = None

                    # First try to get predictions from the primary_model if available
                    try:
                        # Check if we have primary_model data in report_data
                        if 'primary_model' in report_data and isinstance(report_data['primary_model'], dict):
                            primary = report_data['primary_model']
                            # Check for test_predictions and test_labels
                            if 'test_predictions' in primary:
                                y_prob = primary['test_predictions']
                                logger.info(f"  - Found test_predictions in primary_model")
                            if 'test_labels' in primary:
                                y_true = primary['test_labels']
                                logger.info(f"  - Found test_labels in primary_model")
                            # Also check for predictions and labels
                            if y_prob is None and 'predictions' in primary:
                                y_prob = primary['predictions']
                                logger.info(f"  - Found predictions in primary_model")
                            if y_true is None and 'labels' in primary:
                                y_true = primary['labels']
                                logger.info(f"  - Found labels in primary_model")
                    except Exception as e:
                        logger.warning(f"  - Could not get data from primary_model: {e}")

                    # Try to get predictions from the model directly if available
                    try:
                        # Check if we have model and test data in report_data
                        if y_prob is None and 'model' in report_data and 'X_test' in report_data:
                            model = report_data['model']
                            X_test = report_data['X_test']
                            # Get probability predictions
                            if hasattr(model, 'predict_proba'):
                                y_prob = model.predict_proba(X_test)
                                logger.info(f"  - Generated predictions from model.predict_proba")
                            elif hasattr(model, 'predict'):
                                y_prob = model.predict(X_test)
                                logger.info(f"  - Generated predictions from model.predict")
                    except Exception as e:
                        logger.warning(f"  - Could not generate predictions from model: {e}")

                    # Check in report_data directly (these may come from results transformation)
                    if y_prob is None and 'test_predictions' in report_data:
                        y_prob = report_data.get('test_predictions')
                        logger.info(f"  - Found test_predictions in report_data")
                    elif y_prob is None and 'predictions' in report_data:
                        y_prob = report_data.get('predictions')
                        logger.info(f"  - Found predictions in report_data")

                    if 'test_labels' in report_data:
                        y_true = report_data.get('test_labels')
                        logger.info(f"  - Found test_labels in report_data")
                    elif 'ground_truth' in report_data:
                        y_true = report_data.get('ground_truth')
                        logger.info(f"  - Found ground_truth in report_data")
                    elif 'y_test' in report_data:
                        y_true = report_data.get('y_test')
                        logger.info(f"  - Found y_test in report_data")

                    # If not found, check for other naming conventions in report_data
                    if y_true is None and 'y_true' in report_data:
                        y_true = report_data['y_true']
                        logger.info(f"  - Found y_true in report_data")
                    if y_prob is None and 'y_prob' in report_data:
                        y_prob = report_data['y_prob']
                        logger.info(f"  - Found y_prob in report_data")

                    # If we have the data, prepare it for the chart
                    if y_true is not None and y_prob is not None:
                        # Ensure arrays
                        import numpy as np
                        y_true = np.array(y_true)
                        y_prob = np.array(y_prob)

                        # If y_prob is 2D (probability for each class), take the positive class
                        if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
                            y_prob = y_prob[:, 1]

                        reliability_data["Primary Model"] = {
                            "y_true": y_true,
                            "y_prob": y_prob
                        }
                        logger.info(f"  - Primary model data prepared: y_true shape {y_true.shape}, y_prob shape {y_prob.shape}")
                    else:
                        logger.warning(f"  - Missing data: y_true is None: {y_true is None}, y_prob is None: {y_prob is None}")

                    # Add alternative models if available
                    if 'alternative_models' in report_data and isinstance(report_data['alternative_models'], dict):
                        for model_name, model_data in report_data['alternative_models'].items():
                            if 'y_true' in model_data and 'y_prob' in model_data:
                                reliability_data[model_name] = {
                                    "y_true": model_data['y_true'],
                                    "y_prob": model_data['y_prob']
                                }
                                logger.info(f"  - Alternative model {model_name} has reliability data")

                    if reliability_data:
                        reliability_bandwidth_chart = chart_generator.generate_reliability_bandwidth(
                            reliability_data, n_bins=10, confidence=0.95,
                            show_confidence_bands=True, show_histogram=True
                        )
                        if reliability_bandwidth_chart:
                            charts['reliability_bandwidth'] = reliability_bandwidth_chart
                            logger.info("Generated reliability bandwidth chart")

                            # Save chart to PNG if requested
                            if save_chart and charts_dir:
                                charts['reliability_bandwidth'] = self._save_base64_to_file(
                                    reliability_bandwidth_chart, 'reliability_bandwidth', charts_dir, charts_subdir
                                )
                    else:
                        logger.warning("No data available for reliability bandwidth chart")

                except Exception as e:
                    logger.error(f"Error generating reliability bandwidth chart: {str(e)}")

                # Generate reliability performance chart
                try:
                    logger.info("DADOS PARA RELIABILITY PERFORMANCE CHART:")
                    # Use the same reliability_data from above
                    if 'reliability_data' in locals() and reliability_data:
                        reliability_perf_chart = chart_generator.generate_reliability_performance(
                            reliability_data, n_bins=10, metric='accuracy',
                            show_sample_sizes=True, show_confidence_bars=True
                        )
                        if reliability_perf_chart:
                            charts['reliability_performance'] = reliability_perf_chart
                            logger.info("Generated reliability performance chart")

                            # Save chart to PNG if requested
                            if save_chart and charts_dir:
                                charts['reliability_performance'] = self._save_base64_to_file(
                                    reliability_perf_chart, 'reliability_performance', charts_dir, charts_subdir
                                )
                    else:
                        logger.warning("No data available for reliability performance chart")

                except Exception as e:
                    logger.error(f"Error generating reliability performance chart: {str(e)}")

                # Generate model comparison chart
                try:
                    # Log data needed for this chart
                    logger.info("DADOS PARA MODEL COMPARISON CHART:")
                    if 'alternative_models' in report_data:
                        alt_models = report_data['alternative_models']
                        logger.info(f"  - alternative_models: {len(alt_models) if isinstance(alt_models, dict) else 'não é um dicionário'}")

                        if isinstance(alt_models, dict):
                            # Log model names
                            logger.info(f"  - Modelos alternativos: {list(alt_models.keys())}")

                            # Check if models have necessary metrics
                            for model_name, model_data in alt_models.items():
                                logger.info(f"  - Modelo {model_name}: ")
                                if isinstance(model_data, dict):
                                    if 'uncertainty_score' in model_data:
                                        logger.info(f"    - uncertainty_score: {model_data['uncertainty_score']}")
                                    else:
                                        logger.error(f"    - ERRO: uncertainty_score não disponível para {model_name}")

                                    if 'coverage' in model_data:
                                        logger.info(f"    - coverage: {model_data['coverage']}")
                                    else:
                                        logger.error(f"    - ERRO: coverage não disponível para {model_name}")

                                    if 'mean_width' in model_data:
                                        logger.info(f"    - mean_width: {model_data['mean_width']}")
                                    else:
                                        logger.error(f"    - ERRO: mean_width não disponível para {model_name}")
                                else:
                                    logger.error(f"    - ERRO: dados do modelo não são um dicionário: {type(model_data)}")
                        else:
                            logger.error(f"  - ERRO: alternative_models não é um dicionário: {type(alt_models)}")
                    else:
                        logger.error("  - ERRO: alternative_models não está disponível nos dados")

                    comparison_chart = chart_generator.generate_model_comparison(report_data)
                    if comparison_chart:
                        charts['model_comparison'] = comparison_chart
                        logger.info("Generated model comparison chart")

                        # Save chart to PNG if requested
                        if save_chart and charts_dir:
                            charts['model_comparison'] = self._save_base64_to_file(comparison_chart, 'model_comparison', charts_dir, charts_subdir)
                except Exception as e:
                    logger.error(f"Error generating model comparison chart: {str(e)}")

                # Generate performance gap by alpha chart
                try:
                    # Log data needed for this chart
                    logger.info("DADOS PARA PERFORMANCE GAP BY ALPHA CHART:")
                    # Check for alpha levels
                    if 'alpha_levels' in report_data:
                        logger.info(f"  - alpha_levels: {report_data['alpha_levels']}")
                    else:
                        logger.error("  - ERRO: alpha_levels não está disponível nos dados")

                    # Check calibration_results
                    if 'calibration_results' in report_data:
                        cal_results = report_data['calibration_results']
                        logger.info(f"  - calibration_results disponível: {cal_results.keys() if isinstance(cal_results, dict) else 'não é um dicionário'}")

                        # Check for alpha and coverage arrays
                        if isinstance(cal_results, dict):
                            if 'alpha_values' in cal_results and 'coverage_values' in cal_results:
                                logger.info(f"  - alpha_values: {len(cal_results['alpha_values'])} valores")
                                logger.info(f"  - coverage_values: {len(cal_results['coverage_values'])} valores")
                                logger.info(f"  - expected_coverages: {len(cal_results.get('expected_coverages', []))} valores")
                            else:
                                logger.error("  - ERRO: alpha_values, coverage_values ou expected_coverages ausentes em calibration_results")
                    else:
                        logger.error("  - ERRO: calibration_results não está disponível nos dados")

                    # Try to see if signature matches the function
                    import inspect
                    try:
                        from deepbridge.templates.report_types.uncertainty.static.charts import UncertaintyChartGenerator
                        sig = inspect.signature(UncertaintyChartGenerator.generate_performance_gap_by_alpha)
                        logger.info(f"  - Argumentos necessários para generate_performance_gap_by_alpha: {list(sig.parameters.keys())}")
                    except Exception as import_err:
                        logger.error(f"  - Não foi possível inspecionar a assinatura do método: {str(import_err)}")

                    performance_gap_chart = chart_generator.generate_performance_gap_by_alpha(models_data=report_data, title="Performance Gap by Alpha Level", add_annotations=True)
                    if performance_gap_chart:
                        charts['performance_gap_by_alpha'] = performance_gap_chart
                        logger.info("Generated performance gap by alpha chart")

                        # Save chart to PNG if requested
                        if save_chart and charts_dir:
                            charts['performance_gap_by_alpha'] = self._save_base64_to_file(performance_gap_chart, 'performance_gap_by_alpha', charts_dir, charts_subdir)
                except Exception as e:
                    logger.error(f"Error generating performance gap by alpha chart: {str(e)}")
                    
                # Generate enhanced charts if available
                if enhanced_charts:
                    logger.info("Generating enhanced uncertainty charts")
                    
                    # Generate reliability distribution chart for top feature
                    try:
                        # Check if reliability analysis data is available
                        if 'reliability_analysis' in report_data:
                            logger.info("Attempting to generate reliability distribution chart")
                            logger.info(f"reliability_analysis keys: {list(report_data['reliability_analysis'].keys())}")
                            
                            # Detailed logging of reliability_analysis data structure
                            for key, value in report_data['reliability_analysis'].items():
                                if isinstance(value, dict):
                                    logger.info(f"  - {key} is a dictionary with keys: {list(value.keys())}")
                                elif isinstance(value, list):
                                    logger.info(f"  - {key} is a list with {len(value)} items")
                                    if value and len(value) < 10:  # Only log small lists
                                        logger.info(f"    Values: {value}")
                                else:
                                    logger.info(f"  - {key}: {value}")
                            
                            # Check if feature_distributions is available
                            if 'feature_distributions' in report_data['reliability_analysis']:
                                logger.info(f"feature_distributions available with types: {list(report_data['reliability_analysis']['feature_distributions'].keys())}")
                                
                                for dist_type, features in report_data['reliability_analysis']['feature_distributions'].items():
                                    logger.info(f"Type {dist_type} has features: {list(features.keys())}")
                                    
                                    # Log a sample of the data for the first feature
                                    if features:
                                        first_feature = next(iter(features.keys()))
                                        feature_data = features[first_feature]
                                        if isinstance(feature_data, dict):
                                            logger.info(f"Sample data for feature '{first_feature}': {list(feature_data.keys())}")
                                        else:
                                            logger.info(f"Sample data for feature '{first_feature}' is type: {type(feature_data)}")
                            else:
                                logger.warning("feature_distributions not available in reliability_analysis")
                            
                            # Get top feature from feature importance
                            top_feature = None
                            if 'feature_importance' in report_data and report_data['feature_importance']:
                                # Sort by importance and get top feature
                                top_feature = sorted(report_data['feature_importance'].items(), 
                                                   key=lambda x: x[1], reverse=True)[0][0]
                                logger.info(f"Selected top feature: {top_feature}")
                            else:
                                logger.warning("No feature_importance available to select top feature")
                                
                            # Ensure enhanced_charts exists and has the required method
                            if enhanced_charts and hasattr(enhanced_charts, 'generate_reliability_distribution'):
                                logger.info("Calling generate_reliability_distribution")
                                reliability_chart = enhanced_charts.generate_reliability_distribution(report_data, top_feature)
                                
                                if reliability_chart:
                                    charts['reliability_distribution'] = reliability_chart
                                    logger.info("Successfully generated reliability distribution chart")
                                    
                                    # Save chart to PNG if requested
                                    if save_chart and charts_dir:
                                        charts['reliability_distribution'] = self._save_base64_to_file(reliability_chart, 'reliability_distribution', charts_dir, charts_subdir)
                                else:
                                    logger.warning("generate_reliability_distribution returned None")
                            else:
                                logger.error("Enhanced charts not available or missing generate_reliability_distribution method")
                        else:
                            logger.warning("reliability_analysis not available in report_data")
                    except Exception as e:
                        logger.error(f"Error generating reliability distribution chart: {str(e)}")
                        import traceback
                        logger.error(traceback.format_exc())
                    
                    # Generate marginal bandwidth chart for top feature
                    try:
                        # Check if marginal bandwidth data is available
                        if 'marginal_bandwidth' in report_data:
                            logger.info("Attempting to generate marginal bandwidth chart")
                            logger.info(f"marginal_bandwidth type: {type(report_data['marginal_bandwidth'])}")
                            
                            # Verify data structure and log more details
                            if isinstance(report_data['marginal_bandwidth'], dict):
                                logger.info(f"marginal_bandwidth keys: {list(report_data['marginal_bandwidth'].keys())}")
                                
                                # Check if feature_impacts is directly in marginal_bandwidth
                                if 'feature_impacts' in report_data['marginal_bandwidth']:
                                    feature_impacts = report_data['marginal_bandwidth']['feature_impacts']
                                    logger.info(f"feature_impacts found directly in marginal_bandwidth: {type(feature_impacts)}")
                                    
                                    if isinstance(feature_impacts, dict):
                                        logger.info(f"Features in feature_impacts: {list(feature_impacts.keys())}")
                                        # Log a few sample values
                                        sample_features = list(feature_impacts.keys())[:5]  # First 5 features
                                        for feature in sample_features:
                                            logger.info(f"  - {feature}: {feature_impacts[feature]}")
                                
                                # Get first feature in marginal bandwidth data (if structure is feature->data)
                                top_feature = None
                                if report_data['marginal_bandwidth']:
                                    # Try to get first key, skipping 'feature_impacts' if it exists
                                    keys = [k for k in report_data['marginal_bandwidth'].keys() if k != 'feature_impacts']
                                    if keys:
                                        top_feature = keys[0]
                                        logger.info(f"Selected feature for bandwidth chart: {top_feature}")
                                        
                                        # Log the structure of data for this feature
                                        feature_data = report_data['marginal_bandwidth'][top_feature]
                                        if isinstance(feature_data, dict):
                                            logger.info(f"Feature data keys: {list(feature_data.keys())}")
                                            for key, value in feature_data.items():
                                                if isinstance(value, (list, tuple)):
                                                    logger.info(f"  - {key}: {len(value)} values")
                                                    if len(value) < 5:  # Only show values for small arrays
                                                        logger.info(f"    Values: {value}")
                                                else:
                                                    logger.info(f"  - {key}: {value}")
                                        else:
                                            logger.info(f"Feature data is not a dictionary but a {type(feature_data)}")
                                            if isinstance(feature_data, (int, float, str, bool)):
                                                logger.info(f"Value: {feature_data}")
                                    else:
                                        logger.warning("marginal_bandwidth has no feature keys")
                                else:
                                    logger.warning("marginal_bandwidth dictionary is empty")
                            else:
                                logger.warning(f"marginal_bandwidth is not a dictionary but a {type(report_data['marginal_bandwidth'])}")
                                
                            # Ensure enhanced_charts exists and has the required method
                            if enhanced_charts and hasattr(enhanced_charts, 'generate_marginal_bandwidth_chart'):
                                logger.info("Calling generate_marginal_bandwidth_chart")
                                bandwidth_chart = enhanced_charts.generate_marginal_bandwidth_chart(report_data, top_feature)
                                
                                if bandwidth_chart:
                                    charts['marginal_bandwidth'] = bandwidth_chart
                                    logger.info("Successfully generated marginal bandwidth chart")
                                    
                                    # Save chart to PNG if requested
                                    if save_chart and charts_dir:
                                        charts['marginal_bandwidth'] = self._save_base64_to_file(bandwidth_chart, 'marginal_bandwidth', charts_dir, charts_subdir)
                                else:
                                    logger.warning("generate_marginal_bandwidth_chart returned None")
                            else:
                                logger.error("Enhanced charts not available or missing generate_marginal_bandwidth_chart method")
                        else:
                            logger.warning("marginal_bandwidth not available in report_data")
                    except Exception as e:
                        logger.error(f"Error generating marginal bandwidth chart: {str(e)}")
                        import traceback
                        logger.error(traceback.format_exc())
                    
                    # Generate interval widths boxplot
                    try:
                        # Check if interval widths data is available
                        if 'interval_widths' in report_data:
                            logger.info("Attempting to generate interval widths boxplot")
                            
                            # Log information about the interval_widths data
                            if isinstance(report_data['interval_widths'], list):
                                logger.info(f"interval_widths is a list with {len(report_data['interval_widths'])} elements")
                                if report_data['interval_widths'] and isinstance(report_data['interval_widths'][0], list):
                                    logger.info(f"First element is a list with {len(report_data['interval_widths'][0])} values")
                            elif isinstance(report_data['interval_widths'], dict):
                                logger.info(f"interval_widths is a dictionary with keys: {list(report_data['interval_widths'].keys())}")
                                for model, widths in report_data['interval_widths'].items():
                                    if isinstance(widths, list):
                                        logger.info(f"  - {model}: {len(widths)} values")
                            else:
                                logger.warning(f"interval_widths has unexpected type: {type(report_data['interval_widths'])}")
                            
                            # Ensure enhanced_charts exists and has the required method
                            if enhanced_charts and hasattr(enhanced_charts, 'generate_interval_widths_boxplot'):
                                logger.info("Calling generate_interval_widths_boxplot")
                                interval_chart = enhanced_charts.generate_interval_widths_boxplot(report_data)
                                
                                if interval_chart:
                                    charts['interval_widths_boxplot'] = interval_chart
                                    logger.info("Successfully generated interval widths boxplot")
                                    
                                    # Save chart to PNG if requested
                                    if save_chart and charts_dir:
                                        charts['interval_widths_boxplot'] = self._save_base64_to_file(interval_chart, 'interval_widths_boxplot', charts_dir, charts_subdir)
                                else:
                                    logger.warning("generate_interval_widths_boxplot returned None")
                            else:
                                logger.error("Enhanced charts not available or missing generate_interval_widths_boxplot method")
                        else:
                            logger.warning("interval_widths not available in report_data")
                    except Exception as e:
                        logger.error(f"Error generating interval widths boxplot: {str(e)}")
                        import traceback
                        logger.error(traceback.format_exc())
                    
                    # Generate model metrics comparison
                    try:
                        logger.info("Generating model metrics comparison")
                        metrics_chart = enhanced_charts.generate_model_metrics_comparison(report_data)
                        if metrics_chart:
                            charts['model_metrics_comparison'] = metrics_chart
                            logger.info("Generated model metrics comparison")
                            
                            # Save chart to PNG if requested
                            if save_chart and charts_dir:
                                charts['model_metrics_comparison'] = self._save_base64_to_file(metrics_chart, 'model_metrics_comparison', charts_dir, charts_subdir)
                    except Exception as e:
                        logger.error(f"Error generating model metrics comparison: {str(e)}")

            except ImportError as e:
                logger.warning(f"Could not import modular chart generator: {str(e)}")
                logger.warning("Falling back to legacy chart generation")

                # Fallback to legacy chart generation
                # Generate interval widths comparison chart - legacy method
                if 'interval_widths' in report_data:
                    try:
                        from ...utils.uncertainty_charts import generate_interval_widths_comparison
                        import tempfile
                        import os
                        import base64

                        # Use the charts directory if save_chart is True, otherwise use a temp dir
                        if save_chart and charts_dir:
                            chart_dir = charts_dir
                        else:
                            # Create a temporary directory for the chart
                            chart_dir = tempfile.mkdtemp()

                        # Generate the chart
                        chart_path = generate_interval_widths_comparison(report_data, chart_dir)

                        # Add to charts if generated successfully
                        if chart_path and os.path.exists(chart_path):
                            # Get the filename from the path
                            chart_filename = os.path.basename(chart_path)
                            # Use relative URL for the chart
                            if save_chart and charts_dir:
                                charts['interval_widths_comparison'] = f"./{charts_subdir}/{chart_filename}"
                            else:
                                # If not using save_chart, still include as base64
                                with open(chart_path, 'rb') as img_file:
                                    chart_data = base64.b64encode(img_file.read()).decode('utf-8')
                                charts['interval_widths_comparison'] = f"data:image/png;base64,{chart_data}"
                            logger.info(f"Successfully generated interval widths comparison chart at {chart_path}")
                    except Exception as e:
                        logger.error(f"Error generating interval widths chart: {str(e)}")

                # Generate feature reliability chart - legacy method
                if 'feature_reliability' in report_data:
                    try:
                        feature_data = {
                            'Feature': [],
                            'PSI': []
                        }

                        # Extract top features by PSI
                        sorted_features = sorted(
                            [(feature, data.get('psi', 0)) for feature, data in report_data['feature_reliability'].items()],
                            key=lambda x: x[1], reverse=True
                        )[:10]  # Get top 10

                        if sorted_features:
                            feature_data['Feature'] = [feature for feature, _ in sorted_features]
                            feature_data['PSI'] = [score for _, score in sorted_features]

                            charts['feature_reliability'] = self.chart_generator.feature_psi_chart(
                                psi_data=feature_data,
                                title="Feature Reliability"
                            )
                            logger.info("Generated feature reliability chart")
                    except Exception as e:
                        logger.error(f"Error generating feature reliability chart: {str(e)}")

                # Generate additional uncertainty charts if available - legacy method
                try:
                    from ...utils.uncertainty_report_charts import generate_all_uncertainty_charts
                    import tempfile

                    # Use the charts directory if save_chart is True, otherwise use a temp dir
                    if save_chart and charts_dir:
                        additional_charts_dir = charts_dir
                    else:
                        # Create a temporary directory for the charts
                        additional_charts_dir = tempfile.mkdtemp()

                    # Generate charts with real data only, no examples
                    logger.info("Generating static uncertainty charts from real data only")
                    additional_charts = generate_all_uncertainty_charts(additional_charts_dir)

                    if additional_charts:
                        # Convert base64 data to relative paths if save_chart is True
                        if save_chart and charts_dir:
                            for chart_name, chart_data in additional_charts.items():
                                if isinstance(chart_data, str) and chart_data.startswith('data:image/png;base64,'):
                                    # Extract chart filename
                                    chart_filename = f"{chart_name}.png"
                                    # Create relative path
                                    additional_charts[chart_name] = f"./{charts_subdir}/{chart_filename}"
                                    
                                    # Save the file if it doesn't exist already
                                    chart_path = os.path.join(charts_dir, chart_filename)
                                    if not os.path.exists(chart_path):
                                        try:
                                            import base64
                                            img_data = chart_data.split('data:image/png;base64,')[1]
                                            with open(chart_path, 'wb') as f:
                                                f.write(base64.b64decode(img_data))
                                            logger.info(f"Saved additional chart {chart_filename} to {charts_dir}")
                                        except Exception as e:
                                            logger.error(f"Error saving additional chart {chart_name}: {str(e)}")
                                    
                        charts.update(additional_charts)
                        logger.info(f"Added {len(additional_charts)} additional uncertainty charts to {additional_charts_dir}")
                    else:
                        logger.warning("No additional charts were generated")
                except (ImportError, AttributeError):
                    logger.info("Additional uncertainty charts module not available")
                except Exception as e:
                    logger.error(f"Error generating additional uncertainty charts: {str(e)}")
                    logger.error(traceback.format_exc())

        except Exception as e:
            logger.error(f"Error generating charts: {str(e)}")
            logger.error(traceback.format_exc())

        # ========== GENERATE NEW CUSTOM CHARTS ==========
        # Generate our 3 new charts: Interval Boxplot, PSI Analysis, and Top Features Distribution
        try:
            logger.info("=" * 50)
            logger.info("Generating custom DeepBridge charts")
            logger.info("=" * 50)

            # 1. INTERVAL BOXPLOT CHART
            try:
                logger.info("Generating Interval Boxplot chart...")

                # Prepare data for boxplot
                boxplot_data = {'results': []}

                # Try to extract widths from different possible locations

                # Check for interval_widths (from transformed data)
                if 'interval_widths' in report_data:
                    logger.info("Found interval_widths in report_data")
                    interval_widths = report_data['interval_widths']

                    # If it's a dictionary with model names
                    if isinstance(interval_widths, dict):
                        for model_name, widths in interval_widths.items():
                            if isinstance(widths, list) and widths:
                                # Use default alpha values if not available
                                alpha = 0.1 if model_name == 'primary_model' else 0.2
                                boxplot_data['results'].append({
                                    'alpha': alpha,
                                    'widths': widths,
                                    'coverage': report_data.get('coverage', 0.9),
                                    'mean_width': report_data.get('mean_width', np.mean(widths) if widths else 0)
                                })
                    # If it's a list of widths directly
                    elif isinstance(interval_widths, list):
                        if interval_widths:
                            # Check if it's a list of lists (multiple alpha levels)
                            if isinstance(interval_widths[0], list):
                                for i, widths in enumerate(interval_widths):
                                    alpha = report_data.get('alpha_levels', [0.1, 0.2])[i] if i < len(report_data.get('alpha_levels', [])) else 0.1 + i * 0.1
                                    boxplot_data['results'].append({
                                        'alpha': alpha,
                                        'widths': widths,
                                        'coverage': 0.9,  # default
                                        'mean_width': np.mean(widths) if widths else 0
                                    })
                            else:
                                # Single list of widths
                                boxplot_data['results'].append({
                                    'alpha': report_data.get('alpha', 0.1),
                                    'widths': interval_widths,
                                    'coverage': report_data.get('coverage', 0.9),
                                    'mean_width': report_data.get('mean_width', np.mean(interval_widths))
                                })

                # Also check calibration_results for alpha-specific data
                if not boxplot_data['results'] and 'calibration_results' in report_data:
                    calib = report_data['calibration_results']
                    if 'alpha_values' in calib and 'width_values' in calib:
                        logger.info("Using calibration_results for boxplot data")
                        # Note: width_values here are mean widths per alpha, not full distributions
                        # So we can't create a proper boxplot from this data
                        pass

                # Fallback: Check for multiple configurations (different alpha values)
                if not boxplot_data['results'] and 'results' in report_data and isinstance(report_data['results'], list):
                    for result in report_data['results']:
                        if 'widths' in result and 'alpha' in result:
                            boxplot_data['results'].append({
                                'alpha': result['alpha'],
                                'widths': result['widths'],
                                'coverage': result.get('coverage', 0),
                                'mean_width': result.get('mean_width', 0)
                            })
                # Single configuration
                elif not boxplot_data['results'] and 'widths' in report_data:
                    boxplot_data['results'].append({
                        'alpha': report_data.get('alpha', 0.1),
                        'widths': report_data['widths'],
                        'coverage': report_data.get('coverage', 0),
                        'mean_width': report_data.get('mean_width', 0)
                    })

                if boxplot_data['results']:
                    interval_boxplot_chart = chart_generator.generate_interval_boxplot(boxplot_data)
                    if interval_boxplot_chart:
                        charts['interval_boxplot'] = interval_boxplot_chart
                        logger.info("✅ Successfully generated Interval Boxplot chart")

                        # Save chart to file if requested
                        if save_chart and charts_dir:
                            charts['interval_boxplot'] = self._save_base64_to_file(
                                interval_boxplot_chart, 'interval_boxplot', charts_dir, charts_subdir
                            )
                    else:
                        logger.warning("Interval Boxplot chart generation returned None")
                else:
                    logger.warning("No data available for Interval Boxplot chart")

            except Exception as e:
                logger.error(f"Error generating Interval Boxplot chart: {str(e)}")
                logger.error(traceback.format_exc())

            # 2. PSI ANALYSIS CHART
            try:
                logger.info("Generating PSI Analysis chart...")

                # Check for reliability analysis with PSI values
                psi_data = None
                if 'reliability_analysis' in report_data:
                    if 'psi_values' in report_data['reliability_analysis']:
                        psi_data = {
                            'reliability_analysis': report_data['reliability_analysis']
                        }
                # Alternative: check in results
                elif 'results' in report_data:
                    if isinstance(report_data['results'], dict):
                        if 'reliability_analysis' in report_data['results']:
                            psi_data = {'results': report_data['results']}
                    elif isinstance(report_data['results'], list) and report_data['results']:
                        # Check first result
                        if 'reliability_analysis' in report_data['results'][0]:
                            psi_data = {'results': report_data['results'][0]}

                if psi_data:
                    psi_chart = chart_generator.generate_psi_analysis(psi_data)
                    if psi_chart:
                        charts['psi_analysis'] = psi_chart
                        logger.info("✅ Successfully generated PSI Analysis chart")

                        # Save chart to file if requested
                        if save_chart and charts_dir:
                            charts['psi_analysis'] = self._save_base64_to_file(
                                psi_chart, 'psi_analysis', charts_dir, charts_subdir
                            )
                    else:
                        logger.warning("PSI Analysis chart generation returned None")
                else:
                    logger.warning("No PSI data available for PSI Analysis chart")

            except Exception as e:
                logger.error(f"Error generating PSI Analysis chart: {str(e)}")
                logger.error(traceback.format_exc())

            # 3. TOP FEATURES DISTRIBUTION CHART
            try:
                logger.info("Generating Top Features Distribution chart...")

                # Check for reliability analysis with distributions
                dist_data = None
                if 'reliability_analysis' in report_data:
                    if ('psi_values' in report_data['reliability_analysis'] and
                        'feature_distributions' in report_data['reliability_analysis']):
                        dist_data = {
                            'reliability_analysis': report_data['reliability_analysis']
                        }
                # Alternative: check in results
                elif 'results' in report_data:
                    if isinstance(report_data['results'], dict):
                        if 'reliability_analysis' in report_data['results']:
                            dist_data = {'results': report_data['results']}
                    elif isinstance(report_data['results'], list) and report_data['results']:
                        # Check first result
                        if 'reliability_analysis' in report_data['results'][0]:
                            dist_data = {'results': report_data['results'][0]}

                if dist_data:
                    dist_chart = chart_generator.generate_top_features_distribution(dist_data, top_n=3)
                    if dist_chart:
                        charts['top_features_distribution'] = dist_chart
                        logger.info("✅ Successfully generated Top Features Distribution chart")

                        # Save chart to file if requested
                        if save_chart and charts_dir:
                            charts['top_features_distribution'] = self._save_base64_to_file(
                                dist_chart, 'top_features_distribution', charts_dir, charts_subdir
                            )
                    else:
                        logger.warning("Top Features Distribution chart generation returned None")
                else:
                    logger.warning("No distribution data available for Top Features Distribution chart")

            except Exception as e:
                logger.error(f"Error generating Top Features Distribution chart: {str(e)}")
                logger.error(traceback.format_exc())

            logger.info("=" * 50)
            logger.info(f"Custom charts generation completed. Total charts: {len(charts)}")
            logger.info("=" * 50)

        except Exception as e:
            logger.error(f"Error in custom charts generation block: {str(e)}")
            logger.error(traceback.format_exc())

        return charts