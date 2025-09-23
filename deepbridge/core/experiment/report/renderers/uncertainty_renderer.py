"""
Uncertainty report renderer.
"""

import os
import logging
from typing import Dict, Any, Optional, List

# Configure logger
logger = logging.getLogger("deepbridge.reports")

class UncertaintyRenderer:
    """
    Renderer for uncertainty test reports.
    """

    def __init__(self, template_manager, asset_manager):
        """
        Initialize the renderer.

        Parameters:
        -----------
        template_manager : TemplateManager
            Manager for templates
        asset_manager : AssetManager
            Manager for assets (CSS, JS, images)
        """
        from .base_renderer import BaseRenderer
        self.base_renderer = BaseRenderer(template_manager, asset_manager)
        self.template_manager = template_manager
        self.asset_manager = asset_manager

        # Import specific data transformer
        from ..transformers.uncertainty import UncertaintyDataTransformer
        self.data_transformer = UncertaintyDataTransformer()

        # Try to import the new chart generator
        try:
            from deepbridge.templates.report_types.uncertainty.static.charts import UncertaintyChartGenerator
            from ...utils.seaborn_utils import SeabornChartGenerator
            self.chart_generator = UncertaintyChartGenerator(SeabornChartGenerator())
            logger.info("Initialized UncertaintyChartGenerator for rendering")
        except ImportError:
            self.chart_generator = None
            logger.warning("UncertaintyChartGenerator not available, chart generation may be limited")
    
    def render(self, results: Dict[str, Any], file_path: str, model_name: str = "Model", report_type: str = "interactive", save_chart: bool = False) -> str:
        """
        Render uncertainty report from results data.

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
            Whether to save charts as separate files (only for static reports)

        Returns:
        --------
        str : Path to the generated report

        Raises:
        -------
        FileNotFoundError: If template or assets not found
        ValueError: If required data missing
        """
        logger.info(f"Generating uncertainty report to: {file_path} (type: {report_type})")

        # Check if static report was requested
        if report_type.lower() == "static":
            try:
                # Use static renderer
                logger.info("Static report requested, using StaticUncertaintyRenderer")
                from .static.static_uncertainty_renderer import StaticUncertaintyRenderer
                static_renderer = StaticUncertaintyRenderer(self.template_manager, self.asset_manager)
                return static_renderer.render(results, file_path, model_name, report_type, save_chart)
            except Exception as e:
                logger.error(f"Error using static renderer: {str(e)}")
                import traceback
                logger.error(f"Static renderer error traceback: {traceback.format_exc()}")
                raise ValueError(f"Failed to generate static uncertainty report: {str(e)}")

        # Continue with interactive report generation
        try:
            # Find template
            template_paths = self.template_manager.get_template_paths("uncertainty")
            template_path = self.template_manager.find_template(template_paths)

            if not template_path:
                raise FileNotFoundError(f"No template found for uncertainty report in: {template_paths}")

            logger.info(f"Using template: {template_path}")

            # Find CSS and JS paths
            css_dir = self.asset_manager.find_css_path("uncertainty")
            js_dir = self.asset_manager.find_js_path("uncertainty")

            if not css_dir:
                raise FileNotFoundError("CSS directory not found for uncertainty report")

            if not js_dir:
                raise FileNotFoundError("JavaScript directory not found for uncertainty report")

            # Get CSS and JS content
            css_content = self.asset_manager.get_css_content(css_dir)
            js_content = self.asset_manager.get_js_content(js_dir)

            # Load the template
            template = self.template_manager.load_template(template_path)

            # Transform the data
            report_data = self.data_transformer.transform(results, model_name)

            # Create template context
            context = self.base_renderer._create_context(report_data, "uncertainty", css_content, js_content)

            # Add uncertainty-specific context directly from report_data
            context.update({
                'test_type': 'uncertainty'  # Explicit test type
            })

            # Add available metrics and data from report_data without defaults
            if 'uncertainty_score' in report_data:
                context['uncertainty_score'] = report_data['uncertainty_score']
                # For backward compatibility if it exists
                context['robustness_score'] = report_data['uncertainty_score']

            if 'avg_coverage' in report_data:
                context['avg_coverage'] = report_data['avg_coverage']
                context['coverage_score'] = report_data['avg_coverage']

            if 'calibration_error' in report_data:
                context['calibration_error'] = report_data['calibration_error']

            if 'avg_width' in report_data:
                context['avg_width'] = report_data['avg_width']
                context['sharpness'] = report_data['avg_width']

            if 'consistency' in report_data:
                context['consistency'] = report_data['consistency']

            # Add metadata if available
            if 'method' in report_data:
                context['method'] = report_data['method']

            if 'alpha_levels' in report_data:
                context['alpha_levels'] = report_data['alpha_levels']

            # Add features, metrics if available
            if 'features' in report_data:
                context['features'] = report_data['features']

            if 'metrics' in report_data:
                context['metrics'] = report_data['metrics']

            if 'metrics_details' in report_data:
                context['metrics_details'] = report_data['metrics_details']

            # Render the template
            rendered_html = self.template_manager.render_template(template, context)

            # Write the report to file
            return self.base_renderer._write_report(rendered_html, file_path)

        except Exception as e:
            logger.error(f"Error generating uncertainty report: {str(e)}")
            raise ValueError(f"Failed to generate uncertainty report: {str(e)}")