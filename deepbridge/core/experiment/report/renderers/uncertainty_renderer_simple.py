"""
Simple renderer for uncertainty reports - Following resilience pattern.
Uses Plotly for visualizations and single-page template approach.

Refactored in Phase 2 to use BaseRenderer template methods.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger("deepbridge.reports")

# Import BaseRenderer
from .base_renderer import BaseRenderer


class UncertaintyRendererSimple(BaseRenderer):
    """
    Simple renderer for uncertainty experiment reports.
    Inherits from BaseRenderer to use common template methods (Phase 2).
    """

    def __init__(self, template_manager, asset_manager):
        """
        Initialize the uncertainty renderer.

        Parameters:
        -----------
        template_manager : TemplateManager
            Manager for templates
        asset_manager : AssetManager
            Manager for assets (CSS, JS, images)
        """
        # Call parent constructor (initializes css_manager, etc.)
        super().__init__(template_manager, asset_manager)

        # Import data transformer
        from ..transformers.uncertainty_simple import UncertaintyDataTransformerSimple
        self.data_transformer = UncertaintyDataTransformerSimple()

    def render(self, results: Dict[str, Any], file_path: str, model_name: str = "Model",
               report_type: str = "interactive", save_chart: bool = False) -> str:
        """
        Render uncertainty report from results data.

        Parameters:
        -----------
        results : Dict[str, Any]
            Uncertainty experiment results containing:
            - test_results: Test results with primary_model data
            - initial_model_evaluation: Initial evaluation with feature_importance
        file_path : str
            Path where the HTML report will be saved
        model_name : str, optional
            Name for the report title
        report_type : str, optional
            Type of report to generate ('interactive' or 'static')
        save_chart : bool, optional
            Whether to save charts as separate files (not used in simple renderer, kept for compatibility)

        Returns:
        --------
        str : Path to the generated report

        Raises:
        -------
        FileNotFoundError: If template not found
        ValueError: If required data missing
        """
        logger.info(f"Generating SIMPLE uncertainty report to: {file_path}")
        logger.info(f"Report type: {report_type}")

        try:
            # Transform the data
            report_data = self.data_transformer.transform(results, model_name=model_name)

            # Load template using BaseRenderer method
            template = self._load_template('uncertainty', report_type)
            logger.info(f"Template loaded for uncertainty/{report_type}")

            # Get all assets using BaseRenderer method
            assets = self._get_assets('uncertainty')

            # Create base context using BaseRenderer method
            context = self._create_base_context(report_data, 'uncertainty', assets)

            # Add uncertainty-specific context fields
            context.update({
                'report_title': 'Uncertainty Analysis Report',
                'report_subtitle': 'Conformal Prediction and Calibration',
                'uncertainty_score': report_data['summary']['uncertainty_score'],
                'total_alphas': report_data['summary']['total_alphas'],
                'total_features': report_data['features']['total'],
                'avg_coverage': report_data['summary']['avg_coverage'],
                'avg_coverage_error': report_data['summary']['avg_coverage_error'],
                'avg_width': report_data['summary']['avg_width']
            })

            # Render template using BaseRenderer method
            html_content = self._render_template(template, context)

            # Write HTML using BaseRenderer method
            logger.info(f"Report generated and saved to: {file_path} (type: {report_type})")
            return self._write_html(html_content, file_path)

        except Exception as e:
            logger.error(f"Error generating uncertainty report: {str(e)}")
            raise ValueError(f"Failed to generate uncertainty report: {str(e)}")

    # NOTE: All helper methods (_load_template, _get_assets, _get_css_content,
    # _get_js_content, _safe_json_dumps, _write_html, _render_template,
    # _create_base_context) are now inherited from BaseRenderer (Phase 2 refactoring).
    # This eliminates ~180 lines of duplicate code!
