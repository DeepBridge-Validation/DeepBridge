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

            # DEBUG: Log report_data structure
            logger.info("[FEATURE_IMPACT_DEBUG] report_data keys after transform: %s", list(report_data.keys()))
            logger.info("[FEATURE_IMPACT_DEBUG] report_data['charts'] keys: %s", list(report_data.get('charts', {}).keys()))
            logger.info("[FEATURE_IMPACT_DEBUG] 'features' in charts: %s", 'features' in report_data.get('charts', {}))

            # Check if features chart has data
            features_chart = report_data.get('charts', {}).get('features', {})
            logger.info("[FEATURE_IMPACT_DEBUG] features_chart has data: %s traces", len(features_chart.get('data', [])))
            logger.info("[FEATURE_IMPACT_DEBUG] features_chart has layout: %s", bool(features_chart.get('layout')))

            if features_chart.get('data'):
                logger.debug("[FEATURE_IMPACT_DEBUG] First trace type: %s", features_chart['data'][0].get('type', 'unknown'))
                logger.debug("[FEATURE_IMPACT_DEBUG] First trace has y values: %s", bool(features_chart['data'][0].get('y')))

            # Load template using BaseRenderer method
            template = self._load_template('uncertainty', report_type)
            logger.info(f"Template loaded for uncertainty/{report_type}")

            # Get all assets using BaseRenderer method
            assets = self._get_assets('uncertainty')

            # Create base context using BaseRenderer method
            context = self._create_base_context(report_data, 'uncertainty', assets)

            # Add uncertainty-specific context fields
            # All metrics now come from the summary (including base_score and calibration_error)
            context.update({
                'report_title': 'Uncertainty Analysis Report',
                'report_subtitle': 'Conformal Prediction and Calibration',
                'base_score': report_data['summary'].get('base_score', 0.0),
                'uncertainty_score': report_data['summary']['uncertainty_score'],
                'total_alphas': report_data['summary']['total_alphas'],
                'total_features': report_data['features']['total'],
                'avg_coverage': report_data['summary']['avg_coverage'],
                'calibration_error': report_data['summary'].get('calibration_error', report_data['summary']['avg_coverage_error']),
                'avg_coverage_error': report_data['summary']['avg_coverage_error'],
                'avg_width': report_data['summary']['avg_width']
            })

            # DEBUG: Log metrics being sent to template
            logger.info("[METRICS_DEBUG] base_score from summary: %.4f", context.get('base_score', 0.0))
            logger.info("[METRICS_DEBUG] calibration_error from summary: %.4f", context.get('calibration_error', 0.0))
            logger.info("[METRICS_DEBUG] uncertainty_score from summary: %.4f", context.get('uncertainty_score', 0.0))
            logger.info("[METRICS_DEBUG] avg_coverage from summary: %.4f", context.get('avg_coverage', 0.0))

            # DEBUG: Log context being sent to template
            logger.info("[FEATURE_IMPACT_DEBUG] Context keys being sent to template: %s", list(context.keys()))
            logger.info("[FEATURE_IMPACT_DEBUG] Context has report_data: %s", 'report_data' in context)
            logger.info("[FEATURE_IMPACT_DEBUG] Context has report_data_json: %s", 'report_data_json' in context)

            # Check if report_data in context has charts.features
            ctx_report_data = context.get('report_data', {})
            if isinstance(ctx_report_data, dict):
                ctx_charts = ctx_report_data.get('charts', {})
                logger.info("[FEATURE_IMPACT_DEBUG] Context report_data.charts.features exists: %s", 'features' in ctx_charts)
                if 'features' in ctx_charts:
                    logger.info("[FEATURE_IMPACT_DEBUG] Context report_data.charts.features has data: %s traces",
                               len(ctx_charts['features'].get('data', [])))

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
