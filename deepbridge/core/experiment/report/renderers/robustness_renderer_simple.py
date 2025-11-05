"""
Simple renderer for robustness reports - Following resilience/uncertainty pattern.
Uses Plotly for visualizations and single-page template approach.

Refactored in Phase 2 to use BaseRenderer template methods.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger("deepbridge.reports")

# Import BaseRenderer
from .base_renderer import BaseRenderer


class RobustnessRendererSimple(BaseRenderer):
    """
    Simple renderer for robustness experiment reports.
    Inherits from BaseRenderer to use common template methods (Phase 2).
    """

    def __init__(self, template_manager, asset_manager):
        """
        Initialize the robustness renderer.

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
        from ..transformers.robustness_simple import RobustnessDataTransformerSimple
        self.data_transformer = RobustnessDataTransformerSimple()

    def render(self, results: Dict[str, Any], file_path: str, model_name: str = "Model",
               report_type: str = "interactive", save_chart: bool = False) -> str:
        """
        Render robustness report from results data.

        Parameters:
        -----------
        results : Dict[str, Any]
            Robustness experiment results containing:
            - test_results: Test results with primary_model data
            - initial_model_evaluation: Initial evaluation
        file_path : str
            Path where the HTML report will be saved
        model_name : str, optional
            Name for the report title
        report_type : str, optional
            Type of report to generate ('interactive' or 'static')
        save_chart : bool, optional
            Whether to save charts as separate files (not used in simple renderer)

        Returns:
        --------
        str : Path to the generated report

        Raises:
        -------
        FileNotFoundError: If template not found
        ValueError: If required data missing
        """
        logger.info(f"Generating SIMPLE robustness report to: {file_path}")
        logger.info(f"Report type: {report_type}")

        try:
            # Transform the data
            report_data = self.data_transformer.transform(results, model_name=model_name)

            # Load template using BaseRenderer method
            template = self._load_template('robustness', report_type)
            logger.info(f"Template loaded for robustness/{report_type}")

            # Get all assets using BaseRenderer method
            assets = self._get_assets('robustness')

            # Create base context using BaseRenderer method
            context = self._create_base_context(report_data, 'robustness', assets)

            # Add robustness-specific context fields
            context.update({
                'report_title': 'Robustness Analysis Report',
                'report_subtitle': 'Model Stability and Perturbation Resistance',
                'robustness_score': report_data['summary']['robustness_score'],
                'base_score': report_data['summary']['base_score'],
                'avg_impact': report_data['summary']['avg_overall_impact'],
                'metric': report_data['summary']['metric'],
                'total_levels': report_data['metadata']['total_levels'],
                'total_features': report_data['metadata']['total_features'],

                # Advanced robustness tests (WeakSpot and Overfitting)
                'has_weakspot_analysis': 'weakspot_analysis' in results,
                'weakspot_analysis': results.get('weakspot_analysis', {}),
                'weakspot_analysis_json': self._safe_json_dumps(results.get('weakspot_analysis', {})),
                'has_overfitting_analysis': 'overfitting_analysis' in results,
                'overfitting_analysis': results.get('overfitting_analysis', {}),
                'overfitting_analysis_json': self._safe_json_dumps(results.get('overfitting_analysis', {}))
            })

            # Render template using BaseRenderer method
            html_content = self._render_template(template, context)

            # Write HTML using BaseRenderer method
            logger.info(f"Report generated and saved to: {file_path} (type: {report_type})")
            return self._write_html(html_content, file_path)

        except Exception as e:
            logger.error(f"Error generating robustness report: {str(e)}")
            raise

    # NOTE: All helper methods (_load_template, _get_assets, _get_css_content,
    # _get_js_content, _safe_json_dumps, _write_html, _render_template,
    # _create_base_context) are now inherited from BaseRenderer (Phase 2 refactoring).
    # This eliminates ~130 lines of duplicate code!
