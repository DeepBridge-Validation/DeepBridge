"""
Hyperparameter report renderer.

Refactored in Phase 2 to use BaseRenderer template methods.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger("deepbridge.reports")

# Import BaseRenderer
from .base_renderer import BaseRenderer


class HyperparameterRenderer(BaseRenderer):
    """
    Renderer for hyperparameter test reports.
    Inherits from BaseRenderer to use common template methods (Phase 2).
    """

    def __init__(self, template_manager, asset_manager):
        """
        Initialize the hyperparameter renderer.

        Parameters:
        -----------
        template_manager : TemplateManager
            Manager for templates
        asset_manager : AssetManager
            Manager for assets (CSS, JS, images)
        """
        # Call parent constructor (initializes css_manager, etc.)
        super().__init__(template_manager, asset_manager)

        # Import specific data transformer
        from ..transformers.hyperparameter import HyperparameterDataTransformer
        self.data_transformer = HyperparameterDataTransformer()

    def render(self, results: Dict[str, Any], file_path: str, model_name: str = "Model",
               report_type: str = "interactive", save_chart: bool = False) -> str:
        """
        Render hyperparameter report from results data.

        Parameters:
        -----------
        results : Dict[str, Any]
            Hyperparameter test results
        file_path : str
            Path where the HTML report will be saved
        model_name : str, optional
            Name of the model for display in the report
        report_type : str, optional
            Type of report to generate ('interactive' or 'static')
        save_chart : bool, optional
            Whether to save charts as separate files (kept for compatibility)

        Returns:
        --------
        str : Path to the generated report

        Raises:
        -------
        FileNotFoundError: If template or assets not found
        ValueError: If required data missing
        """
        logger.info(f"Generating hyperparameter report to: {file_path}")
        logger.info(f"Report type: {report_type}")

        try:
            # Transform the data
            report_data = self.data_transformer.transform(results, model_name)

            # Load template using BaseRenderer method
            template = self._load_template('hyperparameter', report_type)
            logger.info(f"Template loaded for hyperparameter/{report_type}")

            # Get all assets using BaseRenderer method
            assets = self._get_assets('hyperparameter')

            # Create base context using BaseRenderer method
            context = self._create_base_context(report_data, 'hyperparameter', assets)

            # Add hyperparameter-specific context fields
            context.update({
                'report_title': 'Hyperparameter Tuning Report',
                'report_subtitle': 'Feature Importance and Optimization Results',
                'importance_scores': report_data.get('importance_scores', {}),
                'tuning_order': report_data.get('tuning_order', []),
                'importance_results': report_data.get('importance_results', []),
                'optimization_results': report_data.get('optimization_results', [])
            })

            # Render template using BaseRenderer method
            html_content = self._render_template(template, context)

            # Write HTML using BaseRenderer method
            logger.info(f"Report generated and saved to: {file_path} (type: {report_type})")
            return self._write_html(html_content, file_path)

        except Exception as e:
            logger.error(f"Error generating hyperparameter report: {str(e)}")
            raise ValueError(f"Failed to generate hyperparameter report: {str(e)}")

    # NOTE: All helper methods (_load_template, _get_assets, _get_css_content,
    # _get_js_content, _safe_json_dumps, _write_html, _render_template,
    # _create_base_context) are now inherited from BaseRenderer (Phase 2 refactoring).
    # This eliminates ~50 lines of duplicate code!
