"""
Simple renderer for fairness reports.
Uses Plotly for visualizations and single-page template approach.

Refactored in Phase 2 to use BaseRenderer template methods.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger("deepbridge.reports")

# Import BaseRenderer
from .base_renderer import BaseRenderer


class FairnessRendererSimple(BaseRenderer):
    """
    Simple renderer for fairness experiment reports.
    Inherits from BaseRenderer to use common template methods (Phase 2).
    """

    def __init__(self, template_manager, asset_manager):
        """
        Initialize the fairness renderer.

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
        from ..transformers.fairness_simple import FairnessDataTransformerSimple
        self.data_transformer = FairnessDataTransformerSimple()

    def render(self, results: Dict[str, Any], file_path: str, model_name: str = "Model",
               report_type: str = "interactive", save_chart: bool = False) -> str:
        """
        Render fairness report from results data.

        Parameters:
        -----------
        results : Dict[str, Any]
            Fairness experiment results from FairnessSuite containing:
            - protected_attributes: List of protected attributes
            - pretrain_metrics: Pre-training fairness metrics
            - posttrain_metrics: Post-training fairness metrics
            - confusion_matrix: Confusion matrices by group
            - threshold_analysis: Threshold analysis results
            - warnings: List of warnings
            - critical_issues: List of critical issues
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
        logger.info(f"Generating fairness report to: {file_path}")
        logger.info(f"Report type: {report_type}")

        try:
            # Transform the data
            report_data = self.data_transformer.transform(results, model_name=model_name)

            # Add custom filters to Jinja2 environment (fairness-specific)
            if hasattr(self.template_manager, 'jinja_env'):
                self.template_manager.jinja_env.filters['format_number'] = self._format_number

            # Load template using BaseRenderer method
            template = self._load_template('fairness', report_type)
            logger.info(f"Template loaded for fairness/{report_type}")

            # Get all assets using BaseRenderer method
            assets = self._get_assets('fairness')

            # Create base context using BaseRenderer method
            context = self._create_base_context(report_data, 'fairness', assets)

            # Add fairness-specific context fields
            context.update({
                'report_title': 'Fairness Analysis Report',
                'report_subtitle': 'Model Bias and Fairness Assessment',
                'overall_fairness_score': report_data['summary']['overall_fairness_score'],
                'total_warnings': report_data['summary']['total_warnings'],
                'total_critical': report_data['summary']['total_critical'],
                'total_attributes': report_data['summary']['total_attributes'],
                'assessment': report_data['summary']['assessment'],
                'config': report_data['summary']['config'],
                'protected_attributes': report_data['protected_attributes'],
                'warnings': report_data['issues']['warnings'],
                'critical_issues': report_data['issues']['critical'],
                'has_threshold_analysis': report_data['metadata']['has_threshold_analysis'],
                'has_confusion_matrix': report_data['metadata']['has_confusion_matrix'],
                'charts': report_data['charts']
            })

            # Add optional fields if available
            if 'dataset_info' in report_data:
                context['dataset_info'] = report_data['dataset_info']
            if 'test_config' in report_data:
                context['test_config'] = report_data['test_config']

            # Render template using BaseRenderer method
            html_content = self._render_template(template, context)

            # Write HTML using BaseRenderer method
            logger.info(f"Fairness report generated and saved to: {file_path} (type: {report_type})")
            return self._write_html(html_content, file_path)

        except Exception as e:
            logger.error(f"Error generating fairness report: {str(e)}")
            raise

    # NOTE: Most helper methods (_load_template, _get_assets, _get_css_content,
    # _get_js_content, _safe_json_dumps, _write_html, _render_template,
    # _create_base_context) are now inherited from BaseRenderer (Phase 2 refactoring).
    # This eliminates ~105 lines of duplicate code!

    @staticmethod
    def _format_number(value):
        """
        Custom Jinja2 filter: Format number with thousands separator.
        This is fairness-specific and registered in render() method.
        """
        try:
            return f"{int(value):,}"
        except (ValueError, TypeError):
            return value
