"""
Simple renderer for fairness reports.
Uses Plotly for visualizations and single-page template approach.
"""

import os
import json
import logging
from typing import Dict, Any

logger = logging.getLogger("deepbridge.reports")

# Import CSS Manager
from ..css_manager import CSSManager


class FairnessRendererSimple:
    """
    Simple renderer for fairness experiment reports.
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
        self.template_manager = template_manager
        self.asset_manager = asset_manager

        # Initialize CSS Manager
        self.css_manager = CSSManager()

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

            # Load template
            template_path = self._find_template()
            logger.info(f"Using template: {template_path}")
            template = self.template_manager.load_template(template_path)

            # Get CSS content (inline)
            css_content = self._get_css_content()

            # Get JS content (inline) - minimal, just tab navigation
            js_content = self._get_js_content()

            # Prepare context for template
            context = {
                'model_name': report_data['model_name'],
                'model_type': report_data['model_type'],
                'report_title': 'Fairness Analysis Report',
                'report_subtitle': 'Model Bias and Fairness Assessment',

                # Data as JSON for JavaScript access
                'report_data_json': self._safe_json_dumps(report_data),

                # CSS and JS inline
                'css_content': css_content,
                'js_content': js_content,

                # Summary for display
                'overall_fairness_score': report_data['summary']['overall_fairness_score'],
                'total_warnings': report_data['summary']['total_warnings'],
                'total_critical': report_data['summary']['total_critical'],
                'total_attributes': report_data['summary']['total_attributes'],
                'assessment': report_data['summary']['assessment'],
                'config': report_data['summary']['config'],

                # Protected attributes
                'protected_attributes': report_data['protected_attributes'],

                # Issues
                'warnings': report_data['issues']['warnings'],
                'critical_issues': report_data['issues']['critical'],

                # Metadata
                'has_threshold_analysis': report_data['metadata']['has_threshold_analysis'],
                'has_confusion_matrix': report_data['metadata']['has_confusion_matrix'],

                # Charts
                'charts': report_data['charts']
            }

            # Render template
            html_content = self.template_manager.render_template(template, context)

            # Ensure output directory exists
            output_dir = os.path.dirname(file_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                logger.info(f"Output directory ensured: {output_dir}")

            # Write to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            logger.info(f"Fairness report saved to: {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"Error generating fairness report: {str(e)}")
            raise

    def _find_template(self) -> str:
        """Find the template file."""
        # Try simple template first
        template_path = os.path.join(
            self.template_manager.templates_dir,
            "report_types/fairness/interactive/index_simple.html"
        )

        if not os.path.exists(template_path):
            # Try alternative path
            template_path = os.path.join(
                self.template_manager.templates_dir,
                "report_fairness.html"
            )

        if not os.path.exists(template_path):
            raise FileNotFoundError(
                f"Fairness template not found at: {template_path}"
            )

        return template_path

    def _get_css_content(self) -> str:
        """Get CSS content (inline)."""
        try:
            # Get compiled CSS (base + components + fairness custom)
            compiled_css = self.css_manager.get_compiled_css('fairness')

            # Return the compiled CSS (fairness_custom.css already has all necessary styles)
            return compiled_css

        except Exception as e:
            logger.warning(f"Error loading CSS: {str(e)}")
            return ""

    def _get_js_content(self) -> str:
        """Get JavaScript content (inline)."""
        js_content = """
// Tab navigation
function openTab(evt, tabName) {
    var i, tabcontent, tablinks;

    tabcontent = document.getElementsByClassName("tab-content");
    for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
    }

    tablinks = document.getElementsByClassName("tab-link");
    for (i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" active", "");
    }

    document.getElementById(tabName).style.display = "block";
    evt.currentTarget.className += " active";
}

// Initialize first tab
document.addEventListener('DOMContentLoaded', function() {
    var firstTab = document.querySelector('.tab-link');
    if (firstTab) {
        firstTab.click();
    }
});
"""
        return js_content

    def _safe_json_dumps(self, data: Any) -> str:
        """Safely serialize data to JSON."""
        try:
            return json.dumps(data, default=str, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error serializing data to JSON: {str(e)}")
            return '{}'
