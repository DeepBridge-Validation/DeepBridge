"""
Simple renderer for resilience reports - Following distillation pattern.
Uses Plotly for visualizations and single-page template approach.
"""

import os
import json
import logging
from typing import Dict, Any

logger = logging.getLogger("deepbridge.reports")

# Import CSS Manager
from ..css_manager import CSSManager


class ResilienceRendererSimple:
    """
    Simple renderer for resilience experiment reports.
    Follows the distillation renderer pattern for consistency.
    """

    def __init__(self, template_manager, asset_manager):
        """
        Initialize the resilience renderer.

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
        from ..transformers.resilience_simple import ResilienceDataTransformerSimple
        self.data_transformer = ResilienceDataTransformerSimple()

    def render(self, results: Dict[str, Any], file_path: str, model_name: str = "Model",
               report_type: str = "interactive", save_chart: bool = False) -> str:
        """
        Render resilience report from results data.

        Parameters:
        -----------
        results : Dict[str, Any]
            Resilience experiment results containing:
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
        logger.info(f"Generating SIMPLE resilience report to: {file_path}")
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
                'report_title': 'Resilience Analysis Report',
                'report_subtitle': 'Distribution Shift and Model Resilience',

                # Data as JSON for JavaScript access
                'report_data_json': self._safe_json_dumps(report_data),

                # CSS and JS inline
                'css_content': css_content,
                'js_content': js_content,

                # Summary for display
                'resilience_score': report_data['summary']['resilience_score'],
                'total_scenarios': report_data['summary']['total_scenarios'],
                'valid_scenarios': report_data['summary']['valid_scenarios'],
                'total_features': report_data['features']['total'],

                # Report type
                'report_type': report_type
            }

            # Render template
            html_content = template.render(context)

            # Ensure output directory exists
            output_dir = os.path.dirname(file_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                logger.info(f"Output directory ensured: {output_dir}")

            # Write to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            logger.info(f"Report saved to: {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"Error generating resilience report: {e}", exc_info=True)
            raise

    def _find_template(self) -> str:
        """Find the template file."""
        # Try multiple possible locations
        template_paths = [
            os.path.join(self.template_manager.templates_dir, "report_types/resilience/interactive/index_simple.html"),
            os.path.join(self.template_manager.templates_dir, "resilience/interactive/index_simple.html"),
            os.path.join(self.template_manager.templates_dir, "report_types/resilience/interactive/index.html"),
        ]

        for path in template_paths:
            if os.path.exists(path):
                return path

        raise FileNotFoundError(f"No template found for resilience report in: {template_paths}")

    def _get_css_content(self) -> str:
        """
        Get CSS content using CSSManager for resilience report.

        Returns:
        --------
        str : Compiled CSS (base + components + custom)
        """
        try:
            # Use CSSManager to compile CSS layers
            compiled_css = self.css_manager.get_compiled_css('resilience')
            logger.info(f"CSS compiled successfully using CSSManager: {len(compiled_css)} chars")
            return compiled_css
        except Exception as e:
            logger.error(f"Error loading CSS with CSSManager: {str(e)}")

            # Fallback: return minimal CSS if CSSManager fails
            logger.warning("Using fallback minimal CSS")
            return """
            :root {
                --primary-color: #1b78de;
                --secondary-color: #2c3e50;
            }
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                background-color: #f8f9fa;
                margin: 0;
                padding: 20px;
            }
            """

    def _get_js_content(self) -> str:
        """Get minimal JS for tab navigation."""
        js = """
        // Simple tab navigation
        function initTabs() {
            const tabButtons = document.querySelectorAll('.tab-button');
            tabButtons.forEach(button => {
                button.addEventListener('click', () => {
                    const tabId = button.getAttribute('data-tab');
                    showTab(tabId);
                });
            });
        }

        function showTab(tabId) {
            // Hide all tab contents
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });

            // Remove active from all buttons
            document.querySelectorAll('.tab-button').forEach(button => {
                button.classList.remove('active');
            });

            // Show selected tab
            const selectedTab = document.getElementById(tabId);
            if (selectedTab) {
                selectedTab.classList.add('active');
            }

            // Activate button
            const selectedButton = document.querySelector(`[data-tab="${tabId}"]`);
            if (selectedButton) {
                selectedButton.classList.add('active');
            }
        }
        """
        return js

    def _safe_json_dumps(self, data: Dict) -> str:
        """Safely convert data to JSON string."""
        return json.dumps(data, ensure_ascii=False, indent=None, separators=(',', ':'))
