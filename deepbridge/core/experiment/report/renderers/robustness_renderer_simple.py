"""
Simple renderer for robustness reports - Following resilience/uncertainty pattern.
Uses Plotly for visualizations and single-page template approach.
"""

import os
import json
import logging
from typing import Dict, Any

logger = logging.getLogger("deepbridge.reports")

# Import CSS Manager
from ..css_manager import CSSManager


class RobustnessRendererSimple:
    """
    Simple renderer for robustness experiment reports.
    Follows the resilience/uncertainty renderer pattern for consistency.
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
        self.template_manager = template_manager
        self.asset_manager = asset_manager

        # Initialize CSS Manager
        self.css_manager = CSSManager()

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
                'report_title': 'Robustness Analysis Report',
                'report_subtitle': 'Model Stability and Perturbation Resistance',

                # Data as JSON for JavaScript access
                'report_data_json': self._safe_json_dumps(report_data),

                # CSS and JS inline
                'css_content': css_content,
                'js_content': js_content,

                # Summary for display
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

            logger.info(f"Report saved to: {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"Error generating robustness report: {str(e)}")
            raise

    def _find_template(self) -> str:
        """Find the template file."""
        # Try simple template first
        template_path = os.path.join(
            self.template_manager.templates_dir,
            "report_types/robustness/interactive/index_simple.html"
        )

        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template not found: {template_path}")

        return template_path

    def _get_css_content(self) -> str:
        """
        Get CSS content using CSSManager for robustness report.

        Returns:
        --------
        str : Compiled CSS (base + components + custom)
        """
        try:
            # Use CSSManager to compile CSS layers
            compiled_css = self.css_manager.get_compiled_css('robustness')
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

            // Show first tab by default
            if (tabButtons.length > 0) {
                const firstTabId = tabButtons[0].getAttribute('data-tab');
                showTab(firstTabId);
            }
        }

        function showTab(tabId) {
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });

            // Remove active from all buttons
            document.querySelectorAll('.tab-button').forEach(button => {
                button.classList.remove('active');
            });

            // Show selected tab
            const selectedContent = document.getElementById(tabId);
            if (selectedContent) {
                selectedContent.classList.add('active');
            }

            // Activate selected button
            const selectedButton = document.querySelector(`[data-tab="${tabId}"]`);
            if (selectedButton) {
                selectedButton.classList.add('active');
            }
        }

        // Initialize on page load
        document.addEventListener('DOMContentLoaded', function() {
            console.log('Initializing robustness report...');
            initTabs();
            console.log('Report initialized successfully');
        });
        """
        return js

    def _safe_json_dumps(self, data: Any) -> str:
        """Safely convert data to JSON string."""
        try:
            return json.dumps(data, default=str, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error converting data to JSON: {e}")
            return "{}"
