"""
Simple renderer for uncertainty reports - Following resilience pattern.
Uses Plotly for visualizations and single-page template approach.
"""

import os
import json
import logging
from typing import Dict, Any

logger = logging.getLogger("deepbridge.reports")

# Import CSS Manager
from ..css_manager import CSSManager


class UncertaintyRendererSimple:
    """
    Simple renderer for uncertainty experiment reports.
    Follows the resilience renderer pattern for consistency.
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
        self.template_manager = template_manager
        self.asset_manager = asset_manager

        # Initialize CSS Manager
        self.css_manager = CSSManager()

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
                'report_title': 'Uncertainty Analysis Report',
                'report_subtitle': 'Conformal Prediction and Calibration',

                # Data as JSON for JavaScript access
                'report_data_json': self._safe_json_dumps(report_data),

                # CSS and JS inline
                'css_content': css_content,
                'js_content': js_content,

                # Summary for display
                'uncertainty_score': report_data['summary']['uncertainty_score'],
                'total_alphas': report_data['summary']['total_alphas'],
                'total_features': report_data['features']['total'],
                'avg_coverage': report_data['summary']['avg_coverage'],
                'avg_coverage_error': report_data['summary']['avg_coverage_error'],
                'avg_width': report_data['summary']['avg_width']
            }

            # Render template
            html_content = self.template_manager.render_template(template, context)

            # Ensure output directory exists
            output_dir = os.path.dirname(file_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                logger.info(f"Output directory ensured: {output_dir}")

            # Write to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            logger.info(f"Report saved to: {file_path}")
            logger.info(f"Report generated and saved to: {file_path} (type: {report_type})")

            return file_path

        except Exception as e:
            logger.error(f"Error generating uncertainty report: {str(e)}")
            raise ValueError(f"Failed to generate uncertainty report: {str(e)}")

    def _find_template(self) -> str:
        """Find the simple template."""
        template_path = os.path.join(
            self.template_manager.templates_dir,
            'report_types',
            'uncertainty',
            'interactive',
            'index_simple.html'
        )

        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template not found: {template_path}")

        return template_path

    def _get_css_content(self) -> str:
        """
        Get CSS content using CSSManager for uncertainty report.

        Returns:
        --------
        str : Compiled CSS (base + components + custom)
        """
        try:
            # Use CSSManager to compile CSS layers
            compiled_css = self.css_manager.get_compiled_css('uncertainty')
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
        """Get inline JS content (minimal - just tab navigation)."""
        js = """
        // Simple tab navigation
        function initTabs() {
            const tabButtons = document.querySelectorAll('.tab-button');
            const tabContents = document.querySelectorAll('.tab-content');

            tabButtons.forEach(button => {
                button.addEventListener('click', () => {
                    const targetTab = button.dataset.tab;

                    // Deactivate all
                    tabButtons.forEach(btn => btn.classList.remove('active'));
                    tabContents.forEach(content => content.classList.remove('active'));

                    // Activate target
                    button.classList.add('active');
                    document.getElementById(targetTab).classList.add('active');
                });
            });
        }

        // Initialize on load
        document.addEventListener('DOMContentLoaded', function() {
            console.log('Initializing uncertainty report...');
            console.log('Report data:', window.reportData);

            // Initialize tabs
            initTabs();

            // Render charts if data available
            if (window.reportData && window.reportData.charts) {
                renderCharts(window.reportData.charts);
            }
        });

        function renderCharts(charts) {
            // Render all charts
            for (const [chartName, chartData] of Object.entries(charts)) {
                const elementId = 'chart-' + chartName.replace(/_/g, '-');
                const element = document.getElementById(elementId);

                if (element && chartData.data && chartData.data.length > 0) {
                    // Ensure layout is responsive
                    const layout = {...chartData.layout};
                    layout.autosize = true;
                    delete layout.width; // Remove fixed width if exists

                    // Render with responsive config
                    Plotly.newPlot(element, chartData.data, layout, {
                        responsive: true,
                        displayModeBar: false
                    }).then(() => {
                        // Force resize on window resize
                        window.addEventListener('resize', () => {
                            Plotly.Plots.resize(element);
                        });
                    });
                }
            }
        }
        """
        return js

    def _safe_json_dumps(self, data: Dict) -> str:
        """Safely serialize data to JSON, handling NaN and infinity."""
        def default_handler(obj):
            if isinstance(obj, float):
                if str(obj) == 'nan':
                    return None
                elif str(obj) == 'inf':
                    return None
                elif str(obj) == '-inf':
                    return None
            return str(obj)

        return json.dumps(data, default=default_handler, ensure_ascii=False)
