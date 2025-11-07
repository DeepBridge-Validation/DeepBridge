"""
Uncertainty report renderer.
"""

import os
import logging
from typing import Dict, Any, Optional, List

# Configure logger
logger = logging.getLogger("deepbridge.reports")

# Import CSS Manager
from ..css_manager import CSSManager

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

        # Initialize CSS Manager
        self.css_manager = CSSManager()

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

            # Get CSS and JS content using combined methods
            css_content = self._load_css_content()
            js_content = self._load_js_content()

            # Load the template
            template = self.template_manager.load_template(template_path)

            # Transform the data
            report_data = self.data_transformer.transform(results, model_name)

            # Create template context
            context = self.base_renderer._create_context(report_data, "uncertainty", css_content, js_content, report_type)

            # Add uncertainty-specific context directly from report_data
            context.update({
                'test_type': 'uncertainty',  # Explicit test type
                'report_type': 'uncertainty'  # Required for template includes
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

    def _load_css_content(self) -> str:
        """
        Load and combine CSS files for the uncertainty report using CSSManager.

        Returns:
        --------
        str : Combined CSS content (base + components + custom)
        """
        try:
            # Use CSSManager to compile CSS (base + components + custom)
            compiled_css = self.css_manager.get_compiled_css('uncertainty')
            logger.info(f"CSS compiled successfully using CSSManager: {len(compiled_css)} chars")
            return compiled_css
        except Exception as e:
            logger.error(f"Error loading CSS with CSSManager: {str(e)}")

            # Fallback: try to load CSS from asset manager if CSSManager fails
            try:
                logger.warning("Falling back to asset_manager for CSS loading")
                css_content = self.asset_manager.get_combined_css_content("uncertainty")

                # Add default styles to ensure report functionality even if external CSS is missing
                default_css = """
            /* Base variables and reset */
            :root {
                --primary-color: #1b78de;
                --secondary-color: #2c3e50;
                --success-color: #28a745;
                --danger-color: #dc3545;
                --warning-color: #f39c12;
                --info-color: #17a2b8;
                --light-color: #f8f9fa;
                --dark-color: #343a40;
                --text-color: #333;
                --text-muted: #6c757d;
                --border-color: #ddd;
                --background-color: #f8f9fa;
                --card-bg: #fff;
                --header-bg: #ffffff;
            }

            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }

            html, body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                font-size: 16px;
                line-height: 1.5;
                color: var(--text-color);
                background-color: var(--background-color);
            }

            h1, h2, h3, h4, h5, h6 {
                margin-bottom: 0.5rem;
                font-weight: 500;
                line-height: 1.2;
            }

            p {
                margin-bottom: 1rem;
            }

            /* Layout */
            .report-container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #fff;
            }

            .report-content {
                padding: 20px 0;
            }

            /* Tab navigation */
            .main-tabs {
                display: flex;
                border-bottom: 1px solid var(--border-color);
                margin-bottom: 1.5rem;
                overflow-x: auto;
                flex-wrap: nowrap;
            }

            .tab-btn {
                padding: 0.75rem 1.5rem;
                border: none;
                background: none;
                cursor: pointer;
                font-size: 1rem;
                font-weight: 500;
                color: var(--text-color);
                border-bottom: 2px solid transparent;
                white-space: nowrap;
            }

            .tab-btn:hover {
                color: var(--primary-color);
            }

            .tab-btn.active {
                color: var(--primary-color);
                border-bottom: 2px solid var(--primary-color);
            }

            .tab-content {
                display: none;
            }

            .tab-content.active {
                display: block;
            }

            /* Chart containers */
            .chart-container {
                margin: 1.5rem 0;
                min-height: 300px;
            }

            .chart-plot {
                min-height: 300px;
                background-color: #fff;
                border-radius: 8px;
                border: 1px solid var(--border-color, #ddd);
                margin-bottom: 1.5rem;
            }

            /* Loading indicators */
            .chart-loading-message {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                padding: 30px;
                text-align: center;
            }

            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #3498db;
                border-radius: 50%;
                width: 30px;
                height: 30px;
                animation: spin 1s linear infinite;
                margin-bottom: 10px;
            }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }

            /* Table styles */
            .table-container {
                margin-top: 1.5rem;
                overflow-x: auto;
            }

            .data-table {
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 1.5rem;
            }

            .data-table th,
            .data-table td {
                padding: 0.75rem;
                text-align: left;
                border: 1px solid var(--border-color);
            }

            .data-table th {
                background-color: #f8f9fa;
                font-weight: 600;
            }

            /* Boxplot specific styles */
            .boxplot-table .text-danger {
                color: #d32f2f;
            }

            .boxplot-table .text-warning {
                color: #f0ad4e;
            }

            .boxplot-table .text-success {
                color: #5cb85c;
            }

            .loading-info {
                text-align: center;
                padding: 20px;
            }

            .loading-info .loading-icon {
                font-size: 24px;
                margin-bottom: 10px;
            }

            /* Section styles */
            .section {
                margin-bottom: 2rem;
            }

            .section-title {
                border-left: 4px solid var(--primary-color);
                padding-left: 0.75rem;
                margin-bottom: 1rem;
            }
            """

                # Combine default CSS with loaded CSS
                combined_css = default_css + "\n\n" + css_content
                return combined_css
            except Exception as fallback_error:
                logger.error(f"Fallback CSS loading also failed: {str(fallback_error)}")
                return ""

    def _load_js_content(self) -> str:
        """
        Load and combine JavaScript files for the uncertainty report.

        Returns:
        --------
        str : Combined JavaScript content
        """
        try:
            # Get combined JS content (generic + test-specific)
            js_content = self.asset_manager.get_combined_js_content("uncertainty")

            # Add initialization code to ensure proper tab navigation and chart loading
            init_js = """
            /**
             * Uncertainty Report Initialization
             */
            (function() {
                console.log("Uncertainty report JavaScript loaded");

                // Setup tab navigation when DOM is ready
                document.addEventListener('DOMContentLoaded', function() {
                    console.log("DOM loaded, initializing uncertainty report");

                    // Initialize tab navigation
                    const tabButtons = document.querySelectorAll('.tab-btn');
                    tabButtons.forEach(button => {
                        button.addEventListener('click', function() {
                            // Remove active class from all buttons and content
                            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
                            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

                            // Add active class to clicked button
                            this.classList.add('active');

                            // Show corresponding content
                            const targetTab = this.getAttribute('data-tab');
                            const targetElement = document.getElementById(targetTab);
                            if (targetElement) {
                                targetElement.classList.add('active');
                            }
                        });
                    });

                    // Activate first tab by default
                    if (tabButtons.length > 0 && !document.querySelector('.tab-btn.active')) {
                        tabButtons[0].click();
                    }
                });
            })();
            """

            # Combine all JS
            combined_js = init_js + "\n\n" + js_content
            return combined_js
        except Exception as e:
            logger.error(f"Error loading JavaScript: {str(e)}")
            return ""