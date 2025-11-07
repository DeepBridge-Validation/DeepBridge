"""
Resilience report renderer with enhanced JavaScript handling.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List

# Configure logger
logger = logging.getLogger("deepbridge.reports")

# Import CSS Manager
from ..css_manager import CSSManager

class ResilienceRenderer:
    """
    Renderer for resilience test reports.
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
        from ..transformers.resilience import ResilienceDataTransformer
        self.data_transformer = ResilienceDataTransformer()
    
    def _process_js_content(self, content: str, file_path: str) -> str:
        """
        Process JavaScript content to remove ES6 module syntax and fix common issues.
        
        Parameters:
        -----------
        content : str
            JavaScript file content
        file_path : str
            Path of the file (for logging)
            
        Returns:
        --------
        str : Processed JavaScript content
        """
        try:
            lines = content.split('\n')
            processed_lines = []
            
            for line in lines:
                line_stripped = line.strip()
                
                # Skip import and export statements
                if line_stripped.startswith('import ') and ' from ' in line_stripped:
                    continue
                if line_stripped.startswith('export ') or line_stripped == 'export default' or line_stripped.startswith('export default '):
                    continue
                
                # Convert 'const X =' to 'window.X =' for controllers and managers
                if line_stripped.startswith('const ') and any(pattern in file_path for pattern in ['Controller', 'Manager']):
                    component_name = line_stripped.split('const ')[1].split(' =')[0].strip()
                    if component_name.endswith('Controller') or component_name.endswith('Manager'):
                        line = line.replace(f'const {component_name}', f'window.{component_name}')
                
                # Ensure quotes don't get escaped
                line = line.replace('\\"', '"').replace('\\\'', '\'')
                
                processed_lines.append(line)
            
            processed_content = '\n'.join(processed_lines)
            
            # Replace HTML entities with actual characters
            processed_content = processed_content.replace('&quot;', '"')
            processed_content = processed_content.replace('&#34;', '"')
            processed_content = processed_content.replace('&#39;', '\'')
            processed_content = processed_content.replace('&amp;', '&')
            processed_content = processed_content.replace('&lt;', '<')
            processed_content = processed_content.replace('&gt;', '>')
            
            return processed_content
        except Exception as e:
            logger.error(f"Error processing JS file {file_path}: {str(e)}")
            return f"// Error processing {file_path}: {str(e)}\n" + content
            
    def render(self, results: Dict[str, Any], file_path: str, model_name: str = "Model", report_type: str = "interactive", save_chart: bool = False) -> str:
        """
        Render resilience report from results data.

        Parameters:
        -----------
        results : Dict[str, Any]
            Resilience test results
        file_path : str
            Path where the HTML report will be saved
        model_name : str, optional
            Name of the model for display in the report
        report_type : str, optional
            Type of report to generate ('interactive' or 'static')

        Returns:
        --------
        str : Path to the generated report

        Raises:
        -------
        FileNotFoundError: If template or assets not found
        ValueError: If required data missing
        """
        logger.info(f"Generating resilience report to: {file_path}")
        
        try:
            # Find template
            template_paths = self.template_manager.get_template_paths("resilience")
            template_path = self.template_manager.find_template(template_paths)
            
            if not template_path:
                raise FileNotFoundError(f"No template found for resilience report in: {template_paths}")
            
            logger.info(f"Using template: {template_path}")

            # Get CSS and JS content using combined methods
            css_content = self._load_css_content()
            js_content = self._load_js_content()

            # Load the template
            template = self.template_manager.load_template(template_path)
            
            # Transform the data
            report_data = self.data_transformer.transform(results, model_name)
            
            # Calculate additional metrics for the report
            avg_dist_shift = None
            max_gap = None
            most_affected_scenario = None
            
            # Calculate average distribution shift
            if 'distribution_shift' in report_data and 'by_distance_metric' in report_data['distribution_shift']:
                dist_values = []
                for dm, dm_data in report_data['distribution_shift']['by_distance_metric'].items():
                    for feature, val in dm_data.get('avg_feature_distances', {}).items():
                        dist_values.append(val)
                if dist_values:
                    avg_dist_shift = sum(dist_values) / len(dist_values)
            
            # Find the worst scenario (largest performance gap)
            if 'distribution_shift' in report_data and 'all_results' in report_data['distribution_shift']:
                all_results = report_data['distribution_shift']['all_results']
                if all_results:
                    # Find result with max performance gap
                    max_result = max(all_results, key=lambda x: x.get('performance_gap', 0) 
                                    if isinstance(x.get('performance_gap', 0), (int, float)) else 0)
                    max_gap = max_result.get('performance_gap', 0)
                    # Create a descriptive scenario name
                    scenario_components = []
                    if 'alpha' in max_result:
                        scenario_components.append(f"{int(max_result['alpha'] * 100)}% shift")
                    if 'distance_metric' in max_result:
                        scenario_components.append(f"{max_result['distance_metric']} metric")
                    if scenario_components:
                        most_affected_scenario = " with ".join(scenario_components)
                    else:
                        most_affected_scenario = "Unspecified scenario"
            
            # Calculate outlier sensitivity based on available data
            outlier_sensitivity = None
            if 'distribution_shift' in report_data and 'all_results' in report_data['distribution_shift']:
                sensitivity_values = []
                for result in report_data['distribution_shift']['all_results']:
                    if 'worst_metric' in result and 'remaining_metric' in result and 'alpha' in result:
                        # Sensitivity is how much performance changes per percentage shift
                        # Check if performance_gap is not None before calculating
                        if result.get('performance_gap') is not None and result.get('alpha') is not None:
                            sensitivity = abs(result['performance_gap']) / (result['alpha'] * 100)
                            sensitivity_values.append(sensitivity)
                if sensitivity_values:
                    outlier_sensitivity = sum(sensitivity_values) / len(sensitivity_values)
            
            # Get baseline and target dataset names
            baseline_dataset = None
            target_dataset = None
            if 'dataset_info' in report_data:
                if 'baseline_name' in report_data['dataset_info']:
                    baseline_dataset = report_data['dataset_info']['baseline_name']
                if 'target_name' in report_data['dataset_info']:
                    target_dataset = report_data['dataset_info']['target_name']
            
            # Extract shift scenarios from test results
            shift_scenarios = []
            if 'distribution_shift' in report_data and 'all_results' in report_data['distribution_shift']:
                for result in report_data['distribution_shift']['all_results']:
                    # Handle NaN values - convert to None for JSON serialization
                    import math

                    worst_metric = result.get('worst_metric', 0)
                    if isinstance(worst_metric, float) and math.isnan(worst_metric):
                        worst_metric = None

                    remaining_metric = result.get('remaining_metric', 0)
                    if isinstance(remaining_metric, float) and math.isnan(remaining_metric):
                        remaining_metric = None

                    performance_gap = result.get('performance_gap', 0)
                    if isinstance(performance_gap, float) and math.isnan(performance_gap):
                        performance_gap = None

                    scenario = {
                        'name': result.get('name', f"Scenario {len(shift_scenarios) + 1}"),
                        'alpha': result.get('alpha', 0),
                        'metric': result.get('metric', 'unknown'),
                        'distance_metric': result.get('distance_metric', 'unknown'),
                        'performance_gap': performance_gap,
                        'baseline_performance': worst_metric,  # worst_metric is the baseline
                        'target_performance': remaining_metric,  # remaining_metric is the target
                        'worst_metric': worst_metric,  # Keep original field names too
                        'remaining_metric': remaining_metric,
                        'metrics': result.get('metrics', {})
                    }
                    shift_scenarios.append(scenario)
            
            # Extract sensitive features based on feature distances
            sensitive_features = []
            if 'distribution_shift' in report_data and 'by_distance_metric' in report_data['distribution_shift']:
                all_features = {}
                for dm, dm_data in report_data['distribution_shift']['by_distance_metric'].items():
                    for feature, value in dm_data.get('top_features', {}).items():
                        if feature not in all_features:
                            all_features[feature] = 0
                        all_features[feature] += value
                
                # Get top features across all distance metrics
                sensitive_features = [
                    {'name': feature, 'impact': value}
                    for feature, value in sorted(all_features.items(), key=lambda x: x[1], reverse=True)[:5]
                ]
            
            # Create template context
            context = self.base_renderer._create_context(report_data, "resilience", css_content, js_content, report_type)
            
            # Add resilience-specific context with default values for all variables
            resilience_score = report_data.get('resilience_score', 0)
            avg_performance_gap = report_data.get('avg_performance_gap', 0)
            
            # Prepare report data for JavaScript
            report_data_json = {
                'model_name': model_name,
                'model_type': report_data.get('model_type', 'Unknown'),
                'resilience_score': resilience_score,
                'avg_performance_gap': avg_performance_gap,
                'base_score': report_data.get('base_score', 0),
                'metric': report_data.get('metric', 'accuracy'),
                'feature_importance': report_data.get('feature_importance', {}),
                'model_feature_importance': report_data.get('model_feature_importance', {}),
                'features': report_data.get('features', []),
                'distance_metrics': report_data.get('distance_metrics', []),
                'alphas': report_data.get('alphas', []),
                'feature_subset': report_data.get('feature_subset', []),
                'shift_scenarios': shift_scenarios,
                'sensitive_features': sensitive_features,
                'baseline_dataset': baseline_dataset or "Baseline",
                'target_dataset': target_dataset or "Target",
                'timestamp': report_data.get('timestamp', ''),
                'avg_dist_shift': avg_dist_shift or 0,
                'boxplot_data': report_data.get('boxplot_data', {})
            }
            
            context.update({
                # Core metrics with defaults
                'resilience_score': resilience_score,
                'robustness_score': resilience_score,  # Backward compatibility
                'avg_performance_gap': avg_performance_gap,
                'performance_gap': avg_performance_gap,  # Alternative name
                
                # Additional calculated metrics with defaults
                'avg_dist_shift': avg_dist_shift or 0,
                'outlier_sensitivity': outlier_sensitivity or 0,
                'max_gap': max_gap or 0,
                'most_affected_scenario': most_affected_scenario or "No scenario data",
                
                # Lists and collections with empty defaults
                'distance_metrics': report_data.get('distance_metrics', []),
                'distribution_shift_results': report_data.get('distribution_shift_results', []),
                'alphas': report_data.get('alphas', []),
                'baseline_dataset': baseline_dataset or "Baseline",
                'target_dataset': target_dataset or "Target",
                'shift_scenarios': shift_scenarios or [],
                'sensitive_features': sensitive_features or [],
                
                # Metadata
                'resilience_module_version': report_data.get('module_version', '1.0'),
                'test_type': 'resilience',  # Explicit test type
                'report_type': 'resilience',  # Required for template includes

                # Additional context to ensure backward compatibility
                'features': report_data.get('features', []),
                'metrics': report_data.get('metrics', {}),
                'metrics_details': report_data.get('metrics_details', {}),
                'has_feature_data': bool(sensitive_features),

                # JSON representation for JavaScript config
                'report_data_json': json.dumps(report_data_json)
            })
            
            # Render the template
            rendered_html = self.template_manager.render_template(template, context)

            # Write the report to file
            return self.base_renderer._write_report(rendered_html, file_path)

        except Exception as e:
            logger.error(f"Error generating resilience report: {str(e)}")
            raise ValueError(f"Failed to generate resilience report: {str(e)}")

    def _load_css_content(self) -> str:
        """
        Load and combine CSS files for the resilience report using CSSManager.

        Returns:
        --------
        str : Combined CSS content (base + components + custom)
        """
        try:
            # Use CSSManager to compile CSS (base + components + custom)
            compiled_css = self.css_manager.get_compiled_css('resilience')
            logger.info(f"CSS compiled successfully using CSSManager: {len(compiled_css)} chars")
            return compiled_css
        except Exception as e:
            logger.error(f"Error loading CSS with CSSManager: {str(e)}")

            # Fallback: try to load CSS from asset manager if CSSManager fails
            try:
                logger.warning("Falling back to asset_manager for CSS loading")
                css_content = self.asset_manager.get_combined_css_content("resilience")

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
        Load and combine JavaScript files for the resilience report.

        Returns:
        --------
        str : Combined JavaScript content
        """
        try:
            # Get combined JS content (generic + test-specific)
            js_content = self.asset_manager.get_combined_js_content("resilience")

            # Add initialization code to ensure proper tab navigation and chart loading
            init_js = """
            /**
             * Resilience Report Initialization
             */
            (function() {
                console.log("Resilience report JavaScript loaded");

                // Setup tab navigation when DOM is ready
                document.addEventListener('DOMContentLoaded', function() {
                    console.log("DOM loaded, initializing resilience report");

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