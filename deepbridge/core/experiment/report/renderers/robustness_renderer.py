"""
Robustness report renderer with improved JavaScript loading.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List

# Configure logger
logger = logging.getLogger("deepbridge.reports")

# Import JSON formatter
from ..utils.json_formatter import JsonFormatter

# Import CSS Manager
from ..css_manager import CSSManager

class RobustnessRenderer:
    """
    Renderer for robustness test reports.
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

        # Import data transformers
        from ..transformers.robustness import RobustnessDataTransformer
        from ..transformers.initial_results import InitialResultsTransformer
        self.data_transformer = RobustnessDataTransformer()
        self.initial_results_transformer = InitialResultsTransformer()
    
    def render(self, results: Dict[str, Any], file_path: str, model_name: str = "Model", report_type: str = "interactive", save_chart: bool = False) -> str:
        """
        Render robustness report from results data.

        Parameters:
        -----------
        results : Dict[str, Any]
            Robustness test results
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
        logger.info(f"Generating robustness report to: {file_path}")
        
        try:
            # Find template
            template_paths = self.template_manager.get_template_paths("robustness")
            template_path = self.template_manager.find_template(template_paths)
            
            if not template_path:
                raise FileNotFoundError(f"No template found for robustness report in: {template_paths}")
            
            logger.info(f"Using template: {template_path}")
            
            # Get CSS and JS content for the new modular structure
            css_content = self._load_css_content()
            js_content = self._load_js_content()
            
            # Load the template
            template = self.template_manager.load_template(template_path)
            
            # Transform the robustness data
            report_data = self.data_transformer.transform(results, model_name)
            
            # Transform initial results data if available or use defaults
            logger.info("Checking for initial results data")
            initial_results = {}
            if 'initial_results' in results:
                logger.info("Found initial_results in results, transforming...")
                initial_results = self.initial_results_transformer.transform(results.get('initial_results', {}))
                logger.info(f"Initial results transformed for {len(initial_results.get('models', {}))} models")
            else:
                # If no initial_results, still transform to get default data structure
                logger.info("No initial_results found, using transformer default")
                initial_results = self.initial_results_transformer.transform({})
                logger.info(f"Using default initial results with {len(initial_results.get('models', {}))} models")
            
            # Prepare data for charts (new in modular structure)
            chart_data = self._prepare_chart_data(report_data)
            
            # Add initial results to chart data
            if initial_results:
                chart_data['initial_results'] = initial_results
                
                # Add metrics for radar chart
                models_metrics = {}
                if 'models' in initial_results:
                    for model_id, model_data in initial_results['models'].items():
                        if 'metrics' in model_data:
                            models_metrics[model_id] = {
                                'name': model_data.get('name', model_id),
                                'type': model_data.get('type', 'Unknown'),
                                'metrics': model_data.get('metrics', {})
                            }
                
                chart_data['radar_chart_data'] = {
                    'models': models_metrics
                }
            
            # Create template context
            context = self.base_renderer._create_context(report_data, "robustness", css_content, js_content, report_type)
            
            # Add initial_results directly to the report_data for client-side access
            if initial_results:
                report_data['initial_results'] = initial_results

                # Explicitly log for debugging
                logger.info(f"Added initial_results to report_data with {len(initial_results.get('models', {}))} models")

            # Add advanced robustness tests (WeakSpot and Overfitting) if available
            weakspot_analysis = None
            overfitting_analysis = None

            if 'weakspot_analysis' in results:
                weakspot_analysis = results['weakspot_analysis']
                report_data['weakspot_analysis'] = weakspot_analysis
                logger.info(f"Added weakspot_analysis with {weakspot_analysis.get('summary', {}).get('total_weakspots', 0)} weakspots")

            if 'overfitting_analysis' in results:
                overfitting_analysis = results['overfitting_analysis']
                report_data['overfitting_analysis'] = overfitting_analysis
                logger.info(f"Added overfitting_analysis with data")

            # Add robustness-specific context with default values for all variables
            robustness_score = report_data.get('robustness_score', 0)
            
            # Calculate additional metadata for summary section
            features = report_data.get('features', [])
            feature_count = len(features) if features else 0
            
            # Get perturbation levels count for summary
            perturbation_levels = []
            if 'raw' in report_data and 'by_level' in report_data['raw']:
                perturbation_levels = list(report_data['raw']['by_level'].keys())
            intensity_levels = len(perturbation_levels)
            
            # Get metrics count
            metrics = report_data.get('metrics', {})
            metrics_count = len(metrics) if metrics else 0
            
            # Estimate test sample count (if available)
            test_sample_count = report_data.get('test_sample_count', 0)
            if not test_sample_count and 'raw' in report_data:
                # Try to estimate from the data
                if 'sample_count' in report_data['raw']:
                    test_sample_count = report_data['raw']['sample_count']
                elif 'by_level' in report_data['raw'] and perturbation_levels:
                    # Take the first level's run data if available
                    first_level = perturbation_levels[0]
                    if 'runs' in report_data['raw']['by_level'][first_level]:
                        test_sample_count = report_data['raw']['by_level'][first_level].get('sample_count', 0)
            
            # Get critical features (top features by importance)
            feature_importance = report_data.get('feature_importance', {})
            critical_features = 0
            if feature_importance:
                # Count features with importance above threshold (e.g., 0.05)
                critical_features = sum(1 for imp in feature_importance.values() if imp > 0.05)
            
            context.update({
                # Core metrics with defaults
                'robustness_score': robustness_score,
                'resilience_score': robustness_score,  # Backward compatibility
                'raw_impact': report_data.get('raw_impact', 0),
                'quantile_impact': report_data.get('quantile_impact', 0),

                # Feature importance data
                'feature_importance': report_data.get('feature_importance', {}),
                'model_feature_importance': report_data.get('model_feature_importance', {}),
                'has_feature_importance': bool(report_data.get('feature_importance', {})),
                'has_model_feature_importance': bool(report_data.get('model_feature_importance', {})),

                # Test metadata
                'iterations': report_data.get('n_iterations', 100),
                'chart_data_json': self._sanitize_json(chart_data),  # Use new sanitizer
                'test_type': 'robustness',  # Explicit test type
                'report_type': 'robustness',  # Required for template includes

                # Initial results data (if available)
                'has_initial_results': bool(initial_results),
                'initial_results': initial_results,

                # Advanced robustness tests (WeakSpot and Overfitting Analysis)
                'has_weakspot_analysis': bool(weakspot_analysis),
                'weakspot_analysis': weakspot_analysis or {},
                'weakspot_analysis_json': self._sanitize_json(weakspot_analysis) if weakspot_analysis else '{}',
                'has_overfitting_analysis': bool(overfitting_analysis),
                'overfitting_analysis': overfitting_analysis or {},
                'overfitting_analysis_json': self._sanitize_json(overfitting_analysis) if overfitting_analysis else '{}',

                # Additional context to ensure backward compatibility
                'features': features,
                'metrics': metrics,
                'metrics_details': report_data.get('metrics_details', {}),
                'feature_subset': report_data.get('feature_subset', []),

                # Summary section metadata
                'feature_count': feature_count,
                'test_sample_count': test_sample_count,
                'perturbation_count': report_data.get('n_iterations', 100),
                'intensity_levels': intensity_levels,
                'metrics_count': metrics_count,
                'critical_features': critical_features,
                'framework': report_data.get('framework', 'Scikit-learn')  # Default to Scikit-learn if not specified
            })
            
            # Log debug info for troubleshooting
            logger.info(f"robustness_score: {report_data.get('robustness_score')}")
            if 'alternative_models' in report_data:
                for name, model_data in report_data['alternative_models'].items():
                    logger.info(f"Alternative model {name} robustness_score: {model_data.get('robustness_score')}")
            
            # Ensure template partials directory exists
            try:
                template_partials_dir = os.path.join(
                    self.template_manager.templates_dir, 
                    "report_types", "robustness", "partials"
                )
                if not os.path.exists(template_partials_dir):
                    os.makedirs(template_partials_dir, exist_ok=True)
                    logger.info(f"Created partials directory: {template_partials_dir}")
            except Exception as e:
                logger.warning(f"Could not create partials directory: {str(e)}")
            
            # Create importance_comparison_fixed.html if it doesn't exist
            importance_comparison_path = os.path.join(
                template_partials_dir, "importance_comparison_fixed.html"
            )
            if not os.path.exists(importance_comparison_path):
                try:
                    # Simple template content for importance comparison
                    importance_comparison_content = """<div class="section">
    <h2 class="section-title">Feature Importance Comparison</h2>
    <p>Comparison between model-defined feature importance and robustness-based feature importance.</p>
    
    <div class="chart-container">
        <div id="importance-comparison-chart-plot" class="chart-plot" style="min-height: 500px; width: 100%;"></div>
    </div>
    
    <div class="table-container">
        <h3>Importance Scores Comparison</h3>
        <table class="data-table">
            <thead>
                <tr>
                    <th>Feature</th>
                    <th>Model Importance</th>
                    <th>Robustness Impact</th>
                    <th>Difference</th>
                </tr>
            </thead>
            <tbody id="importance-comparison-data">
                <!-- Will be populated by JavaScript -->
            </tbody>
        </table>
    </div>
</div>"""
                    with open(importance_comparison_path, 'w') as f:
                        f.write(importance_comparison_content)
                    logger.info(f"Created importance comparison template: {importance_comparison_path}")
                except Exception as e:
                    logger.warning(f"Could not create importance comparison template: {str(e)}")
            
            # Before rendering, prepare inline JS for boxplot visualization
            # to avoid external file dependencies
            boxplot_js = self._get_inline_boxplot_js()
            context['inline_boxplot_js'] = boxplot_js

            # Render the template
            rendered_html = self.template_manager.render_template(template, context)

            # Write the report to both files - original destination and standardized location
            self.base_renderer._write_report(rendered_html, file_path)

            # Also write to a standard template location
            templates_dir = self.template_manager.templates_dir
            standard_report_path = os.path.join(templates_dir, "report_robustness.html")

            logger.info(f"Also writing report to standard path: {standard_report_path}")
            return self.base_renderer._write_report(rendered_html, standard_report_path)
            
        except Exception as e:
            logger.error(f"Error generating robustness report: {str(e)}")
            raise ValueError(f"Failed to generate robustness report: {str(e)}")
    
    def _load_css_content(self) -> str:
        """
        Load and combine CSS files for the robustness report using CSSManager.

        Returns:
        --------
        str : Combined CSS content (base + components + custom)
        """
        try:
            # Use CSSManager to compile CSS (base + components + custom)
            compiled_css = self.css_manager.get_compiled_css('robustness')
            logger.info(f"CSS compiled successfully using CSSManager: {len(compiled_css)} chars")
            return compiled_css
        except Exception as e:
            logger.error(f"Error loading CSS with CSSManager: {str(e)}")

            # Fallback: try to load CSS from asset manager if CSSManager fails
            try:
                logger.warning("Falling back to asset_manager for CSS loading")
                css_content = self.asset_manager.get_combined_css_content("robustness")
                return css_content
            except Exception as fallback_error:
                logger.error(f"Fallback CSS loading also failed: {str(fallback_error)}")
                return ""
    
    def _load_js_content(self) -> str:
        """
        Load and combine JavaScript files for the robustness report.

        Returns:
        --------
        str : Combined JavaScript content
        """
        try:
            # Define basic JavaScript utilities for robust operation
            core_js = """
            /**
             * Global ChartUtils object for chart helpers
             * Defined once at the top level to avoid duplicate declaration errors
             */
            (function() {
                // Only define ChartUtils if it doesn't already exist
                if (typeof window.ChartUtils === 'undefined') {
                    window.ChartUtils = {
                        getModelColor: function(modelName, index) {
                            // Predefined colors for known models
                            const modelColors = {
                                'Primary Model': 'rgba(31, 119, 180, 0.7)',
                                'primary_model': 'rgba(31, 119, 180, 0.7)',
                                'GLM_CLASSIFIER': 'rgba(255, 127, 14, 0.7)',
                                'GAM_CLASSIFIER': 'rgba(44, 160, 44, 0.7)',
                                'GBM': 'rgba(214, 39, 40, 0.7)',
                                'XGB': 'rgba(148, 103, 189, 0.7)',
                                'RANDOM_FOREST': 'rgba(140, 86, 75, 0.7)',
                                'SVM': 'rgba(227, 119, 194, 0.7)',
                                'NEURAL_NETWORK': 'rgba(127, 127, 127, 0.7)'
                            };

                    // Return predefined color if available
                    if (modelColors[modelName]) {
                        return modelColors[modelName];
                    }

                    // Generate deterministic color based on model name or index
                    const hash = index || Array.from(modelName).reduce((hash, char) => {
                        return ((hash << 5) - hash) + char.charCodeAt(0);
                    }, 0);

                    const r = Math.abs(hash) % 200 + 55;
                    const g = Math.abs(hash * 31) % 200 + 55;
                    const b = Math.abs(hash * 17) % 200 + 55;

                    return `rgba(${r}, ${g}, ${b}, 0.7)`;
                },

                // Format numbers for display
                formatNumber: function(value, decimals = 4) {
                    if (value === null || value === undefined || isNaN(value)) {
                        return 'N/A';
                    }
                    return Number(value).toFixed(decimals);
                },

                // Get report data safely
                getReportData: function() {
                    return window.reportData || {};
                },

                // Get chart data safely
                getChartData: function() {
                    return window.chartData || {};
                }
            };
                } else {
                    console.log("ChartUtils already exists, skipping initialization");
                }
            })();

            /**
             * Global error handler to catch syntax errors
             */
            window.onerror = function(message, source, lineno, colno, error) {
                console.error("Caught JavaScript error:", {
                    message,
                    source,
                    lineno,
                    colno
                });

                // Check for common errors and try to handle them gracefully
                if (message && (
                    message.includes("Illegal continue") ||
                    message.includes("no surrounding iteration statement") ||
                    message.includes("Unexpected token") ||
                    message.includes("has already been declared")
                )) {
                    console.warn("Known JavaScript error intercepted:", message);
                    return true; // Prevent default error handling
                }

                return false; // Let other errors propagate
            };

            /**
             * Basic navigation and tab functionality
             */
            function setupTabNavigation() {
                const tabButtons = document.querySelectorAll('.tab-btn');
                if (tabButtons.length === 0) {
                    console.warn("No tab buttons found");
                    return;
                }

                tabButtons.forEach(button => {
                    button.addEventListener('click', function() {
                        // Remove active class from all buttons
                        tabButtons.forEach(btn => btn.classList.remove('active'));

                        // Add active class to clicked button
                        this.classList.add('active');

                        // Hide all tab contents
                        document.querySelectorAll('.tab-content').forEach(content => {
                            content.classList.remove('active');
                        });

                        // Show target tab content
                        const targetTab = this.getAttribute('data-tab');
                        const targetElement = document.getElementById(targetTab);

                        if (targetElement) {
                            targetElement.classList.add('active');

                            // Initialize boxplot when boxplot tab is shown
                            if (targetTab === 'boxplot' && window.BoxplotChartManager) {
                                window.BoxplotChartManager.initializeBoxplotChart('boxplot-chart-container');
                            }
                        } else {
                            console.error(`Tab content not found: #${targetTab}`);
                        }
                    });
                });

                // Activate first tab by default if none is active
                if (!document.querySelector('.tab-btn.active')) {
                    tabButtons[0]?.click();
                }
            }

            // Initialize on DOM ready
            document.addEventListener('DOMContentLoaded', function() {
                console.log("Report initialized");

                // Set up tab navigation
                setupTabNavigation();

                // Fix trailing commas in JSON data
                if (window.reportData && window.reportData.chart_data_json) {
                    try {
                        // Already handled in the index.html
                        console.log("Chart data JSON cleaning handled at load time");
                    } catch (e) {
                        console.error("Error handling chart data:", e);
                    }
                }
            });
            """

            # Use the asset manager's combined JS content method
            js_content = self.asset_manager.get_combined_js_content("robustness")

            # Add syntax fixer to handle JavaScript errors safely
            syntax_fixer = """
            /**
             * Helper function to fix common JavaScript errors
             */
            function fixCommonJsErrors() {
                console.log("Fixing common JavaScript errors");

                // Fix missing script references
                // Instead of trying to load external scripts, define empty objects for them
                if (typeof window.FixedSyntax === 'undefined') {
                    window.FixedSyntax = { initialized: true };
                }

                if (typeof window.ModelChartFix === 'undefined') {
                    window.ModelChartFix = { initialized: true };
                }

                if (typeof window.SafeChartManager === 'undefined') {
                    window.SafeChartManager = { initialized: true };
                }

                // Make sure all required controllers exist
                const requiredControllers = [
                    'BoxplotController',
                    'FeatureImportanceController',
                    'DetailsController',
                    'OverviewController'
                ];

                for (const controllerName of requiredControllers) {
                    if (typeof window[controllerName] === 'undefined') {
                        window[controllerName] = {
                            init: function() {
                                console.log(`${controllerName} initialized with fallback`);
                            }
                        };
                    }
                }
            }

            // Run the fixer when the DOM is ready
            document.addEventListener('DOMContentLoaded', fixCommonJsErrors);
            """

            # Ensure specific controllers for details tab are loaded
            details_controller_path = os.path.join(
                self.asset_manager.assets_dir,
                "report_types", "robustness", "js", "controllers", "DetailsController.js"
            )

            details_charts_path = os.path.join(
                self.asset_manager.assets_dir,
                "report_types", "robustness", "js", "charts", "details.js"
            )

            # Add specific content if files exist
            additional_content = ""

            # Check if the files exist
            if os.path.exists(details_controller_path):
                try:
                    with open(details_controller_path, 'r') as file:
                        details_controller_content = file.read()
                        # Process to remove any ES6 imports/exports
                        details_controller_content = self._process_js_content(details_controller_content, details_controller_path)
                        additional_content += f"\n// Details Controller\n{details_controller_content}\n"
                        logger.info(f"Added details controller from {details_controller_path}")
                except Exception as file_error:
                    logger.error(f"Error reading details controller: {str(file_error)}")

            if os.path.exists(details_charts_path):
                try:
                    with open(details_charts_path, 'r') as file:
                        details_charts_content = file.read()
                        # Process to remove any ES6 imports/exports
                        details_charts_content = self._process_js_content(details_charts_content, details_charts_path)
                        additional_content += f"\n// Details Charts\n{details_charts_content}\n"
                        logger.info(f"Added details charts from {details_charts_path}")
                except Exception as file_error:
                    logger.error(f"Error reading details charts: {str(file_error)}")

            # Add the error interceptor script
            error_interceptor_path = os.path.join(
                self.asset_manager.assets_dir,
                "report_types", "robustness", "js", "error_interceptor.js"
            )

            error_interceptor_content = ""
            if os.path.exists(error_interceptor_path):
                try:
                    with open(error_interceptor_path, 'r') as file:
                        error_interceptor_content = file.read()
                        logger.info(f"Added error interceptor from {error_interceptor_path}")
                except Exception as e:
                    logger.error(f"Error reading error interceptor: {str(e)}")

            # Combine all JS content - making sure error interceptor runs first
            combined_js = f"{error_interceptor_content}\n\n{core_js}\n\n{js_content}\n\n{additional_content}\n\n{syntax_fixer}"
            return combined_js
        except Exception as e:
            logger.error(f"Error loading JavaScript: {str(e)}")
            return "// Error loading JavaScript files\n"
    
    def _sanitize_json(self, data: Any) -> str:
        """
        Sanitize data to ensure valid JSON for JavaScript.
        
        Parameters:
        -----------
        data : Any
            Data to sanitize and convert to JSON
            
        Returns:
        --------
        str : Sanitized JSON string
        """
        try:
            # Fix missing iteration scores for boxplot
            if isinstance(data, dict):
                # Ensure boxplot data always has scores array
                if 'boxplot_data' in data and 'models' in data['boxplot_data']:
                    for model in data['boxplot_data']['models']:
                        if 'scores' not in model or model['scores'] is None:
                            model['scores'] = []
                        elif not isinstance(model['scores'], list):
                            # Convert to list if not already a list
                            model['scores'] = [model['scores']]
                
                # Create a safe version of iteration_data if missing
                if 'iterations_by_level' not in data or not data['iterations_by_level']:
                    data['iterations_by_level'] = {}
                    # If we have perturbation levels, create empty arrays for them
                    if 'perturbation_levels' in data and data['perturbation_levels']:
                        for level in data['perturbation_levels']:
                            data['iterations_by_level'][str(level)] = []
                # Ensure keys are strings for iterations_by_level
                elif isinstance(data['iterations_by_level'], dict):
                    # Convert any numeric keys to string keys for JavaScript compatibility
                    string_keys = {}
                    for k, v in data['iterations_by_level'].items():
                        string_keys[str(k)] = v
                    data['iterations_by_level'] = string_keys
                
                # Ensure alternative_models_iterations is initialized if missing
                if 'alternative_models_iterations' not in data:
                    data['alternative_models_iterations'] = {}
                    # If we have alternative models, create empty data structures for them
                    if 'alternative_models' in data and data['alternative_models']:
                        for model_name in data['alternative_models'].keys():
                            data['alternative_models_iterations'][model_name] = {}
                            if 'perturbation_levels' in data and data['perturbation_levels']:
                                for level in data['perturbation_levels']:
                                    data['alternative_models_iterations'][model_name][str(level)] = []
                # If we have alt_iterations_by_level but not alternative_models_iterations
                elif 'alt_iterations_by_level' in data and isinstance(data['alt_iterations_by_level'], dict):
                    # Copy from alt_iterations_by_level to standard format
                    data['alternative_models_iterations'] = data['alt_iterations_by_level']
                    # Ensure all keys are strings
                    for model_name, levels in data['alternative_models_iterations'].items():
                        if isinstance(levels, dict):
                            string_level_keys = {}
                            for k, v in levels.items():
                                string_level_keys[str(k)] = v
                            data['alternative_models_iterations'][model_name] = string_level_keys
                
                # Ensure the most current data about iteration scores is in the boxplot models
                if 'boxplot_data' in data and 'models' in data['boxplot_data'] and len(data['boxplot_data']['models']) > 0:
                    for model in data['boxplot_data']['models']:
                        if 'scores' not in model or not model['scores'] or len(model['scores']) == 0:
                            logger.warning(f"Model {model.get('name', 'unknown')} has no scores in boxplot data")

                            # If this is the primary model, try to get scores from iterations_by_level
                            if model.get('name') == data.get('model_name') and 'iterations_by_level' in data:
                                all_scores = []
                                for level_scores in data['iterations_by_level'].values():
                                    if isinstance(level_scores, list) and level_scores:
                                        all_scores.extend(level_scores)

                                if all_scores:
                                    logger.info(f"Found {len(all_scores)} scores in iterations_by_level for primary model")
                                    model['scores'] = all_scores

                            # If this is an alternative model, try to get scores from alternative_models_iterations
                            elif 'alternative_models_iterations' in data:
                                model_name = model.get('name')
                                if model_name in data['alternative_models_iterations']:
                                    all_scores = []
                                    for level_scores in data['alternative_models_iterations'][model_name].values():
                                        if isinstance(level_scores, list) and level_scores:
                                            all_scores.extend(level_scores)

                                    if all_scores:
                                        logger.info(f"Found {len(all_scores)} scores in alternative_models_iterations for {model_name}")
                                        model['scores'] = all_scores

                # Check if we have alternative models data in initial_results but not in boxplot data
                if ('boxplot_data' in data and
                    'models' in data['boxplot_data'] and
                    'initial_results' in data and
                    'models' in data['initial_results']):

                    # Get all existing model names in boxplot data
                    existing_model_names = set(model.get('name') for model in data['boxplot_data']['models'])

                    # Check initial_results for additional models
                    for model_id, model_data in data['initial_results']['models'].items():
                        model_name = model_data.get('name', model_id)

                        # Skip if model is already in boxplot data
                        if model_name in existing_model_names or model_data.get('is_primary', False):
                            continue

                        # Create a new boxplot model entry
                        scores = []
                        if 'evaluation_results' in model_data and 'scores' in model_data['evaluation_results']:
                            if isinstance(model_data['evaluation_results']['scores'], list):
                                scores = model_data['evaluation_results']['scores']
                            else:
                                # Convert to list if not already a list
                                scores = [model_data['evaluation_results']['scores']]

                        # Add the new model to boxplot data
                        if not scores:
                            logger.warning(f"No scores found in initial_results for model {model_name}")
                            scores = []

                        new_model = {
                            'name': model_name,
                            'modelType': model_data.get('type', 'Unknown'),
                            'baseScore': model_data.get('base_score', 0.0),
                            'scores': scores
                        }

                        logger.info(f"Adding model {model_name} from initial_results to boxplot data with {len(scores)} scores")
                        data['boxplot_data']['models'].append(new_model)
            
            # Use JSON formatter with additional safety
            sanitized = JsonFormatter.format_for_javascript(data)
            
            # Additional safety check for trailing commas
            sanitized = sanitized.replace(',}', '}').replace(',]', ']').replace(',)', ')')
            
            return sanitized
        except Exception as e:
            logger.error(f"Error sanitizing JSON data: {str(e)}")
            return "{}"  # Return empty object as fallback
            
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

    def _get_inline_boxplot_js(self) -> str:
        """
        Get inline JavaScript code for the boxplot visualization.

        Returns:
        --------
        str : Boxplot JavaScript code
        """
        # Add error handler to fix common issues
        error_handler = """
/**
 * Error handler for boxplot visualization
 * Ensures we handle possible JavaScript errors
 */
(function() {
    // Prevent duplicate declarations by checking if objects already exist
    if (typeof window.errorHandlerInstalled === 'boolean' && window.errorHandlerInstalled) {
        console.log("Error handler already installed");
        return;
    }

    // Mark as installed to avoid duplicate execution
    window.errorHandlerInstalled = true;

    // Create placeholders for scripts that might be referenced
    window.FixedSyntax = window.FixedSyntax || { initialized: true };
    window.ModelChartFix = window.ModelChartFix || { initialized: true };
    window.SafeChartManager = window.SafeChartManager || { initialized: true };

    // Override syntax fixer functions to prevent errors
    window.fixTrailingCommas = window.fixTrailingCommas || function() {
        console.log("Using safe fixTrailingCommas");
    };

    window.runFixes = window.runFixes || function() {
        console.log("Using safe runFixes");
    };

    // Add a try-catch wrapper to any function
    window.safeExec = function(fn, fallback) {
        return function() {
            try {
                return fn.apply(this, arguments);
            } catch (error) {
                console.error("Error executing function:", error);
                return typeof fallback === 'function' ? fallback.apply(this, arguments) : undefined;
            }
        };
    };

    console.log("Boxplot error handler installed");
})();
"""

        # This is the core JS needed for boxplot functionality
        return error_handler + """
/**
 * Direct boxplot initialization to avoid duplicate declarations
 */
(function() {
    // Check if BoxplotChartManager already exists to avoid redefinition
    if (typeof window.BoxplotChartManager !== 'undefined') {
        console.log("BoxplotChartManager already exists, using existing implementation");
        return;
    }

    // Define BoxplotChartManager if it doesn't exist yet
    window.BoxplotChartManager = {
    /**
     * Initialize boxplot chart
     * @param {string} elementId - Chart container ID
     */
    initializeBoxplotChart: function(elementId) {
        console.log("BoxplotChartManager initializing boxplot chart in:", elementId);

        const container = document.getElementById(elementId);
        if (!container) {
            console.error("Chart container not found:", elementId);
            return;
        }

        try {
            // Extract data for boxplot
            const chartData = this.extractBoxplotData();

            if (!chartData || !chartData.models || chartData.models.length === 0 ||
                !chartData.models.some(m => m.scores && m.scores.length > 0)) {
                console.error("No valid chart data available");
                this.showNoDataMessage(container, "Dados de boxplot não disponíveis. Execute testes com iterações múltiplas para visualizar a distribuição dos scores.");
                return;
            }

            // Create the Plotly boxplot visualization
            this.createPlotlyBoxplot(container, chartData);

        } catch (error) {
            console.error("Error creating boxplot chart:", error);
            this.showErrorMessage(container, error.message);
        }
    },

    /**
     * Extract data for boxplot from report data
     * @returns {Object} Data for boxplot chart
     */
    extractBoxplotData: function() {
        try {
            // Get data from window.reportData or window.chartData
            const reportData = window.reportData || {};
            const chartData = window.chartData || {};

            // Check if we have pre-processed boxplot data
            if (chartData.boxplot_data && chartData.boxplot_data.models &&
                chartData.boxplot_data.models.length > 0) {
                console.log("Using pre-processed boxplot data");

                // Filter models to only include those with real scores
                const validModels = chartData.boxplot_data.models.filter(model =>
                    model.scores && model.scores.length > 0
                );

                if (validModels.length === 0) {
                    console.error("No models with valid scores found in boxplot_data");
                    return null;
                }

                return {
                    models: validModels,
                    metricName: reportData.metric || 'Score'
                };
            }

            // If no pre-processed data, try to extract from reportData
            if (reportData.boxplot_data && reportData.boxplot_data.models &&
                reportData.boxplot_data.models.length > 0) {
                console.log("Using reportData.boxplot_data");

                // Filter models to only include those with real scores
                const validModels = reportData.boxplot_data.models.filter(model =>
                    model.scores && model.scores.length > 0
                );

                if (validModels.length === 0) {
                    console.error("No models with valid scores found in reportData.boxplot_data");
                    return null;
                }

                return {
                    models: validModels,
                    metricName: reportData.metric || 'Score'
                };
            }

            // No pre-processed data, try to extract from raw data
            console.log("No pre-processed boxplot data, extracting from raw data");

            if (!reportData.raw || !reportData.raw.by_level) {
                console.error("No raw data available for boxplot extraction");
                return null;
            }

            // Extract primary model data
            const primaryModelData = {
                name: reportData.model_name || 'Primary Model',
                modelType: reportData.model_type || 'Unknown',
                baseScore: reportData.base_score || 0,
                scores: []
            };

            // Extract scores from perturbation levels
            Object.keys(reportData.raw.by_level).forEach(level => {
                const levelData = reportData.raw.by_level[level];

                if (levelData.runs && levelData.runs.all_features) {
                    levelData.runs.all_features.forEach(run => {
                        if (run.iterations && run.iterations.scores && run.iterations.scores.length > 0) {
                            primaryModelData.scores.push(...run.iterations.scores);
                        }
                    });
                }
            });

            // Check for iteration data in chartData
            if (primaryModelData.scores.length === 0 && chartData.iterations_by_level) {
                console.log("Trying to extract scores from chartData.iterations_by_level");
                Object.values(chartData.iterations_by_level).forEach(levelScores => {
                    if (Array.isArray(levelScores) && levelScores.length > 0) {
                        primaryModelData.scores.push(...levelScores);
                    }
                });
                console.log(`Extracted ${primaryModelData.scores.length} scores from chartData`);
            }

            if (primaryModelData.scores.length === 0) {
                console.log("No scores found for primary model, checking alternative sources");

                // Try to find scores in initial_results
                if (reportData.initial_results && reportData.initial_results.models) {
                    const primaryModel = Object.values(reportData.initial_results.models).find(
                        model => model.name === primaryModelData.name || model.is_primary
                    );

                    if (primaryModel && primaryModel.evaluation_results && primaryModel.evaluation_results.scores) {
                        console.log("Using scores from initial_results");
                        primaryModelData.scores = primaryModel.evaluation_results.scores;
                    }
                }
            }

            // Only proceed if we have real scores
            if (primaryModelData.scores.length === 0) {
                console.error("No real scores found for primary model");
            }

            const models = [primaryModelData];

            // Extract alternative models if available
            if (reportData.alternative_models) {
                Object.keys(reportData.alternative_models).forEach(modelName => {
                    const modelData = reportData.alternative_models[modelName];

                    const altModelData = {
                        name: modelName,
                        modelType: modelData.model_type || 'Unknown',
                        baseScore: modelData.base_score || 0,
                        scores: []
                    };

                    // Extract scores if raw data is available
                    if (modelData.raw && modelData.raw.by_level) {
                        Object.keys(modelData.raw.by_level).forEach(level => {
                            const levelData = modelData.raw.by_level[level];

                            if (levelData.runs && levelData.runs.all_features) {
                                levelData.runs.all_features.forEach(run => {
                                    if (run.iterations && run.iterations.scores && run.iterations.scores.length > 0) {
                                        altModelData.scores.push(...run.iterations.scores);
                                    }
                                });
                            }
                        });
                    }

                    // Check for iteration data in chartData
                    if (altModelData.scores.length === 0 &&
                        chartData.alternative_models_iterations &&
                        chartData.alternative_models_iterations[modelName]) {

                        console.log(`Trying to extract scores for ${modelName} from chartData.alternative_models_iterations`);
                        Object.values(chartData.alternative_models_iterations[modelName]).forEach(levelScores => {
                            if (Array.isArray(levelScores) && levelScores.length > 0) {
                                altModelData.scores.push(...levelScores);
                            }
                        });
                        console.log(`Extracted ${altModelData.scores.length} scores for ${modelName} from chartData`);
                    }

                    // Try to find data in initial_results
                    if (altModelData.scores.length === 0 &&
                        reportData.initial_results &&
                        reportData.initial_results.models) {

                        console.log(`Looking for ${modelName} scores in initial_results`);
                        // Find the model by name
                        const foundModel = Object.values(reportData.initial_results.models).find(
                            model => model.name === modelName
                        );

                        if (foundModel && foundModel.evaluation_results && foundModel.evaluation_results.scores) {
                            console.log(`Found ${foundModel.evaluation_results.scores.length} scores in initial_results for ${modelName}`);
                            altModelData.scores = foundModel.evaluation_results.scores;
                        }
                    }

                    // Only add if real scores are available
                    if (altModelData.scores.length === 0) {
                        console.error(`No real scores found for model ${modelName}`);
                    }

                    models.push(altModelData);
                });
            } else {
                console.log("No alternative models found in reportData");

                // Check for alternative models in initial_results
                if (reportData.initial_results && reportData.initial_results.models) {
                    const initialModels = reportData.initial_results.models;
                    console.log(`Found ${Object.keys(initialModels).length} models in initial_results`);

                    Object.values(initialModels).forEach(model => {
                        // Skip the primary model which we already processed
                        if (model.name === primaryModelData.name || model.is_primary) {
                            return;
                        }

                        console.log(`Processing initial_results model: ${model.name}`);

                        const altModelData = {
                            name: model.name,
                            modelType: model.type || 'Unknown',
                            baseScore: model.base_score || 0,
                            scores: []
                        };

                        if (model.evaluation_results && model.evaluation_results.scores) {
                            console.log(`Found ${model.evaluation_results.scores.length} scores for ${model.name}`);
                            altModelData.scores = model.evaluation_results.scores;
                        }

                        if (altModelData.scores.length === 0) {
                            console.error(`No scores found for model ${model.name} in initial_results`);
                        }

                        models.push(altModelData);
                    });
                }
            }

            // Ensure at least one model has valid scores
            if (!models.some(model => model.scores && model.scores.length > 0)) {
                console.error("No models with valid scores found");
                return null;
            }

            return {
                models,
                metricName: reportData.metric || 'Score'
            };

        } catch (error) {
            console.error("Error extracting boxplot data:", error);
            return null;
        }
    },

    /**
     * Create Plotly boxplot visualization
     * @param {HTMLElement} container - Chart container element
     * @param {Object} chartData - Data for chart
     */
    createPlotlyBoxplot: function(container, chartData) {
        if (typeof Plotly === 'undefined') {
            console.error("Plotly is not available");
            this.showErrorMessage(container, "Plotly library is not available. Charts cannot be displayed.");
            return;
        }

        const models = chartData.models;
        const traces = [];

        // Define consistent colors for models
        const modelColors = {
            'Primary Model': 'rgba(31, 119, 180, 0.7)',
            'primary_model': 'rgba(31, 119, 180, 0.7)',
            'GLM_CLASSIFIER': 'rgba(255, 127, 14, 0.7)',
            'GAM_CLASSIFIER': 'rgba(44, 160, 44, 0.7)',
            'GBM': 'rgba(214, 39, 40, 0.7)',
            'XGB': 'rgba(148, 103, 189, 0.7)',
            'RANDOM_FOREST': 'rgba(140, 86, 75, 0.7)',
            'SVM': 'rgba(227, 119, 194, 0.7)',
            'NEURAL_NETWORK': 'rgba(127, 127, 127, 0.7)'
        };

        // Track valid models
        let validModelCount = 0;

        // Create traces for each model
        models.forEach(model => {
            // Skip models without real data
            if (!model.scores || model.scores.length === 0) {
                return;
            }

            validModelCount++;

            // Clean up model name for display
            const displayName = model.name.replace(/_/g, ' ').trim();

            // Get color or generate a deterministic color
            let color = modelColors[model.name];
            if (!color) {
                // Generate a deterministic color based on model name
                const hash = Array.from(model.name).reduce((hash, char) => {
                    return ((hash << 5) - hash) + char.charCodeAt(0);
                }, 0);
                const r = Math.abs(hash) % 200 + 55;
                const g = Math.abs(hash * 31) % 200 + 55;
                const b = Math.abs(hash * 17) % 200 + 55;
                color = `rgba(${r}, ${g}, ${b}, 0.7)`;
            }

            // Create violin plot for model
            traces.push({
                type: 'violin',
                y: model.scores,
                x: Array(model.scores.length).fill(displayName),
                name: displayName,
                box: {
                    visible: true,
                    width: 0.6
                },
                meanline: {
                    visible: true
                },
                line: {
                    color: 'black',
                    width: 1
                },
                fillcolor: color,
                opacity: 0.7,
                points: 'all',
                jitter: 0.3,
                pointpos: 0,
                hoverinfo: 'y+x',
                spanmode: 'soft',
                width: 0.5,
                bandwidth: 0.2
            });
        });

        // Add base scores as separate markers
        const baseScoreTrace = {
            type: 'scatter',
            mode: 'markers',
            y: models.map(m => m.baseScore),
            x: models.map(m => m.name.replace(/_/g, ' ').trim()),
            name: 'Base Score',
            marker: {
                size: 12,
                symbol: 'diamond',
                color: models.map(m => modelColors[m.name] || 'rgba(31, 119, 180, 0.7)'),
                line: {
                    color: 'white',
                    width: 1
                }
            },
            text: models.map(m => `Base Score: ${m.baseScore.toFixed(4)}`),
            hoverinfo: 'text+y',
        };

        traces.push(baseScoreTrace);

        // If no valid models, show error
        if (validModelCount === 0) {
            this.showNoDataMessage(container, "Nenhum modelo possui dados reais para visualização");
            return;
        }

        // Get metric name
        const metricName = chartData.metricName || 'Score';

        // Create layout
        const layout = {
            title: {
                text: `Model Performance Distribution - ${metricName}`,
                font: { size: 20 }
            },
            xaxis: {
                title: 'Models',
                tickangle: 0,
                automargin: true,
            },
            yaxis: {
                title: metricName,
                zeroline: false,
                autorange: true,
                automargin: true
            },
            autosize: true,
            violinmode: 'group',
            hoverlabel: {
                bgcolor: "#FFF",
                font: { size: 12 },
                bordercolor: "#333"
            },
            showlegend: true,
            legend: {
                orientation: "h",
                yanchor: "top",
                y: 1.1,
                xanchor: "right",
                x: 1
            },
            hovermode: 'closest',
            margin: {
                l: 50,
                r: 20,
                t: 60,
                b: 80
            },
            annotations: [{
                xref: 'paper',
                yref: 'paper',
                x: 0,
                y: -0.15,
                text: 'The boxplots show model performance distribution under perturbation tests. Diamond markers indicate base scores.',
                showarrow: false,
                font: { size: 12 }
            }]
        };

        try {
            // Render the visualization
            Plotly.newPlot(container, traces, layout, {
                responsive: true,
                displayModeBar: true,
                modeBarButtonsToRemove: ['lasso2d', 'select2d'],
                displaylogo: false,
                staticPlot: false,
                toImageButtonOptions: {
                    format: 'png',
                    filename: 'model_comparison_boxplot',
                    height: 700,
                    width: 1000,
                    scale: 2
                }
            }).then(() => {
                console.log("Boxplot chart successfully rendered");

                // Force a resize event to ensure proper layout
                window.dispatchEvent(new Event('resize'));
            }).catch(error => {
                console.error("Plotly.newPlot failed:", error);
                this.showErrorMessage(container, `Error rendering boxplot: ${error.message}`);
            });
        } catch (error) {
            console.error("Exception during Plotly.newPlot:", error);
            this.showErrorMessage(container, `Error rendering boxplot: ${error.message}`);
        }
    },

    /**
     * Show no data message in container
     * @param {HTMLElement} element - Chart container element
     * @param {string} message - Message to display
     */
    showNoDataMessage: function(element, message) {
        element.innerHTML = `
            <div style="padding: 40px; text-align: center; background-color: #fff0f0; border-radius: 8px; margin: 20px auto; max-width: 600px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
                <div style="font-size: 48px; margin-bottom: 20px;">⚠️</div>
                <h3 style="font-size: 24px; font-weight: bold; margin-bottom: 10px; color: #d32f2f;">Dados não disponíveis</h3>
                <p style="color: #333; font-size: 16px; line-height: 1.4;">${message}</p>
                <p style="color: #333; margin-top: 20px; font-size: 14px;">
                    Não serão gerados dados sintéticos ou demonstrativos. Execute testes com iterações múltiplas (n_iterations > 1).
                </p>
            </div>`;
    },

    /**
     * Show error message in container
     * @param {HTMLElement} element - Chart container element
     * @param {string} errorMessage - Error message to display
     */
    showErrorMessage: function(element, errorMessage) {
        element.innerHTML = `
            <div style="padding: 40px; text-align: center; background-color: #fff0f0; border: 1px solid #ffcccc; border-radius: 8px; margin: 20px auto; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
                <div style="font-size: 48px; margin-bottom: 20px;">⚠️</div>
                <h3 style="font-size: 24px; font-weight: bold; margin-bottom: 10px; color: #cc0000;">Erro ao criar gráfico</h3>
                <p style="color: #666; font-size: 16px; line-height: 1.4;">${errorMessage}</p>
            </div>`;
    }
};

/**
 * Initialize boxplot on DOM ready
 */
document.addEventListener('DOMContentLoaded', function() {
    // Check if boxplot tab is active
    const boxplotTab = document.getElementById('boxplot');
    if (boxplotTab && boxplotTab.classList.contains('active')) {
        window.BoxplotChartManager.initializeBoxplotChart('boxplot-chart-container');
    }

    // Initialize when tab is clicked
    const boxplotTabButton = document.querySelector('[data-tab="boxplot"]');
    if (boxplotTabButton) {
        boxplotTabButton.addEventListener('click', function() {
            window.BoxplotChartManager.initializeBoxplotChart('boxplot-chart-container');
        });
    }

    /**
     * Handle missing external scripts that are referenced in syntax_fixer.js
     * Instead of trying to load these scripts, create placeholders
     */
    function handleMissingScripts() {
        // Create placeholders for scripts that might be referenced
        if (typeof window.FixedSyntax === 'undefined') {
            window.FixedSyntax = { initialized: true };
        }

        if (typeof window.ModelChartFix === 'undefined') {
            window.ModelChartFix = { initialized: true };
        }

        if (typeof window.SafeChartManager === 'undefined') {
            window.SafeChartManager = { initialized: true };
        }

        // Patch external script loading attempts
        const originalCreateElement = document.createElement;
        document.createElement = function(tagName) {
            const element = originalCreateElement.call(document, tagName);

            // If creating a script element, override its src setting to prevent loading external files
            if (tagName.toLowerCase() === 'script') {
                const originalSrcSetter = Object.getOwnPropertyDescriptor(HTMLScriptElement.prototype, 'src').set;

                Object.defineProperty(element, 'src', {
                    set: function(value) {
                        // Check if it's one of our problematic scripts
                        if (value && (
                            value.includes('fixed_syntax.js') ||
                            value.includes('safe_chart_manager.js') ||
                            value.includes('model_chart_fix.js')
                        )) {
                            console.log(`Prevented loading of external script: ${value}`);
                            // Don't actually set the src, but pretend we did
                            setTimeout(() => {
                                // Create a fake load event
                                const event = new Event('load');
                                element.dispatchEvent(event);
                            }, 10);
                            return;
                        }

                        // For all other scripts, use the original setter
                        originalSrcSetter.call(this, value);
                    }
                });
            }

            return element;
        };
    }

    // Run the handler for missing scripts
    handleMissingScripts();
});

// Close the IIFE
})();
"""
    
    def _get_initialization_code(self) -> str:
        """
        Get code for initializing the report on DOM ready.
        
        Returns:
        --------
        str : Initialization code
        """
        return """
// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', function() {
    console.log("Report initialized");
    
    // Initialize main controller if defined
    if (typeof MainController.init === 'function') {
        MainController.init();
    } else {
        // Fallback initialization
        initTabs();
        initCharts();
    }
    
    // Initialize controllers if defined
    if (typeof OverviewController.init === 'function') {
        OverviewController.init();
    }
    
    if (typeof DetailsController.init === 'function') {
        DetailsController.init();
    }
    
    if (typeof BoxplotController.init === 'function') {
        BoxplotController.init();
    }
    
    if (typeof FeatureImportanceController.init === 'function') {
        FeatureImportanceController.init();
    }
    
    // Initialize new feature table controller
    if (typeof FeatureImportanceTableController.init === 'function') {
        FeatureImportanceTableController.init();
    }
});


function initCharts() {
    if (typeof Plotly !== 'undefined') {
        console.log("Initializing charts with Plotly");
        
        // Try to initialize overview charts
        if (typeof ChartManager.initializePerturbationChart === 'function') {
            ChartManager.initializePerturbationChart('perturbation-chart-plot');
        }
        
        if (typeof ChartManager.initializeWorstScoreChart === 'function') {
            ChartManager.initializeWorstScoreChart('worst-score-chart-plot');
        }
        
        if (typeof ChartManager.initializeMeanScoreChart === 'function') {
            ChartManager.initializeMeanScoreChart('mean-score-chart-plot');
        }
        
        if (typeof ChartManager.initializeModelComparisonChart === 'function') {
            ChartManager.initializeModelComparisonChart('model-comparison-chart-plot');
        }
        
        if (typeof ChartManager.initializeModelLevelDetailsChart === 'function') {
            ChartManager.initializeModelLevelDetailsChart('model-level-details-chart-plot');
        }
        
        // Try to initialize feature charts
        if (typeof ChartManager.initializeFeatureImportanceChart === 'function') {
            ChartManager.initializeFeatureImportanceChart('feature-importance-chart');
        }
    } else {
        console.error("Plotly library not available. Charts cannot be displayed.");
        document.querySelectorAll('.chart-plot').forEach(container => {
            container.innerHTML = "<div style='padding: 20px; text-align: center; color: red;'>Plotly library not loaded. Charts cannot be displayed.</div>";
        });
    }
}
"""
    def _prepare_chart_data(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare data for chart rendering.

        Parameters:
        -----------
        report_data : Dict[str, Any]
            Transformed report data

        Returns:
        --------
        Dict[str, Any] : Data structure for chart rendering
        """
        chart_data = {
            # Basic model information
            'model_name': report_data.get('model_name', 'Model'),
            'model_type': report_data.get('model_type', 'Unknown'),
            'base_score': report_data.get('base_score'),
            'metric_name': report_data.get('metric', 'Score'),
            'robustness_score': report_data.get('robustness_score'),
            'timestamp': report_data.get('timestamp', ''),

            # Feature information
            'feature_subset_display': report_data.get('feature_subset_display', 'N/A'),
            'feature_subset': report_data.get('feature_subset', []),

            # Feature importance data (for charts and tables)
            'feature_importance': report_data.get('feature_importance', {}),
            'model_feature_importance': report_data.get('model_feature_importance', {}),
            'features': report_data.get('features', []),
            'has_feature_importance': bool(report_data.get('feature_importance', {})),

            # Add initial results if available
            'initial_results': report_data.get('initial_results', {})
        }
        
        # Extract perturbation levels and scores
        perturbation_levels = []
        perturbed_scores = []
        worst_scores = []
        feature_subset_scores = []
        
        # Collection of all iteration scores for boxplot
        all_iteration_scores = []
        
        # Extract data from raw results
        if 'raw' in report_data and 'by_level' in report_data['raw']:
            raw_data = report_data['raw']['by_level']
            
            # Sort levels numerically
            levels = sorted([float(level) for level in raw_data.keys()])
            perturbation_levels = levels
            
            # Extract scores for each level
            for level in levels:
                level_str = str(level)
                if level_str in raw_data and 'overall_result' in raw_data[level_str]:
                    result = raw_data[level_str]['overall_result']
                    
                    # All features scores
                    if 'all_features' in result:
                        perturbed_scores.append(result['all_features'].get('mean_score', None))
                        worst_scores.append(result['all_features'].get('worst_score', None))
                    else:
                        perturbed_scores.append(None)
                        worst_scores.append(None)
                    
                    # Feature subset scores
                    if 'feature_subset' in result:
                        feature_subset_scores.append(result['feature_subset'].get('mean_score', None))
                    else:
                        feature_subset_scores.append(None)
                else:
                    # Missing data for this level
                    perturbed_scores.append(None)
                    worst_scores.append(None)
                    feature_subset_scores.append(None)
                
                # Extract iteration scores for boxplot
                if level_str in raw_data and 'runs' in raw_data[level_str]:
                    runs = raw_data[level_str]['runs']
                    if 'all_features' in runs and runs['all_features']:
                        for run in runs['all_features']:
                            if 'iterations' in run and 'scores' in run['iterations']:
                                # Store all iteration scores with their perturbation level
                                iteration_scores = run['iterations']['scores']
                                
                                # Add all iteration scores to the overall collection
                                # This is used for the boxplot visualization
                                if iteration_scores and len(iteration_scores) > 0:
                                    logger.info(f"Level {level_str}: Found {len(iteration_scores)} iteration scores")
                                    all_iteration_scores.extend(iteration_scores)
                                else:
                                    logger.warning(f"Level {level_str}: No iteration scores found")
                                
                                # Store level-specific scores for more detailed analysis
                                if not 'iterations_by_level' in chart_data:
                                    chart_data['iterations_by_level'] = {}
                                    
                                if str(level) not in chart_data['iterations_by_level']:
                                    chart_data['iterations_by_level'][str(level)] = []
                                
                                if iteration_scores and len(iteration_scores) > 0:
                                    chart_data['iterations_by_level'][str(level)].extend(iteration_scores)
        
        # Add scores to chart data
        chart_data.update({
            'perturbation_levels': perturbation_levels,
            'perturbed_scores': perturbed_scores,
            'worst_scores': worst_scores,
            'feature_subset_scores': feature_subset_scores
        })
        
        # Add boxplot data
        primary_model_boxplot = {
            'name': report_data.get('model_name', 'Model'),
            'modelType': report_data.get('model_type', 'Unknown'),
            'baseScore': report_data.get('base_score', 0.0),
            'scores': all_iteration_scores
        }
        
        chart_data['raw'] = report_data.get('raw', {})
        chart_data['iterations'] = report_data.get('n_iterations', 0)
        
        # Initialize boxplot data structure
        boxplot_data = {
            'models': [primary_model_boxplot]
        }
        
        # Collect all perturbation levels from all models
        all_perturbation_levels = set(perturbation_levels)
        
        # Add alternative models data if available
        if 'alternative_models' in report_data and report_data['alternative_models']:
            alt_models = {}
            
            # First collect all unique perturbation levels from all models
            for model_name, model_data in report_data['alternative_models'].items():
                if 'raw' in model_data and 'by_level' in model_data['raw']:
                    raw_data = model_data['raw']['by_level']
                    alt_levels = [float(level) for level in raw_data.keys()]
                    all_perturbation_levels.update(alt_levels)
                    logger.info(f"Collected levels from model {model_name}: {alt_levels}")
            
            # Sort all collected levels
            all_levels_sorted = sorted(all_perturbation_levels)
            logger.info(f"All combined perturbation levels: {all_levels_sorted}")
            
            # Update the main perturbation levels
            perturbation_levels = all_levels_sorted
            
            # Now process each model with the complete set of levels
            for model_name, model_data in report_data['alternative_models'].items():
                model_scores = []
                alt_model_iteration_scores = []
                
                # Get base score
                base_score = model_data.get('base_score', 0.0)
                
                # Get perturbation scores if available
                if 'raw' in model_data and 'by_level' in model_data['raw']:
                    raw_data = model_data['raw']['by_level']
                    
                    # Extract scores for each level using the complete set
                    for level in all_levels_sorted:
                        level_str = str(level)
                        if level_str in raw_data and 'overall_result' in raw_data[level_str]:
                            result = raw_data[level_str]['overall_result']
                            
                            # All features scores
                            if 'all_features' in result:
                                model_scores.append(result['all_features'].get('mean_score', None))
                            else:
                                model_scores.append(None)
                        else:
                            # Missing data for this level
                            model_scores.append(None)
                        
                        # Extract iteration scores for alternative model boxplot
                        if level_str in raw_data and 'runs' in raw_data[level_str]:
                            runs = raw_data[level_str]['runs']
                            if 'all_features' in runs and runs['all_features']:
                                for run in runs['all_features']:
                                    if 'iterations' in run and 'scores' in run['iterations']:
                                        # Store all iteration scores for this alternative model
                                        iteration_scores = run['iterations']['scores']
                                        
                                        # Log and add scores to the collection
                                        if iteration_scores and len(iteration_scores) > 0:
                                            logger.info(f"Alternative model {model_name}, level {level_str}: Found {len(iteration_scores)} iteration scores")
                                            alt_model_iteration_scores.extend(iteration_scores)
                                        else:
                                            logger.warning(f"Alternative model {model_name}, level {level_str}: No iteration scores found")
                                        
                                        # Store level-specific scores for each alternative model
                                        if not 'alt_iterations_by_level' in chart_data:
                                            chart_data['alt_iterations_by_level'] = {}
                                            
                                        if model_name not in chart_data['alt_iterations_by_level']:
                                            chart_data['alt_iterations_by_level'][model_name] = {}
                                        
                                        if str(level) not in chart_data['alt_iterations_by_level'][model_name]:
                                            chart_data['alt_iterations_by_level'][model_name][str(level)] = []
                                        
                                        if iteration_scores and len(iteration_scores) > 0:
                                            chart_data['alt_iterations_by_level'][model_name][str(level)].extend(iteration_scores)
                
                # Add model data
                alt_models[model_name] = {
                    'model_type': model_data.get('model_type', 'Unknown'),
                    'base_score': base_score,
                    'scores': model_scores
                }
                
                # Add to boxplot data
                boxplot_data['models'].append({
                    'name': model_name,
                    'modelType': model_data.get('model_type', 'Unknown'),
                    'baseScore': base_score,
                    'scores': alt_model_iteration_scores
                })
            
            chart_data['alternative_models'] = alt_models
        
        # Add boxplot data to chart data
        chart_data['boxplot_data'] = boxplot_data
        
        # Now that we have the complete set of perturbation levels, 
        # we need to update the primary model data to include all levels
        if 'raw' in report_data and 'by_level' in report_data['raw']:
            raw_data = report_data['raw']['by_level']
            
            # Reset primary model data arrays
            updated_perturbed_scores = []
            updated_worst_scores = []
            updated_feature_subset_scores = []
            
            # Extract scores for each level in the complete set
            for level in perturbation_levels:
                level_str = str(level)
                if level_str in raw_data and 'overall_result' in raw_data[level_str]:
                    result = raw_data[level_str]['overall_result']
                    
                    # All features scores
                    if 'all_features' in result:
                        updated_perturbed_scores.append(result['all_features'].get('mean_score', None))
                        updated_worst_scores.append(result['all_features'].get('worst_score', None))
                    else:
                        updated_perturbed_scores.append(None)
                        updated_worst_scores.append(None)
                    
                    # Feature subset scores
                    if 'feature_subset' in result:
                        updated_feature_subset_scores.append(result['feature_subset'].get('mean_score', None))
                    else:
                        updated_feature_subset_scores.append(None)
                else:
                    # Missing data for this level
                    updated_perturbed_scores.append(None)
                    updated_worst_scores.append(None)
                    updated_feature_subset_scores.append(None)
        else:
            updated_perturbed_scores = perturbed_scores
            updated_worst_scores = worst_scores
            updated_feature_subset_scores = feature_subset_scores
        
        # Prepare data for the perturbation_chart_data object
        perturbation_chart_data = {
            'modelName': report_data.get('model_name', 'Model'),
            'levels': perturbation_levels,
            'scores': updated_perturbed_scores,
            'worstScores': updated_worst_scores,
            'baseScore': report_data.get('base_score', 0.0),
            'metric': report_data.get('metric', 'Score')
        }
        
        # Add alternative models data if available
        if 'alternative_models' in report_data and report_data['alternative_models']:
            alternative_models = {}
            
            for model_name, model_data in report_data['alternative_models'].items():
                alt_model_scores = []
                alt_model_worst_scores = []
                
                # If this model has perturbation level data
                if 'raw' in model_data and 'by_level' in model_data['raw']:
                    raw_data = model_data['raw']['by_level']
                    
                    # Extract scores for each level in the complete set
                    for level in perturbation_levels:
                        level_str = str(level)
                        if level_str in raw_data and 'overall_result' in raw_data[level_str]:
                            result = raw_data[level_str]['overall_result']
                            
                            if 'all_features' in result:
                                alt_model_scores.append(result['all_features'].get('mean_score', None))
                                alt_model_worst_scores.append(result['all_features'].get('worst_score', None))
                            else:
                                alt_model_scores.append(None)
                                alt_model_worst_scores.append(None)
                        else:
                            alt_model_scores.append(None)
                            alt_model_worst_scores.append(None)
                
                alternative_models[model_name] = {
                    'baseScore': model_data.get('base_score', 0.0),
                    'scores': alt_model_scores,
                    'worstScores': alt_model_worst_scores
                }
            
            perturbation_chart_data['alternativeModels'] = alternative_models
        
        # Add perturbation_chart_data to chart_data
        chart_data['perturbation_chart_data'] = perturbation_chart_data
        
        # Prepare data for details section and PerturbationResultsManager
        # Only prepare this section if it hasn't already been populated by the iteration collection code
        if 'iterations_by_level' not in chart_data and 'raw' in report_data and 'by_level' in report_data['raw']:
            iteration_data = {}
            raw_data = report_data['raw']['by_level']
            
            # For each perturbation level
            for level_str, level_data in raw_data.items():
                level_float = float(level_str)
                iterations_by_level = []
                
                # If we have runs data with iterations
                if 'runs' in level_data and 'all_features' in level_data['runs']:
                    for run in level_data['runs']['all_features']:
                        if 'iterations' in run and 'scores' in run['iterations']:
                            # Collect ALL iterations from ALL runs
                            iterations_by_level.extend(run['iterations']['scores'])
                
                iteration_data[level_float] = iterations_by_level
            
            chart_data['iterations_by_level'] = iteration_data
        
        # Make sure we have the iterations_by_level data structure even if no data was collected
        if 'iterations_by_level' not in chart_data:
            chart_data['iterations_by_level'] = {}
            
        # Add the alternative models iteration data if available
        if 'alt_iterations_by_level' in chart_data:
            chart_data['alternative_models_iterations'] = chart_data['alt_iterations_by_level']
            # Clean up the temporary data structure to avoid confusion
            del chart_data['alt_iterations_by_level']
        
        return chart_data