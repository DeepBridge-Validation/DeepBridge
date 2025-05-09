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
        
        # Import data transformers
        from ..transformers.robustness import RobustnessDataTransformer
        from ..transformers.initial_results import InitialResultsTransformer
        self.data_transformer = RobustnessDataTransformer()
        self.initial_results_transformer = InitialResultsTransformer()
    
    def render(self, results: Dict[str, Any], file_path: str, model_name: str = "Model") -> str:
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
            # We need to manually inject styles and scripts into the context
            # rather than relying on loading from external files
            context = {
                'report_type': 'robustness',
                'report_data': report_data,
                'report_data_json': self._sanitize_json(report_data),
                'chart_data_json': self._sanitize_json(chart_data),
                'css_content': css_content,
                'js_content': js_content,
                'favicon_base64': self.asset_manager.get_favicon_base64()
            }
            
            # Add initial_results directly to the report_data for client-side access
            if initial_results:
                report_data['initial_results'] = initial_results
                
                # Explicitly log for debugging
                logger.info(f"Added initial_results to report_data with {len(initial_results.get('models', {}))} models")
                
            # Add perturbation chart data to ensure it's accessible in the template
            if chart_data and 'perturbation_chart_data' in chart_data:
                report_data['perturbation_chart_data'] = chart_data['perturbation_chart_data']
                logger.info("Added perturbation_chart_data directly to report_data")
                
            # Add perturbation details data
            if chart_data and 'perturbation_details_data' in chart_data:
                report_data['perturbation_details_data'] = chart_data['perturbation_details_data']
                logger.info("Added perturbation_details_data directly to report_data")
            
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
        Load and combine CSS files for the robustness report.
        
        Returns:
        --------
        str : Combined CSS content
        """
        try:
            # Use the asset manager's combined CSS content method
            css_content = self.asset_manager.get_combined_css_content("robustness")
            return css_content
        except Exception as e:
            logger.error(f"Error loading CSS: {str(e)}")
            return ""
    
    def _load_js_content(self) -> str:
        """
        Load and combine JavaScript files for the robustness report.
        
        Returns:
        --------
        str : Combined JavaScript content with namespace safety
        """
        try:
            # Explicitly identify key JS files to load in order
            js_dir = self.asset_manager.find_js_path("robustness")
            
            # Define a specific loading order for critical files
            critical_files = [
                "global_error_handler.js",
                "utils.js",  # Move utils earlier so it's available for other scripts
                "feature_importance_chart.js",
                "feature_importance_handler.js",
                "importance_comparison_handler.js"
            ]
            
            # Add a preamble to define a namespace safety mechanism
            js_content = """// ----- Namespace Safety Mechanism ----- //
// This prevents duplicate declarations of global objects
window.__deepbridge_loaded_modules = window.__deepbridge_loaded_modules || {};

// Function to safely register a module and prevent duplication
window.registerModule = function(name, factory) {
    if (window.__deepbridge_loaded_modules[name]) {
        console.log(`Module ${name} already registered, using existing instance`);
        return window.__deepbridge_loaded_modules[name];
    }
    console.log(`Registering module ${name}`);
    var module = factory();
    window.__deepbridge_loaded_modules[name] = module;
    return module;
};

// ----- Critical JS Files ----- //
"""
            
            # Load each critical file first with namespace protection
            for filename in critical_files:
                file_path = os.path.join(js_dir, filename)
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            js_content += f"// ----- {filename} ----- //\n"
                            content = f.read()
                            # Don't wrap utility files, they should be available globally
                            if filename in ["utils.js", "global_error_handler.js"]:
                                js_content += content + "\n\n"
                            else:
                                # Wrap other files in IIFE for namespace protection
                                js_content += f"(function() {{\n{content}\n}})();\n\n"
                            logger.info(f"Loaded critical JS file: {filename}")
                    except Exception as e:
                        logger.warning(f"Error loading critical JS file {filename}: {str(e)}")
                else:
                    logger.warning(f"Critical JS file not found: {filename}")
            
            # Load Chart files with namespace protection
            js_content += "// ----- Chart JS Files ----- //\n\n"
            charts_dir = os.path.join(js_dir, "charts")
            if os.path.exists(charts_dir):
                for filename in sorted(os.listdir(charts_dir)):
                    if filename.endswith('.js'):
                        try:
                            file_path = os.path.join(charts_dir, filename)
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                js_content += f"// ----- charts/{filename} ----- //\n"
                                
                                # Special handling for boxplot.js to prevent duplicate declaration
                                if filename == "boxplot.js":
                                    # Store the content minus the variable declaration
                                    if "const BoxplotChartManager = {" in content:
                                        object_content = content.replace("const BoxplotChartManager = {", "{")
                                        js_content += """// Safely register BoxplotChartManager to prevent duplicates
window.BoxplotChartManager = window.registerModule('BoxplotChartManager', function() {
    return """ + object_content + "});\n\n"
                                    else:
                                        # Fallback if the format is unexpected
                                        js_content += f"(function() {{\n{content}\n}})();\n\n"
                                else:
                                    # Normal IIFE wrapping for other chart files
                                    js_content += f"(function() {{\n{content}\n}})();\n\n"
                                
                                logger.info(f"Loaded chart JS file: {filename} with namespace protection")
                        except Exception as e:
                            logger.warning(f"Error loading chart JS file {filename}: {str(e)}")
            
            # Load Controller files with namespace protection
            js_content += "// ----- Controller JS Files ----- //\n\n"
            controllers_dir = os.path.join(js_dir, "controllers")
            if os.path.exists(controllers_dir):
                for filename in sorted(os.listdir(controllers_dir)):
                    if filename.endswith('.js'):
                        try:
                            file_path = os.path.join(controllers_dir, filename)
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                js_content += f"// ----- controllers/{filename} ----- //\n"
                                
                                # Safely register controllers to prevent duplicates
                                controller_name = filename.replace('.js', '')
                                if "Controller" in controller_name:
                                    # Check if the file defines a controller object
                                    const_pattern = f"const {controller_name} = "
                                    var_pattern = f"var {controller_name} = "
                                    
                                    if const_pattern in content:
                                        object_content = content.replace(const_pattern + "{", "{")
                                        js_content += f"window.{controller_name} = window.registerModule('{controller_name}', function() {{\n"
                                        js_content += f"    return {object_content}}};\n\n"
                                    elif var_pattern in content:
                                        object_content = content.replace(var_pattern + "{", "{")
                                        js_content += f"window.{controller_name} = window.registerModule('{controller_name}', function() {{\n"
                                        js_content += f"    return {object_content}}};\n\n"
                                    else:
                                        # Normal IIFE wrapping for controllers without specific patterns
                                        js_content += f"(function() {{\n{content}\n}})();\n\n"
                                else:
                                    # Normal IIFE wrapping for other controllers
                                    js_content += f"(function() {{\n{content}\n}})();\n\n"
                                
                                logger.info(f"Loaded controller JS file: {filename} with namespace protection")
                        except Exception as e:
                            logger.warning(f"Error loading controller JS file {filename}: {str(e)}")
            
            # Always load main.js last, with special handling (no wrapping to allow it to see all other modules)
            main_js_path = os.path.join(js_dir, "main.js")
            if os.path.exists(main_js_path):
                try:
                    with open(main_js_path, 'r', encoding='utf-8') as f:
                        js_content += "// ----- main.js ----- //\n"
                        # Do not wrap main.js since it needs to access all other modules
                        main_content = f.read()
                        
                        # Remove any attempts to load external scripts
                        main_content = main_content.replace("fixScript.src = 'js/fixed_syntax.js';", "// External script loading removed")
                        main_content = main_content.replace("script.src = 'js/", "// External script loading removed: script.src = 'js/")
                        
                        js_content += main_content + "\n\n"
                    logger.info("Loaded main.js")
                except Exception as e:
                    logger.warning(f"Error loading main.js: {str(e)}")
                    
            # Fix any JavaScript syntax issues
            try:
                from ..js_syntax_fixer import JavaScriptSyntaxFixer
                js_content = JavaScriptSyntaxFixer.apply_all_fixes(js_content)
                logger.info("Applied JavaScript syntax fixes")
            except ImportError:
                logger.warning("JavaScript syntax fixer not available")
                
            # Additional safety - remove any remaining external script loading attempts
            js_content = js_content.replace(".src = 'js/", ".src = '//REMOVED_EXTERNAL_PATH_js/")
            js_content = js_content.replace("src='js/", "src='//REMOVED_EXTERNAL_PATH_js/")
            js_content = js_content.replace('src="js/', 'src="//REMOVED_EXTERNAL_PATH_js/')
            
            return js_content
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
            'has_feature_importance': bool(report_data.get('feature_importance', {}))
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
        
        # Prepare data for details section and iteration analysis
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
        
        # Create perturbation_details_data with ONLY real data - no synthetic values
        # This structured data will make it easier to render in the Details tab
        perturbation_details_data = self._prepare_perturbation_details_data(report_data)
        chart_data['perturbation_details_data'] = perturbation_details_data
        
        return chart_data
        
    def _prepare_perturbation_details_data(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare structured data specifically for the Details tab
        perturbation analysis section.
        
        Parameters:
        -----------
        report_data : Dict[str, Any]
            Transformed report data
            
        Returns:
        --------
        Dict[str, Any] : Structured perturbation details data
        """
        # Base structure with model info
        details_data = {
            'modelName': report_data.get('model_name', 'Model'),
            'modelType': report_data.get('model_type', 'Unknown'),
            'metric': report_data.get('metric', 'Score'),
            'baseScore': report_data.get('base_score', 0),
            'results': []
        }
        
        # Extract real perturbation data for each level directly from raw data
        if 'raw' in report_data and 'by_level' in report_data['raw']:
            raw_data = report_data['raw']['by_level']
            
            # Get perturbation levels and sort them
            levels = sorted([float(level) for level in raw_data.keys()])
            
            # Process each level
            for level in levels:
                level_str = str(level)
                level_data = raw_data[level_str]
                
                # Skip if no overall results
                if not level_data or 'overall_result' not in level_data:
                    logger.info(f"No overall result for level {level}")
                    continue
                
                level_result = {
                    'level': level,
                    'allFeatures': None,
                    'featureSubset': None
                }
                
                # Extract all features data
                if 'all_features' in level_data['overall_result']:
                    feature_result = level_data['overall_result']['all_features']
                    base_score = details_data['baseScore']
                    mean_score = feature_result.get('mean_score', 0)
                    
                    # Calculate impact safely
                    impact = 0
                    if base_score > 0:
                        impact = (base_score - mean_score) / base_score
                    
                    level_result['allFeatures'] = {
                        'baseScore': base_score,
                        'meanScore': mean_score,
                        'worstScore': feature_result.get('worst_score', 0),
                        'impact': impact,
                        'iterations': []
                    }
                    
                    # Add iteration scores if available
                    if 'runs' in level_data and 'all_features' in level_data['runs']:
                        iterations = []
                        for run in level_data['runs']['all_features']:
                            if 'iterations' in run and 'scores' in run['iterations']:
                                iterations.extend(run['iterations']['scores'])
                        level_result['allFeatures']['iterations'] = iterations
                
                # Extract feature subset data
                if 'feature_subset' in level_data['overall_result']:
                    feature_result = level_data['overall_result']['feature_subset']
                    base_score = details_data['baseScore']
                    mean_score = feature_result.get('mean_score', 0)
                    
                    # Calculate impact safely
                    impact = 0
                    if base_score > 0:
                        impact = (base_score - mean_score) / base_score
                    
                    level_result['featureSubset'] = {
                        'baseScore': base_score,
                        'meanScore': mean_score,
                        'worstScore': feature_result.get('worst_score', 0),
                        'impact': impact,
                        'iterations': []
                    }
                    
                    # Add iteration scores if available
                    if 'runs' in level_data and 'feature_subset' in level_data['runs']:
                        iterations = []
                        for run in level_data['runs']['feature_subset']:
                            if 'iterations' in run and 'scores' in run['iterations']:
                                iterations.extend(run['iterations']['scores'])
                        level_result['featureSubset']['iterations'] = iterations
                
                details_data['results'].append(level_result)
                logger.info(f"Added perturbation details for level {level}")
        else:
            logger.warning("No raw data available for perturbation details")
        
        return details_data