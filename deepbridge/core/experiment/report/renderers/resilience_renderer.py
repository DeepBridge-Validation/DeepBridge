"""
Resilience report renderer with enhanced JavaScript handling.
"""

import os
import logging
from typing import Dict, Any, Optional, List

# Configure logger
logger = logging.getLogger("deepbridge.reports")

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
            
    def render(self, results: Dict[str, Any], file_path: str, model_name: str = "Model", report_type: str = "interactive") -> str:
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
            
            # Find CSS and JS paths
            css_dir = self.asset_manager.find_css_path("resilience")
            js_dir = self.asset_manager.find_js_path("resilience")
            
            if not css_dir:
                raise FileNotFoundError("CSS directory not found for resilience report")
            
            if not js_dir:
                raise FileNotFoundError("JavaScript directory not found for resilience report")
            
            # Get CSS content
            css_content = self.asset_manager.get_css_content(css_dir)
            
            # Define core components needed for resilience report
            core_components = [
                "MainController",
                "PerformanceGapChartManager", 
                "PerformanceGapController",
                "FeatureChartManager",
                "FeatureController",
                "ResilienceScoreChartManager",
                "ResilienceScoreController",
                "ModelMetricsChartManager",
                "ModelMetricsController",
                "DistributionChartManager",
                "DistributionController",
                "PerturbationResultsManager",
                "PerturbationResultsController",
            ]
            
            # Load JS files in specific order to handle dependencies correctly
            js_files_order = [
                # Core utilities first
                {'path': 'utils.js', 'required': False},
                
                # Charts modules
                {'path': os.path.join('charts', 'gap.js'), 'required': False},
                {'path': os.path.join('charts', 'score.js'), 'required': False},
                {'path': os.path.join('charts', 'metrics.js'), 'required': False},
                {'path': os.path.join('charts', 'feature.js'), 'required': False},
                {'path': os.path.join('charts', 'distribution.js'), 'required': False},
                {'path': os.path.join('charts', 'results.js'), 'required': False},
                
                # Controllers
                {'path': os.path.join('controllers', 'gap.js'), 'required': False},
                {'path': os.path.join('controllers', 'score.js'), 'required': False},
                {'path': os.path.join('controllers', 'metrics.js'), 'required': False},
                {'path': os.path.join('controllers', 'feature.js'), 'required': False},
                {'path': os.path.join('controllers', 'distribution.js'), 'required': False},
                {'path': os.path.join('controllers', 'results.js'), 'required': False},
                
                # Main JS (always last)
                {'path': 'main.js', 'required': True}
            ]
            
            # Combine all JS
            js_content = "// Combined JavaScript for resilience report\n\n"
            
            # Create global objects for namespaces
            js_content += "// Global objects\n"
            for component in core_components:
                js_content += f"window.{component} = window.{component} || {{}};\n"
            
            # Load and combine all JS files in order
            for js_file in js_files_order:
                js_path = os.path.join(js_dir, js_file["path"])
                
                if os.path.exists(js_path):
                    # Check if file has content (skip empty files)
                    file_size = os.path.getsize(js_path)
                    if file_size == 0:
                        logger.warning(f"Skipping empty file: {js_path}")
                        continue
                    
                    try:
                        with open(js_path, 'r', encoding='utf-8') as f:
                            file_content = f.read().strip()
                            
                            # Skip if file is empty
                            if not file_content:
                                logger.warning(f"File is empty: {js_path}")
                                continue
                            
                            # Remove ES6 import/export statements and fix window assignment
                            file_content = self._process_js_content(file_content, js_file["path"])
                            
                            js_content += f"\n\n// File: {js_file['path']}\n"
                            js_content += file_content
                            
                            logger.info(f"Added JS file: {js_path} ({file_size} bytes)")
                    except Exception as e:
                        logger.warning(f"Error reading JS file {js_path}: {str(e)}")
                        if js_file["required"]:
                            logger.error(f"Error in required file: {js_path}")
                else:
                    if js_file["required"]:
                        logger.error(f"Required JavaScript file not found: {js_path}")
                        # Create a placeholder for the required file
                        js_content += f"\n\n// File: {js_file['path']} (NOT FOUND - PLACEHOLDER)\n"
                        js_content += f"// Placeholder for {js_file['path']}\n"
                    else:
                        logger.warning(f"Optional JavaScript file not found: {js_path}")
            
            # Add consolidated report fix JavaScript
            # This is a special module that contains fixed implementations of 
            # components that might be missing or incorrectly implemented
            consolidated_fix_path = os.path.join(js_dir, 'resilience_report_fix.js')
            if os.path.exists(consolidated_fix_path):
                try:
                    with open(consolidated_fix_path, 'r', encoding='utf-8') as f:
                        fix_content = f.read().strip()
                        if fix_content:
                            js_content += "\n\n// Consolidated report fixes\n"
                            js_content += fix_content
                            logger.info(f"Added consolidated fixes from: {consolidated_fix_path}")
                except Exception as e:
                    logger.warning(f"Error reading consolidated fix file: {str(e)}")
            
            # Add initialization code at the end
            js_content += """
            
            // DOM Ready initialization - consolidated handler
            document.addEventListener('DOMContentLoaded', function() {
                console.log("Resilience report initialization");
                
                // Initialize main controller first
                if (typeof MainController !== 'undefined' && typeof MainController.init === 'function') {
                    console.log("Initializing MainController");
                    MainController.init();
                }
                
                // Initialize all component controllers with a short delay
                setTimeout(function() {
                    console.log("Initializing component controllers");
                    
                    // Performance Gap components
                    if (typeof PerformanceGapController !== 'undefined' && typeof PerformanceGapController.init === 'function') {
                        console.log("Initializing PerformanceGapController");
                        PerformanceGapController.init();
                    }
                    
                    // Feature components
                    if (typeof FeatureController !== 'undefined' && typeof FeatureController.init === 'function') {
                        console.log("Initializing FeatureController");
                        FeatureController.init();
                    }
                    
                    // Score components
                    if (typeof ScoreController !== 'undefined' && typeof ScoreController.init === 'function') {
                        console.log("Initializing ScoreController");
                        ScoreController.init();
                    }
                    
                    // Metrics components
                    if (typeof MetricsController !== 'undefined' && typeof MetricsController.init === 'function') {
                        console.log("Initializing MetricsController");
                        MetricsController.init();
                    }
                    
                    // Distribution components 
                    if (typeof DistributionController !== 'undefined' && typeof DistributionController.init === 'function') {
                        console.log("Initializing DistributionController");
                        DistributionController.init();
                    }
                    
                    // Reset tab navigation to ensure consistency
                    const tabContents = document.querySelectorAll('.tab-content');
                    const tabButtons = document.querySelectorAll('.tab-btn');
                    
                    // Reset all tabs first
                    tabContents.forEach(content => {
                        content.classList.remove('active');
                    });
                    
                    tabButtons.forEach(btn => {
                        btn.classList.remove('active');
                    });
                    
                    // Set the first tab (overview) as active
                    const overview = document.getElementById('overview');
                    if (overview) {
                        overview.classList.add('active');
                    }
                    
                    // Set the first button as active
                    if (tabButtons.length > 0) {
                        tabButtons[0].classList.add('active');
                    }
                    
                    console.log("Finished initializing all components");
                }, 500);
            });
            """
            
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
                    scenario = {
                        'name': result.get('name', f"Scenario {len(shift_scenarios) + 1}"),
                        'alpha': result.get('alpha', 0),
                        'metric': result.get('metric', 'unknown'),
                        'distance_metric': result.get('distance_metric', 'unknown'),
                        'performance_gap': result.get('performance_gap', 0),
                        'baseline_performance': result.get('baseline_performance', 0),
                        'target_performance': result.get('target_performance', 0),
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
                'distance_metrics': report_data.get('distance_metrics', []),
                'alphas': report_data.get('alphas', []),
                'feature_subset': report_data.get('feature_subset', []),
                'shift_scenarios': shift_scenarios,
                'sensitive_features': sensitive_features,
                'baseline_dataset': baseline_dataset or "Baseline",
                'target_dataset': target_dataset or "Target",
                'timestamp': report_data.get('timestamp', ''),
                'avg_dist_shift': avg_dist_shift or 0
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
                
                # Additional context to ensure backward compatibility
                'features': report_data.get('features', []),
                'metrics': report_data.get('metrics', {}),
                'metrics_details': report_data.get('metrics_details', {}),
                'has_feature_data': bool(sensitive_features),
                
                # JSON representation for JavaScript config
                'report_data_json': report_data_json
            })
            
            # Render the template
            rendered_html = self.template_manager.render_template(template, context)
            
            # Write the report to file
            return self.base_renderer._write_report(rendered_html, file_path)
            
        except Exception as e:
            logger.error(f"Error generating resilience report: {str(e)}")
            raise ValueError(f"Failed to generate resilience report: {str(e)}")