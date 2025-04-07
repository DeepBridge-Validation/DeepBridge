"""
Report generator for experiment results.
This module provides functionality to generate comprehensive reports from experiment results.
"""

import os
import json
import datetime
import jinja2
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

class ReportGenerator:
    """
    Generates consolidated reports from experiment test results.
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the report generator.
        
        Parameters:
        -----------
        verbose : bool
            Whether to print progress information
        """
        self.verbose = verbose
    
    def generate_report(self, 
                       results: Dict[str, Any], 
                       output_path: str,
                       experiment_name: str = "Experiment") -> str:
        """
        Generate a comprehensive HTML report from experiment results.
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Dictionary containing test results
        output_path : str
            Path to save the generated report
        experiment_name : str
            Name of the experiment for the report
            
        Returns:
        --------
        str : Path to the saved report
        """
        # Ensure directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load template
        template_content = self._get_template()
        
        # Prepare template context
        context = self._prepare_template_context(results, experiment_name)
        
        # Render template
        html_content = self._render_template(template_content, context)
        
        # Save the report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        if self.verbose:
            print(f"Report saved to {output_path}")
            
        return str(output_path)
    
    def _get_template(self) -> str:
        """Get the report template HTML content."""
        # Try to load templates in priority order
        template_names = [
            "interactive_report_template.html",  # First try our interactive template
            "basic_report_template.html",        # Then our simple template
            "experiment.html",                   # Then the regular experiment template
            "robustness_report_template.html"    # Finally try the robustness template
        ]
        
        for template_name in template_names:
            template_path = self._find_template_path(template_name)
            if template_path:
                try:
                    with open(template_path, 'r', encoding='utf-8') as f:
                        if self.verbose:
                            print(f"Using template: {template_path}")
                        return f.read()
                except Exception as e:
                    if self.verbose:
                        print(f"Error loading template {template_path}: {str(e)}")
        
        # Return a basic template if none of the templates are found
        if self.verbose:
            print("No templates found, using built-in basic template")
        return self._get_basic_template()
    
    def _find_template_path(self, template_name: str) -> Optional[str]:
        """Find the path to a template file."""
        # Define possible template locations
        possible_paths = [
            # Current module's directory
            Path(__file__).parent / ".." / ".." / "reports" / "templates" / template_name,
            # Project root
            Path(__file__).parent / ".." / ".." / ".." / "reports" / "templates" / template_name,
            # Absolute path within the package
            Path("/home/guhaase/projetos/DeepBridge/deepbridge/reports/templates") / template_name
        ]
        
        # Return the first path that exists
        for path in possible_paths:
            if path.exists():
                return str(path)
        
        return None
    
    def _get_basic_template(self) -> str:
        """Get a basic HTML template as fallback."""
        return """<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{{experiment_name}} - Validation Report</title>
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; }
                .container { max-width: 1200px; margin: 0 auto; }
                h1, h2, h3 { color: #2c3e50; }
                .header { background: linear-gradient(135deg, #0062cc 0%, #1e88e5 100%); color: white; padding: 20px; margin-bottom: 20px; border-radius: 5px; }
                .section { margin-bottom: 30px; padding: 20px; background: #fff; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
                table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
                th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #f2f2f2; font-weight: bold; }
                .metric-good { color: #28a745; font-weight: bold; }
                .metric-medium { color: #ffc107; font-weight: bold; }
                .metric-poor { color: #dc3545; font-weight: bold; }
                .test-result { margin-bottom: 20px; padding: 15px; border: 1px solid #eee; border-radius: 5px; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>{{experiment_name}} - Validation Report</h1>
                    <p>Generated on: {{generation_time}}</p>
                </div>
                
                <div class="section">
                    <h2>Experiment Overview</h2>
                    <table>
                        <tr><th>Attribute</th><th>Value</th></tr>
                        <tr><td>Experiment Type</td><td>{{experiment_type}}</td></tr>
                        <tr><td>Tests Performed</td><td>{{tests_performed}}</td></tr>
                        <tr><td>Dataset Size</td><td>{{dataset_size}}</td></tr>
                    </table>
                </div>
                
                <!-- Test Results Sections -->
                {{test_sections_html}}
                
                <div class="section">
                    <h2>Recommendations</h2>
                    <div class="test-result">
                        <p>Based on the test results, here are some recommendations:</p>
                        <ul>
                            {{recommendations_html}}
                        </ul>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
    
    def _prepare_template_context(self, results: Dict[str, Any], experiment_name: str) -> Dict[str, Any]:
        """Prepare the context for the template rendering."""
        # Basic context
        context = {
            # Main report information
            'experiment_name': experiment_name,
            'generation_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'experiment_type': results.get('experiment_type', 'Unknown'),
            'tests_performed': ', '.join(results.get('tests_performed', [])),
            'dataset_size': results.get('dataset_size', 'Unknown'),
            'results': results,
            
            # Add placeholders for template variables that might be expected
            'report_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'models_evaluated': 'Primary model',
            'key_finding': 'Model evaluation completed successfully',
            'summary_text': 'See individual test sections for detailed results',
            'best_model': 'Primary model',
            'perturbation_levels': [],
            'model_metrics': [],
            'model_detailed_results': [],
            'top_features': [],
            'boxplot_chart': json.dumps({'data': [], 'layout': {}}),
            'feature_importance_chart': json.dumps({'data': [], 'layout': {}}),
            'a': 3,  # For limiting recommendations in loop
        }
        
        # Generate HTML for test sections
        test_sections_html = []
        recommendations = []
        
        # Process each test result
        for test_name, test_result in results.items():
            if test_name in ['robustness', 'uncertainty', 'resilience', 'hyperparameters']:
                section_html = self._generate_test_section(test_name, test_result)
                test_sections_html.append(section_html)
                
                # Add recommendations for this test
                test_recommendations = self._generate_recommendations(test_name, test_result)
                recommendations.extend(test_recommendations)
        
        context['test_sections_html'] = '\n'.join(test_sections_html)
        
        # Format recommendations
        recommendations_html = []
        for rec in recommendations:
            recommendations_html.append(f'<li><strong>{rec["title"]}:</strong> {rec["content"]}</li>')
        
        context['recommendations_html'] = '\n'.join(recommendations_html)
        
        return context
    
    def _generate_test_section(self, test_name: str, test_result: Dict[str, Any]) -> str:
        """Generate HTML for a test section."""
        capitalize_name = test_name.capitalize()
        
        html = f"""
        <div class="section">
            <h2>{capitalize_name} Test Results</h2>
            <div class="test-result">
        """
        
        # Handle different test types
        if test_name == 'robustness':
            html += self._format_robustness_results(test_result)
        elif test_name == 'uncertainty':
            html += self._format_uncertainty_results(test_result)
        elif test_name == 'resilience':
            html += self._format_resilience_results(test_result)
        elif test_name == 'hyperparameters':
            html += self._format_hyperparameter_results(test_result)
        else:
            # Generic format for unknown test types
            html += '<table><tr><th>Metric</th><th>Value</th></tr>'
            for key, value in test_result.items():
                if isinstance(value, (int, float, str, bool)):
                    html += f'<tr><td>{key}</td><td>{value}</td></tr>'
            html += '</table>'
        
        html += """
            </div>
        </div>
        """
        
        return html
    
    def _format_robustness_results(self, results: Dict[str, Any]) -> str:
        """Format robustness test results."""
        html = f"""
        <h3>Overall Robustness</h3>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Baseline Score</td><td>{results.get('base_score', 'N/A')}</td></tr>
            <tr><td>Average Impact</td><td>{results.get('avg_overall_impact', 'N/A')}</td></tr>
        </table>
        """
        
        # Format raw perturbation results
        if 'raw' in results and 'by_level' in results['raw']:
            html += '<h3>Gaussian Noise Perturbation</h3><table><tr><th>Level</th><th>Mean Score</th><th>Standard Deviation</th></tr>'
            
            for level, level_data in sorted(results['raw']['by_level'].items()):
                overall = level_data.get('overall_result', {})
                if overall:
                    html += f'<tr><td>{level}</td><td>{overall.get("mean_score", "N/A")}</td><td>{overall.get("std_score", "N/A")}</td></tr>'
            
            html += '</table>'
        
        # Format quantile perturbation results
        if 'quantile' in results and 'by_level' in results['quantile']:
            html += '<h3>Quantile Perturbation</h3><table><tr><th>Level</th><th>Mean Score</th><th>Standard Deviation</th></tr>'
            
            for level, level_data in sorted(results['quantile']['by_level'].items()):
                overall = level_data.get('overall_result', {})
                if overall:
                    html += f'<tr><td>{level}</td><td>{overall.get("mean_score", "N/A")}</td><td>{overall.get("std_score", "N/A")}</td></tr>'
            
            html += '</table>'
        
        # Format feature importance
        if 'feature_importance' in results:
            html += '<h3>Feature Importance</h3><table><tr><th>Feature</th><th>Importance</th></tr>'
            
            # Sort features by importance
            sorted_features = sorted(results['feature_importance'].items(), key=lambda x: x[1], reverse=True)
            
            # Display top 10 features
            for feature, value in sorted_features[:10]:
                html += f'<tr><td>{feature}</td><td>{value:.3f}</td></tr>'
            
            html += '</table>'
        
        return html
    
    def _format_uncertainty_results(self, results: Dict[str, Any]) -> str:
        """Format uncertainty test results."""
        html = '<h3>Calibration Metrics</h3><table><tr><th>Metric</th><th>Value</th></tr>'
        
        for key, value in results.items():
            if isinstance(value, (int, float)) and not key.startswith('_'):
                html += f'<tr><td>{key.replace("_", " ").title()}</td><td>{value}</td></tr>'
        
        html += '</table>'
        
        return html
    
    def _format_resilience_results(self, results: Dict[str, Any]) -> str:
        """Format resilience test results."""
        html = '<h3>Resilience Metrics</h3><table><tr><th>Metric</th><th>Value</th></tr>'
        
        for key, value in results.items():
            if isinstance(value, (int, float)) and not key.startswith('_'):
                html += f'<tr><td>{key.replace("_", " ").title()}</td><td>{value}</td></tr>'
        
        html += '</table>'
        
        return html
    
    def _format_hyperparameter_results(self, results: Dict[str, Any]) -> str:
        """Format hyperparameter test results."""
        html = '<h3>Hyperparameter Importance</h3><table><tr><th>Parameter</th><th>Importance</th></tr>'
        
        # Check if results is dictionary with importances
        if isinstance(results, dict):
            # Sort parameters by importance
            sorted_params = sorted(results.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0, reverse=True)
            
            for param, value in sorted_params:
                if isinstance(value, (int, float)):
                    html += f'<tr><td>{param}</td><td>{value:.3f}</td></tr>'
        
        html += '</table>'
        
        return html
    
    def _generate_recommendations(self, test_name: str, test_result: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        if test_name == 'robustness':
            impact = test_result.get('avg_overall_impact', 0)
            if impact > 0.3:
                recommendations.append({
                    'title': 'Improve Model Robustness',
                    'content': 'Consider using data augmentation or adversarial training to improve model robustness to perturbations.'
                })
        elif test_name == 'uncertainty':
            recommendations.append({
                'title': 'Uncertainty Quantification',
                'content': 'Consider using ensemble methods or dropout techniques to better quantify prediction uncertainty.'
            })
        elif test_name == 'resilience':
            recommendations.append({
                'title': 'Enhance Model Resilience',
                'content': 'To improve model resilience to data distribution shifts, consider training with diverse data sources.'
            })
        elif test_name == 'hyperparameters':
            recommendations.append({
                'title': 'Hyperparameter Optimization',
                'content': 'Focus optimization efforts on the most impactful hyperparameters identified in the analysis.'
            })
        
        return recommendations
    
    def _render_template(self, template_content: str, context: Dict[str, Any]) -> str:
        """Render the template with the provided context."""
        try:
            # Add extra safety filters and defaults to catch missing values
            safe_context = context.copy()
            
            # Ensure test_data dictionaries have safe defaults for common keys
            if 'results' in safe_context:
                # Add get() method to results dictionary as a convenience function
                # (This doesn't actually matter for rendering, but makes templates cleaner)
                if not hasattr(safe_context['results'], 'get'):
                    safe_context['results'] = dict(safe_context['results'])
                
                # Ensure all test result subdictionaries exist
                for test_type in ['robustness', 'uncertainty', 'resilience', 'hyperparameters']:
                    if test_type not in safe_context['results']:
                        safe_context['results'][test_type] = {}
                    
                    # Ensure robustness data has expected keys
                    if test_type == 'robustness':
                        robustness_data = safe_context['results'][test_type]
                        for key in ['base_score', 'avg_overall_impact', 'avg_raw_impact', 'avg_quantile_impact', 'feature_importance']:
                            if key not in robustness_data:
                                if key == 'feature_importance':
                                    robustness_data[key] = {}
                                else:
                                    robustness_data[key] = 0.0
            
            # Create a Jinja2 environment with secure settings and error handling
            env = jinja2.Environment(
                undefined=jinja2.StrictUndefined,  # Raises errors for undefined variables
                autoescape=True                    # Auto-escape HTML to prevent XSS
            )
            
            # Add helpful filters
            env.filters['json'] = lambda obj: json.dumps(obj)
            env.filters['string'] = lambda obj: str(obj)
            
            # Add safe dictionary access filter
            def safe_get(obj, key, default=None):
                if isinstance(obj, dict):
                    return obj.get(key, default)
                return default
            
            env.filters['safe_get'] = safe_get
            
            # Create a template from the string
            template = env.from_string(template_content)
            
            try:
                # Try rendering with strict mode first
                return template.render(**safe_context)
            except jinja2.exceptions.UndefinedError as strict_error:
                # If strict mode fails, create a more lenient template
                if self.verbose:
                    print(f"Strict template rendering failed: {str(strict_error)}")
                    print("Trying again with lenient undefined handling...")
                
                # Try with lenient error handling
                lenient_env = jinja2.Environment(
                    undefined=jinja2.Undefined,  # Just returns empty string for undefined
                    autoescape=True
                )
                lenient_env.filters['json'] = lambda obj: json.dumps(obj)
                lenient_env.filters['string'] = lambda obj: str(obj)
                lenient_env.filters['safe_get'] = safe_get
                
                lenient_template = lenient_env.from_string(template_content)
                
                try:
                    return lenient_template.render(**safe_context)
                except Exception as lenient_error:
                    if self.verbose:
                        print(f"Lenient template rendering also failed: {str(lenient_error)}")
                        print("Falling back to most permissive rendering...")
                    
                    # Final attempt with most permissive handling
                    most_permissive_env = jinja2.Environment(
                        undefined=jinja2.ChainableUndefined,  # Allows chained attribute access on undefined
                        autoescape=True
                    )
                    most_permissive_env.filters['json'] = lambda obj: json.dumps(obj) if obj else "{}"
                    most_permissive_env.filters['string'] = lambda obj: str(obj) if obj else ""
                    most_permissive_env.filters['safe_get'] = safe_get
                    
                    most_permissive_template = most_permissive_env.from_string(template_content)
                    return most_permissive_template.render(**safe_context)
            
        except Exception as e:
            if self.verbose:
                print(f"Error rendering template: {str(e)}")
            
            # Fallback to basic template if rendering fails
            return self._render_basic_report(context)
    
    def _render_basic_report(self, context: Dict[str, Any]) -> str:
        """Render a basic report as fallback."""
        html = f"""<!DOCTYPE html>
        <html>
        <head>
            <title>{context['experiment_name']} - Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                .section {{ margin: 20px 0; padding: 10px; border: 1px solid #eee; }}
            </style>
        </head>
        <body>
            <h1>{context['experiment_name']} - Validation Report</h1>
            <p>Generated on: {context['generation_time']}</p>
            
            <div class="section">
                <h2>Experiment Information</h2>
                <p>Type: {context['experiment_type']}</p>
                <p>Tests: {context['tests_performed']}</p>
            </div>
        """
        
        # Add test results sections
        html += context.get('test_sections_html', '')
        
        # Close the HTML
        html += """
        </body>
        </html>
        """
        
        return html


def generate_report_from_results(results: Dict[str, Any], output_path: str, experiment_name: str = "Experiment") -> str:
    """
    Standalone function to generate a report from experiment results.
    
    Args:
        results: Dictionary containing test results or ExperimentResult instance
        output_path: Path to save the generated report
        experiment_name: Name of the experiment for the report
        
    Returns:
        str: Path to the saved report
    """
    # Check if we have an ExperimentResult object from the results module
    if hasattr(results, 'save_report') and callable(getattr(results, 'save_report')):
        # Use the built-in save_report method
        report_path = results.save_report(output_path, f"{experiment_name}.html")
        return str(report_path)
        
    # If not, use the normal report generator
    generator = ReportGenerator(verbose=True)
    return generator.generate_report(results, output_path, experiment_name)


# Don't try to patch the dict class directly, as it's an immutable type
# Instead, we'll use a different approach in report_exporter.py