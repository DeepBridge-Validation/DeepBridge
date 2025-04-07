"""
Report exporter for experiment results.
This module provides a simple interface to export experiment results to HTML reports.
"""

import datetime
import sys
from typing import Dict, Any
from pathlib import Path

# Check for dependencies
from deepbridge.core.experiment.dependencies import check_dependencies, get_install_command

# Import the report generator
try:
    from deepbridge.core.experiment.report_generator import generate_report_from_results
except ImportError as e:
    # Handle import error with more user-friendly message
    print(f"Error importing report_generator: {e}")
    all_installed, missing_required, _ = check_dependencies()
    if not all_installed:
        print(f"\nMissing required dependencies: {', '.join(missing_required)}")
        print(f"Please install them using: {get_install_command(missing_required)}")
    sys.exit(1)

def export_report(results: Dict[str, Any], output_path: str, experiment_name: str = "Experiment") -> str:
    """
    Export experiment results to an HTML report.
    
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
    # Check if required dependencies are installed
    all_installed, missing_required, missing_optional, version_issues = check_dependencies()
    if not all_installed:
        print("⚠️ Warning: Some required dependencies are missing.")
        print(f"Missing required dependencies: {', '.join(missing_required)}")
        print(f"Please install them using: {get_install_command(missing_required)}")
        print("Attempting to generate report with limited functionality...")
    elif missing_optional:
        print("ℹ️ Info: Some optional dependencies are missing.")
        print(f"Missing optional dependencies: {', '.join(missing_optional)}")
        print("Full functionality may not be available.")
    
    # Ensure results dictionary has all required metadata
    if 'experiment_type' not in results:
        results['experiment_type'] = 'binary_classification'
    
    if 'tests_performed' not in results:
        # Extract test types from the results dictionary
        tests_performed = []
        for key in results:
            if key in ['robustness', 'uncertainty', 'resilience', 'hyperparameters']:
                tests_performed.append(key)
        results['tests_performed'] = tests_performed
    
    if not isinstance(results['tests_performed'], list):
        results['tests_performed'] = [results['tests_performed']]
    
    # Add extra metadata to handle common Jinja2 template variables
    enhanced_results = results.copy()
    report_date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Create default structures for expected test data
    if 'robustness' not in enhanced_results:
        enhanced_results['robustness'] = {}
    
    # Check if robustness data has nested structure (primary_model format)
    robustness_data = enhanced_results['robustness']
    
    # Handle nested robustness structure
    if 'primary_model' in robustness_data:
        # Extract primary model data
        primary_data = robustness_data['primary_model']
        
        # Flatten the structure to make it compatible with the template
        # Copy primary model results to top level
        for key, value in primary_data.items():
            robustness_data[key] = value
            
        # Extract feature importance if it exists
        if 'feature_impact' in primary_data:
            robustness_data['feature_importance'] = primary_data['feature_impact']
    
    # Add defaults for commonly accessed robustness keys
    for key in ['base_score', 'avg_overall_impact', 'avg_raw_impact', 'avg_quantile_impact']:
        if key not in robustness_data:
            robustness_data[key] = 0.0
    
    if 'feature_importance' not in robustness_data:
        robustness_data['feature_importance'] = {}
    
    # Add safe defaults for common stats subdictionaries
    for subdict in ['raw', 'quantile']:
        if subdict not in robustness_data:
            robustness_data[subdict] = {'by_level': {}}
    
    # Handle uncertainty results
    if 'uncertainty' not in enhanced_results:
        enhanced_results['uncertainty'] = {}
    elif 'primary_model' in enhanced_results['uncertainty']:
        # Flatten the structure to make it compatible with the template
        primary_data = enhanced_results['uncertainty']['primary_model']
        for key, value in primary_data.items():
            enhanced_results['uncertainty'][key] = value
    
    # Handle resilience results  
    if 'resilience' not in enhanced_results:
        enhanced_results['resilience'] = {}
    elif 'primary_model' in enhanced_results['resilience']:
        # Flatten the structure to make it compatible with the template
        primary_data = enhanced_results['resilience']['primary_model']
        for key, value in primary_data.items():
            enhanced_results['resilience'][key] = value
    
    # Handle hyperparameter results
    if 'hyperparameters' not in enhanced_results:
        enhanced_results['hyperparameters'] = {}
    elif 'primary_model' in enhanced_results['hyperparameters']:
        # Flatten the structure to make it compatible with the template
        primary_data = enhanced_results['hyperparameters']['primary_model']
        for key, value in primary_data.items():
            enhanced_results['hyperparameters'][key] = value
    
    # Add common template variables
    enhanced_results.update({
        # These are common variables used in templates
        'model_name': 'Primary Model',
        'report_date': report_date,
        'generation_time': report_date,
        'models_evaluated': enhanced_results.get('model_name', 'Primary Model'),
        'generation_date': report_date,
        'boxplot_chart': '{"data":[], "layout":{}}',
        'feature_importance_chart': '{"data":[], "layout":{}}',
        'perturbation_methods_chart': '{"data":[], "layout":{}}',
        'perturbation_levels': [],
        'model_metrics': [{'name': 'Main Model', 'robustness_index': '0.0', 'baseline': '0.0', 'perturbed': '0.0', 'drop': '0', 'color': '#3a6ea5'}],
        'recommendations': [{'title': 'Improve Model', 'content': 'Consider further model improvements based on test results.'}],
        'dataset_size': enhanced_results.get('dataset_size', 'Unknown'),
        'experiment_type': enhanced_results.get('experiment_type', 'Classification'),
        'key_finding': 'Model evaluation completed successfully',
        'summary_text': 'See sections below for detailed analysis',
    })
    
    try:
        # Try to generate the full report
        return generate_report_from_results(enhanced_results, output_path, experiment_name)
    except Exception as primary_error:
        print(f"❌ Error generating report: {str(primary_error)}")
        
        try:
            # Try to create a direct simple HTML report as fallback
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create simple HTML without Jinja2
            sections = []
            
            # Add test sections
            for test_name, test_data in results.items():
                if test_name in ['robustness', 'uncertainty', 'resilience', 'hyperparameters']:
                    section = f"<div class='section'><h2>{test_name.capitalize()} Test Results</h2><table><tr><th>Metric</th><th>Value</th></tr>"
                    
                    # Add metrics from this test
                    for key, value in test_data.items():
                        if isinstance(value, (int, float, str, bool)):
                            section += f"<tr><td>{key}</td><td>{value}</td></tr>"
                    
                    section += "</table></div>"
                    sections.append(section)
            
            # Generate a very basic HTML report
            html = f"""<!DOCTYPE html>
            <html>
            <head>
                <title>{experiment_name} - Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                    h1, h2, h3 {{ color: #333; }}
                    .section {{ margin: 20px 0; padding: 15px; border: 1px solid #eee; border-radius: 5px; }}
                    table {{ width: 100%; border-collapse: collapse; }}
                    th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <h1>{experiment_name} - Validation Report</h1>
                <p>Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <div class="section">
                    <h2>Experiment Information</h2>
                    <table>
                        <tr><th>Attribute</th><th>Value</th></tr>
                        <tr><td>Experiment Type</td><td>{results.get('experiment_type', 'Unknown')}</td></tr>
                        <tr><td>Tests Performed</td><td>{', '.join(results.get('tests_performed', []))}</td></tr>
                    </table>
                </div>
                
                {''.join(sections)}
                
                <div class="section">
                    <h2>Recommendations</h2>
                    <p>Based on the results, consider further analysis and model improvements.</p>
                </div>
            </body>
            </html>
            """
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html)
            
            print(f"✅ Created simplified report at: {output_path}")
            return str(output_path)
            
        except Exception as e:
            print(f"❌ Error creating simplified report: {str(e)}")
            
            # Create the most basic emergency report when everything else fails
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Generate a very basic HTML report
            html = f"""<!DOCTYPE html>
            <html>
            <head>
                <title>{experiment_name} - Emergency Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #333; }}
                    pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; }}
                </style>
            </head>
            <body>
                <h1>{experiment_name} - Emergency Report</h1>
                <p>Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Note:</strong> This is an emergency report generated because of errors in the report generation process.</p>
                <h2>Error Information</h2>
                <pre>Primary error: {str(primary_error)}\nSecondary error: {str(e)}</pre>
                <h2>Raw Test Results</h2>
                <pre>{str(results)}</pre>
            </body>
            </html>
            """
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html)
            
            print(f"✅ Created emergency report at: {output_path}")
            return str(output_path)

# Create a wrapper class that extends dict functionality
class ResultsDict(dict):
    """
    A dictionary class with additional methods for experiment results.
    
    This class extends the standard dict with methods for report generation
    and other experiment-related functionality.
    """
    
    def save_report(self, output_path: str, experiment_name: str = "Experiment") -> str:
        """
        Save the dictionary as an HTML report.
        
        Parameters:
        -----------
        output_path : str
            Path to save the report
        experiment_name : str
            Name of the experiment for the report
            
        Returns:
        --------
        str : Path to the saved report
        """
        return export_report(self, output_path, experiment_name)

def wrap_results(results: dict) -> ResultsDict:
    """
    Wrap a regular dictionary in a ResultsDict to add report generation capabilities.
    
    Parameters:
    -----------
    results : dict
        Original results dictionary
        
    Returns:
    --------
    ResultsDict : Enhanced dictionary with save_report method
    """
    return ResultsDict(results)