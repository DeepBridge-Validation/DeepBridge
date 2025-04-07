"""
Generate HTML reports from experiment test results.
This module provides a simple function to convert test results dictionaries into HTML reports.
"""

import os
import json
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union

def generate_report_from_results(results: Dict[str, Any], output_path: str, experiment_name: str = "Experiment", report_type: str = None) -> str:
    """
    Generate an HTML report from experiment test results dictionary.
    
    Parameters:
    -----------
    results : Dict[str, Any]
        Dictionary containing test results from experiment.run_tests()
    output_path : str
        Path to save the generated report
    experiment_name : str
        Name of the experiment for the report
    report_type : str, optional
        Type of report to generate (robustness, uncertainty, etc.). If specified,
        generates a specialized report using the appropriate generator.
        
    Returns:
    --------
    str : Path to the saved report
    """
    # If report_type is specified, try to generate specialized report
    if report_type:
        report_type = report_type.lower()
        try:
            if report_type == 'robustness':
                robustness_data = results.get('robustness', results)
                from deepbridge.reporting.plots.robustness.robustness_report_generator import generate_robustness_report
                return generate_robustness_report(
                    robustness_data,
                    output_path,
                    model_name="Primary Model",
                    experiment_info=results
                )
                
            elif report_type == 'uncertainty':
                uncertainty_data = results.get('uncertainty', results)
                from deepbridge.reporting.plots.uncertainty.uncertainty_report_generator import generate_uncertainty_report
                return generate_uncertainty_report(
                    uncertainty_data,
                    output_path,
                    model_name="Primary Model",
                    experiment_info=results
                )
                
            # Add other report types here as needed
        except ImportError as e:
            print(f"Could not import specialized report generator for {report_type}: {e}")
            # Continue to standard report generation
        except Exception as e:
            import traceback
            print(f"Error generating specialized {report_type} report: {e}")
            print(traceback.format_exc())
            # Continue to standard report generation
            
    # First, try to use the official report generator if available
    try:
        from deepbridge.core.experiment.report_generator import generate_report_from_results as core_generator
        return core_generator(results, output_path, experiment_name)
    except ImportError:
        # Fall back to using the html_report_generator if available
        try:
            from deepbridge.utils.html_report_generator import generate_report_from_results as util_generator
            return util_generator(results, output_path, experiment_name)
        except ImportError:
            # Use our own implementation if neither is available
            return _generate_html_report(results, output_path, experiment_name)

def _generate_html_report(results: Dict[str, Any], output_path: str, experiment_name: str = "Experiment") -> str:
    """
    Generate a basic HTML report from experiment test results dictionary.
    This is a simple implementation for standalone use.
    
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
    
    # Extract test types
    tests_performed = []
    for key in results:
        if key in ['robustness', 'uncertainty', 'resilience', 'hyperparameters']:
            tests_performed.append(key)
    
    # Build report HTML
    html = f"""<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{experiment_name} - Validation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .header {{ background: linear-gradient(135deg, #0062cc 0%, #1e88e5 100%); color: white; padding: 20px; margin-bottom: 20px; border-radius: 5px; }}
            .section {{ margin-bottom: 30px; padding: 20px; background: #fff; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; font-weight: bold; }}
            .metric-good {{ color: #28a745; font-weight: bold; }}
            .metric-medium {{ color: #ffc107; font-weight: bold; }}
            .metric-poor {{ color: #dc3545; font-weight: bold; }}
            .test-result {{ margin-bottom: 20px; padding: 15px; border: 1px solid #eee; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>{experiment_name} - Validation Report</h1>
                <p>Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Experiment Overview</h2>
                <table>
                    <tr><th>Attribute</th><th>Value</th></tr>
                    <tr><td>Experiment Type</td><td>{results.get('experiment_type', 'Unknown')}</td></tr>
                    <tr><td>Tests Performed</td><td>{', '.join(tests_performed)}</td></tr>
                    <tr><td>Dataset Size</td><td>{results.get('dataset_size', 'Unknown')}</td></tr>
                </table>
            </div>
    """
    
    # Add model metrics section (from initial_results if available)
    if 'initial_results' in results:
        initial_results = results['initial_results']
        if 'models' in initial_results:
            models_data = initial_results['models']
            
            html += """
            <div class="section">
                <h2>Model Performance Metrics</h2>
                <div class="test-result">
            """
            
            # Primary model metrics
            if 'primary_model' in models_data:
                primary_model = models_data['primary_model']
                primary_metrics = primary_model.get('metrics', {})
                
                html += '<h3>Primary Model Metrics</h3>'
                html += '<table><tr><th>Metric</th><th>Value</th></tr>'
                
                for metric, value in primary_metrics.items():
                    # Handle numpy float types
                    try:
                        value_str = f"{float(value):.4f}"
                    except (TypeError, ValueError):
                        value_str = str(value)
                    
                    html += f'<tr><td>{metric.replace("_", " ").title()}</td><td>{value_str}</td></tr>'
                
                html += '</table>'
            
            # Alternative models metrics
            alt_models = {k: v for k, v in models_data.items() if k != 'primary_model'}
            if alt_models:
                html += '<h3>Alternative Models Metrics</h3>'
                html += '<table><tr><th>Model</th><th>Accuracy</th><th>F1</th><th>Precision</th><th>Recall</th><th>ROC AUC</th></tr>'
                
                for model_name, model_data in alt_models.items():
                    metrics = model_data.get('metrics', {})
                    html += f'<tr><td>{model_name}</td>'
                    
                    # Add each metric for the model
                    for metric in ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']:
                        value = metrics.get(metric, '-')
                        if value != '-':
                            try:
                                value_str = f"{float(value):.4f}"
                            except (TypeError, ValueError):
                                value_str = str(value)
                        else:
                            value_str = '-'
                        html += f'<td>{value_str}</td>'
                    
                    html += '</tr>'
                
                html += '</table>'
            
            html += """
                </div>
            </div>
            """
    
    # Add sections for each test type
    for test_name in tests_performed:
        if test_name in results:
            test_data = results[test_name]
            
            # Handle nested structure (primary_model format)
            if 'primary_model' in test_data:
                test_data = test_data['primary_model']
            
            # Start section
            html += f"""
            <div class="section">
                <h2>{test_name.capitalize()} Test Results</h2>
                <div class="test-result">
            """
            
            # Generic metrics table
            html += '<table><tr><th>Metric</th><th>Value</th></tr>'
            
            for key, value in test_data.items():
                if isinstance(value, (int, float, str, bool)) and not isinstance(value, dict):
                    try:
                        value_str = f"{float(value):.4f}" if isinstance(value, (int, float)) else str(value)
                    except (TypeError, ValueError):
                        value_str = str(value)
                    html += f'<tr><td>{key.replace("_", " ").title()}</td><td>{value_str}</td></tr>'
            
            html += '</table>'
            
            # Feature importance for robustness
            if test_name == 'robustness' and 'feature_importance' in test_data:
                html += '<h3>Feature Importance</h3><table><tr><th>Feature</th><th>Importance</th></tr>'
                
                # Sort features by importance
                feature_importance = test_data['feature_importance']
                sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
                
                # Display top 10 features
                for feature, value in sorted_features[:10]:
                    try:
                        value_str = f"{float(value):.4f}"
                    except (TypeError, ValueError):
                        value_str = str(value)
                    html += f'<tr><td>{feature}</td><td>{value_str}</td></tr>'
                
                html += '</table>'
            
            # Close the section
            html += """
                </div>
            </div>
            """
    
    # Add recommendations section
    html += """
        <div class="section">
            <h2>Recommendations</h2>
            <div class="test-result">
                <p>Based on the test results, here are some recommendations:</p>
                <ul>
    """
    
    # Add recommendation items based on which tests were performed
    if 'robustness' in results:
        html += '<li><strong>Improve Model Robustness:</strong> Consider data augmentation or adversarial training.</li>'
    
    if 'uncertainty' in results:
        html += '<li><strong>Uncertainty Quantification:</strong> Consider using ensemble methods.</li>'
    
    if 'resilience' in results:
        html += '<li><strong>Enhance Model Resilience:</strong> Train with diverse data sources.</li>'
    
    if 'hyperparameters' in results:
        html += '<li><strong>Hyperparameter Optimization:</strong> Focus on the most impactful hyperparameters.</li>'
    
    # Close the HTML
    html += """
                </ul>
            </div>
        </div>
    </div>
    </body>
    </html>
    """
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    return str(output_path)