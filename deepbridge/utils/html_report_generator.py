"""
Generate HTML reports from experiment test results dictionaries.
This module provides functions to convert experiment.run_tests() dictionaries into HTML reports.
"""

import os
import json
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

def generate_report_from_results(
    results: Dict[str, Any], 
    output_path: str, 
    experiment_name: str = "Experiment"
) -> str:
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
        
    Returns:
    --------
    str : Path to the saved report
    """
    # Try to convert to ExperimentResult first if results module is available
    try:
        from deepbridge.core.experiment.results import wrap_results
        result_obj = wrap_results(results)
        return result_obj.save_report(output_path, f"{experiment_name}.html")
    except ImportError:
        # Use the ReportGenerator directly if available
        try:
            from deepbridge.core.experiment.report_generator import generate_report_from_results as core_generator
            return core_generator(results, output_path, experiment_name)
        except ImportError:
            # Fall back to our own implementation
            return _generate_html_report(results, output_path, experiment_name)

def _generate_html_report(
    results: Dict[str, Any], 
    output_path: str, 
    experiment_name: str = "Experiment"
) -> str:
    """
    Generate a standalone HTML report from experiment test results dictionary.
    This is a simple implementation that works without additional dependencies.
    
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
    
    # Extract test types from results
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
            .tabs {{ display: flex; margin-bottom: 20px; }}
            .tab {{ padding: 10px 15px; cursor: pointer; background-color: #f2f2f2; margin-right: 5px; border-radius: 5px 5px 0 0; }}
            .tab.active {{ background-color: #1e88e5; color: white; }}
            .tab-content {{ display: none; }}
            .tab-content.active {{ display: block; }}
        </style>
        <script>
            function showTab(tabId) {
                // Hide all tab contents
                document.querySelectorAll('.tab-content').forEach(content => {
                    content.classList.remove('active');
                });
                
                // Remove active class from all tabs
                document.querySelectorAll('.tab').forEach(tab => {
                    tab.classList.remove('active');
                });
                
                // Show selected tab content
                document.getElementById(tabId).classList.add('active');
                
                // Activate tab button
                document.querySelectorAll('.tab').forEach(tab => {
                    if (tab.getAttribute('data-target') === tabId) {
                        tab.classList.add('active');
                    }
                });
            }
            
            // When the document is loaded, set up tab functionality
            document.addEventListener('DOMContentLoaded', function() {
                // Add click handlers to tabs
                document.querySelectorAll('.tab').forEach(tab => {
                    tab.addEventListener('click', function() {
                        showTab(this.getAttribute('data-target'));
                    });
                });
                
                // Show the first tab by default
                const firstTab = document.querySelector('.tab');
                if (firstTab) {
                    showTab(firstTab.getAttribute('data-target'));
                }
            });
        </script>
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
                <h2>Model Performance</h2>
                
                <div class="tabs">
                    <div class="tab active" data-target="primary-model-tab">Primary Model</div>
                    <div class="tab" data-target="alt-models-tab">Alternative Models</div>
                    <div class="tab" data-target="metrics-comparison-tab">Metrics Comparison</div>
                </div>
                
                <div id="primary-model-tab" class="tab-content active">
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
                
                # Add hyperparameters if available
                if 'hyperparameters' in primary_model:
                    html += '<h3>Hyperparameters</h3>'
                    html += '<table><tr><th>Parameter</th><th>Value</th></tr>'
                    
                    for param, value in primary_model['hyperparameters'].items():
                        html += f'<tr><td>{param}</td><td>{value}</td></tr>'
                    
                    html += '</table>'
            
            html += """
                </div>
                <div id="alt-models-tab" class="tab-content">
                    <h3>Alternative Models</h3>
            """
            
            # Alternative models metrics
            alt_models = {k: v for k, v in models_data.items() if k != 'primary_model'}
            if alt_models:
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
            else:
                html += '<p>No alternative models were evaluated.</p>'
            
            html += """
                </div>
                <div id="metrics-comparison-tab" class="tab-content">
                    <h3>Metrics Comparison</h3>
            """
            
            # Add comparison metrics from experiment_info if available
            if 'experiment_info' in results and 'comparison_metrics' in results['experiment_info']:
                comparison_metrics = results['experiment_info']['comparison_metrics']
                
                if isinstance(comparison_metrics, dict):
                    # Convert the comparison metrics to a suitable format for display
                    html += '<table><tr><th>Model</th><th>Accuracy</th><th>F1</th><th>Precision</th><th>Recall</th><th>ROC AUC</th></tr>'
                    
                    for model_name, metrics in comparison_metrics.items():
                        html += f'<tr><td>{model_name}</td>'
                        
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
                elif hasattr(comparison_metrics, 'to_html'):
                    # Use pandas DataFrame's to_html method if available
                    html += comparison_metrics.to_html(classes="table", float_format=lambda x: f"{x:.4f}")
                else:
                    html += '<p>Comparison metrics are available but in an unsupported format.</p>'
            else:
                html += '<p>No comparison metrics available.</p>'
            
            html += """
                </div>
            </div>
            """
    
    # Add tabs for each test type
    if tests_performed:
        html += """
        <div class="section">
            <h2>Test Results</h2>
            
            <div class="tabs">
        """
        
        # Create tab navigation
        for i, test_name in enumerate(tests_performed):
            active = ' active' if i == 0 else ''
            html += f'<div class="tab{active}" data-target="{test_name}-tab">{test_name.capitalize()}</div>'
        
        html += """
            </div>
        """
        
        # Create tab content sections
        for i, test_name in enumerate(tests_performed):
            active = ' active' if i == 0 else ''
            html += f'<div id="{test_name}-tab" class="tab-content{active}">'
            
            if test_name in results:
                test_data = results[test_name]
                
                # Handle nested structure (primary_model format)
                if 'primary_model' in test_data:
                    test_data = test_data['primary_model']
                
                # Rendering based on test type
                if test_name == 'robustness':
                    html += _format_robustness_results(test_data)
                elif test_name == 'uncertainty':
                    html += _format_uncertainty_results(test_data)
                elif test_name == 'resilience':
                    html += _format_resilience_results(test_data)
                elif test_name == 'hyperparameters':
                    html += _format_hyperparameter_results(test_data)
                else:
                    # Generic metrics table
                    html += '<h3>Test Metrics</h3>'
                    html += '<table><tr><th>Metric</th><th>Value</th></tr>'
                    
                    for key, value in test_data.items():
                        if isinstance(value, (int, float, str, bool)) and not isinstance(value, dict):
                            try:
                                value_str = f"{float(value):.4f}" if isinstance(value, (int, float)) else str(value)
                            except (TypeError, ValueError):
                                value_str = str(value)
                            html += f'<tr><td>{key.replace("_", " ").title()}</td><td>{value_str}</td></tr>'
                    
                    html += '</table>'
            
            html += '</div>'
        
        html += '</div>'
    
    # Add recommendations section
    html += """
        <div class="section">
            <h2>Recommendations</h2>
            <div class="test-result">
                <p>Based on the test results, here are some recommendations:</p>
                <ul>
    """
    
    # Add recommendation items based on test results
    recommendations = []
    
    if 'robustness' in results:
        robustness_data = results['robustness'].get('primary_model', results['robustness'])
        impact = robustness_data.get('avg_overall_impact', 0)
        
        if impact > 0.2:
            recommendations.append('<li><strong>Improve Model Robustness:</strong> Consider data augmentation or adversarial training to improve performance under perturbations.</li>')
        else:
            recommendations.append('<li><strong>Model Robustness:</strong> The model shows good robustness to feature perturbations.</li>')
    
    if 'uncertainty' in results:
        uncertainty_data = results['uncertainty'].get('primary_model', results['uncertainty'])
        calibration_error = uncertainty_data.get('expected_calibration_error', uncertainty_data.get('calibration_error', 0))
        
        if calibration_error > 0.1:
            recommendations.append('<li><strong>Improve Uncertainty Quantification:</strong> Consider using ensemble methods or temperature scaling to better calibrate prediction probabilities.</li>')
        else:
            recommendations.append('<li><strong>Uncertainty Calibration:</strong> The model shows good calibration of prediction probabilities.</li>')
    
    if 'resilience' in results:
        resilience_data = results['resilience'].get('primary_model', results['resilience'])
        resilience_index = resilience_data.get('resilience_index', 0)
        
        if resilience_index < 0.7:
            recommendations.append('<li><strong>Enhance Model Resilience:</strong> Train with more diverse data sources to improve resilience to distribution shifts.</li>')
        else:
            recommendations.append('<li><strong>Model Resilience:</strong> The model demonstrates good resilience to distribution shifts.</li>')
    
    if 'hyperparameters' in results:
        hyperparameter_data = results['hyperparameters'].get('primary_model', results['hyperparameters'])
        
        # Check if there's a sorted_importance key
        if 'sorted_importance' in hyperparameter_data and isinstance(hyperparameter_data['sorted_importance'], dict):
            important_params = list(hyperparameter_data['sorted_importance'].keys())[:3]
            important_params_str = ', '.join(important_params)
            recommendations.append(f'<li><strong>Hyperparameter Optimization:</strong> Focus on tuning these key parameters: {important_params_str}.</li>')
        else:
            recommendations.append('<li><strong>Hyperparameter Optimization:</strong> Conduct a more thorough hyperparameter search to potentially improve model performance.</li>')
    
    # If no specific recommendations, add a generic one
    if not recommendations:
        recommendations.append('<li><strong>Further Testing:</strong> Consider running additional tests to gain more insights into model behavior.</li>')
    
    # Add the recommendations to the HTML
    html += '\n'.join(recommendations)
    
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

def _format_robustness_results(results: Dict[str, Any]) -> str:
    """Format robustness test results for the HTML report."""
    html = '<h3>Robustness Overview</h3>'
    html += '<table><tr><th>Metric</th><th>Value</th></tr>'
    
    # Extract key metrics
    metrics = {
        'Base Score': results.get('base_score', 'N/A'),
        'Average Overall Impact': results.get('avg_overall_impact', 'N/A'),
        'Average Raw Impact': results.get('avg_raw_impact', 'N/A'),
        'Average Quantile Impact': results.get('avg_quantile_impact', 'N/A')
    }
    
    for metric, value in metrics.items():
        if value != 'N/A':
            try:
                value_str = f"{float(value):.4f}"
            except (TypeError, ValueError):
                value_str = str(value)
        else:
            value_str = 'N/A'
        html += f'<tr><td>{metric}</td><td>{value_str}</td></tr>'
    
    html += '</table>'
    
    # Raw perturbation results
    if 'raw' in results and 'by_level' in results['raw']:
        by_level = results['raw']['by_level']
        
        html += '<h3>Gaussian Noise Perturbation</h3>'
        html += '<table><tr><th>Level</th><th>Mean Score</th><th>Std</th><th>Impact</th></tr>'
        
        for level, level_data in sorted(by_level.items()):
            overall = level_data.get('overall_result', {})
            mean = overall.get('mean_score', 'N/A')
            std = overall.get('std_score', 'N/A')
            
            # Calculate impact if base_score is available
            impact = 'N/A'
            if 'base_score' in results and mean != 'N/A':
                try:
                    base_score = float(results['base_score'])
                    mean_score = float(mean)
                    impact = f"{abs(base_score - mean_score):.4f}"
                except (ValueError, TypeError):
                    pass
            
            html += f'<tr><td>{level}</td><td>{mean}</td><td>{std}</td><td>{impact}</td></tr>'
        
        html += '</table>'
    
    # Quantile perturbation results
    if 'quantile' in results and 'by_level' in results['quantile']:
        by_level = results['quantile']['by_level']
        
        html += '<h3>Quantile Perturbation</h3>'
        html += '<table><tr><th>Level</th><th>Mean Score</th><th>Std</th><th>Impact</th></tr>'
        
        for level, level_data in sorted(by_level.items()):
            overall = level_data.get('overall_result', {})
            mean = overall.get('mean_score', 'N/A')
            std = overall.get('std_score', 'N/A')
            
            # Calculate impact if base_score is available
            impact = 'N/A'
            if 'base_score' in results and mean != 'N/A':
                try:
                    base_score = float(results['base_score'])
                    mean_score = float(mean)
                    impact = f"{abs(base_score - mean_score):.4f}"
                except (ValueError, TypeError):
                    pass
            
            html += f'<tr><td>{level}</td><td>{mean}</td><td>{std}</td><td>{impact}</td></tr>'
        
        html += '</table>'
    
    # Feature importance
    if 'feature_importance' in results:
        feature_importance = results['feature_importance']
        
        html += '<h3>Feature Importance</h3>'
        html += '<table><tr><th>Feature</th><th>Importance</th></tr>'
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Display top 10 features
        for feature, value in sorted_features[:10]:
            try:
                value_str = f"{float(value):.4f}"
            except (TypeError, ValueError):
                value_str = str(value)
            html += f'<tr><td>{feature}</td><td>{value_str}</td></tr>'
        
        html += '</table>'
    
    return html

def _format_uncertainty_results(results: Dict[str, Any]) -> str:
    """Format uncertainty test results for the HTML report."""
    html = '<h3>Uncertainty Metrics</h3>'
    html += '<table><tr><th>Metric</th><th>Value</th></tr>'
    
    # Extract key metrics
    metrics = {
        'Expected Calibration Error': results.get('expected_calibration_error', results.get('calibration_error', 'N/A')),
        'Maximum Calibration Error': results.get('max_calibration_error', 'N/A'),
        'Brier Score': results.get('brier_score', 'N/A')
    }
    
    for metric, value in metrics.items():
        if value != 'N/A':
            try:
                value_str = f"{float(value):.4f}"
            except (TypeError, ValueError):
                value_str = str(value)
        else:
            value_str = 'N/A'
        html += f'<tr><td>{metric}</td><td>{value_str}</td></tr>'
    
    html += '</table>'
    
    # Coverage statistics
    if 'coverage_stats' in results:
        coverage_stats = results['coverage_stats']
        
        html += '<h3>Prediction Intervals</h3>'
        html += '<table><tr><th>Alpha</th><th>Expected Coverage</th><th>Actual Coverage</th><th>Avg Width</th></tr>'
        
        for alpha, stats in sorted(coverage_stats.items()):
            expected = 1 - float(alpha) if not isinstance(alpha, str) else f"1 - {alpha}"
            coverage = stats.get('coverage', 'N/A')
            width = stats.get('avg_width', 'N/A')
            
            html += f'<tr><td>{alpha}</td><td>{expected}</td><td>{coverage}</td><td>{width}</td></tr>'
        
        html += '</table>'
    
    return html

def _format_resilience_results(results: Dict[str, Any]) -> str:
    """Format resilience test results for the HTML report."""
    html = '<h3>Resilience Metrics</h3>'
    html += '<table><tr><th>Metric</th><th>Value</th></tr>'
    
    # Extract key metrics
    base_score = results.get('base_score', 'N/A')
    resilience_index = results.get('resilience_index', 'N/A')
    
    html += f'<tr><td>Base Score</td><td>{base_score}</td></tr>'
    html += f'<tr><td>Resilience Index</td><td>{resilience_index}</td></tr>'
    
    for key, value in results.items():
        if isinstance(value, (int, float)) and key not in ['base_score', 'resilience_index']:
            try:
                value_str = f"{float(value):.4f}"
            except (TypeError, ValueError):
                value_str = str(value)
            html += f'<tr><td>{key.replace("_", " ").title()}</td><td>{value_str}</td></tr>'
    
    html += '</table>'
    
    # Drift results
    drift_types = ['covariate', 'label', 'concept']
    for drift_type in drift_types:
        if drift_type in results and isinstance(results[drift_type], dict):
            drift_data = results[drift_type]
            
            html += f'<h3>{drift_type.capitalize()} Drift Results</h3>'
            html += '<table><tr><th>Intensity</th><th>Score</th><th>Impact</th></tr>'
            
            for intensity, data in sorted(drift_data.items()):
                if isinstance(data, dict):
                    score = data.get('score', 'N/A')
                    impact = data.get('impact', 'N/A')
                else:
                    score = data
                    impact = 'N/A'
                    
                html += f'<tr><td>{intensity}</td><td>{score}</td><td>{impact}</td></tr>'
            
            html += '</table>'
    
    return html

def _format_hyperparameter_results(results: Dict[str, Any]) -> str:
    """Format hyperparameter test results for the HTML report."""
    html = '<h3>Hyperparameter Importance</h3>'
    
    # Check for sorted importance
    if 'sorted_importance' in results and isinstance(results['sorted_importance'], dict):
        importance = results['sorted_importance']
        
        html += '<table><tr><th>Parameter</th><th>Importance</th></tr>'
        
        for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            try:
                value_str = f"{float(imp):.4f}"
            except (TypeError, ValueError):
                value_str = str(imp)
            html += f'<tr><td>{param}</td><td>{value_str}</td></tr>'
        
        html += '</table>'
    elif isinstance(results, dict):
        # Try to find parameter importance values in the results
        importance_dict = {}
        for param, value in results.items():
            if isinstance(value, (int, float)) and not param.startswith('_'):
                importance_dict[param] = value
        
        if importance_dict:
            html += '<table><tr><th>Parameter</th><th>Importance</th></tr>'
            
            for param, imp in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True):
                try:
                    value_str = f"{float(imp):.4f}"
                except (TypeError, ValueError):
                    value_str = str(imp)
                html += f'<tr><td>{param}</td><td>{value_str}</td></tr>'
            
            html += '</table>'
        else:
            html += '<p>No hyperparameter importance data available.</p>'
    else:
        html += '<p>No hyperparameter importance data available.</p>'
    
    # Best parameters
    if 'best_params' in results and isinstance(results['best_params'], dict):
        best_params = results['best_params']
        
        html += '<h3>Best Parameters</h3>'
        html += '<table><tr><th>Parameter</th><th>Value</th></tr>'
        
        for param, value in best_params.items():
            html += f'<tr><td>{param}</td><td>{value}</td></tr>'
        
        html += '</table>'
    
    return html