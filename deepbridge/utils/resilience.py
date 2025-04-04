"""
Utility functions for resilience testing.
"""

import numpy as np
import io
import base64
from typing import Dict, List, Optional, Union, Any

def run_resilience_tests(dataset, config_name='quick', metric='auc', verbose=True):
    """
    Run resilience tests on a dataset to evaluate model performance under distribution shifts.
    
    Parameters:
    -----------
    dataset : DBDataset
        Dataset object containing training/test data and model
    config_name : str
        Name of the configuration to use: 'quick', 'medium', or 'full'
    metric : str
        Performance metric to use for evaluation ('auc', 'f1', 'accuracy', etc.)
    verbose : bool
        Whether to print progress information
        
    Returns:
    --------
    dict : Test results with detailed resilience metrics
    """
    from deepbridge.validation.wrappers.resilience_suite import ResilienceSuite
    
    # Initialize resilience suite
    resilience = ResilienceSuite(dataset, verbose=verbose, metric=metric)
    
    # Configure and run tests
    results = resilience.config(config_name).run()
    
    if verbose:
        print(f"\nResilience Test Summary:")
        print(f"Overall resilience score: {results.get('resilience_score', 0):.3f}")
        
        # Print alpha-specific results
        for alpha, alpha_data in sorted(results.get('distribution_shift', {}).get('by_alpha', {}).items()):
            print(f"Alpha = {alpha}: Average performance gap: {alpha_data.get('avg_performance_gap', 0):.3f}")
    
    return results

def resilience_report_to_html(results, include_details=True):
    """
    Generate HTML report from resilience results.
    
    Parameters:
    -----------
    results : dict
        Resilience test results from run_resilience_tests
    include_details : bool
        Whether to include detailed information in the report
        
    Returns:
    --------
    str : HTML report content
    """
    # Basic report structure
    html = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<title>Resilience Test Report</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 20px; }",
        ".summary { background-color: #f5f5f5; padding: 15px; border-radius: 5px; }",
        ".plot-container { margin: 20px 0; }",
        "table { border-collapse: collapse; width: 100%; }",
        "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
        "th { background-color: #f2f2f2; }",
        "tr:nth-child(even) { background-color: #f9f9f9; }",
        ".feature-importance { margin-top: 20px; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Model Resilience Test Report</h1>"
    ]
    
    # Summary section
    html.append("<div class='summary'>")
    html.append("<h2>Summary</h2>")
    html.append(f"<p><strong>Overall Resilience Score:</strong> {results.get('resilience_score', 0):.3f}</p>")
    html.append("</div>")
    
    # Alpha-specific results
    html.append("<h2>Results by Worst Sample Ratio (Alpha)</h2>")
    html.append("<table>")
    html.append("<tr><th>Alpha</th><th>Average Performance Gap</th></tr>")
    
    for alpha, alpha_data in sorted(results.get('distribution_shift', {}).get('by_alpha', {}).items()):
        html.append("<tr>")
        html.append(f"<td>{alpha}</td>")
        html.append(f"<td>{alpha_data.get('avg_performance_gap', 0):.3f}</td>")
        html.append("</tr>")
    
    html.append("</table>")
    
    # Feature importance by distance metric
    for dm, dm_data in results.get('distribution_shift', {}).get('by_distance_metric', {}).items():
        html.append(f"<h2>Feature Importance ({dm} Distance Metric)</h2>")
        html.append("<table>")
        html.append("<tr><th>Feature</th><th>Average Distance</th></tr>")
        
        # Sort features by distance
        top_features = sorted(dm_data.get('top_features', {}).items(), 
                            key=lambda x: x[1], reverse=True)
        
        for feature, value in top_features[:10]:
            html.append("<tr>")
            html.append(f"<td>{feature}</td>")
            html.append(f"<td>{value:.3f}</td>")
            html.append("</tr>")
        
        html.append("</table>")
    
    # Detailed test results
    if include_details:
        html.append("<h2>Detailed Test Results</h2>")
        
        for i, result in enumerate(results.get('distribution_shift', {}).get('all_results', []), 1):
            html.append(f"<h3>Test {i}</h3>")
            html.append("<table>")
            html.append("<tr><th>Parameter</th><th>Value</th></tr>")
            
            html.append("<tr><td>Method</td><td>Distribution Shift Analysis</td></tr>")
            html.append(f"<tr><td>Alpha</td><td>{result.get('alpha', 0)}</td></tr>")
            html.append(f"<tr><td>Performance Metric</td><td>{result.get('metric', '')}</td></tr>")
            html.append(f"<tr><td>Distance Metric</td><td>{result.get('distance_metric', '')}</td></tr>")
            html.append(f"<tr><td>Worst Samples Score</td><td>{result.get('worst_metric', 0):.3f}</td></tr>")
            html.append(f"<tr><td>Remaining Samples Score</td><td>{result.get('remaining_metric', 0):.3f}</td></tr>")
            html.append(f"<tr><td>Performance Gap</td><td>{result.get('performance_gap', 0):.3f}</td></tr>")
            html.append(f"<tr><td>Worst Sample Count</td><td>{result.get('worst_sample_count', 0)}</td></tr>")
            html.append(f"<tr><td>Remaining Sample Count</td><td>{result.get('remaining_sample_count', 0)}</td></tr>")
            
            html.append("</table>")
            
            # Top features for this test
            html.append("<h4>Top Features by Distance</h4>")
            html.append("<table>")
            html.append("<tr><th>Feature</th><th>Distance</th></tr>")
            
            top_features = sorted(result.get('feature_distances', {}).get('top_features', {}).items(), 
                                key=lambda x: x[1], reverse=True)
            
            for feature, value in top_features[:5]:
                html.append("<tr>")
                html.append(f"<td>{feature}</td>")
                html.append(f"<td>{value:.3f}</td>")
                html.append("</tr>")
            
            html.append("</table>")
    
    # Close HTML
    html.append("</body>")
    html.append("</html>")
    
    return "\n".join(html)

def compare_models_resilience(results_dict):
    """
    Compare resilience of multiple models.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary of model name: results pairs from run_resilience_tests
        
    Returns:
    --------
    dict : Comparison data
    """
    comparison = {
        'model_names': [],
        'resilience_scores': [],
        'performance_gaps': {},
        'feature_importance': {}
    }
    
    for model_name, results in results_dict.items():
        comparison['model_names'].append(model_name)
        comparison['resilience_scores'].append(results.get('resilience_score', 0))
        
        # Collect performance gaps by alpha
        for alpha, alpha_data in results.get('distribution_shift', {}).get('by_alpha', {}).items():
            if alpha not in comparison['performance_gaps']:
                comparison['performance_gaps'][alpha] = []
            comparison['performance_gaps'][alpha].append({
                'model': model_name,
                'gap': alpha_data.get('avg_performance_gap', 0)
            })
            
        # Collect top features by distance metric
        for dm, dm_data in results.get('distribution_shift', {}).get('by_distance_metric', {}).items():
            if dm not in comparison['feature_importance']:
                comparison['feature_importance'][dm] = {}
            
            # Add top features for this model and distance metric
            for feature, value in dm_data.get('top_features', {}).items():
                if feature not in comparison['feature_importance'][dm]:
                    comparison['feature_importance'][dm][feature] = []
                comparison['feature_importance'][dm][feature].append({
                    'model': model_name,
                    'importance': value
                })
    
    return comparison