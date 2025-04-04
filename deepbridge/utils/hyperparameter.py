"""
Utility functions for hyperparameter importance testing.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Any

def run_hyperparameter_tests(dataset, config_name='quick', metric='accuracy', verbose=True):
    """
    Run hyperparameter importance tests on a dataset to identify the most influential parameters.
    
    Parameters:
    -----------
    dataset : DBDataset
        Dataset object containing training data and model
    config_name : str
        Name of the configuration to use: 'quick', 'medium', or 'full'
    metric : str
        Performance metric to use for evaluation ('accuracy', 'auc', 'f1', etc.)
    verbose : bool
        Whether to print progress information
        
    Returns:
    --------
    dict : Test results with detailed hyperparameter importance metrics
    """
    from deepbridge.validation.wrappers.hyperparameter_suite import HyperparameterSuite
    
    # Initialize hyperparameter suite
    hyperparameter = HyperparameterSuite(dataset, verbose=verbose, metric=metric)
    
    # Configure and run tests
    results = hyperparameter.config(config_name).run()
    
    if verbose:
        print(f"\nHyperparameter Importance Summary:")
        print(f"Suggested tuning order:")
        for i, param in enumerate(results.get('tuning_order', []), 1):
            importance = results.get('sorted_importance', {}).get(param, 0)
            print(f"  {i}. {param} (importance: {importance:.4f})")
    
    return results

def hyperparameter_report_to_html(results, include_details=True):
    """
    Generate HTML report from hyperparameter importance results.
    
    Parameters:
    -----------
    results : dict
        Hyperparameter importance test results from run_hyperparameter_tests
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
        "<title>Hyperparameter Importance Report</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 20px; }",
        ".summary { background-color: #f5f5f5; padding: 15px; border-radius: 5px; }",
        ".plot-container { margin: 20px 0; }",
        "table { border-collapse: collapse; width: 100%; }",
        "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
        "th { background-color: #f2f2f2; }",
        "tr:nth-child(even) { background-color: #f9f9f9; }",
        ".tuning-order { margin-top: 20px; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Hyperparameter Importance Report</h1>"
    ]
    
    # Summary section
    html.append("<div class='summary'>")
    html.append("<h2>Summary</h2>")
    html.append("<p>This report shows the relative importance of each hyperparameter for model performance.</p>")
    html.append("</div>")
    
    # Importance scores table
    html.append("<h2>Hyperparameter Importance Scores</h2>")
    html.append("<table>")
    html.append("<tr><th>Hyperparameter</th><th>Importance Score</th></tr>")
    
    for param, score in results.get('sorted_importance', {}).items():
        html.append("<tr>")
        html.append(f"<td>{param}</td>")
        html.append(f"<td>{score:.4f}</td>")
        html.append("</tr>")
    
    html.append("</table>")
    
    # Suggested tuning order
    html.append("<div class='tuning-order'>")
    html.append("<h2>Suggested Hyperparameter Tuning Order</h2>")
    html.append("<ol>")
    
    for param in results.get('tuning_order', []):
        importance = results.get('sorted_importance', {}).get(param, 0)
        html.append(f"<li><strong>{param}</strong> (importance: {importance:.4f})</li>")
    
    html.append("</ol>")
    html.append("</div>")
    
    # Detailed test results
    if include_details:
        html.append("<h2>Detailed Test Results</h2>")
        
        for config_key, config_results in results.get('importance', {}).get('by_config', {}).items():
            html.append(f"<h3>Configuration: {config_key}</h3>")
            html.append("<table>")
            html.append("<tr><th>Parameter</th><th>Value</th></tr>")
            
            html.append("<tr><td>Method</td><td>Hyperparameter Importance</td></tr>")
            html.append(f"<tr><td>CV Folds</td><td>{config_results.get('cv')}</td></tr>")
            html.append(f"<tr><td>Subsamples</td><td>{config_results.get('n_subsamples')}</td></tr>")
            html.append(f"<tr><td>Subsample Size</td><td>{config_results.get('subsample_size')}</td></tr>")
            
            html.append("</table>")
            
            # Parameter importance for this configuration
            html.append("<h4>Importance Scores</h4>")
            html.append("<table>")
            html.append("<tr><th>Parameter</th><th>Importance</th></tr>")
            
            for param, score in config_results.get('sorted_importance', {}).items():
                html.append("<tr>")
                html.append(f"<td>{param}</td>")
                html.append(f"<td>{score:.4f}</td>")
                html.append("</tr>")
            
            html.append("</table>")
            
            # Performance data for most important parameter
            if config_results.get('sorted_importance') and config_results.get('performance_data'):
                top_param = next(iter(config_results.get('sorted_importance', {})), None)
                if top_param:
                    html.append(f"<h4>Performance for different values of {top_param}</h4>")
                    html.append("<table>")
                    html.append("<tr><th>Value</th><th>Performance</th></tr>")
                    
                    perf_data = config_results.get('performance_data', {}).get(top_param, {})
                    for value, score in perf_data.items():
                        html.append("<tr>")
                        html.append(f"<td>{value}</td>")
                        html.append(f"<td>{score:.4f}</td>")
                        html.append("</tr>")
                    
                    html.append("</table>")
    
    # Close HTML
    html.append("</body>")
    html.append("</html>")
    
    return "\n".join(html)

def compare_hyperparameter_importance(results_dict):
    """
    Compare hyperparameter importance across multiple models.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary of model name: results pairs from run_hyperparameter_tests
        
    Returns:
    --------
    dict : Comparison data
    """
    comparison = {
        'model_names': [],
        'common_params': set(),
        'param_importance': {}
    }
    
    # First pass: collect all model names and parameters
    for model_name, results in results_dict.items():
        comparison['model_names'].append(model_name)
        
        # Track all parameters across all models
        importance_scores = results.get('importance_scores', {})
        if not comparison['common_params']:
            comparison['common_params'] = set(importance_scores.keys())
        else:
            comparison['common_params'] &= set(importance_scores.keys())
    
    # Initialize parameter importance tracking
    for param in comparison['common_params']:
        comparison['param_importance'][param] = []
    
    # Second pass: collect importance scores for common parameters
    for model_name, results in results_dict.items():
        importance_scores = results.get('importance_scores', {})
        
        for param in comparison['common_params']:
            comparison['param_importance'][param].append({
                'model': model_name,
                'importance': importance_scores.get(param, 0)
            })
    
    return comparison