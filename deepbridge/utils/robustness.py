
"""
Utility functions for robustness testing.
"""

import numpy as np
import io
import base64
from typing import Dict, List, Optional, Union, Any

def run_robustness_tests(dataset, config_name='full', metric='AUC', verbose=True):
    """
    Run enhanced robustness tests on a dataset with Gaussian noise and Quantile perturbation.
    
    Parameters:
    -----------
    dataset : DBDataset
        Dataset object containing training/test data and model
    config_name : str
        Name of the configuration to use: 'quick', 'medium', or 'full'
    metric : str
        Performance metric to use for evaluation
    verbose : bool
        Whether to print progress information
        
    Returns:
    --------
    dict : Test results with detailed robustness metrics
    """
    from deepbridge.validation.wrappers.robustness_suite import RobustnessSuite
    
    # Initialize robustness suite
    robustness = RobustnessSuite(dataset, verbose=verbose, metric=metric)
    
    # Configure and run tests
    results = robustness.config(config_name).run()
    
    if verbose:
        print(f"\nRobustness Test Summary:")
        print(f"Overall robustness score: {results.get('robustness_score', 0):.3f}")
        print(f"Average impact from Gaussian noise: {results.get('avg_raw_impact', 0):.3f}")
        print(f"Average impact from Quantile perturbation: {results.get('avg_quantile_impact', 0):.3f}")
    
    return results

def plot_robustness_results(results, plot_type='robustness', **kwargs):
    """
    Generate robustness visualizations.
    
    Parameters:
    -----------
    results : dict
        Robustness test results from run_robustness_tests
    plot_type : str
        Type of plot to generate:
        - 'robustness': Robustness by perturbation level
        - 'distribution': Distribution of robustness scores
        - 'feature_importance': Feature importance based on robustness
        - 'methods_comparison': Comparison of perturbation methods
    **kwargs : dict
        Additional arguments for specific plot types
        
    Returns:
    --------
    plotly.graph_objects.Figure : Plotly figure object
    """
    from deepbridge.validation.wrappers.robustness_suite import RobustnessSuite
    import plotly.io as pio
    
    # Create a temporary RobustnessSuite instance for plotting
    # (we don't need a real dataset since we're just using the plotting methods)
    suite = RobustnessSuite(None, verbose=False)
    
    # Generate appropriate plot based on type
    if plot_type == 'robustness':
        return suite.plot_robustness(results, **kwargs)
    elif plot_type == 'distribution':
        return suite.plot_distribution(results, **kwargs)
    elif plot_type == 'feature_importance':
        return suite.plot_feature_importance(results, **kwargs)
    elif plot_type == 'methods_comparison':
        return suite.compare_methods(results, **kwargs)
    else:
        raise ValueError(f"Unknown plot type: {plot_type}")

def compare_models_robustness(results_dict, use_worst=False):
    """
    Compare robustness of multiple models.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary of model name: results pairs from run_robustness_tests
    use_worst : bool
        Whether to use worst-case performance
        
    Returns:
    --------
    plotly.graph_objects.Figure : Comparison plot
    """
    from deepbridge.validation.wrappers.robustness_suite import RobustnessSuite
    
    # Create a temporary RobustnessSuite instance for plotting
    suite = RobustnessSuite(None, verbose=False)
    
    # Generate comparison plot
    return suite.plot_models_comparison(results_dict, use_worst=use_worst)

def is_metric_higher_better(metric_name: str) -> bool:
    """
    Determine if a higher value for a metric is better.
    
    Parameters:
    -----------
    metric_name : str
        Name of the metric to check
        
    Returns:
    --------
    bool : True if higher is better, False otherwise
    """
    # Standard metrics where higher is better
    higher_better_metrics = {
        'accuracy', 'precision', 'recall', 'f1', 'auc', 'roc_auc', 'r2',
        'balanced_accuracy', 'average_precision', 'explained_variance',
        'accuracy_score', 'recall_score', 'precision_score', 'f1_score'
    }
    
    # Standard metrics where lower is better
    lower_better_metrics = {
        'error', 'mae', 'mse', 'rmse', 'log_loss', 'cross_entropy',
        'mean_squared_error', 'mean_absolute_error', 'mean_squared_log_error',
        'median_absolute_error', 'max_error', 'hinge_loss'
    }
    
    # Normalize metric name to lowercase for comparison
    metric_lower = metric_name.lower()
    
    # Check in higher-better set
    if any(m in metric_lower for m in higher_better_metrics):
        return True
    
    # Check in lower-better set
    if any(m in metric_lower for m in lower_better_metrics):
        return False
    
    # Default to higher-better for unknown metrics
    return True

def robustness_report_to_html(results, include_plots=True):
    """
    Generate HTML report from robustness results.
    
    Parameters:
    -----------
    results : dict
        Robustness test results from run_robustness_tests
    include_plots : bool
        Whether to include interactive plots in the report
        
    Returns:
    --------
    str : HTML report content
    """
    import plotly.io as pio
    
    # Basic report structure
    html = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<title>Robustness Test Report</title>",
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
        "<h1>Robustness Test Report</h1>"
    ]
    
    # Summary section
    html.append("<div class='summary'>")
    html.append("<h2>Summary</h2>")
    html.append(f"<p><strong>Overall Robustness Score:</strong> {results.get('robustness_score', 0):.3f}</p>")
    html.append(f"<p><strong>Model Type:</strong> {results.get('model_type', 'Unknown')}</p>")
    html.append(f"<p><strong>Baseline Performance:</strong> {results.get('baseline_performance', {}).get('auc', 0):.3f}</p>")
    html.append(f"<p><strong>Raw Perturbation Impact:</strong> {results.get('avg_raw_impact', 0):.3f}</p>")
    html.append(f"<p><strong>Quantile Perturbation Impact:</strong> {results.get('avg_quantile_impact', 0):.3f}</p>")
    html.append("</div>")
    
    # Include plots if requested
    if include_plots and 'plot_data' in results:
        try:
            from deepbridge.validation.wrappers.robustness_suite import RobustnessSuite
            suite = RobustnessSuite(None, verbose=False)
            
            # Robustness plot
            html.append("<div class='plot-container'>")
            html.append("<h2>Robustness by Perturbation Level</h2>")
            fig = suite.plot_robustness(results)
            html.append(pio.to_html(fig, full_html=False, include_plotlyjs='cdn'))
            html.append("</div>")
            
            # Method comparison plot
            html.append("<div class='plot-container'>")
            html.append("<h2>Comparison of Perturbation Methods</h2>")
            fig = suite.compare_methods(results)
            html.append(pio.to_html(fig, full_html=False, include_plotlyjs='cdn'))
            html.append("</div>")
            
            # Feature importance plot
            if 'feature_importance' in results and results['feature_importance']:
                html.append("<div class='plot-container'>")
                html.append("<h2>Feature Importance</h2>")
                fig = suite.plot_feature_importance(results)
                html.append(pio.to_html(fig, full_html=False, include_plotlyjs='cdn'))
                html.append("</div>")
                
            # Distribution plot
            html.append("<div class='plot-container'>")
            html.append("<h2>Distribution of Robustness Scores</h2>")
            fig = suite.plot_distribution(results)
            html.append(pio.to_html(fig, full_html=False, include_plotlyjs='cdn'))
            html.append("</div>")
            
        except Exception as e:
            html.append(f"<p>Error generating plots: {str(e)}</p>")
    
    # Raw perturbation results table
    html.append("<h2>Raw Perturbation Results</h2>")
    html.append("<table>")
    html.append("<tr><th>Level</th><th>Mean Score</th><th>Std Dev</th></tr>")
    
    for level, level_data in sorted(results.get('raw', {}).get('by_level', {}).items()):
        overall = level_data.get('overall_result', {})
        if overall:
            html.append("<tr>")
            html.append(f"<td>{level}</td>")
            html.append(f"<td>{overall.get('mean_score', 0):.3f}</td>")
            html.append(f"<td>{overall.get('std_score', 0):.3f}</td>")
            html.append("</tr>")
    
    html.append("</table>")
    
    # Quantile perturbation results table
    html.append("<h2>Quantile Perturbation Results</h2>")
    html.append("<table>")
    html.append("<tr><th>Level</th><th>Mean Score</th><th>Std Dev</th></tr>")
    
    for level, level_data in sorted(results.get('quantile', {}).get('by_level', {}).items()):
        overall = level_data.get('overall_result', {})
        if overall:
            html.append("<tr>")
            html.append(f"<td>{level}</td>")
            html.append(f"<td>{overall.get('mean_score', 0):.3f}</td>")
            html.append(f"<td>{overall.get('std_score', 0):.3f}</td>")
            html.append("</tr>")
    
    html.append("</table>")
    
    # Feature importance table
    html.append("<div class='feature-importance'>")
    html.append("<h2>Feature Importance</h2>")
    html.append("<table>")
    html.append("<tr><th>Feature</th><th>Importance</th></tr>")
    
    importance = results.get('feature_importance', {})
    for feature, value in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]:
        html.append("<tr>")
        html.append(f"<td>{feature}</td>")
        html.append(f"<td>{value:.3f}</td>")
        html.append("</tr>")
    
    html.append("</table>")
    html.append("</div>")
    
    # Close HTML
    html.append("</body>")
    html.append("</html>")
    
    return "\n".join(html)
