"""
Utility functions for uncertainty quantification.
"""

import numpy as np
import io
import base64
from typing import Dict, List, Optional, Union, Any

def run_uncertainty_tests(dataset, config_name='full', verbose=True, feature_subset=None):
    """
    Run uncertainty quantification tests on a dataset to estimate prediction intervals.
    
    Parameters:
    -----------
    dataset : DBDataset
        Dataset object containing training/test data and model
    config_name : str
        Name of the configuration to use: 'quick', 'medium', or 'full'
    verbose : bool
        Whether to print progress information
    feature_subset : List[str] or None
        Specific features to focus on for testing (None for all features)
        
    Returns:
    --------
    dict : Test results with detailed uncertainty metrics
    """
    from deepbridge.validation.wrappers.uncertainty_suite import UncertaintySuite
    
    # Initialize uncertainty suite
    uncertainty = UncertaintySuite(dataset, verbose=verbose, feature_subset=feature_subset)
    
    # Configure and run tests with feature subset if specified
    results = uncertainty.config(config_name, feature_subset=feature_subset).run()
    
    if verbose:
        print(f"\nUncertainty Test Summary:")
        print(f"Overall uncertainty quality score: {results.get('uncertainty_quality_score', 0):.3f}")
        print(f"Average coverage error: {results.get('avg_coverage_error', 0):.3f}")
        print(f"Average normalized width: {results.get('avg_normalized_width', 0):.3f}")
    
    return results

def plot_uncertainty_results(results, plot_type='alpha_comparison', **kwargs):
    """
    Generate uncertainty visualizations.
    
    Parameters:
    -----------
    results : dict
        Uncertainty test results from run_uncertainty_tests
    plot_type : str
        Type of plot to generate:
        - 'alpha_comparison': Compare different alpha levels
        - 'width_distribution': Distribution of interval widths
        - 'feature_importance': Feature importance for uncertainty
        - 'coverage_vs_width': Trade-off between coverage and width
    **kwargs : dict
        Additional arguments for specific plot types
        
    Returns:
    --------
    plotly.graph_objects.Figure : Plotly figure object
    """
    from deepbridge.validation.wrappers.uncertainty_suite import UncertaintySuite
    import plotly.io as pio
    
    # Create a temporary UncertaintySuite instance for plotting
    # (we don't need a real dataset since we're just using the plotting methods)
    suite = UncertaintySuite(None, verbose=False)
    
    # Generate appropriate plot based on type
    if plot_type == 'alpha_comparison':
        return suite.plot_alpha_comparison(results, **kwargs)
    elif plot_type == 'width_distribution':
        return suite.plot_width_distribution(results, **kwargs)
    elif plot_type == 'feature_importance':
        return suite.plot_feature_importance(results, **kwargs)
    elif plot_type == 'coverage_vs_width':
        return suite.plot_coverage_vs_width(results, **kwargs)
    else:
        raise ValueError(f"Unknown plot type: {plot_type}")

def compare_models_uncertainty(results_dict, plot_type='coverage'):
    """
    Compare uncertainty quantification of multiple models.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary of model name: results pairs from run_uncertainty_tests
    plot_type : str
        Type of comparison: 'coverage' or 'width'
        
    Returns:
    --------
    plotly.graph_objects.Figure : Comparison plot
    """
    from deepbridge.validation.wrappers.uncertainty_suite import UncertaintySuite
    
    # Create a temporary UncertaintySuite instance for plotting
    suite = UncertaintySuite(None, verbose=False)
    
    # Generate comparison plot
    return suite.plot_models_comparison(results_dict, plot_type=plot_type)

def uncertainty_report_to_html(results, include_plots=True):
    """
    Generate HTML report from uncertainty results.
    
    Parameters:
    -----------
    results : dict
        Uncertainty test results from run_uncertainty_tests
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
        "<title>Uncertainty Quantification Report</title>",
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
        "<h1>Uncertainty Quantification Report</h1>"
    ]
    
    # Summary section
    html.append("<div class='summary'>")
    html.append("<h2>Summary</h2>")
    html.append(f"<p><strong>Overall Uncertainty Quality Score:</strong> {results.get('uncertainty_quality_score', 0):.3f}</p>")
    html.append(f"<p><strong>Average Coverage Error:</strong> {results.get('avg_coverage_error', 0):.3f}</p>")
    html.append(f"<p><strong>Average Normalized Width:</strong> {results.get('avg_normalized_width', 0):.3f}</p>")
    html.append("</div>")
    
    # Include plots if requested
    if include_plots and 'plot_data' in results:
        try:
            from deepbridge.validation.wrappers.uncertainty_suite import UncertaintySuite
            suite = UncertaintySuite(None, verbose=False)
            
            # Alpha comparison plot
            html.append("<div class='plot-container'>")
            html.append("<h2>Alpha Level Comparison</h2>")
            fig = suite.plot_alpha_comparison(results)
            html.append(pio.to_html(fig, full_html=False, include_plotlyjs='cdn'))
            html.append("</div>")
            
            # Width distribution plot
            html.append("<div class='plot-container'>")
            html.append("<h2>Interval Width Distribution</h2>")
            fig = suite.plot_width_distribution(results)
            html.append(pio.to_html(fig, full_html=False, include_plotlyjs='cdn'))
            html.append("</div>")
            
            # Feature importance plot
            if 'feature_importance' in results and results['feature_importance']:
                html.append("<div class='plot-container'>")
                html.append("<h2>Feature Importance</h2>")
                fig = suite.plot_feature_importance(results)
                html.append(pio.to_html(fig, full_html=False, include_plotlyjs='cdn'))
                html.append("</div>")
                
            # Coverage vs width plot
            html.append("<div class='plot-container'>")
            html.append("<h2>Coverage vs Width Trade-off</h2>")
            fig = suite.plot_coverage_vs_width(results)
            html.append(pio.to_html(fig, full_html=False, include_plotlyjs='cdn'))
            html.append("</div>")
            
        except Exception as e:
            html.append(f"<p>Error generating plots: {str(e)}</p>")
    
    # CRQR results by alpha table
    html.append("<h2>Confidence Interval Results by Alpha</h2>")
    html.append("<table>")
    html.append("<tr><th>Alpha</th><th>Expected Coverage</th><th>Actual Coverage</th><th>Mean Width</th></tr>")
    
    for alpha, alpha_data in sorted(results.get('crqr', {}).get('by_alpha', {}).items()):
        overall = alpha_data.get('overall_result', {})
        if overall:
            html.append("<tr>")
            html.append(f"<td>{alpha}</td>")
            html.append(f"<td>{overall.get('expected_coverage', 0):.3f}</td>")
            html.append(f"<td>{overall.get('coverage', 0):.3f}</td>")
            html.append(f"<td>{overall.get('mean_width', 0):.3f}</td>")
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