
"""
Utility functions for robustness testing.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Any

def run_robustness_tests(dataset, config_name='full', metric='AUC', verbose=True, feature_subset=None, n_iterations=None, model_name=None):
    """
    Run enhanced robustness tests on a dataset with Gaussian noise and Quantile perturbation.
    
    Parameters:
    -----------
    dataset : DBDataset
        Dataset object containing training/test data and model
    config_name : str
        Name of the configuration to use: 'quick', 'medium', or 'full'
    metric : str
        Performance metric to use for evaluation ('AUC', 'accuracy', 'f1', etc.)
    verbose : bool
        Whether to print progress information
    feature_subset : List[str] or None
        Specific features to focus on for testing (None for all features)
        When specified, all variables are maintained, but only the specified ones are perturbed
    n_iterations : int or None
        Number of iterations to perform for each perturbation level
        If None, uses defaults based on config_name (3 for 'quick', 6 for 'medium', 10 for 'full')
    model_name : str or None
        Optional name of the model being tested, used for verbose output
        
    Returns:
    --------
    Dict[str, Any] : Test results dictionary containing:
        - base_score: The baseline score of the model
        - metrics: Dictionary of evaluation metrics
        - feature_subset: The features that were perturbed (None for all)
        - feature_importance: Feature importance scores
        - raw: Results for raw (Gaussian) perturbation
        - quantile: Results for quantile-based perturbation
        - avg_raw_impact: Average impact from raw perturbation
        - avg_quantile_impact: Average impact from quantile perturbation
        - avg_overall_impact: Average impact across all perturbation types
        - n_iterations: Number of iterations performed per perturbation level
    """
    from deepbridge.validation.wrappers.robustness_suite import RobustnessSuite
    
    # Set default iterations based on config if not specified
    if n_iterations is None:
        if config_name == 'quick':
            n_iterations = 3
        elif config_name == 'medium':
            n_iterations = 6
        elif config_name == 'full':
            n_iterations = 10
        else:
            n_iterations = 3
    
    # Initialize robustness suite with n_iterations parameter
    robustness = RobustnessSuite(
        dataset, 
        verbose=verbose, 
        metric=metric, 
        feature_subset=feature_subset,
        n_iterations=n_iterations
    )
    
    # Execute two separate tests - one with all features and one with the subset
    
    # First, run test with all features (feature_subset=None)
    robustness_all = RobustnessSuite(
        dataset, 
        verbose=verbose, 
        metric=metric, 
        feature_subset=None,  # Explicitly set to None for all features
        n_iterations=n_iterations
    )
    results = robustness_all.config(config_name).run()
    
    # If feature_subset is specified, run a second test with just those features
    if feature_subset:
        robustness_subset = RobustnessSuite(
            dataset, 
            verbose=verbose, 
            metric=metric, 
            feature_subset=feature_subset,
            n_iterations=n_iterations
        )
        subset_results = robustness_subset.config(config_name).run()
        
        # For each perturbation level, copy the feature_subset results to the main results
        for perturb_type in ['raw', 'quantile']:
            if perturb_type in results and perturb_type in subset_results:
                for level_key in results[perturb_type]['by_level']:
                    if level_key in subset_results[perturb_type]['by_level']:
                        # Create feature_subset key if it doesn't exist
                        if 'feature_subset' not in results[perturb_type]['by_level'][level_key]['runs']:
                            results[perturb_type]['by_level'][level_key]['runs']['feature_subset'] = []
                        
                        # Copy the runs from subset_results to the main results
                        results[perturb_type]['by_level'][level_key]['runs']['feature_subset'] = subset_results[perturb_type]['by_level'][level_key]['runs']['feature_subset']
                        
                        # Copy the overall result
                        if 'feature_subset' not in results[perturb_type]['by_level'][level_key]['overall_result']:
                            results[perturb_type]['by_level'][level_key]['overall_result']['feature_subset'] = {}
                        
                        results[perturb_type]['by_level'][level_key]['overall_result']['feature_subset'] = subset_results[perturb_type]['by_level'][level_key]['overall_result']['feature_subset']
        
        # Store the feature subset in the results
        results['feature_subset'] = feature_subset
    
    # Add the iterations count to the results
    results['n_iterations'] = n_iterations
    
    # Initialize metrics dictionary with just base values
    # Complete metrics will be set from initial_results in the calling code
    metrics = {
        'base_score': results.get('base_score', 0)
    }
    
    # Use base_score as the roc_auc if we don't have anything better
    # This will be replaced by real metrics from initial_results
    if 'base_score' in results:
        metrics['roc_auc'] = results['base_score']
    
    # Add metrics to results
    results['metrics'] = metrics
    
    if verbose:
        model_label = f"[{model_name}] " if model_name else ""
        print(f"\n{model_label}Robustness Test Summary:")
        print(f"{model_label}Overall robustness score: {1.0 - results.get('avg_overall_impact', 0):.3f}")
        print(f"{model_label}Average impact from Gaussian noise: {results.get('avg_raw_impact', 0):.3f}")
        print(f"{model_label}Average impact from Quantile perturbation: {results.get('avg_quantile_impact', 0):.3f}")
        if feature_subset:
            print(f"{model_label}Features perturbed: {', '.join(feature_subset)}")
        
        # Print metrics if available
        if 'metrics' in results and results['metrics']:
            print(f"\n{model_label}Model Metrics:")
            for metric_name, metric_value in results['metrics'].items():
                print(f"  {metric_name}: {metric_value:.3f}")
                
    # Store model_name in results if provided
    if model_name:
        results['model_name'] = model_name
    
    return results

def plot_robustness_results(results, plot_type='robustness', **kwargs):
    """
    This function has been deprecated as visualization functionality has been removed.
    
    Raises:
        NotImplementedError: Always raises this exception
    """
    raise NotImplementedError("Visualization functionality has been removed from this version.")

def compare_models_robustness(results_dict, use_worst=False):
    """
    This function has been deprecated as visualization functionality has been removed.
    
    Raises:
        NotImplementedError: Always raises this exception
    """
    raise NotImplementedError("Visualization functionality has been removed from this version.")

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
    This function has been deprecated as reporting functionality has been removed.
    
    Raises:
        NotImplementedError: Always raises this exception
    """
    raise NotImplementedError("Report generation functionality has been removed from this version.")
