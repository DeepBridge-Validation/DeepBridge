"""
Utility functions for hyperparameter importance testing.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Any

def run_hyperparameter_tests(dataset, config_name='quick', metric='accuracy', verbose=True, feature_subset=None):
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
    feature_subset : List[str] or None
        Specific features to focus on for testing (None for all features)
        
    Returns:
    --------
    dict : Test results with detailed hyperparameter importance metrics
    """
    from deepbridge.validation.wrappers.hyperparameter_suite import HyperparameterSuite
    
    # Initialize hyperparameter suite
    hyperparameter = HyperparameterSuite(dataset, verbose=verbose, metric=metric, feature_subset=feature_subset)
    
    # Configure and run tests with feature subset if specified
    results = hyperparameter.config(config_name, feature_subset=feature_subset).run()
    
    if verbose:
        print(f"\nHyperparameter Importance Summary:")
        print(f"Suggested tuning order:")
        for i, param in enumerate(results.get('tuning_order', []), 1):
            importance = results.get('sorted_importance', {}).get(param, 0)
            print(f"  {i}. {param} (importance: {importance:.4f})")
    
    return results

def hyperparameter_report_to_html(results, include_details=True):
    """
    This function has been deprecated as reporting functionality has been removed.
    
    Raises:
        NotImplementedError: Always raises this exception
    """
    raise NotImplementedError("Report generation functionality has been removed from this version.")

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