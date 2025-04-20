"""
Utility functions for resilience testing.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Any

def run_resilience_tests(dataset, config_name='quick', metric='auc', verbose=True, feature_subset=None):
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
    feature_subset : List[str] or None
        Specific features to focus on for testing (None for all features)
        
    Returns:
    --------
    dict : Test results with detailed resilience metrics
    """
    from deepbridge.validation.wrappers.resilience_suite import ResilienceSuite
    
    # Initialize resilience suite
    resilience = ResilienceSuite(dataset, verbose=verbose, metric=metric, feature_subset=feature_subset)
    
    # Configure and run tests with feature subset if specified
    results = resilience.config(config_name, feature_subset=feature_subset).run()
    
    if verbose:
        print(f"\nResilience Test Summary:")
        print(f"Overall resilience score: {results.get('resilience_score', 0):.3f}")
        
        # Print alpha-specific results
        for alpha, alpha_data in sorted(results.get('distribution_shift', {}).get('by_alpha', {}).items()):
            print(f"Alpha = {alpha}: Average performance gap: {alpha_data.get('avg_performance_gap', 0):.3f}")
    
    return results

def resilience_report_to_html(results, include_details=True):
    """
    This function has been deprecated as reporting functionality has been removed.
    
    Raises:
        NotImplementedError: Always raises this exception
    """
    raise NotImplementedError("Report generation functionality has been removed from this version.")

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