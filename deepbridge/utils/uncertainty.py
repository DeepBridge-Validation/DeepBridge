"""
Utility functions for uncertainty quantification.
"""

import numpy as np
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
    This function has been deprecated as visualization functionality has been removed.
    
    Raises:
        NotImplementedError: Always raises this exception
    """
    raise NotImplementedError("Visualization functionality has been removed from this version.")

def compare_models_uncertainty(results_dict, plot_type='coverage'):
    """
    This function has been deprecated as visualization functionality has been removed.
    
    Raises:
        NotImplementedError: Always raises this exception
    """
    raise NotImplementedError("Visualization functionality has been removed from this version.")

def uncertainty_report_to_html(results, include_plots=True):
    """
    This function has been deprecated as reporting functionality has been removed.
    
    Raises:
        NotImplementedError: Always raises this exception
    """
    raise NotImplementedError("Report generation functionality has been removed from this version.")