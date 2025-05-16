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
    # Try to import the enhanced version first
    try:
        from deepbridge.validation.wrappers.enhanced_uncertainty_suite import run_enhanced_uncertainty_tests
        
        # Run the enhanced version with more detailed metrics and visualizations
        if verbose:
            print("Using enhanced uncertainty analysis with additional metrics and visualizations")
        results = run_enhanced_uncertainty_tests(dataset, config_name, verbose, feature_subset)
        
    except ImportError:
        # Fallback to standard version if enhanced is not available
        from deepbridge.validation.wrappers.uncertainty_suite import UncertaintySuite
        
        if verbose:
            print("Using standard uncertainty analysis")
            
        # Initialize uncertainty suite
        uncertainty = UncertaintySuite(dataset, verbose=verbose, feature_subset=feature_subset)
        
        # Configure and run tests with feature subset if specified
        results = uncertainty.config(config_name, feature_subset=feature_subset).run()
    
    if verbose:
        print(f"\nUncertainty Test Summary:")
        print(f"Overall uncertainty quality score: {results.get('uncertainty_quality_score', 0):.3f}")
        print(f"Average coverage error: {results.get('avg_coverage_error', 0):.3f}")
        print(f"Average normalized width: {results.get('avg_normalized_width', 0):.3f}")
        
        # Print additional metrics if available
        if 'mean_width' in results:
            print(f"Mean interval width: {results.get('mean_width', 0):.3f}")
        if 'threshold_value' in results:
            print(f"Reliability threshold: {results.get('threshold_value', 0):.3f}")
        if 'reliable_count' in results.get('reliability_analysis', {}):
            reliable = results['reliability_analysis']['reliable_count']
            unreliable = results['reliability_analysis']['unreliable_count']
            total = reliable + unreliable
            print(f"Reliable predictions: {reliable}/{total} ({100*reliable/total:.1f}%)")
    
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