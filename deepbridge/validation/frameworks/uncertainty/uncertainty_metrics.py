"""
Metrics for uncertainty estimation and calibration.

This module provides metrics for evaluating uncertainty estimates
and probability calibration.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Any, Callable, Tuple


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    weighted: bool = True
) -> float:
    """
    Compute Expected Calibration Error.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_prob : array-like
        Predicted probabilities
    n_bins : int
        Number of bins for discretization
    weighted : bool
        Whether to weight by bin size
        
    Returns:
    --------
    float : Expected Calibration Error
    """
    # Discretize predictions into bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])
    
    # Calculate statistics for each bin
    bin_sums = np.bincount(bin_indices, weights=y_prob, minlength=n_bins)
    bin_counts = np.bincount(bin_indices, minlength=n_bins)
    bin_true = np.bincount(bin_indices, weights=y_true, minlength=n_bins)
    
    # Calculate mean predicted probability and true fraction in each bin
    mean_pred_probs = bin_sums / (bin_counts + 1e-8)
    true_fractions = bin_true / (bin_counts + 1e-8)
    
    # Calculate absolute difference between predictions and true fractions
    abs_diff = np.abs(mean_pred_probs - true_fractions)
    
    # Calculate ECE
    if weighted:
        # Weighted by bin size
        weights = bin_counts / bin_counts.sum()
        ece = np.sum(weights * abs_diff)
    else:
        # Simple average
        ece = np.mean(abs_diff[bin_counts > 0])
        
    return ece


def maximum_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute Maximum Calibration Error.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_prob : array-like
        Predicted probabilities
    n_bins : int
        Number of bins for discretization
        
    Returns:
    --------
    float : Maximum Calibration Error
    """
    # Discretize predictions into bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])
    
    # Calculate statistics for each bin
    bin_sums = np.bincount(bin_indices, weights=y_prob, minlength=n_bins)
    bin_counts = np.bincount(bin_indices, minlength=n_bins)
    bin_true = np.bincount(bin_indices, weights=y_true, minlength=n_bins)
    
    # Calculate mean predicted probability and true fraction in each bin
    mean_pred_probs = bin_sums / (bin_counts + 1e-8)
    true_fractions = bin_true / (bin_counts + 1e-8)
    
    # Calculate absolute difference between predictions and true fractions
    abs_diff = np.abs(mean_pred_probs - true_fractions)
    
    # Only consider bins with samples
    abs_diff = abs_diff[bin_counts > 0]
    
    # Return maximum difference
    return np.max(abs_diff) if len(abs_diff) > 0 else 0.0


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute Brier Score.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_prob : array-like
        Predicted probabilities
        
    Returns:
    --------
    float : Brier Score
    """
    return np.mean((y_prob - y_true) ** 2)


def negative_log_likelihood(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute Negative Log Likelihood.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_prob : array-like
        Predicted probabilities
        
    Returns:
    --------
    float : Negative Log Likelihood
    """
    # Clip probabilities to avoid log(0)
    eps = 1e-15
    y_prob = np.clip(y_prob, eps, 1 - eps)
    
    # Calculate negative log likelihood
    nll = -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
    return nll


def uncertainty_decomposition(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Dict[str, float]:
    """
    Decompose Brier score into reliability, resolution, and uncertainty.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_prob : array-like
        Predicted probabilities
    n_bins : int
        Number of bins for discretization
        
    Returns:
    --------
    dict : Components of the decomposition
    """
    # Calculate base Brier score
    bs = brier_score(y_true, y_prob)
    
    # Calculate base uncertainty
    p_mean = np.mean(y_true)
    uncertainty = p_mean * (1 - p_mean)
    
    # Discretize predictions into bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])
    
    # Calculate statistics for each bin
    bin_sums = np.bincount(bin_indices, weights=y_prob, minlength=n_bins)
    bin_counts = np.bincount(bin_indices, minlength=n_bins)
    bin_true = np.bincount(bin_indices, weights=y_true, minlength=n_bins)
    
    # Calculate mean predicted probability and true fraction in each bin
    mean_pred_probs = bin_sums / (bin_counts + 1e-8)
    true_fractions = bin_true / (bin_counts + 1e-8)
    
    # Calculate reliability (calibration)
    reliability = np.sum(bin_counts * (mean_pred_probs - true_fractions) ** 2) / len(y_true)
    
    # Calculate resolution (refinement)
    resolution = np.sum(bin_counts * (true_fractions - p_mean) ** 2) / len(y_true)
    
    return {
        'brier_score': bs,
        'uncertainty': uncertainty,
        'reliability': reliability,
        'resolution': resolution
    }


def compute_uncertainty_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Dict[str, float]:
    """
    Compute various uncertainty metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_prob : array-like
        Predicted probabilities
    n_bins : int
        Number of bins for discretization
        
    Returns:
    --------
    dict : Dictionary of metrics
    """
    metrics = {
        'expected_calibration_error': expected_calibration_error(y_true, y_prob, n_bins),
        'maximum_calibration_error': maximum_calibration_error(y_true, y_prob, n_bins),
        'brier_score': brier_score(y_true, y_prob),
        'negative_log_likelihood': negative_log_likelihood(y_true, y_prob)
    }
    
    # Add uncertainty decomposition
    decomposition = uncertainty_decomposition(y_true, y_prob, n_bins)
    metrics.update({
        'uncertainty': decomposition['uncertainty'],
        'reliability': decomposition['reliability'],
        'resolution': decomposition['resolution']
    })
    
    return metrics


def prediction_intervals_coverage(
    y_true: np.ndarray,
    y_pred_low: np.ndarray,
    y_pred_high: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate prediction intervals coverage.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred_low : array-like
        Lower bounds of prediction intervals
    y_pred_high : array-like
        Upper bounds of prediction intervals
        
    Returns:
    --------
    dict : Coverage metrics
    """
    # Calculate empirical coverage
    in_bounds = (y_true >= y_pred_low) & (y_true <= y_pred_high)
    coverage = np.mean(in_bounds)
    
    # Calculate interval widths
    interval_widths = y_pred_high - y_pred_low
    mean_width = np.mean(interval_widths)
    
    # Calculate metrics
    return {
        'coverage': coverage,
        'mean_interval_width': mean_width,
        'normalized_interval_width': mean_width / np.std(y_true) if np.std(y_true) > 0 else np.nan
    }