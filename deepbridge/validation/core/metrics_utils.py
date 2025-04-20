"""
Utility functions for working with model evaluation metrics.

This module provides functions for handling, interpreting, and transforming
various model performance metrics.
"""

import numpy as np
from typing import Optional, Union, List


def is_metric_higher_better(metric_name: str) -> bool:
    """
    Determine if a higher value of the metric indicates better performance.
    
    Parameters:
    -----------
    metric_name : str
        Name of the metric
        
    Returns:
    --------
    bool : True if higher is better, False if lower is better
    """
    # Metrics where lower values indicate better performance
    lower_is_better = [
        'mse', 'mae', 'rmse', 'error', 'loss', 'mean_squared_error', 
        'mean_absolute_error', 'root_mean_squared_error', 'log_loss',
        'hinge_loss', 'perplexity', 'negative_likelihood'
    ]
    
    # Check if the metric name contains any of the lower_is_better strings
    for lower_metric in lower_is_better:
        if lower_metric.lower() in metric_name.lower():
            return False
    
    # By default, assume higher is better for all other metrics
    # (accuracy, F1, AUC, precision, recall, R²)
    return True


def get_metric_direction_multiplier(metric_name: str) -> int:
    """
    Get multiplier to convert a metric to a consistent direction where higher is better.
    
    Parameters:
    -----------
    metric_name : str
        Name of the metric
        
    Returns:
    --------
    int : 1 if higher is better, -1 if lower is better
    """
    return 1 if is_metric_higher_better(metric_name) else -1


def normalize_metric_value(
    value: float, 
    metric_name: str, 
    min_val: Optional[float] = None, 
    max_val: Optional[float] = None
) -> float:
    """
    Normalize a metric value to a 0-1 scale where 1 is always better.
    
    Parameters:
    -----------
    value : float
        Metric value to normalize
    metric_name : str
        Name of the metric
    min_val : float, optional
        Minimum value for normalization
    max_val : float, optional
        Maximum value for normalization
        
    Returns:
    --------
    float : Normalized value
    """
    # If metric is one where lower is better, invert the value
    if not is_metric_higher_better(metric_name):
        if min_val is not None and max_val is not None:
            # Scale from [min_val, max_val] to [0, 1] and invert
            return 1 - ((value - min_val) / (max_val - min_val))
        else:
            # Just invert the sign to make lower values higher
            return -value
    else:
        if min_val is not None and max_val is not None:
            # Scale from [min_val, max_val] to [0, 1]
            return (value - min_val) / (max_val - min_val)
        else:
            # Return as is
            return value


def infer_problem_type(y: np.ndarray) -> str:
    """
    Infer whether the problem is classification or regression based on target values.
    
    Parameters:
    -----------
    y : np.ndarray
        Target values
        
    Returns:
    --------
    str : 'classification' or 'regression'
    """
    # Check if target is binary or categorical
    unique_values = np.unique(y)
    
    # If few unique values or all are integers, likely classification
    if len(unique_values) <= 10:
        # Check if all values are integers or booleans
        is_integer = all(isinstance(val, (int, bool)) or 
                        (isinstance(val, (float, np.float64)) and val.is_integer()) 
                        for val in unique_values)
        return 'classification' if is_integer else 'regression'
    
    return 'regression'


def calculate_relative_change(
    base_value: float, 
    new_value: float, 
    higher_is_better: bool = True
) -> float:
    """
    Calculate relative change between two values.
    
    Parameters:
    -----------
    base_value : float
        Base value
    new_value : float
        New value
    higher_is_better : bool
        Whether higher values represent better performance
        
    Returns:
    --------
    float : Relative change (positive means improvement, negative means degradation)
    """
    if base_value == 0:
        return float('inf') if new_value > 0 else float('-inf')
    
    relative_change = (new_value - base_value) / abs(base_value)
    
    # Adjust sign based on whether higher is better
    if not higher_is_better:
        relative_change = -relative_change
        
    return relative_change


def convert_metrics_to_higher_better(
    metrics_dict: dict, 
    metric_names: Optional[List[str]] = None
) -> dict:
    """
    Convert all metrics in a dictionary to a format where higher is better.
    
    Parameters:
    -----------
    metrics_dict : dict
        Dictionary of metric name to value
    metric_names : list or None
        List of metric names to convert (if None, convert all)
        
    Returns:
    --------
    dict : Dictionary with converted metrics
    """
    result = {}
    
    # Determine which metrics to convert
    if metric_names is None:
        metric_names = list(metrics_dict.keys())
    
    # Convert each metric
    for name in metric_names:
        if name in metrics_dict:
            value = metrics_dict[name]
            
            # Skip non-numeric values
            if not isinstance(value, (int, float)):
                result[name] = value
                continue
                
            # Convert to higher-is-better format
            if not is_metric_higher_better(name):
                # For metrics where lower is better, negate or invert
                if abs(value) < 1e-10:  # Close to zero
                    result[name] = 0.0  # Avoid division by zero
                else:
                    result[name] = -value
            else:
                result[name] = value
    
    return result