"""
Common utilities for validation tasks.

This module provides utility functions for common validation tasks
such as logging, results formatting, and saving/loading results.
"""

import os
import json
import pickle
import logging
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
import datetime
import numpy as np
import pandas as pd


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_str: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger for validation tasks.
    
    Parameters:
    -----------
    name : str
        Name of the logger
    log_file : str or None
        Path to log file (None for console only)
    level : int
        Logging level
    format_str : str or None
        Custom logging format string
        
    Returns:
    --------
    logging.Logger : Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers if any
    if logger.handlers:
        logger.handlers = []
    
    # Set default format if not provided
    if format_str is None:
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
    formatter = logging.Formatter(format_str)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if log file is provided
    if log_file is not None:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger


def format_validation_results(
    results: Dict[str, Any],
    format_type: str = 'json'
) -> Union[str, pd.DataFrame]:
    """
    Format validation results into the specified format.
    
    Parameters:
    -----------
    results : dict
        Validation results
    format_type : str
        Format type: 'json', 'dataframe', 'text'
        
    Returns:
    --------
    str or DataFrame : Formatted results
    """
    if format_type == 'json':
        # Convert results to JSON-friendly format
        clean_results = _make_json_serializable(results)
        return json.dumps(clean_results, indent=2)
        
    elif format_type == 'dataframe':
        # Convert to DataFrame (flatten the dictionary)
        flat_results = _flatten_dict(results)
        return pd.DataFrame.from_dict(flat_results, orient='index').T
        
    elif format_type == 'text':
        # Format as text (human-readable)
        return _dict_to_text(results)
        
    else:
        raise ValueError(f"Unknown format type: {format_type}")


def _make_json_serializable(obj: Any) -> Any:
    """
    Make an object JSON-serializable.
    
    Parameters:
    -----------
    obj : any
        Object to convert
        
    Returns:
    --------
    any : JSON-serializable object
    """
    if isinstance(obj, (np.ndarray, pd.Series)):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    elif hasattr(obj, '__dict__'):
        # Attempt to convert objects with __dict__ attribute
        return _make_json_serializable(obj.__dict__)
    else:
        # Try to convert to string if not serializable
        try:
            json.dumps(obj)
            return obj
        except (TypeError, OverflowError):
            return str(obj)


def _flatten_dict(d: Dict[str, Any], parent_key: str = '') -> Dict[str, Any]:
    """
    Flatten a nested dictionary.
    
    Parameters:
    -----------
    d : dict
        Dictionary to flatten
    parent_key : str
        Key of parent dictionary
        
    Returns:
    --------
    dict : Flattened dictionary
    """
    flat_dict = {}
    for k, v in d.items():
        key = f"{parent_key}.{k}" if parent_key else k
        
        if isinstance(v, dict):
            # Recursively flatten dictionaries
            flat_dict.update(_flatten_dict(v, key))
        elif isinstance(v, (list, tuple)) and all(isinstance(item, dict) for item in v):
            # If the value is a list of dictionaries, flatten each dictionary
            for i, item in enumerate(v):
                flat_dict.update(_flatten_dict(item, f"{key}[{i}]"))
        else:
            # Convert non-serializable objects to string
            if isinstance(v, (np.ndarray, pd.Series)):
                v = v.tolist()
            elif hasattr(v, '__dict__'):
                v = str(v)
                
            flat_dict[key] = v
            
    return flat_dict


def _dict_to_text(d: Dict[str, Any], indent: int = 0) -> str:
    """
    Convert a dictionary to human-readable text.
    
    Parameters:
    -----------
    d : dict
        Dictionary to convert
    indent : int
        Indentation level
        
    Returns:
    --------
    str : Text representation
    """
    text = ""
    indent_str = "  " * indent
    
    for k, v in d.items():
        if isinstance(v, dict):
            text += f"{indent_str}{k}:\n"
            text += _dict_to_text(v, indent + 1)
        elif isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], dict):
            text += f"{indent_str}{k}:\n"
            for i, item in enumerate(v):
                text += f"{indent_str}  [{i}]:\n"
                text += _dict_to_text(item, indent + 2)
        elif isinstance(v, (list, tuple)):
            formatted_list = ', '.join(str(item) for item in v)
            text += f"{indent_str}{k}: [{formatted_list}]\n"
        else:
            text += f"{indent_str}{k}: {v}\n"
            
    return text


def save_validation_results(
    results: Dict[str, Any],
    output_file: str,
    format_type: Optional[str] = None
) -> None:
    """
    Save validation results to a file.
    
    Parameters:
    -----------
    results : dict
        Validation results
    output_file : str
        Path to output file
    format_type : str or None
        Format type (None to infer from file extension)
    """
    # Create directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Infer format type from file extension if not provided
    if format_type is None:
        _, ext = os.path.splitext(output_file)
        ext = ext.lstrip('.').lower()
        
        if ext in ['json']:
            format_type = 'json'
        elif ext in ['csv']:
            format_type = 'dataframe'
        elif ext in ['txt', 'text']:
            format_type = 'text'
        elif ext in ['pickle', 'pkl']:
            format_type = 'pickle'
        else:
            # Default to JSON
            format_type = 'json'
    
    # Save based on format type
    if format_type == 'json':
        # Clean and convert to JSON
        clean_results = _make_json_serializable(results)
        with open(output_file, 'w') as f:
            json.dump(clean_results, f, indent=2)
            
    elif format_type == 'dataframe':
        # Convert to DataFrame and save as CSV
        flat_results = _flatten_dict(results)
        df = pd.DataFrame.from_dict(flat_results, orient='index').T
        df.to_csv(output_file, index=False)
        
    elif format_type == 'text':
        # Format as text and save
        text = _dict_to_text(results)
        with open(output_file, 'w') as f:
            f.write(text)
            
    elif format_type == 'pickle':
        # Save as pickle
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)
            
    else:
        raise ValueError(f"Unknown format type: {format_type}")


def load_validation_results(
    input_file: str,
    format_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load validation results from a file.
    
    Parameters:
    -----------
    input_file : str
        Path to input file
    format_type : str or None
        Format type (None to infer from file extension)
        
    Returns:
    --------
    dict : Validation results
    """
    # Check if file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File not found: {input_file}")
    
    # Infer format type from file extension if not provided
    if format_type is None:
        _, ext = os.path.splitext(input_file)
        ext = ext.lstrip('.').lower()
        
        if ext in ['json']:
            format_type = 'json'
        elif ext in ['csv']:
            format_type = 'dataframe'
        elif ext in ['pickle', 'pkl']:
            format_type = 'pickle'
        else:
            # Default to JSON
            format_type = 'json'
    
    # Load based on format type
    if format_type == 'json':
        # Load JSON
        with open(input_file, 'r') as f:
            return json.load(f)
            
    elif format_type == 'dataframe':
        # Load CSV into DataFrame and convert to dict
        df = pd.read_csv(input_file)
        return df.to_dict(orient='records')[0]
        
    elif format_type == 'pickle':
        # Load pickle
        with open(input_file, 'rb') as f:
            return pickle.load(f)
            
    else:
        raise ValueError(f"Unknown format type: {format_type}")


def create_validation_summary(
    results: Dict[str, Any],
    include_keys: Optional[List[str]] = None,
    exclude_keys: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Create a summary of validation results.
    
    Parameters:
    -----------
    results : dict
        Validation results
    include_keys : list of str or None
        Keys to include in summary (None for all)
    exclude_keys : list of str or None
        Keys to exclude from summary
        
    Returns:
    --------
    dict : Summary of validation results
    """
    # Start with a copy of results
    summary = results.copy()
    
    # Filter by included keys if specified
    if include_keys is not None:
        summary = {k: v for k, v in summary.items() if k in include_keys}
        
    # Filter out excluded keys if specified
    if exclude_keys is not None:
        summary = {k: v for k, v in summary.items() if k not in exclude_keys}
        
    # Extract key metrics from different validation types
    key_metrics = {}
    
    # Extract robustness metrics if available
    if 'robustness' in summary:
        rob = summary['robustness']
        if isinstance(rob, dict):
            if 'robustness_score' in rob:
                key_metrics['robustness_score'] = rob['robustness_score']
            elif 'aggregate' in rob and 'robustness_score' in rob['aggregate']:
                key_metrics['robustness_score'] = rob['aggregate']['robustness_score']
    
    # Extract uncertainty metrics if available
    if 'uncertainty' in summary:
        unc = summary['uncertainty']
        if isinstance(unc, dict):
            if 'metrics' in unc:
                metrics = unc['metrics']
                for key in ['expected_calibration_error', 'brier_score', 'log_loss']:
                    if key in metrics:
                        key_metrics[f'uncertainty_{key}'] = metrics[key]
    
    # Extract hyperparameter metrics if available
    if 'hyperparameters' in summary:
        hyp = summary['hyperparameters']
        if isinstance(hyp, dict):
            if 'best_score' in hyp:
                key_metrics['hyperparameter_best_score'] = hyp['best_score']
            elif 'tuning' in hyp and 'best_score' in hyp['tuning']:
                key_metrics['hyperparameter_best_score'] = hyp['tuning']['best_score']
    
    # Add key metrics to summary
    summary['key_metrics'] = key_metrics
    
    # Add timestamp
    summary['timestamp'] = datetime.datetime.now().isoformat()
    
    return summary


def get_default_metrics(problem_type: str) -> Dict[str, Callable]:
    """
    Get default evaluation metrics for a problem type.
    
    Parameters:
    -----------
    problem_type : str
        Type of problem: 'classification' or 'regression'
        
    Returns:
    --------
    dict : Default metrics
    """
    try:
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
            mean_squared_error, mean_absolute_error, r2_score
        )
    except ImportError:
        raise ImportError("scikit-learn is required for default metrics")
        
    if problem_type.lower() == 'classification':
        return {
            'accuracy': accuracy_score,
            'f1': lambda y_true, y_pred: f1_score(
                y_true, y_pred, average='weighted', zero_division=0
            ),
            'precision': lambda y_true, y_pred: precision_score(
                y_true, y_pred, average='weighted', zero_division=0
            ),
            'recall': lambda y_true, y_pred: recall_score(
                y_true, y_pred, average='weighted', zero_division=0
            ),
            'roc_auc': lambda y_true, y_score: roc_auc_score(
                y_true, y_score, multi_class='ovr', average='weighted'
            ) if hasattr(y_score, 'shape') and y_score.ndim > 1 else roc_auc_score(y_true, y_score)
        }
    elif problem_type.lower() == 'regression':
        return {
            'mse': mean_squared_error,
            'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error,
            'r2': r2_score
        }
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")