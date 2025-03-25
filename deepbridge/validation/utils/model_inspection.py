"""
Model inspection utilities for the validation framework.

This module provides functions for extracting information and
parameters from machine learning models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
import warnings


def get_model_info(model: Any) -> Dict[str, Any]:
    """
    Get general information about a model.
    
    Parameters:
    -----------
    model : Any
        Machine learning model
        
    Returns:
    --------
    dict : Model information
    """
    info = {
        'model_type': type(model).__name__,
        'model_module': type(model).__module__
    }
    
    # Get model parameters
    if hasattr(model, 'get_params'):
        info['parameters'] = model.get_params()
        
    # Get feature importances if available
    importance = extract_feature_importance(model)
    if importance is not None:
        info['feature_importance'] = importance
        
    # Get coefficients if available
    coefficients = get_model_coefficients(model)
    if coefficients is not None:
        info['coefficients'] = coefficients
        
    # Get model type
    info['is_classifier'] = is_classifier(model)
    info['is_regressor'] = is_regressor(model)
    
    # Get classes if it's a classifier
    if info['is_classifier'] and hasattr(model, 'classes_'):
        info['classes'] = model.classes_.tolist() if hasattr(model.classes_, 'tolist') else model.classes_
        
    return info


def extract_model_parameters(model: Any) -> Dict[str, Any]:
    """
    Extract parameters from a model.
    
    Parameters:
    -----------
    model : Any
        Machine learning model
        
    Returns:
    --------
    dict : Model parameters
    """
    # Initialize parameters dictionary
    params = {}
    
    # Try getting parameters using get_params method
    if hasattr(model, 'get_params'):
        params = model.get_params()
        
    # Add additional model attributes that are often useful
    common_attributes = [
        'n_estimators', 'max_depth', 'learning_rate', 'C', 'gamma',
        'alpha', 'l1_ratio', 'kernel', 'degree', 'n_neighbors',
        'hidden_layer_sizes', 'activation', 'solver', 'max_iter'
    ]
    
    for attr in common_attributes:
        if hasattr(model, attr) and attr not in params:
            params[attr] = getattr(model, attr)
            
    return params


def extract_feature_importance(
    model: Any,
    feature_names: Optional[List[str]] = None
) -> Optional[Dict[str, float]]:
    """
    Extract feature importances from a model.
    
    Parameters:
    -----------
    model : Any
        Machine learning model
    feature_names : list of str or None
        Names of features for mapping importance values
        If None, use generic names
        
    Returns:
    --------
    dict or None : Feature importance mapping
    """
    importance = None
    
    # Check for various attributes that may contain feature importance
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # For models with coefficients, use absolute values as importance
        if model.coef_.ndim == 1:
            importance = np.abs(model.coef_)
        else:
            # For multi-class models, use mean absolute coefficient
            importance = np.mean(np.abs(model.coef_), axis=0)
    
    # Return None if no importance found
    if importance is None:
        return None
        
    # Create mapping with feature names
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(importance))]
        
    # Ensure we have the correct number of feature names
    if len(feature_names) != len(importance):
        warnings.warn(
            f"Number of feature names ({len(feature_names)}) does not match "
            f"number of importance values ({len(importance)}). "
            "Using generic feature names."
        )
        feature_names = [f"feature_{i}" for i in range(len(importance))]
        
    # Create dictionary mapping feature names to importance
    importance_dict = dict(zip(feature_names, importance))
    
    return importance_dict


def get_model_coefficients(
    model: Any,
    feature_names: Optional[List[str]] = None
) -> Optional[Dict[str, Any]]:
    """
    Get coefficients from a linear model.
    
    Parameters:
    -----------
    model : Any
        Machine learning model
    feature_names : list of str or None
        Names of features for mapping coefficients
        If None, use generic names
        
    Returns:
    --------
    dict or None : Coefficient mapping
    """
    if not hasattr(model, 'coef_'):
        return None
        
    coef = model.coef_
    
    # Generate feature names if not provided
    if feature_names is None:
        if coef.ndim == 1:
            feature_names = [f"feature_{i}" for i in range(len(coef))]
        else:
            feature_names = [f"feature_{i}" for i in range(coef.shape[1])]
            
    # Ensure we have the correct number of feature names
    if coef.ndim == 1 and len(feature_names) != len(coef):
        warnings.warn(
            f"Number of feature names ({len(feature_names)}) does not match "
            f"number of coefficients ({len(coef)}). "
            "Using generic feature names."
        )
        feature_names = [f"feature_{i}" for i in range(len(coef))]
    elif coef.ndim > 1 and len(feature_names) != coef.shape[1]:
        warnings.warn(
            f"Number of feature names ({len(feature_names)}) does not match "
            f"number of coefficients ({coef.shape[1]}). "
            "Using generic feature names."
        )
        feature_names = [f"feature_{i}" for i in range(coef.shape[1])]
        
    # Create coefficient dictionary based on model type
    if coef.ndim == 1:
        # For single-output models
        coefficients = dict(zip(feature_names, coef))
        
        # Add intercept if available
        if hasattr(model, 'intercept_'):
            coefficients['intercept'] = model.intercept_
            
        return coefficients
    else:
        # For multi-output/multi-class models
        if hasattr(model, 'classes_'):
            class_labels = model.classes_
        else:
            class_labels = [f"class_{i}" for i in range(coef.shape[0])]
            
        # Create nested dictionary for multi-class coefficients
        coefficients = {}
        for i, cls in enumerate(class_labels):
            cls_key = str(cls)
            coefficients[cls_key] = dict(zip(feature_names, coef[i]))
            
            # Add intercept if available
            if hasattr(model, 'intercept_'):
                coefficients[cls_key]['intercept'] = model.intercept_[i] if isinstance(model.intercept_, np.ndarray) else model.intercept_
                
        return coefficients


def is_classifier(model: Any) -> bool:
    """
    Check if a model is a classifier.
    
    Parameters:
    -----------
    model : Any
        Machine learning model
        
    Returns:
    --------
    bool : True if the model is a classifier
    """
    # Check for classification-specific attributes
    if hasattr(model, 'classes_'):
        return True
        
    # Check the model's class name
    model_name = type(model).__name__.lower()
    classifier_indicators = ['classifier', 'svm', 'logistic', 'naive', 'kneighbors', 'forest']
    
    for indicator in classifier_indicators:
        if indicator in model_name:
            return True
            
    # Check for predict_proba method (common in classifiers)
    if hasattr(model, 'predict_proba'):
        return True
        
    # Try to use sklearn's is_classifier utility if available
    try:
        from sklearn.base import is_classifier as sklearn_is_classifier
        return sklearn_is_classifier(model)
    except ImportError:
        pass
        
    # If no clear indicators, assume it's not a classifier
    return False


def is_regressor(model: Any) -> bool:
    """
    Check if a model is a regressor.
    
    Parameters:
    -----------
    model : Any
        Machine learning model
        
    Returns:
    --------
    bool : True if the model is a regressor
    """
    # Check the model's class name
    model_name = type(model).__name__.lower()
    regressor_indicators = ['regressor', 'svr', 'linear', 'ridge', 'lasso', 'elastic', 'kneighbors']
    
    for indicator in regressor_indicators:
        if indicator in model_name:
            return True
            
    # Check predict method output type to differentiate from classifiers
    if hasattr(model, 'predict') and hasattr(model, 'fit'):
        # Try to use sklearn's is_regressor utility if available
        try:
            from sklearn.base import is_regressor as sklearn_is_regressor
            return sklearn_is_regressor(model)
        except ImportError:
            pass
            
        # If model doesn't have classifier-specific attributes, assume it's a regressor
        if not (hasattr(model, 'classes_') or hasattr(model, 'predict_proba')):
            return True
            
    # If no clear indicators, assume it's not a regressor
    return False


def get_model_type(model: Any) -> str:
    """
    Get the type of a model (classifier or regressor).
    
    Parameters:
    -----------
    model : Any
        Machine learning model
        
    Returns:
    --------
    str : 'classifier', 'regressor', or 'unknown'
    """
    if is_classifier(model):
        return 'classifier'
    elif is_regressor(model):
        return 'regressor'
    else:
        return 'unknown'


def get_model_library(model: Any) -> str:
    """
    Identify which library a model is from.
    
    Parameters:
    -----------
    model : Any
        Machine learning model
        
    Returns:
    --------
    str : Library name ('sklearn', 'xgboost', 'lightgbm', etc.)
    """
    # Get model module
    model_module = type(model).__module__
    
    # Check for common machine learning libraries
    if model_module.startswith('sklearn'):
        return 'sklearn'
    elif model_module.startswith('xgboost'):
        return 'xgboost'
    elif model_module.startswith('lightgbm'):
        return 'lightgbm'
    elif model_module.startswith('catboost'):
        return 'catboost'
    elif model_module.startswith('keras') or model_module.startswith('tensorflow'):
        return 'tensorflow'
    elif model_module.startswith('torch'):
        return 'pytorch'
    else:
        return 'unknown'


def is_ensemble_model(model: Any) -> bool:
    """
    Check if a model is an ensemble model.
    
    Parameters:
    -----------
    model : Any
        Machine learning model
        
    Returns:
    --------
    bool : True if the model is an ensemble
    """
    # Check for common ensemble model attributes
    if hasattr(model, 'estimators_'):
        return True
        
    # Check the model's class name
    model_name = type(model).__name__.lower()
    ensemble_indicators = ['forest', 'boost', 'bagging', 'voting', 'stack']
    
    for indicator in ensemble_indicators:
        if indicator in model_name:
            return True
            
    # If no clear indicators, assume it's not an ensemble
    return False


def get_base_estimator(model: Any) -> Optional[Any]:
    """
    Get the base estimator from an ensemble model.
    
    Parameters:
    -----------
    model : Any
        Machine learning model
        
    Returns:
    --------
    Any or None : Base estimator if model is an ensemble
    """
    # Check for base_estimator attribute (common in scikit-learn)
    if hasattr(model, 'base_estimator_'):
        return model.base_estimator_
    elif hasattr(model, 'base_estimator'):
        return model.base_estimator
        
    # Check for estimator attribute (common in some ensembles)
    if hasattr(model, 'estimator'):
        return model.estimator
        
    # Check for estimators_ list (for models with multiple estimators)
    if hasattr(model, 'estimators_') and model.estimators_:
        # Return first estimator as an example
        return model.estimators_[0]
        
    return None