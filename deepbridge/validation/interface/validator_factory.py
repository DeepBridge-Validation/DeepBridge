"""
Factory for creating validators.

This module provides factory functions for creating validator instances
based on model type and validation requirements.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Callable, Tuple


def create_validator(
    validator_type: str,
    model: Any,
    **kwargs
) -> Any:
    """
    Create a validator instance.
    
    Parameters:
    -----------
    validator_type : str
        Type of validator to create: 'robustness', 'uncertainty', 'hyperparameter'
    model : Any
        Machine learning model
    **kwargs : dict
        Additional parameters for the validator
        
    Returns:
    --------
    Any : Validator instance
    """
    if validator_type == 'robustness':
        return create_robustness_validator(model, **kwargs)
    elif validator_type == 'uncertainty':
        return create_uncertainty_validator(model, **kwargs)
    elif validator_type == 'hyperparameter':
        return create_hyperparameter_validator(model, **kwargs)
    else:
        raise ValueError(f"Unknown validator type: {validator_type}")


def create_robustness_validator(
    model: Any,
    problem_type: str = 'auto',
    perturbation_types: Optional[List[str]] = None,
    perturbation_levels: Optional[List[float]] = None,
    **kwargs
) -> Any:
    """
    Create a robustness validator.
    
    Parameters:
    -----------
    model : Any
        Machine learning model
    problem_type : str
        Type of problem: 'classification', 'regression', or 'auto' (detect)
    perturbation_types : list of str or None
        Types of perturbations to apply
    perturbation_levels : list of float or None
        Levels of perturbation to apply
    **kwargs : dict
        Additional parameters for the validator
        
    Returns:
    --------
    Any : Robustness validator instance
    """
    from ..frameworks.robustness import RobustnessTester
    
    # Set default perturbation types if not provided
    if perturbation_types is None:
        perturbation_types = ['noise', 'zero']
        
    # Set default perturbation levels if not provided
    if perturbation_levels is None:
        perturbation_levels = [0.1, 0.3, 0.5]
        
    # Extract required parameters
    X = kwargs.get('X')
    y = kwargs.get('y')
    
    if X is None or y is None:
        raise ValueError("X and y must be provided")
        
    # Create validator
    validator = RobustnessTester(
        model=model,
        X=X,
        y=y,
        problem_type=problem_type,
        **{k: v for k, v in kwargs.items() if k not in ['X', 'y']}
    )
    
    return validator


def create_uncertainty_validator(
    model: Any,
    problem_type: str = 'auto',
    uncertainty_method: str = 'conformal',
    **kwargs
) -> Any:
    """
    Create an uncertainty validator.
    
    Parameters:
    -----------
    model : Any
        Machine learning model
    problem_type : str
        Type of problem: 'classification', 'regression', or 'auto' (detect)
    uncertainty_method : str
        Method for uncertainty estimation: 'conformal', 'calibration', 'bootstrap'
    **kwargs : dict
        Additional parameters for the validator
        
    Returns:
    --------
    Any : Uncertainty validator instance
    """
    # Create validator based on method
    if uncertainty_method == 'conformal':
        from ..frameworks.uncertainty import ConformalPredictor
        
        # Extract parameters
        alpha = kwargs.get('alpha', 0.1)
        
        # Create validator
        validator = ConformalPredictor(
            model=model,
            problem_type=problem_type,
            alpha=alpha,
            **{k: v for k, v in kwargs.items() if k != 'alpha'}
        )
        
    elif uncertainty_method == 'calibration':
        from ..frameworks.uncertainty import CalibrationAnalyzer
        
        # Extract parameters
        method = kwargs.get('calibration_method', 'isotonic')
        
        # Check if model has predict_proba
        if not hasattr(model, 'predict_proba'):
            raise ValueError("Model must have predict_proba method for calibration")
            
        # Create validator
        validator = CalibrationAnalyzer(
            model=model,
            method=method,
            **{k: v for k, v in kwargs.items() if k != 'calibration_method'}
        )
        
    elif uncertainty_method == 'bootstrap':
        from ..frameworks.uncertainty import BootstrapUncertainty
        
        # Extract parameters
        n_bootstrap = kwargs.get('n_bootstrap', 100)
        
        # Create validator
        validator = BootstrapUncertainty(
            base_model=model,
            n_bootstrap=n_bootstrap,
            **{k: v for k, v in kwargs.items() if k != 'n_bootstrap'}
        )
        
    else:
        raise ValueError(f"Unknown uncertainty method: {uncertainty_method}")
        
    return validator


def create_hyperparameter_validator(
    model: Any,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    tuning_method: str = 'bayesian',
    **kwargs
) -> Any:
    """
    Create a hyperparameter validator.
    
    Parameters:
    -----------
    model : Any
        Machine learning model
    param_grid : dict or None
        Grid of hyperparameters to search
    tuning_method : str
        Method for tuning: 'bayesian', 'importance'
    **kwargs : dict
        Additional parameters for the validator
        
    Returns:
    --------
    Any : Hyperparameter validator instance
    """
    # Check if model has get_params method
    if not hasattr(model, 'get_params'):
        raise ValueError("Model must have get_params method for hyperparameter validation")
        
    # Get default param grid if not provided
    if param_grid is None:
        # Try to get default param grid based on model type
        from ..interface.simplified_api import _get_default_param_grid
        param_grid = _get_default_param_grid(model)
        
    # Create validator based on method
    if tuning_method == 'bayesian':
        from ..frameworks.hyperparameters import EfficientTuner
        
        # Extract parameters
        n_iterations = kwargs.get('n_iterations', 20)
        
        # Create validator
        validator = EfficientTuner(
            param_space=param_grid,
            n_iterations=n_iterations,
            method='bayesian',
            **{k: v for k, v in kwargs.items() if k not in ['n_iterations']}
        )
        
    elif tuning_method == 'importance':
        from ..frameworks.hyperparameters import HyperparameterImportance
        
        # Extract parameters
        n_samples = kwargs.get('n_samples', 50)
        
        # Create validator
        validator = HyperparameterImportance(
            param_space=param_grid,
            n_samples=n_samples,
            **{k: v for k, v in kwargs.items() if k != 'n_samples'}
        )
        
    else:
        raise ValueError(f"Unknown tuning method: {tuning_method}")
        
    return validator


def detect_model_type(model: Any) -> str:
    """
    Detect the type of a machine learning model.
    
    Parameters:
    -----------
    model : Any
        Machine learning model
        
    Returns:
    --------
    str : Model type
    """
    model_name = model.__class__.__name__.lower()
    
    # Detect model type based on name
    if any(x in model_name for x in ['classifier', 'svc', 'logistic', 'forest']):
        return 'classification'
    elif any(x in model_name for x in ['regressor', 'svr', 'linear']):
        return 'regression'
    else:
        # Try to infer from model attributes
        if hasattr(model, 'predict_proba'):
            return 'classification'
        else:
            return 'regression'


def auto_create_validator(
    model: Any,
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    validation_goal: str = 'robustness'
) -> Any:
    """
    Automatically create an appropriate validator based on model type and goal.
    
    Parameters:
    -----------
    model : Any
        Machine learning model
    X : array-like or DataFrame
        Input features
    y : array-like or Series
        Target values
    validation_goal : str
        Validation goal: 'robustness', 'uncertainty', 'hyperparameter'
        
    Returns:
    --------
    Any : Validator instance
    """
    # Detect problem type
    problem_type = detect_model_type(model)
    
    # Create appropriate validator
    if validation_goal == 'robustness':
        return create_robustness_validator(model, problem_type=problem_type, X=X, y=y)
    elif validation_goal == 'uncertainty':
        # Choose appropriate uncertainty method based on model type
        if problem_type == 'classification' and hasattr(model, 'predict_proba'):
            uncertainty_method = 'calibration'
        else:
            uncertainty_method = 'conformal'
            
        return create_uncertainty_validator(
            model, 
            problem_type=problem_type, 
            uncertainty_method=uncertainty_method
        )
    elif validation_goal == 'hyperparameter':
        return create_hyperparameter_validator(model)
    else:
        raise ValueError(f"Unknown validation goal: {validation_goal}")