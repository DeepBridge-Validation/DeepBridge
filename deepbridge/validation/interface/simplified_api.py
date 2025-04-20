"""
Simplified API for the validation framework.

This module provides a simplified, high-level API for common
validation tasks without requiring detailed knowledge of the
underlying components.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
import matplotlib.pyplot as plt
from ..core import ReportGenerator


def validate_model(
    model: Any,
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    validation_types: List[str] = ['robustness', 'uncertainty'],
    problem_type: str = 'auto',
    report_format: str = 'markdown'
) -> Dict[str, Any]:
    """
    Perform comprehensive model validation.
    
    Parameters:
    -----------
    model : Any
        Trained machine learning model
    X : array-like or DataFrame
        Input features
    y : array-like or Series
        Target values
    validation_types : list of str
        Types of validation to perform
    problem_type : str
        Type of problem: 'classification', 'regression', or 'auto' (detect)
    report_format : str
        Format for the generated report
        
    Returns:
    --------
    dict : Validation results and report
    """
    results = {}
    
    # Detect problem type if auto
    if problem_type == 'auto':
        n_unique = len(np.unique(y))
        if n_unique < 10:
            problem_type = 'classification'
        else:
            problem_type = 'regression'
            
    # Perform each validation type
    if 'robustness' in validation_types:
        robustness_results = evaluate_robustness(model, X, y, problem_type)
        results['robustness'] = robustness_results
        
    if 'uncertainty' in validation_types:
        uncertainty_results = evaluate_uncertainty(model, X, y, problem_type)
        results['uncertainty'] = uncertainty_results
        
    if 'hyperparameters' in validation_types:
        # For hyperparameters, we need a model class, not a fitted instance
        # Here we just check if this is possible
        if hasattr(model, 'get_params'):
            hyperparameter_results = tune_hyperparameters(model, X, y, problem_type)
            results['hyperparameters'] = hyperparameter_results
            
    # Generate report
    report = generate_report(results, report_format)
    results['report'] = report
    
    return results


def evaluate_robustness(
    model: Any,
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    problem_type: str = 'auto',
    perturbation_types: Optional[List[str]] = None,
    perturbation_levels: Optional[List[float]] = None,
    test_adversarial: bool = False
) -> Dict[str, Any]:
    """
    Evaluate model robustness against perturbations.
    
    Parameters:
    -----------
    model : Any
        Trained machine learning model
    X : array-like or DataFrame
        Input features
    y : array-like or Series
        Target values
    problem_type : str
        Type of problem: 'classification', 'regression', or 'auto' (detect)
    perturbation_types : list of str or None
        Types of perturbations to apply
    perturbation_levels : list of float or None
        Levels of perturbations to apply
    test_adversarial : bool
        Whether to test adversarial examples
        
    Returns:
    --------
    dict : Robustness evaluation results
    """
    from ..frameworks.robustness import run_robustness_test_suite
    
    # Set default perturbation types if not provided
    if perturbation_types is None:
        perturbation_types = ['noise', 'zero']
        
    # Set default perturbation levels if not provided
    if perturbation_levels is None:
        perturbation_levels = [0.1, 0.3, 0.5]
        
    # Run robustness test suite
    results = run_robustness_test_suite(
        model=model,
        X=X,
        y=y,
        problem_type=problem_type,
        perturbation_types=perturbation_types,
        perturbation_levels=perturbation_levels
    )
    
    # Test adversarial examples if requested and it's a classification problem
    if test_adversarial and problem_type == 'classification':
        try:
            from ..frameworks.robustness.adversarial import generate_fgsm_attack
            
            # Try to generate adversarial examples
            X_adv = generate_fgsm_attack(
                model=model,
                X=X,
                y=y,
                epsilon=0.1
            )
            
            # Evaluate model on adversarial examples
            if hasattr(model, 'predict'):
                y_pred = model.predict(X_adv)
                
                # Calculate accuracy
                accuracy = np.mean(y_pred == y)
                
                # Add to results
                results['adversarial'] = {
                    'accuracy': accuracy,
                    'epsilon': 0.1
                }
        except (ImportError, ValueError) as e:
            # If error occurs, skip adversarial testing
            results['adversarial'] = {
                'error': str(e)
            }
    
    return results


def evaluate_uncertainty(
    model: Any,
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    problem_type: str = 'auto',
    calibration_method: str = 'isotonic',
    uncertainty_method: str = 'conformal'
) -> Dict[str, Any]:
    """
    Evaluate model uncertainty estimates.
    
    Parameters:
    -----------
    model : Any
        Trained machine learning model
    X : array-like or DataFrame
        Input features
    y : array-like or Series
        Target values
    problem_type : str
        Type of problem: 'classification', 'regression', or 'auto' (detect)
    calibration_method : str
        Method for probability calibration
    uncertainty_method : str
        Method for uncertainty estimation
        
    Returns:
    --------
    dict : Uncertainty evaluation results
    """
    results = {}
    
    # Detect problem type if auto
    if problem_type == 'auto':
        n_unique = len(np.unique(y))
        if n_unique < 10:
            problem_type = 'classification'
        else:
            problem_type = 'regression'
    
    # For classification, evaluate calibration
    if problem_type == 'classification':
        from ..frameworks.uncertainty import (
            CalibrationAnalyzer,
            compute_uncertainty_metrics
        )
        
        # Check if model has predict_proba method
        if hasattr(model, 'predict_proba'):
            # Initialize calibration analyzer
            calibration_analyzer = CalibrationAnalyzer(
                model=model,
                method=calibration_method
            )
            
            # Analyze calibration
            calibration_results = calibration_analyzer.analyze_calibration(X, y)
            
            # Add to results
            results['calibration'] = calibration_results
            
            # Create calibration plot
            try:
                calibration_plot = calibration_analyzer.plot_calibration(X, y)
                plt.close(calibration_plot)  # Close to prevent display
                
                # Add plot to results
                if 'plots' not in results:
                    results['plots'] = {}
                    
                results['plots']['calibration_curve'] = calibration_plot
            except Exception as e:
                # Skip plotting if error occurs
                pass
                
            # Compute uncertainty metrics
            y_prob = model.predict_proba(X)
            
            # For binary classification
            if y_prob.shape[1] == 2:
                y_prob = y_prob[:, 1]
                
                # Convert targets to binary
                if isinstance(y, pd.Series):
                    y_binary = (y.values == model.classes_[1]).astype(int)
                else:
                    y_binary = (y == model.classes_[1]).astype(int)
                    
                # Compute metrics
                metrics = compute_uncertainty_metrics(y_binary, y_prob)
                
                # Add to results
                results['metrics'] = metrics
    
    # For regression or classification, evaluate prediction intervals/sets
    if uncertainty_method == 'conformal':
        from ..frameworks.uncertainty import ConformalPredictor
        
        try:
            # Initialize conformal predictor
            conformal_predictor = ConformalPredictor(
                model=model,
                problem_type=problem_type,
                alpha=0.1  # 90% coverage
            )
            
            # Generate prediction intervals/sets
            intervals_or_sets = conformal_predictor.calibrate_and_predict(X, y)
            
            # Evaluate coverage
            if problem_type == 'regression':
                # For regression, calculate coverage
                coverage = np.mean((y >= intervals_or_sets[:, 0]) & (y <= intervals_or_sets[:, 1]))
                avg_width = np.mean(intervals_or_sets[:, 1] - intervals_or_sets[:, 0])
                
                # Add to results
                results['conformal'] = {
                    'coverage': coverage,
                    'average_width': avg_width,
                    'target_coverage': 0.9
                }
            else:
                # For classification, calculate set size and coverage
                set_sizes = [len(s) for s in intervals_or_sets]
                avg_set_size = np.mean(set_sizes)
                
                # Convert y to array if needed
                if isinstance(y, pd.Series):
                    y_values = y.values
                else:
                    y_values = y
                    
                # Calculate coverage
                coverage = np.mean([y_values[i] in s for i, s in enumerate(intervals_or_sets)])
                
                # Add to results
                results['conformal'] = {
                    'coverage': coverage,
                    'average_set_size': avg_set_size,
                    'target_coverage': 0.9
                }
                
            # Create conformal coverage plot
            try:
                coverage_plot = conformal_predictor.plot_coverage()
                plt.close(coverage_plot)  # Close to prevent display
                
                # Add plot to results
                if 'plots' not in results:
                    results['plots'] = {}
                    
                results['plots']['conformal_coverage'] = coverage_plot
            except Exception as e:
                # Skip plotting if error occurs
                pass
                
        except Exception as e:
            # Skip conformal prediction if error occurs
            results['conformal'] = {
                'error': str(e)
            }
    
    return results


def tune_hyperparameters(
    model: Any,
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    problem_type: str = 'auto',
    param_grid: Optional[Dict[str, List[Any]]] = None,
    n_iterations: int = 20,
    tuning_method: str = 'bayesian'
) -> Dict[str, Any]:
    """
    Tune model hyperparameters.
    
    Parameters:
    -----------
    model : Any
        Machine learning model class or instance
    X : array-like or DataFrame
        Input features
    y : array-like or Series
        Target values
    problem_type : str
        Type of problem: 'classification', 'regression', or 'auto' (detect)
    param_grid : dict or None
        Grid of hyperparameters to search
    n_iterations : int
        Number of tuning iterations
    tuning_method : str
        Method for tuning: 'bayesian', 'random', or 'grid'
        
    Returns:
    --------
    dict : Hyperparameter tuning results
    """
    results = {}
    
    # Check if model has get_params method
    if not hasattr(model, 'get_params'):
        return {'error': 'Model does not support hyperparameter tuning'}
        
    # Get default param grid if not provided
    if param_grid is None:
        # Try to get default param grid based on model type
        try:
            param_grid = _get_default_param_grid(model)
        except:
            return {'error': 'Could not determine parameter grid for model'}
    
    # Use scikit-learn for hyperparameter tuning
    try:
        # For bayesian optimization
        if tuning_method == 'bayesian':
            from ..frameworks.hyperparameters import bayesian_optimization_tuner
            
            # Define scoring function
            from sklearn.model_selection import cross_val_score
            
            def scoring_fn(params):
                # Create model with params
                model_instance = model.__class__(**{**model.get_params(), **params})
                
                # Evaluate using cross-validation
                scores = cross_val_score(model_instance, X, y, cv=3)
                return np.mean(scores)
                
            # Run bayesian optimization
            tuning_results = bayesian_optimization_tuner(
                param_space=param_grid,
                scoring_function=scoring_fn,
                n_iterations=n_iterations
            )
            
        # For random search
        elif tuning_method == 'random':
            from sklearn.model_selection import RandomizedSearchCV
            
            # Create random search
            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=n_iterations,
                cv=3,
                random_state=42
            )
            
            # Fit random search
            random_search.fit(X, y)
            
            # Extract results
            tuning_results = {
                'best_params': random_search.best_params_,
                'best_score': random_search.best_score_,
                'evaluations': [
                    {'params': params, 'score': score}
                    for params, score in zip(random_search.cv_results_['params'], 
                                           random_search.cv_results_['mean_test_score'])
                ]
            }
            
        # For grid search
        elif tuning_method == 'grid':
            from sklearn.model_selection import GridSearchCV
            
            # Create grid search
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=3
            )
            
            # Fit grid search
            grid_search.fit(X, y)
            
            # Extract results
            tuning_results = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'evaluations': [
                    {'params': params, 'score': score}
                    for params, score in zip(grid_search.cv_results_['params'], 
                                           grid_search.cv_results_['mean_test_score'])
                ]
            }
            
        else:
            return {'error': f'Unsupported tuning method: {tuning_method}'}
            
        # Add results
        results = tuning_results
        
    except Exception as e:
        # If error occurs, return error message
        results = {'error': str(e)}
    
    return results


def generate_report(
    results: Dict[str, Any],
    format: str = 'markdown',
    title: str = "Validation Report"
) -> Union[str, Dict[str, Any]]:
    """
    Generate a validation report.
    
    Parameters:
    -----------
    results : dict
        Validation results
    format : str
        Report format: 'markdown', 'html', 'dict', or 'dataframe'
    title : str
        Report title
        
    Returns:
    --------
    str or dict : Formatted report
    """
    from ..interface.report_formatter import format_report
    
    return format_report(results, format, title)


def _get_default_param_grid(model: Any) -> Dict[str, List[Any]]:
    """
    Get default parameter grid for a model.
    
    Parameters:
    -----------
    model : Any
        Machine learning model
        
    Returns:
    --------
    dict : Default parameter grid
    """
    model_name = model.__class__.__name__.lower()
    
    # Default grids for common models
    if 'randomforest' in model_name:
        return {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif 'gradientboosting' in model_name:
        return {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        }
    elif 'svm' in model_name or 'svc' in model_name:
        return {
            'C': [0.1, 1.0, 10.0],
            'gamma': ['scale', 'auto', 0.1, 0.01],
            'kernel': ['rbf', 'linear']
        }
    elif 'logistic' in model_name:
        return {
            'C': [0.1, 1.0, 10.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
    elif 'knn' in model_name or 'neighbor' in model_name:
        return {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]
        }
    else:
        # Generic grid
        return {}