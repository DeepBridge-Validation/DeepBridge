"""
Hyperparameter tuning for classification models.

This module provides specialized tools for hyperparameter optimization
and analysis tailored for classification models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
)


class ClassificationHyperparameterTests:
    """
    Hyperparameter tests for classification models.
    
    This class provides methods for testing and optimizing hyperparameters
    of classification models.
    """
    
    def __init__(
        self,
        model: Any,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        scoring: str = 'accuracy',
        cv: int = 5,
        random_state: Optional[int] = None,
        verbose: bool = False,
        n_jobs: int = 1
    ):
        """
        Initialize the classification hyperparameter tests.
        
        Parameters:
        -----------
        model : Any
            Classification model to test
        param_grid : dict or None
            Grid of hyperparameters to search
        scoring : str
            Scoring metric: 'accuracy', 'f1', 'precision', 'recall', 'roc_auc'
        cv : int
            Number of cross-validation folds
        random_state : int or None
            Random seed for reproducibility
        verbose : bool
            Whether to print progress information
        n_jobs : int
            Number of parallel jobs
        """
        self.model = model
        self.param_grid = param_grid or get_classification_param_grid(model)
        self.scoring = scoring
        self.cv = cv
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs
        
        # Initialize results storage
        self.results = {}
    
    def validate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        method: str = 'bayesian',
        n_iterations: int = 20,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Validate hyperparameters of the classification model.
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Feature data
        y : array-like or Series
            Target data
        method : str
            Optimization method: 'bayesian', 'random', 'grid'
        n_iterations : int
            Number of iterations for optimization
        **kwargs : dict
            Additional parameters
            
        Returns:
        --------
        dict : Validation results
        """
        # Ensure model type is appropriate
        self._check_model_type()
        
        # Tune hyperparameters
        tuning_results = tune_classification_hyperparameters(
            self.model,
            X, y,
            param_grid=self.param_grid,
            method=method,
            scoring=self.scoring,
            cv=self.cv,
            n_iterations=n_iterations,
            random_state=self.random_state,
            verbose=self.verbose,
            n_jobs=self.n_jobs,
            **kwargs
        )
        
        # Store results
        self.results = tuning_results
        
        # Evaluate best model
        evaluation_results = evaluate_classifier_hyperparameters(
            self.model,
            X, y,
            best_params=tuning_results['best_params'],
            cv=self.cv,
            random_state=self.random_state,
            verbose=self.verbose
        )
        
        # Combine results
        combined_results = {
            'tuning': tuning_results,
            'evaluation': evaluation_results
        }
        
        return combined_results
    
    def plot_tuning_results(
        self,
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Plot hyperparameter tuning results.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
            
        Returns:
        --------
        matplotlib.Figure : Tuning results plot
        """
        if not self.results:
            raise ValueError("No tuning results available. Call validate() first.")
            
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Optimization history
        scores = [result['score'] for result in self.results.get('evaluations', [])]
        iterations = list(range(1, len(scores) + 1))
        
        axes[0].plot(iterations, scores, 'o-', color='blue')
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel(f'{self.scoring.replace("_", " ").title()} Score')
        axes[0].set_title('Hyperparameter Optimization History')
        
        # Add best score line
        best_score = self.results.get('best_score', max(scores) if scores else 0)
        axes[0].axhline(y=best_score, color='r', linestyle='--', 
                      label=f'Best Score: {best_score:.4f}')
        axes[0].legend()
        
        # Plot 2: Parameter importance
        if 'parameter_importance' in self.results:
            importance = self.results['parameter_importance']
            
            # Sort parameters by importance
            sorted_params = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            params = [p[0] for p in sorted_params]
            values = [p[1] for p in sorted_params]
            
            # Plot
            y_pos = np.arange(len(params))
            axes[1].barh(y_pos, values, align='center')
            axes[1].set_yticks(y_pos)
            axes[1].set_yticklabels(params)
            axes[1].invert_yaxis()  # Labels read top-to-bottom
            axes[1].set_xlabel('Importance Score')
            axes[1].set_title('Hyperparameter Importance')
        else:
            axes[1].set_title('Parameter Importance Not Available')
            axes[1].axis('off')
        
        plt.tight_layout()
        return fig
    
    def plot_evaluation_results(
        self,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot hyperparameter evaluation results.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
            
        Returns:
        --------
        matplotlib.Figure : Evaluation results plot
        """
        if not self.results or 'evaluation' not in self.results:
            raise ValueError("No evaluation results available. Call validate() first.")
            
        evaluation = self.results.get('evaluation', {})
        
        # Get metric scores
        metrics = evaluation.get('metrics', {})
        
        if not metrics:
            raise ValueError("No metric data available in evaluation results.")
            
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Format metric names and values
        metric_names = [m.replace('_', ' ').title() for m in metrics.keys()]
        metric_values = list(metrics.values())
        
        # Plot bar chart
        x_pos = np.arange(len(metric_names))
        bars = ax.bar(x_pos, metric_values, align='center', alpha=0.7)
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                  f'{height:.4f}', ha='center', va='bottom')
        
        # Set labels and title
        ax.set_xlabel('Metric')
        ax.set_ylabel('Score')
        ax.set_title('Classification Performance Metrics')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metric_names)
        ax.set_ylim(0, 1.1)  # Most classification metrics are in [0, 1]
        
        plt.tight_layout()
        return fig
    
    def _check_model_type(self):
        """Check if model is appropriate for classification."""
        # Try to infer if it's a classifier
        model_name = self.model.__class__.__name__.lower()
        
        is_classifier = (
            'classifier' in model_name or
            hasattr(self.model, 'predict_proba') or
            hasattr(self.model, 'classes_')
        )
        
        if not is_classifier:
            warnings.warn("Model does not appear to be a classifier. "
                        "Some tests may not work as expected.")


def tune_classification_hyperparameters(
    model: Any,
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    param_grid: Optional[Dict[str, List[Any]]] = None,
    method: str = 'bayesian',
    scoring: str = 'accuracy',
    cv: int = 5,
    n_iterations: int = 20,
    random_state: Optional[int] = None,
    verbose: bool = False,
    n_jobs: int = 1,
    **kwargs
) -> Dict[str, Any]:
    """
    Tune hyperparameters of a classification model.
    
    Parameters:
    -----------
    model : Any
        Classification model to tune
    X : array-like or DataFrame
        Feature data
    y : array-like or Series
        Target data
    param_grid : dict or None
        Grid of hyperparameters to search
    method : str
        Optimization method: 'bayesian', 'random', 'grid'
    scoring : str
        Scoring metric: 'accuracy', 'f1', 'precision', 'recall', 'roc_auc'
    cv : int
        Number of cross-validation folds
    n_iterations : int
        Number of iterations for optimization
    random_state : int or None
        Random seed for reproducibility
    verbose : bool
        Whether to print progress information
    n_jobs : int
        Number of parallel jobs
    **kwargs : dict
        Additional parameters
        
    Returns:
    --------
    dict : Tuning results
    """
    # Get default param grid if not provided
    if param_grid is None:
        param_grid = get_classification_param_grid(model)
        
    # Convert scoring string to scorer function
    scorer = get_classification_scorer(scoring)
    
    # Configure cross-validation
    cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    # Define scoring function for optimization
    def score_params(params):
        # Clone model and set parameters
        from sklearn.base import clone
        model_clone = clone(model)
        model_clone.set_params(**params)
        
        # Evaluate using cross-validation
        scores = cross_val_score(
            model_clone, X, y, 
            scoring=scorer, 
            cv=cv_obj, 
            n_jobs=n_jobs
        )
        
        return np.mean(scores)
    
    # Perform hyperparameter tuning
    if method == 'bayesian':
        try:
            from sklearn.model_selection import cross_val_score
            from skopt import gp_minimize
            from skopt.space import Real, Integer, Categorical
            from skopt.utils import use_named_args
            
            # Convert param grid to skopt dimensions
            dimensions = []
            dimension_names = []
            
            for param_name, param_values in param_grid.items():
                if isinstance(param_values, list):
                    if all(isinstance(v, (int, float)) for v in param_values):
                        # Numerical list - consider as a categorical choice
                        dimensions.append(Categorical(param_values, name=param_name))
                    elif all(isinstance(v, str) for v in param_values):
                        # String list - categorical
                        dimensions.append(Categorical(param_values, name=param_name))
                    else:
                        # Mixed list - treat as categorical
                        dimensions.append(Categorical(param_values, name=param_name))
                elif isinstance(param_values, tuple) and len(param_values) == 2:
                    # Range (min, max)
                    min_val, max_val = param_values
                    if all(isinstance(v, int) for v in param_values):
                        # Integer range
                        dimensions.append(Integer(min_val, max_val, name=param_name))
                    else:
                        # Float range
                        dimensions.append(Real(min_val, max_val, name=param_name))
                
                dimension_names.append(param_name)
            
            # Objective function for optimization
            @use_named_args(dimensions)
            def objective(**params):
                return -score_params(params)  # Negate for minimization
            
            # Run Bayesian optimization
            result = gp_minimize(
                objective,
                dimensions,
                n_calls=n_iterations,
                random_state=random_state,
                verbose=verbose
            )
            
            # Extract results
            best_params = {dim.name: result.x[i] for i, dim in enumerate(dimensions)}
            best_score = -result.fun  # Un-negate
            
            # Create evaluations list
            evaluations = []
            for i, (x, y) in enumerate(zip(result.x_iters, result.func_vals)):
                params = {dimensions[j].name: x[j] for j in range(len(dimensions))}
                evaluations.append({
                    'iteration': i,
                    'params': params,
                    'score': -y  # Un-negate
                })
                
            # Calculate parameter importance if possible
            try:
                from skopt.plots import plot_objective
                parameter_importance = {}
                for i, dim in enumerate(dimensions):
                    importance = result.space.dimension_names[i]
                    parameter_importance[dim.name] = importance
            except:
                parameter_importance = None
                
        except ImportError:
            # Fallback to random search
            if verbose:
                print("Warning: scikit-optimize not available. Falling back to random search.")
            
            from sklearn.model_selection import RandomizedSearchCV
            
            random_search = RandomizedSearchCV(
                model, param_grid, 
                n_iter=n_iterations,
                scoring=scorer,
                cv=cv_obj,
                random_state=random_state,
                verbose=verbose,
                n_jobs=n_jobs
            )
            
            random_search.fit(X, y)
            
            best_params = random_search.best_params_
            best_score = random_search.best_score_
            
            # Create evaluations list
            evaluations = []
            for i, (params, score) in enumerate(zip(
                random_search.cv_results_['params'],
                random_search.cv_results_['mean_test_score']
            )):
                evaluations.append({
                    'iteration': i,
                    'params': params,
                    'score': score
                })
                
            parameter_importance = None
            
    elif method == 'random':
        from sklearn.model_selection import RandomizedSearchCV
        
        random_search = RandomizedSearchCV(
            model, param_grid, 
            n_iter=n_iterations,
            scoring=scorer,
            cv=cv_obj,
            random_state=random_state,
            verbose=verbose,
            n_jobs=n_jobs
        )
        
        random_search.fit(X, y)
        
        best_params = random_search.best_params_
        best_score = random_search.best_score_
        
        # Create evaluations list
        evaluations = []
        for i, (params, score) in enumerate(zip(
            random_search.cv_results_['params'],
            random_search.cv_results_['mean_test_score']
        )):
            evaluations.append({
                'iteration': i,
                'params': params,
                'score': score
            })
            
        # Calculate simple parameter importance
        parameter_importance = {}
        for param_name in param_grid.keys():
            if param_name in random_search.cv_results_['param_' + param_name]:
                importance = np.std(random_search.cv_results_['mean_test_score'])
                parameter_importance[param_name] = importance
            
    elif method == 'grid':
        from sklearn.model_selection import GridSearchCV
        
        grid_search = GridSearchCV(
            model, param_grid, 
            scoring=scorer,
            cv=cv_obj,
            verbose=verbose,
            n_jobs=n_jobs
        )
        
        grid_search.fit(X, y)
        
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        # Create evaluations list
        evaluations = []
        for i, (params, score) in enumerate(zip(
            grid_search.cv_results_['params'],
            grid_search.cv_results_['mean_test_score']
        )):
            evaluations.append({
                'iteration': i,
                'params': params,
                'score': score
            })
            
        # Calculate simple parameter importance
        parameter_importance = {}
        for param_name in param_grid.keys():
            if param_name in grid_search.cv_results_['param_' + param_name]:
                importance = np.std(grid_search.cv_results_['mean_test_score'])
                parameter_importance[param_name] = importance
            
    else:
        raise ValueError(f"Unknown optimization method: {method}")
        
    # Normalize parameter importance if available
    if parameter_importance:
        max_importance = max(parameter_importance.values())
        if max_importance > 0:
            parameter_importance = {
                k: v / max_importance for k, v in parameter_importance.items()
            }
    
    # Return results
    return {
        'best_params': best_params,
        'best_score': best_score,
        'scoring': scoring,
        'evaluations': evaluations,
        'parameter_importance': parameter_importance
    }


def evaluate_classifier_hyperparameters(
    model: Any,
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    best_params: Dict[str, Any],
    cv: int = 5,
    random_state: Optional[int] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Evaluate a classifier with optimal hyperparameters.
    
    Parameters:
    -----------
    model : Any
        Classification model
    X : array-like or DataFrame
        Feature data
    y : array-like or Series
        Target data
    best_params : dict
        Optimal hyperparameters
    cv : int
        Number of cross-validation folds
    random_state : int or None
        Random seed for reproducibility
    verbose : bool
        Whether to print progress information
        
    Returns:
    --------
    dict : Evaluation results
    """
    from sklearn.base import clone
    from sklearn.model_selection import cross_validate, StratifiedKFold
    
    # Clone model and set best parameters
    best_model = clone(model)
    best_model.set_params(**best_params)
    
    # Define scoring metrics
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision_weighted',
        'recall': 'recall_weighted',
        'f1': 'f1_weighted',
        'roc_auc': 'roc_auc_ovr_weighted'
    }
    
    # Configure cross-validation
    cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    # Perform cross-validation
    cv_results = cross_validate(
        best_model, X, y,
        scoring=scoring,
        cv=cv_obj,
        return_train_score=True,
        verbose=verbose
    )
    
    # Extract and format results
    metrics = {}
    for metric in scoring.keys():
        test_key = f'test_{metric}'
        train_key = f'train_{metric}'
        
        if test_key in cv_results and train_key in cv_results:
            metrics[metric] = float(np.mean(cv_results[test_key]))
            metrics[f'{metric}_train'] = float(np.mean(cv_results[train_key]))
            metrics[f'{metric}_std'] = float(np.std(cv_results[test_key]))
    
    # Calculate overfitting metrics
    for metric in scoring.keys():
        if f'{metric}' in metrics and f'{metric}_train' in metrics:
            metrics[f'{metric}_overfit'] = metrics[f'{metric}_train'] - metrics[f'{metric}']
    
    # Return results
    return {
        'best_params': best_params,
        'metrics': metrics,
        'cv_results': {k: v.tolist() for k, v in cv_results.items()}
    }


def get_classification_param_grid(model: Any) -> Dict[str, List[Any]]:
    """
    Get default hyperparameter grid for a classification model.
    
    Parameters:
    -----------
    model : Any
        Classification model
        
    Returns:
    --------
    dict : Default hyperparameter grid
    """
    # Get model class name
    model_name = model.__class__.__name__.lower()
    
    # Define default grids for common classifiers
    if 'randomforest' in model_name:
        return {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }
    elif 'gradientboosting' in model_name:
        return {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.8, 1.0]
        }
    elif 'svc' in model_name or 'svm' in model_name:
        return {
            'C': [0.1, 1.0, 10.0, 100.0],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
            'degree': [2, 3, 4],
            'coef0': [0.0, 0.1, 0.5]
        }
    elif 'logistic' in model_name:
        return {
            'C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'penalty': ['l1', 'l2', 'elasticnet', None],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'max_iter': [100, 500, 1000],
            'l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0]
        }
    elif 'decisiontree' in model_name:
        return {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': [None, 'sqrt', 'log2']
        }
    elif 'kneighbors' in model_name:
        return {
            'n_neighbors': [3, 5, 7, 9, 11, 15],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'p': [1, 2]
        }
    elif 'adaboost' in model_name:
        return {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.5, 1.0],
            'algorithm': ['SAMME', 'SAMME.R']
        }
    elif 'naive_bayes' in model_name or 'naivebayes' in model_name:
        if 'gaussian' in model_name:
            return {
                'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
            }
        elif 'multinomial' in model_name:
            return {
                'alpha': [0.1, 0.5, 1.0, 2.0],
                'fit_prior': [True, False]
            }
        elif 'bernoulli' in model_name:
            return {
                'alpha': [0.1, 0.5, 1.0, 2.0],
                'binarize': [0.0, 0.5, None],
                'fit_prior': [True, False]
            }
        else:
            return {}
    elif 'xgb' in model_name:
        return {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
    elif 'lgbm' in model_name:
        return {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7, -1],
            'num_leaves': [31, 63, 127],
            'min_child_samples': [20, 50, 100],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
    elif 'catboost' in model_name:
        return {
            'iterations': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'depth': [4, 6, 8, 10],
            'l2_leaf_reg': [1, 3, 5, 7, 9],
            'border_count': [32, 64, 128]
        }
    else:
        # Generic classifier params
        # Generic classifier params
        return {
            'max_iter': [100, 500, 1000],
            'tol': [1e-4, 1e-3, 1e-2]
        }


def get_classification_scorer(scoring: str) -> Callable:
    """
    Get a scorer function for classification models.
    
    Parameters:
    -----------
    scoring : str
        Scoring metric: 'accuracy', 'f1', 'precision', 'recall', 'roc_auc'
        
    Returns:
    --------
    callable : Scorer function
    """
    from sklearn.metrics import make_scorer
    
    if scoring == 'accuracy':
        return make_scorer(accuracy_score)
    elif scoring == 'f1':
        return make_scorer(f1_score, average='weighted')
    elif scoring == 'precision':
        return make_scorer(precision_score, average='weighted')
    elif scoring == 'recall':
        return make_scorer(recall_score, average='weighted')
    elif scoring == 'roc_auc':
        return make_scorer(roc_auc_score, average='weighted', multi_class='ovr')
    else:
        raise ValueError(f"Unknown scoring metric: {scoring}")


def plot_classification_hyperparameter_importance(
    importance_scores: Dict[str, float],
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot hyperparameter importance for classification models.
    
    Parameters:
    -----------
    importance_scores : dict
        Dictionary mapping parameter names to importance scores
    figsize : tuple
        Figure size (width, height)
        
    Returns:
    --------
    matplotlib.Figure : Importance plot
    """
    # Sort parameters by importance
    sorted_params = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
    params = [p[0] for p in sorted_params]
    values = [p[1] for p in sorted_params]
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot horizontal bar chart
    y_pos = np.arange(len(params))
    ax.barh(y_pos, values, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(params)
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel('Importance Score')
    ax.set_title('Hyperparameter Importance')
    
    plt.tight_layout()
    return fig


def plot_classification_hyperparameter_effects(
    evaluations: List[Dict[str, Any]],
    param_name: str,
    scoring: str = 'accuracy',
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot effects of a hyperparameter on classification performance.
    
    Parameters:
    -----------
    evaluations : list of dict
        List of evaluation results
    param_name : str
        Name of the parameter to plot
    scoring : str
        Scoring metric used for evaluation
    figsize : tuple
        Figure size (width, height)
        
    Returns:
    --------
    matplotlib.Figure : Effects plot
    """
    # Extract parameter values and scores
    param_values = []
    scores = []
    
    for eval_data in evaluations:
        if param_name in eval_data['params']:
            param_values.append(eval_data['params'][param_name])
            scores.append(eval_data['score'])
    
    if not param_values:
        raise ValueError(f"Parameter {param_name} not found in evaluations")
        
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Check parameter type
    if isinstance(param_values[0], (int, float)):
        # Numerical parameter - scatter plot
        ax.scatter(param_values, scores, alpha=0.7)
        
        # Try to fit a smoothed curve
        try:
            import scipy.stats as stats
            
            # Sort points for smoother line
            sorted_indices = np.argsort(param_values)
            sorted_values = [param_values[i] for i in sorted_indices]
            sorted_scores = [scores[i] for i in sorted_indices]
            
            # Fit smoothing spline
            if len(sorted_values) > 3:
                from scipy.interpolate import make_interp_spline
                
                # Create smoothing spline
                spl = make_interp_spline(sorted_values, sorted_scores, k=min(3, len(sorted_values)-1))
                
                # Generate points for plotting
                x_smooth = np.linspace(min(sorted_values), max(sorted_values), 100)
                y_smooth = spl(x_smooth)
                
                # Plot smoothed curve
                ax.plot(x_smooth, y_smooth, 'r-', alpha=0.7)
        except (ImportError, Exception) as e:
            # Skip curve fitting if there's an error
            pass
            
        ax.set_xlabel(param_name)
        
    else:
        # Categorical parameter - box plot
        param_score_dict = {}
        
        for pv, s in zip(param_values, scores):
            pv_str = str(pv)
            if pv_str not in param_score_dict:
                param_score_dict[pv_str] = []
            param_score_dict[pv_str].append(s)
            
        # Create box plot data
        box_data = [param_score_dict[key] for key in param_score_dict.keys()]
        
        # Plot
        ax.boxplot(box_data)
        ax.set_xticklabels(list(param_score_dict.keys()))
        ax.set_xlabel(param_name)
    
    # Set labels and title
    ax.set_ylabel(f'{scoring.replace("_", " ").title()} Score')
    ax.set_title(f'Effect of {param_name} on Classification Performance')
    
    plt.tight_layout()
    return fig