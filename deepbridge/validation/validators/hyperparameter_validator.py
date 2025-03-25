"""
Hyperparameter validator for machine learning models.

This module provides a validator for optimizing and analyzing
hyperparameters of machine learning models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
import matplotlib.pyplot as plt
import os
import warnings

from ..core.base_validator import BaseValidator
from ..utils.model_inspection import get_model_type, is_classifier
from ..utils.validation_utils import save_validation_results


class HyperparameterValidator(BaseValidator):
    """
    Validator for optimizing and analyzing model hyperparameters.
    
    This class provides methods for hyperparameter tuning and analyzing
    the importance of different hyperparameters.
    """
    
    def __init__(
        self,
        model: Any,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        scoring: Optional[str] = None,
        n_iterations: int = 20,
        cv_folds: int = 5,
        random_state: Optional[int] = None,
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize the hyperparameter validator.
        
        Parameters:
        -----------
        model : Any
            Machine learning model to validate
        param_grid : dict or None
            Grid of hyperparameters to search
        scoring : str or None
            Scoring metric for hyperparameter optimization
        n_iterations : int
            Number of iterations for optimization
        cv_folds : int
            Number of cross-validation folds
        random_state : int or None
            Random seed for reproducibility
        verbose : bool
            Whether to print progress information
        **kwargs : dict
            Additional parameters
        """
        super().__init__(model=model, **kwargs)
        
        self.param_grid = param_grid
        self.n_iterations = n_iterations
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.verbose = verbose
        
        # Determine model type
        self.model_type = kwargs.get('model_type', get_model_type(model))
        
        # Set scoring based on model type if not provided
        if scoring is None:
            if self.model_type == 'classifier':
                self.scoring = 'accuracy'
            else:
                self.scoring = 'neg_mean_squared_error'
        else:
            self.scoring = scoring
            
        # Infer param_grid if not provided
        if self.param_grid is None:
            self.param_grid = self._infer_param_grid()
            
        # Initialize results
        self.results = {}
        
        # Initialize best model
        self.best_model = None
    
    def validate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Validate model hyperparameters.
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Features to validate with
        y : array-like or Series
            Target values
        **kwargs : dict
            Additional parameters
            
        Returns:
        --------
        dict : Validation results
        """
        # Override parameters if provided
        param_grid = kwargs.get('param_grid', self.param_grid)
        scoring = kwargs.get('scoring', self.scoring)
        n_iterations = kwargs.get('n_iterations', self.n_iterations)
        cv_folds = kwargs.get('cv_folds', self.cv_folds)
        
        # Start with clean results
        results = {
            'model_type': self.model_type,
            'scoring': scoring,
            'best_params': {},
            'best_score': 0.0,
            'param_importance': {},
            'optimization_history': []
        }
        
        # Check if scikit-learn is available
        try:
            import sklearn
            has_sklearn = True
        except ImportError:
            has_sklearn = False
            warnings.warn("scikit-learn is required for hyperparameter validation")
            
            # Skip validation
            results['error'] = "scikit-learn not available"
            self.results = results
            return results
            
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X
            
        if isinstance(y, pd.Series):
            y_values = y.values
        else:
            y_values = y
            
        # Determine optimization method based on param_grid size
        total_combinations = 1
        for param, values in param_grid.items():
            total_combinations *= len(values)
            
        if total_combinations <= n_iterations:
            # Use grid search if total combinations is manageable
            optimization_method = 'grid'
        else:
            # Use random search if too many combinations
            optimization_method = 'random'
            
        # Check if Bayesian optimization is available
        try:
            import skopt
            has_skopt = True
        except ImportError:
            has_skopt = False
            
        if has_skopt and total_combinations > n_iterations:
            # Use Bayesian optimization if available and beneficial
            optimization_method = 'bayesian'
            
        # Log optimization method
        if self.verbose:
            print(f"Using {optimization_method} search for hyperparameter optimization")
            
        # Perform hyperparameter optimization
        optimization_results = self._optimize_hyperparameters(
            X_values, y_values, param_grid, scoring, n_iterations, cv_folds, optimization_method
        )
        
        # Update results with optimization results
        results.update(optimization_results)
        
        # Calculate hyperparameter importance
        importance_results = self._calculate_importance(
            optimization_results.get('optimization_history', [])
        )
        
        results['param_importance'] = importance_results
        
        # Store results
        self.results = results
        
        return results
    
    def _infer_param_grid(self) -> Dict[str, List[Any]]:
        """
        Infer parameter grid based on model type.
        
        Returns:
        --------
        dict : Parameter grid
        """
        # Get model class name
        model_name = type(self.model).__name__.lower()
        
        # Define common parameter grids for different model types
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
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif 'svc' in model_name or 'svm' in model_name:
            return {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto', 0.1, 0.01]
            }
        elif 'logistic' in model_name:
            return {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
        elif 'knn' in model_name or 'neighbor' in model_name:
            return {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'p': [1, 2]
            }
        else:
            # Generic parameter grid
            if hasattr(self.model, 'get_params'):
                params = self.model.get_params()
                
                # Create a minimal grid based on common parameters
                grid = {}
                
                if 'n_estimators' in params:
                    grid['n_estimators'] = [50, 100, 200]
                    
                if 'max_depth' in params:
                    grid['max_depth'] = [3, 5, 7, None]
                    
                if 'learning_rate' in params:
                    grid['learning_rate'] = [0.01, 0.1, 0.2]
                    
                if 'C' in params:
                    grid['C'] = [0.1, 1.0, 10.0]
                    
                if len(grid) > 0:
                    return grid
                    
            # If no parameters found, return an empty grid
            warnings.warn("Could not infer parameter grid for this model. "
                        "Please provide a param_grid.")
            return {}
    
    def _optimize_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        param_grid: Dict[str, List[Any]],
        scoring: str,
        n_iterations: int,
        cv_folds: int,
        method: str
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters.
        
        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            Target values
        param_grid : dict
            Grid of hyperparameters to search
        scoring : str
            Scoring metric
        n_iterations : int
            Number of iterations
        cv_folds : int
            Number of cross-validation folds
        method : str
            Optimization method
            
        Returns:
        --------
        dict : Optimization results
        """
        # Initialize results
        results = {
            'best_params': {},
            'best_score': 0.0,
            'optimization_history': []
        }
        
        # Use different optimization methods
        if method == 'grid':
            # Grid search
            from sklearn.model_selection import GridSearchCV
            
            grid_search = GridSearchCV(
                self.model,
                param_grid,
                scoring=scoring,
                cv=cv_folds,
                n_jobs=-1,
                verbose=1 if self.verbose else 0,
                return_train_score=True
            )
            
            # Fit grid search
            grid_search.fit(X, y)
            
            # Store best results
            results['best_params'] = grid_search.best_params_
            results['best_score'] = float(grid_search.best_score_)
            
            # Store optimization history
            for i, params in enumerate(grid_search.cv_results_['params']):
                results['optimization_history'].append({
                    'iteration': i,
                    'params': params,
                    'score': float(grid_search.cv_results_['mean_test_score'][i]),
                    'std': float(grid_search.cv_results_['std_test_score'][i])
                })
                
            # Store best model
            self.best_model = grid_search.best_estimator_
            
        elif method == 'random':
            # Random search
            from sklearn.model_selection import RandomizedSearchCV
            
            random_search = RandomizedSearchCV(
                self.model,
                param_grid,
                n_iter=n_iterations,
                scoring=scoring,
                cv=cv_folds,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=1 if self.verbose else 0,
                return_train_score=True
            )
            
            # Fit random search
            random_search.fit(X, y)
            
            # Store best results
            results['best_params'] = random_search.best_params_
            results['best_score'] = float(random_search.best_score_)
            
            # Store optimization history
            for i, params in enumerate(random_search.cv_results_['params']):
                results['optimization_history'].append({
                    'iteration': i,
                    'params': params,
                    'score': float(random_search.cv_results_['mean_test_score'][i]),
                    'std': float(random_search.cv_results_['std_test_score'][i])
                })
                
            # Store best model
            self.best_model = random_search.best_estimator_
            
        elif method == 'bayesian':
            # Bayesian optimization with scikit-optimize
            try:
                from skopt import BayesSearchCV
                from skopt.space import Real, Integer, Categorical
                
                # Convert param_grid to skopt search space
                search_space = {}
                
                for param, values in param_grid.items():
                    if isinstance(values, list):
                        if all(isinstance(v, (int, float)) for v in values) and len(values) > 2:
                            # Numerical list with more than 2 values - consider as a categorical choice
                            search_space[param] = Categorical(values)
                        elif all(isinstance(v, str) for v in values):
                            # String list - categorical
                            search_space[param] = Categorical(values)
                        else:
                            # Mixed list - treat as categorical
                            search_space[param] = Categorical(values)
                    elif isinstance(values, tuple) and len(values) == 2:
                        # Range (min, max)
                        min_val, max_val = values
                        if all(isinstance(v, int) for v in values):
                            # Integer range
                            search_space[param] = Integer(min_val, max_val)
                        else:
                            # Float range
                            search_space[param] = Real(min_val, max_val)
                    else:
                        # Unknown type - skip
                        continue
                        
                # Create Bayesian search
                bayes_search = BayesSearchCV(
                    self.model,
                    search_space,
                    n_iter=n_iterations,
                    scoring=scoring,
                    cv=cv_folds,
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbose=1 if self.verbose else 0,
                    return_train_score=True
                )
                
                # Fit Bayesian search
                bayes_search.fit(X, y)
                
                # Store best results
                results['best_params'] = bayes_search.best_params_
                results['best_score'] = float(bayes_search.best_score_)
                
                # Store optimization history
                for i, params in enumerate(bayes_search.cv_results_['params']):
                    results['optimization_history'].append({
                        'iteration': i,
                        'params': params,
                        'score': float(bayes_search.cv_results_['mean_test_score'][i]),
                        'std': float(bayes_search.cv_results_['std_test_score'][i])
                    })
                    
                # Store best model
                self.best_model = bayes_search.best_estimator_
                
            except (ImportError, ValueError) as e:
                # Fall back to random search if Bayesian fails
                if self.verbose:
                    print(f"Error with Bayesian optimization: {str(e)}")
                    print("Falling back to random search")
                    
                # Use random search instead
                return self._optimize_hyperparameters(
                    X, y, param_grid, scoring, n_iterations, cv_folds, 'random'
                )
                
        else:
            raise ValueError(f"Unknown optimization method: {method}")
            
        return results
    
    def _calculate_importance(
        self,
        optimization_history: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate hyperparameter importance.
        
        Parameters:
        -----------
        optimization_history : list of dict
            Optimization history
            
        Returns:
        --------
        dict : Parameter importance
        """
        # Check if we have enough results
        if len(optimization_history) < 5:
            # Not enough data for importance calculation
            return {}
            
        # Collect all parameters and values
        params_data = {}
        scores = []
        
        for result in optimization_history:
            # Store score
            scores.append(result['score'])
            
            # Store parameter values
            for param, value in result['params'].items():
                if param not in params_data:
                    params_data[param] = []
                    
                params_data[param].append(value)
                
        # Calculate importance for each parameter
        importance = {}
        
        for param, values in params_data.items():
            # Skip if only one unique value
            if len(set(values)) <= 1:
                continue
                
            # Try to calculate correlation for numerical parameters
            try:
                # Convert values to numeric if possible
                numeric_values = []
                
                for value in values:
                    if value is None:
                        numeric_values.append(0)  # Replace None with 0
                    else:
                        numeric_values.append(float(value))
                        
                # Calculate correlation
                correlation = np.corrcoef(numeric_values, scores)[0, 1]
                
                # Use absolute correlation as importance
                importance[param] = float(abs(correlation))
                
            except (TypeError, ValueError):
                # For non-numeric parameters, use a different approach
                # Calculate score variance for each unique value
                unique_values = set(values)
                value_scores = {value: [] for value in unique_values}
                
                for i, value in enumerate(values):
                    value_scores[value].append(scores[i])
                    
                # Calculate mean score for each value
                mean_scores = {value: np.mean(scores_list) for value, scores_list in value_scores.items()}
                
                # Calculate variance of mean scores
                if len(mean_scores) > 1:
                    variance = np.var(list(mean_scores.values()))
                    
                    # Higher variance means more important
                    importance[param] = float(variance)
                    
        # Normalize importance scores
        if importance:
            max_importance = max(importance.values())
            
            if max_importance > 0:
                importance = {
                    param: value / max_importance
                    for param, value in importance.items()
                }
                
        return importance
    
    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """
        Generate predictions using the best model.
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Features
            
        Returns:
        --------
        numpy.ndarray : Predictions
        """
        if self.best_model is None:
            raise ValueError("No best model available. Run validate() first.")
            
        return self.best_model.predict(X)
    
    def plot_optimization_history(
        self,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot optimization history.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.Figure : Optimization history plot
        """
        if not self.results or 'optimization_history' not in self.results:
            raise ValueError("No optimization history available. Run validate() first.")
            
        # Get optimization history
        history = self.results['optimization_history']
        
        # Sort by iteration
        history = sorted(history, key=lambda x: x['iteration'])
        
        # Extract iterations and scores
        iterations = [result['iteration'] for result in history]
        scores = [result['score'] for result in history]
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot scores
        ax.plot(iterations, scores, 'o-', label='Score')
        
        # Plot best score
        best_score = self.results.get('best_score', max(scores))
        ax.axhline(y=best_score, color='r', linestyle='--', label=f'Best Score: {best_score:.4f}')
        
        # Set labels and title
        ax.set_xlabel('Iteration')
        ax.set_ylabel(f"Score ({self.scoring})")
        ax.set_title('Hyperparameter Optimization History')
        
        # Add legend
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_param_importance(
        self,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot parameter importance.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.Figure : Parameter importance plot
        """
        if not self.results or 'param_importance' not in self.results:
            raise ValueError("No parameter importance data available. Run validate() first.")
            
        # Get parameter importance
        importance = self.results['param_importance']
        
        if not importance:
            raise ValueError("No parameter importance data available")
            
        # Sort parameters by importance
        sorted_params = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        # Extract parameter names and importance values
        param_names = [param[0] for param in sorted_params]
        importance_values = [param[1] for param in sorted_params]
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create horizontal bar chart
        y_pos = range(len(param_names))
        ax.barh(y_pos, importance_values, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(param_names)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Importance')
        ax.set_title('Hyperparameter Importance')
        
        plt.tight_layout()
        return fig
    
    def plot_param_interactions(
        self,
        figsize: Tuple[int, int] = (12, 10)
    ) -> plt.Figure:
        """
        Plot parameter interactions.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.Figure : Parameter interaction plot
        """
        if not self.results or 'optimization_history' not in self.results:
            raise ValueError("No optimization history available. Run validate() first.")
            
        # Get optimization history
        history = self.results['optimization_history']
        
        # Get parameter importance
        importance = self.results.get('param_importance', {})
        
        if not importance:
            # Calculate importance if not available
            importance = self._calculate_importance(history)
            
        if not importance:
            raise ValueError("No parameter importance data available")
            
        # Get top 2 most important parameters
        sorted_params = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        if len(sorted_params) < 2:
            raise ValueError("Need at least 2 parameters for interaction plot")
            
        param1 = sorted_params[0][0]
        param2 = sorted_params[1][0]
        
        # Extract parameter values and scores
        param1_values = []
        param2_values = []
        scores = []
        
        for result in history:
            params = result['params']
            
            if param1 in params and param2 in params:
                # Try to convert values to numeric
                try:
                    value1 = float(params[param1]) if params[param1] is not None else 0
                    value2 = float(params[param2]) if params[param2] is not None else 0
                    
                    param1_values.append(value1)
                    param2_values.append(value2)
                    scores.append(result['score'])
                except (TypeError, ValueError):
                    # Skip non-numeric values
                    continue
                    
        # Check if we have enough numeric data
        if len(param1_values) < 5:
            raise ValueError("Not enough numeric data for interaction plot")
            
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create scatter plot with scores as colors
        scatter = ax.scatter(param1_values, param2_values, c=scores, cmap='viridis', 
                            alpha=0.8, edgecolors='k', s=100)
        
        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label(f"Score ({self.scoring})")
        
        # Set labels and title
        ax.set_xlabel(param1)
        ax.set_ylabel(param2)
        ax.set_title('Parameter Interaction Plot')
        
        plt.tight_layout()
        return fig
    
    def save_results(
        self,
        output_dir: str,
        prefix: str = 'hyperparameter',
        include_plots: bool = True
    ) -> Dict[str, str]:
        """
        Save validation results and plots.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save results in
        prefix : str
            Prefix for output files
        include_plots : bool
            Whether to save plots
            
        Returns:
        --------
        dict : Paths to saved files
        """
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Paths for saved files
        saved_files = {}
        
        # Save results as JSON
        results_path = os.path.join(output_dir, f"{prefix}_results.json")
        save_validation_results(self.results, results_path, 'json')
        saved_files['results_json'] = results_path
        
        # Save results as CSV
        csv_path = os.path.join(output_dir, f"{prefix}_results.csv")
        save_validation_results(self.results, csv_path, 'dataframe')
        saved_files['results_csv'] = csv_path
        
        # Save best parameters
        if 'best_params' in self.results and self.results['best_params']:
            params_path = os.path.join(output_dir, f"{prefix}_best_params.json")
            
            with open(params_path, 'w') as f:
                import json
                json.dump(self.results['best_params'], f, indent=2)
                
            saved_files['best_params'] = params_path
            
        # Save plots if requested
        if include_plots and self.results:
            # Save optimization history plot
            try:
                fig = self.plot_optimization_history()
                fig_path = os.path.join(output_dir, f"{prefix}_optimization_history.png")
                fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                saved_files['optimization_history_plot'] = fig_path
                plt.close(fig)
            except Exception as e:
                if self.verbose:
                    print(f"Error saving optimization history plot: {str(e)}")
                    
            # Save parameter importance plot
            try:
                fig = self.plot_param_importance()
                fig_path = os.path.join(output_dir, f"{prefix}_param_importance.png")
                fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                saved_files['param_importance_plot'] = fig_path
                plt.close(fig)
            except Exception as e:
                if self.verbose:
                    print(f"Error saving parameter importance plot: {str(e)}")
                    
            # Save parameter interaction plot
            try:
                fig = self.plot_param_interactions()
                fig_path = os.path.join(output_dir, f"{prefix}_param_interactions.png")
                fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                saved_files['param_interactions_plot'] = fig_path
                plt.close(fig)
            except Exception as e:
                if self.verbose:
                    print(f"Error saving parameter interaction plot: {str(e)}")
                    
        return saved_files