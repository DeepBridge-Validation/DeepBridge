"""
Efficient hyperparameter tuning strategies.

This module provides methods for efficient hyperparameter tuning,
including Bayesian optimization and multi-fidelity methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import ParameterGrid


class EfficientTuner:
    """
    Efficiently tune hyperparameters using advanced algorithms.
    
    This class provides methods for tuning hyperparameters using
    strategies more efficient than grid or random search.
    """
    
    def __init__(
        self,
        param_space: Dict,
        scoring_function: Callable,
        n_iterations: int = 20,
        method: str = 'bayesian',
        verbose: bool = False,
        random_state: Optional[int] = None
    ):
        """
        Initialize the hyperparameter tuner.
        
        Parameters:
        -----------
        param_space : dict
            Dictionary defining the hyperparameter search space
        scoring_function : callable
            Function that takes a set of hyperparameters and returns a score
        n_iterations : int
            Number of iterations/evaluations to perform
        method : str
            Tuning method to use:
            - 'bayesian': Bayesian optimization
            - 'successive_halving': Successive halving (multi-fidelity)
        verbose : bool
            Whether to print progress information
        random_state : int or None
            Random seed for reproducibility
        """
        self.param_space = param_space
        self.scoring_function = scoring_function
        self.n_iterations = n_iterations
        self.method = method
        self.verbose = verbose
        self.random_state = random_state
        self.results = None
        self.best_params = None
        self.best_score = None
        
        # Seeds for reproducibility
        np.random.seed(random_state)
    
    def tune(self) -> Dict:
        """
        Perform hyperparameter tuning.
        
        Returns:
        --------
        dict : Dictionary with best parameters and results
        """
        if self.method == 'bayesian':
            results = bayesian_optimization_tuner(
                self.param_space,
                self.scoring_function,
                n_iterations=self.n_iterations,
                verbose=self.verbose,
                random_state=self.random_state
            )
        elif self.method == 'successive_halving':
            results = successive_halving_tuner(
                self.param_space,
                self.scoring_function,
                n_iterations=self.n_iterations,
                verbose=self.verbose,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown tuning method: {self.method}")
        
        self.results = results['evaluations']
        self.best_params = results['best_params']
        self.best_score = results['best_score']
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'evaluations': self.results
        }
    
    def plot_optimization_history(self, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot the optimization history.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
            
        Returns:
        --------
        matplotlib.Figure : Optimization history plot
        """
        if self.results is None:
            raise ValueError("No tuning results available. Call tune() first.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Convert results to DataFrame if not already
        if not isinstance(self.results, pd.DataFrame):
            results_df = pd.DataFrame(self.results)
        else:
            results_df = self.results
        
        # Sort by iteration if available
        if 'iteration' in results_df.columns:
            results_df = results_df.sort_values('iteration')
        
        # Plot optimization progress
        ax.plot(results_df.index, results_df['score'], 'o-', label='Score')
        
        # Plot best score found so far
        best_so_far = results_df['score'].cummax() if is_metric_higher_better('score') else results_df['score'].cummin()
        ax.plot(results_df.index, best_so_far, 'r--', label='Best So Far')
        
        ax.set_xlabel('Evaluation')
        ax.set_ylabel('Score')
        ax.set_title('Hyperparameter Optimization History')
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_parallel_coordinates(self, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Create a parallel coordinates plot of hyperparameters and scores.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
            
        Returns:
        --------
        matplotlib.Figure : Parallel coordinates plot
        """
        if self.results is None:
            raise ValueError("No tuning results available. Call tune() first.")
        
        try:
            import pandas as pd
            from pandas.plotting import parallel_coordinates
        except ImportError:
            raise ImportError("pandas is required for parallel coordinates plots")
        
        # Convert results to DataFrame if not already
        if not isinstance(self.results, pd.DataFrame):
            results_df = pd.DataFrame(self.results)
        else:
            results_df = self.results.copy()
        
        # Create a class column based on score quantiles
        results_df['performance_class'] = pd.qcut(
            results_df['score'], 
            q=min(4, len(results_df['score'].unique())),
            labels=['poor', 'fair', 'good', 'excellent'][:min(4, len(results_df['score'].unique()))]
        )
        
        # Select columns for plot: all parameters + class
        param_cols = [col for col in results_df.columns 
                      if col not in ['score', 'performance_class', 'iteration']]
        
        fig, ax = plt.subplots(figsize=figsize)
        parallel_coordinates(
            results_df[param_cols + ['performance_class']], 
            'performance_class',
            ax=ax
        )
        
        ax.set_title('Parallel Coordinates Plot of Hyperparameters')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig


def bayesian_optimization_tuner(
    param_space: Dict,
    scoring_function: Callable,
    n_iterations: int = 20,
    verbose: bool = False,
    random_state: Optional[int] = None
) -> Dict:
    """
    Tune hyperparameters using Bayesian optimization.
    
    Parameters:
    -----------
    param_space : dict
        Dictionary defining the hyperparameter search space
    scoring_function : callable
        Function that takes a set of hyperparameters and returns a score
    n_iterations : int
        Number of iterations to perform
    verbose : bool
        Whether to print progress information
    random_state : int or None
        Random seed for reproducibility
        
    Returns:
    --------
    dict : Dictionary with best parameters and results
    """
    try:
        # Try to import skopt for Bayesian optimization
        from skopt import gp_minimize
        from skopt.space import Real, Integer, Categorical
    except ImportError:
        warnings.warn("scikit-optimize not installed. Using simple random search instead.")
        return _random_search_fallback(
            param_space, scoring_function, n_iterations, verbose, random_state
        )
    
    # Define the search space for skopt
    dimensions = []
    dimension_names = []
    
    for param_name, param_range in param_space.items():
        if isinstance(param_range, list):
            # Categorical parameter
            dimensions.append(Categorical(param_range, name=param_name))
        elif isinstance(param_range, tuple) and len(param_range) == 2:
            # Numerical parameter (min, max)
            min_val, max_val = param_range
            if all(isinstance(x, int) for x in param_range):
                # Integer parameter
                dimensions.append(Integer(min_val, max_val, name=param_name))
            else:
                # Float parameter
                dimensions.append(Real(min_val, max_val, name=param_name))
        else:
            raise ValueError(f"Unsupported parameter space for {param_name}: {param_range}")
            
        dimension_names.append(param_name)
    
    # Define objective function for gp_minimize
    def objective(x):
        # Convert list of parameter values to dictionary
        params = {dim_name: x[i] for i, dim_name in enumerate(dimension_names)}
        
        try:
            # Call the scoring function with the parameters
            score = scoring_function(params)
            
            # Bayesian optimization minimizes the objective,
            # so negate the score if higher is better
            return -score  # Assuming higher scores are better
        except Exception as e:
            if verbose:
                print(f"Error evaluating parameters {params}: {str(e)}")
            # Return a very poor score in case of error
            return float('inf')
    
    # Run Bayesian optimization
    result = gp_minimize(
        objective,
        dimensions,
        n_calls=n_iterations,
        random_state=random_state,
        verbose=verbose
    )
    
    # Collect results
    best_params = {name: result.x[i] for i, name in enumerate(dimension_names)}
    best_score = -result.fun  # Negate back to get the original score
    
    evaluations = []
    for i, (x, y) in enumerate(zip(result.x_iters, result.func_vals)):
        params = {name: x[i] for i, name in enumerate(dimension_names)}
        evaluations.append({
            'iteration': i,
            **params,
            'score': -y  # Negate to get the original score
        })
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'evaluations': evaluations
    }


def successive_halving_tuner(
    param_space: Dict,
    scoring_function: Callable,
    n_iterations: int = 20,
    verbose: bool = False,
    random_state: Optional[int] = None
) -> Dict:
    """
    Tune hyperparameters using successive halving (multi-fidelity optimization).
    
    Parameters:
    -----------
    param_space : dict
        Dictionary defining the hyperparameter search space
    scoring_function : callable
        Function that takes a set of hyperparameters and returns a score
    n_iterations : int
        Total computation budget (number of model evaluations)
    verbose : bool
        Whether to print progress information
    random_state : int or None
        Random seed for reproducibility
        
    Returns:
    --------
    dict : Dictionary with best parameters and results
    """
    try:
        import math
    except ImportError:
        # This should never happen as math is a built-in module
        return _random_search_fallback(
            param_space, scoring_function, n_iterations, verbose, random_state
        )
    
    # Generate configurations
    if all(isinstance(values, list) for values in param_space.values()):
        # If all parameter spaces are lists, use ParameterGrid
        all_configs = list(ParameterGrid(param_space))
        
        # If more configurations than budget, randomly sample
        np.random.seed(random_state)
        if len(all_configs) > n_iterations:
            indices = np.random.choice(len(all_configs), n_iterations, replace=False)
            configurations = [all_configs[i] for i in indices]
        else:
            configurations = all_configs
    else:
        # For continuous parameters, sample randomly
        np.random.seed(random_state)
        configurations = []
        for _ in range(n_iterations):
            config = {}
            for param, space in param_space.items():
                if isinstance(space, list):
                    # Categorical parameter
                    config[param] = np.random.choice(space)
                elif isinstance(space, tuple) and len(space) == 2:
                    # Numerical parameter (min, max)
                    min_val, max_val = space
                    if all(isinstance(x, int) for x in space):
                        # Integer parameter
                        config[param] = np.random.randint(min_val, max_val + 1)
                    else:
                        # Float parameter
                        config[param] = np.random.uniform(min_val, max_val)
                else:
                    raise ValueError(f"Unsupported parameter space for {param}: {space}")
            configurations.append(config)
    
    # Successive Halving algorithm
    n_configs = len(configurations)
    s_max = int(math.log2(n_configs))
    
    # Track all evaluations
    all_evaluations = []
    iteration = 0
    
    # Initialize with all configurations
    remaining_configs = configurations.copy()
    
    for s in range(s_max, -1, -1):
        # Determine number of configurations to keep after this round
        n_i = math.ceil(n_configs / (2 ** (s_max - s)))
        
        # Determine resource allocation (e.g., epochs or data fraction)
        r_i = n_iterations / (s_max + 1) / n_i
        
        # Evaluate all configurations with the allocated resources
        scores = []
        
        for config in remaining_configs:
            try:
                # We're using the same scoring function, but in a real implementation,
                # you'd pass r_i to the scoring function to control resources used
                score = scoring_function(config)
                scores.append(score)
                
                # Record the evaluation
                all_evaluations.append({
                    'iteration': iteration,
                    **config,
                    'score': score,
                    'stage': s
                })
                
                iteration += 1
                
                if verbose:
                    print(f"Stage {s}, Config: {config}, Score: {score}")
            except Exception as e:
                if verbose:
                    print(f"Error evaluating configuration {config}: {str(e)}")
                scores.append(float('-inf'))  # Assuming higher is better
        
        # Keep the top half of configurations
        indices = np.argsort(scores)[::-1][:n_i]  # Assuming higher is better
        remaining_configs = [remaining_configs[i] for i in indices]
    
    # Best configuration is the last one remaining
    best_config = remaining_configs[0]
    best_score = scoring_function(best_config)
    
    return {
        'best_params': best_config,
        'best_score': best_score,
        'evaluations': all_evaluations
    }


def _random_search_fallback(
    param_space: Dict,
    scoring_function: Callable,
    n_iterations: int = 20,
    verbose: bool = False,
    random_state: Optional[int] = None
) -> Dict:
    """
    Fallback to random search when required libraries are not available.
    
    Parameters:
    -----------
    param_space : dict
        Dictionary defining the hyperparameter search space
    scoring_function : callable
        Function that takes a set of hyperparameters and returns a score
    n_iterations : int
        Number of iterations to perform
    verbose : bool
        Whether to print progress information
    random_state : int or None
        Random seed for reproducibility
        
    Returns:
    --------
    dict : Dictionary with best parameters and results
    """
    np.random.seed(random_state)
    
    evaluations = []
    best_score = float('-inf')  # Assuming higher is better
    best_params = None
    
    for i in range(n_iterations):
        # Sample random parameters
        params = {}
        for param, space in param_space.items():
            if isinstance(space, list):
                # Categorical parameter
                params[param] = np.random.choice(space)
            elif isinstance(space, tuple) and len(space) == 2:
                # Numerical parameter (min, max)
                min_val, max_val = space
                if all(isinstance(x, int) for x in space):
                    # Integer parameter
                    params[param] = np.random.randint(min_val, max_val + 1)
                else:
                    # Float parameter
                    params[param] = np.random.uniform(min_val, max_val)
            else:
                raise ValueError(f"Unsupported parameter space for {param}: {space}")
        
        try:
            # Evaluate the parameters
            score = scoring_function(params)
            
            evaluations.append({
                'iteration': i,
                **params,
                'score': score
            })
            
            if score > best_score:  # Assuming higher is better
                best_score = score
                best_params = params.copy()
            
            if verbose:
                print(f"Iteration {i}: Params = {params}, Score = {score}")
                
        except Exception as e:
            if verbose:
                print(f"Error evaluating parameters {params}: {str(e)}")
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'evaluations': evaluations
    }


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
    # (accuracy, F1, AUC, precision, recall
    return True