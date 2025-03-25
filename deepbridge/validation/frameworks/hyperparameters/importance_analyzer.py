"""
Hyperparameter importance analysis tools.

This module provides methods for analyzing the importance and influence
of hyperparameters on model performance using various algorithms.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
import warnings


class HyperparameterImportance:
    """
    Analyze the importance of hyperparameters for a machine learning model.
    
    This class provides methods to quantify the impact of different
    hyperparameters on model performance using various techniques.
    """
    
    def __init__(
        self,
        param_space: Dict,
        scoring_function: Callable,
        n_samples: int = 50,
        random_state: Optional[int] = None
    ):
        """
        Initialize the hyperparameter importance analyzer.
        
        Parameters:
        -----------
        param_space : dict
            Dictionary defining the hyperparameter search space
        scoring_function : callable
            Function that takes a set of hyperparameters and returns a score
        n_samples : int
            Number of samples to use for importance estimation
        random_state : int or None
            Random seed for reproducibility
        """
        self.param_space = param_space
        self.scoring_function = scoring_function
        self.n_samples = n_samples
        self.random_state = random_state
        self.evaluation_results = None
        self.importance_scores = None
        
        # Seeds for reproducibility
        np.random.seed(random_state)
    
    def evaluate_configurations(self, configurations: Optional[List[Dict]] = None) -> pd.DataFrame:
        """
        Evaluate a set of hyperparameter configurations.
        
        Parameters:
        -----------
        configurations : list of dict or None
            List of hyperparameter configurations to evaluate.
            If None, generates random configurations from the parameter space.
            
        Returns:
        --------
        pd.DataFrame : Evaluation results with configurations and scores
        """
        if configurations is None:
            # Generate configurations from parameter space
            if all(isinstance(values, list) for values in self.param_space.values()):
                # If all parameter spaces are lists, use ParameterGrid
                all_configs = list(ParameterGrid(self.param_space))
                
                # Subsample if there are too many configurations
                if len(all_configs) > self.n_samples:
                    indices = np.random.choice(
                        len(all_configs), self.n_samples, replace=False
                    )
                    configurations = [all_configs[i] for i in indices]
                else:
                    configurations = all_configs
            else:
                # For continuous parameters, sample randomly
                configurations = []
                for _ in range(self.n_samples):
                    config = {}
                    for param, space in self.param_space.items():
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
        
        # Evaluate each configuration
        results = []
        for config in configurations:
            try:
                score = self.scoring_function(config)
                results.append({**config, 'score': score})
            except Exception as e:
                warnings.warn(f"Error evaluating configuration {config}: {str(e)}")
        
        # Convert to DataFrame
        self.evaluation_results = pd.DataFrame(results)
        return self.evaluation_results
    
    def compute_importance(self, method: str = 'permutation') -> Dict[str, float]:
        """
        Compute the importance of each hyperparameter.
        
        Parameters:
        -----------
        method : str
            Method to use for importance calculation:
            - 'permutation': Permutation-based importance
            - 'fanova': Functional ANOVA-based importance
            
        Returns:
        --------
        dict : Dictionary mapping hyperparameter names to importance scores
        """
        if self.evaluation_results is None:
            self.evaluate_configurations()
        
        if method == 'permutation':
            self.importance_scores = compute_importance_permutation(
                self.evaluation_results, 
                list(self.param_space.keys())
            )
        elif method == 'fanova':
            self.importance_scores = compute_importance_fanova(
                self.evaluation_results, 
                list(self.param_space.keys())
            )
        else:
            raise ValueError(f"Unknown importance method: {method}")
        
        return self.importance_scores
    
    def plot_importance(self, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot the importance scores for each hyperparameter.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
            
        Returns:
        --------
        matplotlib.Figure : Importance plot
        """
        if self.importance_scores is None:
            self.compute_importance()
        
        # Sort parameters by importance
        sorted_params = sorted(
            self.importance_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        params = [p[0] for p in sorted_params]
        scores = [p[1] for p in sorted_params]
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        y_pos = np.arange(len(params))
        
        ax.barh(y_pos, scores, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(params)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Importance Score')
        ax.set_title('Hyperparameter Importance')
        
        plt.tight_layout()
        return fig
    
    def plot_parameter_influence(
        self, 
        param_name: str, 
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot the influence of a specific parameter on the score.
        
        Parameters:
        -----------
        param_name : str
            Name of the parameter to analyze
        figsize : tuple
            Figure size (width, height)
            
        Returns:
        --------
        matplotlib.Figure : Parameter influence plot
        """
        if self.evaluation_results is None:
            self.evaluate_configurations()
        
        if param_name not in self.evaluation_results.columns:
            raise ValueError(f"Parameter {param_name} not found in evaluation results")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Check if parameter is categorical or numeric
        param_values = self.evaluation_results[param_name].values
        unique_values = sorted(self.evaluation_results[param_name].unique())
        
        if len(unique_values) < 10 or isinstance(param_values[0], (str, bool)):
            # Categorical parameter - use box plot
            self.evaluation_results.boxplot(
                column='score', 
                by=param_name, 
                ax=ax
            )
            ax.set_title(f'Impact of {param_name} on Score')
            ax.set_ylabel('Score')
        else:
            # Numerical parameter - use scatter plot
            ax.scatter(
                self.evaluation_results[param_name],
                self.evaluation_results['score'],
                alpha=0.6
            )
            ax.set_xlabel(param_name)
            ax.set_ylabel('Score')
            ax.set_title(f'Impact of {param_name} on Score')
            
            # Add trend line
            try:
                from scipy import stats
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    self.evaluation_results[param_name],
                    self.evaluation_results['score']
                )
                x = np.array([min(param_values), max(param_values)])
                y = intercept + slope * x
                ax.plot(x, y, 'r--', 
                        label=f'Trend: y={slope:.4f}x+{intercept:.4f}, R²={r_value**2:.4f}')
                ax.legend()
            except Exception:
                # If trend line fitting fails, just skip it
                pass
        
        plt.tight_layout()
        return fig


def compute_importance_permutation(results_df: pd.DataFrame, param_names: List[str]) -> Dict[str, float]:
    """
    Compute hyperparameter importance using a permutation-based approach.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame containing hyperparameter configurations and scores
    param_names : list of str
        List of hyperparameter names to analyze
        
    Returns:
    --------
    dict : Dictionary mapping hyperparameter names to importance scores
    """
    base_score_variance = results_df['score'].var()
    importance_scores = {}
    
    for param in param_names:
        if param not in results_df.columns or param == 'score':
            continue
            
        # Create a copy of the results with permuted parameter values
        permuted_df = results_df.copy()
        permuted_df[param] = np.random.permutation(permuted_df[param].values)
        
        # Calculate how much variance is explained by the parameter
        permuted_variance = permuted_df['score'].var()
        
        # Importance is reduction in variance when parameter is permuted
        # Normalize by the original variance
        if base_score_variance > 0:
            importance = max(0, (base_score_variance - permuted_variance) / base_score_variance)
        else:
            importance = 0
            
        importance_scores[param] = importance
    
    # Normalize importance scores to sum to 1
    total_importance = sum(importance_scores.values())
    if total_importance > 0:
        importance_scores = {k: v / total_importance for k, v in importance_scores.items()}
    
    return importance_scores


def compute_importance_fanova(results_df: pd.DataFrame, param_names: List[str]) -> Dict[str, float]:
    """
    Compute hyperparameter importance using functional ANOVA.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame containing hyperparameter configurations and scores
    param_names : list of str
        List of hyperparameter names to analyze
        
    Returns:
    --------
    dict : Dictionary mapping hyperparameter names to importance scores
    """
    try:
        # Try to import fANOVA
        from fanova import fANOVA
        from fanova.visualizer import Visualizer
    except ImportError:
        warnings.warn("fANOVA not installed. Using permutation importance instead.")
        return compute_importance_permutation(results_df, param_names)
    
    # Prepare data for fANOVA
    X = results_df[param_names].values
    y = results_df['score'].values
    
    # Create parameter space for fANOVA
    param_space = {}
    for i, param in enumerate(param_names):
        unique_values = sorted(results_df[param].unique())
        param_space[i] = unique_values
    
    # Run fANOVA
    try:
        f = fANOVA(X, y, param_space)
        importance_scores = {}
        
        for i, param in enumerate(param_names):
            importance = f.quantify_importance((i,))[(i,)]['individual importance']
            importance_scores[param] = importance
    except Exception as e:
        warnings.warn(f"Error computing fANOVA: {str(e)}. Using permutation importance instead.")
        return compute_importance_permutation(results_df, param_names)
    
    return importance_scores