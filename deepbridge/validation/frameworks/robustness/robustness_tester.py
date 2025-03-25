"""
Core robustness testing functionality.

This module provides the base classes and functions for testing
model robustness against various types of perturbations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score
import warnings
from collections import defaultdict


class RobustnessTester:
    """
    Test model robustness against various perturbations.
    
    This class provides methods for evaluating how well a model
    maintains its performance when input data is perturbed.
    """
    
    def __init__(
        self,
        model: Any,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        feature_names: Optional[List[str]] = None,
        problem_type: str = 'auto',
        metrics: Optional[Dict[str, Callable]] = None,
        verbose: bool = False
    ):
        """
        Initialize the robustness tester.
        
        Parameters:
        -----------
        model : Any
            Trained machine learning model with predict and/or predict_proba methods
        X : array-like or DataFrame
            Input features
        y : array-like or Series
            Target values
        feature_names : list of str or None
            Names of features (required if X is not a DataFrame)
        problem_type : str
            Type of problem: 'classification', 'regression', or 'auto' (detect)
        metrics : dict or None
            Dictionary mapping metric names to metric functions
        verbose : bool
            Whether to print progress information
        """
        self.model = model
        
        # Convert X to DataFrame if it's a numpy array
        if isinstance(X, np.ndarray):
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            self.X = pd.DataFrame(X, columns=feature_names)
        else:
            self.X = X
            
        # Store feature names
        self.feature_names = list(self.X.columns)
        
        # Convert y to numpy array if it's a Series
        if isinstance(y, pd.Series):
            self.y = y.values
        else:
            self.y = y
            
        # Determine problem type if auto
        if problem_type == 'auto':
            n_unique = len(np.unique(self.y))
            self.problem_type = 'classification' if n_unique < 10 else 'regression'
        else:
            self.problem_type = problem_type
            
        # Set up metrics based on problem type
        if metrics is None:
            if self.problem_type == 'classification':
                self.metrics = {
                    'accuracy': accuracy_score,
                    'f1': lambda y_true, y_pred: f1_score(
                        y_true, y_pred, average='weighted', zero_division=0
                    )
                }
            else:  # regression
                self.metrics = {
                    'mse': mean_squared_error,
                    'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))
                }
        else:
            self.metrics = metrics
            
        self.verbose = verbose
        self.results = {}
        
        # Sanity check model
        self._check_model()
        
        # Get baseline performance
        self.baseline_performance = self._evaluate_model(self.X, self.y)
        if self.verbose:
            print(f"Baseline performance: {self.baseline_performance}")
    
    def _check_model(self):
        """
        Check that the model has the required prediction methods.
        """
        if self.problem_type == 'classification':
            if not hasattr(self.model, 'predict'):
                raise ValueError("Model must have a predict method")
            
            # Check if model has predict_proba for classification
            self.has_predict_proba = hasattr(self.model, 'predict_proba')
        else:
            if not hasattr(self.model, 'predict'):
                raise ValueError("Model must have a predict method")
    
    def _evaluate_model(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance on a dataset.
        
        Parameters:
        -----------
        X : DataFrame
            Input features
        y : array-like
            Target values
            
        Returns:
        --------
        dict : Dictionary of metric name to score
        """
        try:
            y_pred = self.model.predict(X)
            
            # Calculate metrics
            results = {}
            for metric_name, metric_func in self.metrics.items():
                results[metric_name] = metric_func(y, y_pred)
                
            return results
        except Exception as e:
            warnings.warn(f"Error evaluating model: {str(e)}")
            return {metric: np.nan for metric in self.metrics}
    
    def test_feature_perturbation(
        self,
        feature_name: str,
        perturbation_type: str = 'noise',
        perturbation_level: Union[float, List[float]] = 0.1,
        n_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Test model robustness against perturbation of a single feature.
        
        Parameters:
        -----------
        feature_name : str
            Name of the feature to perturb
        perturbation_type : str
            Type of perturbation to apply:
            - 'noise': Add Gaussian noise
            - 'flip': Randomly flip binary values
            - 'zero': Set values to zero
            - 'quantile': Move values to their quantiles
        perturbation_level : float or list of floats
            Level of perturbation to apply
        n_samples : int or None
            Number of samples to use (None uses all)
            
        Returns:
        --------
        dict : Dictionary with perturbation results
        """
        if feature_name not in self.feature_names:
            raise ValueError(f"Feature {feature_name} not found")
            
        # Convert perturbation_level to list if it's a single value
        if isinstance(perturbation_level, (int, float)):
            perturbation_levels = [perturbation_level]
        else:
            perturbation_levels = perturbation_level
            
        # Subsample if needed
        if n_samples is not None and n_samples < len(self.X):
            idx = np.random.choice(len(self.X), n_samples, replace=False)
            X_test = self.X.iloc[idx].copy()
            y_test = self.y[idx].copy()
        else:
            X_test = self.X.copy()
            y_test = self.y.copy()
            
        # Get feature values
        feature_values = X_test[feature_name].values
        
        results = {
            'feature': feature_name,
            'perturbation_type': perturbation_type,
            'perturbation_levels': perturbation_levels,
            'performance': []
        }
        
        for level in perturbation_levels:
            # Apply perturbation
            X_perturbed = X_test.copy()
            
            if perturbation_type == 'noise':
                # Add Gaussian noise
                noise = np.random.normal(0, level * np.std(feature_values), size=len(feature_values))
                X_perturbed[feature_name] = feature_values + noise
            elif perturbation_type == 'flip':
                # Flip binary values
                if set(np.unique(feature_values)) <= {0, 1}:
                    # For binary features
                    mask = np.random.random(len(feature_values)) < level
                    X_perturbed.loc[mask, feature_name] = 1 - X_perturbed.loc[mask, feature_name]
                else:
                    warnings.warn(f"Flip perturbation only works for binary features. Feature {feature_name} is not binary.")
            elif perturbation_type == 'zero':
                # Set values to zero
                mask = np.random.random(len(feature_values)) < level
                X_perturbed.loc[mask, feature_name] = 0
            elif perturbation_type == 'quantile':
                # Move values to their quantiles
                quantiles = np.quantile(feature_values, [level, 1 - level])
                mask_lower = np.random.random(len(feature_values)) < 0.5
                mask_upper = ~mask_lower
                
                X_perturbed.loc[mask_lower, feature_name] = quantiles[0]
                X_perturbed.loc[mask_upper, feature_name] = quantiles[1]
            else:
                raise ValueError(f"Unknown perturbation type: {perturbation_type}")
                
            # Evaluate model on perturbed data
            performance = self._evaluate_model(X_perturbed, y_test)
            
            # Calculate relative change in performance
            relative_change = {}
            for metric, value in performance.items():
                baseline = self.baseline_performance[metric]
                if baseline != 0:
                    relative_change[metric] = (value - baseline) / abs(baseline)
                else:
                    relative_change[metric] = float('inf') if value > 0 else float('-inf')
            
            results['performance'].append({
                'level': level,
                'metrics': performance,
                'relative_change': relative_change
            })
            
            if self.verbose:
                print(f"Perturbation level {level}: {performance}")
        
        # Store results
        result_key = f"feature_perturbation_{feature_name}_{perturbation_type}"
        self.results[result_key] = results
        
        return results
    
    def test_all_features(
        self,
        perturbation_type: str = 'noise',
        perturbation_level: float = 0.1,
        n_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Test model robustness against perturbation of all features.
        
        Parameters:
        -----------
        perturbation_type : str
            Type of perturbation to apply
        perturbation_level : float
            Level of perturbation to apply
        n_samples : int or None
            Number of samples to use (None uses all)
            
        Returns:
        --------
        dict : Dictionary with perturbation results for all features
        """
        if self.verbose:
            print(f"Testing robustness for all features...")
            
        all_results = {}
        for feature in self.feature_names:
            result = self.test_feature_perturbation(
                feature,
                perturbation_type=perturbation_type,
                perturbation_level=perturbation_level,
                n_samples=n_samples
            )
            all_results[feature] = result
            
        # Calculate feature importance based on robustness
        feature_importance = self._calculate_robustness_importance(all_results)
        
        # Store results
        result_key = f"all_features_{perturbation_type}_{perturbation_level}"
        self.results[result_key] = {
            'individual_results': all_results,
            'feature_importance': feature_importance
        }
        
        return self.results[result_key]
    
    def _calculate_robustness_importance(self, all_results: Dict[str, Dict]) -> Dict[str, float]:
        """
        Calculate feature importance based on robustness results.
        
        Parameters:
        -----------
        all_results : dict
            Dictionary with perturbation results for all features
            
        Returns:
        --------
        dict : Dictionary mapping feature names to importance scores
        """
        importance = {}
        
        # Choose a representative metric based on problem type
        if self.problem_type == 'classification':
            metric = 'accuracy' if 'accuracy' in self.baseline_performance else list(self.baseline_performance.keys())[0]
        else:
            metric = 'mse' if 'mse' in self.baseline_performance else list(self.baseline_performance.keys())[0]
            
        for feature, result in all_results.items():
            # Get the performance change for the feature
            if not result['performance']:
                importance[feature] = 0.0
                continue
                
            # Use the first perturbation level's result
            performance = result['performance'][0]
            relative_change = performance['relative_change'][metric]
            
            # For metrics where higher is better (e.g., accuracy),
            # a negative change means the feature is important
            # For metrics where lower is better (e.g., mse),
            # a positive change means the feature is important
            if metric in ['mse', 'rmse']:
                importance[feature] = relative_change  # Higher means more important
            else:
                importance[feature] = -relative_change  # Lower means more important
        
        # Normalize importance values to [0, 1]
        if importance:
            max_importance = max(importance.values())
            min_importance = min(importance.values())
            
            if max_importance > min_importance:
                for feature in importance:
                    importance[feature] = (importance[feature] - min_importance) / (max_importance - min_importance)
        
        return importance
    
    def plot_feature_perturbation(
        self,
        feature_name: str,
        perturbation_type: str = 'noise',
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot the effect of perturbing a feature on model performance.
        
        Parameters:
        -----------
        feature_name : str
            Name of the feature to plot
        perturbation_type : str
            Type of perturbation to plot
        figsize : tuple
            Figure size (width, height)
            
        Returns:
        --------
        matplotlib.Figure : Performance plot
        """
        result_key = f"feature_perturbation_{feature_name}_{perturbation_type}"
        
        if result_key not in self.results:
            raise ValueError(f"No results found for feature {feature_name} with perturbation {perturbation_type}")
            
        results = self.results[result_key]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract perturbation levels and performance values
        levels = [p['level'] for p in results['performance']]
        metrics = list(self.baseline_performance.keys())
        
        for metric in metrics:
            baseline = self.baseline_performance[metric]
            values = [p['metrics'][metric] for p in results['performance']]
            
            # Plot relative change instead of absolute values
            relative_values = [(v - baseline) / abs(baseline) if baseline != 0 else 0 
                             for v in values]
            
            ax.plot(levels, relative_values, 'o-', label=f"{metric} (rel. change)")
        
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel(f"Perturbation Level ({perturbation_type})")
        ax.set_ylabel("Relative Performance Change")
        ax.set_title(f"Effect of Perturbing {feature_name}")
        ax.legend()
        
        return fig
    
    def plot_feature_importance(self, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot feature importance based on robustness.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
            
        Returns:
        --------
        matplotlib.Figure : Feature importance plot
        """
        # Find the most recent all-features result
        all_features_keys = [k for k in self.results.keys() if k.startswith('all_features_')]
        
        if not all_features_keys:
            raise ValueError("No all-features robustness test results found")
            
        result_key = all_features_keys[-1]
        results = self.results[result_key]
        
        if 'feature_importance' not in results:
            raise ValueError("No feature importance found in results")
            
        importance = results['feature_importance']
        
        # Sort features by importance
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        features = [f[0] for f in sorted_features]
        values = [f[1] for f in sorted_features]
        
        fig, ax = plt.subplots(figsize=figsize)
        y_pos = np.arange(len(features))
        
        ax.barh(y_pos, values, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Robustness Importance')
        ax.set_title('Feature Importance Based on Robustness')
        
        return fig


def run_robustness_test_suite(
    model: Any,
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    feature_names: Optional[List[str]] = None,
    problem_type: str = 'auto',
    perturbation_types: Optional[List[str]] = None,
    perturbation_levels: Optional[List[float]] = None,
    metrics: Optional[Dict[str, Callable]] = None,
    n_samples: Optional[int] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run a comprehensive robustness test suite on a model.
    
    Parameters:
    -----------
    model : Any
        Trained machine learning model
    X : array-like or DataFrame
        Input features
    y : array-like or Series
        Target values
    feature_names : list of str or None
        Names of features (required if X is not a DataFrame)
    problem_type : str
        Type of problem: 'classification', 'regression', or 'auto' (detect)
    perturbation_types : list of str or None
        Types of perturbations to apply (default: ['noise', 'zero'])
    perturbation_levels : list of float or None
        Levels of perturbation to apply (default: [0.1, 0.3, 0.5])
    metrics : dict or None
        Dictionary mapping metric names to metric functions
    n_samples : int or None
        Number of samples to use (None uses all)
    verbose : bool
        Whether to print progress information
        
    Returns:
    --------
    dict : Dictionary with robustness test results
    """
    if perturbation_types is None:
        perturbation_types = ['noise', 'zero']
        
    if perturbation_levels is None:
        perturbation_levels = [0.1, 0.3, 0.5]
        
    # Create robustness tester
    tester = RobustnessTester(
        model=model,
        X=X,
        y=y,
        feature_names=feature_names,
        problem_type=problem_type,
        metrics=metrics,
        verbose=verbose
    )
    
    results = {
        'baseline': tester.baseline_performance,
        'per_feature': defaultdict(dict),
        'summary': {}
    }
    
    # Test individual features
    for feature in tester.feature_names:
        for p_type in perturbation_types:
            result = tester.test_feature_perturbation(
                feature_name=feature,
                perturbation_type=p_type,
                perturbation_level=perturbation_levels,
                n_samples=n_samples
            )
            
            results['per_feature'][feature][p_type] = result
            
    # Run feature importance for each perturbation type
    for p_type in perturbation_types:
        all_features_result = tester.test_all_features(
            perturbation_type=p_type,
            perturbation_level=perturbation_levels[0],  # Use first level for feature importance
            n_samples=n_samples
        )
        
        results['summary'][p_type] = {
            'feature_importance': all_features_result['feature_importance']
        }
        
    return results