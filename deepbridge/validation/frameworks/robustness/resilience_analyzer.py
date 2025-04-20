"""
Resilience analysis for machine learning models.

This module provides tools for assessing model resilience against
various types of data shifts and perturbations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score
from sklearn.model_selection import train_test_split
import warnings


class ResilienceAnalyzer:
    """
    Analyze model resilience against data shifts and perturbations.
    
    This class provides methods for quantifying a model's ability to
    maintain performance when faced with various types of data shifts.
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
        Initialize the resilience analyzer.
        
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
        metrics : dict or None
            Dictionary mapping metric names to metric functions
        verbose : bool
            Whether to print progress information
        """
        # Initialize a robustness tester for base functionality
        from .robustness_tester import RobustnessTester
        self.base_tester = RobustnessTester(
            model=model,
            X=X,
            y=y,
            feature_names=feature_names,
            problem_type=problem_type,
            metrics=metrics,
            verbose=verbose
        )
        
        self.model = model
        
        # Convert X to DataFrame if it's a numpy array
        if isinstance(X, np.ndarray):
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            self.X = pd.DataFrame(X, columns=feature_names)
        else:
            self.X = X
            
        # Convert y to numpy array if it's a Series
        if isinstance(y, pd.Series):
            self.y = y.values
        else:
            self.y = y
            
        self.feature_names = list(self.X.columns)
        self.problem_type = self.base_tester.problem_type
        self.metrics = self.base_tester.metrics
        self.verbose = verbose
        self.results = {}
        
        # Get baseline performance
        self.baseline_performance = self.base_tester.baseline_performance
    
    def test_distribution_shift(
        self,
        feature_name: str,
        shift_type: str = 'mean',
        shift_levels: Union[float, List[float]] = [0.1, 0.3, 0.5, 1.0],
        n_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Test model resilience against distribution shift in a single feature.
        
        Parameters:
        -----------
        feature_name : str
            Name of the feature to shift
        shift_type : str
            Type of distribution shift to apply:
            - 'mean': Shift the mean
            - 'variance': Increase the variance
            - 'skew': Apply skewness
        shift_levels : float or list of floats
            Levels of shift to apply
        n_samples : int or None
            Number of samples to use (None uses all)
            
        Returns:
        --------
        dict : Dictionary with distribution shift results
        """
        if feature_name not in self.feature_names:
            raise ValueError(f"Feature {feature_name} not found")
            
        # Convert shift_levels to list if it's a single value
        if isinstance(shift_levels, (int, float)):
            shift_levels = [shift_levels]
            
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
        feature_mean = np.mean(feature_values)
        feature_std = np.std(feature_values)
        
        results = {
            'feature': feature_name,
            'shift_type': shift_type,
            'shift_levels': shift_levels,
            'performance': []
        }
        
        for level in shift_levels:
            # Apply shift
            X_shifted = X_test.copy()
            
            if shift_type == 'mean':
                # Shift mean by level * std
                X_shifted[feature_name] = feature_values + level * feature_std
            elif shift_type == 'variance':
                # Increase variance by factor of (1 + level)
                centered = feature_values - feature_mean
                X_shifted[feature_name] = feature_mean + centered * np.sqrt(1 + level)
            elif shift_type == 'skew':
                # Apply skewness using exponential transformation
                normalized = (feature_values - feature_mean) / feature_std
                skewed = np.exp(level * normalized) - 1
                X_shifted[feature_name] = feature_mean + skewed * feature_std
            else:
                raise ValueError(f"Unknown shift type: {shift_type}")
                
            # Evaluate model on shifted data
            performance = self.base_tester._evaluate_model(X_shifted, y_test)
            
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
                print(f"Shift level {level}: {performance}")
        
        # Store results
        result_key = f"distribution_shift_{feature_name}_{shift_type}"
        self.results[result_key] = results
        
        return results
    
    def compute_resilience_score(
        self,
        feature_names: Optional[List[str]] = None,
        shift_types: Optional[List[str]] = None,
        shift_level: float = 0.5
    ) -> Dict[str, float]:
        """
        Compute an overall resilience score for the model.
        
        Parameters:
        -----------
        feature_names : list of str or None
            Names of features to include (None uses all)
        shift_types : list of str or None
            Types of shifts to include
            (None uses ['mean', 'variance', 'skew'])
        shift_level : float
            Level of shift to use for the score
            
        Returns:
        --------
        dict : Dictionary with resilience scores
        """
        if feature_names is None:
            feature_names = self.feature_names
            
        if shift_types is None:
            shift_types = ['mean', 'variance', 'skew']
            
        # Select a representative metric
        if self.problem_type == 'classification':
            metric = 'accuracy' if 'accuracy' in self.metrics else list(self.metrics.keys())[0]
        else:
            metric = 'mse' if 'mse' in self.metrics else list(self.metrics.keys())[0]
            
        # Check if we need to compute results
        results_to_compute = []
        for feature in feature_names:
            for shift_type in shift_types:
                result_key = f"distribution_shift_{feature}_{shift_type}"
                if result_key not in self.results:
                    results_to_compute.append((feature, shift_type))
        
        # Compute missing results
        for feature, shift_type in results_to_compute:
            self.test_distribution_shift(
                feature_name=feature,
                shift_type=shift_type,
                shift_levels=[shift_level],
                n_samples=None
            )
        
        # Compute average relative change for each feature
        feature_scores = {}
        for feature in feature_names:
            feature_changes = []
            
            for shift_type in shift_types:
                result_key = f"distribution_shift_{feature}_{shift_type}"
                if result_key in self.results:
                    result = self.results[result_key]
                    
                    # Find the closest shift level
                    closest_idx = np.argmin([abs(level - shift_level) 
                                           for level in result['shift_levels']])
                    
                    performance = result['performance'][closest_idx]
                    relative_change = performance['relative_change'][metric]
                    
                    # For metrics where lower is better, flip the sign
                    if metric in ['mse', 'rmse']:
                        relative_change = -relative_change
                        
                    feature_changes.append(relative_change)
            
            if feature_changes:
                # Negative values mean worse performance after shift
                feature_scores[feature] = np.mean(feature_changes)
        
        # Compute overall resilience score
        # Higher (less negative) values mean more resilient
        overall_score = np.mean(list(feature_scores.values()))
        
        # Normalize score to [0, 1] where higher is more resilient
        # A perfect model would have a score of 1
        # The worst score would be -inf, but we'll clamp it to 0
        normalized_score = max(0, 1 + overall_score)
        normalized_score = min(1, normalized_score)  # Cap at 1
        
        return {
            'feature_scores': feature_scores,
            'overall_score': overall_score,
            'normalized_score': normalized_score
        }
    
    def plot_distribution_shift(
        self,
        feature_name: str,
        shift_type: str = 'mean',
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot the effect of distribution shift on model performance.
        
        Parameters:
        -----------
        feature_name : str
            Name of the feature to plot
        shift_type : str
            Type of distribution shift to plot
        figsize : tuple
            Figure size (width, height)
            
        Returns:
        --------
        matplotlib.Figure : Performance plot
        """
        result_key = f"distribution_shift_{feature_name}_{shift_type}"
        
        if result_key not in self.results:
            raise ValueError(f"No results found for feature {feature_name} with shift {shift_type}")
            
        results = self.results[result_key]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract shift levels and performance values
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
        ax.set_xlabel(f"Shift Level ({shift_type})")
        ax.set_ylabel("Relative Performance Change")
        ax.set_title(f"Effect of {shift_type.title()} Shift in {feature_name}")
        ax.legend()
        
        return fig
    
    def plot_resilience_scores(self, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot resilience scores for each feature.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
            
        Returns:
        --------
        matplotlib.Figure : Resilience scores plot
        """
        # Compute resilience scores if not already done
        scores = self.compute_resilience_score()
        feature_scores = scores['feature_scores']
        
        # Sort features by score
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1])
        features = [f[0] for f in sorted_features]
        values = [f[1] for f in sorted_features]
        
        fig, ax = plt.subplots(figsize=figsize)
        y_pos = np.arange(len(features))
        
        bars = ax.barh(y_pos, values, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Resilience Score')
        ax.set_title('Feature Resilience Scores')
        
        # Add a reference line at y=0
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        
        # Color bars based on value
        for i, bar in enumerate(bars):
            if values[i] < 0:
                bar.set_color('r')
            else:
                bar.set_color('g')
        
        return fig


def compute_resilience_score(
    model: Any,
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    feature_names: Optional[List[str]] = None,
    problem_type: str = 'auto',
    shift_level: float = 0.5,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Compute an overall resilience score for a model.
    
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
    shift_level : float
        Level of shift to use for the score
    verbose : bool
        Whether to print progress information
        
    Returns:
    --------
    dict : Dictionary with resilience scores
    """
    # Create resilience analyzer
    analyzer = ResilienceAnalyzer(
        model=model,
        X=X,
        y=y,
        feature_names=feature_names,
        problem_type=problem_type,
        verbose=verbose
    )
    
    # Compute resilience score
    return analyzer.compute_resilience_score(shift_level=shift_level)