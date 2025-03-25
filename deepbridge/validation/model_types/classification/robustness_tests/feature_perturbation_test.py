"""
Feature perturbation tests for classification models.

This module provides tools for testing the robustness of classification
models against various types of feature perturbations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


class ClassificationFeaturePerturbationTest:
    """
    Test classification model robustness against feature perturbations.
    
    This class provides methods for evaluating how well a classification
    model maintains its performance when input features are perturbed.
    """
    
    def __init__(
        self,
        model: Any,
        perturbation_types: Optional[List[str]] = None,
        perturbation_levels: Optional[List[float]] = None,
        metrics: Optional[List[str]] = None,
        feature_subset: Optional[List[str]] = None,
        random_state: Optional[int] = None,
        verbose: bool = False
    ):
        """
        Initialize the feature perturbation test.
        
        Parameters:
        -----------
        model : Any
            Classification model to test
        perturbation_types : list of str or None
            Types of perturbations to apply:
            - 'noise': Add Gaussian noise
            - 'zero': Set values to zero
            - 'swap': Swap feature values between samples
            - 'missing': Simulate missing values
        perturbation_levels : list of float or None
            Levels of perturbation to apply
        metrics : list of str or None
            Metrics to use for evaluation:
            - 'accuracy': Classification accuracy
            - 'f1': F1 score
            - 'roc_auc': ROC AUC score
        feature_subset : list of str or None
            Subset of features to test (None for all)
        random_state : int or None
            Random seed for reproducibility
        verbose : bool
            Whether to print progress information
        """
        self.model = model
        
        # Default perturbation types
        self.perturbation_types = perturbation_types or ['noise', 'zero', 'swap', 'missing']
        
        # Default perturbation levels
        self.perturbation_levels = perturbation_levels or [0.1, 0.2, 0.3, 0.5]
        
        # Default metrics
        self.metrics = metrics or ['accuracy', 'f1', 'roc_auc']
        
        self.feature_subset = feature_subset
        self.random_state = random_state
        self.verbose = verbose
        
        # Set random seed
        np.random.seed(random_state)
        
        # Initialize results storage
        self.results = {}
    
    def test(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Test model robustness against feature perturbations.
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Feature data
        y : array-like or Series
            Target data
        **kwargs : dict
            Additional parameters that override initialization
            
        Returns:
        --------
        dict : Feature perturbation test results
        """
        # Override parameters if provided
        perturbation_types = kwargs.get('perturbation_types', self.perturbation_types)
        perturbation_levels = kwargs.get('perturbation_levels', self.perturbation_levels)
        metrics = kwargs.get('metrics', self.metrics)
        feature_subset = kwargs.get('feature_subset', self.feature_subset)
        
        # Handle data format
        is_dataframe = isinstance(X, pd.DataFrame)
        
        if is_dataframe:
            feature_names = X.columns
            X_values = X.values
        else:
            # If not a DataFrame, create feature names
            X_values = X
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            
        # Determine features to test
        if feature_subset is not None:
            if is_dataframe:
                # Get indices of specified features
                feature_indices = [i for i, name in enumerate(feature_names) if name in feature_subset]
                test_features = [name for name in feature_names if name in feature_subset]
            else:
                # Use indices directly
                feature_indices = [int(f.split('_')[1]) for f in feature_subset]
                test_features = feature_subset
        else:
            # Test all features
            feature_indices = list(range(X.shape[1]))
            test_features = feature_names
            
        # Compute baseline performance
        baseline_performance = self._evaluate_model(X_values, y, metrics)
        
        # Initialize results
        results = {
            'baseline': baseline_performance,
            'per_feature': {},
            'aggregate': {},
            'robustness_score': 0.0
        }
        
        # Test each feature
        for i, feature_idx in enumerate(feature_indices):
            feature_name = test_features[i]
            
            if self.verbose:
                print(f"Testing feature '{feature_name}' ({i+1}/{len(feature_indices)})")
                
            # Test each perturbation type
            feature_results = {}
            for p_type in perturbation_types:
                type_results = []
                
                for level in perturbation_levels:
                    # Apply perturbation
                    X_perturbed = self._perturb_feature(X_values, feature_idx, p_type, level)
                    
                    # Evaluate model
                    performance = self._evaluate_model(X_perturbed, y, metrics)
                    
                    # Calculate relative change
                    relative_change = {}
                    for metric in metrics:
                        baseline = baseline_performance[metric]
                        current = performance[metric]
                        if baseline != 0:
                            relative_change[metric] = (current - baseline) / baseline
                        else:
                            relative_change[metric] = 0
                            
                    # Store results
                    type_results.append({
                        'level': level,
                        'performance': performance,
                        'relative_change': relative_change
                    })
                    
                feature_results[p_type] = type_results
                
            # Store feature results
            results['per_feature'][feature_name] = feature_results
            
        # Compute aggregate results
        aggregate_results = self._compute_aggregate_results(results['per_feature'], metrics)
        results['aggregate'] = aggregate_results
        
        # Compute robustness score
        robustness_score = self._compute_robustness_score(aggregate_results, metrics)
        results['robustness_score'] = robustness_score
        
        # Store results
        self.results = results
        
        return results
    
    def _perturb_feature(
        self,
        X: np.ndarray,
        feature_idx: int,
        perturbation_type: str,
        level: float
    ) -> np.ndarray:
        """
        Apply perturbation to a single feature.
        
        Parameters:
        -----------
        X : array-like
            Feature data
        feature_idx : int
            Index of feature to perturb
        perturbation_type : str
            Type of perturbation to apply
        level : float
            Level of perturbation to apply
            
        Returns:
        --------
        array-like : Perturbed data
        """
        # Create a copy
        X_perturbed = X.copy()
        
        # Get feature values
        feature_values = X[:, feature_idx]
        
        # Apply perturbation
        if perturbation_type == 'noise':
            # Add Gaussian noise
            std = np.std(feature_values)
            noise = np.random.normal(0, std * level, size=len(feature_values))
            X_perturbed[:, feature_idx] = feature_values + noise
            
        elif perturbation_type == 'zero':
            # Set values to zero
            mask = np.random.random(len(feature_values)) < level
            X_perturbed[mask, feature_idx] = 0
            
        elif perturbation_type == 'swap':
            # Swap values between samples
            n_swaps = int(len(feature_values) * level)
            for _ in range(n_swaps):
                i, j = np.random.choice(len(feature_values), 2, replace=False)
                X_perturbed[i, feature_idx], X_perturbed[j, feature_idx] = X_perturbed[j, feature_idx], X_perturbed[i, feature_idx]
                
        elif perturbation_type == 'missing':
            # Simulate missing values (replace with mean)
            mask = np.random.random(len(feature_values)) < level
            mean_value = np.mean(feature_values)
            X_perturbed[mask, feature_idx] = mean_value
            
        else:
            raise ValueError(f"Unknown perturbation type: {perturbation_type}")
            
        return X_perturbed
    
    def _evaluate_model(
        self,
        X: np.ndarray,
        y: Union[np.ndarray, pd.Series],
        metrics: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate model performance on data.
        
        Parameters:
        -----------
        X : array-like
            Feature data
        y : array-like or Series
            Target data
        metrics : list of str
            Metrics to compute
            
        Returns:
        --------
        dict : Performance metrics
        """
        # Convert y to numpy array if needed
        if isinstance(y, pd.Series):
            y_values = y.values
        else:
            y_values = y
            
        # Make predictions
        try:
            # Class predictions
            y_pred = self.model.predict(X)
            
            # Probability predictions if available
            has_proba = hasattr(self.model, 'predict_proba')
            if has_proba:
                y_proba = self.model.predict_proba(X)
                
                # Get number of classes
                if hasattr(self.model, 'classes_'):
                    n_classes = len(self.model.classes_)
                else:
                    n_classes = y_proba.shape[1]
        except Exception as e:
            # If prediction fails, return zeros
            return {metric: 0.0 for metric in metrics}
            
        # Compute metrics
        results = {}
        
        for metric in metrics:
            if metric == 'accuracy':
                results[metric] = accuracy_score(y_values, y_pred)
            elif metric == 'f1':
                if len(np.unique(y_values)) > 2:
                    # Multiclass
                    results[metric] = f1_score(y_values, y_pred, average='weighted')
                else:
                    # Binary
                    results[metric] = f1_score(y_values, y_pred)
            elif metric == 'roc_auc':
                if has_proba:
                    if n_classes > 2:
                        # Multiclass
                        results[metric] = roc_auc_score(y_values, y_proba, multi_class='ovr')
                    else:
                        # Binary
                        results[metric] = roc_auc_score(y_values, y_proba[:, 1])
                else:
                    # ROC AUC requires probabilities
                    results[metric] = 0.5
            else:
                # Unknown metric
                results[metric] = 0.0
                
        return results
    
    def _compute_aggregate_results(
        self,
        per_feature_results: Dict[str, Dict[str, List[Dict[str, Any]]]],
        metrics: List[str]
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Compute aggregate results across features.
        
        Parameters:
        -----------
        per_feature_results : dict
            Results for each feature
        metrics : list of str
            Metrics used
            
        Returns:
        --------
        dict : Aggregate results
        """
        # Initialize aggregate results
        aggregate = {}
        
        # Get perturbation types and levels
        p_types = list(next(iter(per_feature_results.values())).keys())
        
        for p_type in p_types:
            type_aggregate = {}
            
            # Get all levels for this perturbation type
            first_feature = next(iter(per_feature_results.keys()))
            levels = [r['level'] for r in per_feature_results[first_feature][p_type]]
            
            for level_idx, level in enumerate(levels):
                level_results = {}
                
                for metric in metrics:
                    # Collect relative changes for this metric across all features
                    changes = []
                    
                    for feature in per_feature_results:
                        change = per_feature_results[feature][p_type][level_idx]['relative_change'][metric]
                        changes.append(change)
                        
                    # Compute statistics
                    level_results[metric] = {
                        'mean': float(np.mean(changes)),
                        'std': float(np.std(changes)),
                        'min': float(np.min(changes)),
                        'max': float(np.max(changes)),
                        'median': float(np.median(changes))
                    }
                    
                type_aggregate[level] = level_results
                
            aggregate[p_type] = type_aggregate
            
        return aggregate
    
    def _compute_robustness_score(
        self,
        aggregate_results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
        metrics: List[str]
    ) -> float:
        """
        Compute overall robustness score.
        
        Parameters:
        -----------
        aggregate_results : dict
            Aggregate results
        metrics : list of str
            Metrics used
            
        Returns:
        --------
        float : Robustness score
        """
        # Collect all mean relative changes
        all_changes = []
        
        for p_type in aggregate_results:
            for level in aggregate_results[p_type]:
                for metric in metrics:
                    mean_change = aggregate_results[p_type][level][metric]['mean']
                    all_changes.append(mean_change)
                    
        # Compute score
        # Lower (less negative) changes mean more robust model
        # Normalize to [0, 1] where 1 is most robust
        if not all_changes:
            return 0.0
            
        # Get mean change (typically negative for degrading performance)
        mean_change = np.mean(all_changes)
        
        # Convert to robustness score (higher is better)
        # Limit range to [0, 1]
        score = max(0, 1 + mean_change)
        score = min(1, score)
        
        return float(score)
    
    def plot_feature_robustness(
        self,
        feature_name: str,
        perturbation_type: str = 'noise',
        metric: str = 'accuracy',
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot robustness of a specific feature.
        
        Parameters:
        -----------
        feature_name : str
            Name of the feature to plot
        perturbation_type : str
            Type of perturbation to plot
        metric : str
            Metric to plot
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.Figure : Robustness plot
        """
        if not self.results:
            raise ValueError("No results available. Call test() first.")
            
        if feature_name not in self.results['per_feature']:
            raise ValueError(f"Feature '{feature_name}' not found in results.")
            
        if perturbation_type not in self.results['per_feature'][feature_name]:
            raise ValueError(f"Perturbation type '{perturbation_type}' not found for feature '{feature_name}'.")
            
        # Get feature results
        feature_results = self.results['per_feature'][feature_name][perturbation_type]
        
        # Extract data
        levels = [r['level'] for r in feature_results]
        relative_changes = [r['relative_change'][metric] for r in feature_results]
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot relative change
        ax.plot(levels, relative_changes, 'o-', label=f'Relative change in {metric}')
        
        # Add reference line at 0
        ax.axhline(y=0, color='gray', linestyle='--')
        
        # Set labels and title
        ax.set_xlabel('Perturbation Level')
        ax.set_ylabel(f'Relative Change in {metric.upper()}')
        ax.set_title(f'Robustness of Feature "{feature_name}" to {perturbation_type.title()} Perturbation')
        
        # Add legend
        ax.legend()
        
        return fig
    
    def plot_aggregate_robustness(
        self,
        perturbation_type: str = 'noise',
        metric: str = 'accuracy',
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot aggregate robustness across all features.
        
        Parameters:
        -----------
        perturbation_type : str
            Type of perturbation to plot
        metric : str
            Metric to plot
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.Figure : Aggregate robustness plot
        """
        if not self.results:
            raise ValueError("No results available. Call test() first.")
            
        if 'aggregate' not in self.results:
            raise ValueError("Aggregate results not available.")
            
        if perturbation_type not in self.results['aggregate']:
            raise ValueError(f"Perturbation type '{perturbation_type}' not found in aggregate results.")
            
        # Get aggregate results
        agg_results = self.results['aggregate'][perturbation_type]
        
        # Extract data
        levels = sorted(agg_results.keys())
        means = [agg_results[level][metric]['mean'] for level in levels]
        stds = [agg_results[level][metric]['std'] for level in levels]
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot mean with error bars
        ax.errorbar(levels, means, yerr=stds, marker='o', linestyle='-', 
                   label=f'Mean relative change in {metric}')
        
        # Add reference line at 0
        ax.axhline(y=0, color='gray', linestyle='--')
        
        # Set labels and title
        ax.set_xlabel('Perturbation Level')
        ax.set_ylabel(f'Mean Relative Change in {metric.upper()}')
        ax.set_title(f'Aggregate Robustness to {perturbation_type.title()} Perturbation')
        
        # Add legend
        ax.legend()
        
        return fig


def test_classifier_feature_robustness(
    model: Any,
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    perturbation_types: Optional[List[str]] = None,
    perturbation_levels: Optional[List[float]] = None,
    metrics: Optional[List[str]] = None,
    feature_subset: Optional[List[str]] = None,
    random_state: Optional[int] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Test classification model robustness against feature perturbations.
    
    Parameters:
    -----------
    model : Any
        Classification model to test
    X : array-like or DataFrame
        Feature data
    y : array-like or Series
        Target data
    perturbation_types : list of str or None
        Types of perturbations to apply
    perturbation_levels : list of float or None
        Levels of perturbation to apply
    metrics : list of str or None
        Metrics to use for evaluation
    feature_subset : list of str or None
        Subset of features to test (None for all)
    random_state : int or None
        Random seed for reproducibility
    verbose : bool
        Whether to print progress information
        
    Returns:
    --------
    dict : Feature robustness test results
    """
    tester = ClassificationFeaturePerturbationTest(
        model=model,
        perturbation_types=perturbation_types,
        perturbation_levels=perturbation_levels,
        metrics=metrics,
        feature_subset=feature_subset,
        random_state=random_state,
        verbose=verbose
    )
    
    return tester.test(X, y)


def plot_feature_perturbation_impact(
    results: Dict[str, Any],
    feature_name: str,
    perturbation_type: str = 'noise',
    metric: str = 'accuracy',
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot impact of feature perturbation on model performance.
    
    Parameters:
    -----------
    results : dict
        Results from feature robustness test
    feature_name : str
        Name of the feature to plot
    perturbation_type : str
        Type of perturbation to plot
    metric : str
        Metric to plot
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.Figure : Impact plot
    """
    if 'per_feature' not in results:
        raise ValueError("Invalid results format. 'per_feature' section not found.")
        
    if feature_name not in results['per_feature']:
        raise ValueError(f"Feature '{feature_name}' not found in results.")
        
    if perturbation_type not in results['per_feature'][feature_name]:
        raise ValueError(f"Perturbation type '{perturbation_type}' not found for feature '{feature_name}'.")
        
    # Get feature results
    feature_results = results['per_feature'][feature_name][perturbation_type]
    
    # Extract data
    levels = [r['level'] for r in feature_results]
    relative_changes = [r['relative_change'][metric] for r in feature_results]
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot relative change
    ax.plot(levels, relative_changes, 'o-', label=f'Relative change in {metric}')
    
    # Add reference line at 0
    ax.axhline(y=0, color='gray', linestyle='--')
    
    # Set labels and title
    ax.set_xlabel('Perturbation Level')
    ax.set_ylabel(f'Relative Change in {metric.upper()}')
    ax.set_title(f'Impact of {perturbation_type.title()} Perturbation on Feature "{feature_name}"')
    
    # Add legend
    ax.legend()
    
    return fig