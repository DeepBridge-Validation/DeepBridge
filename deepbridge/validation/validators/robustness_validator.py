
# robustness_validator.py - placeholder file
"""
Robustness validator for machine learning models.

This module provides a validator for testing model robustness
against data perturbations and noise.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
import matplotlib.pyplot as plt
import os
import warnings

from ..core.base_validator import BaseValidator
from ..utils.model_inspection import is_classifier, is_regressor, get_model_type
from ..utils.validation_utils import save_validation_results


class RobustnessValidator(BaseValidator):
    """
    Validator for evaluating model robustness.
    
    This class provides methods for testing model robustness against
    various types of data perturbations and noise.
    """
    
    def __init__(
        self,
        model: Any,
        perturbation_types: Optional[List[str]] = None,
        perturbation_levels: Optional[List[float]] = None,
        feature_subset: Optional[List[str]] = None,
        metrics: Optional[Dict[str, Callable]] = None,
        random_state: Optional[int] = None,
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize the robustness validator.
        
        Parameters:
        -----------
        model : Any
            Machine learning model to validate
        perturbation_types : list of str or None
            Types of perturbations to apply
        perturbation_levels : list of float or None
            Levels of perturbations to apply
        feature_subset : list of str or None
            Subset of features to test (None for all)
        metrics : dict or None
            Custom evaluation metrics
        random_state : int or None
            Random seed for reproducibility
        verbose : bool
            Whether to print progress information
        **kwargs : dict
            Additional parameters
        """
        super().__init__(model=model, **kwargs)
        
        # Default perturbation types if not provided
        self.perturbation_types = perturbation_types or ['noise', 'zero', 'missing']
        
        # Default perturbation levels if not provided
        self.perturbation_levels = perturbation_levels or [0.1, 0.2, 0.5]
        
        self.feature_subset = feature_subset
        self.random_state = random_state
        self.verbose = verbose
        
        # Determine model type if not provided
        self.model_type = kwargs.get('model_type', get_model_type(model))
        
        # Set up metrics based on model type
        if metrics is None:
            from ..utils.validation_utils import get_default_metrics
            self.metrics = get_default_metrics(self.model_type)
        else:
            self.metrics = metrics
            
        # Initialize results
        self.results = {}
    
    def validate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Validate model robustness.
        
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
        perturbation_types = kwargs.get('perturbation_types', self.perturbation_types)
        perturbation_levels = kwargs.get('perturbation_levels', self.perturbation_levels)
        feature_subset = kwargs.get('feature_subset', self.feature_subset)
        
        # Start with clean results
        results = {
            'model_type': self.model_type,
            'perturbation_types': perturbation_types,
            'perturbation_levels': perturbation_levels,
            'baseline': {},
            'perturbations': {},
            'feature_importance': {},
            'robustness_score': 0.0
        }
        
        # Get baseline performance
        baseline_performance = self._evaluate_baseline(X, y)
        results['baseline'] = baseline_performance
        
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            if feature_subset is not None:
                warnings.warn("Feature subset specified but X is not a DataFrame. Using indices instead.")
                
            # Convert to DataFrame
            X = pd.DataFrame(X)
            
        # Get feature names
        feature_names = X.columns.tolist()
        
        # Determine features to test
        if feature_subset is not None:
            test_features = [f for f in feature_subset if f in feature_names]
            
            if len(test_features) == 0:
                warnings.warn("No features in the specified subset were found in the data.")
                test_features = feature_names
        else:
            test_features = feature_names
            
        # Run perturbation tests
        all_perturbation_results = {}
        for p_type in perturbation_types:
            if self.verbose:
                print(f"Testing perturbation type: {p_type}")
                
            type_results = self._test_perturbation(
                X, y, p_type, perturbation_levels, test_features
            )
            
            all_perturbation_results[p_type] = type_results
            
        results['perturbations'] = all_perturbation_results
        
        # Calculate feature importance based on robustness
        feature_importance = self._calculate_feature_importance(all_perturbation_results)
        results['feature_importance'] = feature_importance
        
        # Calculate overall robustness score
        robustness_score = self._calculate_robustness_score(all_perturbation_results)
        results['robustness_score'] = robustness_score
        
        # Store results
        self.results = results
        
        return results
    
    def _evaluate_baseline(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series]
    ) -> Dict[str, float]:
        """
        Evaluate baseline model performance.
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Features
        y : array-like or Series
            Target values
            
        Returns:
        --------
        dict : Baseline performance metrics
        """
        try:
            # Make predictions
            y_pred = self.model.predict(X)
            
            # For classifiers with predict_proba
            has_proba = hasattr(self.model, 'predict_proba') and is_classifier(self.model)
            if has_proba:
                y_proba = self.model.predict_proba(X)
            else:
                y_proba = None
                
            # Calculate metrics
            metrics_results = {}
            for metric_name, metric_func in self.metrics.items():
                try:
                    if metric_name == 'roc_auc' and y_proba is not None:
                        # ROC AUC requires probabilities
                        if y_proba.shape[1] == 2:
                            # Binary classification
                            score = metric_func(y, y_proba[:, 1])
                        else:
                            # Multiclass classification
                            score = metric_func(y, y_proba)
                    else:
                        # Standard metric
                        score = metric_func(y, y_pred)
                        
                    metrics_results[metric_name] = float(score)
                except Exception as e:
                    if self.verbose:
                        print(f"Error calculating metric {metric_name}: {str(e)}")
                    metrics_results[metric_name] = float('nan')
                    
            return metrics_results
            
        except Exception as e:
            if self.verbose:
                print(f"Error evaluating baseline: {str(e)}")
            return {metric: float('nan') for metric in self.metrics}
    
    def _test_perturbation(
        self,
        X: pd.DataFrame,
        y: Union[np.ndarray, pd.Series],
        perturbation_type: str,
        perturbation_levels: List[float],
        test_features: List[str]
    ) -> Dict[str, Any]:
        """
        Test a specific type of perturbation.
        
        Parameters:
        -----------
        X : DataFrame
            Features
        y : array-like or Series
            Target values
        perturbation_type : str
            Type of perturbation to apply
        perturbation_levels : list of float
            Levels of perturbation to apply
        test_features : list of str
            Features to test
            
        Returns:
        --------
        dict : Perturbation test results
        """
        # Initialize results
        results = {
            'feature_results': {},
            'aggregate_results': {}
        }
        
        # Test each feature
        for feature in test_features:
            if self.verbose:
                print(f"  Testing feature: {feature}")
                
            feature_results = []
            
            for level in perturbation_levels:
                # Create perturbed data
                X_perturbed = X.copy()
                
                # Apply perturbation to feature
                X_perturbed[feature] = self._perturb_feature(
                    X[feature].values, perturbation_type, level
                )
                
                # Evaluate model on perturbed data
                perturbed_performance = self._evaluate_baseline(X_perturbed, y)
                
                # Calculate relative change from baseline
                relative_change = {}
                for metric, value in perturbed_performance.items():
                    if metric in self.results.get('baseline', {}):
                        baseline_value = self.results['baseline'][metric]
                        if baseline_value != 0 and not np.isnan(baseline_value):
                            relative_change[metric] = (value - baseline_value) / abs(baseline_value)
                        else:
                            relative_change[metric] = 0.0
                    else:
                        relative_change[metric] = 0.0
                        
                # Store results for this level
                feature_results.append({
                    'level': level,
                    'performance': perturbed_performance,
                    'relative_change': relative_change
                })
                
            # Store feature results
            results['feature_results'][feature] = feature_results
            
        # Compute aggregate results
        aggregate_results = self._compute_aggregate_results(
            results['feature_results'], perturbation_levels
        )
        results['aggregate_results'] = aggregate_results
        
        return results
    
    def _perturb_feature(
        self,
        feature_values: np.ndarray,
        perturbation_type: str,
        level: float
    ) -> np.ndarray:
        """
        Apply perturbation to a feature.
        
        Parameters:
        -----------
        feature_values : array-like
            Feature values to perturb
        perturbation_type : str
            Type of perturbation to apply
        level : float
            Level of perturbation to apply
            
        Returns:
        --------
        numpy.ndarray : Perturbed feature values
        """
        # Set random seed
        np.random.seed(self.random_state)
        
        # Apply perturbation based on type
        if perturbation_type == 'noise':
            # Add Gaussian noise
            std = np.std(feature_values)
            noise = np.random.normal(0, level * std, size=len(feature_values))
            return feature_values + noise
            
        elif perturbation_type == 'zero':
            # Set a fraction of values to zero
            perturbed = feature_values.copy()
            mask = np.random.random(size=len(feature_values)) < level
            perturbed[mask] = 0
            return perturbed
            
        elif perturbation_type == 'missing':
            # Simulate missing values (replace with mean)
            perturbed = feature_values.copy()
            mask = np.random.random(size=len(feature_values)) < level
            perturbed[mask] = np.nanmean(feature_values)
            return perturbed
            
        elif perturbation_type == 'swap':
            # Randomly swap values
            perturbed = feature_values.copy()
            n_swaps = int(len(feature_values) * level)
            
            for _ in range(n_swaps):
                i, j = np.random.choice(len(feature_values), 2, replace=False)
                perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
                
            return perturbed
            
        else:
            raise ValueError(f"Unknown perturbation type: {perturbation_type}")
    
    def _compute_aggregate_results(
        self,
        feature_results: Dict[str, List[Dict[str, Any]]],
        perturbation_levels: List[float]
    ) -> Dict[str, Any]:
        """
        Compute aggregate results across all features.
        
        Parameters:
        -----------
        feature_results : dict
            Results for each feature
        perturbation_levels : list of float
            Levels of perturbation
            
        Returns:
        --------
        dict : Aggregate results
        """
        # Initialize aggregate results
        aggregate = {level: {} for level in perturbation_levels}
        
        # Get all metrics
        all_metrics = set()
        for feature, results in feature_results.items():
            for level_result in results:
                all_metrics.update(level_result['relative_change'].keys())
                
        # Compute aggregates for each level and metric
        for level_idx, level in enumerate(perturbation_levels):
            for metric in all_metrics:
                # Collect relative changes across features
                changes = []
                
                for feature, results in feature_results.items():
                    if level_idx < len(results) and metric in results[level_idx]['relative_change']:
                        changes.append(results[level_idx]['relative_change'][metric])
                        
                # Compute statistics if we have data
                if changes:
                    aggregate[level][metric] = {
                        'mean': float(np.mean(changes)),
                        'std': float(np.std(changes)),
                        'min': float(np.min(changes)),
                        'max': float(np.max(changes)),
                        'median': float(np.median(changes))
                    }
                    
        return aggregate
    
    def _calculate_feature_importance(
        self,
        perturbation_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate feature importance based on robustness.
        
        Parameters:
        -----------
        perturbation_results : dict
            Results from perturbation tests
            
        Returns:
        --------
        dict : Feature importance based on robustness
        """
        # Initialize importance
        importance = {}
        
        # Choose primary metric based on model type
        if self.model_type == 'classifier':
            primary_metric = 'accuracy' if 'accuracy' in self.metrics else next(iter(self.metrics))
        else:
            primary_metric = 'mse' if 'mse' in self.metrics else next(iter(self.metrics))
            
        # Collect impact of perturbations for each feature
        for p_type, type_results in perturbation_results.items():
            if 'feature_results' not in type_results:
                continue
                
            for feature, feature_results in type_results['feature_results'].items():
                # Skip if no results
                if not feature_results:
                    continue
                    
                # Get average relative change for this feature
                changes = []
                
                for level_result in feature_results:
                    if primary_metric in level_result['relative_change']:
                        changes.append(level_result['relative_change'][primary_metric])
                        
                if changes:
                    # Calculate average change (negative means worse performance)
                    avg_change = np.mean(changes)
                    
                    # Store importance (higher magnitude of negative change = more important)
                    if feature not in importance:
                        importance[feature] = 0.0
                        
                    # We want the largest negative impact
                    importance[feature] += max(0, -avg_change)
                    
        # Normalize importance to [0, 1]
        if importance:
            max_importance = max(importance.values())
            if max_importance > 0:
                importance = {
                    feature: value / max_importance
                    for feature, value in importance.items()
                }
                
        return importance
    
    def _calculate_robustness_score(
        self,
        perturbation_results: Dict[str, Dict[str, Any]]
    ) -> float:
        """
        Calculate overall robustness score.
        
        Parameters:
        -----------
        perturbation_results : dict
            Results from perturbation tests
            
        Returns:
        --------
        float : Robustness score [0, 1]
        """
        # Choose primary metric based on model type
        if self.model_type == 'classifier':
            primary_metric = 'accuracy' if 'accuracy' in self.metrics else next(iter(self.metrics))
        else:
            primary_metric = 'mse' if 'mse' in self.metrics else next(iter(self.metrics))
            
        # Collect all relative changes
        all_changes = []
        
        for p_type, type_results in perturbation_results.items():
            if 'aggregate_results' not in type_results:
                continue
                
            for level, level_results in type_results['aggregate_results'].items():
                if primary_metric in level_results:
                    all_changes.append(level_results[primary_metric]['mean'])
                    
        # Compute robustness score
        if not all_changes:
            return 0.5  # Default score if no data
            
        # Average of relative changes (negative means worse performance)
        avg_change = float(np.mean(all_changes))
        
        # Convert to [0, 1] score (robust models have changes closer to 0)
        # Higher score = more robust
        return max(0, min(1, 1 + avg_change))
    
    def plot_feature_importance(
        self,
        top_n: int = 10,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot feature importance based on robustness.
        
        Parameters:
        -----------
        top_n : int
            Number of top features to show
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.Figure : Feature importance plot
        """
        if not self.results or 'feature_importance' not in self.results:
            raise ValueError("No feature importance data available. Run validate() first.")
            
        # Get feature importance
        importance = self.results['feature_importance']
        
        # Sort features by importance
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        # Limit to top N features
        top_features = sorted_features[:top_n]
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract feature names and importance values
        feature_names = [f[0] for f in top_features]
        importance_values = [f[1] for f in top_features]
        
        # Create horizontal bar chart
        y_pos = range(len(feature_names))
        ax.barh(y_pos, importance_values, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Robustness Importance')
        ax.set_title('Feature Importance Based on Robustness')
        
        plt.tight_layout()
        return fig
    
    def plot_perturbation_impact(
        self,
        perturbation_type: Optional[str] = None,
        metric: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Plot impact of perturbations on model performance.
        
        Parameters:
        -----------
        perturbation_type : str or None
            Type of perturbation to plot (None for first available)
        metric : str or None
            Metric to plot (None for primary metric)
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.Figure : Perturbation impact plot
        """
        if not self.results or 'perturbations' not in self.results:
            raise ValueError("No perturbation data available. Run validate() first.")
            
        # Choose perturbation type if not specified
        if perturbation_type is None:
            perturbation_type = next(iter(self.results['perturbations'].keys()))
            
        if perturbation_type not in self.results['perturbations']:
            raise ValueError(f"Perturbation type '{perturbation_type}' not found in results")
            
        # Choose metric if not specified
        if metric is None:
            if self.model_type == 'classifier':
                metric = 'accuracy' if 'accuracy' in self.metrics else next(iter(self.metrics))
            else:
                metric = 'mse' if 'mse' in self.metrics else next(iter(self.metrics))
                
        # Get aggregate results
        if 'aggregate_results' not in self.results['perturbations'][perturbation_type]:
            raise ValueError(f"No aggregate results found for perturbation type '{perturbation_type}'")
            
        aggregate = self.results['perturbations'][perturbation_type]['aggregate_results']
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract data
        levels = sorted([float(level) for level in aggregate.keys()])
        means = []
        stds = []
        
        for level in levels:
            if str(level) in aggregate and metric in aggregate[str(level)]:
                means.append(aggregate[str(level)][metric]['mean'])
                stds.append(aggregate[str(level)][metric]['std'])
            else:
                means.append(0)
                stds.append(0)
                
        # Plot means with error bars
        ax.errorbar(levels, means, yerr=stds, marker='o', linestyle='-', 
                   label=f'{metric.upper()} (relative change)')
        
        # Add reference line at 0
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Set labels and title
        ax.set_xlabel('Perturbation Level')
        ax.set_ylabel(f'Relative Change in {metric.upper()}')
        ax.set_title(f'Impact of {perturbation_type.title()} Perturbation on {metric.upper()}')
        
        # Add legend
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def save_results(
        self,
        output_dir: str,
        prefix: str = 'robustness',
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
        
        # Save plots if requested
        if include_plots and self.results:
            # Save feature importance plot
            try:
                fig = self.plot_feature_importance()
                fig_path = os.path.join(output_dir, f"{prefix}_feature_importance.png")
                fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                saved_files['feature_importance_plot'] = fig_path
                plt.close(fig)
            except Exception as e:
                if self.verbose:
                    print(f"Error saving feature importance plot: {str(e)}")
                    
            # Save perturbation impact plot for each perturbation type
            for p_type in self.results.get('perturbations', {}):
                try:
                    fig = self.plot_perturbation_impact(p_type)
                    fig_path = os.path.join(output_dir, f"{prefix}_{p_type}_impact.png")
                    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                    saved_files[f'{p_type}_impact_plot'] = fig_path
                    plt.close(fig)
                except Exception as e:
                    if self.verbose:
                        print(f"Error saving {p_type} impact plot: {str(e)}")
                        
        return saved_files