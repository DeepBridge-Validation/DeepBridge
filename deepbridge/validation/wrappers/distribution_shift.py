"""
Distribution shift tests for model validation.

This module provides a wrapper for testing model robustness
against various types of distribution shifts using DBDataset objects.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
import matplotlib.pyplot as plt

from .base_wrapper import BaseWrapper
from deepbridge.validation.frameworks.robustness.resilience_analyzer import ResilienceAnalyzer


class DistributionShiftTests(BaseWrapper):
    """
    Tests for model robustness against distribution shifts.
    
    This class provides methods for testing how well a model maintains
    its performance when the distribution of feature values changes.
    """
    
    def __init__(self, dataset, verbose: bool = False):
        """
        Initialize the distribution shift tests.
        
        Parameters:
        -----------
        dataset : DBDataset
            Dataset object containing training/test data and model
        verbose : bool
            Whether to print progress information
        """
        super().__init__(dataset, verbose)
        
        # Initialize the underlying resilience analyzer
        self._analyzer = ResilienceAnalyzer(
            model=self.model,
            X=self._get_feature_data(),
            y=self._get_target_data(),
            problem_type=self._problem_type,
            verbose=self.verbose
        )
        
        # Store baseline performance for reference
        self.baseline_performance = self._analyzer.baseline_performance
        
        if self.verbose:
            print(f"Baseline performance: {self.baseline_performance}")
    
    def test_mean_shift(self, feature_name: Optional[str] = None, levels: Union[List[float], float] = [0.1, 0.3, 0.5]) -> Dict[str, Any]:
        """
        Test robustness against mean shift in a feature distribution.
        
        This test shifts the mean of a feature's distribution by specific levels,
        measured in multiples of the feature's standard deviation.
        
        Parameters:
        -----------
        feature_name : str or None
            Name of the feature to shift (None for first feature)
        levels : list of float or float
            Levels of shift to apply (multiples of standard deviation)
            
        Returns:
        --------
        dict : Distribution shift test results
        """
        # Validate feature name
        feature_name = self._validate_feature(feature_name)
        
        # Convert levels to list if it's a single value
        if isinstance(levels, (int, float)):
            levels = [levels]
        
        if self.verbose:
            print(f"Testing mean shift on feature '{feature_name}' with levels {levels}")
        
        # Run the test
        results = self._analyzer.test_distribution_shift(
            feature_name=feature_name,
            shift_type='mean',
            shift_levels=levels
        )
        
        # Add metadata to results
        results['feature'] = feature_name
        results['shift_type'] = 'mean'
        results['baseline_performance'] = self.baseline_performance
        
        return results
    
    def test_variance_shift(self, feature_name: Optional[str] = None, levels: Union[List[float], float] = [0.1, 0.3, 0.5]) -> Dict[str, Any]:
        """
        Test robustness against variance shift in a feature distribution.
        
        This test changes the variance of a feature's distribution by specific factors.
        
        Parameters:
        -----------
        feature_name : str or None
            Name of the feature to shift (None for first feature)
        levels : list of float or float
            Levels of shift to apply (factors to multiply variance by)
            
        Returns:
        --------
        dict : Distribution shift test results
        """
        # Validate feature name
        feature_name = self._validate_feature(feature_name)
        
        # Convert levels to list if it's a single value
        if isinstance(levels, (int, float)):
            levels = [levels]
        
        if self.verbose:
            print(f"Testing variance shift on feature '{feature_name}' with levels {levels}")
        
        # Run the test
        results = self._analyzer.test_distribution_shift(
            feature_name=feature_name,
            shift_type='variance',
            shift_levels=levels
        )
        
        # Add metadata to results
        results['feature'] = feature_name
        results['shift_type'] = 'variance'
        results['baseline_performance'] = self.baseline_performance
        
        return results
    
    def test_skew_shift(self, feature_name: Optional[str] = None, levels: Union[List[float], float] = [0.1, 0.3, 0.5]) -> Dict[str, Any]:
        """
        Test robustness against skewness shift in a feature distribution.
        
        This test changes the skewness of a feature's distribution by applying
        exponential transformations of varying strengths.
        
        Parameters:
        -----------
        feature_name : str or None
            Name of the feature to shift (None for first feature)
        levels : list of float or float
            Levels of shift to apply (strength of skewness transformation)
            
        Returns:
        --------
        dict : Distribution shift test results
        """
        # Validate feature name
        feature_name = self._validate_feature(feature_name)
        
        # Convert levels to list if it's a single value
        if isinstance(levels, (int, float)):
            levels = [levels]
        
        if self.verbose:
            print(f"Testing skew shift on feature '{feature_name}' with levels {levels}")
        
        # Run the test
        results = self._analyzer.test_distribution_shift(
            feature_name=feature_name,
            shift_type='skew',
            shift_levels=levels
        )
        
        # Add metadata to results
        results['feature'] = feature_name
        results['shift_type'] = 'skew'
        results['baseline_performance'] = self.baseline_performance
        
        return results
    
    def compute_resilience_score(self, feature_names: Optional[List[str]] = None, shift_types: Optional[List[str]] = None, shift_level: float = 0.5) -> Dict[str, float]:
        """
        Compute an overall resilience score for the model.
        
        This function evaluates the model's resilience to different types
        of distribution shifts across multiple features.
        
        Parameters:
        -----------
        feature_names : list of str or None
            Names of features to include (None uses all)
        shift_types : list of str or None
            Types of shifts to include (None uses ['mean', 'variance', 'skew'])
        shift_level : float
            Level of shift to use for the score
            
        Returns:
        --------
        dict : Dictionary with resilience scores
        """
        if feature_names is None:
            feature_names = self._features
            
        if shift_types is None:
            shift_types = ['mean', 'variance', 'skew']
            
        if self.verbose:
            print(f"Computing resilience score for features: {feature_names}")
            print(f"Using shift types: {shift_types} at level {shift_level}")
        
        # Compute resilience score
        scores = self._analyzer.compute_resilience_score(
            feature_names=feature_names,
            shift_types=shift_types,
            shift_level=shift_level
        )
        
        return scores
    
    def plot_distribution_shift(self, feature_name: Optional[str] = None, shift_type: str = 'mean', figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot the effect of distribution shift on model performance.
        
        Parameters:
        -----------
        feature_name : str or None
            Name of the feature to plot (None for first feature)
        shift_type : str
            Type of shift to plot: 'mean', 'variance', or 'skew'
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.Figure : Shift impact plot
        """
        # Validate feature name
        feature_name = self._validate_feature(feature_name)
        
        # Validate shift type
        if shift_type not in ['mean', 'variance', 'skew']:
            raise ValueError(f"Shift type '{shift_type}' not supported. Must be 'mean', 'variance', or 'skew'")
        
        if self.verbose:
            print(f"Plotting {shift_type} shift for feature '{feature_name}'")
        
        # Run the test with multiple levels for better visualization
        levels = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
        
        if shift_type == 'mean':
            results = self.test_mean_shift(feature_name, levels)
        elif shift_type == 'variance':
            results = self.test_variance_shift(feature_name, levels)
        elif shift_type == 'skew':
            results = self.test_skew_shift(feature_name, levels)
        
        # Plot shift impact
        fig = self._analyzer.plot_distribution_shift(feature_name, shift_type, figsize)
        
        return fig
    
    def plot_resilience_scores(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot resilience scores for each feature.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.Figure : Resilience scores plot
        """
        if self.verbose:
            print("Plotting resilience scores for all features")
        
        # Compute resilience scores if not already done
        scores = self.compute_resilience_score()
        
        # Plot resilience scores
        fig = self._analyzer.plot_resilience_scores(figsize)
        
        return fig
    
    def compare_features(self, feature_names: List[str], shift_type: str = 'mean', level: float = 0.5, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Compare the impact of distribution shift across multiple features.
        
        Parameters:
        -----------
        feature_names : list of str
            Names of features to compare
        shift_type : str
            Type of shift to apply: 'mean', 'variance', or 'skew'
        level : float
            Level of shift to apply
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.Figure : Feature comparison plot
        """
        # Validate feature names
        for feature in feature_names:
            self._validate_feature(feature)
            
        # Validate shift type
        if shift_type not in ['mean', 'variance', 'skew']:
            raise ValueError(f"Shift type '{shift_type}' not supported. Must be 'mean', 'variance', or 'skew'")
        
        if self.verbose:
            print(f"Comparing {shift_type} shift at level {level} across features: {feature_names}")
        
        # Get primary metric based on problem type
        if self._problem_type == 'classification':
            metric = 'accuracy'
        else:
            metric = 'mse'
        
        # Run tests for each feature
        results = []
        for feature in feature_names:
            if shift_type == 'mean':
                result = self.test_mean_shift(feature, level)
            elif shift_type == 'variance':
                result = self.test_variance_shift(feature, level)
            elif shift_type == 'skew':
                result = self.test_skew_shift(feature, level)
                
            # Get performance metrics
            performance = result['performance'][0]
            rel_change = performance['relative_change'][metric]
            results.append((feature, rel_change))
        
        # Sort by impact (most negative change first)
        results.sort(key=lambda x: x[1])
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        features = [r[0] for r in results]
        changes = [r[1] for r in results]
        
        # Create horizontal bar chart
        y_pos = range(len(features))
        bars = ax.barh(y_pos, changes, align='center')
        
        # Color bars based on impact
        for i, bar in enumerate(bars):
            if changes[i] < 0:
                bar.set_color('red')
            else:
                bar.set_color('green')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()  # Labels read top-to-bottom
        
        # Add vertical line at 0
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        ax.set_xlabel(f'Relative Change in {metric.upper()}')
        ax.set_title(f'Impact of {shift_type.title()} Shift (Level {level}) Across Features')
        
        return fig
    
    def batch_test_features(self, n_top_features: int = 5, shift_types: Optional[List[str]] = None, level: float = 0.5) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Run batch distribution shift tests on top features.
        
        Parameters:
        -----------
        n_top_features : int
            Number of top features to test
        shift_types : list of str or None
            Types of shifts to apply (None uses ['mean', 'variance', 'skew'])
        level : float
            Level of shift to apply
            
        Returns:
        --------
        dict : Dictionary with test results for each feature and shift type
        """
        if shift_types is None:
            shift_types = ['mean', 'variance', 'skew']
            
        # Compute resilience score to identify important features
        resilience_scores = self.compute_resilience_score(shift_level=level)
        feature_scores = resilience_scores['feature_scores']
        
        # Sort features by importance (most vulnerable first - lower scores)
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1])
        top_features = [f[0] for f in sorted_features[:n_top_features]]
        
        if self.verbose:
            print(f"Running batch tests on top {n_top_features} features: {top_features}")
            print(f"Using shift types: {shift_types} at level {level}")
        
        # Get primary metric based on problem type
        if self._problem_type == 'classification':
            metric = 'accuracy'
        else:
            metric = 'mse'
        
        # Run tests for each feature and shift type
        results = {}
        
        for feature in top_features:
            feature_results = {}
            
            for shift_type in shift_types:
                if shift_type == 'mean':
                    result = self.test_mean_shift(feature, level)
                elif shift_type == 'variance':
                    result = self.test_variance_shift(feature, level)
                elif shift_type == 'skew':
                    result = self.test_skew_shift(feature, level)
                
                # Get performance metrics
                performance = result['performance'][0]
                metrics_results = performance['metrics']
                rel_change = performance['relative_change']
                
                feature_results[shift_type] = {
                    'metrics': metrics_results,
                    'relative_change': rel_change
                }
            
            results[feature] = feature_results
        
        return {
            'batch_results': results,
            'top_features': top_features,
            'resilience_scores': resilience_scores
        }