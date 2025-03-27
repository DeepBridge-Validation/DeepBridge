"""
Feature perturbation tests for model validation.

This module provides a wrapper for testing model robustness
against various types of feature perturbations using DBDataset objects.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
import matplotlib.pyplot as plt

from .base_wrapper import BaseWrapper
from deepbridge.validation.frameworks.robustness.robustness_tester import RobustnessTester
from deepbridge.validation.frameworks.robustness.perturbation.gaussian_perturbation import apply_gaussian_noise
from deepbridge.validation.frameworks.robustness.perturbation.salt_pepper_perturbation import apply_salt_pepper_noise
from deepbridge.validation.frameworks.robustness.perturbation.quantile_perturbation import apply_feature_perturbation


class FeaturePerturbationTests(BaseWrapper):
    """
    Tests for model robustness against feature perturbations.
    
    This class provides methods for testing how well a model maintains
    its performance when input features are perturbed in various ways.
    """
    
    def __init__(self, dataset, verbose: bool = False):
        """
        Initialize the feature perturbation tests.
        
        Parameters:
        -----------
        dataset : DBDataset
            Dataset object containing training/test data and model
        verbose : bool
            Whether to print progress information
        """
        super().__init__(dataset, verbose)
        
        # Initialize the underlying robustness tester
        self._tester = RobustnessTester(
            model=self.model,
            X=self._get_feature_data(),
            y=self._get_target_data(),
            problem_type=self._problem_type,
            verbose=self.verbose
        )
        
        # Store baseline performance for reference
        self.baseline_performance = self._tester.baseline_performance
        
        if self.verbose:
            print(f"Baseline performance: {self.baseline_performance}")
    
    def test_noise(self, feature_name: Optional[str] = None, level: float = 0.2) -> Dict[str, Any]:
        """
        Test robustness against Gaussian noise perturbation.
        
        Parameters:
        -----------
        feature_name : str or None
            Name of the feature to perturb (None for first feature)
        level : float
            Level of perturbation to apply (0-1)
            
        Returns:
        --------
        dict : Perturbation test results
        """
        # Validate feature name
        feature_name = self._validate_feature(feature_name)
        
        if self.verbose:
            print(f"Testing noise perturbation on feature '{feature_name}' at level {level}")
        
        # Run the test
        results = self._tester.test_feature_perturbation(
            feature_name=feature_name,
            perturbation_type='noise',
            perturbation_level=level
        )
        
        # Add metadata to results
        results['feature'] = feature_name
        results['perturbation_type'] = 'noise'
        results['baseline_performance'] = self.baseline_performance
        
        return results
    
    def test_zero(self, feature_name: Optional[str] = None, level: float = 0.2) -> Dict[str, Any]:
        """
        Test robustness against zeroing feature values.
        
        Parameters:
        -----------
        feature_name : str or None
            Name of the feature to perturb (None for first feature)
        level : float
            Level of perturbation to apply (0-1) - proportion of values to zero
            
        Returns:
        --------
        dict : Perturbation test results
        """
        # Validate feature name
        feature_name = self._validate_feature(feature_name)
        
        if self.verbose:
            print(f"Testing zero perturbation on feature '{feature_name}' at level {level}")
        
        # Run the test
        results = self._tester.test_feature_perturbation(
            feature_name=feature_name,
            perturbation_type='zero',
            perturbation_level=level
        )
        
        # Add metadata to results
        results['feature'] = feature_name
        results['perturbation_type'] = 'zero'
        results['baseline_performance'] = self.baseline_performance
        
        return results
    
    def test_flip(self, feature_name: Optional[str] = None, level: float = 0.2) -> Dict[str, Any]:
        """
        Test robustness against flipping binary feature values.
        
        Parameters:
        -----------
        feature_name : str or None
            Name of the feature to perturb (None for first binary feature)
        level : float
            Level of perturbation to apply (0-1) - proportion of values to flip
            
        Returns:
        --------
        dict : Perturbation test results
        """
        # If feature_name is None, find a binary feature
        if feature_name is None:
            binary_features = self._find_binary_features()
            if not binary_features:
                raise ValueError("No binary features found in the dataset")
            feature_name = binary_features[0]
        else:
            # Validate that the feature is binary
            X = self._get_feature_data()
            unique_values = X[feature_name].nunique()
            if unique_values > 2:
                raise ValueError(f"Feature '{feature_name}' is not binary (has {unique_values} unique values)")
        
        if self.verbose:
            print(f"Testing flip perturbation on feature '{feature_name}' at level {level}")
        
        # Run the test
        results = self._tester.test_feature_perturbation(
            feature_name=feature_name,
            perturbation_type='flip',
            perturbation_level=level
        )
        
        # Add metadata to results
        results['feature'] = feature_name
        results['perturbation_type'] = 'flip'
        results['baseline_performance'] = self.baseline_performance
        
        return results
    
    def test_quantile(self, feature_name: Optional[str] = None, level: float = 0.2) -> Dict[str, Any]:
        """
        Test robustness against moving values to their quantiles.
        
        Parameters:
        -----------
        feature_name : str or None
            Name of the feature to perturb (None for first feature)
        level : float
            Level of perturbation to apply (0-1)
            
        Returns:
        --------
        dict : Perturbation test results
        """
        # Validate feature name
        feature_name = self._validate_feature(feature_name)
        
        if self.verbose:
            print(f"Testing quantile perturbation on feature '{feature_name}' at level {level}")
        
        # Run the test
        results = self._tester.test_feature_perturbation(
            feature_name=feature_name,
            perturbation_type='quantile',
            perturbation_level=level
        )
        
        # Add metadata to results
        results['feature'] = feature_name
        results['perturbation_type'] = 'quantile'
        results['baseline_performance'] = self.baseline_performance
        
        return results
    
    def test_missing(self, feature_name: Optional[str] = None, level: float = 0.2) -> Dict[str, Any]:
        """
        Test robustness against missing values (simulated by using mean).
        
        Parameters:
        -----------
        feature_name : str or None
            Name of the feature to perturb (None for first feature)
        level : float
            Level of perturbation to apply (0-1) - proportion of values to replace
            
        Returns:
        --------
        dict : Perturbation test results
        """
        # Validate feature name
        feature_name = self._validate_feature(feature_name)
        
        if self.verbose:
            print(f"Testing missing value perturbation on feature '{feature_name}' at level {level}")
        
        # Run the test - use 'quantile' with custom implementation to simulate missing values
        X_test = self._get_feature_data()
        y_test = self._get_target_data()
        
        # Create perturbed data
        X_perturbed = X_test.copy()
        feature_values = X_test[feature_name].values
        
        # Replace random values with mean
        mask = np.random.random(len(feature_values)) < level
        mean_value = np.mean(feature_values)
        X_perturbed.loc[mask, feature_name] = mean_value
        
        # Evaluate model on perturbed data
        y_pred = self.model.predict(X_perturbed)
        
        # Calculate metrics based on problem type
        if self._problem_type == 'classification':
            from sklearn.metrics import accuracy_score, f1_score
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred, average='weighted')
            }
        else:
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred)
            }
        
        # Calculate relative change
        relative_change = {}
        for metric, value in metrics.items():
            if metric in self.baseline_performance:
                baseline = self.baseline_performance[metric]
                if baseline != 0:
                    relative_change[metric] = (value - baseline) / abs(baseline)
                else:
                    relative_change[metric] = 0.0
        
        # Format results to match RobustnessTester output
        results = {
            'feature': feature_name,
            'perturbation_type': 'missing',
            'perturbation_level': level,
            'performance': [{
                'level': level,
                'metrics': metrics,
                'relative_change': relative_change
            }],
            'baseline_performance': self.baseline_performance
        }
        
        return results
    
    def test_salt_pepper(self, feature_name: Optional[str] = None, level: float = 0.1, salt_prob: float = 0.5) -> Dict[str, Any]:
        """
        Test robustness against salt and pepper noise.
        
        Parameters:
        -----------
        feature_name : str or None
            Name of the feature to perturb (None for first feature)
        level : float
            Level of perturbation to apply (0-1) - proportion of values to perturb
        salt_prob : float
            Probability of salt vs. pepper (0-1)
            
        Returns:
        --------
        dict : Perturbation test results
        """
        # Validate feature name
        feature_name = self._validate_feature(feature_name)
        
        if self.verbose:
            print(f"Testing salt & pepper noise on feature '{feature_name}' at level {level}")
        
        # Get data
        X_test = self._get_feature_data()
        y_test = self._get_target_data()
        
        # Create perturbed data
        X_perturbed = X_test.copy()
        feature_values = X_test[feature_name].values
        
        # Determine max and min values
        max_val = np.max(feature_values)
        min_val = np.min(feature_values)
        
        # Generate mask for values to perturb
        mask = np.random.random(len(feature_values)) < level
        
        # Generate salt or pepper mask
        salt_mask = np.random.random(len(feature_values)) < salt_prob
        
        # Apply salt (max value)
        X_perturbed.loc[mask & salt_mask, feature_name] = max_val
        
        # Apply pepper (min value)
        X_perturbed.loc[mask & (~salt_mask), feature_name] = min_val
        
        # Evaluate model on perturbed data
        y_pred = self.model.predict(X_perturbed)
        
        # Calculate metrics based on problem type
        if self._problem_type == 'classification':
            from sklearn.metrics import accuracy_score, f1_score
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred, average='weighted')
            }
        else:
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred)
            }
        
        # Calculate relative change
        relative_change = {}
        for metric, value in metrics.items():
            if metric in self.baseline_performance:
                baseline = self.baseline_performance[metric]
                if baseline != 0:
                    relative_change[metric] = (value - baseline) / abs(baseline)
                else:
                    relative_change[metric] = 0.0
        
        # Format results to match RobustnessTester output
        results = {
            'feature': feature_name,
            'perturbation_type': 'salt_pepper',
            'perturbation_level': level,
            'salt_prob': salt_prob,
            'performance': [{
                'level': level,
                'metrics': metrics,
                'relative_change': relative_change
            }],
            'baseline_performance': self.baseline_performance
        }
        
        return results
    
    def test_all_features(self, perturbation_type: str = 'noise', level: float = 0.2) -> Dict[str, Any]:
        """
        Test perturbation on all features.
        
        Parameters:
        -----------
        perturbation_type : str
            Type of perturbation to apply
        level : float
            Level of perturbation to apply
            
        Returns:
        --------
        dict : Perturbation test results for all features
        """
        if self.verbose:
            print(f"Testing {perturbation_type} perturbation on all features at level {level}")
        
        # Run the test
        results = self._tester.test_all_features(
            perturbation_type=perturbation_type,
            perturbation_level=level
        )
        
        return results
    
    def plot_feature_impact(self, feature_name: Optional[str] = None, perturbation_type: str = 'noise', figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot impact of perturbation on a feature.
        
        Parameters:
        -----------
        feature_name : str or None
            Name of the feature to plot (None for first feature)
        perturbation_type : str
            Type of perturbation to plot
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.Figure : Impact plot
        """
        # Validate feature name
        feature_name = self._validate_feature(feature_name)
        
        # Ensure we have test results for this feature and perturbation type
        if perturbation_type not in ['noise', 'zero', 'flip', 'quantile']:
            raise ValueError(f"Perturbation type '{perturbation_type}' not supported for plotting")
        
        # Run test with multiple levels to get better plot
        levels = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
        results = []
        
        for level in levels:
            if perturbation_type == 'noise':
                result = self.test_noise(feature_name, level)
            elif perturbation_type == 'zero':
                result = self.test_zero(feature_name, level)
            elif perturbation_type == 'flip':
                result = self.test_flip(feature_name, level)
            elif perturbation_type == 'quantile':
                result = self.test_quantile(feature_name, level)
            
            results.append(result)
        
        # Extract data for plotting
        if self._problem_type == 'classification':
            metric = 'accuracy'
        else:
            metric = 'mse'
        
        plot_levels = levels
        rel_changes = [r['performance'][0]['relative_change'][metric] for r in results]
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot relative change
        ax.plot(plot_levels, rel_changes, 'o-', linewidth=2, markersize=8)
        
        # Add horizontal line at 0
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Set labels and title
        ax.set_xlabel(f'{perturbation_type.title()} Perturbation Level')
        ax.set_ylabel(f'Relative Change in {metric.upper()}')
        ax.set_title(f'Impact of {perturbation_type.title()} Perturbation on {feature_name}')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_feature_importance(self, perturbation_type: str = 'noise', level: float = 0.2, top_n: int = 10, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot feature importance based on robustness.
        
        Parameters:
        -----------
        perturbation_type : str
            Type of perturbation to use
        level : float
            Level of perturbation to apply
        top_n : int
            Number of top features to show
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.Figure : Feature importance plot
        """
        # Run test_all_features to get importance
        results = self.test_all_features(perturbation_type, level)
        
        if 'feature_importance' not in results:
            raise ValueError("Feature importance not found in results")
        
        importance = results['feature_importance']
        
        # Sort features by importance
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        # Limit to top N features
        if top_n and len(sorted_features) > top_n:
            sorted_features = sorted_features[:top_n]
        
        features = [f[0] for f in sorted_features]
        importance_values = [f[1] for f in sorted_features]
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create horizontal bar chart
        y_pos = range(len(features))
        ax.barh(y_pos, importance_values, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Robustness Importance')
        ax.set_title(f'Feature Importance Based on {perturbation_type.title()} Perturbation')
        
        return fig
    
    def _find_binary_features(self) -> List[str]:
        """
        Find binary features in the dataset.
        
        Returns:
        --------
        list of str : Names of binary features
        """
        X = self._get_feature_data()
        binary_features = []
        
        for feature in self._features:
            unique_values = X[feature].nunique()
            if unique_values == 2:
                binary_features.append(feature)
                
        return binary_features