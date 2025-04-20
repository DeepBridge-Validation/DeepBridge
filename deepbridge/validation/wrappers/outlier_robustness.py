"""
Outlier robustness tests for model validation.

This module provides a wrapper for testing model robustness
against outliers and extreme values using DBDataset objects.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
import matplotlib.pyplot as plt

from .base_wrapper import BaseWrapper
from deepbridge.validation.frameworks.robustness.outlier_tester import OutlierRobustnessTester, detect_outliers


class OutlierRobustnessTests(BaseWrapper):
    """
    Tests for model robustness against outliers.
    
    This class provides methods for testing how well a model maintains
    its performance when dealing with outlier data points.
    """
    
    def __init__(self, dataset, verbose: bool = False):
        """
        Initialize the outlier robustness tests.
        
        Parameters:
        -----------
        dataset : DBDataset
            Dataset object containing training/test data and model
        verbose : bool
            Whether to print progress information
        """
        super().__init__(dataset, verbose)
        
        # Initialize the underlying outlier tester
        self._tester = OutlierRobustnessTester(
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
    
    def test_isolation_forest(self, contamination: float = 0.1, feature_subset: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Test robustness against outliers detected by Isolation Forest.
        
        Parameters:
        -----------
        contamination : float
            Expected proportion of outliers in the dataset (0-1)
        feature_subset : list of str or None
            Subset of features to use for outlier detection (None uses all)
            
        Returns:
        --------
        dict : Outlier robustness test results
        """
        # Validate feature subset if provided
        if feature_subset is not None:
            for feature in feature_subset:
                self._validate_feature(feature)
        
        if self.verbose:
            print(f"Testing robustness against outliers (Isolation Forest, contamination={contamination})")
            if feature_subset:
                print(f"Using feature subset: {feature_subset}")
        
        # A classe OutlierRobustnessTester usa o método de detecção de outliers como 
        # um parâmetro de inicialização, não como um parâmetro do método test_outlier_robustness
        
        # Verifica se temos que criar uma nova instância ou se a atual já usa Isolation Forest
        if not hasattr(self, '_current_detection_method') or self._current_detection_method != 'isolation_forest':
            # Armazena o método de detecção atual
            self._current_detection_method = 'isolation_forest'
            
            # Cria uma nova instância com o método correto
            self._tester = OutlierRobustnessTester(
                model=self.model,
                X=self._get_feature_data(),
                y=self._get_target_data(),
                problem_type=self._problem_type,
                outlier_detection_method='isolation_forest',
                verbose=self.verbose
            )
        
        # Agora execute o teste sem passar o método como parâmetro
        results = self._tester.test_outlier_robustness(
            contamination=contamination,
            feature_subset=feature_subset
        )
        
        # Add metadata to results
        results['outlier_detection_method'] = 'isolation_forest'
        results['contamination'] = contamination
        results['feature_subset'] = feature_subset
        results['baseline_performance'] = self.baseline_performance
        
        return results
    
    def test_lof(self, contamination: float = 0.1, feature_subset: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Test robustness against outliers detected by Local Outlier Factor.
        
        Parameters:
        -----------
        contamination : float
            Expected proportion of outliers in the dataset (0-1)
        feature_subset : list of str or None
            Subset of features to use for outlier detection (None uses all)
            
        Returns:
        --------
        dict : Outlier robustness test results
        """
        # Validate feature subset if provided
        if feature_subset is not None:
            for feature in feature_subset:
                self._validate_feature(feature)
        
        if self.verbose:
            print(f"Testing robustness against outliers (Local Outlier Factor, contamination={contamination})")
            if feature_subset:
                print(f"Using feature subset: {feature_subset}")
        
        if not hasattr(self, '_current_detection_method') or self._current_detection_method != 'lof':
            self._current_detection_method = 'lof'
            
            self._tester = OutlierRobustnessTester(
                model=self.model,
                X=self._get_feature_data(),
                y=self._get_target_data(),
                problem_type=self._problem_type,
                outlier_detection_method='lof',
                verbose=self.verbose
            )
        
        results = self._tester.test_outlier_robustness(
            contamination=contamination,
            feature_subset=feature_subset
        )
        
        # Add metadata to results
        results['outlier_detection_method'] = 'lof'
        results['contamination'] = contamination
        results['feature_subset'] = feature_subset
        results['baseline_performance'] = self.baseline_performance
        
        return results
    
    def test_quantile(self, contamination: float = 0.1, feature_subset: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Test robustness against outliers detected by quantile-based method.
        
        Parameters:
        -----------
        contamination : float
            Expected proportion of outliers in the dataset (0-1)
        feature_subset : list of str or None
            Subset of features to use for outlier detection (None uses all)
            
        Returns:
        --------
        dict : Outlier robustness test results
        """
        # Validate feature subset if provided
        if feature_subset is not None:
            for feature in feature_subset:
                self._validate_feature(feature)
        
        if self.verbose:
            print(f"Testing robustness against outliers (Quantile-based, contamination={contamination})")
            if feature_subset:
                print(f"Using feature subset: {feature_subset}")
        
        # Verifica se temos que criar uma nova instância ou se a atual já usa quantile
        if not hasattr(self, '_current_detection_method') or self._current_detection_method != 'quantile':
            # Armazena o método de detecção atual
            self._current_detection_method = 'quantile'
            
            # Cria uma nova instância com o método correto
            self._tester = OutlierRobustnessTester(
                model=self.model,
                X=self._get_feature_data(),
                y=self._get_target_data(),
                problem_type=self._problem_type,
                outlier_detection_method='quantile',
                verbose=self.verbose
            )
        
        # Agora execute o teste sem passar o método como parâmetro
        results = self._tester.test_outlier_robustness(
            contamination=contamination,
            feature_subset=feature_subset
        )
        
        # Add metadata to results
        results['outlier_detection_method'] = 'quantile'
        results['contamination'] = contamination
        results['feature_subset'] = feature_subset
        results['baseline_performance'] = self.baseline_performance
        
        return results
    
    def detect_outliers(self, method: str = 'isolation_forest', contamination: float = 0.1, feature_subset: Optional[List[str]] = None) -> Tuple[np.ndarray, Any]:
        """
        Detect outliers in the dataset.
        
        Parameters:
        -----------
        method : str
            Outlier detection method: 'isolation_forest', 'lof', or 'quantile'
        contamination : float
            Expected proportion of outliers in the dataset (0-1)
        feature_subset : list of str or None
            Subset of features to use for outlier detection (None uses all)
            
        Returns:
        --------
        tuple : (outlier_mask, detector)
            - outlier_mask: Boolean array where True indicates outlier
            - detector: Trained outlier detection model (None for quantile method)
        """
        # Validate feature subset if provided
        if feature_subset is not None:
            for feature in feature_subset:
                self._validate_feature(feature)
        
        if self.verbose:
            print(f"Detecting outliers ({method}, contamination={contamination})")
            if feature_subset:
                print(f"Using feature subset: {feature_subset}")
        
        # Get feature data
        X = self._get_feature_data()
        
        # Filter to feature subset if provided
        if feature_subset is not None:
            X = X[feature_subset]
        
        # Detect outliers
        outlier_mask, detector = detect_outliers(
            X=X,
            method=method,
            contamination=contamination
        )
        
        return outlier_mask, detector
    
    def analyze_outliers(self, method: str = 'isolation_forest', contamination: float = 0.1, feature_subset: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze outliers and their characteristics.
        
        Parameters:
        -----------
        method : str
            Outlier detection method: 'isolation_forest', 'lof', or 'quantile'
        contamination : float
            Expected proportion of outliers in the dataset (0-1)
        feature_subset : list of str or None
            Subset of features to use for outlier detection (None uses all)
            
        Returns:
        --------
        dict : Outlier analysis results
        """
        # Detect outliers
        outlier_mask, detector = self.detect_outliers(
            method=method,
            contamination=contamination,
            feature_subset=feature_subset
        )
        
        # Get data
        X = self._get_feature_data()
        y = self._get_target_data()
        
        # Split data into outliers and inliers
        X_outliers = X[outlier_mask]
        y_outliers = y[outlier_mask]
        
        X_inliers = X[~outlier_mask]
        y_inliers = y[~outlier_mask]
        
        # Calculate statistics for each feature
        outlier_stats = {}
        for feature in self._features:
            outlier_values = X_outliers[feature]
            inlier_values = X_inliers[feature]
            
            outlier_stats[feature] = {
                'mean_outliers': float(outlier_values.mean()),
                'mean_inliers': float(inlier_values.mean()),
                'std_outliers': float(outlier_values.std()),
                'std_inliers': float(inlier_values.std()),
                'min_outliers': float(outlier_values.min()),
                'min_inliers': float(inlier_values.min()),
                'max_outliers': float(outlier_values.max()),
                'max_inliers': float(inlier_values.max()),
                'mean_ratio': float(outlier_values.mean() / inlier_values.mean()) if inlier_values.mean() != 0 else float('inf'),
                'std_ratio': float(outlier_values.std() / inlier_values.std()) if inlier_values.std() != 0 else float('inf')
            }
        
        # Calculate target distribution for classification
        target_distribution = {}
        if self.is_classification():
            unique_y, counts_y = np.unique(y, return_counts=True)
            unique_y_outliers, counts_y_outliers = np.unique(y_outliers, return_counts=True)
            unique_y_inliers, counts_y_inliers = np.unique(y_inliers, return_counts=True)
            
            # Convert to dictionaries for easier comparison
            total_dist = {unique_y[i]: counts_y[i] / len(y) for i in range(len(unique_y))}
            outlier_dist = {unique_y_outliers[i]: counts_y_outliers[i] / len(y_outliers) 
                           for i in range(len(unique_y_outliers))}
            inlier_dist = {unique_y_inliers[i]: counts_y_inliers[i] / len(y_inliers) 
                          for i in range(len(unique_y_inliers))}
            
            target_distribution = {
                'total': total_dist,
                'outliers': outlier_dist,
                'inliers': inlier_dist
            }
        
        return {
            'outlier_detection_method': method,
            'contamination': contamination,
            'feature_subset': feature_subset,
            'n_outliers': int(np.sum(outlier_mask)),
            'n_inliers': int(np.sum(~outlier_mask)),
            'outlier_ratio': float(np.mean(outlier_mask)),
            'feature_stats': outlier_stats,
            'target_distribution': target_distribution
        }
    
    def plot_outlier_comparison(self, method: str = 'isolation_forest', contamination: float = 0.1, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot comparison of model performance on outliers vs. inliers.
        
        Parameters:
        -----------
        method : str
            Outlier detection method: 'isolation_forest', 'lof', or 'quantile'
        contamination : float
            Expected proportion of outliers in the dataset (0-1)
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.Figure : Comparison plot
        """
        # Run outlier test
        if method == 'isolation_forest':
            results = self.test_isolation_forest(contamination)
        elif method == 'lof':
            results = self.test_lof(contamination)
        elif method == 'quantile':
            results = self.test_quantile(contamination)
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        # Create plot
        fig = self._tester.plot_outlier_comparison(figsize)
        
        return fig
    
    def plot_outlier_distribution(self, feature_name: Optional[str] = None, method: str = 'isolation_forest', contamination: float = 0.1, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot distribution of outliers vs. inliers for a feature.
        
        Parameters:
        -----------
        feature_name : str or None
            Name of feature to plot (None plots first two features as scatter)
        method : str
            Outlier detection method: 'isolation_forest', 'lof', or 'quantile'
        contamination : float
            Expected proportion of outliers in the dataset (0-1)
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.Figure : Distribution plot
        """
        # Run outlier test
        if method == 'isolation_forest':
            results = self.test_isolation_forest(contamination)
        elif method == 'lof':
            results = self.test_lof(contamination)
        elif method == 'quantile':
            results = self.test_quantile(contamination)
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        # Create plot
        fig = self._tester.plot_outlier_distribution(feature_name, figsize)
        
        return fig
    
    def compare_methods(self, contamination: float = 0.1, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Compare different outlier detection methods.
        
        Parameters:
        -----------
        contamination : float
            Expected proportion of outliers in the dataset (0-1)
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.Figure : Methods comparison plot
        """
        if self.verbose:
            print(f"Comparing outlier detection methods (contamination={contamination})")
        
        # Run tests for each method
        methods = ['isolation_forest', 'lof', 'quantile']
        results = {}
        
        for method in methods:
            if method == 'isolation_forest':
                results[method] = self.test_isolation_forest(contamination)
            elif method == 'lof':
                results[method] = self.test_lof(contamination)
            elif method == 'quantile':
                results[method] = self.test_quantile(contamination)
        
        # Get primary metric based on problem type
        if self._problem_type == 'classification':
            metric = 'accuracy'
        else:
            metric = 'mse'
        
        # Extract metrics for comparison
        method_metrics = {
            'Method': [],
            'Outlier Performance': [],
            'Inlier Performance': [],
            'Relative Change': []
        }
        
        for method, result in results.items():
            method_metrics['Method'].append(method)
            method_metrics['Outlier Performance'].append(result['outlier_performance'][metric])
            method_metrics['Inlier Performance'].append(result['inlier_performance'][metric])
            
            # Calculate relative change
            outlier_perf = result['outlier_performance'][metric]
            inlier_perf = result['inlier_performance'][metric]
            rel_change = (outlier_perf - inlier_perf) / inlier_perf if inlier_perf != 0 else 0
            method_metrics['Relative Change'].append(rel_change)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Position of bars
        x = np.arange(len(methods))
        width = 0.35
        
        # Create bars for inlier and outlier performance
        ax.bar(x - width/2, method_metrics['Inlier Performance'], width, label='Inliers')
        ax.bar(x + width/2, method_metrics['Outlier Performance'], width, label='Outliers')
        
        # Set labels and title
        ax.set_xlabel('Outlier Detection Method')
        ax.set_ylabel(f'{metric.upper()} Performance')
        ax.set_title(f'Model Performance on Outliers vs. Inliers by Detection Method')
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.legend()
        
        # Add relative change as text
        for i, method in enumerate(methods):
            rel_change = method_metrics['Relative Change'][i]
            color = 'red' if rel_change < 0 else 'green'
            ax.annotate(f'{rel_change:.2%}',
                      xy=(i, min(method_metrics['Inlier Performance'][i], method_metrics['Outlier Performance'][i])),
                      xytext=(0, -20),
                      textcoords='offset points',
                      ha='center',
                      color=color)
        
        return fig
    
    def find_outlier_features(self, method: str = 'isolation_forest', contamination: float = 0.1, threshold: float = 1.5) -> Dict[str, float]:
        """
        Find features that most contribute to outlier detection.
        
        Parameters:
        -----------
        method : str
            Outlier detection method: 'isolation_forest', 'lof', or 'quantile'
        contamination : float
            Expected proportion of outliers in the dataset (0-1)
        threshold : float
            Threshold for considering a feature important for outlier detection
            
        Returns:
        --------
        dict : Dictionary mapping feature names to importance scores
        """
        # Analyze outliers
        analysis = self.analyze_outliers(method, contamination)
        
        # Calculate feature importance based on how different outliers are from inliers
        feature_importance = {}
        
        for feature, stats in analysis['feature_stats'].items():
            # Normalize by standard deviation to get z-score-like measure
            if stats['std_inliers'] > 0:
                mean_diff_normalized = abs(stats['mean_outliers'] - stats['mean_inliers']) / stats['std_inliers']
            else:
                mean_diff_normalized = 0
                
            # Also consider variance ratio
            std_ratio = stats['std_ratio'] if stats['std_ratio'] != float('inf') else 5.0
            std_ratio = min(std_ratio, 5.0)  # Cap at 5x
            
            # Combine metrics
            importance = mean_diff_normalized * (1 + np.log1p(std_ratio))
            
            feature_importance[feature] = float(importance)
        
        # Filter by threshold
        important_features = {f: score for f, score in feature_importance.items() if score >= threshold}
        
        # Sort by importance (descending)
        important_features = dict(sorted(important_features.items(), key=lambda x: x[1], reverse=True))
        
        return important_features