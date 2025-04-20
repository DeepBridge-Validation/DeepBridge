"""
Outlier robustness testing.

This module provides tools for testing model robustness against
outliers and extreme values in the data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


class OutlierRobustnessTester:
    """
    Test model robustness against outliers and extreme values.
    
    This class provides methods for evaluating how well a model
    performs on outlier data points.
    """
    
    def __init__(
        self,
        model: Any,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        feature_names: Optional[List[str]] = None,
        problem_type: str = 'auto',
        metrics: Optional[Dict[str, Callable]] = None,
        outlier_detection_method: str = 'isolation_forest',
        verbose: bool = False
    ):
        """
        Initialize the outlier robustness tester.
        
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
        outlier_detection_method : str
            Method for outlier detection:
            - 'isolation_forest': Use Isolation Forest algorithm
            - 'lof': Use Local Outlier Factor
            - 'quantile': Use simple quantile-based detection
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
        self.outlier_detection_method = outlier_detection_method
        self.verbose = verbose
        self.results = {}
        
        # Get baseline performance
        self.baseline_performance = self.base_tester.baseline_performance
    
    def detect_outliers(
        self,
        contamination: float = 0.05,
        feature_subset: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Detect outliers in the dataset.
        
        Parameters:
        -----------
        contamination : float
            Expected proportion of outliers in the dataset
        feature_subset : list of str or None
            Subset of features to use for outlier detection
            
        Returns:
        --------
        numpy.ndarray : Boolean mask for outlier samples
        """
        if feature_subset is not None:
            # Use only the specified features
            X_subset = self.X[feature_subset]
        else:
            X_subset = self.X
            
        # Detect outliers using the specified method
        if self.outlier_detection_method == 'isolation_forest':
            detector = IsolationForest(
                contamination=contamination,
                random_state=42
            )
            # -1 for outliers, 1 for inliers
            outlier_labels = detector.fit_predict(X_subset)
            outlier_mask = outlier_labels == -1
            
        elif self.outlier_detection_method == 'lof':
            detector = LocalOutlierFactor(
                n_neighbors=20,
                contamination=contamination
            )
            # -1 for outliers, 1 for inliers
            outlier_labels = detector.fit_predict(X_subset)
            outlier_mask = outlier_labels == -1
            
        elif self.outlier_detection_method == 'quantile':
            # Simple approach: detect values outside of quantile range
            lower_quantile = contamination / 2
            upper_quantile = 1 - lower_quantile
            
            # Initialize mask
            outlier_mask = np.zeros(len(self.X), dtype=bool)
            
            # For each feature, mark samples outside the quantile range
            for feature in X_subset.columns:
                values = X_subset[feature].values
                q_low = np.quantile(values, lower_quantile)
                q_high = np.quantile(values, upper_quantile)
                
                # Mark as outlier if outside the range
                feature_outliers = (values < q_low) | (values > q_high)
                outlier_mask |= feature_outliers
                
        else:
            raise ValueError(f"Unknown outlier detection method: {self.outlier_detection_method}")
            
        if self.verbose:
            n_outliers = np.sum(outlier_mask)
            print(f"Detected {n_outliers} outliers ({n_outliers/len(self.X):.2%})")
            
        return outlier_mask
    
    def test_outlier_robustness(
        self,
        contamination: float = 0.05,
        feature_subset: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Test model robustness against outliers.
        
        Parameters:
        -----------
        contamination : float
            Expected proportion of outliers in the dataset
        feature_subset : list of str or None
            Subset of features to use for outlier detection
            
        Returns:
        --------
        dict : Dictionary with outlier robustness test results
        """
        # Detect outliers
        outlier_mask = self.detect_outliers(
            contamination=contamination,
            feature_subset=feature_subset
        )
        
        # Inlier mask is the opposite of outlier mask
        inlier_mask = ~outlier_mask
        
        # Separate data into outliers and inliers
        X_outliers = self.X[outlier_mask]
        y_outliers = self.y[outlier_mask]
        
        X_inliers = self.X[inlier_mask]
        y_inliers = self.y[inlier_mask]
        
        # Evaluate model on outliers and inliers
        if len(X_outliers) > 0:
            outlier_performance = self.base_tester._evaluate_model(X_outliers, y_outliers)
        else:
            outlier_performance = {metric: np.nan for metric in self.metrics}
            
        inlier_performance = self.base_tester._evaluate_model(X_inliers, y_inliers)
        
        # Calculate relative change in performance
        relative_change = {}
        for metric in self.metrics:
            if metric in outlier_performance and metric in inlier_performance:
                relative_change[metric] = (
                    (outlier_performance[metric] - inlier_performance[metric]) 
                    / abs(inlier_performance[metric]) if inlier_performance[metric] != 0 else float('inf')
                )
        
        results = {
            'outlier_detection_method': self.outlier_detection_method,
            'contamination': contamination,
            'feature_subset': feature_subset,
            'n_outliers': np.sum(outlier_mask),
            'n_inliers': np.sum(inlier_mask),
            'outlier_performance': outlier_performance,
            'inlier_performance': inlier_performance,
            'relative_change': relative_change,
            'outlier_mask': outlier_mask
        }
        
        # Store results
        result_key = f"outlier_test_{self.outlier_detection_method}_{contamination}"
        self.results[result_key] = results
        
        if self.verbose:
            print(f"Outlier performance: {outlier_performance}")
            print(f"Inlier performance: {inlier_performance}")
            print(f"Relative change: {relative_change}")
            
        return results
    
    def plot_outlier_comparison(self, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot comparison of model performance on outliers vs. inliers.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
            
        Returns:
        --------
        matplotlib.Figure : Comparison plot
        """
        # Find the most recent outlier test result
        outlier_test_keys = [k for k in self.results.keys() if k.startswith('outlier_test_')]
        
        if not outlier_test_keys:
            raise ValueError("No outlier test results found")
            
        result_key = outlier_test_keys[-1]
        results = self.results[result_key]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        metrics = list(self.metrics.keys())
        x = np.arange(len(metrics))
        width = 0.35

# Get performance values
        inlier_values = [results['inlier_performance'][m] for m in metrics]
        outlier_values = [results['outlier_performance'][m] for m in metrics]
        
        # Create grouped bar chart
        rects1 = ax.bar(x - width/2, inlier_values, width, label='Inliers')
        rects2 = ax.bar(x + width/2, outlier_values, width, label='Outliers')
        
        # Add labels and legend
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Performance')
        ax.set_title('Model Performance: Outliers vs. Inliers')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        
        # Add value labels on bars
        for rect in rects1:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
                        
        for rect in rects2:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def plot_outlier_distribution(
        self,
        feature_name: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Plot distribution of outliers vs. inliers.
        
        Parameters:
        -----------
        feature_name : str or None
            Name of feature to plot (if None, plot first two features)
        figsize : tuple
            Figure size (width, height)
            
        Returns:
        --------
        matplotlib.Figure : Distribution plot
        """
        # Find the most recent outlier test result
        outlier_test_keys = [k for k in self.results.keys() if k.startswith('outlier_test_')]
        
        if not outlier_test_keys:
            raise ValueError("No outlier test results found")
            
        result_key = outlier_test_keys[-1]
        results = self.results[result_key]
        
        outlier_mask = results['outlier_mask']
        
        if feature_name is not None:
            # Plot single feature distribution
            if feature_name not in self.feature_names:
                raise ValueError(f"Feature {feature_name} not found")
                
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot histograms for inliers and outliers
            inlier_values = self.X.loc[~outlier_mask, feature_name]
            outlier_values = self.X.loc[outlier_mask, feature_name]
            
            ax.hist(inlier_values, bins=30, alpha=0.5, label='Inliers')
            ax.hist(outlier_values, bins=30, alpha=0.5, label='Outliers')
            
            ax.set_xlabel(feature_name)
            ax.set_ylabel('Count')
            ax.set_title(f'Distribution of {feature_name}: Inliers vs. Outliers')
            ax.legend()
            
        else:
            # Plot scatter of first two features
            if len(self.feature_names) < 2:
                raise ValueError("Need at least two features for scatter plot")
                
            fig, ax = plt.subplots(figsize=figsize)
            
            # Get first two features
            feature1 = self.feature_names[0]
            feature2 = self.feature_names[1]
            
            # Scatter plot for inliers and outliers
            ax.scatter(
                self.X.loc[~outlier_mask, feature1],
                self.X.loc[~outlier_mask, feature2],
                alpha=0.5, label='Inliers'
            )
            ax.scatter(
                self.X.loc[outlier_mask, feature1],
                self.X.loc[outlier_mask, feature2],
                alpha=0.5, label='Outliers', c='r'
            )
            
            ax.set_xlabel(feature1)
            ax.set_ylabel(feature2)
            ax.set_title(f'Scatter Plot: Inliers vs. Outliers')
            ax.legend()
        
        plt.tight_layout()
        return fig


def detect_outliers(
    X: Union[np.ndarray, pd.DataFrame],
    method: str = 'isolation_forest',
    contamination: float = 0.05,
    feature_names: Optional[List[str]] = None
) -> Tuple[np.ndarray, Any]:
    """
    Detect outliers in a dataset.
    
    Parameters:
    -----------
    X : array-like or DataFrame
        Input features
    method : str
        Outlier detection method:
        - 'isolation_forest': Use Isolation Forest algorithm
        - 'lof': Use Local Outlier Factor
        - 'quantile': Use simple quantile-based detection
    contamination : float
        Expected proportion of outliers in the dataset
    feature_names : list of str or None
        Names of features (required if X is not a DataFrame)
        
    Returns:
    --------
    tuple : (outlier_mask, detector)
        - outlier_mask: Boolean array where True indicates outlier
        - detector: Trained outlier detection model (None for quantile method)
    """
    # Convert X to DataFrame if it's a numpy array
    if isinstance(X, np.ndarray):
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
    else:
        X_df = X
        
    # Detect outliers using the specified method
    if method == 'isolation_forest':
        detector = IsolationForest(
            contamination=contamination,
            random_state=42
        )
        # -1 for outliers, 1 for inliers
        outlier_labels = detector.fit_predict(X_df)
        outlier_mask = outlier_labels == -1
        
    elif method == 'lof':
        detector = LocalOutlierFactor(
            n_neighbors=20,
            contamination=contamination
        )
        # -1 for outliers, 1 for inliers
        outlier_labels = detector.fit_predict(X_df)
        outlier_mask = outlier_labels == -1
        detector = None  # LOF doesn't have a fitted model attribute
        
    elif method == 'quantile':
        # Simple approach: detect values outside of quantile range
        lower_quantile = contamination / 2
        upper_quantile = 1 - lower_quantile
        
        # Initialize mask
        outlier_mask = np.zeros(len(X_df), dtype=bool)
        
        # For each feature, mark samples outside the quantile range
        for feature in X_df.columns:
            values = X_df[feature].values
            q_low = np.quantile(values, lower_quantile)
            q_high = np.quantile(values, upper_quantile)
            
            # Mark as outlier if outside the range
            feature_outliers = (values < q_low) | (values > q_high)
            outlier_mask |= feature_outliers
            
        detector = None
        
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")
        
    return outlier_mask, detector