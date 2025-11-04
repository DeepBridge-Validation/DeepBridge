"""
WeakSpot Detection for ML Models

This module identifies regions (slices) of the feature space where model
performance degrades significantly. Unlike global metrics, weakspot detection
reveals localized failures that might be hidden in aggregate statistics.

Use Cases:
- Credit scoring: Model fails for extreme incomes (< $10k or > $200k)
- Healthcare: Poor performance for pediatric or geriatric patients
- Insurance: High error rates in specific geographic regions

Based on research from Google's Slice Finder and Microsoft's Spotlight.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from sklearn.tree import DecisionTreeRegressor
import warnings


class WeakspotDetector:
    """
    Detects weak regions (slices) in the feature space where model performance degrades.

    The detector uses various slicing strategies to partition the feature space
    and identifies slices with significantly worse performance than global average.

    Example:
    --------
    >>> from deepbridge.validation.robustness.weakspot_detector import WeakspotDetector
    >>>
    >>> # Create detector
    >>> detector = WeakspotDetector(
    ...     slice_method='quantile',
    ...     n_slices=10,
    ...     severity_threshold=0.2
    ... )
    >>>
    >>> # Detect weakspots
    >>> results = detector.detect_weak_regions(
    ...     X=X_test,
    ...     y_true=y_test,
    ...     y_pred=predictions,
    ...     slice_features=['income', 'age'],
    ...     metric='mae'
    ... )
    >>>
    >>> # Analyze results
    >>> print(f"Found {results['summary']['total_weakspots']} weakspots")
    >>> for ws in results['weakspots'][:5]:
    ...     print(f"  {ws['feature']}: {ws['range']} (severity={ws['severity']:.2f})")
    """

    def __init__(self,
                 slice_method: str = 'quantile',
                 n_slices: int = 10,
                 min_samples_per_slice: int = 30,
                 severity_threshold: float = 0.15):
        """
        Initialize WeakSpot Detector.

        Parameters:
        -----------
        slice_method : str, default='quantile'
            Method for creating slices:
            - 'uniform': Equal-width bins
            - 'quantile': Equal-frequency bins (recommended)
            - 'tree-based': Adaptive splitting using decision tree
        n_slices : int, default=10
            Number of slices to create per feature
        min_samples_per_slice : int, default=30
            Minimum samples required in a slice to be considered
            (prevents false positives from small samples)
        severity_threshold : float, default=0.15
            Relative degradation threshold to classify as "weak"
            0.15 = 15% worse than global average
        """
        valid_methods = ['uniform', 'quantile', 'tree-based']
        if slice_method not in valid_methods:
            raise ValueError(f"slice_method must be one of {valid_methods}")

        self.slice_method = slice_method
        self.n_slices = n_slices
        self.min_samples_per_slice = min_samples_per_slice
        self.severity_threshold = severity_threshold

    def detect_weak_regions(self,
                           X: pd.DataFrame,
                           y_true: np.ndarray,
                           y_pred: np.ndarray,
                           slice_features: Optional[List[str]] = None,
                           metric: str = 'mae') -> Dict[str, Any]:
        """
        Identify weak regions where model performance degrades.

        Parameters:
        -----------
        X : DataFrame
            Features (must include slice_features)
        y_true : array-like
            True labels/values
        y_pred : array-like
            Predicted labels/values
        slice_features : List[str], optional
            Features to analyze (None = all numeric features)
        metric : str, default='mae'
            Metric for evaluating weakness:
            - 'mae': Mean Absolute Error (regression)
            - 'mse': Mean Squared Error (regression)
            - 'residual': Raw residuals (regression)
            - 'error_rate': Classification error rate

        Returns:
        --------
        dict : Comprehensive weakspot analysis
            {
                'weakspots': List[Dict],  # Ordered by severity (worst first)
                'summary': {
                    'total_weakspots': int,
                    'features_with_weakspots': int,
                    'avg_severity': float,
                    'max_severity': float,
                    'critical_weakspots': int  # severity > 0.5
                },
                'slice_analysis': Dict[str, Dict],  # Per-feature detailed analysis
                'global_mean_residual': float,
                'config': Dict  # Configuration used
            }
        """
        # Validate inputs
        if len(y_true) != len(y_pred):
            raise ValueError(f"y_true and y_pred must have same length ({len(y_true)} vs {len(y_pred)})")

        if len(X) != len(y_true):
            raise ValueError(f"X and y_true must have same length ({len(X)} vs {len(y_true)})")

        # Select features to analyze
        if slice_features is None:
            # Auto-select numeric features
            slice_features = X.select_dtypes(include=[np.number]).columns.tolist()
            if not slice_features:
                raise ValueError("No numeric features found in X. Specify slice_features manually.")
        else:
            # Validate that features exist
            missing = [f for f in slice_features if f not in X.columns]
            if missing:
                raise ValueError(f"Features not found in X: {missing}")

        # Calculate residuals/errors
        residuals = self._calculate_residuals(y_true, y_pred, metric)
        global_mean_residual = np.mean(np.abs(residuals))

        weakspots = []
        slice_analysis = {}

        # Analyze each feature
        for feature in slice_features:
            feature_values = X[feature].values

            # Skip if feature has too many missing values
            if np.isnan(feature_values).sum() / len(feature_values) > 0.5:
                warnings.warn(f"Feature '{feature}' has >50% missing values, skipping")
                continue

            # Create slices for this feature
            slices = self._create_slices(feature_values, method=self.slice_method)

            feature_analysis = {
                'feature': feature,
                'slices': [],
                'worst_slice': None,
                'best_slice': None,
                'n_slices_evaluated': 0
            }

            for slice_idx, (slice_range, slice_mask) in enumerate(slices):
                n_samples = np.sum(slice_mask)

                # Skip slices with too few samples
                if n_samples < self.min_samples_per_slice:
                    continue

                feature_analysis['n_slices_evaluated'] += 1

                # Calculate metrics for this slice
                slice_residuals = residuals[slice_mask]
                slice_mean_residual = np.mean(np.abs(slice_residuals))
                slice_std_residual = np.std(slice_residuals)
                slice_max_residual = np.max(np.abs(slice_residuals))

                # Severity: relative degradation vs global average
                severity = (slice_mean_residual - global_mean_residual) / global_mean_residual

                slice_info = {
                    'slice_idx': slice_idx,
                    'feature': feature,
                    'range': slice_range,
                    'range_str': f"[{slice_range[0]:.2f}, {slice_range[1]:.2f}]",
                    'n_samples': int(n_samples),
                    'mean_residual': float(slice_mean_residual),
                    'std_residual': float(slice_std_residual),
                    'max_residual': float(slice_max_residual),
                    'global_mean_residual': float(global_mean_residual),
                    'severity': float(severity),
                    'is_weak': severity > self.severity_threshold
                }

                feature_analysis['slices'].append(slice_info)

                # Identify if it's a weakspot
                if slice_info['is_weak']:
                    weakspots.append(slice_info)

            # Find worst and best slices for this feature
            if feature_analysis['slices']:
                feature_analysis['worst_slice'] = max(
                    feature_analysis['slices'],
                    key=lambda s: s['severity']
                )
                feature_analysis['best_slice'] = min(
                    feature_analysis['slices'],
                    key=lambda s: s['severity']
                )

            slice_analysis[feature] = feature_analysis

        # Sort weakspots by severity (worst first)
        weakspots = sorted(weakspots, key=lambda w: w['severity'], reverse=True)

        # Generate summary
        summary = {
            'total_weakspots': len(weakspots),
            'features_with_weakspots': len(set(w['feature'] for w in weakspots)),
            'features_analyzed': len(slice_features),
            'avg_severity': float(np.mean([w['severity'] for w in weakspots])) if weakspots else 0.0,
            'max_severity': float(max([w['severity'] for w in weakspots])) if weakspots else 0.0,
            'critical_weakspots': len([w for w in weakspots if w['severity'] > 0.5])
        }

        return {
            'weakspots': weakspots,
            'summary': summary,
            'slice_analysis': slice_analysis,
            'global_mean_residual': float(global_mean_residual),
            'config': {
                'slice_method': self.slice_method,
                'n_slices': self.n_slices,
                'min_samples_per_slice': self.min_samples_per_slice,
                'severity_threshold': self.severity_threshold,
                'metric': metric
            }
        }

    def _calculate_residuals(self,
                            y_true: np.ndarray,
                            y_pred: np.ndarray,
                            metric: str) -> np.ndarray:
        """
        Calculate residuals/errors based on metric.

        Parameters:
        -----------
        y_true, y_pred : array-like
            True and predicted values
        metric : str
            Metric type

        Returns:
        --------
        array : Residuals/errors
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if metric in ['mae', 'residual']:
            return y_true - y_pred
        elif metric == 'mse':
            return (y_true - y_pred) ** 2
        elif metric == 'error_rate':
            # For classification
            return (y_true != y_pred).astype(float)
        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'mae', 'mse', 'residual', or 'error_rate'")

    def _create_slices(self,
                      feature_values: np.ndarray,
                      method: str) -> List[Tuple[Tuple[float, float], np.ndarray]]:
        """
        Create slices of a feature using specified method.

        Parameters:
        -----------
        feature_values : array
            Feature values to slice
        method : str
            Slicing method

        Returns:
        --------
        List[Tuple[Tuple[float, float], np.ndarray]] :
            List of (slice_range, slice_mask) tuples
        """
        # Remove NaN values for slicing (mask will handle them)
        valid_mask = ~np.isnan(feature_values)
        valid_values = feature_values[valid_mask]

        if len(valid_values) == 0:
            return []

        if method == 'uniform':
            return self._uniform_slices(feature_values, valid_values)
        elif method == 'quantile':
            return self._quantile_slices(feature_values, valid_values)
        elif method == 'tree-based':
            return self._tree_based_slices(feature_values, valid_values)
        else:
            raise ValueError(f"Unknown slice method: {method}")

    def _uniform_slices(self,
                       feature_values: np.ndarray,
                       valid_values: np.ndarray) -> List[Tuple[Tuple[float, float], np.ndarray]]:
        """
        Create slices with equal-width bins.

        Good for: Features with uniform distribution
        """
        min_val = np.min(valid_values)
        max_val = np.max(valid_values)

        # Handle edge case where all values are the same
        if min_val == max_val:
            return [((min_val, max_val), np.ones(len(feature_values), dtype=bool))]

        bin_edges = np.linspace(min_val, max_val, self.n_slices + 1)

        slices = []
        for i in range(self.n_slices):
            slice_range = (float(bin_edges[i]), float(bin_edges[i+1]))

            if i == self.n_slices - 1:
                # Last slice includes upper edge
                slice_mask = (feature_values >= slice_range[0]) & (feature_values <= slice_range[1])
            else:
                slice_mask = (feature_values >= slice_range[0]) & (feature_values < slice_range[1])

            slices.append((slice_range, slice_mask))

        return slices

    def _quantile_slices(self,
                        feature_values: np.ndarray,
                        valid_values: np.ndarray) -> List[Tuple[Tuple[float, float], np.ndarray]]:
        """
        Create slices with equal-frequency bins (quantiles).

        Good for: Most features (recommended default)
        Ensures each slice has approximately same number of samples.
        """
        quantiles = np.linspace(0, 1, self.n_slices + 1)
        bin_edges = np.quantile(valid_values, quantiles)

        # Handle duplicate edges (can happen with discrete features)
        bin_edges = np.unique(bin_edges)
        actual_n_slices = len(bin_edges) - 1

        if actual_n_slices == 0:
            # All values are the same
            return [((float(bin_edges[0]), float(bin_edges[0])),
                    np.ones(len(feature_values), dtype=bool))]

        slices = []
        for i in range(actual_n_slices):
            slice_range = (float(bin_edges[i]), float(bin_edges[i+1]))

            if i == actual_n_slices - 1:
                slice_mask = (feature_values >= slice_range[0]) & (feature_values <= slice_range[1])
            else:
                slice_mask = (feature_values >= slice_range[0]) & (feature_values < slice_range[1])

            slices.append((slice_range, slice_mask))

        return slices

    def _tree_based_slices(self,
                          feature_values: np.ndarray,
                          valid_values: np.ndarray) -> List[Tuple[Tuple[float, float], np.ndarray]]:
        """
        Create adaptive slices using decision tree splits.

        Good for: Finding natural breakpoints in the feature
        Uses decision tree to find optimal split points based on residuals.

        Note: Requires residuals to be available. This is a placeholder
        for now - proper implementation would need residuals passed in.
        """
        # TODO: Full implementation would use decision tree to find optimal splits
        # For now, fall back to quantile slicing
        warnings.warn(
            "tree-based slicing not fully implemented yet, using quantile method",
            UserWarning
        )
        return self._quantile_slices(feature_values, valid_values)

    def get_top_weakspots(self,
                         results: Dict[str, Any],
                         n: int = 5) -> List[Dict]:
        """
        Get top N weakspots ordered by severity.

        Parameters:
        -----------
        results : dict
            Results from detect_weak_regions()
        n : int, default=5
            Number of top weakspots to return

        Returns:
        --------
        List[Dict] : Top N weakspots
        """
        return results['weakspots'][:n]

    def print_summary(self, results: Dict[str, Any], verbose: bool = True):
        """
        Print human-readable summary of weakspot detection results.

        Parameters:
        -----------
        results : dict
            Results from detect_weak_regions()
        verbose : bool, default=True
            If True, print detailed info for top weakspots
        """
        summary = results['summary']

        print("\n" + "="*70)
        print("WEAKSPOT DETECTION SUMMARY")
        print("="*70)
        print(f"Total Weakspots Found: {summary['total_weakspots']}")
        print(f"Features with Weakspots: {summary['features_with_weakspots']} / {summary['features_analyzed']}")
        print(f"Average Severity: {summary['avg_severity']:.2%}")
        print(f"Max Severity: {summary['max_severity']:.2%}")
        print(f"Critical Weakspots (>50% degradation): {summary['critical_weakspots']}")
        print(f"\nGlobal Mean Residual: {results['global_mean_residual']:.4f}")

        if verbose and results['weakspots']:
            print("\n" + "-"*70)
            print("TOP 5 WEAKSPOTS (Ordered by Severity)")
            print("-"*70)

            for i, ws in enumerate(results['weakspots'][:5], 1):
                print(f"\n{i}. Feature: {ws['feature']}")
                print(f"   Range: {ws['range_str']}")
                print(f"   Samples: {ws['n_samples']}")
                print(f"   Mean Residual: {ws['mean_residual']:.4f} (global: {ws['global_mean_residual']:.4f})")
                print(f"   Severity: {ws['severity']:.2%} {'🚨 CRITICAL' if ws['severity'] > 0.5 else '⚠️  WARNING'}")

        print("="*70 + "\n")
