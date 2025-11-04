"""
Sliced Overfitting Analysis for ML Models

This module detects localized overfitting by analyzing train-test performance
gaps across different slices of the feature space.

Why This Matters:
-----------------
A model might show acceptable global train-test gap (e.g., train AUC=0.92, test AUC=0.90),
but exhibit severe overfitting in specific regions:

Example:
    Feature: Income
    Overall:  train=0.92, test=0.90 (gap=0.02) ✓ OK
    Income > $150k: train=0.98, test=0.75 (gap=0.23) ✗ SEVERE OVERFITTING

This localized overfitting is hidden in global metrics but critical for:
- Model reliability in production
- Fairness (certain groups may be affected)
- Risk management

Based on research from PiML-Toolbox and interpretable ML literature.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Callable, Tuple
import warnings


class OverfitAnalyzer:
    """
    Analyzes overfitting in specific slices of the dataset.

    Identifies regions of the feature space where the model exhibits
    large train-test performance gaps, indicating localized overfitting.

    Example:
    --------
    >>> from deepbridge.validation.robustness.overfit_analyzer import OverfitAnalyzer
    >>>
    >>> analyzer = OverfitAnalyzer(
    ...     n_slices=10,
    ...     slice_method='quantile',
    ...     gap_threshold=0.1
    ... )
    >>>
    >>> results = analyzer.compute_gap_by_slice(
    ...     X_train=X_train,
    ...     X_test=X_test,
    ...     y_train=y_train,
    ...     y_test=y_test,
    ...     model=model,
    ...     slice_feature='income',
    ...     metric_func=lambda y_true, y_pred: roc_auc_score(y_true, y_pred)
    ... )
    >>>
    >>> print(f"Overfit slices: {results['summary']['overfit_slices_count']}")
    >>> print(f"Max gap: {results['max_gap']:.3f}")
    """

    def __init__(self,
                 n_slices: int = 10,
                 slice_method: str = 'quantile',
                 gap_threshold: float = 0.1,
                 min_samples_per_slice: int = 30):
        """
        Initialize Overfit Analyzer.

        Parameters:
        -----------
        n_slices : int, default=10
            Number of slices to create per feature
        slice_method : str, default='quantile'
            Slicing method: 'uniform' or 'quantile'
        gap_threshold : float, default=0.1
            Threshold for considering a gap as significant
            0.1 = 10% performance difference between train and test
        min_samples_per_slice : int, default=30
            Minimum samples required in both train and test slices
        """
        self.n_slices = n_slices
        self.slice_method = slice_method
        self.gap_threshold = gap_threshold
        self.min_samples_per_slice = min_samples_per_slice

    def compute_gap_by_slice(self,
                            X_train: pd.DataFrame,
                            X_test: pd.DataFrame,
                            y_train: np.ndarray,
                            y_test: np.ndarray,
                            model: Any,
                            slice_feature: str,
                            metric_func: Callable[[np.ndarray, np.ndarray], float]) -> Dict[str, Any]:
        """
        Calculate train-test performance gap across slices of a feature.

        Parameters:
        -----------
        X_train, X_test : DataFrame
            Training and test features
        y_train, y_test : array-like
            Training and test labels
        model : fitted model
            Model with predict() or predict_proba() method
        slice_feature : str
            Feature to slice on (must exist in both X_train and X_test)
        metric_func : callable
            Function that computes metric: metric_func(y_true, y_pred) -> float
            Higher values should mean better performance

        Returns:
        --------
        dict : Detailed overfitting analysis
            {
                'feature': str,
                'slices': List[Dict],  # Per-slice analysis
                'max_gap': float,
                'avg_gap': float,
                'std_gap': float,
                'overfit_slices': List[Dict],  # Slices with gap > threshold
                'summary': {
                    'total_slices': int,
                    'overfit_slices_count': int,
                    'overfit_percentage': float
                }
            }
        """
        # Validate inputs
        if slice_feature not in X_train.columns:
            raise ValueError(f"Feature '{slice_feature}' not found in X_train")
        if slice_feature not in X_test.columns:
            raise ValueError(f"Feature '{slice_feature}' not found in X_test")

        # Get feature values
        train_feature = X_train[slice_feature].values
        test_feature = X_test[slice_feature].values

        # Create slices
        train_slices = self._create_slices(train_feature)
        test_slices = self._create_slices(test_feature)

        slice_results = []
        overfit_slices = []

        # Analyze each slice
        for slice_idx in range(min(len(train_slices), len(test_slices))):
            train_range, train_mask = train_slices[slice_idx]
            test_range, test_mask = test_slices[slice_idx]

            n_train = np.sum(train_mask)
            n_test = np.sum(test_mask)

            # Skip if too few samples
            if n_train < self.min_samples_per_slice or n_test < self.min_samples_per_slice:
                continue

            # Get slice data
            X_train_slice = X_train[train_mask]
            y_train_slice = y_train[train_mask]
            X_test_slice = X_test[test_mask]
            y_test_slice = y_test[test_mask]

            # Compute metrics
            try:
                # Train metric
                y_train_pred = self._predict(model, X_train_slice, y_train_slice)
                train_metric = metric_func(y_train_slice, y_train_pred)

                # Test metric
                y_test_pred = self._predict(model, X_test_slice, y_test_slice)
                test_metric = metric_func(y_test_slice, y_test_pred)

                # Gap (train - test; positive means overfitting)
                gap = train_metric - test_metric

                slice_info = {
                    'slice_idx': slice_idx,
                    'train_range': train_range,
                    'test_range': test_range,
                    'range_str': f"[{train_range[0]:.2f}, {train_range[1]:.2f}]",
                    'train_samples': int(n_train),
                    'test_samples': int(n_test),
                    'train_metric': float(train_metric),
                    'test_metric': float(test_metric),
                    'gap': float(gap),
                    'gap_percentage': float(gap / train_metric * 100) if train_metric != 0 else 0.0,
                    'is_overfitting': gap > self.gap_threshold
                }

                slice_results.append(slice_info)

                if slice_info['is_overfitting']:
                    overfit_slices.append(slice_info)

            except Exception as e:
                warnings.warn(f"Error computing metrics for slice {slice_idx}: {str(e)}")
                continue

        # Calculate summary statistics
        if slice_results:
            gaps = [s['gap'] for s in slice_results]
            max_gap = max(gaps)
            avg_gap = np.mean(gaps)
            std_gap = np.std(gaps)
        else:
            max_gap = avg_gap = std_gap = 0.0

        summary = {
            'total_slices': len(slice_results),
            'overfit_slices_count': len(overfit_slices),
            'overfit_percentage': (len(overfit_slices) / len(slice_results) * 100) if slice_results else 0.0
        }

        return {
            'feature': slice_feature,
            'slices': slice_results,
            'max_gap': float(max_gap),
            'avg_gap': float(avg_gap),
            'std_gap': float(std_gap),
            'overfit_slices': overfit_slices,
            'summary': summary,
            'config': {
                'n_slices': self.n_slices,
                'slice_method': self.slice_method,
                'gap_threshold': self.gap_threshold
            }
        }

    def analyze_multiple_features(self,
                                  X_train: pd.DataFrame,
                                  X_test: pd.DataFrame,
                                  y_train: np.ndarray,
                                  y_test: np.ndarray,
                                  model: Any,
                                  features: List[str],
                                  metric_func: Callable) -> Dict[str, Any]:
        """
        Analyze overfitting across multiple features.

        Parameters:
        -----------
        X_train, X_test : DataFrame
            Training and test features
        y_train, y_test : array-like
            Training and test labels
        model : fitted model
            Trained model
        features : List[str]
            Features to analyze
        metric_func : callable
            Metric function

        Returns:
        --------
        dict : Results for all features
            {
                'features': Dict[str, Dict],  # Per-feature results
                'worst_feature': str,  # Feature with highest max gap
                'summary': {
                    'total_features': int,
                    'features_with_overfitting': int,
                    'global_max_gap': float
                }
            }
        """
        feature_results = {}
        max_gaps = {}

        for feature in features:
            if feature not in X_train.columns or feature not in X_test.columns:
                warnings.warn(f"Feature '{feature}' not found, skipping")
                continue

            result = self.compute_gap_by_slice(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                model=model,
                slice_feature=feature,
                metric_func=metric_func
            )

            feature_results[feature] = result
            max_gaps[feature] = result['max_gap']

        # Find worst feature
        worst_feature = max(max_gaps, key=max_gaps.get) if max_gaps else None

        # Summary
        features_with_overfitting = sum(
            1 for r in feature_results.values()
            if r['summary']['overfit_slices_count'] > 0
        )

        summary = {
            'total_features': len(feature_results),
            'features_with_overfitting': features_with_overfitting,
            'global_max_gap': max(max_gaps.values()) if max_gaps else 0.0
        }

        return {
            'features': feature_results,
            'worst_feature': worst_feature,
            'summary': summary
        }

    def _predict(self, model: Any, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """
        Get predictions from model.

        Handles both classifiers (predict_proba) and regressors (predict).
        """
        # Check if classification (predict_proba available)
        if hasattr(model, 'predict_proba'):
            # Binary classification: get probability of positive class
            proba = model.predict_proba(X)
            if proba.shape[1] == 2:
                return proba[:, 1]
            else:
                # Multi-class: use predict
                return model.predict(X)
        else:
            # Regression
            return model.predict(X)

    def _create_slices(self, feature_values: np.ndarray) -> List[Tuple[Tuple[float, float], np.ndarray]]:
        """
        Create slices using configured method.

        Returns:
        --------
        List[Tuple[Tuple[float, float], np.ndarray]] :
            List of (slice_range, slice_mask) tuples
        """
        # Remove NaN
        valid_mask = ~np.isnan(feature_values)
        valid_values = feature_values[valid_mask]

        if len(valid_values) == 0:
            return []

        if self.slice_method == 'quantile':
            return self._quantile_slices(feature_values, valid_values)
        elif self.slice_method == 'uniform':
            return self._uniform_slices(feature_values, valid_values)
        else:
            raise ValueError(f"Unknown slice method: {self.slice_method}")

    def _quantile_slices(self,
                        feature_values: np.ndarray,
                        valid_values: np.ndarray) -> List[Tuple[Tuple[float, float], np.ndarray]]:
        """Create quantile-based slices"""
        quantiles = np.linspace(0, 1, self.n_slices + 1)
        bin_edges = np.quantile(valid_values, quantiles)
        bin_edges = np.unique(bin_edges)
        actual_n_slices = len(bin_edges) - 1

        if actual_n_slices == 0:
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

    def _uniform_slices(self,
                       feature_values: np.ndarray,
                       valid_values: np.ndarray) -> List[Tuple[Tuple[float, float], np.ndarray]]:
        """Create uniform-width slices"""
        min_val = np.min(valid_values)
        max_val = np.max(valid_values)

        if min_val == max_val:
            return [((min_val, max_val), np.ones(len(feature_values), dtype=bool))]

        bin_edges = np.linspace(min_val, max_val, self.n_slices + 1)

        slices = []
        for i in range(self.n_slices):
            slice_range = (float(bin_edges[i]), float(bin_edges[i+1]))

            if i == self.n_slices - 1:
                slice_mask = (feature_values >= slice_range[0]) & (feature_values <= slice_range[1])
            else:
                slice_mask = (feature_values >= slice_range[0]) & (feature_values < slice_range[1])

            slices.append((slice_range, slice_mask))

        return slices

    def print_summary(self, results: Dict[str, Any], verbose: bool = True):
        """
        Print human-readable summary of overfitting analysis.

        Parameters:
        -----------
        results : dict
            Results from compute_gap_by_slice() or analyze_multiple_features()
        verbose : bool, default=True
            If True, print detailed info
        """
        # Check if single feature or multiple features
        if 'features' in results:
            # Multiple features
            self._print_multifeature_summary(results, verbose)
        else:
            # Single feature
            self._print_singlefeature_summary(results, verbose)

    def _print_singlefeature_summary(self, results: Dict[str, Any], verbose: bool):
        """Print summary for single feature analysis"""
        print("\n" + "="*70)
        print("SLICED OVERFITTING ANALYSIS")
        print("="*70)
        print(f"Feature: {results['feature']}")
        print(f"Total Slices: {results['summary']['total_slices']}")
        print(f"Overfit Slices: {results['summary']['overfit_slices_count']} ({results['summary']['overfit_percentage']:.1f}%)")
        print(f"Max Gap: {results['max_gap']:.4f}")
        print(f"Avg Gap: {results['avg_gap']:.4f} ± {results['std_gap']:.4f}")

        if verbose and results['overfit_slices']:
            print("\n" + "-"*70)
            print("OVERFIT SLICES (Gap > Threshold)")
            print("-"*70)

            for i, s in enumerate(results['overfit_slices'][:5], 1):
                print(f"\n{i}. Range: {s['range_str']}")
                print(f"   Train: {s['train_metric']:.4f} ({s['train_samples']} samples)")
                print(f"   Test:  {s['test_metric']:.4f} ({s['test_samples']} samples)")
                print(f"   Gap:   {s['gap']:.4f} ({s['gap_percentage']:.1f}%) {'🚨' if s['gap'] > 0.2 else '⚠️'}")

        print("="*70 + "\n")

    def _print_multifeature_summary(self, results: Dict[str, Any], verbose: bool):
        """Print summary for multiple features analysis"""
        print("\n" + "="*70)
        print("MULTI-FEATURE OVERFITTING ANALYSIS")
        print("="*70)
        print(f"Features Analyzed: {results['summary']['total_features']}")
        print(f"Features with Overfitting: {results['summary']['features_with_overfitting']}")
        print(f"Global Max Gap: {results['summary']['global_max_gap']:.4f}")
        print(f"Worst Feature: {results['worst_feature']}")

        if verbose:
            print("\n" + "-"*70)
            print("PER-FEATURE SUMMARY")
            print("-"*70)

            for feature, feature_result in results['features'].items():
                print(f"\n{feature}:")
                print(f"  Max Gap: {feature_result['max_gap']:.4f}")
                print(f"  Overfit Slices: {feature_result['summary']['overfit_slices_count']} / {feature_result['summary']['total_slices']}")

        print("="*70 + "\n")
