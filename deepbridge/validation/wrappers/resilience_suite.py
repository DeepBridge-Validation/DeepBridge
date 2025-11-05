"""
Simplified resilience testing suite for machine learning models.

This module provides a streamlined interface for evaluating model resilience
when faced with changing input distributions and identifying areas for enhancement.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
import time
import datetime
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score,
                            recall_score, f1_score, accuracy_score, mean_squared_error,
                            mean_absolute_error, r2_score)

from deepbridge.core.experiment.parameter_standards import (
    get_test_config, TestType, ConfigName, is_valid_config_name
)

class ResilienceSuite:
    """
    Focused suite for model resilience testing under distribution shifts.
    """

    # Load configurations from centralized parameter standards
    def _get_config_templates(self):
        """Get resilience configurations from the centralized parameter standards."""
        try:
            # Convert the drift-based configurations to test specific format
            central_configs = {
                config_name: get_test_config(TestType.RESILIENCE.value, config_name)
                for config_name in [ConfigName.QUICK.value, ConfigName.MEDIUM.value, ConfigName.FULL.value]
            }

            # Transform the format to match what the resilience suite expects
            test_configs = {}
            for config_name, config in central_configs.items():
                tests = []
                drift_types = config.get('drift_types', [])
                drift_intensities = config.get('drift_intensities', [])

                # Create test configurations based on drift types and intensities (distribution_shift)
                for drift_type in drift_types:
                    for intensity in drift_intensities:
                        # Create corresponding alpha and metric settings
                        distance_metric = 'PSI'  # Default
                        if drift_type == 'covariate':
                            distance_metric = 'PSI'
                        elif drift_type == 'concept':
                            distance_metric = 'KS'
                        elif drift_type == 'label':
                            distance_metric = 'WD1'
                        elif drift_type == 'distribution':
                            distance_metric = 'KL'
                        elif drift_type == 'statistical':
                            distance_metric = 'CM'

                        # Add test configuration
                        tests.append({
                            'method': 'distribution_shift',
                            'params': {
                                'alpha': intensity,
                                'metric': 'auc',  # Default metric
                                'distance_metric': distance_metric
                            }
                        })

                # Add new test scenarios from configuration
                test_scenarios = config.get('test_scenarios', [])
                for scenario in test_scenarios:
                    method = scenario['method']

                    if method == 'worst_sample':
                        alphas = scenario.get('alphas', [0.1])
                        ranking_methods = scenario.get('ranking_methods', ['residual'])
                        for alpha in alphas:
                            for ranking_method in ranking_methods:
                                tests.append({
                                    'method': 'worst_sample',
                                    'params': {
                                        'alpha': alpha,
                                        'metric': 'auc',
                                        'ranking_method': ranking_method
                                    }
                                })

                    elif method == 'worst_cluster':
                        n_clusters_list = scenario.get('n_clusters_list', [5])
                        for n_clusters in n_clusters_list:
                            tests.append({
                                'method': 'worst_cluster',
                                'params': {
                                    'n_clusters': n_clusters,
                                    'metric': 'auc',
                                    'random_state': None
                                }
                            })

                    elif method == 'outer_sample':
                        alphas = scenario.get('alphas', [0.05])
                        outlier_methods = scenario.get('outlier_methods', ['isolation_forest'])
                        for alpha in alphas:
                            for outlier_method in outlier_methods:
                                tests.append({
                                    'method': 'outer_sample',
                                    'params': {
                                        'alpha': alpha,
                                        'metric': 'auc',
                                        'outlier_method': outlier_method,
                                        'random_state': None
                                    }
                                })

                    elif method == 'hard_sample':
                        disagreement_thresholds = scenario.get('disagreement_thresholds', [0.3])
                        for threshold in disagreement_thresholds:
                            tests.append({
                                'method': 'hard_sample',
                                'params': {
                                    'disagreement_threshold': threshold,
                                    'metric': 'auc',
                                    'auxiliary_models': None  # Will use all available
                                }
                            })

                test_configs[config_name] = tests

            return test_configs
        except Exception as e:
            import logging
            logging.getLogger("deepbridge.resilience").error(f"Error loading centralized configs: {str(e)}")
            # Fallback to empty templates if centralized configs fail
            return {
                'quick': [],
                'medium': [],
                'full': []
            }
    
    def __init__(self, dataset, verbose: bool = False, feature_subset: Optional[List[str]] = None, random_state: Optional[int] = None, metric: str = 'auc'):
        """
        Initialize the resilience testing suite.
        
        Parameters:
        -----------
        dataset : DBDataset
            Dataset object containing training/test data and model
        verbose : bool
            Whether to print progress information
        feature_subset : List[str] or None
            Subset of features to analyze (None for all features)
        random_state : int or None
            Random seed for reproducibility
        metric : str
            Performance metric to use ('auc', 'f1', 'accuracy', etc.)
        """
        self.dataset = dataset
        self.verbose = verbose
        self.feature_subset = feature_subset
        self.random_state = random_state
        self.metric = metric
        
        # Store current configuration
        self.current_config = None
        
        # Store results
        self.results = {}
        
        # Determine problem type based on dataset or model
        self._problem_type = self._determine_problem_type()
        
        # Initialize distance metrics
        self.distance_metrics = {
            "PSI": self._calculate_psi,
            "KS": self._calculate_ks,
            "WD1": self._calculate_wasserstein,
            "KL": self._calculate_kl_divergence,
            "CM": self._calculate_cm_statistic
        }
        
        if self.verbose:
            print(f"Problem type detected: {self._problem_type}")
    
    def _determine_problem_type(self):
        """Determine if the problem is classification or regression"""
        # Try to get problem type from dataset
        if hasattr(self.dataset, 'problem_type'):
            return self.dataset.problem_type
        
        # Try to infer from the model
        if hasattr(self.dataset, 'model'):
            model = self.dataset.model
            if hasattr(model, 'predict_proba'):
                return 'classification'
            else:
                return 'regression'
        
        # Default to classification
        return 'classification'
    
    def config(self, config_name: str = 'quick', feature_subset: Optional[List[str]] = None) -> 'ResilienceSuite':
        """
        Set a predefined configuration for resilience tests.

        Parameters:
        -----------
        config_name : str
            Name of the configuration to use: 'quick', 'medium', or 'full'
        feature_subset : List[str] or None
            Subset of features to test (overrides the one set in constructor)

        Returns:
        --------
        self : Returns self to allow method chaining
        """
        self.feature_subset = feature_subset if feature_subset is not None else self.feature_subset

        # Validate config_name
        if not is_valid_config_name(config_name):
            raise ValueError(f"Unknown configuration: {config_name}. Available options: {[ConfigName.QUICK.value, ConfigName.MEDIUM.value, ConfigName.FULL.value]}")

        # Get the configuration templates from central location
        config_templates = self._get_config_templates()

        if config_name not in config_templates:
            raise ValueError(f"Configuration '{config_name}' not found in templates. Available options: {list(config_templates.keys())}")

        # Clone the configuration template
        self.current_config = self._clone_config(config_templates[config_name])

        # Update feature_subset in tests if specified
        if self.feature_subset:
            for test in self.current_config:
                if 'params' in test:
                    test['params']['feature_subset'] = self.feature_subset

        if self.verbose:
            print(f"\nConfigured for {config_name} resilience test suite")
            if self.feature_subset:
                print(f"Feature subset: {self.feature_subset}")
            print(f"\nTests that will be executed:")

            # Print all configured tests
            for i, test in enumerate(self.current_config, 1):
                test_method = test['method']
                params = test.get('params', {})
                param_str = ', '.join(f"{k}={v}" for k, v in params.items())
                print(f"  {i}. {test_method} ({param_str})")

        return self
    
    def _clone_config(self, config):
        """Clone configuration to avoid modifying original templates."""
        import copy
        return copy.deepcopy(config)
    
    def _calculate_residuals(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate residuals based on the problem type.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True target values
        y_pred : np.ndarray
            Predicted values or probabilities
            
        Returns:
        --------
        np.ndarray
            Calculated residuals
        """
        if self._problem_type == "classification":
            # For classification, use absolute difference between predicted prob and true class
            return np.abs(y_pred - y_true)
        else:  # regression
            # For regression, use absolute residuals
            return np.abs(y_true - y_pred)
    
    def _select_worst_samples(self, X: pd.DataFrame, residuals: np.ndarray,
                            alpha: float = 0.3) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Select worst samples based on residuals.

        Parameters:
        -----------
        X : pd.DataFrame
            Feature data
        residuals : np.ndarray
            Residuals for each sample
        alpha : float
            Ratio of samples to select as worst samples (0 < alpha < 1)

        Returns:
        --------
        Tuple containing worst samples, remaining samples, worst indices, and remaining indices
        """
        # Sort indices by residual value in descending order
        sorted_indices = np.argsort(-residuals)

        # Calculate number of worst samples
        n_worst = int(alpha * len(X))

        # Ensure we have at least 1 sample in each group (worst and remaining)
        # This prevents errors when alpha is very small or dataset is very small
        total_samples = len(X)
        if total_samples < 2:
            raise ValueError(f"Need at least 2 samples for distribution shift analysis, got {total_samples}")

        # Ensure n_worst is at least 1 and at most (total_samples - 1)
        n_worst = max(1, min(n_worst, total_samples - 1))

        # Get worst and remaining sample indices
        worst_indices = sorted_indices[:n_worst]
        remaining_indices = sorted_indices[n_worst:]

        # Return selected samples and indices
        return X.iloc[worst_indices], X.iloc[remaining_indices], worst_indices, remaining_indices
    
    def _calculate_psi(self, dist1: np.ndarray, dist2: np.ndarray) -> float:
        """
        Calculate Population Stability Index (PSI) between two distributions.
        
        Parameters:
        -----------
        dist1 : np.ndarray
            First distribution
        dist2 : np.ndarray
            Second distribution
            
        Returns:
        --------
        float
            PSI value
        """
        # Create bins based on combined data to ensure consistency
        combined = np.concatenate([dist1, dist2])
        bins = np.linspace(combined.min(), combined.max(), 11)  # 10 bins
        
        # Calculate histograms
        hist1, _ = np.histogram(dist1, bins=bins, density=True)
        hist2, _ = np.histogram(dist2, bins=bins, density=True)
        
        # Add small epsilon to avoid division by zero or log(0)
        epsilon = 1e-10
        hist1 = hist1 + epsilon
        hist2 = hist2 + epsilon
        
        # Normalize histograms to get probabilities
        hist1 = hist1 / hist1.sum()
        hist2 = hist2 / hist2.sum()
        
        # Calculate PSI
        psi = np.sum((hist1 - hist2) * np.log(hist1 / hist2))
        
        return psi
    
    def _calculate_ks(self, dist1: np.ndarray, dist2: np.ndarray) -> float:
        """
        Calculate Kolmogorov-Smirnov statistic between two distributions.
        
        Parameters:
        -----------
        dist1 : np.ndarray
            First distribution
        dist2 : np.ndarray
            Second distribution
            
        Returns:
        --------
        float
            KS statistic
        """
        # Calculate KS statistic
        ks_stat, _ = stats.ks_2samp(dist1, dist2)
        return ks_stat
    
    def _calculate_wasserstein(self, dist1: np.ndarray, dist2: np.ndarray) -> float:
        """
        Calculate 1-Wasserstein distance (Earth Mover's Distance) between two distributions.
        
        Parameters:
        -----------
        dist1 : np.ndarray
            First distribution
        dist2 : np.ndarray
            Second distribution
            
        Returns:
        --------
        float
            Wasserstein distance
        """
        # Calculate Wasserstein distance
        wd = stats.wasserstein_distance(dist1, dist2)
        return wd

    def _calculate_kl_divergence(self, dist1: np.ndarray, dist2: np.ndarray, bins: int = 10) -> float:
        """
        Calculate Kullback-Leibler Divergence between two distributions.

        Parameters:
        -----------
        dist1 : np.ndarray
            First distribution (typically training/baseline)
        dist2 : np.ndarray
            Second distribution (typically test/shifted)
        bins : int
            Number of bins for histogram

        Returns:
        --------
        float
            KL divergence value (non-negative, asymmetric)

        Notes:
        ------
        - KL(P||Q) measures how much information is lost when approximating P with Q
        - Asymmetric: KL(P||Q) ≠ KL(Q||P)
        - Returns infinity if Q has zero probability where P doesn't
        """
        # Create bins based on combined data to ensure consistency
        combined = np.concatenate([dist1, dist2])
        hist_range = (combined.min(), combined.max())

        # Calculate histograms
        p, _ = np.histogram(dist1, bins=bins, range=hist_range, density=True)
        q, _ = np.histogram(dist2, bins=bins, range=hist_range, density=True)

        # Add small epsilon to avoid division by zero or log(0)
        epsilon = 1e-10
        p = p + epsilon
        q = q + epsilon

        # Normalize to get probabilities
        p = p / p.sum()
        q = q / q.sum()

        # Calculate KL divergence: sum(P(x) * log(P(x) / Q(x)))
        kl = np.sum(p * np.log(p / q))

        return kl

    def _calculate_cm_statistic(self, dist1: np.ndarray, dist2: np.ndarray) -> float:
        """
        Calculate Cramér-von Mises statistic between two distributions.

        Parameters:
        -----------
        dist1 : np.ndarray
            First distribution
        dist2 : np.ndarray
            Second distribution

        Returns:
        --------
        float
            CM statistic value (non-negative)

        Notes:
        ------
        - Tests if two samples come from the same distribution
        - More sensitive to differences in the middle of distributions
        - KS test is more sensitive to differences at the tails
        """
        from scipy.stats import cramervonmises_2samp

        # Calculate Cramér-von Mises statistic
        result = cramervonmises_2samp(dist1, dist2)

        # Return only the statistic value (ignore p-value for now)
        # The statistic is more useful for comparing relative drift
        return result.statistic

    def _calculate_feature_distances(self, 
                                   worst_samples: pd.DataFrame,
                                   remaining_samples: pd.DataFrame,
                                   distance_metric: str = "PSI") -> Dict:
        """
        Calculate distribution shift between worst and remaining samples for each feature.
        
        Parameters:
        -----------
        worst_samples : pd.DataFrame
            Worst samples based on residuals
        remaining_samples : pd.DataFrame
            Remaining samples
        distance_metric : str
            Distance metric to use ('PSI', 'KS', or 'WD1')
            
        Returns:
        --------
        Dict
            Dictionary containing distance metrics for each feature
        """
        if distance_metric not in self.distance_metrics:
            raise ValueError(f"Distance metric {distance_metric} not supported. "
                            f"Choose from {list(self.distance_metrics.keys())}")
        
        dist_func = self.distance_metrics[distance_metric]
        feature_distances = {}
        
        for col in worst_samples.columns:
            # Skip non-numeric columns
            if not np.issubdtype(worst_samples[col].dtype, np.number):
                continue
                
            try:
                dist = dist_func(worst_samples[col].values, remaining_samples[col].values)
                feature_distances[col] = dist
            except Exception as e:
                if self.verbose:
                    print(f"Could not calculate {distance_metric} for feature {col}: {str(e)}")
                continue
        
        # Sort features by distance
        sorted_features = sorted(feature_distances.items(), 
                               key=lambda x: x[1], reverse=True)
        
        # Get top 10 features
        top_features = dict(sorted_features[:10])
        
        return {
            "distance_metric": distance_metric,
            "all_feature_distances": feature_distances,
            "top_features": top_features
        }
    
    def evaluate_distribution_shift(self, method: str, params: Dict) -> Dict[str, Any]:
        """
        Evaluate model resilience using distribution shift analysis.
        
        Parameters:
        -----------
        method : str
            Method to use ('distribution_shift')
        params : Dict
            Parameters for the resilience method
            
        Returns:
        --------
        dict : Detailed evaluation results
        """
        # Get parameters
        alpha = params.get('alpha', 0.3)
        metric = params.get('metric', 'auc')
        distance_metric = params.get('distance_metric', 'PSI')
        
        # Get dataset
        X = self.dataset.get_feature_data()
        y = self.dataset.get_target_data()
        
        # Convert any numpy arrays to pandas objects if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        
        # Store original full feature set for predictions
        X_full = X.copy()
        
        # Create feature subset view for analysis only
        X_analysis = X.copy()
        if self.feature_subset:
            # Ensure all features in feature_subset are in X
            valid_features = [f for f in self.feature_subset if f in X.columns]
            if len(valid_features) < len(self.feature_subset):
                missing = set(self.feature_subset) - set(valid_features)
                if self.verbose:
                    print(f"Warning: Some requested features not found in dataset: {missing}")
            if valid_features:
                X_analysis = X[valid_features]
            elif self.verbose:
                print("No valid features in subset. Using all features.")
        
        # Get model
        model = self.dataset.model
        
        # Get predictions using the FULL feature set to avoid scikit-learn feature name mismatch error
        if self._problem_type == "classification" and hasattr(model, "predict_proba"):
            y_pred = model.predict_proba(X_full)[:, 1]
        else:
            y_pred = model.predict(X_full)
        
        # Calculate residuals
        residuals = self._calculate_residuals(y, y_pred)
        
        # Select worst samples using the analysis feature set (subset if specified)
        worst_samples, remaining_samples, worst_indices, remaining_indices = self._select_worst_samples(X_analysis, residuals, alpha)
        
        # Split target values
        y_worst = y.iloc[worst_indices]
        y_remaining = y.iloc[remaining_indices]
        
        # Create full feature views of the worst and remaining samples for prediction
        X_worst_full = X_full.iloc[worst_indices]
        X_remaining_full = X_full.iloc[remaining_indices]
        
        # Calculate performance metrics using the FULL feature set for predictions
        if self._problem_type == "classification":
            if hasattr(model, "predict_proba"):
                worst_pred = model.predict_proba(X_worst_full)[:, 1]
                remaining_pred = model.predict_proba(X_remaining_full)[:, 1]
            else:
                worst_pred = model.predict(X_worst_full)
                remaining_pred = model.predict(X_remaining_full)
                
            # Calculate appropriate metrics based on problem type
            if metric == "auc":
                worst_metric = roc_auc_score(y_worst, worst_pred)
                remaining_metric = roc_auc_score(y_remaining, remaining_pred)
            elif metric == "aucpr":
                worst_metric = average_precision_score(y_worst, worst_pred)
                remaining_metric = average_precision_score(y_remaining, remaining_pred)
            elif metric == "precision":
                worst_pred_binary = (worst_pred > 0.5).astype(int)
                remaining_pred_binary = (remaining_pred > 0.5).astype(int)
                worst_metric = precision_score(y_worst, worst_pred_binary)
                remaining_metric = precision_score(y_remaining, remaining_pred_binary)
            elif metric == "recall":
                worst_pred_binary = (worst_pred > 0.5).astype(int)
                remaining_pred_binary = (remaining_pred > 0.5).astype(int)
                worst_metric = recall_score(y_worst, worst_pred_binary)
                remaining_metric = recall_score(y_remaining, remaining_pred_binary)
            elif metric == "f1":
                worst_pred_binary = (worst_pred > 0.5).astype(int)
                remaining_pred_binary = (remaining_pred > 0.5).astype(int)
                worst_metric = f1_score(y_worst, worst_pred_binary)
                remaining_metric = f1_score(y_remaining, remaining_pred_binary)
            elif metric == "accuracy":
                worst_pred_binary = (worst_pred > 0.5).astype(int)
                remaining_pred_binary = (remaining_pred > 0.5).astype(int)
                worst_metric = accuracy_score(y_worst, worst_pred_binary)
                remaining_metric = accuracy_score(y_remaining, remaining_pred_binary)
            else:
                raise ValueError(f"Unsupported metric for classification: {metric}")
        else:  # regression
            worst_pred = model.predict(X_worst_full)
            remaining_pred = model.predict(X_remaining_full)
            
            if metric == "mse":
                worst_metric = mean_squared_error(y_worst, worst_pred)
                remaining_metric = mean_squared_error(y_remaining, remaining_pred)
            elif metric == "mae":
                worst_metric = mean_absolute_error(y_worst, worst_pred)
                remaining_metric = mean_absolute_error(y_remaining, remaining_pred)
            elif metric == "r2":
                worst_metric = r2_score(y_worst, worst_pred)
                remaining_metric = r2_score(y_remaining, remaining_pred)
            elif metric == "smape":
                # Symmetric Mean Absolute Percentage Error
                worst_metric = np.mean(np.abs(y_worst - worst_pred) / ((np.abs(y_worst) + np.abs(worst_pred)) / 2)) * 100
                remaining_metric = np.mean(np.abs(y_remaining - remaining_pred) / ((np.abs(y_remaining) + np.abs(remaining_pred)) / 2)) * 100
            else:
                raise ValueError(f"Unsupported metric for regression: {metric}")
        
        # Calculate performance gap
        performance_gap = remaining_metric - worst_metric
        
        # Calculate feature distribution shift
        feature_distances = self._calculate_feature_distances(
            worst_samples, remaining_samples, distance_metric
        )
        
        # Return detailed results
        return {
            "method": "distribution_shift",
            "alpha": alpha,
            "metric": metric,
            "distance_metric": distance_metric,
            "worst_metric": worst_metric,
            "remaining_metric": remaining_metric,
            "performance_gap": performance_gap,
            "feature_distances": feature_distances,
            "worst_sample_count": len(worst_samples),
            "remaining_sample_count": len(remaining_samples)
        }

    def evaluate_worst_sample(self, method: str, params: Dict) -> Dict[str, Any]:
        """
        Evaluate model performance on worst-performing samples.

        Unlike distribution_shift which uses feature distance metrics,
        this directly selects samples with highest prediction errors.

        Parameters:
        -----------
        method : str
            Method to use ('worst_sample')
        params : Dict
            Parameters including:
            - alpha: Ratio of worst samples to select (0 < alpha < 1)
            - metric: Performance metric to use
            - ranking_method: How to rank samples ('residual', 'entropy', 'margin')

        Returns:
        --------
        dict : Detailed evaluation results
        """
        # Get parameters
        alpha = params.get('alpha', 0.1)
        metric = params.get('metric', 'auc')
        ranking_method = params.get('ranking_method', 'residual')

        # Get dataset
        X = self.dataset.get_feature_data()
        y = self.dataset.get_target_data()

        # Convert to pandas if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        # Get model
        model = self.dataset.model

        # Get predictions
        if self._problem_type == "classification" and hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X)
            y_pred = y_pred_proba[:, 1]
        else:
            y_pred = model.predict(X)
            if self._problem_type == "classification":
                y_pred_proba = np.column_stack([1 - y_pred, y_pred])  # Create proba array for binary classification

        # Rank samples by error
        if ranking_method == 'residual':
            # Use absolute prediction error
            errors = np.abs(y.values - y_pred)
        elif ranking_method == 'entropy':
            # Use prediction entropy (for classification only)
            if self._problem_type == "classification":
                # Calculate entropy: -sum(p * log(p))
                epsilon = 1e-10
                y_pred_proba_safe = np.clip(y_pred_proba, epsilon, 1 - epsilon)
                errors = -np.sum(y_pred_proba_safe * np.log(y_pred_proba_safe), axis=1)
            else:
                raise ValueError("Entropy ranking only available for classification")
        elif ranking_method == 'margin':
            # Use prediction margin (for classification only)
            if self._problem_type == "classification":
                # Margin = difference between top two class probabilities
                sorted_proba = np.sort(y_pred_proba, axis=1)
                errors = -(sorted_proba[:, -1] - sorted_proba[:, -2])  # Negative so higher margin = lower error
            else:
                raise ValueError("Margin ranking only available for classification")
        else:
            raise ValueError(f"Unknown ranking method: {ranking_method}")

        # Sort indices by error (descending)
        sorted_indices = np.argsort(-errors)

        # Select worst samples
        n_worst = max(1, min(int(alpha * len(X)), len(X) - 1))
        worst_indices = sorted_indices[:n_worst]
        remaining_indices = sorted_indices[n_worst:]

        # Split data
        X_worst = X.iloc[worst_indices]
        X_remaining = X.iloc[remaining_indices]
        y_worst = y.iloc[worst_indices]
        y_remaining = y.iloc[remaining_indices]

        # Calculate performance metrics
        if self._problem_type == "classification":
            if hasattr(model, "predict_proba"):
                worst_pred = model.predict_proba(X_worst)[:, 1]
                remaining_pred = model.predict_proba(X_remaining)[:, 1]
            else:
                worst_pred = model.predict(X_worst)
                remaining_pred = model.predict(X_remaining)

            if metric == "auc":
                # Check if we have both classes
                if len(np.unique(y_worst)) < 2:
                    worst_metric = np.nan
                else:
                    worst_metric = roc_auc_score(y_worst, worst_pred)
                if len(np.unique(y_remaining)) < 2:
                    remaining_metric = np.nan
                else:
                    remaining_metric = roc_auc_score(y_remaining, remaining_pred)
            elif metric == "f1":
                worst_pred_binary = (worst_pred > 0.5).astype(int)
                remaining_pred_binary = (remaining_pred > 0.5).astype(int)
                worst_metric = f1_score(y_worst, worst_pred_binary, zero_division=0)
                remaining_metric = f1_score(y_remaining, remaining_pred_binary, zero_division=0)
            elif metric == "accuracy":
                worst_pred_binary = (worst_pred > 0.5).astype(int)
                remaining_pred_binary = (remaining_pred > 0.5).astype(int)
                worst_metric = accuracy_score(y_worst, worst_pred_binary)
                remaining_metric = accuracy_score(y_remaining, remaining_pred_binary)
            elif metric == "precision":
                worst_pred_binary = (worst_pred > 0.5).astype(int)
                remaining_pred_binary = (remaining_pred > 0.5).astype(int)
                worst_metric = precision_score(y_worst, worst_pred_binary, zero_division=0)
                remaining_metric = precision_score(y_remaining, remaining_pred_binary, zero_division=0)
            elif metric == "recall":
                worst_pred_binary = (worst_pred > 0.5).astype(int)
                remaining_pred_binary = (remaining_pred > 0.5).astype(int)
                worst_metric = recall_score(y_worst, worst_pred_binary, zero_division=0)
                remaining_metric = recall_score(y_remaining, remaining_pred_binary, zero_division=0)
            else:
                raise ValueError(f"Unsupported metric: {metric}")
        else:  # regression
            worst_pred = model.predict(X_worst)
            remaining_pred = model.predict(X_remaining)

            if metric == "mse":
                worst_metric = mean_squared_error(y_worst, worst_pred)
                remaining_metric = mean_squared_error(y_remaining, remaining_pred)
            elif metric == "mae":
                worst_metric = mean_absolute_error(y_worst, worst_pred)
                remaining_metric = mean_absolute_error(y_remaining, remaining_pred)
            elif metric == "r2":
                worst_metric = r2_score(y_worst, worst_pred)
                remaining_metric = r2_score(y_remaining, remaining_pred)
            else:
                raise ValueError(f"Unsupported metric: {metric}")

        # Calculate performance gap
        if not np.isnan(worst_metric) and not np.isnan(remaining_metric):
            performance_gap = remaining_metric - worst_metric
        else:
            performance_gap = np.nan

        # Calculate feature statistics for worst samples
        feature_statistics = {}
        for col in X.columns:
            if np.issubdtype(X[col].dtype, np.number):
                feature_statistics[col] = {
                    'worst_mean': float(X_worst[col].mean()),
                    'worst_std': float(X_worst[col].std()),
                    'remaining_mean': float(X_remaining[col].mean()),
                    'remaining_std': float(X_remaining[col].std()),
                    'mean_diff': float(X_worst[col].mean() - X_remaining[col].mean())
                }

        # Sort features by mean difference
        sorted_features = sorted(
            feature_statistics.items(),
            key=lambda x: abs(x[1]['mean_diff']),
            reverse=True
        )
        top_features = dict(sorted_features[:10])

        # Return results
        return {
            "method": "worst_sample",
            "alpha": alpha,
            "metric": metric,
            "ranking_method": ranking_method,
            "worst_metric": float(worst_metric) if not np.isnan(worst_metric) else None,
            "remaining_metric": float(remaining_metric) if not np.isnan(remaining_metric) else None,
            "performance_gap": float(performance_gap) if not np.isnan(performance_gap) else None,
            "worst_indices": worst_indices.tolist(),
            "worst_errors": errors[worst_indices].tolist(),
            "worst_sample_count": len(worst_indices),
            "remaining_sample_count": len(remaining_indices),
            "feature_statistics": feature_statistics,
            "top_features": top_features
        }

    def evaluate_worst_cluster(self, method: str, params: Dict) -> Dict[str, Any]:
        """
        Identify worst-performing cluster of samples using K-means clustering.

        Parameters:
        -----------
        method : str
            Method to use ('worst_cluster')
        params : Dict
            Parameters including:
            - n_clusters: Number of clusters to create
            - metric: Performance metric to use
            - random_state: Random seed for reproducibility

        Returns:
        --------
        dict : Detailed evaluation results
        """
        from sklearn.cluster import KMeans

        # Get parameters
        n_clusters = params.get('n_clusters', 5)
        metric = params.get('metric', 'auc')
        random_state = params.get('random_state', self.random_state)

        # Get dataset
        X = self.dataset.get_feature_data()
        y = self.dataset.get_target_data()

        # Convert to pandas if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        # Get model
        model = self.dataset.model

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        cluster_centers = kmeans.cluster_centers_

        # Calculate performance for each cluster
        cluster_metrics = []
        cluster_sizes = []

        for cluster_id in range(n_clusters):
            # Get samples in this cluster
            cluster_mask = cluster_labels == cluster_id
            X_cluster = X[cluster_mask]
            y_cluster = y[cluster_mask]

            cluster_sizes.append(len(X_cluster))

            # Skip if cluster is too small
            if len(X_cluster) < 2:
                cluster_metrics.append(np.nan)
                continue

            # Calculate performance on this cluster
            if self._problem_type == "classification":
                if hasattr(model, "predict_proba"):
                    y_pred = model.predict_proba(X_cluster)[:, 1]
                else:
                    y_pred = model.predict(X_cluster)

                if metric == "auc":
                    # Check if we have both classes in cluster
                    if len(np.unique(y_cluster)) < 2:
                        cluster_metric = np.nan
                    else:
                        cluster_metric = roc_auc_score(y_cluster, y_pred)
                elif metric == "f1":
                    y_pred_binary = (y_pred > 0.5).astype(int)
                    cluster_metric = f1_score(y_cluster, y_pred_binary, zero_division=0)
                elif metric == "accuracy":
                    y_pred_binary = (y_pred > 0.5).astype(int)
                    cluster_metric = accuracy_score(y_cluster, y_pred_binary)
                elif metric == "precision":
                    y_pred_binary = (y_pred > 0.5).astype(int)
                    cluster_metric = precision_score(y_cluster, y_pred_binary, zero_division=0)
                elif metric == "recall":
                    y_pred_binary = (y_pred > 0.5).astype(int)
                    cluster_metric = recall_score(y_cluster, y_pred_binary, zero_division=0)
                else:
                    raise ValueError(f"Unsupported metric: {metric}")
            else:  # regression
                y_pred = model.predict(X_cluster)

                if metric == "mse":
                    cluster_metric = mean_squared_error(y_cluster, y_pred)
                elif metric == "mae":
                    cluster_metric = mean_absolute_error(y_cluster, y_pred)
                elif metric == "r2":
                    cluster_metric = r2_score(y_cluster, y_pred)
                else:
                    raise ValueError(f"Unsupported metric: {metric}")

            cluster_metrics.append(cluster_metric)

        # Find worst cluster (lowest metric for classification, highest for regression error metrics)
        valid_metrics = [(i, m) for i, m in enumerate(cluster_metrics) if not np.isnan(m)]

        if not valid_metrics:
            # All clusters have invalid metrics, return null results
            return {
                "method": "worst_cluster",
                "n_clusters": n_clusters,
                "metric": metric,
                "worst_cluster_id": None,
                "worst_cluster_metric": None,
                "remaining_metric": None,
                "performance_gap": None,
                "worst_cluster_size": 0,
                "remaining_size": len(X),
                "cluster_centers": cluster_centers.tolist(),
                "cluster_sizes": cluster_sizes,
                "cluster_metrics": [float(m) if not np.isnan(m) else None for m in cluster_metrics],
                "feature_importance": {},
                "top_features": {}
            }

        if metric in ['mse', 'mae']:  # Higher is worse for error metrics
            worst_cluster_id = max(valid_metrics, key=lambda x: x[1])[0]
        else:  # Lower is worse for score metrics
            worst_cluster_id = min(valid_metrics, key=lambda x: x[1])[0]

        worst_cluster_metric = cluster_metrics[worst_cluster_id]

        # Calculate performance on remaining clusters
        remaining_mask = cluster_labels != worst_cluster_id
        X_remaining = X[remaining_mask]
        y_remaining = y[remaining_mask]

        if len(X_remaining) < 2:
            remaining_metric = np.nan
            performance_gap = np.nan
        else:
            if self._problem_type == "classification":
                if hasattr(model, "predict_proba"):
                    y_pred_remaining = model.predict_proba(X_remaining)[:, 1]
                else:
                    y_pred_remaining = model.predict(X_remaining)

                if metric == "auc":
                    if len(np.unique(y_remaining)) < 2:
                        remaining_metric = np.nan
                    else:
                        remaining_metric = roc_auc_score(y_remaining, y_pred_remaining)
                elif metric == "f1":
                    y_pred_binary = (y_pred_remaining > 0.5).astype(int)
                    remaining_metric = f1_score(y_remaining, y_pred_binary, zero_division=0)
                elif metric == "accuracy":
                    y_pred_binary = (y_pred_remaining > 0.5).astype(int)
                    remaining_metric = accuracy_score(y_remaining, y_pred_binary)
                elif metric == "precision":
                    y_pred_binary = (y_pred_remaining > 0.5).astype(int)
                    remaining_metric = precision_score(y_remaining, y_pred_binary, zero_division=0)
                elif metric == "recall":
                    y_pred_binary = (y_pred_remaining > 0.5).astype(int)
                    remaining_metric = recall_score(y_remaining, y_pred_binary, zero_division=0)
            else:  # regression
                y_pred_remaining = model.predict(X_remaining)

                if metric == "mse":
                    remaining_metric = mean_squared_error(y_remaining, y_pred_remaining)
                elif metric == "mae":
                    remaining_metric = mean_absolute_error(y_remaining, y_pred_remaining)
                elif metric == "r2":
                    remaining_metric = r2_score(y_remaining, y_pred_remaining)

            # Calculate performance gap
            if not np.isnan(worst_cluster_metric) and not np.isnan(remaining_metric):
                if metric in ['mse', 'mae']:  # Error metrics
                    performance_gap = worst_cluster_metric - remaining_metric
                else:  # Score metrics
                    performance_gap = remaining_metric - worst_cluster_metric
            else:
                performance_gap = np.nan

        # Calculate feature importance (features that define worst cluster)
        worst_cluster_mask = cluster_labels == worst_cluster_id
        X_worst_cluster = X[worst_cluster_mask]

        feature_importance = {}
        for col in X.columns:
            if np.issubdtype(X[col].dtype, np.number):
                # Calculate how much this feature differs in worst cluster vs others
                worst_mean = X_worst_cluster[col].mean()
                overall_std = X[col].std()

                if len(X_remaining) > 0:
                    remaining_mean = X_remaining[col].mean()
                    # Normalized difference
                    if overall_std > 0:
                        importance = abs(worst_mean - remaining_mean) / overall_std
                    else:
                        importance = 0
                else:
                    importance = 0

                feature_importance[col] = float(importance)

        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = dict(sorted_features[:10])

        # Return results
        return {
            "method": "worst_cluster",
            "n_clusters": n_clusters,
            "metric": metric,
            "worst_cluster_id": int(worst_cluster_id),
            "worst_cluster_metric": float(worst_cluster_metric) if not np.isnan(worst_cluster_metric) else None,
            "remaining_metric": float(remaining_metric) if not np.isnan(remaining_metric) else None,
            "performance_gap": float(performance_gap) if not np.isnan(performance_gap) else None,
            "worst_cluster_size": cluster_sizes[worst_cluster_id],
            "remaining_size": sum(cluster_sizes) - cluster_sizes[worst_cluster_id],
            "cluster_centers": cluster_centers.tolist(),
            "cluster_sizes": cluster_sizes,
            "cluster_metrics": [float(m) if not np.isnan(m) else None for m in cluster_metrics],
            "feature_importance": feature_importance,
            "top_features": top_features
        }

    def evaluate_outer_sample(self, method: str, params: Dict) -> Dict[str, Any]:
        """
        Evaluate model performance on boundary/outlier samples.

        Parameters:
        -----------
        method : str
            Method to use ('outer_sample')
        params : Dict
            Parameters including:
            - alpha: Ratio of outer samples to select
            - metric: Performance metric to use
            - outlier_method: Detection method ('isolation_forest', 'lof')

        Returns:
        --------
        dict : Detailed evaluation results
        """
        from sklearn.ensemble import IsolationForest
        from sklearn.neighbors import LocalOutlierFactor

        # Get parameters
        alpha = params.get('alpha', 0.05)
        metric = params.get('metric', 'auc')
        outlier_method = params.get('outlier_method', 'isolation_forest')
        random_state = params.get('random_state', self.random_state)

        # Get dataset
        X = self.dataset.get_feature_data()
        y = self.dataset.get_target_data()

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        # Get model
        model = self.dataset.model

        # Calculate outlier scores
        if outlier_method == 'isolation_forest':
            outlier_detector = IsolationForest(
                contamination=alpha,
                random_state=random_state,
                n_estimators=100
            )
            outlier_detector.fit(X)
            outlier_scores = -outlier_detector.score_samples(X)  # Negative so higher = more outlier

        elif outlier_method == 'lof':
            n_neighbors = min(20, len(X) - 1)
            outlier_detector = LocalOutlierFactor(
                n_neighbors=n_neighbors,
                contamination=alpha
            )
            outlier_scores = -outlier_detector.fit(X).negative_outlier_factor_

        else:
            raise ValueError(f"Unknown outlier method: {outlier_method}")

        # Select outer samples (highest outlier scores)
        n_outer = max(1, min(int(alpha * len(X)), len(X) - 1))
        sorted_indices = np.argsort(-outlier_scores)
        outer_indices = sorted_indices[:n_outer]
        inner_indices = sorted_indices[n_outer:]

        # Split data
        X_outer = X.iloc[outer_indices]
        X_inner = X.iloc[inner_indices]
        y_outer = y.iloc[outer_indices]
        y_inner = y.iloc[inner_indices]

        # Calculate performance metrics
        if self._problem_type == "classification":
            if hasattr(model, "predict_proba"):
                outer_pred = model.predict_proba(X_outer)[:, 1]
                inner_pred = model.predict_proba(X_inner)[:, 1]
            else:
                outer_pred = model.predict(X_outer)
                inner_pred = model.predict(X_inner)

            if metric == "auc":
                # Check if we have both classes
                if len(np.unique(y_outer)) < 2:
                    outer_metric = np.nan
                else:
                    outer_metric = roc_auc_score(y_outer, outer_pred)
                if len(np.unique(y_inner)) < 2:
                    inner_metric = np.nan
                else:
                    inner_metric = roc_auc_score(y_inner, inner_pred)
            elif metric == "f1":
                outer_pred_binary = (outer_pred > 0.5).astype(int)
                inner_pred_binary = (inner_pred > 0.5).astype(int)
                outer_metric = f1_score(y_outer, outer_pred_binary, zero_division=0)
                inner_metric = f1_score(y_inner, inner_pred_binary, zero_division=0)
            elif metric == "accuracy":
                outer_pred_binary = (outer_pred > 0.5).astype(int)
                inner_pred_binary = (inner_pred > 0.5).astype(int)
                outer_metric = accuracy_score(y_outer, outer_pred_binary)
                inner_metric = accuracy_score(y_inner, inner_pred_binary)
            elif metric == "precision":
                outer_pred_binary = (outer_pred > 0.5).astype(int)
                inner_pred_binary = (inner_pred > 0.5).astype(int)
                outer_metric = precision_score(y_outer, outer_pred_binary, zero_division=0)
                inner_metric = precision_score(y_inner, inner_pred_binary, zero_division=0)
            elif metric == "recall":
                outer_pred_binary = (outer_pred > 0.5).astype(int)
                inner_pred_binary = (inner_pred > 0.5).astype(int)
                outer_metric = recall_score(y_outer, outer_pred_binary, zero_division=0)
                inner_metric = recall_score(y_inner, inner_pred_binary, zero_division=0)
            else:
                raise ValueError(f"Unsupported metric: {metric}")
        else:  # regression
            outer_pred = model.predict(X_outer)
            inner_pred = model.predict(X_inner)

            if metric == "mse":
                outer_metric = mean_squared_error(y_outer, outer_pred)
                inner_metric = mean_squared_error(y_inner, inner_pred)
            elif metric == "mae":
                outer_metric = mean_absolute_error(y_outer, outer_pred)
                inner_metric = mean_absolute_error(y_inner, inner_pred)
            elif metric == "r2":
                outer_metric = r2_score(y_outer, outer_pred)
                inner_metric = r2_score(y_inner, inner_pred)
            else:
                raise ValueError(f"Unsupported metric: {metric}")

        # Calculate performance gap
        if not np.isnan(outer_metric) and not np.isnan(inner_metric):
            if metric in ['mse', 'mae']:  # Error metrics
                performance_gap = outer_metric - inner_metric
            else:  # Score metrics
                performance_gap = inner_metric - outer_metric
        else:
            performance_gap = np.nan

        # Calculate feature deviations
        feature_deviations = {}
        for col in X.columns:
            if np.issubdtype(X[col].dtype, np.number):
                outer_mean = X_outer[col].mean()
                inner_mean = X_inner[col].mean()
                overall_std = X[col].std()

                feature_deviations[col] = {
                    'outer_mean': float(outer_mean),
                    'inner_mean': float(inner_mean),
                    'deviation': float((outer_mean - inner_mean) / overall_std if overall_std > 0 else 0)
                }

        # Sort features by absolute deviation
        sorted_deviations = sorted(
            feature_deviations.items(),
            key=lambda x: abs(x[1]['deviation']),
            reverse=True
        )
        top_features = dict(sorted_deviations[:10])

        # Return results
        return {
            "method": "outer_sample",
            "alpha": alpha,
            "metric": metric,
            "outlier_detection_method": outlier_method,
            "outer_metric": float(outer_metric) if not np.isnan(outer_metric) else None,
            "inner_metric": float(inner_metric) if not np.isnan(inner_metric) else None,
            "performance_gap": float(performance_gap) if not np.isnan(performance_gap) else None,
            "outer_indices": outer_indices.tolist(),
            "outer_scores": outlier_scores[outer_indices].tolist(),
            "outer_sample_count": len(outer_indices),
            "inner_sample_count": len(inner_indices),
            "feature_deviations": feature_deviations,
            "top_features": top_features
        }

    def evaluate_hard_sample(self, method: str, params: Dict) -> Dict[str, Any]:
        """
        Evaluate model performance on intrinsically hard samples identified by
        model disagreement.

        Requires alternative models to be available in the dataset.

        Parameters:
        -----------
        method : str
            Method to use ('hard_sample')
        params : Dict
            Parameters including:
            - disagreement_threshold: Threshold for considering samples as hard
            - metric: Performance metric to use
            - auxiliary_models: List of model names to compare with

        Returns:
        --------
        dict : Detailed evaluation results
        """
        # Get parameters
        disagreement_threshold = params.get('disagreement_threshold', 0.3)
        metric = params.get('metric', 'auc')
        auxiliary_model_names = params.get('auxiliary_models', None)

        # Get dataset
        X = self.dataset.get_feature_data()
        y = self.dataset.get_target_data()

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        # Get primary model
        primary_model = self.dataset.model

        # Get auxiliary models
        if not hasattr(self.dataset, 'alternative_models'):
            # If no alternative models available, return null results
            return {
                "method": "hard_sample",
                "disagreement_threshold": disagreement_threshold,
                "metric": metric,
                "auxiliary_models": [],
                "hard_metric": None,
                "easy_metric": None,
                "performance_gap": None,
                "hard_indices": [],
                "disagreement_scores": [],
                "hard_sample_count": 0,
                "easy_sample_count": len(X),
                "model_predictions": {},
                "feature_complexity": {}
            }

        alternative_models = self.dataset.alternative_models

        if auxiliary_model_names is None:
            auxiliary_model_names = list(alternative_models.keys())

        # Get predictions from all models
        model_predictions = {}

        if self._problem_type == "classification":
            if hasattr(primary_model, "predict_proba"):
                model_predictions['primary'] = primary_model.predict_proba(X)[:, 1]
            else:
                model_predictions['primary'] = primary_model.predict(X)

            for model_name in auxiliary_model_names:
                if model_name in alternative_models:
                    aux_model = alternative_models[model_name]
                    if hasattr(aux_model, "predict_proba"):
                        model_predictions[model_name] = aux_model.predict_proba(X)[:, 1]
                    else:
                        model_predictions[model_name] = aux_model.predict(X)
        else:  # regression
            model_predictions['primary'] = primary_model.predict(X)

            for model_name in auxiliary_model_names:
                if model_name in alternative_models:
                    model_predictions[model_name] = alternative_models[model_name].predict(X)

        # Need at least 2 models to calculate disagreement
        if len(model_predictions) < 2:
            return {
                "method": "hard_sample",
                "disagreement_threshold": disagreement_threshold,
                "metric": metric,
                "auxiliary_models": list(model_predictions.keys()),
                "hard_metric": None,
                "easy_metric": None,
                "performance_gap": None,
                "hard_indices": [],
                "disagreement_scores": [],
                "hard_sample_count": 0,
                "easy_sample_count": len(X),
                "model_predictions": {k: v.tolist() for k, v in model_predictions.items()},
                "feature_complexity": {}
            }

        # Calculate disagreement scores
        predictions_array = np.array([model_predictions[name] for name in model_predictions.keys()])

        # Use standard deviation across models as disagreement metric
        disagreement_scores = np.std(predictions_array, axis=0)

        # Select hard samples (high disagreement)
        hard_threshold = np.percentile(disagreement_scores, (1 - disagreement_threshold) * 100)
        hard_mask = disagreement_scores >= hard_threshold
        easy_mask = ~hard_mask

        hard_indices = np.where(hard_mask)[0]
        easy_indices = np.where(easy_mask)[0]

        # Ensure we have samples in both groups
        if len(hard_indices) == 0 or len(easy_indices) == 0:
            return {
                "method": "hard_sample",
                "disagreement_threshold": disagreement_threshold,
                "metric": metric,
                "auxiliary_models": list(model_predictions.keys()),
                "hard_metric": None,
                "easy_metric": None,
                "performance_gap": None,
                "hard_indices": hard_indices.tolist(),
                "disagreement_scores": disagreement_scores.tolist(),
                "hard_sample_count": len(hard_indices),
                "easy_sample_count": len(easy_indices),
                "model_predictions": {k: v.tolist() for k, v in model_predictions.items()},
                "feature_complexity": {}
            }

        # Split data
        X_hard = X.iloc[hard_indices]
        X_easy = X.iloc[easy_indices]
        y_hard = y.iloc[hard_indices]
        y_easy = y.iloc[easy_indices]

        # Calculate performance metrics on primary model
        if self._problem_type == "classification":
            if hasattr(primary_model, "predict_proba"):
                hard_pred = primary_model.predict_proba(X_hard)[:, 1]
                easy_pred = primary_model.predict_proba(X_easy)[:, 1]
            else:
                hard_pred = primary_model.predict(X_hard)
                easy_pred = primary_model.predict(X_easy)

            if metric == "auc":
                # Check if we have both classes
                if len(np.unique(y_hard)) < 2:
                    hard_metric = np.nan
                else:
                    hard_metric = roc_auc_score(y_hard, hard_pred)
                if len(np.unique(y_easy)) < 2:
                    easy_metric = np.nan
                else:
                    easy_metric = roc_auc_score(y_easy, easy_pred)
            elif metric == "f1":
                hard_pred_binary = (hard_pred > 0.5).astype(int)
                easy_pred_binary = (easy_pred > 0.5).astype(int)
                hard_metric = f1_score(y_hard, hard_pred_binary, zero_division=0)
                easy_metric = f1_score(y_easy, easy_pred_binary, zero_division=0)
            elif metric == "accuracy":
                hard_pred_binary = (hard_pred > 0.5).astype(int)
                easy_pred_binary = (easy_pred > 0.5).astype(int)
                hard_metric = accuracy_score(y_hard, hard_pred_binary)
                easy_metric = accuracy_score(y_easy, easy_pred_binary)
            elif metric == "precision":
                hard_pred_binary = (hard_pred > 0.5).astype(int)
                easy_pred_binary = (easy_pred > 0.5).astype(int)
                hard_metric = precision_score(y_hard, hard_pred_binary, zero_division=0)
                easy_metric = precision_score(y_easy, easy_pred_binary, zero_division=0)
            elif metric == "recall":
                hard_pred_binary = (hard_pred > 0.5).astype(int)
                easy_pred_binary = (easy_pred > 0.5).astype(int)
                hard_metric = recall_score(y_hard, hard_pred_binary, zero_division=0)
                easy_metric = recall_score(y_easy, easy_pred_binary, zero_division=0)
            else:
                raise ValueError(f"Unsupported metric: {metric}")
        else:  # regression
            hard_pred = primary_model.predict(X_hard)
            easy_pred = primary_model.predict(X_easy)

            if metric == "mse":
                hard_metric = mean_squared_error(y_hard, hard_pred)
                easy_metric = mean_squared_error(y_easy, easy_pred)
            elif metric == "mae":
                hard_metric = mean_absolute_error(y_hard, hard_pred)
                easy_metric = mean_absolute_error(y_easy, easy_pred)
            elif metric == "r2":
                hard_metric = r2_score(y_hard, hard_pred)
                easy_metric = r2_score(y_easy, easy_pred)
            else:
                raise ValueError(f"Unsupported metric: {metric}")

        # Calculate performance gap
        if not np.isnan(hard_metric) and not np.isnan(easy_metric):
            if metric in ['mse', 'mae']:  # Error metrics
                performance_gap = hard_metric - easy_metric
            else:  # Score metrics
                performance_gap = easy_metric - hard_metric
        else:
            performance_gap = np.nan

        # Calculate feature complexity (variance in hard samples)
        feature_complexity = {}
        for col in X.columns:
            if np.issubdtype(X[col].dtype, np.number):
                hard_var = X_hard[col].var()
                easy_var = X_easy[col].var()

                feature_complexity[col] = {
                    'hard_variance': float(hard_var),
                    'easy_variance': float(easy_var),
                    'variance_ratio': float(hard_var / easy_var if easy_var > 0 else 0)
                }

        # Sort features by variance ratio
        sorted_complexity = sorted(
            feature_complexity.items(),
            key=lambda x: x[1]['variance_ratio'],
            reverse=True
        )
        top_features = dict(sorted_complexity[:10])

        # Return results
        return {
            "method": "hard_sample",
            "disagreement_threshold": disagreement_threshold,
            "metric": metric,
            "auxiliary_models": list(model_predictions.keys()),
            "hard_metric": float(hard_metric) if not np.isnan(hard_metric) else None,
            "easy_metric": float(easy_metric) if not np.isnan(easy_metric) else None,
            "performance_gap": float(performance_gap) if not np.isnan(performance_gap) else None,
            "hard_indices": hard_indices.tolist(),
            "disagreement_scores": disagreement_scores.tolist(),
            "hard_sample_count": len(hard_indices),
            "easy_sample_count": len(easy_indices),
            "model_predictions": {k: v.tolist() for k, v in model_predictions.items()},
            "feature_complexity": feature_complexity,
            "top_features": top_features
        }

    def run(self) -> Dict[str, Any]:
        """
        Run the configured resilience tests.
        
        Returns:
        --------
        dict : Test results with detailed performance metrics
        """
        if self.current_config is None:
            # Default to quick config if none selected
            if self.verbose:
                print("No configuration set, using 'quick' configuration")
            self.config('quick')
                
        if self.verbose:
            print(f"Running resilience test suite...")
            start_time = time.time()
        
        # Initialize results
        results = {
            'distribution_shift': {
                'by_alpha': {},           # Results organized by alpha level
                'by_distance_metric': {}, # Results organized by distance metric
                'by_metric': {},          # Results organized by performance metric
                'all_results': []         # All raw test results
            },
            'worst_sample': {
                'by_alpha': {},           # Results organized by alpha level
                'all_results': []         # All raw test results
            },
            'worst_cluster': {
                'by_n_clusters': {},      # Results organized by number of clusters
                'all_results': []         # All raw test results
            },
            'outer_sample': {
                'by_alpha': {},           # Results organized by alpha level
                'all_results': []         # All raw test results
            },
            'hard_sample': {
                'by_threshold': {},       # Results organized by disagreement threshold
                'all_results': []         # All raw test results
            }
        }
        
        # Track parameters for summary
        all_alphas = []
        all_distance_metrics = []
        all_metrics = []
        
        # Run all configured tests
        for test_config in self.current_config:
            method = test_config['method']
            params = test_config.get('params', {})
            
            if method == 'distribution_shift':
                alpha = params.get('alpha', 0.3)
                metric = params.get('metric', 'auc')
                distance_metric = params.get('distance_metric', 'PSI')
                
                # Track parameters
                if alpha not in all_alphas:
                    all_alphas.append(alpha)
                if distance_metric not in all_distance_metrics:
                    all_distance_metrics.append(distance_metric)
                if metric not in all_metrics:
                    all_metrics.append(metric)
                
                if self.verbose:
                    print(f"Running distribution shift analysis with alpha={alpha}, " 
                          f"metric={metric}, distance_metric={distance_metric}")
                
                # Run the resilience evaluation
                test_result = self.evaluate_distribution_shift(method, params)
                results['distribution_shift']['all_results'].append(test_result)
                
                # Organize results by alpha
                if alpha not in results['distribution_shift']['by_alpha']:
                    results['distribution_shift']['by_alpha'][alpha] = []
                results['distribution_shift']['by_alpha'][alpha].append(test_result)
                
                # Organize results by distance metric
                if distance_metric not in results['distribution_shift']['by_distance_metric']:
                    results['distribution_shift']['by_distance_metric'][distance_metric] = []
                results['distribution_shift']['by_distance_metric'][distance_metric].append(test_result)
                
                # Organize results by performance metric
                if metric not in results['distribution_shift']['by_metric']:
                    results['distribution_shift']['by_metric'][metric] = []
                results['distribution_shift']['by_metric'][metric].append(test_result)

            elif method == 'worst_sample':
                alpha = params.get('alpha', 0.1)
                metric = params.get('metric', 'auc')
                ranking_method = params.get('ranking_method', 'residual')

                # Track parameters
                if alpha not in all_alphas:
                    all_alphas.append(alpha)
                if metric not in all_metrics:
                    all_metrics.append(metric)

                if self.verbose:
                    print(f"Running worst-sample analysis with alpha={alpha}, "
                          f"metric={metric}, ranking={ranking_method}")

                # Run the worst-sample evaluation
                test_result = self.evaluate_worst_sample(method, params)
                results['worst_sample']['all_results'].append(test_result)

                # Organize results by alpha
                if alpha not in results['worst_sample']['by_alpha']:
                    results['worst_sample']['by_alpha'][alpha] = []
                results['worst_sample']['by_alpha'][alpha].append(test_result)

            elif method == 'worst_cluster':
                n_clusters = params.get('n_clusters', 5)
                metric = params.get('metric', 'auc')
                random_state = params.get('random_state', self.random_state)

                # Track parameters
                if metric not in all_metrics:
                    all_metrics.append(metric)

                if self.verbose:
                    print(f"Running worst-cluster analysis with n_clusters={n_clusters}, "
                          f"metric={metric}")

                # Run the worst-cluster evaluation
                test_result = self.evaluate_worst_cluster(method, params)
                results['worst_cluster']['all_results'].append(test_result)

                # Organize results by n_clusters
                if n_clusters not in results['worst_cluster']['by_n_clusters']:
                    results['worst_cluster']['by_n_clusters'][n_clusters] = []
                results['worst_cluster']['by_n_clusters'][n_clusters].append(test_result)

            elif method == 'outer_sample':
                alpha = params.get('alpha', 0.05)
                metric = params.get('metric', 'auc')
                outlier_method = params.get('outlier_method', 'isolation_forest')

                # Track parameters
                if alpha not in all_alphas:
                    all_alphas.append(alpha)
                if metric not in all_metrics:
                    all_metrics.append(metric)

                if self.verbose:
                    print(f"Running outer-sample analysis with alpha={alpha}, "
                          f"metric={metric}, method={outlier_method}")

                # Run the outer-sample evaluation
                test_result = self.evaluate_outer_sample(method, params)
                results['outer_sample']['all_results'].append(test_result)

                # Organize results by alpha
                if alpha not in results['outer_sample']['by_alpha']:
                    results['outer_sample']['by_alpha'][alpha] = []
                results['outer_sample']['by_alpha'][alpha].append(test_result)

            elif method == 'hard_sample':
                disagreement_threshold = params.get('disagreement_threshold', 0.3)
                metric = params.get('metric', 'auc')
                auxiliary_models = params.get('auxiliary_models', None)

                # Track parameters
                if metric not in all_metrics:
                    all_metrics.append(metric)

                if self.verbose:
                    print(f"Running hard-sample analysis with threshold={disagreement_threshold}, "
                          f"metric={metric}")

                # Run the hard-sample evaluation
                test_result = self.evaluate_hard_sample(method, params)
                results['hard_sample']['all_results'].append(test_result)

                # Organize results by threshold
                if disagreement_threshold not in results['hard_sample']['by_threshold']:
                    results['hard_sample']['by_threshold'][disagreement_threshold] = []
                results['hard_sample']['by_threshold'][disagreement_threshold].append(test_result)

        # Calculate overall resilience metrics
        # For each alpha level, calculate average performance gap
        for alpha, alpha_results in results['distribution_shift']['by_alpha'].items():
            avg_performance_gap = np.mean([r['performance_gap'] for r in alpha_results])
            results['distribution_shift']['by_alpha'][alpha] = {
                'results': alpha_results,
                'avg_performance_gap': avg_performance_gap
            }
        
        # For each distance metric, find the features with highest shift
        for dm, dm_results in results['distribution_shift']['by_distance_metric'].items():
            # Combine feature distances from all tests with this distance metric
            all_feature_distances = {}
            for result in dm_results:
                feature_distances = result['feature_distances']['all_feature_distances']
                for feature, distance in feature_distances.items():
                    if feature not in all_feature_distances:
                        all_feature_distances[feature] = []
                    all_feature_distances[feature].append(distance)
            
            # Calculate average distance for each feature
            avg_feature_distances = {
                feature: np.mean(distances) 
                for feature, distances in all_feature_distances.items()
            }
            
            # Sort features by average distance
            sorted_features = sorted(avg_feature_distances.items(), 
                                   key=lambda x: x[1], reverse=True)
            
            # Store most shifted features
            results['distribution_shift']['by_distance_metric'][dm] = {
                'results': dm_results,
                'avg_feature_distances': avg_feature_distances,
                'top_features': dict(sorted_features[:10])
            }

        # Calculate worst-sample summaries
        for alpha, alpha_results in results['worst_sample']['by_alpha'].items():
            # Filter out None performance gaps
            valid_gaps = [r['performance_gap'] for r in alpha_results if r['performance_gap'] is not None]
            if valid_gaps:
                avg_performance_gap = np.mean(valid_gaps)
            else:
                avg_performance_gap = 0.0
            results['worst_sample']['by_alpha'][alpha] = {
                'results': alpha_results,
                'avg_performance_gap': avg_performance_gap
            }

        # Calculate worst-cluster summaries
        for n_clusters, cluster_results in results['worst_cluster']['by_n_clusters'].items():
            # Filter out None performance gaps
            valid_gaps = [r['performance_gap'] for r in cluster_results if r['performance_gap'] is not None]
            if valid_gaps:
                avg_performance_gap = np.mean(valid_gaps)
            else:
                avg_performance_gap = 0.0
            results['worst_cluster']['by_n_clusters'][n_clusters] = {
                'results': cluster_results,
                'avg_performance_gap': avg_performance_gap
            }

        # Calculate outer-sample summaries
        for alpha, alpha_results in results['outer_sample']['by_alpha'].items():
            # Filter out None performance gaps
            valid_gaps = [r['performance_gap'] for r in alpha_results if r['performance_gap'] is not None]
            if valid_gaps:
                avg_performance_gap = np.mean(valid_gaps)
            else:
                avg_performance_gap = 0.0
            results['outer_sample']['by_alpha'][alpha] = {
                'results': alpha_results,
                'avg_performance_gap': avg_performance_gap
            }

        # Calculate hard-sample summaries
        for threshold, threshold_results in results['hard_sample']['by_threshold'].items():
            # Filter out None performance gaps
            valid_gaps = [r['performance_gap'] for r in threshold_results if r['performance_gap'] is not None]
            if valid_gaps:
                avg_performance_gap = np.mean(valid_gaps)
            else:
                avg_performance_gap = 0.0
            results['hard_sample']['by_threshold'][threshold] = {
                'results': threshold_results,
                'avg_performance_gap': avg_performance_gap
            }

        # Calculate overall resilience score considering all test types
        all_gaps = []

        # Add distribution_shift gaps
        if results['distribution_shift']['by_alpha']:
            for alpha in results['distribution_shift']['by_alpha'].keys():
                all_gaps.append(results['distribution_shift']['by_alpha'][alpha]['avg_performance_gap'])

        # Add worst_sample gaps
        if results['worst_sample']['by_alpha']:
            for alpha in results['worst_sample']['by_alpha'].keys():
                gap = results['worst_sample']['by_alpha'][alpha]['avg_performance_gap']
                if gap is not None and not np.isnan(gap):
                    all_gaps.append(gap)

        # Add worst_cluster gaps
        if results['worst_cluster']['by_n_clusters']:
            for n_clusters in results['worst_cluster']['by_n_clusters'].keys():
                gap = results['worst_cluster']['by_n_clusters'][n_clusters]['avg_performance_gap']
                if gap is not None and not np.isnan(gap):
                    all_gaps.append(gap)

        # Add outer_sample gaps
        if results['outer_sample']['by_alpha']:
            for alpha in results['outer_sample']['by_alpha'].keys():
                gap = results['outer_sample']['by_alpha'][alpha]['avg_performance_gap']
                if gap is not None and not np.isnan(gap):
                    all_gaps.append(gap)

        # Add hard_sample gaps
        if results['hard_sample']['by_threshold']:
            for threshold in results['hard_sample']['by_threshold'].keys():
                gap = results['hard_sample']['by_threshold'][threshold]['avg_performance_gap']
                if gap is not None and not np.isnan(gap):
                    all_gaps.append(gap)

        # Calculate composite resilience score
        if all_gaps:
            results['resilience_score'] = 1.0 - min(1.0, max(0.0, np.mean(all_gaps)))
        else:
            results['resilience_score'] = 1.0

        # Add test-specific scores
        results['test_scores'] = {}
        if results['distribution_shift']['by_alpha']:
            ds_gaps = [results['distribution_shift']['by_alpha'][a]['avg_performance_gap']
                       for a in results['distribution_shift']['by_alpha'].keys()]
            if ds_gaps:
                results['test_scores']['distribution_shift'] = 1.0 - min(1.0, max(0.0, np.mean(ds_gaps)))

        if results['worst_sample']['by_alpha']:
            ws_gaps = [results['worst_sample']['by_alpha'][a]['avg_performance_gap']
                       for a in results['worst_sample']['by_alpha'].keys()
                       if results['worst_sample']['by_alpha'][a]['avg_performance_gap'] is not None]
            if ws_gaps:
                results['test_scores']['worst_sample'] = 1.0 - min(1.0, max(0.0, np.mean(ws_gaps)))

        if results['worst_cluster']['by_n_clusters']:
            wc_gaps = [results['worst_cluster']['by_n_clusters'][n]['avg_performance_gap']
                       for n in results['worst_cluster']['by_n_clusters'].keys()
                       if results['worst_cluster']['by_n_clusters'][n]['avg_performance_gap'] is not None]
            if wc_gaps:
                results['test_scores']['worst_cluster'] = 1.0 - min(1.0, max(0.0, np.mean(wc_gaps)))

        if results['outer_sample']['by_alpha']:
            os_gaps = [results['outer_sample']['by_alpha'][a]['avg_performance_gap']
                       for a in results['outer_sample']['by_alpha'].keys()
                       if results['outer_sample']['by_alpha'][a]['avg_performance_gap'] is not None]
            if os_gaps:
                results['test_scores']['outer_sample'] = 1.0 - min(1.0, max(0.0, np.mean(os_gaps)))

        if results['hard_sample']['by_threshold']:
            hs_gaps = [results['hard_sample']['by_threshold'][t]['avg_performance_gap']
                       for t in results['hard_sample']['by_threshold'].keys()
                       if results['hard_sample']['by_threshold'][t]['avg_performance_gap'] is not None]
            if hs_gaps:
                results['test_scores']['hard_sample'] = 1.0 - min(1.0, max(0.0, np.mean(hs_gaps)))

        # Store parameters
        results['alphas'] = sorted(all_alphas)
        results['distance_metrics'] = all_distance_metrics
        results['metrics'] = all_metrics
        
        # Add execution time
        if self.verbose:
            elapsed_time = time.time() - start_time
            # Não armazenamos mais o tempo de execução nos resultados
            print(f"Test suite completed in {elapsed_time:.2f} seconds")
            print(f"Overall resilience score: {results['resilience_score']:.3f}")
        
        # Store results
        test_id = f"test_{int(time.time())}"
        self.results[test_id] = results
                
        return results
    
    def save_report(self, output_path: str) -> None:
        """
        Save resilience test results to a simple text report file.
        
        Parameters:
        -----------
        output_path : str
            Path where the report should be saved
        """
        if not self.results:
            raise ValueError("No results available. Run a test first.")
        
        # Get the most recent test result
        last_test_key = list(self.results.keys())[-1]
        test_results = self.results[last_test_key]
        
        # Create a simple report
        report_lines = [
            "# Model Resilience Report",
            f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Model: {self.dataset.model.__class__.__name__}",
            f"Problem type: {self._problem_type}",
            "",
            "## Summary",
            f"Overall resilience score: {test_results.get('resilience_score', 0):.3f}",
            "",
            "## Distribution Shift Results"
        ]
        
        # Add results by alpha
        for alpha, alpha_data in sorted(test_results.get('distribution_shift', {}).get('by_alpha', {}).items()):
            report_lines.append(f"\n### Alpha = {alpha} (Worst {int(alpha*100)}% of samples)")
            report_lines.append(f"Average performance gap: {alpha_data.get('avg_performance_gap', 0):.3f}")
            
            # Add individual test results
            for i, result in enumerate(alpha_data.get('results', []), 1):
                report_lines.append(f"\n#### Test {i}")
                report_lines.append(f"Metric: {result.get('metric', '')}")
                report_lines.append(f"Distance metric: {result.get('distance_metric', '')}")
                report_lines.append(f"Worst samples {result.get('metric', '')} score: {result.get('worst_metric', 0):.3f}")
                report_lines.append(f"Remaining samples {result.get('metric', '')} score: {result.get('remaining_metric', 0):.3f}")
                report_lines.append(f"Performance gap: {result.get('performance_gap', 0):.3f}")
        
        # Add feature importance section
        report_lines.append("\n## Feature Importance by Distance Metric")
        
        # For each distance metric, show top features
        for dm, dm_data in test_results.get('distribution_shift', {}).get('by_distance_metric', {}).items():
            report_lines.append(f"\n### {dm} Distance Metric")
            
            # Sort features by importance
            top_features = sorted(dm_data.get('top_features', {}).items(), 
                                key=lambda x: x[1], reverse=True)
            
            # Limit to top 10 features
            if top_features:
                report_lines.append("Top 10 most important features:")
                for feature, value in top_features[:10]:
                    report_lines.append(f"- {feature}: {value:.3f}")
            else:
                report_lines.append("No feature importance data available")
        
        # Add execution time
        if 'execution_time' in test_results:
            report_lines.append(f"\nExecution time: {test_results['execution_time']:.2f} seconds")
        
        # Write report to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
            
        if self.verbose:
            print(f"Report saved to {output_path}")