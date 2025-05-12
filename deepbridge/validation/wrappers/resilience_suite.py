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

                # Create test configurations based on drift types and intensities
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

                        # Add test configuration
                        tests.append({
                            'method': 'distribution_shift',
                            'params': {
                                'alpha': intensity,
                                'metric': 'auc',  # Default metric
                                'distance_metric': distance_metric
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
            "WD1": self._calculate_wasserstein
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
        
        # Calculate overall resilience score
        avg_performance_gaps = [
            results['distribution_shift']['by_alpha'][alpha]['avg_performance_gap'] 
            for alpha in all_alphas
        ]
        results['resilience_score'] = 1.0 - min(1.0, max(0.0, np.mean(avg_performance_gaps)))
        
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