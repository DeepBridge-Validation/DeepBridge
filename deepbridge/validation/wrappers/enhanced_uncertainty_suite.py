"""
Enhanced uncertainty estimation suite with additional metrics and visualizations.

This module extends the base UncertaintySuite to provide more detailed uncertainty
analysis and data for comprehensive reports and visualizations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
import time
import datetime
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging

from deepbridge.validation.wrappers.uncertainty_suite import UncertaintySuite, CRQR

# Configure logger
logger = logging.getLogger("deepbridge.uncertainty")

class EnhancedUncertaintySuite(UncertaintySuite):
    """
    Enhanced uncertainty estimation suite with additional metrics and feature analysis.
    """
    
    def __init__(self, dataset, verbose: bool = False, feature_subset: Optional[List[str]] = None, random_state: Optional[int] = None):
        """
        Initialize the enhanced uncertainty estimation suite.
        
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
        """
        super().__init__(dataset, verbose, feature_subset, random_state)
        
        # Additional parameters for enhanced analysis
        self.reliable_threshold_ratio = 1.1  # Multiplier for classification of reliable/unreliable points
        self.bin_count = 10  # Number of bins for marginal bandwidth analysis
        
    def evaluate_uncertainty(self, method: str, params: Dict, feature=None) -> Dict[str, Any]:
        """
        Enhanced uncertainty evaluation with additional metrics and feature analysis.
        
        Parameters:
        -----------
        method : str
            Method to use ('crqr' supported)
        params : Dict
            Parameters for the uncertainty method
        feature : str or None
            Specific feature to analyze (for feature importance)
            
        Returns:
        --------
        dict : Detailed evaluation results with enhanced metrics
        """
        # Get the standard uncertainty evaluation results
        results = super().evaluate_uncertainty(method, params, feature)
        
        logger.info(f"[EVALUATE_DEBUG] Standard results keys: {list(results.keys())}")
        
        # Get dataset
        X = self.dataset.get_feature_data()
        y = self.dataset.get_target_data()
        
        # Convert any numpy arrays to pandas objects if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        
        # Apply feature subset if specified
        if self.feature_subset:
            X = X[self.feature_subset]
        
        # Get the model and predictions
        if method == 'crqr':
            model_key = results['model_key']
            model = self.uncertainty_models[model_key]
            
            # Get prediction intervals
            lower_bound, upper_bound = results['lower_bounds'], results['upper_bounds']
            widths = results['widths']
            
            logger.info(f"[EVALUATE_DEBUG] Starting enhanced analysis with data: X shape={X.shape}, widths shape={widths.shape}")
            
            # Calculate additional metrics
            additional_metrics = self._calculate_additional_metrics(X, y, model, lower_bound, upper_bound, widths)
            results.update(additional_metrics)
            logger.info(f"[EVALUATE_DEBUG] Added additional metrics: {list(additional_metrics.keys())}")
            
            # Add reliability analysis
            reliability_analysis = self._analyze_reliability(X, y, lower_bound, upper_bound, widths)
            results['reliability_analysis'] = reliability_analysis
            logger.info(f"[EVALUATE_DEBUG] Added reliability_analysis with keys: {list(reliability_analysis.keys())}")
            
            # Add marginal bandwidth analysis for important features
            marginal_bandwidth = self._analyze_marginal_bandwidth(X, widths)
            results['marginal_bandwidth'] = marginal_bandwidth
            logger.info(f"[EVALUATE_DEBUG] Added marginal_bandwidth for features: {list(marginal_bandwidth.keys())}")
            
            # Verify that the data is in the results
            for key in ['reliability_analysis', 'marginal_bandwidth']:
                if key in results:
                    logger.info(f"[EVALUATE_DEBUG] Verified {key} is in the results")
                else:
                    logger.error(f"[EVALUATE_DEBUG] ERROR: {key} is missing from the results")
            
        # Log the final set of top-level keys
        logger.info(f"[EVALUATE_DEBUG] Final evaluation results keys: {list(results.keys())}")
        
        return results
    
    def _calculate_additional_metrics(self, X, y, model, lower_bound, upper_bound, widths):
        """Calculate enhanced metrics for uncertainty evaluation."""
        # Point predictions from base model
        y_pred = model.predict(X)
        
        # Basic regression metrics
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        
        # Enhanced interval metrics
        median_width = np.median(widths)
        min_width = np.min(widths)
        max_width = np.max(widths)
        
        # Coverage statistics for different threshold values
        threshold_values = [0.80, 0.85, 0.90, 0.95, 0.99]
        coverage_stats = {}
        for threshold in threshold_values:
            # Calculate expected interval width for this coverage
            width_threshold = np.quantile(widths, threshold)
            # Filter to only use intervals smaller than threshold
            mask = widths <= width_threshold
            if np.sum(mask) > 0:
                # Calculate actual coverage for these intervals
                coverage = np.mean((y[mask] >= lower_bound[mask]) & (y[mask] <= upper_bound[mask]))
                coverage_stats[f"coverage_{int(threshold*100)}"] = coverage
        
        return {
            "mse": mse,
            "mae": mae,
            "median_width": median_width,
            "min_width": min_width,
            "max_width": max_width,
            "coverage_stats": coverage_stats,
            "interval_widths": widths.tolist()  # Convert to list for JSON serialization
        }
    
    def _analyze_reliability(self, X, y, lower_bound, upper_bound, widths):
        """
        Analyze the reliability of predictions by categorizing points as reliable/unreliable.
        
        A point is classified as unreliable if its prediction interval width is greater
        than the threshold_ratio times the median width.
        """
        logger.info("[RELIABILITY_DEBUG] Starting reliability analysis")
        logger.info(f"[RELIABILITY_DEBUG] X shape: {X.shape}, y shape: {y.shape}")
        logger.info(f"[RELIABILITY_DEBUG] lower_bound shape: {lower_bound.shape}, upper_bound shape: {upper_bound.shape}")
        logger.info(f"[RELIABILITY_DEBUG] widths shape: {widths.shape}")
        
        # Calculate threshold for reliability classification
        threshold = np.median(widths) * self.reliable_threshold_ratio
        logger.info(f"[RELIABILITY_DEBUG] Calculated threshold: {threshold} (median width: {np.median(widths)} × ratio: {self.reliable_threshold_ratio})")
        
        # Classify points
        reliable_mask = widths <= threshold
        unreliable_mask = ~reliable_mask
        
        # Count reliable/unreliable points
        reliable_count = np.sum(reliable_mask)
        unreliable_count = np.sum(unreliable_mask)
        logger.info(f"[RELIABILITY_DEBUG] Reliable points: {reliable_count}, Unreliable points: {unreliable_count}")
        
        # Calculate PSI (Population Stability Index) for each feature
        psi_values = {}
        feature_distributions = {"reliable": {}, "unreliable": {}}
        
        # Select top features for analysis (max 5 to keep size reasonable)
        num_features = min(5, X.shape[1])
        feature_names = X.columns[:num_features].tolist()
        logger.info(f"[RELIABILITY_DEBUG] Selected features for analysis: {feature_names}")
        
        for feature in feature_names:
            # Get feature values for reliable and unreliable points
            reliable_values = X[feature][reliable_mask].values
            unreliable_values = X[feature][unreliable_mask].values
            
            logger.info(f"[RELIABILITY_DEBUG] Feature '{feature}': reliable={len(reliable_values)} values, unreliable={len(unreliable_values)} values")
            
            # Store distributions for visualization
            reliable_list = reliable_values.tolist()
            unreliable_list = unreliable_values.tolist()
            
            # Log some sample values
            if len(reliable_list) > 0:
                logger.info(f"[RELIABILITY_DEBUG] '{feature}' reliable sample: {reliable_list[:3]}")
            if len(unreliable_list) > 0:
                logger.info(f"[RELIABILITY_DEBUG] '{feature}' unreliable sample: {unreliable_list[:3]}")
                
            feature_distributions["reliable"][feature] = reliable_list
            feature_distributions["unreliable"][feature] = unreliable_list
            
            # Calculate PSI if we have enough data in both groups
            if len(reliable_values) > 10 and len(unreliable_values) > 10:
                psi = self._calculate_psi(reliable_values, unreliable_values)
                psi_values[feature] = psi
                logger.info(f"[RELIABILITY_DEBUG] Feature '{feature}' PSI: {psi}")
        
        # Verify feature_distributions is properly structured
        logger.info(f"[RELIABILITY_DEBUG] feature_distributions types: {list(feature_distributions.keys())}")
        for dist_type, features in feature_distributions.items():
            logger.info(f"[RELIABILITY_DEBUG] '{dist_type}' has {len(features)} features")
        
        result = {
            "threshold_value": float(threshold),
            "reliable_count": int(reliable_count),
            "unreliable_count": int(unreliable_count),
            "psi_values": psi_values,
            "feature_distributions": feature_distributions
        }
        
        logger.info(f"[RELIABILITY_DEBUG] Result keys: {list(result.keys())}")
        logger.info("[RELIABILITY_DEBUG] Finished reliability analysis")
        return result
    
    def _calculate_psi(self, expected, actual, bins=10):
        """
        Calculate Population Stability Index (PSI) between two distributions.
        PSI measures how much a distribution has shifted.
        
        Parameters:
        -----------
        expected : array-like
            Expected (reference) distribution
        actual : array-like
            Actual (current) distribution to compare
        bins : int
            Number of bins for histogram
            
        Returns:
        --------
        float : PSI value (higher means more shift)
        """
        try:
            # Handle empty arrays
            if len(expected) == 0 or len(actual) == 0:
                return 0.0
                
            # Find the min and max to define the bin range
            all_values = np.concatenate([expected, actual])
            min_val, max_val = np.min(all_values), np.max(all_values)
            
            # Add a small range to avoid division by zero in edge cases
            if min_val == max_val:
                min_val = min_val - 0.1
                max_val = max_val + 0.1
                
            # Create bins
            bin_edges = np.linspace(min_val, max_val, bins + 1)
            
            # Count observations in each bin
            expected_counts, _ = np.histogram(expected, bins=bin_edges)
            actual_counts, _ = np.histogram(actual, bins=bin_edges)
            
            # Convert to percentages (adding small value to avoid division by zero)
            expected_pct = expected_counts / (np.sum(expected_counts) + 1e-6)
            actual_pct = actual_counts / (np.sum(actual_counts) + 1e-6)
            
            # Replace zeros with small value to avoid division by zero or log(0)
            expected_pct = np.where(expected_pct == 0, 1e-6, expected_pct)
            actual_pct = np.where(actual_pct == 0, 1e-6, actual_pct)
            
            # Calculate PSI
            psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
            
            return float(psi)
        except Exception as e:
            logger.warning(f"Error calculating PSI: {str(e)}")
            return 0.0
    
    def _analyze_marginal_bandwidth(self, X, widths):
        """
        Analyze how interval width varies with feature values.
        
        For each important feature, bins the feature values and calculates
        average interval width in each bin.
        """
        logger.info("[MARGINAL_DEBUG] Starting marginal bandwidth analysis")
        logger.info(f"[MARGINAL_DEBUG] X shape: {X.shape}, widths shape: {widths.shape}")
        
        result = {}
        
        # Calculate threshold for reliable vs unreliable
        threshold = np.median(widths) * self.reliable_threshold_ratio
        logger.info(f"[MARGINAL_DEBUG] Threshold: {threshold}")
        
        # Select top features for analysis (max 3 to keep size reasonable)
        num_features = min(3, X.shape[1])
        feature_names = X.columns[:num_features].tolist()
        logger.info(f"[MARGINAL_DEBUG] Selected features for analysis: {feature_names}")
        
        for feature in feature_names:
            logger.info(f"[MARGINAL_DEBUG] Analyzing feature: {feature}")
            feature_values = X[feature].values
            
            # Log feature value range
            min_val = np.min(feature_values)
            max_val = np.max(feature_values)
            logger.info(f"[MARGINAL_DEBUG] Feature '{feature}' range: [{min_val}, {max_val}]")
            
            # Create bins for the feature values
            bin_edges = np.linspace(min_val, max_val, self.bin_count + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            logger.info(f"[MARGINAL_DEBUG] Created {len(bin_centers)} bins with centers: {bin_centers}")
            
            # Initialize arrays for average widths and counts
            avg_widths = []
            counts_below = []
            counts_above = []
            
            # Calculate statistics for each bin
            for i in range(len(bin_centers)):
                # Get mask for values in this bin
                mask = (feature_values >= bin_edges[i]) & (feature_values < bin_edges[i+1])
                bin_count = np.sum(mask)
                
                # Skip if no values in this bin
                if bin_count == 0:
                    logger.info(f"[MARGINAL_DEBUG] Bin {i} has no values, adding zeros")
                    avg_widths.append(0)
                    counts_below.append(0)
                    counts_above.append(0)
                    continue
                
                # Calculate average width for this bin
                bin_widths = widths[mask]
                avg_width = np.mean(bin_widths)
                avg_widths.append(float(avg_width))
                
                below_count = int(np.sum(bin_widths <= threshold))
                above_count = int(np.sum(bin_widths > threshold))
                
                # Count reliable vs unreliable in this bin
                counts_below.append(below_count)
                counts_above.append(above_count)
                
                logger.info(f"[MARGINAL_DEBUG] Bin {i} stats: avg_width={avg_width:.4f}, reliable={below_count}, unreliable={above_count}")
            
            # Store results for this feature
            feature_result = {
                "bin_centers": bin_centers.tolist(),
                "avg_widths": avg_widths,
                "counts_below_threshold": counts_below,
                "counts_above_threshold": counts_above,
                "threshold": float(threshold)
            }
            
            # Log data lengths for verification
            for key, value in feature_result.items():
                if isinstance(value, list):
                    logger.info(f"[MARGINAL_DEBUG] {key} length: {len(value)}")
                else:
                    logger.info(f"[MARGINAL_DEBUG] {key}: {value}")
            
            result[feature] = feature_result
        
        # Verify result structure
        logger.info(f"[MARGINAL_DEBUG] Result contains data for {len(result)} features: {list(result.keys())}")
        logger.info("[MARGINAL_DEBUG] Finished marginal bandwidth analysis")
        return result
    
    def run(self) -> Dict[str, Any]:
        """
        Run the configured uncertainty tests with enhanced analysis.
        
        Returns:
        --------
        dict : Enhanced test results with detailed performance metrics
        """
        logger.info("[RUN_DEBUG] Starting enhanced uncertainty analysis")

        # Get standard results first
        results = super().run()
        logger.info(f"[RUN_DEBUG] Base results contain top-level keys: {list(results.keys())}")

        # Preserve test_predictions and test_labels if they exist
        test_predictions = None
        test_labels = None
        if 'test_predictions' in results:
            test_predictions = results['test_predictions']
            logger.info(f"[RUN_DEBUG] Preserving test_predictions with shape: {test_predictions.shape if hasattr(test_predictions, 'shape') else 'N/A'}")
        if 'test_labels' in results:
            test_labels = results['test_labels']
            logger.info(f"[RUN_DEBUG] Preserving test_labels with shape: {test_labels.shape if hasattr(test_labels, 'shape') else 'N/A'}")
        
        # Check CRQR results
        if 'crqr' in results:
            logger.info(f"[RUN_DEBUG] CRQR results keys: {list(results['crqr'].keys())}")
            if 'all_results' in results['crqr']:
                logger.info(f"[RUN_DEBUG] Found {len(results['crqr']['all_results'])} CRQR test results")
        
        # Additional enhancements for the overall results
        if 'crqr' in results and 'all_results' in results['crqr'] and results['crqr']['all_results']:
            # Collect interval widths across all tests
            all_widths = []
            for i, result in enumerate(results['crqr']['all_results']):
                logger.info(f"[RUN_DEBUG] Checking result {i} keys: {list(result.keys())}")
                if 'widths' in result:
                    widths_count = len(result['widths'])
                    logger.info(f"[RUN_DEBUG] Result {i} has {widths_count} widths")
                    all_widths.extend(result['widths'])
            
            if all_widths:
                logger.info(f"[RUN_DEBUG] Collected {len(all_widths)} total interval widths")
                
                # Store them for the overall model
                results['interval_widths'] = {
                    "primary_model": all_widths
                }
                logger.info("[RUN_DEBUG] Added interval_widths to results")
                
                # Calculate interval width statistics
                results['mean_width'] = float(np.mean(all_widths))
                results['median_width'] = float(np.median(all_widths))
                results['min_width'] = float(np.min(all_widths))
                results['max_width'] = float(np.max(all_widths))
                logger.info(f"[RUN_DEBUG] Width statistics: mean={results['mean_width']:.4f}, median={results['median_width']:.4f}")
                
                # Add reliable/unreliable threshold
                threshold = np.median(all_widths) * self.reliable_threshold_ratio
                results['threshold_value'] = float(threshold)
                logger.info(f"[RUN_DEBUG] Set threshold_value: {results['threshold_value']}")
                
                # Add coverage value
                if 'avg_coverage' not in results and results['crqr']['all_results']:
                    coverages = [r.get('coverage', 0) for r in results['crqr']['all_results']]
                    if coverages:
                        results['avg_coverage'] = float(np.mean(coverages))
                        results['coverage'] = results['avg_coverage']  # Alias for compatibility
                        logger.info(f"[RUN_DEBUG] Set coverage: {results['coverage']}")
                        
                # Get dataset for enhanced analysis
                X = self.dataset.get_feature_data()
                y = self.dataset.get_target_data()
                
                # Convert any numpy arrays to pandas objects if needed
                if not isinstance(X, pd.DataFrame):
                    X = pd.DataFrame(X)
                if not isinstance(y, pd.Series):
                    y = pd.Series(y)
                
                # Apply feature subset if specified
                if self.feature_subset:
                    X = X[self.feature_subset]
                
                # Get the first test result for enhanced analysis
                test_result = results['crqr']['all_results'][0]
                if 'lower_bounds' in test_result and 'upper_bounds' in test_result and 'widths' in test_result:
                    # Extract needed data
                    lower_bound = test_result['lower_bounds']
                    upper_bound = test_result['upper_bounds']
                    widths = test_result['widths']
                    
                    # Add reliability analysis
                    reliability_analysis = self._analyze_reliability(X, y, lower_bound, upper_bound, widths)
                    results['reliability_analysis'] = reliability_analysis
                    logger.info(f"[RUN_DEBUG] Added reliability_analysis with keys: {list(reliability_analysis.keys())}")
                    
                    # Add marginal bandwidth analysis
                    marginal_bandwidth = self._analyze_marginal_bandwidth(X, widths)
                    results['marginal_bandwidth'] = marginal_bandwidth
                    logger.info(f"[RUN_DEBUG] Added marginal_bandwidth for features: {list(marginal_bandwidth.keys())}")
                else:
                    logger.warning("[RUN_DEBUG] Missing required data in CRQR results for enhanced analysis")
            else:
                logger.warning("[RUN_DEBUG] No interval widths found in any CRQR result")
        else:
            logger.warning("[RUN_DEBUG] No CRQR results found to process")
            
        # Check if key distribution data was added
        distribution_keys = ['reliability_analysis', 'marginal_bandwidth', 'interval_widths']
        for key in distribution_keys:
            if key in results:
                logger.info(f"[RUN_DEBUG] Found {key} in results")
                if isinstance(results[key], dict):
                    logger.info(f"[RUN_DEBUG] {key} has keys: {list(results[key].keys())}")
            else:
                logger.warning(f"[RUN_DEBUG] Missing {key} in results")
                
        # Log final set of top-level keys
        logger.info(f"[RUN_DEBUG] Final results contain keys: {list(results.keys())}")
        logger.info("[RUN_DEBUG] Finished enhanced uncertainty analysis")
        
        # Add alpha levels for visualization
        if 'alphas' in results:
            results['alpha_levels'] = results['alphas']
        
        # Add timestamp for reporting
        results['timestamp'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Get the model type
        if hasattr(self.dataset, 'model') and self.dataset.model is not None:
            results['model_type'] = type(self.dataset.model).__name__
        
        # Add dataset information
        X = self.dataset.get_feature_data()
        y = self.dataset.get_target_data()
        
        results['dataset'] = {
            "feature_names": X.columns.tolist() if hasattr(X, 'columns') else [f"feature_{i}" for i in range(X.shape[1])],
            "X_test_shape": list(X.shape),
            "y_test_shape": [len(y)] if len(y.shape) == 1 else list(y.shape)
        }
        
        # Add model predictions
        if hasattr(self.dataset, 'model') and self.dataset.model is not None:
            model = self.dataset.model
            try:
                y_pred = model.predict(X)
                
                # Get prediction intervals from our uncertainty model
                if self.uncertainty_models:
                    model_key = list(self.uncertainty_models.keys())[0]
                    uncertainty_model = self.uncertainty_models[model_key]
                    lower_bound, upper_bound = uncertainty_model.predict_interval(X)
                    
                    # Store predictions for visualization
                    results['predictions'] = {
                        "model": results.get('model_type', "Model"),
                        "y_pred": y_pred.tolist(),
                        "lower_bound": lower_bound.tolist(),
                        "upper_bound": upper_bound.tolist()
                    }
            except Exception as e:
                if self.verbose:
                    print(f"Error generating predictions: {str(e)}")
        
        # Add or update the config section
        results['config'] = {
            "alpha": results['alpha_levels'][0] if 'alpha_levels' in results and results['alpha_levels'] else 0.1,
            "test_size": 0.3,  # Default or from params
            "calib_ratio": 1/3,  # Default or from params
            "random_state": self.random_state,
            "threshold_ratio": self.reliable_threshold_ratio
        }

        # Re-add test_predictions and test_labels if they were preserved
        if test_predictions is not None:
            results['test_predictions'] = test_predictions
            logger.info(f"[RUN_DEBUG] Re-added test_predictions to final results")
        if test_labels is not None:
            results['test_labels'] = test_labels
            logger.info(f"[RUN_DEBUG] Re-added test_labels to final results")

        # IMPORTANT: The base run() returns a structured result with primary_model
        # We need to preserve this structure and add our enhancements to the right place
        if 'primary_model' in results:
            logger.info(f"[RUN_DEBUG] primary_model exists, keys: {list(results['primary_model'].keys())[:10]}")

            # Check if plot_data exists
            if 'plot_data' in results['primary_model']:
                logger.info(f"[RUN_DEBUG] plot_data exists in primary_model with keys: {list(results['primary_model']['plot_data'].keys())}")
            else:
                logger.warning("[RUN_DEBUG] plot_data NOT in primary_model")

            # Add our enhancements to the primary_model
            if 'reliability_analysis' in results:
                results['primary_model']['reliability_analysis'] = results.pop('reliability_analysis')
                logger.info("[RUN_DEBUG] Moved reliability_analysis to primary_model")
            if 'marginal_bandwidth' in results:
                results['primary_model']['marginal_bandwidth'] = results.pop('marginal_bandwidth')
                logger.info("[RUN_DEBUG] Moved marginal_bandwidth to primary_model")
            if 'interval_widths' in results:
                results['primary_model']['interval_widths'] = results.pop('interval_widths')
                logger.info("[RUN_DEBUG] Moved interval_widths to primary_model")

            # Ensure plot_data is preserved
            if 'plot_data' not in results['primary_model'] and 'plot_data' in results:
                results['primary_model']['plot_data'] = results['plot_data']
                logger.info("[RUN_DEBUG] Copied plot_data to primary_model")

            logger.info(f"[RUN_DEBUG] Final primary_model keys: {list(results['primary_model'].keys())[:15]}")

        return results


# Enhancement for the main run_uncertainty_tests function
def run_enhanced_uncertainty_tests(dataset, config_name='full', verbose=True, feature_subset=None):
    """
    Run enhanced uncertainty quantification tests on a dataset with additional metrics and visualizations.
    
    Parameters:
    -----------
    dataset : DBDataset
        Dataset object containing training/test data and model
    config_name : str
        Name of the configuration to use: 'quick', 'medium', or 'full'
    verbose : bool
        Whether to print progress information
    feature_subset : List[str] or None
        Specific features to focus on for testing (None for all features)
        
    Returns:
    --------
    dict : Test results with comprehensive uncertainty metrics and visualization data
    """
    # Initialize enhanced uncertainty suite
    uncertainty = EnhancedUncertaintySuite(dataset, verbose=verbose, feature_subset=feature_subset)
    
    # Configure and run tests with feature subset if specified
    results = uncertainty.config(config_name, feature_subset=feature_subset).run()
    
    if verbose:
        print(f"\nEnhanced Uncertainty Test Summary:")
        print(f"Overall uncertainty quality score: {results.get('uncertainty_quality_score', 0):.3f}")
        print(f"Average coverage: {results.get('coverage', 0):.3f}")
        print(f"Average interval width: {results.get('mean_width', 0):.3f}")
    
    return results