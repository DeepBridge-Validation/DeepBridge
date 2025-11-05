"""
Simplified uncertainty estimation suite for machine learning models.

This module provides a streamlined interface for estimating prediction
uncertainty using conformal prediction techniques.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
import time
import datetime
from sklearn.model_selection import train_test_split

from deepbridge.core.experiment.parameter_standards import (
    get_test_config, TestType, ConfigName, is_valid_config_name
)

class UncertaintySuite:
    """
    Focused suite for model uncertainty quantification using conformal prediction.
    """

    # Load configurations from centralized parameter standards
    def _get_config_templates(self):
        """Get uncertainty configurations from the centralized parameter standards."""
        try:
            uncertainty_configs = {
                config_name: get_test_config(TestType.UNCERTAINTY.value, config_name)
                for config_name in [ConfigName.QUICK.value, ConfigName.MEDIUM.value, ConfigName.FULL.value]
            }
            return uncertainty_configs
        except Exception as e:
            import logging
            logging.getLogger("deepbridge.uncertainty").error(f"Error loading centralized configs: {str(e)}")
            # Fallback to empty templates if centralized configs fail
            return {
                'quick': [],
                'medium': [],
                'full': []
            }
    
    def __init__(self, dataset, verbose: bool = False, feature_subset: Optional[List[str]] = None, random_state: Optional[int] = None):
        """
        Initialize the uncertainty estimation suite.
        
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
        self.dataset = dataset
        self.verbose = verbose
        self.feature_subset = feature_subset
        self.random_state = random_state
        
        # Store current configuration
        self.current_config = None
        
        # Store results
        self.results = {}
        
        # Determine problem type based on dataset or model
        self._problem_type = self._determine_problem_type()
        
        # Store models
        self.uncertainty_models = {}
        
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
        
        # Default to regression for uncertainty estimation
        return 'regression'
    
    def config(self, config_name: str = 'quick', feature_subset: Optional[List[str]] = None) -> 'UncertaintySuite':
        """
        Set a predefined configuration for uncertainty tests.

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
        self.current_config_name = config_name

        # Update feature_subset in tests if specified
        if self.feature_subset:
            for test in self.current_config:
                if 'params' in test:
                    test['params']['feature_subset'] = self.feature_subset

        if self.verbose:
            print(f"\nConfigured for {config_name} uncertainty test suite")
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
    
    def _create_crqr_model(self, alpha=0.1, test_size=0.3, calib_ratio=1/3):
        """Create a CRQR model with specified parameters."""
        return CRQR(base_model=None, 
                   alpha=alpha, 
                   test_size=test_size, 
                   calib_ratio=calib_ratio, 
                   random_state=self.random_state)
    
    def evaluate_uncertainty(self, method: str, params: Dict, feature=None) -> Dict[str, Any]:
        """
        Evaluate model uncertainty using the specified method.
        
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
        dict : Detailed evaluation results
        """
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
        
        # Create and train the uncertainty estimation model
        if method == 'crqr':
            alpha = params.get('alpha', 0.1)
            test_size = params.get('test_size', 0.3)
            calib_ratio = params.get('calib_ratio', 1/3)
            
            # Create the model
            model = self._create_crqr_model(alpha, test_size, calib_ratio)
            
            # Fit the model
            model.fit(X, y)
            
            # Get prediction intervals
            lower_bound, upper_bound = model.predict_interval(X)
            
            # Evaluate performance
            scores = model.score(X, y)
            
            # Calculate mean and median interval widths
            widths = upper_bound - lower_bound
            mean_width = np.mean(widths)
            median_width = np.median(widths)
            
            # Calculate feature importance if a specific feature is provided
            feature_importance = {}
            if feature:
                # Simple approach: remove feature and see how it affects interval width
                X_without_feature = X.drop(columns=[feature])
                model_without_feature = self._create_crqr_model(alpha, test_size, calib_ratio)
                model_without_feature.fit(X_without_feature, y)
                
                # Get intervals without the feature
                lower_bound_without, upper_bound_without = model_without_feature.predict_interval(X_without_feature)
                widths_without = upper_bound_without - lower_bound_without
                
                # Calculate importance as relative change in interval width
                importance = (np.mean(widths_without) - mean_width) / mean_width
                feature_importance[feature] = abs(importance)
            
            # Store key information for results
            model_key = f"crqr_alpha_{alpha}"
            self.uncertainty_models[model_key] = model
            
            # Store predictions for calibration charts if classification
            test_predictions = None
            test_labels = None

            # Only capture predictions for the overall model (not per-feature tests)
            # to ensure we get predictions for the full dataset
            if feature is None and hasattr(self.dataset, 'model'):
                original_model = self.dataset.model

                # Try to get test data and predictions
                try:
                    # IMPORTANT: Get the FULL test data from dataset, not the subset X
                    # because the original model was trained on all features
                    if hasattr(self.dataset, 'get_test_features'):
                        X_test_full = self.dataset.get_test_features()
                        y_test_full = self.dataset.get_test_target()
                    else:
                        # Fall back to getting the full feature data
                        X_test_full = self.dataset.get_feature_data()
                        y_test_full = self.dataset.get_target_data()

                    # Get probability predictions if model supports it
                    if hasattr(original_model, 'predict_proba'):
                        test_predictions = original_model.predict_proba(X_test_full)
                        test_labels = np.array(y_test_full)
                        if self.verbose:
                            print(f"    Stored probability predictions for calibration charts")
                except Exception as e:
                    if self.verbose:
                        print(f"    Could not store probability predictions: {e}")

            # Return detailed results
            result = {
                'method': 'crqr',
                'alpha': alpha,
                'coverage': scores['coverage'],
                'expected_coverage': 1 - alpha,
                'mean_width': mean_width,
                'median_width': median_width,
                'widths': widths,
                'lower_bounds': lower_bound,
                'upper_bounds': upper_bound,
                'feature_importance': feature_importance,
                'model_key': model_key,
                'test_size': test_size,
                'calib_ratio': calib_ratio,
                'split_sizes': model.get_split_sizes()
            }

            # Add predictions if available for calibration charts
            if test_predictions is not None and test_labels is not None:
                result['test_predictions'] = test_predictions
                result['test_labels'] = test_labels

            return result
        else:
            raise ValueError(f"Uncertainty method '{method}' not supported")
    
    def run(self) -> Dict[str, Any]:
        """
        Run the configured uncertainty tests.

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
            print(f"Running uncertainty test suite...")
            start_time = time.time()

        # Initialize results
        results = {
            'crqr': {
                'by_alpha': {},           # Results organized by alpha level
                'by_feature': {},         # Results organized by feature
                'all_results': []         # All raw test results
            }
        }

        # Get the dataset
        X = self.dataset.get_feature_data()
        y = self.dataset.get_target_data()

        # Track alpha levels for summary
        all_alphas = []
        
        # Run all configured tests
        for test_config in self.current_config:
            method = test_config['method']
            params = test_config.get('params', {})
            
            if method == 'crqr':
                alpha = params.get('alpha', 0.1)
                
                # Track alpha
                if alpha not in all_alphas:
                    all_alphas.append(alpha)
                
                if self.verbose:
                    print(f"Running CRQR with alpha={alpha}")
                
                # Initialize alpha results if needed
                if alpha not in results['crqr']['by_alpha']:
                    results['crqr']['by_alpha'][alpha] = {}
                
                # Run the uncertainty estimation
                overall_result = self.evaluate_uncertainty(method, params)
                results['crqr']['all_results'].append(overall_result)
                
                # Add to alpha-specific results
                results['crqr']['by_alpha'][alpha]['overall_result'] = overall_result
                
                # Test individual features if we have a feature subset
                if self.feature_subset:
                    features_to_test = self.feature_subset
                elif isinstance(X, pd.DataFrame):
                    # Test a sample of features (max 5) to keep runtime reasonable
                    features_to_test = X.columns.tolist()[:5] if len(X.columns) > 5 else X.columns.tolist()
                else:
                    features_to_test = []
                
                # Run per-feature tests
                for feature in features_to_test:
                    if self.verbose:
                        print(f"  - Testing feature: {feature}")
                    
                    # Initialize feature results if needed
                    if feature not in results['crqr']['by_feature']:
                        results['crqr']['by_feature'][feature] = {}
                    
                    # Run feature-specific test
                    feature_result = self.evaluate_uncertainty(method, params, feature=feature)
                    
                    # Store results
                    results['crqr']['by_feature'][feature][alpha] = feature_result
        
        # Organize results for easier analysis
        results['alphas'] = sorted(all_alphas)

        # Calculate feature importance
        results['feature_importance'] = self._calculate_feature_importance(results)

        # Identify top features for detailed reliability analysis
        top_features = self._identify_top_features(results['feature_importance'])

        # Run detailed reliability analysis on top features
        if top_features:
            if self.verbose:
                print(f"\nRunning detailed reliability analysis on top features: {top_features}")
            results['reliability_analysis'] = self._analyze_feature_reliability(top_features, X, y)

            # Generate individual reliability charts for each top feature
            results['reliability_charts'] = self._generate_reliability_charts(results['reliability_analysis'])

            if self.verbose:
                print(f"\nGenerated {len(results['reliability_charts'])} reliability charts")
        else:
            results['reliability_analysis'] = {}
            results['reliability_charts'] = {}
        
        # Calculate overall performance metrics
        coverage_error = []
        normalized_width = []
        
        # Process CRQR results
        for result in results['crqr']['all_results']:
            # Coverage error: difference between actual and expected coverage
            coverage_error.append(abs(result['coverage'] - result['expected_coverage']))
            
            # Normalized width: width compared to the range of the target variable
            if hasattr(y, 'min') and hasattr(y, 'max'):
                y_range = y.max() - y.min()
                if y_range > 0:
                    normalized_width.append(result['mean_width'] / y_range)
        
        # Calculate average metrics
        if coverage_error:
            results['avg_coverage_error'] = np.mean(coverage_error)
        if normalized_width:
            results['avg_normalized_width'] = np.mean(normalized_width)
        
        # Calculate overall quality score (higher is better)
        if coverage_error and normalized_width:
            # Balance between coverage accuracy and interval efficiency
            # Lower coverage error and narrower intervals (relative to data range) are better
            coverage_score = 1 - min(1, np.mean(coverage_error) * 2)  # Penalize coverage errors
            width_score = 1 - min(1, np.mean(normalized_width))      # Reward narrower intervals
            
            # Combine scores with higher weight on coverage
            results['uncertainty_quality_score'] = 0.7 * coverage_score + 0.3 * width_score
        else:
            results['uncertainty_quality_score'] = 0.5  # Default if no metrics calculated
        
        # Collect test predictions and labels if available (for calibration charts)
        test_predictions = None
        test_labels = None
        for result in results['crqr']['all_results']:
            if 'test_predictions' in result and 'test_labels' in result:
                test_predictions = result['test_predictions']
                test_labels = result['test_labels']
                break  # Use the first available set

        # Store predictions at the top level for easy access
        if test_predictions is not None and test_labels is not None:
            results['test_predictions'] = test_predictions
            results['test_labels'] = test_labels
            if self.verbose:
                print(f"Stored test predictions for calibration charts")

        # Prepare data for plotting
        results['plot_data'] = self._prepare_plot_data(results)

        # Add execution time
        if self.verbose:
            elapsed_time = time.time() - start_time
            # Não armazenamos mais o tempo de execução nos resultados
            print(f"Test suite completed in {elapsed_time:.2f} seconds")
            print(f"Overall uncertainty quality score: {results['uncertainty_quality_score']:.3f}")
        
        # Store results
        test_id = f"test_{int(time.time())}"
        self.results[test_id] = results

        # Create primary_model structure with test predictions if available
        primary_model = {
            'crqr': results.get('crqr', {}),
            'alphas': results.get('alphas', []),
            'feature_importance': results.get('feature_importance', {}),
            'avg_coverage_error': results.get('avg_coverage_error', 0),
            'avg_normalized_width': results.get('avg_normalized_width', 0),
            'uncertainty_quality_score': results.get('uncertainty_quality_score', 0.5),
            'plot_data': results.get('plot_data', {}),
            'reliability_analysis': results.get('reliability_analysis', {}),
            'reliability_charts': results.get('reliability_charts', {})
        }

        # Add test predictions if available
        if 'test_predictions' in results:
            primary_model['test_predictions'] = results['test_predictions']
            primary_model['test_labels'] = results['test_labels']
            if self.verbose:
                print(f"Added test predictions to primary_model for charts")

        # Structure results for report generation
        final_results = {
            'primary_model': primary_model,
            'config': {
                'test_id': test_id,
                'config_name': self.current_config_name if hasattr(self, 'current_config_name') else 'custom'
            }
        }

        # Copy other top-level fields
        for key in ['test_predictions', 'test_labels', 'uncertainty_quality_score',
                    'avg_coverage_error', 'avg_normalized_width', 'alphas', 'feature_importance']:
            if key in results:
                final_results[key] = results[key]

        return final_results
    
    def _calculate_feature_importance(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate feature importance based on uncertainty impact."""
        feature_importance = {}

        # Process feature tests for CRQR
        for feature, alphas in results['crqr']['by_feature'].items():
            # Calculate average importance across alpha levels
            importances = []
            for alpha, result in alphas.items():
                if feature in result.get('feature_importance', {}):
                    importances.append(result['feature_importance'][feature])

            # Average importance across alpha levels
            if importances:
                feature_importance[feature] = np.mean(importances)

        # Normalize to [0, 1] scale
        if feature_importance:
            max_importance = max(feature_importance.values())
            if max_importance > 0:
                feature_importance = {feature: value / max_importance for feature, value in feature_importance.items()}

        return feature_importance

    def _identify_top_features(self, feature_importance: Dict[str, float]) -> List[str]:
        """Identify top features for detailed analysis.

        If feature_subset was provided, use those features.
        Otherwise, use top 3 most important features.
        """
        # If feature_subset was explicitly provided, prioritize those
        if self.feature_subset:
            # Return features that exist in both feature_subset and feature_importance
            valid_features = [f for f in self.feature_subset if f in feature_importance]
            if valid_features:
                return valid_features[:3]  # Limit to top 3
            # If none of the subset features have importance scores, return subset anyway
            return self.feature_subset[:3]

        # Otherwise, use top 3 features by importance
        if feature_importance:
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            return [f[0] for f in sorted_features[:3]]

        return []

    def _analyze_feature_reliability(self, features: List[str], X, y) -> Dict[str, Any]:
        """Analyze reliability for specific features.

        This performs detailed reliability analysis on individual features,
        identifying regions of high and low confidence.
        """
        reliability_results = {}

        for feature in features:
            if self.verbose:
                print(f"  - Analyzing reliability for feature: {feature}")

            # Get feature data
            if isinstance(X, pd.DataFrame) and feature in X.columns:
                feature_data = X[feature].values
            else:
                # Skip if feature not found
                continue

            # Calculate reliability regions
            reliability_results[feature] = self._calculate_reliability_regions(
                feature_data, y, feature_name=feature
            )

        return reliability_results

    def _calculate_reliability_regions(self, feature_data: np.ndarray, y: np.ndarray,
                                      feature_name: str, n_bins: int = 10) -> Dict[str, Any]:
        """Calculate reliability regions for a specific feature.

        Identifies regions where the model predictions are more or less reliable
        based on the feature values.
        """
        # Create bins for the feature
        min_val, max_val = np.min(feature_data), np.max(feature_data)
        bin_edges = np.linspace(min_val, max_val, n_bins + 1)

        # Initialize results
        regions = {
            'bins': [],
            'confidence_scores': [],
            'sample_counts': [],
            'low_confidence_regions': [],
            'high_confidence_regions': []
        }

        # Analyze each bin
        for i in range(n_bins):
            bin_start, bin_end = bin_edges[i], bin_edges[i + 1]

            # Find samples in this bin
            if i == n_bins - 1:  # Last bin includes upper edge
                bin_mask = (feature_data >= bin_start) & (feature_data <= bin_end)
            else:
                bin_mask = (feature_data >= bin_start) & (feature_data < bin_end)

            n_samples = np.sum(bin_mask)

            if n_samples > 0:
                # Calculate confidence metric for this bin
                # Using a simple variance-based metric as proxy for confidence
                bin_y = y[bin_mask] if hasattr(y, '__getitem__') else y

                # Calculate prediction variance or uncertainty metric
                if hasattr(bin_y, 'std'):
                    # Higher variance = lower confidence
                    variance = np.std(bin_y)
                    confidence_score = 1.0 / (1.0 + variance)  # Normalize to [0, 1]
                else:
                    confidence_score = 0.5  # Default neutral confidence

                regions['bins'].append({
                    'start': bin_start,
                    'end': bin_end,
                    'center': (bin_start + bin_end) / 2
                })
                regions['confidence_scores'].append(confidence_score)
                regions['sample_counts'].append(n_samples)

                # Identify low/high confidence regions
                if confidence_score < 0.4:  # Low confidence threshold
                    regions['low_confidence_regions'].append({
                        'range': [bin_start, bin_end],
                        'confidence': confidence_score,
                        'samples': n_samples
                    })
                elif confidence_score > 0.7:  # High confidence threshold
                    regions['high_confidence_regions'].append({
                        'range': [bin_start, bin_end],
                        'confidence': confidence_score,
                        'samples': n_samples
                    })

        # Add summary statistics
        if regions['confidence_scores']:
            regions['avg_confidence'] = np.mean(regions['confidence_scores'])
            regions['min_confidence'] = np.min(regions['confidence_scores'])
            regions['max_confidence'] = np.max(regions['confidence_scores'])

        regions['feature_name'] = feature_name
        regions['n_bins'] = n_bins

        return regions

    def _generate_reliability_charts(self, reliability_analysis: Dict[str, Any]) -> Dict[str, str]:
        """Generate individual reliability charts for each analyzed feature.

        Returns dictionary with feature names as keys and base64 encoded charts as values.
        """
        charts = {}

        try:
            # Try to import chart generator
            from deepbridge.templates.report_types.uncertainty.static.charts.reliability_regions import ReliabilityRegionsChart
            chart_generator = ReliabilityRegionsChart()

            # Generate a chart for each feature
            for feature_name, feature_data in reliability_analysis.items():
                if self.verbose:
                    print(f"  Generating reliability chart for {feature_name}...")

                try:
                    # Generate single feature chart
                    chart_base64 = chart_generator.generate_single_feature(
                        feature_data=feature_data,
                        feature_name=feature_name,
                        title=f"Reliability Analysis: {feature_name}"
                    )

                    if chart_base64:
                        charts[feature_name] = chart_base64
                        if self.verbose:
                            print(f"    ✓ Chart generated for {feature_name}")
                    else:
                        if self.verbose:
                            print(f"    ✗ Failed to generate chart for {feature_name}")

                except Exception as e:
                    if self.verbose:
                        print(f"    ✗ Error generating chart for {feature_name}: {str(e)}")
                    continue

            # Also generate a comparison chart for all features
            if len(reliability_analysis) > 1:
                try:
                    comparison_chart = chart_generator.generate_comparison(
                        reliability_data=reliability_analysis,
                        title="Feature Confidence Comparison"
                    )
                    if comparison_chart:
                        charts['_comparison'] = comparison_chart
                        if self.verbose:
                            print(f"    ✓ Comparison chart generated")
                except Exception as e:
                    if self.verbose:
                        print(f"    ✗ Error generating comparison chart: {str(e)}")

        except ImportError as e:
            if self.verbose:
                print(f"Warning: Could not import chart generator: {e}")
                print("Charts will not be generated but analysis data is available")
        except Exception as e:
            if self.verbose:
                print(f"Error in chart generation: {str(e)}")

        return charts
    
    def _prepare_plot_data(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare formatted data for various plots."""
        plot_data = {
            'alpha_comparison': {},
            'width_distribution': [],
            'feature_importance': [],
            'coverage_vs_width': {}
        }
        
        # 1. Alpha comparison data
        alphas = []
        coverages = []
        expected_coverages = []
        mean_widths = []
        
        # Collect data for each alpha level
        for alpha, alpha_data in sorted(results['crqr']['by_alpha'].items()):
            overall = alpha_data.get('overall_result', {})
            if overall:
                alphas.append(alpha)
                coverages.append(overall.get('coverage', 0))
                expected_coverages.append(overall.get('expected_coverage', 0))
                mean_widths.append(overall.get('mean_width', 0))
        
        plot_data['alpha_comparison'] = {
            'alphas': alphas,
            'coverages': coverages,
            'expected_coverages': expected_coverages,
            'mean_widths': mean_widths
        }
        
        # 2. Width distribution data
        for alpha, alpha_data in sorted(results['crqr']['by_alpha'].items()):
            overall = alpha_data.get('overall_result', {})
            if overall and 'widths' in overall:
                plot_data['width_distribution'].append({
                    'alpha': alpha,
                    'widths': overall['widths']
                })
        
        # 3. Feature importance data
        importance = results.get('feature_importance', {})
        for feature, value in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            plot_data['feature_importance'].append({
                'feature': feature,
                'importance': value
            })
        
        # 4. Coverage vs width data
        plot_data['coverage_vs_width'] = {
            'coverages': coverages,
            'mean_widths': mean_widths,
            'alphas': alphas  # Include alphas as reference
        }
        
        return plot_data
    
    def predict_interval(self, X, alpha=0.1):
        """
        Predict confidence intervals for new data.
        
        Parameters:
        -----------
        X : DataFrame or array-like
            Features to predict intervals for
        alpha : float
            Confidence level (default: 0.1 for 90% confidence intervals)
        
        Returns:
        --------
        tuple: (lower_bounds, upper_bounds, point_predictions)
        """
        # Find the CRQR model with the closest alpha
        model_key = f"crqr_alpha_{alpha}"
        
        # If exact alpha not found, find the closest one
        if model_key not in self.uncertainty_models:
            available_alphas = [float(key.split('_')[-1]) for key in self.uncertainty_models.keys() 
                               if key.startswith('crqr_alpha_')]
            if not available_alphas:
                raise ValueError("No CRQR models available. Run the test suite first.")
            
            closest_alpha = min(available_alphas, key=lambda x: abs(x - alpha))
            model_key = f"crqr_alpha_{closest_alpha}"
            
            if self.verbose:
                print(f"No model with alpha={alpha} found. Using closest alpha={closest_alpha}")
        
        model = self.uncertainty_models[model_key]
        
        # Create a copy of the original X for predictions
        X_full = X.copy()
        
        # Check if we need to create a separate view for feature analysis
        X_analysis = X.copy()
        if self.feature_subset and isinstance(X, pd.DataFrame):
            # Ensure all features in feature_subset are in X
            valid_features = [f for f in self.feature_subset if f in X.columns]
            if len(valid_features) < len(self.feature_subset) and self.verbose:
                missing = set(self.feature_subset) - set(valid_features)
                print(f"Warning: Some requested features not found in dataset: {missing}")
            if valid_features:
                X_analysis = X[valid_features]
            elif self.verbose:
                print("No valid features in subset. Using all features.")
        
        # For uncertainty models like CRQR, we can use the subset
        # as they are trained specifically for the subset during evaluation
        # But for regular models we should use full feature set
        if isinstance(model, dict) and "model" in model:
            # This is a CRQR model that can handle the subset directly
            lower_bound, upper_bound = model.predict_interval(X_analysis)
            point_pred = model.predict(X_analysis)
        else:
            # For standard models, use the full feature set to avoid feature mismatch errors
            lower_bound, upper_bound = model.predict_interval(X_full)
            point_pred = model.predict(X_full)
        
        return lower_bound, upper_bound, point_pred
    
    def save_report(self, output_path: str) -> None:
        """
        This method has been deprecated as reporting functionality has been removed.
        
        Raises:
            NotImplementedError: Always raises this exception
        """
        raise NotImplementedError("Report generation functionality has been removed from this version.")


class CRQR:
    """
    Conformalized Residual Quantile Regression (CRQR)
    
    Uma abordagem model-agnostic para avaliar a confiabilidade de modelos de regressão 
    dentro do framework de predição conformal.
    """
    
    def __init__(self, base_model=None, alpha=0.1, test_size=0.6, calib_ratio=1/3, random_state=None):
        """
        Inicializador da classe CRQR.
        
        Parâmetros:
        - base_model: modelo de regressão base (padrão: None, usa o modelo padrão)
        - alpha: nível de significância (padrão: 0.1 para intervalos de confiança de 90%)
        - test_size: proporção dos dados a serem usados para teste+calibração (padrão: 0.6 = 60%)
        - calib_ratio: proporção do conjunto test_size a ser usada para calibração (padrão: 1/3, 
                      resultando em 20% do total para calibração e 40% para teste)
        - random_state: semente aleatória para reprodutibilidade
        """
        self.alpha = alpha
        self.test_size = test_size
        self.calib_ratio = calib_ratio
        self.random_state = random_state
        
        # Calcula as proporções efetivas
        self.train_size = 1 - test_size
        self.calib_size = test_size * calib_ratio
        self.test_size_final = test_size * (1 - calib_ratio)
        
        # Modelo base para regressão
        if base_model is None:
            self.base_model = None  # Será configurado durante o fit
        else:
            self.base_model = base_model
            
        # Modelos de regressão quantil
        self.quantile_model_lower = None
        self.quantile_model_upper = None
        
        # Valor de calibração
        self.q_hat = None
        
    def fit(self, X, y):
        """
        Treina o modelo base e os modelos de regressão quantil.
        
        Parâmetros:
        - X: features de treinamento
        - y: target de treinamento
        """
        # Divisão dos dados em conjuntos de treinamento e conjunto temporário (calibração + teste)
        # com base nos parâmetros definidos pelo usuário
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Divisão do conjunto temporário em calibração e teste
        # calib_ratio do conjunto temporário para calibração, o resto para teste
        X_calib, X_test, y_calib, y_test = train_test_split(
            X_temp, y_temp, 
            test_size=(1 - self.calib_ratio), 
            random_state=self.random_state
        )
        
        # Se não foi fornecido um modelo base, cria um modelo padrão
        if self.base_model is None:
            from sklearn.ensemble import HistGradientBoostingRegressor
            self.base_model = HistGradientBoostingRegressor(random_state=self.random_state)
        
        # Treina o modelo base com os dados de treinamento
        self.base_model.fit(X_train, y_train)
        
        # Prediz os valores para os dados de treinamento
        y_pred_train = self.base_model.predict(X_train)
        
        # Calcula os resíduos
        residuals = y_train - y_pred_train
        
        # Configura os modelos de regressão quantil
        try:
            # Tenta usar HistGradientBoostingRegressor com loss='quantile'
            import sklearn
            if sklearn.__version__ >= '1.1':
                from sklearn.ensemble import HistGradientBoostingRegressor
                self.quantile_model_lower = HistGradientBoostingRegressor(
                    loss='quantile', quantile=self.alpha/2, max_depth=5, random_state=self.random_state)
                self.quantile_model_upper = HistGradientBoostingRegressor(
                    loss='quantile', quantile=1-self.alpha/2, max_depth=5, random_state=self.random_state)
            else:
                # Para versões mais antigas do sklearn, usa GBDT diretamente
                from sklearn.ensemble import GradientBoostingRegressor
                self.quantile_model_lower = GradientBoostingRegressor(
                    loss='quantile', alpha=self.alpha/2, max_depth=5, random_state=self.random_state)
                self.quantile_model_upper = GradientBoostingRegressor(
                    loss='quantile', alpha=1-self.alpha/2, max_depth=5, random_state=self.random_state)
        except Exception as e:
            # Fallback para GradientBoostingRegressor
            from sklearn.ensemble import GradientBoostingRegressor
            self.quantile_model_lower = GradientBoostingRegressor(
                loss='quantile', alpha=self.alpha/2, max_depth=5, random_state=self.random_state)
            self.quantile_model_upper = GradientBoostingRegressor(
                loss='quantile', alpha=1-self.alpha/2, max_depth=5, random_state=self.random_state)
        
        # Treina os modelos de regressão quantil nos resíduos
        self.quantile_model_lower.fit(X_train, residuals)
        self.quantile_model_upper.fit(X_train, residuals)
        
        # Calibração usando o conjunto de calibração
        y_pred_calib = self.base_model.predict(X_calib)
        residuals_calib = y_calib - y_pred_calib
        
        # Prediz os limites dos resíduos para o conjunto de calibração
        lower_pred = self.quantile_model_lower.predict(X_calib)
        upper_pred = self.quantile_model_upper.predict(X_calib)
        
        # Calcula os scores de conformidade
        scores = np.maximum(lower_pred - residuals_calib, residuals_calib - upper_pred)
        
        # Calcula o quantil para o intervalo de confiança
        n = len(scores)
        level = np.ceil((n+1) * (1-self.alpha)) / n
        # Garante que level não exceda 1.0
        level = min(level, 1.0)
        self.q_hat = np.quantile(scores, level)
        
        # Armazena os conjuntos de dados para possível uso posterior
        self.X_train_ = X_train
        self.y_train_ = y_train
        self.X_calib_ = X_calib
        self.y_calib_ = y_calib
        self.X_test_ = X_test
        self.y_test_ = y_test
        
        return self
    
    def predict_interval(self, X):
        """
        Constrói intervalos de confiança para novas previsões.
        
        Parâmetros:
        - X: features para previsão
        
        Retorna:
        - Limites inferior e superior do intervalo de confiança
        """
        if self.base_model is None or self.quantile_model_lower is None or self.quantile_model_upper is None:
            raise ValueError("O modelo deve ser treinado antes de fazer previsões.")
        
        # Predição do modelo base
        y_pred = self.base_model.predict(X)
        
        # Predição dos limites dos resíduos
        lower_pred = self.quantile_model_lower.predict(X)
        upper_pred = self.quantile_model_upper.predict(X)
        
        # Constrói os intervalos de confiança
        lower_bound = y_pred + lower_pred - self.q_hat
        upper_bound = y_pred + upper_pred + self.q_hat
        
        return lower_bound, upper_bound
    
    def predict(self, X):
        """
        Realiza a previsão pontual usando o modelo base.
        
        Parâmetros:
        - X: features para previsão
        
        Retorna:
        - Previsões pontuais
        """
        if self.base_model is None:
            raise ValueError("O modelo deve ser treinado antes de fazer previsões.")
        
        return self.base_model.predict(X)
    
    def score(self, X, y):
        """
        Avalia a qualidade dos intervalos de predição.
        
        Parâmetros:
        - X: features de teste
        - y: targets de teste
        
        Retorna:
        - coverage: proporção de valores verdadeiros contidos nos intervalos
        - width: largura média dos intervalos
        """
        lower_bound, upper_bound = self.predict_interval(X)
        
        # Calcula a cobertura (porcentagem de valores verdadeiros dentro dos intervalos)
        coverage = np.mean((y >= lower_bound) & (y <= upper_bound))
        
        # Calcula a largura média dos intervalos
        width = np.mean(upper_bound - lower_bound)
        
        return {"coverage": coverage, "width": width}
    
    def get_split_sizes(self):
        """
        Retorna as proporções dos conjuntos de dados utilizados.
        
        Retorna:
        - Dicionário com os tamanhos proporcionais de cada conjunto
        """
        return {
            "training_set": self.train_size,
            "calibration_set": self.calib_size,
            "test_set": self.test_size_final
        }