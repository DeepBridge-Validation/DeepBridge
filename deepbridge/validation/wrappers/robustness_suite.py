"""
Simplified robustness testing suite for machine learning models.

This module provides a streamlined interface for testing model robustness
against feature perturbations using DBDataset objects.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
import time
import datetime

class RobustnessSuite:
    """
    Focused suite for model robustness testing with Gaussian noise and Quantile perturbation.
    """
    
    # Predefined configurations with varying perturbation levels
    _CONFIG_TEMPLATES = {
        'quick': [
            {'type': 'raw', 'params': {'level': 0.1}},
            {'type': 'raw', 'params': {'level': 0.2}},
            {'type': 'quantile', 'params': {'level': 0.1}},
            {'type': 'quantile', 'params': {'level': 0.2}}
        ],
        
        'medium': [
            {'type': 'raw', 'params': {'level': 0.1}},
            {'type': 'raw', 'params': {'level': 0.2}},
            {'type': 'raw', 'params': {'level': 0.4}},
            {'type': 'quantile', 'params': {'level': 0.1}},
            {'type': 'quantile', 'params': {'level': 0.2}},
            {'type': 'quantile', 'params': {'level': 0.4}}
        ],
        
        'full': [
            {'type': 'raw', 'params': {'level': 0.1}},
            {'type': 'raw', 'params': {'level': 0.2}},
            {'type': 'raw', 'params': {'level': 0.4}},
            {'type': 'raw', 'params': {'level': 0.6}},
            {'type': 'raw', 'params': {'level': 0.8}},
            {'type': 'raw', 'params': {'level': 1.0}},
            {'type': 'quantile', 'params': {'level': 0.1}},
            {'type': 'quantile', 'params': {'level': 0.2}},
            {'type': 'quantile', 'params': {'level': 0.4}},
            {'type': 'quantile', 'params': {'level': 0.6}},
            {'type': 'quantile', 'params': {'level': 0.8}},
            {'type': 'quantile', 'params': {'level': 1.0}}
        ]
    }
    
    def __init__(self, dataset, verbose: bool = False, metric: str = 'AUC', feature_subset: Optional[List[str]] = None):
        """
        Initialize the robustness testing suite.
        
        Parameters:
        -----------
        dataset : DBDataset
            Dataset object containing training/test data and model
        verbose : bool
            Whether to print progress information
        metric : str
            Performance metric to use for evaluation ('AUC', 'accuracy', 'mse', etc.)
        feature_subset : List[str] or None
            Subset of features to test (None for all features)
        """
        self.dataset = dataset
        self.verbose = verbose
        self.feature_subset = feature_subset
        self.metric = metric
        
        # Store current configuration
        self.current_config = None
        
        # Store results
        self.results = {}
        
        # Determine problem type based on dataset or model
        self._problem_type = self._determine_problem_type()
        
        if self.verbose:
            print(f"Problem type detected: {self._problem_type}")
            print(f"Using metric: {self.metric}")
    
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
    
    def config(self, config_name: str = 'quick', feature_subset: Optional[List[str]] = None) -> 'RobustnessSuite':
        """
        Set a predefined configuration for robustness tests.
        
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

        if config_name not in self._CONFIG_TEMPLATES:
            raise ValueError(f"Unknown configuration: {config_name}. Available options: {list(self._CONFIG_TEMPLATES.keys())}")
        
        # Clone the configuration template
        self.current_config = self._clone_config(self._CONFIG_TEMPLATES[config_name])
        
        # Update feature_subset in tests if specified
        if self.feature_subset:
            for test in self.current_config:
                if 'params' in test:
                    test['params']['feature_subset'] = self.feature_subset
        
        if self.verbose:
            print(f"\nConfigured for {config_name} robustness test suite")
            if self.feature_subset:
                print(f"Feature subset: {self.feature_subset}")
            print(f"\nTests that will be executed:")
            
            # Print all configured tests
            for i, test in enumerate(self.current_config, 1):
                test_type = test['type']
                params = test.get('params', {})
                param_str = ', '.join(f"{k}={v}" for k, v in params.items())
                print(f"  {i}. {test_type} ({param_str})")
        
        return self
    
    def _clone_config(self, config):
        """Clone configuration to avoid modifying original templates."""
        import copy
        return copy.deepcopy(config)
    
    def _perturb_data(self, X, perturb_method, level, perturb_features=None):
        """
        Perturb data using specified method and level.
        
        Parameters:
        -----------
        X : DataFrame or ndarray
            Feature data to perturb
        perturb_method : str
            Method to use ('raw' or 'quantile')
        level : float
            Level of perturbation to apply
        perturb_features : List[str] or None
            Specific features to perturb (None for all)
            
        Returns:
        --------
        DataFrame or ndarray : Perturbed data
        """
        if perturb_features is None:
            perturb_features = X.columns if isinstance(X, pd.DataFrame) else range(X.shape[1])
        
        X_perturbed = X.copy()
        for feature in perturb_features:
            if isinstance(X, pd.DataFrame):
                col = X.columns.get_loc(feature)
            else:
                col = feature
            
            if perturb_method == 'raw':
                # Apply Gaussian noise proportional to standard deviation
                feature_values = X.iloc[:, col] if isinstance(X, pd.DataFrame) else X[:, col]
                feature_std = np.std(feature_values)
                noise = np.random.normal(0, level * feature_std, X_perturbed.shape[0])
                
                if isinstance(X, pd.DataFrame):
                    X_perturbed.iloc[:, col] += noise
                else:
                    X_perturbed[:, col] += noise
                    
            elif perturb_method == 'quantile':
                # Apply quantile-based perturbation
                feature_values = X.iloc[:, col] if isinstance(X, pd.DataFrame) else X[:, col]
                quantiles = np.quantile(feature_values, [0.25, 0.75])
                perturbation = np.random.uniform(
                    quantiles[0] * (1 - level), 
                    quantiles[1] * (1 + level), 
                    X_perturbed.shape[0]
                )
                
                if isinstance(X, pd.DataFrame):
                    X_perturbed.iloc[:, col] = perturbation
                else:
                    X_perturbed[:, col] = perturbation
            else:
                raise ValueError(f"Perturb method {perturb_method} not supported.")
        
        return X_perturbed
    
    def _evaluate_performance(self, model, X, y):
        """Evaluate model performance using the specified metric."""
        if self._problem_type == 'classification':
            if self.metric.lower() == 'auc':
                if hasattr(model, 'predict_proba'):
                    from sklearn.metrics import roc_auc_score
                    y_prob = model.predict_proba(X)[:, 1]
                    return roc_auc_score(y, y_prob)
                else:
                    from sklearn.metrics import roc_auc_score
                    y_pred = model.predict(X)
                    return roc_auc_score(y, y_pred)
            else:
                from sklearn.metrics import accuracy_score, f1_score
                y_pred = model.predict(X)
                if self.metric.lower() == 'accuracy':
                    return accuracy_score(y, y_pred)
                elif self.metric.lower() == 'f1':
                    return f1_score(y, y_pred, average='weighted')
                else:
                    return accuracy_score(y, y_pred)
        else:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            y_pred = model.predict(X)
            if self.metric.lower() == 'mse':
                return mean_squared_error(y, y_pred)
            elif self.metric.lower() == 'rmse':
                return np.sqrt(mean_squared_error(y, y_pred))
            elif self.metric.lower() == 'mae':
                return mean_absolute_error(y, y_pred)
            elif self.metric.lower() == 'r2':
                return r2_score(y, y_pred)
            else:
                return mean_squared_error(y, y_pred)
    
    def evaluate_robustness(self, perturb_method: str, level: float, feature=None, n_iterations: int = 10) -> Dict[str, Any]:
        """
        Evaluate model robustness using multiple iterations of perturbation.
        
        Parameters:
        -----------
        perturb_method : str
            Method to use ('raw' or 'quantile')
        level : float
            Level of perturbation to apply
        feature : str or None
            Specific feature to perturb (None for all features)
        n_iterations : int
            Number of iterations for more reliable results
            
        Returns:
        --------
        dict : Detailed evaluation results
        """
        # Get dataset
        X_test = self.dataset.get_feature_data('test')
        y_test = self.dataset.get_target_data('test')
        model = self.dataset.model
        
        # Determine features to perturb
        if feature:
            perturb_features = [feature]
        else:
            perturb_features = None  # Will use all features
        
        # Get baseline performance
        baseline_score = self._evaluate_performance(model, X_test, y_test)
        
        # Run multiple iterations
        scores = []
        perturbed_data = []
        
        for i in range(n_iterations):
            # Perturb data
            X_perturbed = self._perturb_data(X_test.copy(), perturb_method, level, perturb_features)
            
            # Keep a sample of perturbed data (first 5 rows) for visualization
            if i == 0:
                perturbed_data = X_perturbed.iloc[:5].copy() if isinstance(X_perturbed, pd.DataFrame) else X_perturbed[:5].copy()
            
            # Evaluate performance
            score = self._evaluate_performance(model, X_perturbed, y_test)
            scores.append(score)
        
        # Calculate relative change
        mean_score = np.mean(scores)
        relative_change = (mean_score - baseline_score) / abs(baseline_score) if baseline_score != 0 else 0
        
        # Return detailed results
        return {
            'perturbation_type': perturb_method,
            'level': level,
            'feature': feature,
            'baseline_score': baseline_score,
            'individual_scores': scores,
            'mean_score': mean_score,
            'std_score': np.std(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'relative_change': relative_change,
            'perturbed_data_sample': perturbed_data
        }
    
    def run(self) -> Dict[str, Any]:
        """
        Run the configured robustness tests.
        
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
            print(f"Running robustness test suite...")
            start_time = time.time()
        
        # Initialize results
        results = {
            'raw': {
                'by_level': {},          # Results organized by perturbation level
                'by_feature': {},        # Results organized by feature
                'all_results': []        # All raw test results
            },
            'quantile': {
                'by_level': {},
                'by_feature': {},
                'all_results': []
            }
        }
        
        # Get the dataset's test data
        X_test = self.dataset.get_feature_data('test')
        y_test = self.dataset.get_target_data('test')
        features = X_test.columns.tolist()
        
        # Get baseline performance
        model = self.dataset.model
        baseline_score = self._evaluate_performance(model, X_test, y_test)
        
        # Store baseline performance
        results['baseline_performance'] = {
            self.metric.lower(): baseline_score
        }
        
        # Track levels for summary plots
        all_levels = {
            'raw': [],
            'quantile': []
        }
        
        # Run all configured tests
        for test_config in self.current_config:
            test_type = test_config['type']
            params = test_config.get('params', {})
            level = params.get('level', 0.1)
            feature_subset = params.get('feature_subset', None)
            
            # Track level
            if level not in all_levels[test_type]:
                all_levels[test_type].append(level)
            
            if self.verbose:
                print(f"Running {test_type} perturbation test with level {level}")
            
            # Initialize level results if needed
            if level not in results[test_type]['by_level']:
                results[test_type]['by_level'][level] = {
                    'individual_scores': [],
                    'feature_results': {}
                }
            
            # Test all features together first
            overall_result = self.evaluate_robustness(test_type, level, feature=None, n_iterations=10)
            results[test_type]['all_results'].append(overall_result)
            
            # Add to level-specific results
            results[test_type]['by_level'][level]['overall_result'] = overall_result
            results[test_type]['by_level'][level]['individual_scores'].extend(overall_result['individual_scores'])
            
            # Test individual features if a subset is specified
            if feature_subset:
                features_to_test = feature_subset
            else:
                # Test a sample of features (max 5) to keep runtime reasonable
                features_to_test = features[:5] if len(features) > 5 else features
            
            # Run per-feature tests
            for feature in features_to_test:
                if self.verbose:
                    print(f"  - Testing feature: {feature}")
                
                # Initialize feature results if needed
                if feature not in results[test_type]['by_feature']:
                    results[test_type]['by_feature'][feature] = {}
                
                # Run feature-specific test
                feature_result = self.evaluate_robustness(test_type, level, feature=feature, n_iterations=5)
                
                # Store results
                results[test_type]['by_feature'][feature][level] = feature_result
                results[test_type]['by_level'][level]['feature_results'][feature] = feature_result
        
        # Organize results for easier plotting
        results['perturbation_levels'] = all_levels
        
        # Calculate feature importance
        results['feature_importance'] = self._calculate_feature_importance(results)
        
        # Calculate method comparison data
        results['method_comparison'] = self._prepare_method_comparison(results)
        
        # Calculate overall performance impact
        raw_impact = []
        quantile_impact = []
        
        # Process raw perturbation results
        for result in results['raw']['all_results']:
            raw_impact.append(result['relative_change'])
        
        # Process quantile perturbation results
        for result in results['quantile']['all_results']:
            quantile_impact.append(result['relative_change'])
        
        # Calculate average impacts
        if raw_impact:
            results['avg_raw_impact'] = np.mean(raw_impact)
        if quantile_impact:
            results['avg_quantile_impact'] = np.mean(quantile_impact)
        
        # Calculate overall robustness score (higher is better)
        if raw_impact or quantile_impact:
            # Combine impacts, converting to positive scores (higher is better)
            if self._problem_type == 'classification' or self.metric.lower() not in ['mse', 'rmse', 'mae']:
                # For metrics where higher is better, less negative impact means better robustness
                combined_impact = []
                if raw_impact:
                    combined_impact.extend(raw_impact)
                if quantile_impact:
                    combined_impact.extend(quantile_impact)
                
                # Convert relative change to robustness score (0-1 scale)
                # Smaller negative impact → higher score
                average_impact = np.mean(combined_impact)
                robustness_score = max(0, min(1, 1 + average_impact))
            else:
                # For metrics where lower is better (like MSE), positive impact means better robustness
                combined_impact = []
                if raw_impact:
                    combined_impact.extend([-x for x in raw_impact])  # Invert so negative is good
                if quantile_impact:
                    combined_impact.extend([-x for x in quantile_impact])  # Invert so negative is good
                
                # Convert relative change to robustness score (0-1 scale)
                average_impact = np.mean(combined_impact)
                robustness_score = max(0, min(1, 1 + average_impact))
                
            results['robustness_score'] = robustness_score
        else:
            results['robustness_score'] = 0.5  # Default if no impacts calculated
        
        # Prepare data for plotting
        results['plot_data'] = self._prepare_plot_data(results)
        
        # Add execution time
        if self.verbose:
            elapsed_time = time.time() - start_time
            results['execution_time'] = elapsed_time
            print(f"Test suite completed in {elapsed_time:.2f} seconds")
            print(f"Overall robustness score: {results['robustness_score']:.3f}")
        
        # Store results
        test_id = f"test_{int(time.time())}"
        self.results[test_id] = results
                
        return results
    
    def _calculate_feature_importance(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate feature importance based on perturbation impact."""
        feature_importance = {}
        
        # Process both perturbation types
        for perturb_type in ['raw', 'quantile']:
            for feature, levels in results[perturb_type]['by_feature'].items():
                # Calculate average impact across levels
                impacts = []
                for level, result in levels.items():
                    impacts.append(result['relative_change'])
                
                # Average impact (negative = more important)
                if impacts:
                    avg_impact = np.mean(impacts)
                    
                    # Convert to importance score (higher = more important)
                    if self._problem_type == 'classification' or self.metric.lower() not in ['mse', 'rmse', 'mae']:
                        # For metrics where higher is better, negative impact = important
                        importance = max(0, -avg_impact)
                    else:
                        # For metrics where lower is better, positive impact = important
                        importance = max(0, avg_impact)
                    
                    # Update feature importance (use maximum from different methods)
                    if feature in feature_importance:
                        feature_importance[feature] = max(feature_importance[feature], importance)
                    else:
                        feature_importance[feature] = importance
        
        # Normalize to [0, 1] scale
        if feature_importance:
            max_importance = max(feature_importance.values())
            if max_importance > 0:
                feature_importance = {feature: value / max_importance for feature, value in feature_importance.items()}
        
        return feature_importance
    
    def _prepare_method_comparison(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for comparing perturbation methods."""
        comparison = {}
        
        # Get common levels if possible
        raw_levels = results['perturbation_levels']['raw']
        quantile_levels = results['perturbation_levels']['quantile']
        
        # Get scores for each method and level
        raw_scores = []
        raw_stds = []
        for level in raw_levels:
            if level in results['raw']['by_level']:
                level_result = results['raw']['by_level'][level].get('overall_result', {})
                raw_scores.append(level_result.get('mean_score', 0))
                raw_stds.append(level_result.get('std_score', 0))
            else:
                raw_scores.append(0)
                raw_stds.append(0)
                
        quantile_scores = []
        quantile_stds = []
        for level in quantile_levels:
            if level in results['quantile']['by_level']:
                level_result = results['quantile']['by_level'][level].get('overall_result', {})
                quantile_scores.append(level_result.get('mean_score', 0))
                quantile_stds.append(level_result.get('std_score', 0))
            else:
                quantile_scores.append(0)
                quantile_stds.append(0)
        
        # Store comparison data
        comparison = {
            'raw': {
                'levels': raw_levels,
                'scores': raw_scores,
                'stds': raw_stds
            },
            'quantile': {
                'levels': quantile_levels,
                'scores': quantile_scores,
                'stds': quantile_stds
            }
        }
        
        return comparison
    
    def _prepare_plot_data(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare formatted data for various plots."""
        plot_data = {
            'robustness_by_level': {},
            'distribution_data': [],
            'feature_importance': [],
            'method_comparison': {}
        }
        
        # 1. Robustness by level data
        for method in ['raw', 'quantile']:
            levels = []
            means = []
            stds = []
            mins = []
            maxes = []
            
            for level, level_data in sorted(results[method]['by_level'].items()):
                overall = level_data.get('overall_result', {})
                if overall:
                    levels.append(level)
                    means.append(overall.get('mean_score', 0))
                    stds.append(overall.get('std_score', 0))
                    mins.append(overall.get('min_score', 0))
                    maxes.append(overall.get('max_score', 0))
            
            plot_data['robustness_by_level'][method] = {
                'levels': levels,
                'means': means,
                'stds': stds,
                'mins': mins,
                'maxes': maxes
            }
            
        # 2. Distribution data for boxplots
        for method in ['raw', 'quantile']:
            for level, level_data in sorted(results[method]['by_level'].items()):
                scores = level_data.get('individual_scores', [])
                if scores:
                    plot_data['distribution_data'].append({
                        'method': method,
                        'level': level,
                        'scores': scores
                    })
        
        # 3. Feature importance data
        importance = results.get('feature_importance', {})
        for feature, value in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            plot_data['feature_importance'].append({
                'feature': feature,
                'importance': value
            })
        
        # 4. Method comparison data
        if 'method_comparison' in results:
            plot_data['method_comparison'] = results['method_comparison']
        
        return plot_data
    
    def plot_robustness(self, model_results=None, title="Robustness by Perturbation Level"):
        """
        Plot robustness scores for different perturbation levels.
        
        Parameters:
        -----------
        model_results : dict or None
            Results to plot (uses most recent results if None)
        title : str
            Plot title
            
        Returns:
        --------
        plotly.graph_objects.Figure
        """
        try:
            import plotly.graph_objects as go
            
            # Use most recent results if none provided
            if model_results is None:
                if not self.results:
                    raise ValueError("No results available. Run a test first.")
                last_test_key = list(self.results.keys())[-1]
                model_results = self.results[last_test_key]
            
            # Extract plot data
            plot_data = model_results.get('plot_data', {}).get('robustness_by_level', {})
            
            # Create figure
            fig = go.Figure()
            
            # Add traces for each perturbation method
            for method, data in plot_data.items():
                # Skip if no data
                if not data.get('levels'):
                    continue
                    
                # Add line with error bars
                fig.add_trace(go.Scatter(
                    x=data['levels'],
                    y=data['means'],
                    mode='lines+markers',
                    name=f"{method.title()} Perturbation",
                    error_y=dict(
                        type='data',
                        array=data['stds'],
                        visible=True
                    )
                ))
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title="Perturbation Level",
                yaxis_title=f"{self.metric} Score",
                height=500,
                width=800,
                template="plotly_white"
            )
            
            return fig
        except ImportError:
            print("Plotly is required for plotting. Install with: pip install plotly")
            return None
    
    def plot_feature_importance(self, model_results=None, top_n=10):
        """
        Plot feature importance based on robustness.
        
        Parameters:
        -----------
        model_results : dict or None
            Results to plot (uses most recent results if None)
        top_n : int
            Number of top features to include
            
        Returns:
        --------
        plotly.graph_objects.Figure
        """
        try:
            import plotly.graph_objects as go
            
            # Use most recent results if none provided
            if model_results is None:
                if not self.results:
                    raise ValueError("No results available. Run a test first.")
                last_test_key = list(self.results.keys())[-1]
                model_results = self.results[last_test_key]
            
            # Extract feature importance data
            feature_importance = model_results.get('feature_importance', {})
            
            # Sort features by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Limit to top N features
            if top_n > 0 and len(sorted_features) > top_n:
                sorted_features = sorted_features[:top_n]
            
            features = [f[0] for f in sorted_features]
            importance_values = [f[1] for f in sorted_features]
            
            # Create horizontal bar chart
            fig = go.Figure([go.Bar(
                x=importance_values,
                y=features,
                orientation='h',
                marker=dict(
                    color=importance_values,
                    colorscale='Viridis',
                    colorbar=dict(title='Importance')
                )
            )])
            
            # Update layout
            fig.update_layout(
                title="Feature Importance Based on Robustness",
                xaxis_title="Importance Score",
                yaxis_title="Feature",
                height=max(400, len(features) * 25),  # Adjust height based on number of features
                width=800,
                template="plotly_white",
                yaxis=dict(
                    autorange="reversed"  # Labels read top-to-bottom
                )
            )
            
            return fig
        except ImportError:
            print("Plotly is required for plotting. Install with: pip install plotly")
            return None
    
    def save_report(self, output_path: str) -> None:
        """
        Save robustness test results to a simple text report file.
        
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
            "# Robustness Test Report",
            f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Model: {self.dataset.model.__class__.__name__}",
            f"Problem type: {self._problem_type}",
            f"Metric used: {self.metric}",
            "",
            "## Summary",
            f"Overall robustness score: {test_results.get('robustness_score', 0):.3f}",
            f"Baseline {self.metric} performance: {test_results.get('baseline_performance', {}).get(self.metric.lower(), 0):.3f}",
            "",
            "## Perturbation Results"
        ]
        
        # Add Raw perturbation results
        report_lines.append("\n### Gaussian Noise Perturbation")
        report_lines.append(f"Average impact: {test_results.get('avg_raw_impact', 0):.3f}")
        
        # Add results by level
        for level, level_data in sorted(test_results.get('raw', {}).get('by_level', {}).items()):
            overall = level_data.get('overall_result', {})
            if overall:
                report_lines.append(f"- Level: {level}, Mean Score: {overall.get('mean_score', 0):.3f}, Std: {overall.get('std_score', 0):.3f}")
        
        # Add Quantile perturbation results
        report_lines.append("\n### Quantile Perturbation")
        report_lines.append(f"Average impact: {test_results.get('avg_quantile_impact', 0):.3f}")
        
        # Add results by level
        for level, level_data in sorted(test_results.get('quantile', {}).get('by_level', {}).items()):
            overall = level_data.get('overall_result', {})
            if overall:
                report_lines.append(f"- Level: {level}, Mean Score: {overall.get('mean_score', 0):.3f}, Std: {overall.get('std_score', 0):.3f}")
        
        # Add feature importance section
        report_lines.append("\n## Feature Importance")
        
        # Get feature importance
        importance = test_results.get('feature_importance', {})
        
        # Sort features by importance
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        # Limit to top 10 features
        if len(sorted_features) > 10:
            sorted_features = sorted_features[:10]
            report_lines.append("Top 10 most important features:")
        else:
            report_lines.append("Feature importance:")
            
        for feature, value in sorted_features:
            report_lines.append(f"- {feature}: {value:.3f}")
        
        # Add execution time
        if 'execution_time' in test_results:
            report_lines.append(f"\nExecution time: {test_results['execution_time']:.2f} seconds")
        
        # Write report to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
            
        if self.verbose:
            print(f"Report saved to {output_path}")