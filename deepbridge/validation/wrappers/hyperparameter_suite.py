"""
Simplified hyperparameter importance testing suite for machine learning models.

This module provides a streamlined interface for evaluating the importance of 
different hyperparameters in model performance using subsampling techniques.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
import time
import datetime
from sklearn.model_selection import cross_val_score, train_test_split

class HyperparameterSuite:
    """
    Focused suite for analyzing hyperparameter importance in machine learning models.
    """
    
    # Predefined configurations with varying test intensities
    _CONFIG_TEMPLATES = {
        'quick': [
            {'method': 'importance', 'params': {'cv': 3, 'n_subsamples': 5, 'subsample_size': 0.5}}
        ],
        
        'medium': [
            {'method': 'importance', 'params': {'cv': 5, 'n_subsamples': 10, 'subsample_size': 0.5}}
        ],
        
        'full': [
            {'method': 'importance', 'params': {'cv': 5, 'n_subsamples': 20, 'subsample_size': 0.5}},
            {'method': 'importance', 'params': {'cv': 5, 'n_subsamples': 10, 'subsample_size': 0.7}}
        ]
    }
    
    def __init__(self, dataset, verbose: bool = False, random_state: Optional[int] = None, metric: str = 'accuracy', feature_subset: Optional[List[str]] = None):
        """
        Initialize the hyperparameter importance testing suite.
        
        Parameters:
        -----------
        dataset : DBDataset
            Dataset object containing training data and model
        verbose : bool
            Whether to print progress information
        random_state : int or None
            Random seed for reproducibility
        metric : str
            Performance metric to use ('accuracy', 'auc', 'f1', 'mse', etc.)
        feature_subset : List[str] or None
            Specific features to focus on for testing (None for all features)
        """
        self.dataset = dataset
        self.verbose = verbose
        self.random_state = random_state
        self.metric = metric
        self.feature_subset = feature_subset
        
        # Store current configuration
        self.current_config = None
        
        # Store results
        self.results = {}
        
        # Determine problem type based on dataset or model
        self._problem_type = self._determine_problem_type()
        
        if self.verbose:
            print(f"Problem type detected: {self._problem_type}")
            
        # Get model and param grid
        self.model = self._get_model()
        self.param_grid = self._get_param_grid()
    
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
    
    def _get_model(self):
        """Get the model from the dataset"""
        if hasattr(self.dataset, 'model') and self.dataset.model is not None:
            return self.dataset.model
        
        raise ValueError("No model found in dataset")
    
    def _get_param_grid(self):
        """
        Get the parameter grid for the model.
        This either comes from the dataset or is inferred from the model.
        """
        if hasattr(self.dataset, 'param_grid') and self.dataset.param_grid is not None:
            return self.dataset.param_grid
        
        # Infer basic param grid based on model type
        model = self.model
        model_type = type(model).__name__
        
        # Default parameter grids for common models
        if model_type == 'RandomForestClassifier' or model_type == 'RandomForestRegressor':
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif model_type == 'GradientBoostingClassifier' or model_type == 'GradientBoostingRegressor':
            return {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 10],
                'min_samples_split': [2, 5, 10]
            }
        elif model_type == 'LogisticRegression':
            return {
                'C': [0.01, 0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        elif model_type == 'SVC' or model_type == 'SVR':
            return {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto', 0.1, 1.0]
            }
        else:
            # Try to get all params and use a simple grid
            try:
                all_params = model.get_params()
                param_grid = {}
                
                # Create a simple grid for numeric parameters
                for param, value in all_params.items():
                    if isinstance(value, (int, float)) and param != 'random_state':
                        if isinstance(value, int):
                            param_grid[param] = [max(1, value // 2), value, value * 2]
                        else:
                            param_grid[param] = [value / 2, value, value * 2]
                
                if param_grid:
                    return param_grid
            except:
                pass
            
            # If we get here, we couldn't infer a good grid
            if self.verbose:
                print(f"Could not infer parameter grid for model type: {model_type}")
                print("Using a minimal default grid")
            
            # Return a minimal grid that should work for most models
            return {'random_state': [self.random_state or 42]}
    
    def config(self, config_name: str = 'quick', feature_subset: Optional[List[str]] = None) -> 'HyperparameterSuite':
        """
        Set a predefined configuration for hyperparameter importance tests.
        
        Parameters:
        -----------
        config_name : str
            Name of the configuration to use: 'quick', 'medium', or 'full'
        feature_subset : List[str] or None
            Specific features to focus on for testing (overrides the one set in constructor)
                
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
            print(f"\nConfigured for {config_name} hyperparameter importance test suite")
            if self.feature_subset:
                print(f"Feature subset: {self.feature_subset}")
            print(f"\nTests that will be executed:")
            
            # Print all configured tests
            for i, test in enumerate(self.current_config, 1):
                test_method = test['method']
                params = test.get('params', {})
                # Filter out feature_subset from display for cleaner output
                display_params = {k: v for k, v in params.items() if k != 'feature_subset'}
                param_str = ', '.join(f"{k}={v}" for k, v in display_params.items())
                print(f"  {i}. {test_method} ({param_str})")
        
        return self
    
    def _clone_config(self, config):
        """Clone configuration to avoid modifying original templates."""
        import copy
        return copy.deepcopy(config)
    
    def _evaluate_model(self, X, y, params, cv=5):
        """
        Evaluate the model with given parameters using cross-validation.
        
        Parameters:
        -----------
        X : DataFrame or array-like
            The feature data
        y : Series or array-like
            The target data
        params : dict
            Parameter values to set
        cv : int
            Number of cross-validation folds
            
        Returns:
        --------
        float : Mean cross-validation score
        """
        # Set the parameters on a copy of the model
        try:
            model = self.model.__class__(**{**self.model.get_params(), **params})
        except Exception as e:
            if self.verbose:
                print(f"Error setting parameters: {str(e)}")
                print(f"Using default model")
            model = self.model
        
        # Determine the scoring metric
        if self._problem_type == 'classification':
            if self.metric == 'accuracy':
                scoring = 'accuracy'
            elif self.metric == 'auc':
                scoring = 'roc_auc'
            elif self.metric == 'f1':
                scoring = 'f1'
            else:
                scoring = self.metric
        else:  # regression
            if self.metric == 'mse':
                scoring = 'neg_mean_squared_error'
            elif self.metric == 'mae':
                scoring = 'neg_mean_absolute_error'
            elif self.metric == 'r2':
                scoring = 'r2'
            else:
                scoring = self.metric
        
        # Perform cross-validation
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            return np.mean(scores)
        except Exception as e:
            if self.verbose:
                print(f"Error during cross-validation: {str(e)}")
            return float('-inf')  # Return a very low score on error
    
    def evaluate_importance(self, method: str, params: Dict) -> Dict[str, Any]:
        """
        Evaluate hyperparameter importance using subsampling.
        
        Parameters:
        -----------
        method : str
            Method to use ('importance')
        params : Dict
            Parameters for the importance calculation method
            
        Returns:
        --------
        dict : Detailed evaluation results
        """
        # Get parameters
        cv = params.get('cv', 5)
        n_subsamples = params.get('n_subsamples', 10)
        subsample_size = params.get('subsample_size', 0.5)
        feature_subset = params.get('feature_subset', self.feature_subset)
        
        # Get dataset
        X = self.dataset.get_feature_data()
        y = self.dataset.get_target_data()
        
        # Convert any numpy arrays to pandas objects if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        
        # Create a copy of the full feature set for model predictions
        X_full = X.copy()
            
        # Create a filtered version for analysis if feature_subset is provided
        X_analysis = X.copy()
        if feature_subset:
            if self.verbose:
                print(f"Focusing on feature subset: {feature_subset}")
            # Make sure all features in feature_subset are in X
            valid_features = [f for f in feature_subset if f in X.columns]
            if len(valid_features) < len(feature_subset):
                missing = set(feature_subset) - set(valid_features)
                if self.verbose:
                    print(f"Warning: Some requested features not found in dataset: {missing}")
            if valid_features:
                X_analysis = X[valid_features]
                # For hyperparameter importance evaluation, we need to set X to X_analysis
                # as we're training models on this subset
                X = X_analysis
            elif self.verbose:
                print("No valid features in subset. Using all features.")
        
        # Initialize dictionary to store variation in performance for each parameter
        param_variations = {param: [] for param in self.param_grid.keys()}
        
        # Performance data for detailed analysis
        performance_data = {param: {} for param in self.param_grid.keys()}
        
        # For each subsample
        for i in range(n_subsamples):
            if self.verbose and (i == 0 or (i+1) % 5 == 0):
                print(f"Processing subsample {i+1}/{n_subsamples}")
            
            # Create a subsample of the data
            X_sub, _, y_sub, _ = train_test_split(
                X, y, 
                test_size=1-subsample_size, 
                random_state=(self.random_state or 0) + i
            )
            
            # For each hyperparameter
            for param_name, param_values in self.param_grid.items():
                # Store performances with different values of the hyperparameter
                param_performances = []
                
                # For each possible value of the hyperparameter
                for value in param_values:
                    # Define default parameters (using first value from each parameter range)
                    default_params = {p: v[0] for p, v in self.param_grid.items()}
                    
                    # Modify only the current hyperparameter
                    default_params[param_name] = value
                    
                    # Evaluate the model
                    performance = self._evaluate_model(X_sub, y_sub, default_params, cv=cv)
                    param_performances.append(performance)
                    
                    # Store in performance data
                    if str(value) not in performance_data[param_name]:
                        performance_data[param_name][str(value)] = []
                    performance_data[param_name][str(value)].append(performance)
                
                # Calculate the variation in performance for this hyperparameter in this subsample
                variation = np.std(param_performances)
                param_variations[param_name].append(variation)
        
        # Calculate importance as the mean variation in performance across subsamples
        raw_importance_scores = {param: np.mean(variations) for param, variations in param_variations.items()}
        
        # Normalize scores to sum to 1
        total = sum(raw_importance_scores.values()) or 1.0  # Avoid division by zero
        normalized_importance = {param: score/total for param, score in raw_importance_scores.items()}
        
        # Sort parameters by importance
        sorted_importance = dict(sorted(normalized_importance.items(), key=lambda x: x[1], reverse=True))
        
        # Calculate average performance for each parameter value
        avg_performance = {}
        for param, values_dict in performance_data.items():
            avg_performance[param] = {value: np.mean(perfs) for value, perfs in values_dict.items()}
        
        # Return detailed results
        return {
            "method": "importance",
            "cv": cv,
            "n_subsamples": n_subsamples,
            "subsample_size": subsample_size,
            "raw_importance_scores": raw_importance_scores,
            "normalized_importance": normalized_importance,
            "sorted_importance": sorted_importance,
            "param_variations": param_variations,
            "performance_data": avg_performance,
            "tuning_order": list(sorted_importance.keys())
        }
    
    def run(self) -> Dict[str, Any]:
        """
        Run the configured hyperparameter importance tests.
        
        Returns:
        --------
        dict : Test results with detailed metrics
        """
        if self.current_config is None:
            # Default to quick config if none selected
            if self.verbose:
                print("No configuration set, using 'quick' configuration")
            self.config('quick')
                
        if self.verbose:
            print(f"Running hyperparameter importance test suite...")
            if self.feature_subset:
                print(f"Using feature subset: {self.feature_subset}")
            start_time = time.time()
        
        # Initialize results
        results = {
            'importance': {
                'by_config': {},           # Results organized by test configuration
                'all_results': []          # All raw test results
            }
        }
        
        # Run all configured tests
        for test_config in self.current_config:
            method = test_config['method']
            params = test_config.get('params', {})
            
            if method == 'importance':
                cv = params.get('cv', 5)
                n_subsamples = params.get('n_subsamples', 10)
                subsample_size = params.get('subsample_size', 0.5)
                
                if self.verbose:
                    print(f"Running hyperparameter importance analysis with cv={cv}, " 
                          f"n_subsamples={n_subsamples}, subsample_size={subsample_size}")
                
                # Run the importance evaluation
                test_result = self.evaluate_importance(method, params)
                results['importance']['all_results'].append(test_result)
                
                # Organize results by configuration
                config_key = f"cv{cv}_subs{n_subsamples}_size{subsample_size}"
                results['importance']['by_config'][config_key] = test_result
        
        # Aggregate importance scores across all configurations
        all_importance_scores = {}
        for result in results['importance']['all_results']:
            for param, score in result['normalized_importance'].items():
                if param not in all_importance_scores:
                    all_importance_scores[param] = []
                all_importance_scores[param].append(score)
        
        # Calculate average importance score for each parameter
        avg_importance = {param: np.mean(scores) for param, scores in all_importance_scores.items()}
        
        # Normalize average scores to sum to 1
        total = sum(avg_importance.values()) or 1.0  # Avoid division by zero
        normalized_avg_importance = {param: score/total for param, score in avg_importance.items()}
        
        # Sort parameters by average importance
        sorted_avg_importance = dict(sorted(normalized_avg_importance.items(), key=lambda x: x[1], reverse=True))
        
        # Store overall results
        results['importance_scores'] = normalized_avg_importance
        results['sorted_importance'] = sorted_avg_importance
        results['tuning_order'] = list(sorted_avg_importance.keys())
        
        # Add execution time
        if self.verbose:
            elapsed_time = time.time() - start_time
            # Não armazenamos mais o tempo de execução nos resultados
            print(f"Test suite completed in {elapsed_time:.2f} seconds")
            
            # Print importance scores
            print("\nHyperparameter importance scores:")
            for param, score in sorted_avg_importance.items():
                print(f"  {param}: {score:.4f}")
            
            # Print suggested tuning order
            print("\nSuggested hyperparameter tuning order:")
            for i, param in enumerate(results['tuning_order'], 1):
                print(f"  {i}. {param} (importance: {sorted_avg_importance[param]:.4f})")
        
        # Store results
        test_id = f"test_{int(time.time())}"
        self.results[test_id] = results
                
        return results
    
    def save_report(self, output_path: str) -> None:
        """
        Save hyperparameter importance test results to a simple text report file.
        
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
            "# Hyperparameter Importance Report",
            f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Model: {type(self.model).__name__}",
            f"Problem type: {self._problem_type}",
            "",
            "## Hyperparameter Importance Scores"
        ]
        
        # Add importance scores sorted by importance
        for param, score in test_results.get('sorted_importance', {}).items():
            report_lines.append(f"- {param}: {score:.4f}")
        
        # Add suggested tuning order
        report_lines.append("\n## Suggested Hyperparameter Tuning Order")
        for i, param in enumerate(test_results.get('tuning_order', []), 1):
            report_lines.append(f"{i}. {param}")
        
        # Add detailed results for each test configuration
        report_lines.append("\n## Detailed Test Results")
        for config_key, config_results in test_results.get('importance', {}).get('by_config', {}).items():
            report_lines.append(f"\n### Configuration: {config_key}")
            report_lines.append(f"CV folds: {config_results.get('cv')}")
            report_lines.append(f"Subsamples: {config_results.get('n_subsamples')}")
            report_lines.append(f"Subsample size: {config_results.get('subsample_size')}")
            
            report_lines.append("\nImportance scores:")
            for param, score in config_results.get('sorted_importance', {}).items():
                report_lines.append(f"- {param}: {score:.4f}")
            
            # Add performance data for most important parameter
            if config_results.get('sorted_importance') and config_results.get('performance_data'):
                top_param = next(iter(config_results.get('sorted_importance', {})), None)
                if top_param:
                    report_lines.append(f"\nPerformance for different values of {top_param}:")
                    perf_data = config_results.get('performance_data', {}).get(top_param, {})
                    for value, score in perf_data.items():
                        report_lines.append(f"- {value}: {score:.4f}")
        
        # Add execution time
        if 'execution_time' in test_results:
            report_lines.append(f"\nExecution time: {test_results['execution_time']:.2f} seconds")
        
        # Write report to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
            
        if self.verbose:
            print(f"Report saved to {output_path}")