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
import os

from deepbridge.validation.wrappers.robustness import (
    DataPerturber, 
    RobustnessEvaluator
)

class RobustnessSuite:
    """
    Focused suite for model robustness testing with Gaussian noise and Quantile perturbation.
    This class has been refactored to use specialized components for data perturbation,
    robustness evaluation, visualization, and reporting.
    """
    
    # Predefined configurations with varying perturbation levels
    _CONFIG_TEMPLATES = {
        'quick': [
            {'type': 'raw', 'params': {'level': 0.1}},
            {'type': 'raw', 'params': {'level': 0.2}}
        ],

        'medium': [
            {'type': 'raw', 'params': {'level': 0.1}},
            {'type': 'raw', 'params': {'level': 0.2}},
            {'type': 'raw', 'params': {'level': 0.4}}
        ],

        'full': [
            {'type': 'raw', 'params': {'level': 0.1}},
            {'type': 'raw', 'params': {'level': 0.2}},
            {'type': 'raw', 'params': {'level': 0.4}},
            {'type': 'raw', 'params': {'level': 0.6}},
            {'type': 'raw', 'params': {'level': 0.8}},
            {'type': 'raw', 'params': {'level': 1.0}}
        ],

        # Method comparison configurations (includes both raw and quantile)
        'quick_compare': [
            {'type': 'raw', 'params': {'level': 0.1}},
            {'type': 'raw', 'params': {'level': 0.2}},
            {'type': 'quantile', 'params': {'level': 0.1}},
            {'type': 'quantile', 'params': {'level': 0.2}}
        ],

        'medium_compare': [
            {'type': 'raw', 'params': {'level': 0.1}},
            {'type': 'raw', 'params': {'level': 0.2}},
            {'type': 'raw', 'params': {'level': 0.4}},
            {'type': 'quantile', 'params': {'level': 0.1}},
            {'type': 'quantile', 'params': {'level': 0.2}},
            {'type': 'quantile', 'params': {'level': 0.4}}
        ],

        'full_compare': [
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
    
    def __init__(self, 
                 dataset, 
                 verbose: bool = False, 
                 metric: str = 'AUC', 
                 feature_subset: Optional[List[str]] = None, 
                 random_state: Optional[int] = None,
                 n_iterations: int = 1):
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
        random_state : int or None
            Random seed for reproducibility
        n_iterations : int
            Number of iterations to perform for each perturbation level to get statistical robustness
        """
        self.dataset = dataset
        self.verbose = verbose
        self.feature_subset = feature_subset
        self.metric = metric
        self.n_iterations = n_iterations
        
        # Initialize components
        self.data_perturber = DataPerturber()
        if random_state is not None:
            self.data_perturber.set_random_state(random_state)
            
        self.evaluator = RobustnessEvaluator(dataset, metric, verbose, random_state, n_iterations)
        # Visualization and reporting functionality has been removed
        
        # Store current configuration
        self.current_config = None
        
        # Store results
        self.results = {}
        
        if self.verbose:
            print(f"Robustness Suite initialized with metric: {self.metric}")
            print(f"Using {self.n_iterations} iterations per perturbation level")
    
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
                param_str = ', '.join(f"{k}={v}" for k, v in params.items() if k != 'feature_subset')
                print(f"  {i}. {test_type} ({param_str})")
        
        return self
    
    def _clone_config(self, config):
        """Clone configuration to avoid modifying original templates."""
        import copy
        return copy.deepcopy(config)
    
    def run(self, X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Run the configured robustness tests.
        
        Parameters:
        -----------
        X : DataFrame, optional
            Feature data to use (if None, will use test data from dataset)
        y : Series, optional
            Target variable (if None, will use test target from dataset)
            
        Returns:
        --------
        Dict[str, Any] : Dictionary with test results
        """
        if self.current_config is None:
            # Use default configuration if none specified
            self.config('quick')
            
        if X is None or y is None:
            # Use test data from dataset if not provided
            if hasattr(self.dataset, 'test_data') and self.dataset.test_data is not None:
                X = self.dataset.get_feature_data('test')
                y = self.dataset.get_target_data('test')
            else:
                raise ValueError("No test data available in dataset. Please provide X and y.")
        
        # Track execution time
        start_time = time.time()
        
        if self.verbose:
            print(f"\nRunning robustness tests on dataset with {X.shape[0]} rows and {X.shape[1]} columns")
            
        # Initialize results structure
        results = {
            'base_score': 0,
            'raw': {'by_level': {}, 'overall': {}},
            'quantile': {'by_level': {}, 'overall': {}},
            'feature_importance': {},
            'feature_subset': self.feature_subset,  # Store the feature subset used in the test
            'metric': self.metric  # Store the metric used for evaluation
        }
        
        # Get model's native feature importance
        try:
            model_feature_importance = self.evaluator.get_model_feature_importance()
            
            if model_feature_importance:
                if self.verbose:
                    print(f"Model feature importance detected with {len(model_feature_importance)} features")
                    top_features = sorted(model_feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                    print(f"Top 5 important features: {top_features}")
                results['model_feature_importance'] = model_feature_importance
            else:
                if self.verbose:
                    print("WARNING: Model does not provide native feature importance")
                # Initialize with empty dictionary to avoid None
                results['model_feature_importance'] = {}
        except Exception as e:
            print(f"ERROR getting model feature importance: {str(e)}")
            # Initialize with empty dictionary
            results['model_feature_importance'] = {}
        
        # Calculate baseline score
        base_score = self.evaluator.calculate_base_score(X, y)
        results['base_score'] = base_score
        
        if self.verbose:
            print(f"Baseline score: {base_score:.3f}")
            
        # Process each test configuration
        all_raw_impacts = []
        all_quantile_impacts = []
        all_impacts = []
        
        for test_idx, test_config in enumerate(self.current_config, 1):
            test_type = test_config['type']
            params = test_config.get('params', {})
            
            # Extract parameters with defaults
            level = params.get('level', 0.1)
            test_feature_subset = params.get('feature_subset', self.feature_subset)
            
            if self.verbose:
                print(f"\nRunning test {test_idx}/{len(self.current_config)}: {test_type}, level={level}")
                
            # Store result by level
            level_key = str(level)

            # For feature subset comparison, test both all_features and feature_subset
            if test_feature_subset:
                # Test both all features and feature subset for comparison
                eval_results = {}

                # Test all features
                eval_results['all_features'] = self.evaluator.evaluate_perturbation(
                    X, y, test_type, level, None  # None = all features
                )

                # Test feature subset
                eval_results['feature_subset'] = self.evaluator.evaluate_perturbation(
                    X, y, test_type, level, test_feature_subset
                )
            else:
                # Only test all features if no subset specified
                eval_results = {
                    'all_features': self.evaluator.evaluate_perturbation(
                        X, y, test_type, level, None
                    )
                }
            
            # Process results for raw method
            if test_type == 'raw':
                if level_key not in results['raw']['by_level']:
                    results['raw']['by_level'][level_key] = {'runs': {}, 'overall_result': {}}

                # Initialize storage for both all_features and feature_subset if they don't exist
                if 'all_features' not in results['raw']['by_level'][level_key]['runs']:
                    results['raw']['by_level'][level_key]['runs']['all_features'] = []
                if 'feature_subset' not in results['raw']['by_level'][level_key]['runs']:
                    results['raw']['by_level'][level_key]['runs']['feature_subset'] = []

                # Process each eval_result
                for run_key, eval_result in eval_results.items():
                    # Add result to the appropriate category
                    results['raw']['by_level'][level_key]['runs'][run_key].append(eval_result)
                    all_raw_impacts.append(eval_result['impact'])
                    all_impacts.append(eval_result['impact'])

                    # Calculate overall result for this level and run_key
                    runs = results['raw']['by_level'][level_key]['runs'][run_key]
                    mean_score = np.mean([run['perturbed_score'] for run in runs])
                    std_score = np.std([run['perturbed_score'] for run in runs])
                    worst_scores = [run.get('worst_score', 0) for run in runs]

                    # Store results under the appropriate key
                    results['raw']['by_level'][level_key]['overall_result'][run_key] = {
                        'mean_score': mean_score,
                        'std_score': std_score,
                        'impact': np.mean([run['impact'] for run in runs]),
                        'worst_score': np.mean(worst_scores) if worst_scores else 0
                    }

            # Process results for quantile method
            elif test_type == 'quantile':
                if level_key not in results['quantile']['by_level']:
                    results['quantile']['by_level'][level_key] = {'runs': {}, 'overall_result': {}}

                # Initialize storage for both all_features and feature_subset if they don't exist
                if 'all_features' not in results['quantile']['by_level'][level_key]['runs']:
                    results['quantile']['by_level'][level_key]['runs']['all_features'] = []
                if 'feature_subset' not in results['quantile']['by_level'][level_key]['runs']:
                    results['quantile']['by_level'][level_key]['runs']['feature_subset'] = []

                # Process each eval_result
                for run_key, eval_result in eval_results.items():
                    # Add result to the appropriate category
                    results['quantile']['by_level'][level_key]['runs'][run_key].append(eval_result)
                    all_quantile_impacts.append(eval_result['impact'])
                    all_impacts.append(eval_result['impact'])

                    # Calculate overall result for this level and run_key
                    runs = results['quantile']['by_level'][level_key]['runs'][run_key]
                    mean_score = np.mean([run['perturbed_score'] for run in runs])
                    std_score = np.std([run['perturbed_score'] for run in runs])
                    worst_scores = [run.get('worst_score', 0) for run in runs]

                    # Store results under the appropriate key
                    results['quantile']['by_level'][level_key]['overall_result'][run_key] = {
                        'mean_score': mean_score,
                        'std_score': std_score,
                        'impact': np.mean([run['impact'] for run in runs]),
                        'worst_score': np.mean(worst_scores) if worst_scores else 0
                    }
        
        # Evaluate feature importance using the median level from configurations
        if self.verbose:
            print("\nEvaluating feature importance...")
            
        # Find the median level for raw perturbation
        raw_levels = [test['params'].get('level', 0.1) for test in self.current_config if test['type'] == 'raw']
        if raw_levels:
            median_level = np.median(raw_levels)
            
            try:
                # Evaluate feature importance
                if self.verbose:
                    print(f"Calculating feature importance using raw perturbation at level {median_level}")
                    
                feature_importance = self.evaluator.evaluate_feature_importance(
                    X, y, 'raw', median_level, self.feature_subset
                )
                
                if feature_importance:
                    if self.verbose:
                        print(f"Feature importance calculation successful - found {len(feature_importance)} features")
                        # Print top 5 features by importance
                        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                        print(f"Top 5 features by robustness importance: {top_features}")
                    
                    results['feature_importance'] = feature_importance
                else:
                    print("WARNING: Feature importance calculation returned empty results")
                    # Initialize with empty dictionary to avoid None
                    results['feature_importance'] = {}
            except Exception as e:
                print(f"ERROR calculating feature importance: {str(e)}")
                # Initialize with empty dictionary
                results['feature_importance'] = {}
        
        # Calculate average impacts
        results['avg_raw_impact'] = np.mean(all_raw_impacts) if all_raw_impacts else 0
        results['avg_quantile_impact'] = np.mean(all_quantile_impacts) if all_quantile_impacts else 0
        results['avg_overall_impact'] = np.mean(all_impacts) if all_impacts else 0
        
        # No longer storing visualizations in the results dictionary
        if self.verbose:
            print("\nVisualizations available through the VisualizationManager")
        
        # Calculando tempo apenas para impressão, sem armazenar no dicionário
        if self.verbose:
            execution_time = time.time() - start_time
            print(f"\nTests completed in {execution_time:.2f} seconds")
            print(f"Average raw impact: {results['avg_raw_impact']:.3f}")
            print(f"Average quantile impact: {results['avg_quantile_impact']:.3f}")
            print(f"Overall average impact: {results['avg_overall_impact']:.3f}")
        
        # Store results
        self.results = results
        
        # Add the model_type to the results
        if hasattr(self.dataset, 'model'):
            results['model_type'] = type(self.dataset.model).__name__
        
        return results
    
    def compare(self, alternative_models: Dict[str, Any], X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None) -> Dict[str, Dict[str, Any]]:
        """
        Compare robustness of multiple models using the same configuration.
        
        Parameters:
        -----------
        alternative_models : Dict
            Dictionary mapping model names to model objects
        X : DataFrame, optional
            Feature data to use (if None, will use test data from dataset)
        y : Series, optional
            Target variable (if None, will use test target from dataset)
            
        Returns:
        --------
        Dict[str, Dict[str, Any]] : Dictionary mapping model names to test results
        """
        if not alternative_models:
            raise ValueError("No alternative models provided")
        
        if self.current_config is None:
            # Use default configuration if none specified
            self.config('quick')
            
        if X is None or y is None:
            # Use test data from dataset if not provided
            if hasattr(self.dataset, 'test_data') and self.dataset.test_data is not None:
                X = self.dataset.get_feature_data('test')
                y = self.dataset.get_target_data('test')
            else:
                raise ValueError("No test data available in dataset. Please provide X and y.")
        
        # Run tests for primary model first
        primary_results = self.run(X, y)
        
        # Store results
        all_results = {
            'primary_model': primary_results
        }
        
        # Add the model_type to the primary_model results
        if hasattr(self.dataset, 'model'):
            all_results['primary_model']['model_type'] = type(self.dataset.model).__name__
        
        # Test alternative models
        for model_name, model in alternative_models.items():
            if self.verbose:
                print(f"\nTesting robustness of alternative model: {model_name}")
            
            # Create a temporary dataset with the alternative model
            original_model = self.dataset.model
            self.dataset.model = model
            
            # Run the same tests on this model
            model_results = self.run(X, y)
            
            # Restore original model
            self.dataset.model = original_model
            
            # Store results
            all_results[model_name] = model_results
        
        # Model comparison visualization has been removed
        if self.verbose:
            print("\nModel comparison visualization has been removed in this version.")
        
        # Update stored results
        self.results = all_results['primary_model']
        
        return all_results
    
    def save_report(self, output_path: str = None, model_name: str = "Main Model", format: str = "html") -> str:
        """
        Generate and save a HTML report of robustness test results.
        
        Parameters:
        -----------
        output_path : str, optional
            Path where to save the report. If None, use a timestamped filename in current directory.
        model_name : str, optional
            Name to display in the report
        format : str, optional
            Report format ('html' only for now)
            
        Returns:
        --------
        str : Path to the saved report
        """
        if not self.results:
            raise ValueError("No results to generate report from. Run tests first.")
            
        if format.lower() != 'html':
            raise ValueError("Only HTML report format is supported")
            
        # Create default output path if none specified
        if output_path is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"robustness_report_{timestamp}.html"
            
        # Import report manager here to avoid circular imports
        from deepbridge.core.experiment.report_manager import ReportManager
        
        # Create report manager and generate report
        report_manager = ReportManager()
        try:
            report_path = report_manager.generate_robustness_report(
                self.results, 
                output_path, 
                model_name
            )
            
            if self.verbose:
                print(f"Report saved to: {report_path}")
                
            return report_path
            
        except Exception as e:
            print(f"Error generating report: {str(e)}")
            raise
    
    def get_results(self) -> Dict[str, Any]:
        """Get the test results."""
        return self.results
    
    def get_visualizations(self) -> Dict[str, Any]:
        """
        Get visualizations for the robustness tests.
        
        This method now returns a simple message instructing to use the HTML report
        for visualizations as they are generated dynamically in the report.
        
        Returns:
        --------
        Dict[str, Any]: Empty dict with a message about using the HTML report
        """
        print("Visualizations are now generated in the HTML report.")
        print("Please use save_report() to generate a full interactive HTML report.")
        
        return {
            "message": "Visualizations are available in the HTML report. Use save_report() method."
        }
        
    def update_model_name(self, original_results: Dict, model_type: str) -> Dict:
        """
        Update model name in results dictionary, replacing 'primary_model' with the actual model type.
        
        Parameters:
        -----------
        original_results : Dict
            Original results dictionary to update
        model_type : str
            Actual model type name to replace 'primary_model' with
        
        Returns:
        --------
        Dict : Updated results dictionary
        """
        # Create a copy to avoid modifying the original
        import copy
        results = copy.deepcopy(original_results)
        
        # Replace 'primary_model' with model_type in the top level
        if 'primary_model' in results:
            results[model_type] = results['primary_model']
            del results['primary_model']
        
        # Handle nested dictionaries
        for key, value in results.items():
            if isinstance(value, dict):
                # Update model_name in current level
                if 'model_name' in value and value['model_name'] == 'primary_model':
                    value['model_name'] = model_type
                
                # Process alternative_models if present
                if key == 'alternative_models' and isinstance(value, dict):
                    # Keep alternative_models as is, they should retain their original names
                    continue
                
                # Recursively process deeper nested dictionaries
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, dict):
                        # Update model_name in nested level
                        if 'model_name' in sub_value and sub_value['model_name'] == 'primary_model':
                            sub_value['model_name'] = model_type
        
        return results