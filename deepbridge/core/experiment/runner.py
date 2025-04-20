"""
Enhanced test runner with better separation of concerns.
This module provides a standardized way to run different types of tests.
"""

import typing as t
from abc import ABC, abstractmethod
import numpy as np

from deepbridge.core.experiment.interfaces import ITestRunner, TestResult
from deepbridge.core.experiment.parameter_standards import TestConfigDict, TestResultsDict
try:
    # Try to use the factory first
    from deepbridge.core.experiment.test_result_factory import TestResultFactory
    create_test_result = TestResultFactory.create_test_result
except ImportError:
    # Fall back to the direct function in results.py
    from deepbridge.core.experiment.results import create_test_result
from deepbridge.utils.dataset_factory import DBDatasetFactory

class TestRunner(ITestRunner):
    """
    An enhanced test runner that implements the ITestRunner interface and provides
    better separation of concerns compared to the original implementation.
    """
    
    def __init__(
        self,
        dataset: 'DBDataset',
        alternative_models: dict,
        tests: t.List[str],
        X_train,
        X_test,
        y_train,
        y_test,
        verbose: bool = False,
        feature_subset: t.Optional[t.List[str]] = None,  # Consistently use feature_subset instead of features_select
        test_managers: t.Optional[dict] = None
    ):
        """
        Initialize the test runner with dataset and model information.
        
        Args:
            dataset: The DBDataset containing model and data
            alternative_models: Dictionary of alternative models
            tests: List of tests to run
            X_train: Training features
            X_test: Testing features
            y_train: Training target
            y_test: Testing target
            verbose: Whether to print verbose output
            feature_subset: List of feature names to specifically test in the experiments
            test_managers: Dictionary of test manager instances keyed by test type
        """
        # Set attributes using standardized names
        self.dataset = dataset
        self.alternative_models = alternative_models
        self.tests = tests
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.verbose = verbose
        self.feature_subset = feature_subset  # Standardized name
        
        # Store test results
        self.test_results: TestResultsDict = {}
        
        # Initialize test managers (lazy loaded if not provided)
        self._test_managers: t.Dict[str, t.Any] = test_managers or {}
        
    def run_tests(self, config_name: str = 'quick', **kwargs) -> TestResultsDict:
        """
        Run all tests specified during initialization with the given configuration.
        
        Args:
            config_name: Name of the configuration to use: 'quick', 'medium', or 'full'
            **kwargs: Additional configuration parameters
            
        Returns:
            dict: Dictionary with test results
        """
        if self.verbose:
            print(f"Running tests with {config_name} configuration...")
            
        # Check if we have a model to test
        if not hasattr(self.dataset, 'model') or self.dataset.model is None:
            if self.verbose:
                print("No model found in dataset. Skipping tests.")
            return {}
        
        # Initialize results dictionary
        results = {}
        
        # Run each type of test
        for test_type in self.tests:
            if self.verbose:
                print(f"Running {test_type} tests...")
            
            try:
                # Run the test
                test_result = self.run_test(test_type, config_name, **kwargs)
                
                # Add to results dictionary
                results[test_type] = test_result.results
            except Exception as e:
                if self.verbose:
                    print(f"Error running {test_type} tests: {str(e)}")
                results[test_type] = {'error': str(e)}
        
        # Store results in the object for future reference
        self.test_results.update(results)
        
        return results
    
    def run_test(self, test_type: str, config_name: str = 'quick', **kwargs) -> TestResult:
        """
        Run a specific test with the given configuration.
        
        Args:
            test_type: Type of test to run
            config_name: Configuration profile to use
            **kwargs: Additional configuration parameters
            
        Returns:
            TestResult: Result object for the test
        """
        if test_type not in self.tests:
            raise ValueError(f"Test type '{test_type}' not in configured tests: {self.tests}")
        
        # Get the appropriate manager for this test type
        manager = self._get_manager(test_type)
        
        # Run primary model test
        if self.verbose:
            print(f"Testing {test_type} of primary model...")
        
        # Get default configuration and update with kwargs
        config = self.get_test_config(test_type, config_name)
        config.update(kwargs)
        
        # Run the test
        result = self._run_standard_test(test_type, config)
        
        # Test alternative models if available
        alt_results = {}
        if self.alternative_models:
            for model_name, model in self.alternative_models.items():
                if self.verbose:
                    print(f"Testing {test_type} of alternative model: {model_name}")
                
                # Create a dataset with the alternative model
                alt_dataset = self._create_alternative_dataset(model)
                
                # Run the test on the alternative model
                alt_result = self._run_model_test(test_type, alt_dataset, config)
                
                # Store results
                alt_results[model_name] = alt_result
        
        # Combine results
        combined_results = {
            'primary_model': result,
            'alternative_models': alt_results
        }
        
        # Create test result object
        test_result = create_test_result(test_type, combined_results)
        
        # Update stored results
        self.test_results[test_type] = combined_results
        
        return test_result
    
    def get_test_results(self, test_type: t.Optional[str] = None) -> t.Union[TestResultsDict, TestResult]:
        """
        Get results for a specific test or all results.
        
        Args:
            test_type: Type of test to get results for. If None, returns all results.
            
        Returns:
            Union[dict, TestResult]: Test results
        """
        if test_type is None:
            return self.test_results
            
        results = self.test_results.get(test_type)
        
        if results is None:
            return None
            
        return create_test_result(test_type, results)
    
    def _get_manager(self, test_type: str):
        """
        Get the appropriate test manager for a test type.
        Uses the ManagerFactory if available, otherwise falls back to local creation.
        
        Args:
            test_type: Type of test to get a manager for
            
        Returns:
            Manager instance for the specified test type
        """
        # Check if we already have a manager instance
        if test_type in self._test_managers:
            return self._test_managers[test_type]
            
        try:
            # Try to use the ManagerFactory
            from deepbridge.core.experiment.manager_factory import ManagerFactory
            
            # Get a manager instance from the factory
            manager = ManagerFactory.get_manager(
                test_type=test_type,
                dataset=self.dataset,
                alternative_models=self.alternative_models,
                verbose=self.verbose
            )
            
            # Store the manager instance
            self._test_managers[test_type] = manager
            
            return manager
            
        except ImportError:
            # Fall back to local creation
            self._test_managers[test_type] = self._create_manager(test_type)
            return self._test_managers[test_type]
    
    def _create_manager(self, test_type: str):
        """
        Create a manager instance for a test type.
        Uses a factory-like approach to create the appropriate manager.
        
        Args:
            test_type: Type of test to create a manager for
            
        Returns:
            Manager instance for the specified test type
            
        Raises:
            ValueError: If the test_type is not supported
        """
        # Import manager classes
        from deepbridge.core.experiment.managers import (
            RobustnessManager, UncertaintyManager, ResilienceManager, HyperparameterManager
        )
        
        # Create a manager registry (factory pattern)
        manager_registry = {
            'robustness': RobustnessManager,
            'uncertainty': UncertaintyManager,
            'resilience': ResilienceManager,
            'hyperparameters': HyperparameterManager
        }
        
        # Check if the test type is supported
        if test_type not in manager_registry:
            raise ValueError(f"Unsupported test type: {test_type}")
        
        # Create and return the manager instance
        manager_class = manager_registry[test_type]
        return manager_class(self.dataset, self.alternative_models, self.verbose)
    
    def _run_standard_test(self, test_type: str, config: dict):
        """
        Run a standard test using the appropriate strategy.
        Uses the Strategy pattern to encapsulate test logic.
        
        Args:
            test_type: Type of test to run
            config: Test configuration parameters
            
        Returns:
            dict: Test results
            
        Raises:
            ValueError: If the test_type is not supported
        """
        try:
            # Try to use the strategy factory
            from deepbridge.core.experiment.test_strategies import TestStrategyFactory
            
            # Get the appropriate strategy for this test type
            strategy = TestStrategyFactory.create_strategy(test_type)
            
            # Run the test using the strategy
            return strategy.run_test(
                dataset=self.dataset,
                config=config,
                feature_subset=self.feature_subset,
                verbose=self.verbose
            )
            
        except ImportError:
            # Fall back to direct implementation
            if test_type == 'robustness':
                from deepbridge.utils.robustness import run_robustness_tests
                
                # Initialize the results dictionary
                results = {}
                
                # Run test with feature_subset (if any) 
                # The robustness tests now internally handle both all features and feature subset cases
                results = run_robustness_tests(
                    self.dataset, 
                    config_name=config.get('config_name', 'quick'),
                    metric=config.get('metric', 'auc'),
                    verbose=self.verbose,
                    feature_subset=self.feature_subset  # Pass feature_subset directly (None or specific features)
                )
                
                return results
            
            elif test_type == 'uncertainty':
                from deepbridge.utils.uncertainty import run_uncertainty_tests
                
                # Initialize the results dictionary
                results = {}
                
                # Run test with all features first
                all_features_result = run_uncertainty_tests(
                    self.dataset, 
                    config_name=config.get('config_name', 'quick'),
                    verbose=self.verbose,
                    feature_subset=None  # Explicitly set to None to use all features
                )
                
                # Initialize results as a copy of the result with all features
                results = all_features_result
                
                # If feature_subset specified, run a second test and store separately
                if self.feature_subset:
                    # Run test with feature subset
                    feature_subset_result = run_uncertainty_tests(
                        self.dataset, 
                        config_name=config.get('config_name', 'quick'),
                        verbose=self.verbose,
                        feature_subset=self.feature_subset
                    )
                    
                    # Add the subset results at the top level with a distinct key
                    results['feature_subset_results'] = feature_subset_result
                
                return results
                
            elif test_type == 'resilience':
                from deepbridge.utils.resilience import run_resilience_tests
                
                # Initialize the results dictionary
                results = {}
                
                # Run test with all features first
                all_features_result = run_resilience_tests(
                    self.dataset, 
                    config_name=config.get('config_name', 'quick'),
                    metric=config.get('metric', 'auc'),
                    verbose=self.verbose,
                    feature_subset=None  # Explicitly set to None to use all features
                )
                
                # Initialize results as a copy of the result with all features
                results = all_features_result
                
                # If feature_subset specified, run a second test and store separately
                if self.feature_subset:
                    # Run test with feature subset
                    feature_subset_result = run_resilience_tests(
                        self.dataset, 
                        config_name=config.get('config_name', 'quick'),
                        metric=config.get('metric', 'auc'),
                        verbose=self.verbose,
                        feature_subset=self.feature_subset
                    )
                    
                    # Add the subset results at the top level with a distinct key
                    results['feature_subset_results'] = feature_subset_result
                
                return results
                
            elif test_type == 'hyperparameters':
                from deepbridge.utils.hyperparameter import run_hyperparameter_tests
                
                # Initialize the results dictionary
                results = {}
                
                # Run test with all features first
                all_features_result = run_hyperparameter_tests(
                    self.dataset, 
                    config_name=config.get('config_name', 'quick'),
                    metric=config.get('metric', 'accuracy'),
                    verbose=self.verbose,
                    feature_subset=None  # Explicitly set to None to use all features
                )
                
                # Initialize results as a copy of the result with all features
                results = all_features_result
                
                # If feature_subset specified, run a second test and store separately
                if self.feature_subset:
                    # Run test with feature subset
                    feature_subset_result = run_hyperparameter_tests(
                        self.dataset, 
                        config_name=config.get('config_name', 'quick'),
                        metric=config.get('metric', 'accuracy'),
                        verbose=self.verbose,
                        feature_subset=self.feature_subset
                    )
                    
                    # Add the subset results at the top level with a distinct key
                    results['feature_subset_results'] = feature_subset_result
                
                return results
                
            else:
                raise ValueError(f"Unsupported test type: {test_type}")
    
    def _run_model_test(self, test_type: str, model_dataset, config: dict):
        """Run a test on a specific model dataset"""
        # The test functions expect a dataset object
        return self._run_standard_test(test_type, config)
    
    def _create_alternative_dataset(self, model):
        """
        Helper method to create a dataset with an alternative model.
        Uses DBDatasetFactory to ensure consistent dataset creation.
        """
        return DBDatasetFactory.create_for_alternative_model(
            original_dataset=self.dataset,
            model=model
        )
    
    def get_test_config(self, test_type: str, config_name: str = 'quick') -> TestConfigDict:
        """
        Get configuration options for a specific test type.
        Uses the TestStrategyFactory to get configurations.
        
        Args:
            test_type: Type of test
            config_name: Configuration level ('quick', 'medium', 'full')
            
        Returns:
            dict: Test configuration parameters
        """
        try:
            # Try to use the factory
            from deepbridge.core.experiment.test_strategies import TestStrategyFactory
            return TestStrategyFactory.get_configuration(test_type, config_name)
        except ImportError:
            # Fall back to local implementation
            config_options = {}
            
            if test_type == 'robustness':
                if config_name == 'quick':
                    config_options = {
                        'perturbation_methods': ['raw', 'quantile'],
                        'levels': [0.1, 0.2],
                        'n_trials': 5,
                        'config_name': config_name
                    }
                elif config_name == 'medium':
                    config_options = {
                        'perturbation_methods': ['raw', 'quantile', 'adversarial'],
                        'levels': [0.05, 0.1, 0.2],
                        'n_trials': 10,
                        'config_name': config_name
                    }
                elif config_name == 'full':
                    config_options = {
                        'perturbation_methods': ['raw', 'quantile', 'adversarial', 'custom'],
                        'levels': [0.01, 0.05, 0.1, 0.2, 0.3],
                        'n_trials': 20,
                        'config_name': config_name
                    }
            
            elif test_type == 'uncertainty':
                if config_name == 'quick':
                    config_options = {
                        'methods': ['crqr'],
                        'alpha_levels': [0.1, 0.2],
                        'config_name': config_name
                    }
                elif config_name == 'medium':
                    config_options = {
                        'methods': ['crqr'],
                        'alpha_levels': [0.05, 0.1, 0.2],
                        'config_name': config_name
                    }
                elif config_name == 'full':
                    config_options = {
                        'methods': ['crqr'],
                        'alpha_levels': [0.01, 0.05, 0.1, 0.2, 0.3],
                        'config_name': config_name
                    }
            
            elif test_type == 'resilience':
                if config_name == 'quick':
                    config_options = {
                        'drift_types': ['covariate', 'label'],
                        'drift_intensities': [0.1, 0.2],
                        'config_name': config_name
                    }
                elif config_name == 'medium':
                    config_options = {
                        'drift_types': ['covariate', 'label', 'concept'],
                        'drift_intensities': [0.05, 0.1, 0.2],
                        'config_name': config_name
                    }
                elif config_name == 'full':
                    config_options = {
                        'drift_types': ['covariate', 'label', 'concept', 'temporal'],
                        'drift_intensities': [0.01, 0.05, 0.1, 0.2, 0.3],
                        'config_name': config_name
                    }
            
            elif test_type == 'hyperparameters':
                if config_name == 'quick':
                    config_options = {
                        'n_trials': 10,
                        'optimization_metric': 'accuracy',
                        'config_name': config_name
                    }
                elif config_name == 'medium':
                    config_options = {
                        'n_trials': 30,
                        'optimization_metric': 'accuracy',
                        'config_name': config_name
                    }
                elif config_name == 'full':
                    config_options = {
                        'n_trials': 100,
                        'optimization_metric': 'accuracy',
                        'config_name': config_name
                    }
            
            return config_options
    
    def run_initial_tests(self) -> dict:
        """
        Get basic metrics and configuration for the experiment.
        
        Returns:
            dict: Dictionary with experiment configuration and model metrics
        """
        if self.verbose:
            print(f"Initializing experiment with tests: {self.tests}")
            
        # Initialize results dictionary
        results = {
            'config': self._get_experiment_config(),
            'models': {}
        }
        
        # Check if we have models to evaluate
        if not hasattr(self.dataset, 'model') or self.dataset.model is None:
            if self.verbose:
                print("No model found in dataset.")
            
            # Still include configuration details
            return results
            
        # Calculate metrics for primary model
        primary_metrics = self._calculate_model_metrics(self.dataset.model, "primary_model")
        results['models']['primary_model'] = primary_metrics
        
        # Get model feature importance if available for primary model
        try:
            if hasattr(self.dataset.model, 'feature_importances_') and hasattr(self.dataset, 'features'):
                features = self.dataset.features
                importances = self.dataset.model.feature_importances_
                if len(importances) == len(features):
                    total = sum(importances)
                    if total > 0:
                        results['models']['primary_model']['feature_importance'] = {
                            features[i]: float(importances[i]) / total for i in range(len(features))
                        }
            elif hasattr(self.dataset.model, 'coef_') and hasattr(self.dataset, 'features'):
                features = self.dataset.features
                if hasattr(self.dataset.model.coef_, 'shape') and len(self.dataset.model.coef_.shape) > 1:
                    # For multi-class models, take the average of coefficients
                    importances = np.abs(self.dataset.model.coef_).mean(axis=0)
                else:
                    importances = np.abs(self.dataset.model.coef_)
                    
                if len(importances) == len(features):
                    total = sum(importances)
                    if total > 0:
                        results['models']['primary_model']['feature_importance'] = {
                            features[i]: float(importances[i]) / total for i in range(len(features))
                        }
        except Exception as e:
            if self.verbose:
                print(f"Could not extract feature importances: {str(e)}")
            
        # Calculate metrics for alternative models
        if self.alternative_models:
            for model_name, model in self.alternative_models.items():
                if self.verbose:
                    print(f"Calculating metrics for alternative model: {model_name}")
                
                model_metrics = self._calculate_model_metrics(model, model_name)
                results['models'][model_name] = model_metrics
        
        # Add available test configurations
        test_configs = {}
        for test_type in self.tests:
            test_configs[test_type] = {
                'quick': self.get_test_config(test_type, 'quick'),
                'medium': self.get_test_config(test_type, 'medium'),
                'full': self.get_test_config(test_type, 'full')
            }
        
        results['test_configs'] = test_configs
            
        # Store results for future reference
        self.test_results.update(results)
        
        return results
    
    def _get_experiment_config(self) -> dict:
        """Get experiment configuration parameters"""
        config = {
            'tests': self.tests,
            'verbose': self.verbose,
            'dataset_info': {
                'n_samples': len(self.X_train) + len(self.X_test),
                'n_features': self.X_train.shape[1] if hasattr(self.X_train, 'shape') else 'unknown',
                'test_size': len(self.X_test) / (len(self.X_train) + len(self.X_test)) if len(self.X_train) + len(self.X_test) > 0 else 0,
            }
        }
        
        # Add any additional dataset configurations if available
        if hasattr(self.dataset, 'config') and self.dataset.config:
            config['dataset_config'] = self.dataset.config
            
        return config
    
    def _calculate_model_metrics(self, model, model_name: str) -> dict:
        """Calculate basic metrics for a model"""
        from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
        
        model_info = {
            'name': model_name,
            'type': type(model).__name__,
            'metrics': {},
            'hyperparameters': self._get_model_hyperparameters(model)
        }
        
        # Skip metrics calculation if no data is available
        if self.X_test is None or self.y_test is None:
            return model_info
            
        try:
            # Try to get predictions
            y_pred = model.predict(self.X_test)
            
            # Calculate basic metrics
            metrics = {
                'accuracy': accuracy_score(self.y_test, y_pred)
            }
            
            # Try to calculate ROC AUC if it's a classification problem
            try:
                if hasattr(model, 'predict_proba'):
                    y_prob = model.predict_proba(self.X_test)
                    if y_prob.shape[1] > 1:  # Multi-class
                        metrics['roc_auc'] = roc_auc_score(self.y_test, y_prob, multi_class='ovr')
                    else:  # Binary
                        metrics['roc_auc'] = roc_auc_score(self.y_test, y_prob[:, 1])
            except:
                # Skip ROC AUC if not applicable
                pass
                
            # Try to calculate F1, precision, and recall for classification
            try:
                metrics['f1'] = f1_score(self.y_test, y_pred, average='weighted')
                metrics['precision'] = precision_score(self.y_test, y_pred, average='weighted')
                metrics['recall'] = recall_score(self.y_test, y_pred, average='weighted')
            except:
                # Skip if not applicable
                pass
                
            model_info['metrics'] = metrics
            
        except Exception as e:
            if self.verbose:
                print(f"Error calculating metrics for {model_name}: {str(e)}")
            model_info['metrics'] = {'error': str(e)}
            
        return model_info
    
    def _get_model_hyperparameters(self, model) -> dict:
        """Extract hyperparameters from a model"""
        try:
            # For scikit-learn models
            if hasattr(model, 'get_params'):
                return model.get_params()
                
            # For other model types
            elif hasattr(model, '__dict__'):
                # Filter out private attributes and callable methods
                hyperparams = {}
                for key, value in model.__dict__.items():
                    if not key.startswith('_') and not callable(value):
                        hyperparams[key] = str(value)
                return hyperparams
                
            return {}
            
        except Exception as e:
            if self.verbose:
                print(f"Error extracting hyperparameters: {str(e)}")
            return {'error': str(e)}