import typing as t
import numpy as np
import inspect

# Import dataset factory directly since it doesn't create circular dependencies
from deepbridge.utils.dataset_factory import DBDatasetFactory
from deepbridge.utils.logger import get_logger
from deepbridge.core.experiment.parameter_standards import (
    get_test_config, TestType, ConfigName, is_valid_test_type, is_valid_config_name
)

# Manager imports are moved to appropriate methods to avoid potential circular dependencies

class TestRunner:
    """
    Responsible for running various tests on models.
    Extracted from Experiment class to separate test execution responsibilities.
    """
    # Initialize logger for this class
    logger = get_logger("deepbridge.testrunner")
    
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
        features_select: t.Optional[t.List[str]] = None
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
            features_select: List of feature names to specifically test in the experiments
        """
        self.dataset = dataset
        self.alternative_models = alternative_models
        self.tests = tests
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.verbose = verbose
        self.features_select = features_select
        
        # Set logger verbosity
        self.logger.set_verbose(verbose)
        
        # Log initialization
        self.logger.debug(f"Initializing test runner with tests: {tests}")
        if features_select:
            self.logger.debug(f"Feature subset for testing: {features_select}")
        
        # Store test results
        self.test_results = {}
        
    def run_initial_tests(self, **kwargs) -> dict:
        """
        Simplified version that only calculates basic metrics for original and alternative models,
        and returns experiment configurations.
        
        Args:
            **kwargs : dict
                Additional parameters. Can include:
                - experiment: The parent Experiment object
        
        Returns:
        --------
        dict : Dictionary with model metrics and experiment configurations
        """
        self.logger.info(f"Initializing experiment with tests: {self.tests}")
            
        # Initialize results dictionary
        results = {
            'config': self._get_experiment_config(),
            'models': {}
        }
        
        # Get the experiment object if provided
        experiment = kwargs.get('experiment', None)
        experiment_model = None
        
        # If experiment is provided, try to get the surrogate model
        if experiment is not None and hasattr(experiment, 'distillation_model') and experiment.distillation_model is not None:
            experiment_model = experiment.distillation_model
            self.logger.info("Found surrogate model in provided experiment object for initial tests.")
        else:
            # Fallback to stack inspection method if experiment not provided directly
            try:
                frame = inspect.currentframe()
                # Go up through the stack frames
                while frame:
                    if 'self' in frame.f_locals and isinstance(frame.f_locals['self'], object):
                        parent = frame.f_locals['self']
                        if hasattr(parent, 'distillation_model') and parent.distillation_model is not None:
                            experiment_model = parent.distillation_model
                            self.logger.info("Found surrogate model through stack inspection for initial tests.")
                            break
                    frame = frame.f_back
            except Exception as e:
                self.logger.warning(f"Error during stack inspection: {e}")
                
        # If no model in dataset and no surrogate model in experiment, skip further evaluation
        if (not hasattr(self.dataset, 'model') or self.dataset.model is None) and experiment_model is None:
            self.logger.warning("No model found in dataset or parent experiment.")
            
            # Still include configuration details
            return results
            
        # If we have a surrogate model in the experiment but not in dataset, use it for testing
        if (not hasattr(self.dataset, 'model') or self.dataset.model is None) and experiment_model is not None:
            self.logger.info("Using surrogate model from parent experiment for initial metrics.")
            # Temporarily set the model in the dataset for testing using set_model method
            if hasattr(self.dataset, 'set_model'):
                self.dataset.set_model(experiment_model)
                self.logger.info("Successfully set surrogate model in dataset.")
            else:
                self.logger.warning("Dataset does not have set_model method. Cannot use surrogate model.")
            
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
            self.logger.debug(f"Could not extract feature importances: {str(e)}")
            
        # Calculate metrics for alternative models
        if self.alternative_models:
            for model_name, model in self.alternative_models.items():
                self.logger.debug(f"Calculating metrics for alternative model: {model_name}")
                
                model_metrics = self._calculate_model_metrics(model, model_name)
                results['models'][model_name] = model_metrics
        
        # Add available test configurations
        test_configs = {}
        if "robustness" in self.tests:
            test_configs['robustness'] = self._get_test_config('robustness')
        if "uncertainty" in self.tests:
            test_configs['uncertainty'] = self._get_test_config('uncertainty')
        if "resilience" in self.tests:
            test_configs['resilience'] = self._get_test_config('resilience')
        if "hyperparameters" in self.tests:
            test_configs['hyperparameters'] = self._get_test_config('hyperparameters')
        
        results['test_configs'] = test_configs
            
        # Store results for future reference
        self.test_results.update(results)
        
        # Print initialization completion message
        if 'models' in results and 'primary_model' in results['models']:
            model_type = results['models']['primary_model'].get('type', 'Unknown')
            print(f"✅ Initial model evaluation complete: {model_type}")
        
        return results
        
    def _get_experiment_config(self) -> dict:
        """Get experiment configuration parameters."""
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
    
    def _standardize_metrics(self, metrics: dict) -> None:
        """
        Standardize metrics format and naming conventions.
        Centralizes metric normalization to avoid redundancy.
        
        Args:
            metrics: Dictionary of metrics to standardize
        """
        # Standard metric name is 'roc_auc', convert any 'auc' to 'roc_auc' 
        if 'auc' in metrics:
            # Copy 'auc' value to 'roc_auc' if not already present
            if 'roc_auc' not in metrics:
                metrics['roc_auc'] = float(metrics['auc'])
            # Always remove 'auc' to maintain standardization
            del metrics['auc']
            
        # Ensure all metric values are float type for consistency
        for key, value in metrics.items():
            if value is not None and not isinstance(value, str):
                metrics[key] = float(value)
    
    def _calculate_metrics_for_model(self, model, X, y) -> dict:
        """
        Calculate standard metrics for a model and dataset.
        Centralized helper method to avoid duplicate code.

        Args:
            model: The model to evaluate
            X: Features to use for prediction
            y: Target values for evaluation

        Returns:
            Dictionary of calculated metrics
        """
        from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
        
        # Initialize metrics dictionary
        metrics = {}
        
        # Get predictions
        y_pred = model.predict(X)
        
        # Basic accuracy metric
        metrics['accuracy'] = float(accuracy_score(y, y_pred))
        
        # Calculate ROC AUC if model supports predict_proba
        if hasattr(model, 'predict_proba'):
            try:
                y_prob = model.predict_proba(X)
                if y_prob.shape[1] > 1:  # For binary classification
                    metrics['roc_auc'] = float(roc_auc_score(y, y_prob[:, 1]))
            except Exception:
                # Skip if not applicable
                pass
        
        # Classification metrics
        try:
            metrics['f1'] = float(f1_score(y, y_pred, average='weighted'))
            metrics['precision'] = float(precision_score(y, y_pred, average='weighted'))
            metrics['recall'] = float(recall_score(y, y_pred, average='weighted'))
        except Exception:
            # Skip if not applicable
            pass
            
        return metrics

    def _calculate_model_metrics(self, model, model_name: str) -> dict:
        """Calculate basic metrics for a model."""
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
            # Use the centralized metrics calculation method
            metrics = self._calculate_metrics_for_model(model, self.X_test, self.y_test)
            
            # Standardize metrics names and formats
            self._standardize_metrics(metrics)
            
            model_info['metrics'] = metrics
            
        except Exception as e:
            model_info['metrics'] = {'error': str(e)}
            
        return model_info
    
    def _get_model_hyperparameters(self, model) -> dict:
        """Extract hyperparameters from a model."""
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
            return {'error': str(e)}
    
    def _get_test_config(self, test_type: str) -> dict:
        """
        Get configuration options for a specific test type.

        Args:
            test_type: Type of test to get configurations for

        Returns:
            Dictionary containing configuration options for different levels ('quick', 'medium', 'full')
        """
        # Validate test type
        if not is_valid_test_type(test_type):
            self.logger.warning(f"Invalid test type: {test_type}. Using fallback configurations.")
            return {
                'quick': {},
                'medium': {},
                'full': {}
            }

        try:
            # Use the centralized configuration from parameter_standards.py
            quick_config = get_test_config(test_type, ConfigName.QUICK.value)
            medium_config = get_test_config(test_type, ConfigName.MEDIUM.value)
            full_config = get_test_config(test_type, ConfigName.FULL.value)

            return {
                'quick': quick_config,
                'medium': medium_config,
                'full': full_config
            }
        except Exception as e:
            self.logger.error(f"Error getting test config: {str(e)}")
            # Return empty configurations as fallback
            return {
                'quick': {},
                'medium': {},
                'full': {}
            }
        
    def _get_test_arguments(self, test_type: str, config_name: str, model_name: str, metric_param: str = None) -> dict:
        """
        Get the appropriate arguments for a specific test function based on test type.
        Different test functions have different parameter signatures.
        
        Args:
            test_type: Type of test being run
            config_name: Configuration level
            model_name: Name of the model being tested
            metric_param: Optional metric parameter name
            
        Returns:
            Dictionary of arguments appropriate for the test function
        """
        # Base arguments for all tests
        args = {
            'config_name': config_name,
            'verbose': self.verbose,
            'feature_subset': self.features_select,
        }
        
        # Test-specific arguments
        if test_type == 'uncertainty':
            # Uncertainty tests don't use model_name or metric parameters
            pass
        elif test_type == 'robustness':
            # Robustness tests use model_name but not metric parameter
            args['model_name'] = model_name
            # The 'metric' parameter is now removed, as it was redundant with 'metrics'
        elif test_type == 'resilience':
            # Resilience tests use metric but not model_name
            if metric_param:
                args[metric_param.lower()] = metric_param
        elif test_type == 'hyperparameters':
            # Hyperparameter tests use metric but not model_name
            if metric_param:
                args[metric_param.lower()] = metric_param
        
        return args

    def _run_model_test(self, test_type: str, config_name: str, run_test_fn, metric_param: str = 'AUC') -> dict:
        """
        Run a specific type of test for all models.
        
        Args:
            test_type: Type of test to run ('robustness', 'uncertainty', etc.)
            config_name: Configuration level ('quick', 'medium', 'full')
            run_test_fn: Function to run the test
            metric_param: Name of metric parameter to pass to the test function
            
        Returns:
            Dictionary with test results for all models
        """
        # Initialize results dictionary
        test_results = {
            'primary_model': {},
            'alternative_models': {}
        }
        
        # Test primary model
        self.logger.info(f"Testing {test_type} of primary model...")
        
        # Get appropriate arguments for this test type for primary model
        test_args = self._get_test_arguments(test_type, config_name, 'primary_model', metric_param)
        
        # Run the test
        primary_results = run_test_fn(self.dataset, **test_args)
        
        # Apply metrics from initial results if available for primary model
        if 'models' in self.test_results and 'primary_model' in self.test_results['models']:
            primary_metrics = self.test_results['models']['primary_model'].get('metrics', {})
            if primary_metrics and isinstance(primary_metrics, dict):
                # Create a deep copy of the metrics
                metrics_copy = {}
                for key, value in primary_metrics.items():
                    metrics_copy[key] = value
                
                # Ensure metrics exist in results
                if 'metrics' not in primary_results:
                    primary_results['metrics'] = {}
                
                # Replace with real metrics
                primary_results['metrics'] = metrics_copy
                
                # Standardize metrics names
                self._standardize_metrics(primary_results['metrics'])
        
        # Store primary model results
        # Check if primary_results already has the structure with primary_model
        if 'primary_model' in primary_results:
            # Use the primary_model from the results directly
            test_results['primary_model'] = primary_results['primary_model']
            # Add other top-level keys that might be important
            for key in primary_results:
                if key not in ['primary_model', 'alternative_models']:
                    test_results[key] = primary_results[key]
        else:
            # If no nested structure, use as is
            test_results['primary_model'] = primary_results

        # Add model_type to primary model results
        if hasattr(self.dataset, 'model'):
            test_results['primary_model']['model_type'] = type(self.dataset.model).__name__
        
        # Handle alternative_models if they exist in primary_results
        if 'alternative_models' in primary_results:
            test_results['alternative_models'] = primary_results['alternative_models']

        # Test additional alternative models
        if self.alternative_models:
            for model_name, model in self.alternative_models.items():
                self.logger.info(f"Testing {test_type} of alternative model: {model_name}")
                
                # Create dataset with the alternative model
                alt_dataset = self._create_alternative_dataset(model)
                
                # Get appropriate arguments for alternative model
                test_args = self._get_test_arguments(test_type, config_name, model_name, metric_param)
                
                # Run test for alternative model
                alt_results = run_test_fn(alt_dataset, **test_args)
                
                # Apply metrics from initial results if available
                if 'models' in self.test_results and model_name in self.test_results['models']:
                    model_metrics = self.test_results['models'][model_name].get('metrics', {})
                    if model_metrics and isinstance(model_metrics, dict):
                        # Ensure metrics exist in results
                        if 'metrics' not in alt_results:
                            alt_results['metrics'] = {}
                        
                        # Replace with real metrics
                        alt_results['metrics'] = model_metrics.copy()
                
                # Store results for this alternative model
                test_results['alternative_models'][model_name] = alt_results
                
                # Add model_type to alternative model results
                test_results['alternative_models'][model_name]['model_type'] = type(model).__name__
        
        return test_results
    
    def run_tests(self, config_name: str = 'quick', **kwargs) -> dict:
        """
        Run all tests specified during initialization with the given configuration.
        
        Parameters:
        -----------
        config_name : str
            Name of the configuration to use: 'quick', 'medium', or 'full'
        **kwargs : dict
            Additional parameters for tests. Can include:
            - experiment: The parent Experiment object
            
        Returns:
        --------
        dict : Dictionary with test results
        """
        self.logger.info(f"Running tests with {config_name} configuration...")
            
        # Get the experiment object if provided
        experiment = kwargs.get('experiment', None)
        experiment_model = None
        
        # If experiment is provided, try to get the surrogate model
        if experiment is not None and hasattr(experiment, 'distillation_model') and experiment.distillation_model is not None:
            experiment_model = experiment.distillation_model
            self.logger.info("Found surrogate model in provided experiment object.")
        else:
            # Fallback to stack inspection method if experiment not provided directly
            try:
                frame = inspect.currentframe()
                # Go up through the stack frames
                while frame:
                    if 'self' in frame.f_locals and isinstance(frame.f_locals['self'], object):
                        parent = frame.f_locals['self']
                        if hasattr(parent, 'distillation_model') and parent.distillation_model is not None:
                            experiment_model = parent.distillation_model
                            self.logger.info("Found surrogate model through stack inspection.")
                            break
                    frame = frame.f_back
            except Exception as e:
                self.logger.warning(f"Error during stack inspection: {e}")
                
        # If no model in dataset and no surrogate model in experiment, skip tests
        if (not hasattr(self.dataset, 'model') or self.dataset.model is None) and experiment_model is None:
            self.logger.warning("No model found in dataset or parent experiment. Skipping tests.")
            return {}
        
        # If we have a surrogate model in the experiment but not in dataset, use it for testing
        if (not hasattr(self.dataset, 'model') or self.dataset.model is None) and experiment_model is not None:
            self.logger.info("Using surrogate model from parent experiment for tests.")
            # Temporarily set the model in the dataset for testing using set_model method
            if hasattr(self.dataset, 'set_model'):
                self.dataset.set_model(experiment_model)
                self.logger.info("Successfully set surrogate model in dataset.")
            else:
                self.logger.warning("Dataset does not have set_model method. Cannot use surrogate model.")
            
        # Make sure we have run initial tests first to get base metrics
        if not hasattr(self, 'test_results') or not self.test_results or not 'models' in self.test_results:
            self.run_initial_tests()
            
        # Initialize results dictionary
        results = {}
        
        # Run robustness tests if requested
        if "robustness" in self.tests:
            from deepbridge.utils.robustness import run_robustness_tests
            self.logger.info("Starting robustness tests...")
            results['robustness'] = self._run_model_test(
                test_type='robustness',  # Accepts model_name but not metric parameter
                config_name=config_name,
                run_test_fn=run_robustness_tests,
                metric_param=None
            )
            print("✅ Robustness Tests Finished!")
            
        # Run uncertainty tests if requested
        if "uncertainty" in self.tests:
            from deepbridge.utils.uncertainty import run_uncertainty_tests
            self.logger.info("Starting uncertainty tests...")
            results['uncertainty'] = self._run_model_test(
                test_type='uncertainty',  # Doesn't accept model_name or metric
                config_name=config_name,
                run_test_fn=run_uncertainty_tests,
                metric_param=None
            )
            print("✅ Uncertainty Tests Finished!")
            
        # Run resilience tests if requested
        if "resilience" in self.tests:
            from deepbridge.utils.resilience import run_resilience_tests
            self.logger.info("Starting resilience tests...")
            results['resilience'] = self._run_model_test(
                test_type='resilience',  # Accepts metric but not model_name
                config_name=config_name,
                run_test_fn=run_resilience_tests,
                metric_param='metric'  # We keep this one since it's not redundant
            )
            print("✅ Resilience Tests Finished!")
            
        # Run hyperparameter tests if requested
        if "hyperparameters" in self.tests:
            from deepbridge.utils.hyperparameter import run_hyperparameter_tests
            self.logger.info("Starting hyperparameter tests...")
            results['hyperparameters'] = self._run_model_test(
                test_type='hyperparameters',  # Accepts metric but not model_name
                config_name=config_name,
                run_test_fn=run_hyperparameter_tests,
                metric_param='metric'
            )
            print("✅ Hyperparameter Tests Finished!")

        # Run fairness tests if requested
        if "fairness" in self.tests:
            from deepbridge.validation.wrappers.fairness_suite import FairnessSuite
            self.logger.info("Starting fairness tests...")

            # Get protected attributes from kwargs
            protected_attributes = kwargs.get('sensitive_features', kwargs.get('protected_attributes', []))

            if not protected_attributes:
                self.logger.warning("No protected attributes provided for fairness test. Skipping.")
            else:
                try:
                    # Initialize fairness suite
                    fairness = FairnessSuite(
                        dataset=self.dataset,
                        protected_attributes=protected_attributes,
                        verbose=False
                    )

                    # Run fairness test with config
                    fairness_config = kwargs.get('config', config_name)
                    fairness_result = fairness.config(fairness_config).run()

                    results['fairness'] = fairness_result
                    print("✅ Fairness Tests Finished!")
                except Exception as e:
                    self.logger.error(f"Error running fairness tests: {e}")
                    results['fairness'] = {}

        # Store results in the object for future reference
        self.test_results.update(results)
        
        # Print completion message for all tests
        if results:
            completed_tests = list(results.keys())
            if len(completed_tests) == 1:
                print(f"🎉 Test completed successfully: {completed_tests[0]}")
            else:
                print(f"🎉 All {len(completed_tests)} tests completed successfully!")
        
        return results

    def run_test(self, test_type: str, config_name: str = 'quick', **kwargs):
        """
        Run a single specific test with the given configuration.

        Parameters:
        -----------
        test_type : str
            Type of test to run (robustness, uncertainty, resilience, hyperparameters, fairness)
        config_name : str
            Name of the configuration to use: 'quick', 'medium', or 'full'
        **kwargs : dict
            Additional parameters for the test

        Returns:
        --------
        TestResult or dict : Test result object or dictionary with results
        """
        # Validate test type
        valid_tests = ["robustness", "uncertainty", "resilience", "hyperparameters", "fairness"]
        if test_type not in valid_tests:
            raise ValueError(f"Invalid test type '{test_type}'. Valid types: {valid_tests}")

        # Temporarily override the tests list to run only this test
        original_tests = self.tests
        self.tests = [test_type]

        try:
            # Run the single test
            results = self.run_tests(config_name=config_name, **kwargs)

            # Return the result for the specific test
            if test_type in results:
                return results[test_type]
            else:
                # Return the full results if test_type not found as key
                return results

        finally:
            # Restore original tests list
            self.tests = original_tests

    def _create_alternative_dataset(self, model):
        """
        Helper method to create a dataset with an alternative model.
        Uses DBDatasetFactory to ensure consistent dataset creation.
        """
        return DBDatasetFactory.create_for_alternative_model(
            original_dataset=self.dataset,
            model=model
        )

    def get_test_results(self, test_type: str = None):
        """
        Get test results for a specific test type or all results.
        
        Args:
            test_type: The type of test to get results for. If None, returns all results.
            
        Returns:
            dict: Dictionary with test results
        """
        if test_type:
            return self.test_results.get(test_type)
        return self.test_results