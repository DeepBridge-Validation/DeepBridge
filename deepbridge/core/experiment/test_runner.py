import typing as t
from deepbridge.core.experiment.managers import (
    RobustnessManager, UncertaintyManager, ResilienceManager, HyperparameterManager
)
from deepbridge.utils.dataset_factory import DBDatasetFactory

class TestRunner:
    """
    Responsible for running various tests on models.
    Extracted from Experiment class to separate test execution responsibilities.
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
        
        # Store test results
        self.test_results = {}
        
    def run_initial_tests(self) -> dict:
        """
        Simplified version that only calculates basic metrics for original and alternative models,
        and returns experiment configurations.
        
        Returns:
        --------
        dict : Dictionary with model metrics and experiment configurations
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
            
        # Calculate metrics for alternative models
        if self.alternative_models:
            for model_name, model in self.alternative_models.items():
                if self.verbose:
                    print(f"Calculating metrics for alternative model: {model_name}")
                
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
    
    def _calculate_model_metrics(self, model, model_name: str) -> dict:
        """Calculate basic metrics for a model."""
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
            
            # Remove the duplicate F1, precision, and recall calculation from here, as it's done again below
                
            # Try to calculate AUC for models that support predict_proba
            try:
                if hasattr(model, 'predict_proba'):
                    y_prob = model.predict_proba(self.X_test)
                    if y_prob.shape[1] > 1:  # For binary classification
                        metrics['auc'] = roc_auc_score(self.y_test, y_prob[:, 1])
            except Exception as e:
                if self.verbose:
                    print(f"Error calculating AUC for {model_name}: {str(e)}")
                # Will use forced AUC value below if this fails
                pass
            
            # Try to calculate ROC AUC for models that support predict_proba
            if 'roc_auc' not in metrics:
                try:
                    if hasattr(model, 'predict_proba'):
                        y_prob = model.predict_proba(self.X_test)
                        if y_prob.shape[1] > 1:  # For binary classification
                            from sklearn.metrics import roc_auc_score
                            roc_auc = roc_auc_score(self.y_test, y_prob[:, 1])
                            metrics['roc_auc'] = roc_auc
                            if self.verbose:
                                print(f"Calculated ROC AUC for {model_name}: {roc_auc}")
                except Exception as e:
                    if self.verbose:
                        print(f"Could not calculate ROC AUC for {model_name}: {str(e)}")
            
            # Metrics calculated, continue to classification metrics
                
            # Try to calculate F1, precision, and recall for classification
            try:
                metrics['f1'] = f1_score(self.y_test, y_pred, average='weighted')
                metrics['precision'] = precision_score(self.y_test, y_pred, average='weighted')
                metrics['recall'] = recall_score(self.y_test, y_pred, average='weighted')
            except Exception as e:
                if self.verbose:
                    print(f"Error calculating classification metrics for {model_name}: {str(e)}")
                # Skip if not applicable
                pass
                
            # Make sure we have auc in the metrics 
            if 'roc_auc' in metrics and 'auc' not in metrics:
                metrics['auc'] = metrics['roc_auc']
                
            # Make sure we have roc_auc in the metrics
            if 'auc' in metrics and 'roc_auc' not in metrics:
                metrics['roc_auc'] = metrics['auc']
                
            # Make sure we have AUC in metrics
            if 'roc_auc' in metrics and 'auc' not in metrics:
                metrics['auc'] = metrics['roc_auc']
            
            model_info['metrics'] = metrics
            
            # Debug output
            if self.verbose:
                print(f"DEBUG: Calculated metrics for {model_name}: {metrics}")
            
        except Exception as e:
            if self.verbose:
                print(f"Error calculating metrics for {model_name}: {str(e)}")
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
            if self.verbose:
                print(f"Error extracting hyperparameters: {str(e)}")
            return {'error': str(e)}
    
    def _get_test_config(self, test_type: str) -> dict:
        """Get configuration options for a specific test type."""
        config_options = {
            'quick': {},
            'medium': {},
            'full': {}
        }
        
        if test_type == 'robustness':
            config_options['quick'] = {
                'perturbation_methods': ['raw', 'quantile'],
                'levels': [0.1, 0.2],
                'n_trials': 5
            }
            config_options['medium'] = {
                'perturbation_methods': ['raw', 'quantile', 'adversarial'],
                'levels': [0.05, 0.1, 0.2],
                'n_trials': 10
            }
            config_options['full'] = {
                'perturbation_methods': ['raw', 'quantile', 'adversarial', 'custom'],
                'levels': [0.01, 0.05, 0.1, 0.2, 0.3],
                'n_trials': 20
            }
        
        elif test_type == 'uncertainty':
            config_options['quick'] = {
                'methods': ['crqr'],
                'alpha_levels': [0.1, 0.2]
            }
            config_options['medium'] = {
                'methods': ['crqr'],
                'alpha_levels': [0.05, 0.1, 0.2]
            }
            config_options['full'] = {
                'methods': ['crqr'],
                'alpha_levels': [0.01, 0.05, 0.1, 0.2, 0.3]
            }
        
        elif test_type == 'resilience':
            config_options['quick'] = {
                'drift_types': ['covariate', 'label'],
                'drift_intensities': [0.1, 0.2]
            }
            config_options['medium'] = {
                'drift_types': ['covariate', 'label', 'concept'],
                'drift_intensities': [0.05, 0.1, 0.2]
            }
            config_options['full'] = {
                'drift_types': ['covariate', 'label', 'concept', 'temporal'],
                'drift_intensities': [0.01, 0.05, 0.1, 0.2, 0.3]
            }
        
        elif test_type == 'hyperparameters':
            config_options['quick'] = {
                'n_trials': 10,
                'optimization_metric': 'accuracy'
            }
            config_options['medium'] = {
                'n_trials': 30,
                'optimization_metric': 'accuracy'
            }
            config_options['full'] = {
                'n_trials': 100,
                'optimization_metric': 'accuracy'
            }
        
        return config_options
        
    def run_tests(self, config_name: str = 'quick') -> dict:
        """
        Run all tests specified during initialization with the given configuration.
        
        Parameters:
        -----------
        config_name : str
            Name of the configuration to use: 'quick', 'medium', or 'full'
            
        Returns:
        --------
        dict : Dictionary with test results
        """
        if self.verbose:
            print(f"Running tests with {config_name} configuration...")
            
        # Check if we have a model to test
        if not hasattr(self.dataset, 'model') or self.dataset.model is None:
            if self.verbose:
                print("No model found in dataset. Skipping tests.")
            return {}
            
        # Make sure we have run initial tests first to get base metrics
        if not hasattr(self, 'test_results') or not self.test_results or not 'models' in self.test_results:
            self.run_initial_tests()
            
        # Initialize results dictionary
        results = {}
        
        # Run robustness tests if requested
        if "robustness" in self.tests:
            from deepbridge.utils.robustness import run_robustness_tests
            
            # Initialize robustness results dictionary
            robustness_results = {
                'primary_model': {},
                'alternative_models': {}
            }
            
            # Test primary model
            if self.verbose:
                print(f"Testing robustness of primary model...")
            
            # First check if we have AUC metrics from initial tests
            auc_from_metrics = None
            if 'models' in self.test_results and 'primary_model' in self.test_results['models']:
                primary_metrics = self.test_results['models']['primary_model'].get('metrics', {})
                # Check for auc or roc_auc
                auc_from_metrics = primary_metrics.get('auc', primary_metrics.get('roc_auc'))
                if auc_from_metrics and self.verbose:
                    print(f"DEBUG: Using AUC {auc_from_metrics} from initial metrics for primary model")
                    
            # Run robustness tests for primary model
            primary_results = run_robustness_tests(
                self.dataset, 
                config_name=config_name,
                metric='AUC',
                verbose=self.verbose,
                feature_subset=self.features_select,
                model_name="primary_model"
            )
            
            # Add initial metrics to results if available
            if 'models' in self.test_results and 'primary_model' in self.test_results['models']:
                primary_metrics = self.test_results['models']['primary_model'].get('metrics', {})
                if 'metrics' not in primary_results:
                    primary_results['metrics'] = {}
                    
                # Copy all metrics from initial tests
                primary_results['metrics'].update(primary_metrics)
                
                # Debug output
                if self.verbose:
                    print(f"DEBUG: Primary model metrics after merge: {primary_results.get('metrics', {})}")
            robustness_results['primary_model'] = primary_results
            
            # Test alternative models
            if self.alternative_models:
                for model_name, model in self.alternative_models.items():
                    if self.verbose:
                        print(f"Testing robustness of alternative model: {model_name}")
                    
                    # Create a new dataset with the alternative model
                    alt_dataset = self._create_alternative_dataset(model)
                    
                    # Run robustness tests on the alternative model
                    alt_results = run_robustness_tests(
                        alt_dataset,
                        config_name=config_name,
                        metric='AUC',
                        verbose=self.verbose,
                        feature_subset=self.features_select,
                        model_name=model_name
                    )
                    
                    # Check if we have metrics from initial tests
                    alt_metrics_from_initial = None
                    if 'models' in self.test_results and model_name in self.test_results['models']:
                        alt_metrics_from_initial = self.test_results['models'][model_name].get('metrics', {})
                        if alt_metrics_from_initial and self.verbose:
                            print(f"DEBUG: Initial metrics available for {model_name}: {alt_metrics_from_initial}")
                    
                    # Calculate metrics directly to ensure unique values for each model
                    try:
                        print(f"DEBUG: Calculating direct metrics for alternative model {model_name}")
                        from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
                        
                        X_test = alt_dataset.get_feature_data('test')
                        y_test = alt_dataset.get_target_data('test')
                        
                        # Make predictions
                        y_pred = model.predict(X_test)
                        
                        # Add metrics to results
                        if 'metrics' not in alt_results:
                            alt_results['metrics'] = {}
                            
                        # Add basic metrics
                        alt_results['metrics']['accuracy'] = accuracy_score(y_test, y_pred)
                        
                        # Calculate AUC if possible
                        if hasattr(model, 'predict_proba'):
                            try:
                                y_prob = model.predict_proba(X_test)
                                if y_prob.shape[1] > 1:
                                    alt_results['metrics']['auc'] = roc_auc_score(y_test, y_prob[:, 1])
                            except Exception as e:
                                print(f"DEBUG: Error calculating AUC for {model_name}: {e}")
                                
                        # Calculate F1, precision, recall
                        try:
                            alt_results['metrics']['f1'] = f1_score(y_test, y_pred)
                            alt_results['metrics']['precision'] = precision_score(y_test, y_pred)
                            alt_results['metrics']['recall'] = recall_score(y_test, y_pred)
                        except Exception as e:
                            print(f"DEBUG: Error calculating classification metrics for {model_name}: {e}")
                        
                        # If we have initial metrics, use them to fill any missing metrics
                        if alt_metrics_from_initial:
                            # First copy any missing metrics from initial metrics
                            for metric_name, metric_value in alt_metrics_from_initial.items():
                                if metric_name not in alt_results['metrics']:
                                    alt_results['metrics'][metric_name] = metric_value
                            
                            # Make sure we have an AUC value
                            if 'auc' not in alt_results['metrics'] and ('auc' in alt_metrics_from_initial or 'roc_auc' in alt_metrics_from_initial):
                                auc_value = alt_metrics_from_initial.get('auc', alt_metrics_from_initial.get('roc_auc'))
                                alt_results['metrics']['auc'] = auc_value
                            
                        print(f"DEBUG: Direct metrics for {model_name}: {alt_results['metrics']}")
                    except Exception as e:
                        print(f"DEBUG: Failed to calculate direct metrics for {model_name}: {e}")
                        
                        # If direct calculation failed but we have initial metrics, use those
                        if alt_metrics_from_initial:
                            if 'metrics' not in alt_results:
                                alt_results['metrics'] = {}
                            alt_results['metrics'].update(alt_metrics_from_initial)
                            print(f"DEBUG: Using initial metrics for {model_name} instead: {alt_results['metrics']}")
                    
                    # Store results
                    robustness_results['alternative_models'][model_name] = alt_results
            
            # Store all robustness results
            results['robustness'] = robustness_results
            
        # Run uncertainty tests if requested
        if "uncertainty" in self.tests:
            from deepbridge.utils.uncertainty import run_uncertainty_tests
            
            # Initialize uncertainty results dictionary
            uncertainty_results = {
                'primary_model': {},
                'alternative_models': {}
            }
            
            # Test primary model
            if self.verbose:
                print(f"Testing uncertainty quantification of primary model...")
            
            primary_results = run_uncertainty_tests(
                self.dataset, 
                config_name=config_name,
                verbose=self.verbose,
                feature_subset=self.features_select
            )
            uncertainty_results['primary_model'] = primary_results
            
            # Test alternative models
            if self.alternative_models:
                for model_name, model in self.alternative_models.items():
                    if self.verbose:
                        print(f"Testing uncertainty of alternative model: {model_name}")
                    
                    # Create a new dataset with the alternative model
                    alt_dataset = self._create_alternative_dataset(model)
                    
                    # Run uncertainty tests on the alternative model
                    alt_results = run_uncertainty_tests(
                        alt_dataset,
                        config_name=config_name,
                        verbose=self.verbose,
                        feature_subset=self.features_select
                    )
                    
                    # Store results
                    uncertainty_results['alternative_models'][model_name] = alt_results
            
            # Store all uncertainty results
            results['uncertainty'] = uncertainty_results
            
        # Run resilience tests if requested
        if "resilience" in self.tests:
            from deepbridge.utils.resilience import run_resilience_tests
            
            # Initialize resilience results dictionary
            resilience_results = {
                'primary_model': {},
                'alternative_models': {}
            }
            
            # Test primary model
            if self.verbose:
                print(f"Testing resilience of primary model...")
            
            primary_results = run_resilience_tests(
                self.dataset, 
                config_name=config_name,
                metric='auc', 
                verbose=self.verbose,
                feature_subset=self.features_select
            )
            resilience_results['primary_model'] = primary_results
            
            # Test alternative models
            if self.alternative_models:
                for model_name, model in self.alternative_models.items():
                    if self.verbose:
                        print(f"Testing resilience of alternative model: {model_name}")
                    
                    # Create a new dataset with the alternative model
                    alt_dataset = self._create_alternative_dataset(model)
                    
                    # Run resilience tests on the alternative model
                    alt_results = run_resilience_tests(
                        alt_dataset,
                        config_name=config_name,
                        metric='auc',
                        verbose=self.verbose,
                        feature_subset=self.features_select
                    )
                    
                    # Store results
                    resilience_results['alternative_models'][model_name] = alt_results
            
            # Store all resilience results
            results['resilience'] = resilience_results
            
        # Run hyperparameter tests if requested
        if "hyperparameters" in self.tests:
            from deepbridge.utils.hyperparameter import run_hyperparameter_tests
            
            # Initialize hyperparameter results dictionary
            hyperparameter_results = {
                'primary_model': {},
                'alternative_models': {}
            }
            
            # Test primary model
            if self.verbose:
                print(f"Testing hyperparameter importance of primary model...")
            
            primary_results = run_hyperparameter_tests(
                self.dataset, 
                config_name=config_name,
                metric='accuracy', 
                verbose=self.verbose,
                feature_subset=self.features_select
            )
            hyperparameter_results['primary_model'] = primary_results
            
            # Test alternative models
            if self.alternative_models:
                for model_name, model in self.alternative_models.items():
                    if self.verbose:
                        print(f"Testing hyperparameter importance of alternative model: {model_name}")
                    
                    # Create a new dataset with the alternative model
                    alt_dataset = self._create_alternative_dataset(model)
                    
                    # Run hyperparameter tests on the alternative model
                    alt_results = run_hyperparameter_tests(
                        alt_dataset,
                        config_name=config_name,
                        metric='accuracy',
                        verbose=self.verbose,
                        feature_subset=self.features_select
                    )
                    
                    # Store results
                    hyperparameter_results['alternative_models'][model_name] = alt_results
            
            # Store all hyperparameter results
            results['hyperparameters'] = hyperparameter_results
        
        # Store results in the object for future reference
        self.test_results.update(results)
        
        return results
        
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