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
        verbose: bool = False
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
        """
        self.dataset = dataset
        self.alternative_models = alternative_models
        self.tests = tests
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.verbose = verbose
        
        # Store test results
        self.test_results = {}
        
    def run_initial_tests(self) -> dict:
        """Run the tests specified in self.tests using manager classes."""
        if self.verbose:
            print(f"Running the following tests: {self.tests}")
            
        # Check if we have a model to test
        if not hasattr(self.dataset, 'model') or self.dataset.model is None:
            if self.verbose:
                print("No model found in dataset. Skipping tests.")
            return {}
            
        results = {}
            
        # Run robustness tests if requested
        if "robustness" in self.tests:
            robustness_manager = RobustnessManager(
                self.dataset, 
                self.alternative_models, 
                self.verbose
            )
            results['robustness'] = robustness_manager.run_tests()
            
        # Run uncertainty tests if requested
        if "uncertainty" in self.tests:
            uncertainty_manager = UncertaintyManager(
                self.dataset, 
                self.alternative_models, 
                self.verbose
            )
            results['uncertainty'] = uncertainty_manager.run_tests()
            
        # Run resilience tests if requested
        if "resilience" in self.tests:
            resilience_manager = ResilienceManager(
                self.dataset, 
                self.alternative_models, 
                self.verbose
            )
            results['resilience'] = resilience_manager.run_tests()
            
        # Run hyperparameter tests if requested
        if "hyperparameters" in self.tests:
            hyperparameter_manager = HyperparameterManager(
                self.dataset, 
                self.alternative_models, 
                self.verbose
            )
            results['hyperparameters'] = hyperparameter_manager.run_tests()
            
        # Store results for future reference
        self.test_results.update(results)
        
        return results
        
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
            
            primary_results = run_robustness_tests(
                self.dataset, 
                config_name=config_name,
                metric='AUC', 
                verbose=self.verbose
            )
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
                        verbose=self.verbose
                    )
                    
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
                verbose=self.verbose
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
                        verbose=self.verbose
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
                verbose=self.verbose
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
                        verbose=self.verbose
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
                verbose=self.verbose
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
                        verbose=self.verbose
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