"""
Hyperparameter manager for model evaluation.
"""

import typing as t

class HyperparameterManager:
    """
    Manager class for running hyperparameter importance tests on models.
    """
    
    def __init__(self, dataset, alternative_models=None, verbose=False):
        """
        Initialize the hyperparameter manager.
        
        Args:
            dataset: DBDataset instance containing the primary model
            alternative_models: Dictionary of alternative models for comparison
            verbose: Whether to print progress information
        """
        self.dataset = dataset
        self.alternative_models = alternative_models or {}
        self.verbose = verbose
        
    def run_tests(self, config_name='quick', metric='accuracy'):
        """
        Run standard hyperparameter importance tests on the primary model.
        
        Args:
            config_name: Configuration profile ('quick', 'medium', 'full')
            metric: Performance metric to use for evaluation
            
        Returns:
            dict: Results of hyperparameter importance tests
        """
        if self.verbose:
            print("Running hyperparameter importance tests...")
            
        from deepbridge.utils.hyperparameter import run_hyperparameter_tests
        
        # Run tests on primary model
        results = run_hyperparameter_tests(
            self.dataset,
            config_name=config_name,
            metric=metric,
            verbose=self.verbose
        )
        
        if self.verbose:
            print("Hyperparameter importance tests completed.")
            
        return results
    
    def compare_models(self, config_name='quick', metric='accuracy'):
        """
        Compare hyperparameter importance across all models.
        
        Args:
            config_name: Configuration profile ('quick', 'medium', 'full')
            metric: Performance metric to use for evaluation
            
        Returns:
            dict: Comparison results for all models
        """
        if self.verbose:
            print("Comparing hyperparameter importance across models...")
            
        from deepbridge.utils.hyperparameter import run_hyperparameter_tests
        from deepbridge.core.db_data import DBDataset
        
        # Initialize results
        results = {
            'primary_model': {},
            'alternative_models': {}
        }
        
        # Test primary model
        if self.verbose:
            print("Testing primary model hyperparameter importance...")
            
        primary_results = run_hyperparameter_tests(
            self.dataset,
            config_name=config_name,
            metric=metric,
            verbose=self.verbose
        )
        results['primary_model'] = primary_results
        
        # Test alternative models
        if self.alternative_models:
            for model_name, model in self.alternative_models.items():
                if self.verbose:
                    print(f"Testing hyperparameter importance of alternative model: {model_name}")
                
                # Create a new dataset with the alternative model
                alt_dataset = DBDataset(
                    train_data=self.dataset.train_data,
                    test_data=self.dataset.test_data,
                    target_column=self.dataset.target_name,
                    model=model
                )
                
                # Run hyperparameter tests on the alternative model
                alt_results = run_hyperparameter_tests(
                    alt_dataset,
                    config_name=config_name,
                    metric=metric,
                    verbose=self.verbose
                )
                
                # Store results
                results['alternative_models'][model_name] = alt_results
        
        return results