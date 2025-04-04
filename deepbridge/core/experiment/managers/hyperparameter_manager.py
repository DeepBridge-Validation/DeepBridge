"""
Hyperparameter manager for model evaluation.
"""

import typing as t
from deepbridge.core.experiment.managers.base_manager import BaseManager
from deepbridge.utils.dataset_factory import DBDatasetFactory

class HyperparameterManager(BaseManager):
    """
    Manager class for running hyperparameter importance tests on models.
    Implements the BaseManager interface.
    """
    
    def run_tests(self, config_name='quick', metric='accuracy'):
        """
        Run standard hyperparameter importance tests on the primary model.
        
        Args:
            config_name: Configuration profile ('quick', 'medium', 'full')
            metric: Performance metric to use for evaluation
            
        Returns:
            dict: Results of hyperparameter importance tests
        """
        self.log("Running hyperparameter importance tests...")
            
        from deepbridge.utils.hyperparameter import run_hyperparameter_tests
        
        # Run tests on primary model
        results = run_hyperparameter_tests(
            self.dataset,
            config_name=config_name,
            metric=metric,
            verbose=self.verbose
        )
        
        self.log("Hyperparameter importance tests completed.")
            
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
        self.log("Comparing hyperparameter importance across models...")
            
        from deepbridge.utils.hyperparameter import run_hyperparameter_tests
        
        # Initialize results
        results = {
            'primary_model': {},
            'alternative_models': {}
        }
        
        # Test primary model
        self.log("Testing primary model hyperparameter importance...")
            
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
                self.log(f"Testing hyperparameter importance of alternative model: {model_name}")
                
                # Create a new dataset with the alternative model
                alt_dataset = DBDatasetFactory.create_for_alternative_model(
                    original_dataset=self.dataset,
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