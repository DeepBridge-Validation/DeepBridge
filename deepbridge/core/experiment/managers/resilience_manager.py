"""
Resilience manager for model evaluation.
"""

import typing as t
from deepbridge.core.experiment.managers.base_manager import BaseManager
from deepbridge.utils.dataset_factory import DBDatasetFactory

class ResilienceManager(BaseManager):
    """
    Manager class for running resilience tests on models.
    Implements the BaseManager interface.
    """
    
    def run_tests(self, config_name='quick', metric='auc'):
        """
        Run standard resilience tests on the primary model.
        
        Args:
            config_name: Configuration profile ('quick', 'medium', 'full')
            metric: Performance metric to use for evaluation
            
        Returns:
            dict: Results of resilience tests
        """
        self.log("Running resilience tests...")
            
        from deepbridge.utils.resilience import run_resilience_tests
        
        # Run tests on primary model
        results = run_resilience_tests(
            self.dataset,
            config_name=config_name,
            metric=metric,
            verbose=self.verbose
        )
        
        self.log("Resilience tests completed.")
            
        return results
    
    def compare_models(self, config_name='quick', metric='auc'):
        """
        Compare resilience across all models.
        
        Args:
            config_name: Configuration profile ('quick', 'medium', 'full')
            metric: Performance metric to use for evaluation
            
        Returns:
            dict: Comparison results for all models
        """
        self.log("Comparing resilience across models...")
            
        from deepbridge.utils.resilience import run_resilience_tests
        
        # Initialize results
        results = {
            'primary_model': {},
            'alternative_models': {}
        }
        
        # Test primary model
        self.log("Testing primary model resilience...")
            
        primary_results = run_resilience_tests(
            self.dataset,
            config_name=config_name,
            metric=metric,
            verbose=self.verbose
        )
        results['primary_model'] = primary_results
        
        # Test alternative models
        if self.alternative_models:
            for model_name, model in self.alternative_models.items():
                self.log(f"Testing resilience of alternative model: {model_name}")
                
                # Create a new dataset with the alternative model
                alt_dataset = DBDatasetFactory.create_for_alternative_model(
                    original_dataset=self.dataset,
                    model=model
                )
                
                # Run resilience tests on the alternative model
                alt_results = run_resilience_tests(
                    alt_dataset,
                    config_name=config_name,
                    metric=metric,
                    verbose=self.verbose
                )
                
                # Store results
                results['alternative_models'][model_name] = alt_results
        
        return results