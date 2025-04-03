"""
Resilience manager for model evaluation.
"""

import typing as t

class ResilienceManager:
    """
    Manager class for running resilience tests on models.
    """
    
    def __init__(self, dataset, alternative_models=None, verbose=False):
        """
        Initialize the resilience manager.
        
        Args:
            dataset: DBDataset instance containing the primary model
            alternative_models: Dictionary of alternative models for comparison
            verbose: Whether to print progress information
        """
        self.dataset = dataset
        self.alternative_models = alternative_models or {}
        self.verbose = verbose
        
    def run_tests(self, config_name='quick', metric='auc'):
        """
        Run standard resilience tests on the primary model.
        
        Args:
            config_name: Configuration profile ('quick', 'medium', 'full')
            metric: Performance metric to use for evaluation
            
        Returns:
            dict: Results of resilience tests
        """
        if self.verbose:
            print("Running resilience tests...")
            
        from deepbridge.utils.resilience import run_resilience_tests
        
        # Run tests on primary model
        results = run_resilience_tests(
            self.dataset,
            config_name=config_name,
            metric=metric,
            verbose=self.verbose
        )
        
        if self.verbose:
            print("Resilience tests completed.")
            
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
        if self.verbose:
            print("Comparing resilience across models...")
            
        from deepbridge.utils.resilience import run_resilience_tests
        from deepbridge.core.db_data import DBDataset
        
        # Initialize results
        results = {
            'primary_model': {},
            'alternative_models': {}
        }
        
        # Test primary model
        if self.verbose:
            print("Testing primary model resilience...")
            
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
                if self.verbose:
                    print(f"Testing resilience of alternative model: {model_name}")
                
                # Create a new dataset with the alternative model
                alt_dataset = DBDataset(
                    train_data=self.dataset.train_data,
                    test_data=self.dataset.test_data,
                    target_column=self.dataset.target_name,
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