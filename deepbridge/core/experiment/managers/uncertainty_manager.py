"""
Uncertainty manager for model evaluation.
"""

import typing as t

class UncertaintyManager:
    """
    Manager class for running uncertainty tests on models.
    """
    
    def __init__(self, dataset, alternative_models=None, verbose=False):
        """
        Initialize the uncertainty manager.
        
        Args:
            dataset: DBDataset instance containing the primary model
            alternative_models: Dictionary of alternative models for comparison
            verbose: Whether to print progress information
        """
        self.dataset = dataset
        self.alternative_models = alternative_models or {}
        self.verbose = verbose
        
    def run_tests(self, config_name='quick'):
        """
        Run standard uncertainty tests on the primary model.
        
        Args:
            config_name: Configuration profile ('quick', 'medium', 'full')
            
        Returns:
            dict: Results of uncertainty tests
        """
        if self.verbose:
            print("Running uncertainty tests...")
            
        from deepbridge.utils.uncertainty import run_uncertainty_tests
        
        # Run tests on primary model
        results = run_uncertainty_tests(
            self.dataset,
            config_name=config_name,
            verbose=self.verbose
        )
        
        if self.verbose:
            print("Uncertainty tests completed.")
            
        return results
    
    def compare_models(self, config_name='quick'):
        """
        Compare uncertainty quantification across all models.
        
        Args:
            config_name: Configuration profile ('quick', 'medium', 'full')
            
        Returns:
            dict: Comparison results for all models
        """
        if self.verbose:
            print("Comparing uncertainty quantification across models...")
            
        from deepbridge.utils.uncertainty import run_uncertainty_tests
        from deepbridge.core.db_data import DBDataset
        
        # Initialize results
        results = {
            'primary_model': {},
            'alternative_models': {}
        }
        
        # Test primary model
        if self.verbose:
            print("Testing primary model uncertainty...")
            
        primary_results = run_uncertainty_tests(
            self.dataset,
            config_name=config_name,
            verbose=self.verbose
        )
        results['primary_model'] = primary_results
        
        # Test alternative models
        if self.alternative_models:
            for model_name, model in self.alternative_models.items():
                if self.verbose:
                    print(f"Testing uncertainty of alternative model: {model_name}")
                
                # Create a new dataset with the alternative model
                alt_dataset = DBDataset(
                    train_data=self.dataset.train_data,
                    test_data=self.dataset.test_data,
                    target_column=self.dataset.target_name,
                    model=model
                )
                
                # Run uncertainty tests on the alternative model
                alt_results = run_uncertainty_tests(
                    alt_dataset,
                    config_name=config_name,
                    verbose=self.verbose
                )
                
                # Store results
                results['alternative_models'][model_name] = alt_results
        
        return results