"""
Uncertainty manager for model evaluation.
"""

import typing as t
from deepbridge.core.experiment.managers.base_manager import BaseManager
from deepbridge.utils.dataset_factory import DBDatasetFactory
from deepbridge.core.experiment.parameter_standards import (
    get_test_config, TestType, ConfigName, is_valid_config_name
)

class UncertaintyManager(BaseManager):
    """
    Manager class for running uncertainty tests on models.
    Implements the BaseManager interface.
    """
    
    def run_tests(self, config_name='quick', **kwargs):
        """
        Run standard uncertainty tests on the primary model.

        Args:
            config_name: Configuration profile ('quick', 'medium', 'full')
            **kwargs: Additional test parameters

        Returns:
            dict: Results of uncertainty tests
        """
        self.log("Running uncertainty tests...")

        # Validate configuration name
        if not is_valid_config_name(config_name):
            self.log(f"Warning: Invalid configuration name '{config_name}'. Using 'quick' instead.")
            config_name = ConfigName.QUICK.value

        from deepbridge.utils.uncertainty import run_uncertainty_tests

        # Run tests on primary model
        results = run_uncertainty_tests(
            self.dataset,
            config_name=config_name,
            verbose=self.verbose,
            **kwargs
        )

        self.log("Uncertainty tests completed.")

        return results
    
    def compare_models(self, config_name='quick', **kwargs):
        """
        Compare uncertainty quantification across all models.

        Args:
            config_name: Configuration profile ('quick', 'medium', 'full')
            **kwargs: Additional test parameters

        Returns:
            dict: Comparison results for all models
        """
        self.log("Comparing uncertainty quantification across models...")

        # Validate configuration name
        if not is_valid_config_name(config_name):
            self.log(f"Warning: Invalid configuration name '{config_name}'. Using 'quick' instead.")
            config_name = ConfigName.QUICK.value

        from deepbridge.utils.uncertainty import run_uncertainty_tests

        # Initialize results
        results = {
            'primary_model': {},
            'alternative_models': {}
        }

        # Test primary model
        self.log("Testing primary model uncertainty...")

        primary_results = run_uncertainty_tests(
            self.dataset,
            config_name=config_name,
            verbose=self.verbose,
            **kwargs
        )
        results['primary_model'] = primary_results

        # Test alternative models
        if self.alternative_models:
            for model_name, model in self.alternative_models.items():
                self.log(f"Testing uncertainty of alternative model: {model_name}")

                # Create a new dataset with the alternative model
                alt_dataset = DBDatasetFactory.create_for_alternative_model(
                    original_dataset=self.dataset,
                    model=model
                )

                # Run uncertainty tests on the alternative model
                alt_results = run_uncertainty_tests(
                    alt_dataset,
                    config_name=config_name,
                    verbose=self.verbose,
                    **kwargs
                )

                # Store results
                results['alternative_models'][model_name] = alt_results

        return results