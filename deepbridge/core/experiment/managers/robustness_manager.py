"""
Robustness manager for model evaluation.
"""

import typing as t
import copy
import pandas as pd
import numpy as np
from deepbridge.core.experiment.managers.base_manager import BaseManager

class RobustnessManager(BaseManager):
    """
    Handles robustness testing and comparison between models.
    Implements the BaseManager interface.
    """
    
    def run_tests(self, config_name='quick', **kwargs) -> dict:
        """
        Run robustness tests using RobustnessSuite and compare
        the original model with alternative models.
        
        Args:
            config_name: Configuration profile ('quick', 'medium', 'full')
            **kwargs: Additional test parameters
                - n_iterations: Number of iterations per perturbation level (default: 1)
            
        Returns:
            dict: Results of robustness tests
        """
        from deepbridge.validation.wrappers.robustness_suite import RobustnessSuite
        
        # Extract n_iterations from kwargs with default value
        n_iterations = kwargs.get('n_iterations', 1)
        
        self.log(f"Running robustness tests with {n_iterations} iterations per perturbation level...")
            
        # Initialize results storage
        results = {
            'main_model': {},
            'alternative_models': {}
        }
        
        # Test main model with n_iterations
        suite = RobustnessSuite(
            dataset=self.dataset, 
            verbose=self.verbose,
            n_iterations=n_iterations
        )
        
        results['main_model'] = suite.config(config_name).run()
        
        # Report generation has been removed
        
        # Test alternative models if we have any
        if self.alternative_models:
            for name, model in self.alternative_models.items():
                self.log(f"Testing robustness of alternative model: {name}")
                
                # Create temporary dataset with alternative model
                temp_dataset = copy.deepcopy(self.dataset)
                temp_dataset.set_model(model)
                
                # Run robustness tests on alternative model
                alt_suite = RobustnessSuite(temp_dataset, verbose=self.verbose)
                alt_results = alt_suite.config(config_name).run()
                results['alternative_models'][name] = alt_results
                
                # Report generation has been removed
        
        # Compare models based on robustness scores
        results['comparison'] = self.compare_models_robustness(results)
        
        return results
    
    def compare_models(self, config_name='quick', **kwargs) -> dict:
        """
        Compare robustness of multiple models.
        
        Args:
            config_name: Configuration profile ('quick', 'medium', 'full')
            **kwargs: Additional test parameters
            
        Returns:
            dict: Comparison results for all models
        """
        # Run full tests that include model comparison
        return self.run_tests(config_name, **kwargs)
    
    def compare_models_robustness(self, robustness_results) -> dict:
        """
        Compare the robustness scores of the main model and alternative models.
        """
        comparison = {}
        
        # Extract overall score for main model
        main_score = None
        if 'main_model' in robustness_results:
            # Try different possible paths to find the robustness score
            if 'robustness_scores' in robustness_results['main_model']:
                main_score = robustness_results['main_model']['robustness_scores'].get('overall_score', 0)
            elif 'robustness_score' in robustness_results['main_model']:
                main_score = robustness_results['main_model']['robustness_score']
            
        # Extract scores for alternative models
        alt_scores = {}
        if 'alternative_models' in robustness_results:
            for model_name, model_results in robustness_results['alternative_models'].items():
                model_score = None
                # Try different possible paths to find the robustness score
                if 'robustness_scores' in model_results:
                    model_score = model_results['robustness_scores'].get('overall_score', 0)
                elif 'robustness_score' in model_results:
                    model_score = model_results['robustness_score']
                
                # Only add if we found a score
                if model_score is not None:
                    alt_scores[model_name] = model_score
        
        # Identify the most robust model
        all_scores = {
            'main_model': main_score
        }
        all_scores.update(alt_scores)
        
        # Handle None values by replacing them with 0 for comparison
        most_robust_model = max(all_scores.items(), key=lambda x: x[1] if x[1] is not None else 0)
        
        # Store comparison results
        comparison = {
            'all_scores': all_scores,
            'most_robust_model': most_robust_model[0],
            'most_robust_score': most_robust_model[1]
        }
        
        self.log("\nRobustness comparison results:")
        for model_name, score in all_scores.items():
            if score is not None:
                self.log(f"{model_name}: {score:.4f}")
            else:
                self.log(f"{model_name}: None")
                
        if most_robust_model[1] is not None:
            self.log(f"\nMost robust model: {most_robust_model[0]} (score: {most_robust_model[1]:.4f})")
        else:
            self.log(f"\nMost robust model: {most_robust_model[0]} (score: None)")
            
        return comparison
    
    # Visualization methods have been removed