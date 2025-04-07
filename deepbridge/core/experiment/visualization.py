"""
Visualization manager with improved separation of concerns.
This module provides a standardized way to access visualizations from test results.
"""

import typing as t

from deepbridge.core.experiment.interfaces import IVisualizationManager
from deepbridge.core.experiment.interfaces import ITestRunner

class VisualizationManager(IVisualizationManager):
    """
    Implementation of the IVisualizationManager interface.
    Centralizes access to visualizations from test results.
    """
    
    def __init__(self, test_runner: ITestRunner):
        """
        Initialize the visualization manager with a test runner.
        
        Args:
            test_runner: The TestRunner instance containing test results
        """
        self.test_runner = test_runner
    
    def get_visualization(self, test_type: str, visualization_name: str) -> t.Any:
        """
        Get a specific visualization for a test type.
        
        Args:
            test_type: Type of test ('robustness', 'uncertainty', etc.)
            visualization_name: Name of the visualization to retrieve
            
        Returns:
            The requested visualization or None if not available
        """
        visualizations = self.get_visualizations(test_type)
        return visualizations.get(visualization_name)
    
    def get_visualizations(self, test_type: str) -> dict:
        """
        Get all visualizations for a test type.
        
        Args:
            test_type: Type of test ('robustness', 'uncertainty', etc.)
            
        Returns:
            Dictionary of visualizations for the test type
        """
        test_results = self._get_test_results(test_type)
        if not test_results:
            return {}
            
        # Extract visualizations from the test results
        primary_results = test_results.get('primary_model', {})
        return primary_results.get('visualizations', {})
        
    def get_test_results(self, test_type: str) -> dict:
        """
        Get results for a specific test type.
        
        Args:
            test_type: Type of test (robustness, uncertainty, etc.)
            
        Returns:
            Dictionary with test results
        """
        # This public method is required by the IVisualizationManager interface
        return self._get_test_results(test_type)
    
    def _get_test_results(self, test_type: str) -> dict:
        """
        Get results for a specific test type, running the test if not available.
        
        Args:
            test_type: Type of test ('robustness', 'uncertainty', etc.)
            
        Returns:
            Dictionary with test results or an empty dict if not available
        """
        results = self.test_runner.get_test_results(test_type)
        
        # If results are not available but test type is in configured tests,
        # try to run the test manager
        if results is None and test_type in self.test_runner.tests:
            self._run_manager(test_type)
            results = self.test_runner.get_test_results(test_type)
            
        if results is None:
            return {}
            
        if hasattr(results, 'results'):
            return results.results
            
        return results
    
    def _run_manager(self, test_type: str) -> None:
        """
        Run the appropriate manager for a test type.
        
        Args:
            test_type: Type of test ('robustness', 'uncertainty', etc.)
        """
        from deepbridge.core.experiment.managers import (
            RobustnessManager, UncertaintyManager, ResilienceManager, HyperparameterManager
        )
        
        # Create and run the appropriate manager
        if test_type == 'robustness':
            manager = RobustnessManager(
                self.test_runner.dataset, 
                self.test_runner.alternative_models, 
                self.test_runner.verbose
            )
        elif test_type == 'uncertainty':
            manager = UncertaintyManager(
                self.test_runner.dataset, 
                self.test_runner.alternative_models, 
                self.test_runner.verbose
            )
        elif test_type == 'resilience':
            manager = ResilienceManager(
                self.test_runner.dataset, 
                self.test_runner.alternative_models, 
                self.test_runner.verbose
            )
        elif test_type == 'hyperparameters':
            manager = HyperparameterManager(
                self.test_runner.dataset, 
                self.test_runner.alternative_models, 
                self.test_runner.verbose
            )
        else:
            raise ValueError(f"Unsupported test type: {test_type}")
            
        # Run the manager and update test results
        results = manager.run_tests()
        self.test_runner.test_results[test_type] = results
    
    # Convenience methods for accessing common visualizations
    
    # Robustness visualization methods
    def get_robustness_results(self):
        """Get the robustness test results."""
        return self._get_test_results('robustness')
        
    def get_robustness_visualizations(self):
        """Get the robustness visualizations generated by the tests."""
        return self.get_visualizations('robustness')
        
    def plot_robustness_comparison(self):
        """Get the plotly figure showing the comparison of robustness across models."""
        return self.get_visualization('robustness', 'models_comparison')
    
    def plot_robustness_distribution(self):
        """Get the boxplot showing distribution of robustness scores."""
        return self.get_visualization('robustness', 'score_distribution')
    
    def plot_feature_importance_robustness(self):
        """Get the plotly figure showing feature importance for robustness."""
        return self.get_visualization('robustness', 'feature_importance')
    
    def plot_perturbation_methods_comparison(self):
        """Get the plotly figure comparing different perturbation methods."""
        return self.get_visualization('robustness', 'perturbation_methods')
        
    # Uncertainty visualization methods
    def get_uncertainty_results(self):
        """Get the uncertainty test results."""
        return self._get_test_results('uncertainty')
        
    def get_uncertainty_visualizations(self):
        """Get the uncertainty visualizations generated by the tests."""
        return self.get_visualizations('uncertainty')
        
    def plot_uncertainty_alpha_comparison(self):
        """Get the plotly figure showing the comparison of different alpha levels."""
        return self.get_visualization('uncertainty', 'alpha_comparison')
    
    def plot_uncertainty_width_distribution(self):
        """Get the boxplot showing distribution of interval widths."""
        return self.get_visualization('uncertainty', 'width_distribution')
    
    def plot_feature_importance_uncertainty(self):
        """Get the plotly figure showing feature importance for uncertainty."""
        return self.get_visualization('uncertainty', 'feature_importance')
    
    def plot_coverage_vs_width(self):
        """Get the plotly figure showing trade-off between coverage and width."""
        return self.get_visualization('uncertainty', 'coverage_vs_width')
        
    # Resilience results methods
    def get_resilience_results(self):
        """Get the resilience test results."""
        return self._get_test_results('resilience')
    
    # Hyperparameter methods
    def get_hyperparameter_results(self):
        """Get the hyperparameter importance test results."""
        return self._get_test_results('hyperparameters')
    
    def get_hyperparameter_importance(self):
        """Get the hyperparameter importance scores for the primary model."""
        results = self.get_hyperparameter_results()
        if results and 'primary_model' in results:
            return results['primary_model'].get('sorted_importance', {})
        return None
    
    def get_hyperparameter_tuning_order(self):
        """Get the suggested hyperparameter tuning order for the primary model."""
        results = self.get_hyperparameter_results()
        if results and 'primary_model' in results:
            return results['primary_model'].get('tuning_order', [])
        return None