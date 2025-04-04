"""
Base manager interface for model evaluation components.
"""

import typing as t
from abc import ABC, abstractmethod

class BaseManager(ABC):
    """
    Abstract base class for all manager components.
    Defines the common interface that all managers should implement.
    """
    
    def __init__(self, dataset, alternative_models=None, verbose=False):
        """
        Initialize the base manager.
        
        Args:
            dataset: DBDataset instance containing the primary model
            alternative_models: Dictionary of alternative models for comparison
            verbose: Whether to print progress information
        """
        self.dataset = dataset
        self.alternative_models = alternative_models or {}
        self.verbose = verbose
    
    @abstractmethod
    def run_tests(self, config_name: str = 'quick', **kwargs) -> dict:
        """
        Run standard tests on the primary model.
        
        Args:
            config_name: Configuration profile ('quick', 'medium', 'full')
            **kwargs: Additional test parameters
            
        Returns:
            dict: Results of the tests
        """
        pass
    
    @abstractmethod
    def compare_models(self, config_name: str = 'quick', **kwargs) -> dict:
        """
        Compare test results across all models.
        
        Args:
            config_name: Configuration profile ('quick', 'medium', 'full')
            **kwargs: Additional test parameters
            
        Returns:
            dict: Comparison results for all models
        """
        pass
    
    def log(self, message: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def get_results(self, result_type: str = None) -> t.Dict[str, t.Any]:
        """
        Get the results of the most recent tests.
        
        Args:
            result_type: Specific type of results to return
            
        Returns:
            dict: Test results
        """
        if hasattr(self, 'results'):
            if result_type is not None and result_type in self.results:
                return self.results[result_type]
            return self.results
        return {}