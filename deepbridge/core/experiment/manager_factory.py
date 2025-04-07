"""
Factory for test managers.
This module provides a centralized way to create and access test manager instances.
"""

import typing as t
from abc import ABC, abstractmethod

class ManagerFactory:
    """
    Factory for creating and managing test manager instances.
    This class implements the Factory pattern to create various manager types.
    """
    
    # Store manager class registry
    _manager_classes = {}
    
    # Store singleton instances
    _instances = {}
    
    @classmethod
    def register_manager(cls, test_type: str, manager_class: t.Type) -> None:
        """
        Register a manager class for a test type
        
        Args:
            test_type: Type of test the manager handles
            manager_class: Manager class to register
        """
        cls._manager_classes[test_type] = manager_class
    
    @classmethod
    def get_manager(cls, test_type: str, dataset, alternative_models=None, verbose=False):
        """
        Get a manager instance for a test type.
        If an instance already exists, return it (singleton pattern).
        
        Args:
            test_type: Type of test
            dataset: Dataset to use with the manager
            alternative_models: Dictionary of alternative models
            verbose: Whether to enable verbose output
            
        Returns:
            Manager instance for the specified test type
            
        Raises:
            ValueError: If there is no manager registered for the test type
        """
        # Check if manager type is registered
        if test_type not in cls._manager_classes:
            # Try to import standard managers
            cls._import_standard_managers()
            
            # Check again after import
            if test_type not in cls._manager_classes:
                raise ValueError(f"No manager registered for test type: {test_type}")
        
        # Get the manager class
        manager_class = cls._manager_classes[test_type]
        
        # Create a key for the instance
        # Use the dataset and test_type to uniquely identify the manager
        instance_key = (test_type, id(dataset))
        
        # Check if we already have an instance
        if instance_key not in cls._instances:
            # Create a new instance
            cls._instances[instance_key] = manager_class(
                dataset=dataset,
                alternative_models=alternative_models,
                verbose=verbose
            )
            
        return cls._instances[instance_key]
    
    @classmethod
    def _import_standard_managers(cls) -> None:
        """Import and register standard manager classes."""
        try:
            # Import managers
            from deepbridge.core.experiment.managers import (
                RobustnessManager, UncertaintyManager, ResilienceManager, HyperparameterManager
            )
            
            # Register managers
            cls.register_manager('robustness', RobustnessManager)
            cls.register_manager('uncertainty', UncertaintyManager)
            cls.register_manager('resilience', ResilienceManager)
            cls.register_manager('hyperparameters', HyperparameterManager)
            
        except ImportError:
            # Failed to import managers
            pass
    
    @classmethod
    def clear_instances(cls) -> None:
        """Clear all manager instances."""
        cls._instances.clear()
    
    @classmethod
    def get_supported_types(cls) -> t.List[str]:
        """
        Get a list of supported manager types
        
        Returns:
            List of supported test types
        """
        # Make sure standard managers are imported
        cls._import_standard_managers()
        
        # Return the keys of the manager registry
        return list(cls._manager_classes.keys())