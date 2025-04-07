"""
Factory for creating test results of various types.
This module provides a unified interface for creating test results.
"""

import typing as t
from pathlib import Path

# Import various result classes
from deepbridge.core.experiment.interfaces import TestResult, ModelResult
from deepbridge.core.experiment.results import (
    BaseTestResult, RobustnessResult, UncertaintyResult, 
    ResilienceResult, HyperparameterResult, ExperimentResult
)

try:
    from deepbridge.core.experiment.model_result import (
        BaseModelResult, ClassificationModelResult, RegressionModelResult, 
        create_model_result
    )
except ImportError:
    # Create simplified versions
    from deepbridge.core.experiment.results import BaseModelResult, create_model_result


class TestResultFactory:
    """
    Factory class for creating test results.
    This class centralizes the creation of various types of test results.
    """
    
    @staticmethod
    def create_test_result(test_type: str, results: dict, metadata: t.Optional[dict] = None) -> TestResult:
        """
        Create a test result object of the appropriate type
        
        Args:
            test_type: Type of test ('robustness', 'uncertainty', etc.)
            results: Raw test results
            metadata: Additional test metadata
            
        Returns:
            TestResult: Appropriate test result object
        """
        test_type = test_type.lower()
        
        if test_type == 'robustness':
            return RobustnessResult(results, metadata)
        elif test_type == 'uncertainty':
            return UncertaintyResult(results, metadata)
        elif test_type == 'resilience':
            return ResilienceResult(results, metadata)
        elif test_type in ('hyperparameters', 'hyperparameter'):
            return HyperparameterResult(results, metadata)
        else:
            return BaseTestResult(test_type.capitalize(), results, metadata)
    
    @staticmethod
    def create_model_result(
        model_name: str,
        model_type: str,
        metrics: dict,
        problem_type: str = 'classification',
        **kwargs
    ) -> ModelResult:
        """
        Create a model result object of the appropriate type
        
        Args:
            model_name: Name of the model
            model_type: Type or class of the model
            metrics: Model performance metrics
            problem_type: Type of problem ('classification', 'regression', 'forecasting')
            **kwargs: Additional parameters for specific model result types
            
        Returns:
            ModelResult: Appropriate model result object
        """
        return create_model_result(model_name, model_type, metrics, problem_type, **kwargs)
    
    @staticmethod
    def create_experiment_result(results: dict) -> ExperimentResult:
        """
        Create an experiment result from a dictionary of results
        
        Args:
            results: Dictionary of test results
            
        Returns:
            ExperimentResult: Experiment result object
        """
        return ExperimentResult.from_dict(results)
    
    @staticmethod
    def combine_results(*results: TestResult, experiment_type: str = "binary_classification") -> ExperimentResult:
        """
        Combine multiple test results into a single experiment result
        
        Args:
            *results: Test result objects to combine
            experiment_type: Type of experiment
            
        Returns:
            ExperimentResult: Combined experiment result
        """
        # Create a new experiment result
        experiment_result = ExperimentResult(
            experiment_type=experiment_type,
            config={}
        )
        
        # Add each test result
        for result in results:
            experiment_result.add_result(result)
            
        return experiment_result