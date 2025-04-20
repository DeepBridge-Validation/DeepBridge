"""
Interfaces and abstract base classes for the experiment system.
These interfaces define the contract between components, allowing for better separation of concerns.
"""

import typing as t
from abc import ABC, abstractmethod
from pathlib import Path

# Import standardized parameter names and types
try:
    from deepbridge.core.experiment.parameter_standards import (
        ParameterNames, TestType, ConfigName, ExperimentType,
        DatasetType, ModelType, FeatureImportanceDict, TestConfigDict, TestResultsDict
    )
except ImportError:
    # If parameter standards aren't available, define placeholders
    class ParameterNames:
        """Placeholder for parameter names"""
        DATASET = "dataset"
        FEATURE_SUBSET = "feature_subset"
        CONFIG_NAME = "config_name"
        VERBOSE = "verbose"
        METRIC = "metric"
        
    # Use strings for test types
    class TestType:
        """Placeholder for test types"""
        ROBUSTNESS = "robustness"
        UNCERTAINTY = "uncertainty"
        RESILIENCE = "resilience"
        HYPERPARAMETERS = "hyperparameters"
    
    # Placeholder type aliases    
    DatasetType = t.TypeVar('DatasetType')
    ModelType = t.TypeVar('ModelType')
    FeatureImportanceDict = t.Dict[str, float]
    TestConfigDict = t.Dict[str, t.Any]
    TestResultsDict = t.Dict[str, t.Any]

# Result Interfaces
class TestResult(ABC):
    """Base interface for all test result objects"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the test"""
        pass
    
    @property
    @abstractmethod
    def results(self) -> dict:
        """Get the raw results dictionary"""
        pass
    
    @property
    @abstractmethod
    def metadata(self) -> dict:
        """Get the test metadata"""
        pass
        
    @abstractmethod
    def to_dict(self) -> dict:
        """Convert test result to a dictionary format"""
        pass

class ModelResult(ABC):
    """Interface for model evaluation result"""
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the name of the model"""
        pass
    
    @property
    @abstractmethod
    def model_type(self) -> str:
        """Get the type of the model"""
        pass
    
    @property
    @abstractmethod
    def metrics(self) -> dict:
        """Get model performance metrics"""
        pass
    
    @property
    @abstractmethod
    def hyperparameters(self) -> dict:
        """Get model hyperparameters"""
        pass
        
    @property
    @abstractmethod
    def metadata(self) -> dict:
        """Get additional metadata"""
        pass
        
    @abstractmethod
    def to_dict(self) -> dict:
        """Convert the model result to a dictionary"""
        pass
        
    @abstractmethod
    def get_metric(self, metric_name: str, default: t.Any = None) -> t.Any:
        """Get a specific metric by name with optional default"""
        pass

# Test Runner Interfaces
class ITestRunner(ABC):
    """Interface for test runner components"""
    
    @abstractmethod
    def run_tests(self, config_name: str = 'quick', **kwargs) -> TestResultsDict:
        """
        Run all configured tests with the specified configuration.
        
        Args:
            config_name: Name of the configuration to use (quick, medium, full)
            **kwargs: Additional test parameters
            
        Returns:
            Dictionary containing all test results
        """
        pass
    
    @abstractmethod
    def run_test(self, test_type: str, config_name: str = 'quick', **kwargs) -> TestResult:
        """
        Run a specific test with the given configuration.
        
        Args:
            test_type: Type of test to run (robustness, uncertainty, etc.)
            config_name: Name of the configuration to use (quick, medium, full)
            **kwargs: Additional test parameters
            
        Returns:
            TestResult object containing the test results
        """
        pass
    
    @abstractmethod
    def get_test_results(self, test_type: t.Optional[str] = None) -> t.Union[TestResultsDict, TestResult]:
        """
        Get results for a specific test or all results.
        
        Args:
            test_type: Type of test to get results for (None for all tests)
            
        Returns:
            TestResult object or dictionary of all test results
        """
        pass
        
    @abstractmethod
    def get_test_config(self, test_type: str, config_name: str = 'quick') -> TestConfigDict:
        """
        Get configuration for a specific test and configuration level.
        
        Args:
            test_type: Type of test to get configuration for
            config_name: Name of the configuration level
            
        Returns:
            Dictionary with test configuration parameters
        """
        pass

# Visualization and reporting interfaces have been completely removed in this refactoring

# Experiment Interface
class IExperiment(ABC):
    """Interface for the main experiment class"""
    
    @property
    @abstractmethod
    def experiment_type(self) -> str:
        """
        Get the experiment type.
        
        Returns:
            String indicating the experiment type (binary_classification, regression, etc.)
        """
        pass
    
    @property
    @abstractmethod
    def test_results(self) -> TestResultsDict:
        """
        Get all test results.
        
        Returns:
            Dictionary containing all test results
        """
        pass
    
    @property
    @abstractmethod
    def model(self) -> t.Any:
        """
        Get the primary model.
        
        Returns:
            The primary model used in the experiment
        """
        pass
    
    @abstractmethod
    def run_tests(self, config_name: str = 'quick', **kwargs) -> TestResultsDict:
        """
        Run all tests with the specified configuration.
        
        Args:
            config_name: Name of the configuration to use (quick, medium, full)
            **kwargs: Additional test parameters
            
        Returns:
            Dictionary containing all test results
        """
        pass
    
    @abstractmethod
    def run_test(self, test_type: str, config_name: str = 'quick', **kwargs) -> TestResult:
        """
        Run a specific test with the specified configuration.
        
        Args:
            test_type: Type of test to run (robustness, uncertainty, etc.)
            config_name: Name of the configuration to use (quick, medium, full)
            **kwargs: Additional test parameters
            
        Returns:
            TestResult object containing the test results
        """
        pass
    
    # save_report method has been removed as part of the visualization/reporting cleanup
        
    @abstractmethod
    def fit(self, **kwargs) -> 'IExperiment':
        """
        Fit a model to the data.
        
        Args:
            **kwargs: Parameters for model fitting
            
        Returns:
            Self (for method chaining)
        """
        pass