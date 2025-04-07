"""
Test strategies for different types of tests.
This module uses the Strategy pattern to encapsulate different test algorithms.
"""

import typing as t
from abc import ABC, abstractmethod

class TestStrategy(ABC):
    """Abstract base class for test strategies"""
    
    @abstractmethod
    def run_test(self, dataset, config: dict, feature_subset: t.Optional[t.List[str]] = None, 
                verbose: bool = False) -> dict:
        """
        Run a test using this strategy
        
        Args:
            dataset: Dataset to test
            config: Test configuration
            feature_subset: Features to specifically test
            verbose: Whether to output verbose logging
            
        Returns:
            dict: Test results
        """
        pass
    
    @staticmethod
    def get_default_metric() -> str:
        """Get the default metric for this test type"""
        return "accuracy"
    
    @staticmethod
    def get_configuration(config_name: str) -> dict:
        """
        Get configuration parameters for the specified configuration level
        
        Args:
            config_name: Configuration level ('quick', 'medium', 'full')
            
        Returns:
            dict: Configuration parameters
        """
        return {}

# Specific Strategy Implementations

class RobustnessTestStrategy(TestStrategy):
    """Strategy for robustness testing"""
    
    def run_test(self, dataset, config: dict, feature_subset: t.Optional[t.List[str]] = None, 
                verbose: bool = False) -> dict:
        """
        Run robustness tests on a dataset
        
        Args:
            dataset: Dataset to test
            config: Test configuration
            feature_subset: Features to specifically test
            verbose: Whether to output verbose logging
            
        Returns:
            dict: Robustness test results
        """
        from deepbridge.utils.robustness import run_robustness_tests
        
        return run_robustness_tests(
            dataset,
            config_name=config.get('config_name', 'quick'),
            metric=config.get('metric', self.get_default_metric()),
            verbose=verbose,
            feature_subset=feature_subset
        )
    
    @staticmethod
    def get_default_metric() -> str:
        """Get the default metric for robustness tests"""
        return "AUC"
    
    @staticmethod
    def get_configuration(config_name: str) -> dict:
        """
        Get robustness test configuration for the specified level
        
        Args:
            config_name: Configuration level ('quick', 'medium', 'full')
            
        Returns:
            dict: Robustness configuration parameters
        """
        configs = {
            'quick': {
                'perturbation_methods': ['raw', 'quantile'],
                'levels': [0.1, 0.2],
                'n_trials': 5,
                'config_name': 'quick'
            },
            'medium': {
                'perturbation_methods': ['raw', 'quantile', 'adversarial'],
                'levels': [0.05, 0.1, 0.2],
                'n_trials': 10,
                'config_name': 'medium'
            },
            'full': {
                'perturbation_methods': ['raw', 'quantile', 'adversarial', 'custom'],
                'levels': [0.01, 0.05, 0.1, 0.2, 0.3],
                'n_trials': 20,
                'config_name': 'full'
            }
        }
        
        return configs.get(config_name, configs['quick'])

class UncertaintyTestStrategy(TestStrategy):
    """Strategy for uncertainty testing"""
    
    def run_test(self, dataset, config: dict, feature_subset: t.Optional[t.List[str]] = None, 
                verbose: bool = False) -> dict:
        """
        Run uncertainty tests on a dataset
        
        Args:
            dataset: Dataset to test
            config: Test configuration
            feature_subset: Features to specifically test
            verbose: Whether to output verbose logging
            
        Returns:
            dict: Uncertainty test results
        """
        from deepbridge.utils.uncertainty import run_uncertainty_tests
        
        return run_uncertainty_tests(
            dataset,
            config_name=config.get('config_name', 'quick'),
            verbose=verbose,
            feature_subset=feature_subset
        )
    
    @staticmethod
    def get_configuration(config_name: str) -> dict:
        """
        Get uncertainty test configuration for the specified level
        
        Args:
            config_name: Configuration level ('quick', 'medium', 'full')
            
        Returns:
            dict: Uncertainty configuration parameters
        """
        configs = {
            'quick': {
                'methods': ['crqr'],
                'alpha_levels': [0.1, 0.2],
                'config_name': 'quick'
            },
            'medium': {
                'methods': ['crqr'],
                'alpha_levels': [0.05, 0.1, 0.2],
                'config_name': 'medium'
            },
            'full': {
                'methods': ['crqr'],
                'alpha_levels': [0.01, 0.05, 0.1, 0.2, 0.3],
                'config_name': 'full'
            }
        }
        
        return configs.get(config_name, configs['quick'])

class ResilienceTestStrategy(TestStrategy):
    """Strategy for resilience testing"""
    
    def run_test(self, dataset, config: dict, feature_subset: t.Optional[t.List[str]] = None, 
                verbose: bool = False) -> dict:
        """
        Run resilience tests on a dataset
        
        Args:
            dataset: Dataset to test
            config: Test configuration
            feature_subset: Features to specifically test
            verbose: Whether to output verbose logging
            
        Returns:
            dict: Resilience test results
        """
        from deepbridge.utils.resilience import run_resilience_tests
        
        return run_resilience_tests(
            dataset,
            config_name=config.get('config_name', 'quick'),
            metric=config.get('metric', self.get_default_metric()),
            verbose=verbose,
            feature_subset=feature_subset
        )
        
    @staticmethod
    def get_default_metric() -> str:
        """Get the default metric for resilience tests"""
        return "auc"
        
    @staticmethod
    def get_configuration(config_name: str) -> dict:
        """
        Get resilience test configuration for the specified level
        
        Args:
            config_name: Configuration level ('quick', 'medium', 'full')
            
        Returns:
            dict: Resilience configuration parameters
        """
        configs = {
            'quick': {
                'drift_types': ['covariate', 'label'],
                'drift_intensities': [0.1, 0.2],
                'config_name': 'quick'
            },
            'medium': {
                'drift_types': ['covariate', 'label', 'concept'],
                'drift_intensities': [0.05, 0.1, 0.2],
                'config_name': 'medium'
            },
            'full': {
                'drift_types': ['covariate', 'label', 'concept', 'temporal'],
                'drift_intensities': [0.01, 0.05, 0.1, 0.2, 0.3],
                'config_name': 'full'
            }
        }
        
        return configs.get(config_name, configs['quick'])

class HyperparameterTestStrategy(TestStrategy):
    """Strategy for hyperparameter testing"""
    
    def run_test(self, dataset, config: dict, feature_subset: t.Optional[t.List[str]] = None, 
                verbose: bool = False) -> dict:
        """
        Run hyperparameter tests on a dataset
        
        Args:
            dataset: Dataset to test
            config: Test configuration
            feature_subset: Features to specifically test
            verbose: Whether to output verbose logging
            
        Returns:
            dict: Hyperparameter test results
        """
        from deepbridge.utils.hyperparameter import run_hyperparameter_tests
        
        return run_hyperparameter_tests(
            dataset,
            config_name=config.get('config_name', 'quick'),
            metric=config.get('metric', self.get_default_metric()),
            verbose=verbose,
            feature_subset=feature_subset
        )
    
    @staticmethod
    def get_configuration(config_name: str) -> dict:
        """
        Get hyperparameter test configuration for the specified level
        
        Args:
            config_name: Configuration level ('quick', 'medium', 'full')
            
        Returns:
            dict: Hyperparameter configuration parameters
        """
        configs = {
            'quick': {
                'n_trials': 10,
                'optimization_metric': 'accuracy',
                'config_name': 'quick'
            },
            'medium': {
                'n_trials': 30,
                'optimization_metric': 'accuracy',
                'config_name': 'medium'
            },
            'full': {
                'n_trials': 100,
                'optimization_metric': 'accuracy',
                'config_name': 'full'
            }
        }
        
        return configs.get(config_name, configs['quick'])

# Factory for test strategies

class TestStrategyFactory:
    """Factory for creating test strategies"""
    
    # Registry of test strategies
    _strategies = {
        'robustness': RobustnessTestStrategy,
        'uncertainty': UncertaintyTestStrategy,
        'resilience': ResilienceTestStrategy,
        'hyperparameters': HyperparameterTestStrategy
    }
    
    @classmethod
    def create_strategy(cls, test_type: str) -> TestStrategy:
        """
        Create a test strategy of the specified type
        
        Args:
            test_type: Type of test strategy to create
            
        Returns:
            TestStrategy: Instance of the appropriate test strategy
            
        Raises:
            ValueError: If the test_type is not supported
        """
        if test_type not in cls._strategies:
            raise ValueError(f"Unsupported test type: {test_type}")
            
        return cls._strategies[test_type]()
    
    @classmethod
    def register_strategy(cls, test_type: str, strategy_class: t.Type[TestStrategy]) -> None:
        """
        Register a new test strategy
        
        Args:
            test_type: Type of test for the strategy
            strategy_class: Class for the strategy
        """
        cls._strategies[test_type] = strategy_class
    
    @classmethod
    def get_supported_types(cls) -> t.List[str]:
        """
        Get a list of supported test types
        
        Returns:
            List[str]: Supported test types
        """
        return list(cls._strategies.keys())
    
    @classmethod
    def get_configuration(cls, test_type: str, config_name: str) -> dict:
        """
        Get configuration for a specific test type and configuration level
        
        Args:
            test_type: Type of test
            config_name: Configuration level ('quick', 'medium', 'full')
            
        Returns:
            dict: Configuration parameters
            
        Raises:
            ValueError: If the test_type is not supported
        """
        if test_type not in cls._strategies:
            raise ValueError(f"Unsupported test type: {test_type}")
            
        strategy = cls._strategies[test_type]()
        return strategy.get_configuration(config_name)