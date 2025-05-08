"""
Core experiment module for model validation and testing.
This package provides a standard interface for running experiments on ML models.
"""

try:
    from deepbridge.core.experiment.dependencies import check_dependencies, print_dependency_status
    from deepbridge.core.experiment.experiment import Experiment
except ImportError:
    from core.experiment.dependencies import check_dependencies, print_dependency_status
    from core.experiment.experiment import Experiment

# Import report manager if available
try:
    try:
        from deepbridge.core.experiment.report.report_manager import ReportManager
    except ImportError:
        from core.experiment.report.report_manager import ReportManager
except ImportError:
    ReportManager = None


# Use relative path with os.path for cross-platform compatibility
import os

# Get the base directory of the package
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
templates_dir = os.path.join(base_dir, 'templates')
report_manager = ReportManager(templates_dir=templates_dir)



# Try to import new interfaces and implementations
try:
    from deepbridge.core.experiment.interfaces import (
        IExperiment, ITestRunner, TestResult, ModelResult
    )
    from deepbridge.core.experiment.results import (
        ExperimentResult, RobustnessResult, UncertaintyResult, 
        ResilienceResult, HyperparameterResult, wrap_results
    )
    from deepbridge.core.experiment.runner import TestRunner
    
    # Import new model result classes
    try:
        from deepbridge.core.experiment.model_result import (
            BaseModelResult, ClassificationModelResult, RegressionModelResult, 
            create_model_result
        )
    except ImportError:
        pass
        
    # Import result factory
    try:
        from deepbridge.core.experiment.test_result_factory import TestResultFactory
    except ImportError:
        pass
    
    # Check if all dependencies are available
    all_required_installed, missing_required, missing_optional, version_issues = check_dependencies()
    
    if all_required_installed:
        __all__ = [
            'Experiment', 'TestRunner',
            'IExperiment', 'ITestRunner', 'TestResult', 'ModelResult',
            'ExperimentResult', 'RobustnessResult', 'UncertaintyResult', 
            'ResilienceResult', 'HyperparameterResult', 'wrap_results',
            'check_dependencies', 'print_dependency_status'
        ]
        
        # Add model result classes if available
        try:
            if 'BaseModelResult' in globals():
                __all__.extend([
                    'BaseModelResult', 'ClassificationModelResult', 'RegressionModelResult',
                    'create_model_result'
                ])
        except:
            pass
            
        # Add TestResultFactory if available
        if 'TestResultFactory' in globals():
            __all__.append('TestResultFactory')
            
        # Import strategy and manager factories if available
        try:
            from deepbridge.core.experiment.test_strategies import (
                TestStrategy, TestStrategyFactory, 
                RobustnessTestStrategy, UncertaintyTestStrategy,
                ResilienceTestStrategy, HyperparameterTestStrategy
            )
            from deepbridge.core.experiment.manager_factory import ManagerFactory
            
            # Add to __all__
            __all__.extend([
                'TestStrategy', 'TestStrategyFactory', 'ManagerFactory',
                'RobustnessTestStrategy', 'UncertaintyTestStrategy',
                'ResilienceTestStrategy', 'HyperparameterTestStrategy'
            ])
        except ImportError:
            pass
    else:
        # Reduced functionality when dependencies are missing
        __all__ = ['Experiment', 'check_dependencies', 'print_dependency_status']

except ImportError as e:
    # Fallback to basic functionality
    print(f"Warning: Some experiment functionality is not available: {str(e)}")
    __all__ = ['Experiment', 'check_dependencies', 'print_dependency_status']
    
    # Create dummy functions for backward compatibility
    def wrap_results(results):
        """Dummy implementation when dependencies are missing."""
        print("Warning: Report generation functionality has been removed.")
        return results
            
    __all__.extend(['wrap_results'])
