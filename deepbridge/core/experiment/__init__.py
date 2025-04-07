"""
Core experiment module for model validation and testing.
This package provides a standard interface for running experiments on ML models.
"""

from deepbridge.core.experiment.dependencies import check_dependencies, print_dependency_status
from deepbridge.core.experiment.experiment import Experiment

# Try to import new interfaces and implementations
try:
    from deepbridge.core.experiment.interfaces import (
        IExperiment, ITestRunner, IVisualizationManager, IReportGenerator, 
        TestResult, ModelResult
    )
    from deepbridge.core.experiment.results import (
        ExperimentResult, RobustnessResult, UncertaintyResult, 
        ResilienceResult, HyperparameterResult, wrap_results
    )
    from deepbridge.core.experiment.runner import TestRunner
    from deepbridge.core.experiment.visualization import VisualizationManager
    
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
            'Experiment', 'TestRunner', 'VisualizationManager',
            'IExperiment', 'ITestRunner', 'IVisualizationManager', 'IReportGenerator',
            'TestResult', 'ModelResult',
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
    
    # For backward compatibility, import these from report_exporter if available
    try:
        from deepbridge.core.experiment.report_exporter import export_report, wrap_results, ResultsDict
        __all__.extend(['export_report', 'wrap_results', 'ResultsDict'])
    except ImportError:
        # Create dummy classes for backward compatibility
        def export_report(*args, **kwargs):
            """Placeholder for export_report when dependencies are missing."""
            print("Error: Unable to export report due to missing dependencies.")
            print_dependency_status()
            return None
        
        class ResultsDict(dict):
            """Dummy implementation when dependencies are missing."""
            def save_report(self, *args, **kwargs):
                print("Error: Unable to save report due to missing dependencies.")
                print_dependency_status()
                return None
        
        def wrap_results(results):
            """Dummy implementation when dependencies are missing."""
            print("Warning: Report generation functionality is limited due to missing dependencies.")
            print_dependency_status()
            return ResultsDict(results)
            
        __all__.extend(['export_report', 'wrap_results', 'ResultsDict'])
