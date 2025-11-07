"""
Report generation package for experiment results.
Provides functionality for generating HTML reports from experiment results.

Phase 4 includes multi-format adapters (PDF, Markdown) and async batch generation.
"""

from .base import DataTransformer
from .transformers import (
    RobustnessDataTransformer,
    UncertaintyDataTransformer,
    ResilienceDataTransformer,
    HyperparameterDataTransformer
)
from .report_manager import ReportManager
from .asset_manager import AssetManager
from .file_discovery import FileDiscoveryManager
from .asset_processor import AssetProcessor
from .data_integration import DataIntegrationManager
from .template_manager import TemplateManager
from .js_syntax_fixer import JavaScriptSyntaxFixer

# Phase 4: Async generation
from .async_generator import (
    AsyncReportGenerator,
    ReportTask,
    ProgressTracker,
    ExecutorType,
    TaskStatus,
    generate_report_async,
    generate_reports_async
)

# Factory function to get the appropriate transformer for a report type
def get_transformer(report_type):
    """
    Get the appropriate data transformer for a specific report type.
    
    Parameters:
    -----------
    report_type : str
        Type of report ('robustness', 'uncertainty', 'resilience', 'hyperparameter')
        
    Returns:
    --------
    DataTransformer : Instance of the appropriate transformer
    
    Raises:
    -------
    ValueError : If an unsupported report type is requested
    """
    transformers = {
        'robustness': RobustnessDataTransformer,
        'uncertainty': UncertaintyDataTransformer,
        'resilience': ResilienceDataTransformer,
        'hyperparameter': HyperparameterDataTransformer
    }
    
    if report_type.lower() not in transformers:
        raise ValueError(f"Unsupported report type: {report_type}. " +
                         f"Supported types are: {', '.join(transformers.keys())}")
    
    return transformers[report_type.lower()]()

__all__ = [
    'DataTransformer',
    'RobustnessDataTransformer',
    'UncertaintyDataTransformer',
    'ResilienceDataTransformer',
    'HyperparameterDataTransformer',
    'ReportManager',
    'AssetManager',
    'FileDiscoveryManager',
    'AssetProcessor',
    'DataIntegrationManager',
    'TemplateManager',
    'JavaScriptSyntaxFixer',
    'get_transformer',
    # Phase 4 async generation
    'AsyncReportGenerator',
    'ReportTask',
    'ProgressTracker',
    'ExecutorType',
    'TaskStatus',
    'generate_report_async',
    'generate_reports_async',
]