"""
Report generation package for experiment results.
Provides functionality for generating HTML reports from experiment results.

**NEW (Recommended):** Use the unified `ReportGenerator` API for all report types.
See MIGRATION_GUIDE_REPORT_GENERATION.md for migration instructions.

Example:
    >>> from deepbridge.core.experiment.report import ReportGenerator, RenderConfig
    >>> generator = ReportGenerator()
    >>> generator.generate_robustness_report(results, output_path=Path("report.html"))

**OLD (Deprecated):** Individual renderer classes (RobustnessRenderer, etc.) are deprecated
and will be removed in a future version.
"""

# ==============================================================================
# NEW UNIFIED API (✅ Recommended)
# ==============================================================================
from .api import ReportGenerator
from .config import OutputFormat, RenderConfig, ReportStyle, get_preset_config

# New data layer (type-safe dataclasses)
from .data.base import DataTransformer as BaseDataTransformer
from .data.base import MetricValue, ModelResult, ReportData
from .data.fairness import FairnessDataTransformer, FairnessReportData
from .data.resilience import ResilienceDataTransformer, ResilienceReportData
from .data.robustness import RobustnessDataTransformer, RobustnessReportData
from .data.uncertainty import UncertaintyDataTransformer, UncertaintyReportData

# New renderers
from .renderers.html import HTMLRenderer, HTMLRendererWithAssets
from .renderers.json import JSONLinesRenderer, JSONRenderer

# Template engine
from .templates import TemplateEngine

# ==============================================================================
# OLD API (⚠️ Deprecated - will be removed in future versions)
# ==============================================================================
import warnings

from .asset_manager import AssetManager
from .asset_processor import AssetProcessor

# Phase 4: Async generation
from .async_generator import (
    AsyncReportGenerator,
    ExecutorType,
    ProgressTracker,
    ReportTask,
    TaskStatus,
    generate_report_async,
    generate_reports_async,
)
from .base import DataTransformer
from .data_integration import DataIntegrationManager
from .file_discovery import FileDiscoveryManager
from .js_syntax_fixer import JavaScriptSyntaxFixer
from .report_manager import ReportManager
from .template_manager import TemplateManager
from .transformers import (
    HyperparameterDataTransformer,
    ResilienceDataTransformer as OldResilienceDataTransformer,
    RobustnessDataTransformer as OldRobustnessDataTransformer,
    UncertaintyDataTransformer as OldUncertaintyDataTransformer,
)


# Factory function to get the appropriate transformer for a report type (OLD API)
def get_transformer(report_type):
    """
    Get the appropriate data transformer for a specific report type.

    **DEPRECATED:** This function uses the old transformer API.
    Use the new ReportGenerator API instead.

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
    warnings.warn(
        "get_transformer() is deprecated. Use ReportGenerator API instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    transformers = {
        'robustness': OldRobustnessDataTransformer,
        'uncertainty': OldUncertaintyDataTransformer,
        'resilience': OldResilienceDataTransformer,
        'hyperparameter': HyperparameterDataTransformer,
    }

    if report_type.lower() not in transformers:
        raise ValueError(
            f'Unsupported report type: {report_type}. '
            + f"Supported types are: {', '.join(transformers.keys())}"
        )

    return transformers[report_type.lower()]()


__all__ = [
    # ==============================================================================
    # NEW UNIFIED API (✅ Recommended)
    # ==============================================================================
    'ReportGenerator',
    'RenderConfig',
    'ReportStyle',
    'OutputFormat',
    'get_preset_config',
    # New data layer
    'BaseDataTransformer',
    'ReportData',
    'ModelResult',
    'MetricValue',
    'RobustnessDataTransformer',
    'RobustnessReportData',
    'ResilienceDataTransformer',
    'ResilienceReportData',
    'UncertaintyDataTransformer',
    'UncertaintyReportData',
    'FairnessDataTransformer',
    'FairnessReportData',
    # New renderers
    'HTMLRenderer',
    'HTMLRendererWithAssets',
    'JSONRenderer',
    'JSONLinesRenderer',
    # Template engine
    'TemplateEngine',
    # ==============================================================================
    # OLD API (⚠️ Deprecated - will be removed in future versions)
    # ==============================================================================
    'DataTransformer',
    'OldRobustnessDataTransformer',
    'OldUncertaintyDataTransformer',
    'OldResilienceDataTransformer',
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
