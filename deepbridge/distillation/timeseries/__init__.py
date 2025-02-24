"""
Model distillation techniques and utilities.
Includes Time Series-specific implementations.
"""

from deepbridge.distillation.timeseries.ts_auto_distiller import TSConfig, TSAutoDistiller
from deepbridge.distillation.timeseries.ts_knowledge_distillation import TSKnowledgeDistillation
from deepbridge.distillation.timeseries.ts_metrics import TimeSeriesMetrics
from deepbridge.distillation.timeseries.ts_model_registry import BaseTimeSeriesAdapter, SklearnTSAdapter, ARIMATSAdapter, TSModelType, TSModelConfig, TSModelRegistry
from deepbridge.distillation.timeseries.ts_preprocessing import TimeSeriesPreprocessor


__all__ = [
    "TSConfig",
    "TSAutoDistiller",
    "TSKnowledgeDistillation",
    "TimeSeriesMetrics",
    "BaseTimeSeriesAdapter", 
    "SklearnTSAdapter", 
    "ARIMATSAdapter", 
    "TSModelType", 
    "TSModelConfig", 
    "TSModelRegistry", 
    "TimeSeriesPreprocessor"
]