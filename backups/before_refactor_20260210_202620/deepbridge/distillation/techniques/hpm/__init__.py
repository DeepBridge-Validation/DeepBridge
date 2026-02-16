"""
HPM-KD: Hierarchical Progressive Multi-Teacher Knowledge Distillation

This module implements an advanced knowledge distillation technique that combines:
- Adaptive configuration selection using Bayesian optimization
- Progressive distillation chain (simple to complex models)
- Multi-teacher ensemble with attention mechanisms
- Intelligent caching and parallel processing
"""

from .adaptive_config import AdaptiveConfigurationManager
from .cache_system import IntelligentCache
from .hpm_distiller import HPMConfig, HPMDistiller
from .meta_scheduler import MetaTemperatureScheduler
from .multi_teacher import AttentionWeightedMultiTeacher
from .parallel_pipeline import ParallelDistillationPipeline
from .progressive_chain import ProgressiveDistillationChain
from .shared_memory import SharedOptimizationMemory

__all__ = [
    'AdaptiveConfigurationManager',
    'SharedOptimizationMemory',
    'IntelligentCache',
    'ProgressiveDistillationChain',
    'AttentionWeightedMultiTeacher',
    'MetaTemperatureScheduler',
    'ParallelDistillationPipeline',
    'HPMDistiller',
    'HPMConfig',
]
