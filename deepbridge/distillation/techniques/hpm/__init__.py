"""
HPM-KD: Hierarchical Progressive Multi-Teacher Knowledge Distillation

This module implements an advanced knowledge distillation technique that combines:
- Adaptive configuration selection using Bayesian optimization
- Progressive distillation chain (simple to complex models)
- Multi-teacher ensemble with attention mechanisms
- Intelligent caching and parallel processing
"""

from .adaptive_config import AdaptiveConfigurationManager
from .shared_memory import SharedOptimizationMemory
from .cache_system import IntelligentCache
from .progressive_chain import ProgressiveDistillationChain
from .multi_teacher import AttentionWeightedMultiTeacher
from .meta_scheduler import MetaTemperatureScheduler
from .parallel_pipeline import ParallelDistillationPipeline
from .hpm_distiller import HPMDistiller, HPMConfig

__all__ = [
    'AdaptiveConfigurationManager',
    'SharedOptimizationMemory',
    'IntelligentCache',
    'ProgressiveDistillationChain',
    'AttentionWeightedMultiTeacher',
    'MetaTemperatureScheduler',
    'ParallelDistillationPipeline',
    'HPMDistiller',
    'HPMConfig'
]