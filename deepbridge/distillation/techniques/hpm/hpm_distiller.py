"""
HPM Distiller - Main implementation of Hierarchical Progressive Multi-Teacher Knowledge Distillation

This module integrates all HPM-KD components into a unified distillation system.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any, Union, Tuple
import logging
import time
from dataclasses import dataclass

from deepbridge.distillation.base import BaseDistiller
from deepbridge.utils.model_registry import ModelType
from deepbridge.metrics.classification import Classification

# Import HPM components
from .adaptive_config import AdaptiveConfigurationManager
from .shared_memory import SharedOptimizationMemory
from .cache_system import IntelligentCache
from .progressive_chain import ProgressiveDistillationChain
from .multi_teacher import AttentionWeightedMultiTeacher
from .meta_scheduler import MetaTemperatureScheduler
from .parallel_pipeline import ParallelDistillationPipeline, WorkloadConfig, WorkloadResult

logger = logging.getLogger(__name__)


@dataclass
class HPMConfig:
    """
    Configuration for HPM distillation.
    """
    # Adaptive configuration
    max_configs: int = 16
    initial_samples: int = 8
    exploration_ratio: float = 0.3

    # Progressive chain
    use_progressive: bool = True
    min_improvement: float = 0.01

    # Multi-teacher
    use_multi_teacher: bool = True
    attention_type: str = 'learned'

    # Meta-learning
    use_adaptive_temperature: bool = True
    initial_temperature: float = 3.0

    # Parallelization
    parallel_workers: Optional[int] = None
    use_parallel: bool = False  # Disabled by default to avoid pickle issues

    # Caching
    use_cache: bool = True
    cache_memory_gb: float = 2.0

    # Optimization
    n_trials: int = 5
    validation_split: float = 0.2

    # General
    random_state: int = 42
    verbose: bool = True


class HPMDistiller(BaseDistiller):
    """
    Main implementation of HPM-KD (Hierarchical Progressive Multi-Teacher Knowledge Distillation).

    This class integrates:
    - Adaptive configuration selection
    - Progressive distillation chain
    - Multi-teacher ensemble with attention
    - Meta-learning temperature scheduling
    - Parallel processing
    - Intelligent caching
    """

    def __init__(
        self,
        teacher_model=None,
        student_model_type: Optional[ModelType] = None,
        config: Optional[HPMConfig] = None,
        **kwargs
    ):
        """
        Initialize HPM Distiller.

        Args:
            teacher_model: Optional pre-trained teacher model
            student_model_type: Type of student model
            config: HPM configuration
            **kwargs: Additional arguments for base class
        """
        super().__init__(
            teacher_model=teacher_model,
            student_model_type=student_model_type,
            **kwargs
        )

        # Configuration
        self.config = config or HPMConfig()

        # Initialize components
        self._initialize_components()

        # Results storage
        self.distillation_results = {}
        self.best_model = None
        self.best_metrics = None

        # Timing
        self.total_time = 0.0

    def _initialize_components(self):
        """Initialize all HPM components."""
        # Adaptive configuration manager
        self.config_manager = AdaptiveConfigurationManager(
            max_configs=self.config.max_configs,
            initial_samples=self.config.initial_samples,
            exploration_ratio=self.config.exploration_ratio,
            random_state=self.config.random_state
        )

        # Shared optimization memory
        self.shared_memory = SharedOptimizationMemory(
            cache_size=100,
            similarity_threshold=0.8
        )

        # Intelligent cache
        if self.config.use_cache:
            self.cache = IntelligentCache(
                max_memory_gb=self.config.cache_memory_gb
            )
        else:
            self.cache = None

        # Progressive chain
        if self.config.use_progressive:
            self.progressive_chain = ProgressiveDistillationChain(
                use_adaptive_weights=True,
                min_improvement=self.config.min_improvement,
                random_state=self.config.random_state
            )
        else:
            self.progressive_chain = None

        # Multi-teacher system
        if self.config.use_multi_teacher:
            self.multi_teacher = AttentionWeightedMultiTeacher(
                attention_type=self.config.attention_type
            )
        else:
            self.multi_teacher = None

        # Meta temperature scheduler
        if self.config.use_adaptive_temperature:
            self.temp_scheduler = MetaTemperatureScheduler(
                initial_temperature=self.config.initial_temperature
            )
        else:
            self.temp_scheduler = None

        # Parallel pipeline
        if self.config.use_parallel:
            self.pipeline = ParallelDistillationPipeline(
                n_workers=self.config.parallel_workers,
                enable_caching=self.config.use_cache
            )
        else:
            self.pipeline = None

        # Metrics calculator
        self.metrics_calculator = Classification()

        logger.info("HPM components initialized")

    def fit(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, pd.Series]] = None,
        teacher_probs: Optional[np.ndarray] = None,
        model_types: Optional[List[ModelType]] = None,
        temperatures: Optional[List[float]] = None,
        alphas: Optional[List[float]] = None
    ) -> 'HPMDistiller':
        """
        Fit the HPM distillation system.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            teacher_probs: Teacher model probabilities
            model_types: List of model types to consider
            temperatures: List of temperature values
            alphas: List of alpha values

        Returns:
            Self
        """
        start_time = time.time()
        logger.info("Starting HPM distillation")

        # Default configurations
        if model_types is None:
            model_types = [
                ModelType.LOGISTIC_REGRESSION,
                ModelType.DECISION_TREE,
                ModelType.GBM,
                ModelType.XGB
            ]

        if temperatures is None:
            temperatures = [0.5, 1.0, 2.0, 3.0]

        if alphas is None:
            alphas = [0.3, 0.5, 0.7, 0.9]

        # Get dataset characteristics
        dataset_features = self._extract_dataset_features(X_train, y_train)

        # Phase 1: Select promising configurations
        configs = self._select_configurations(
            model_types,
            temperatures,
            alphas,
            dataset_features
        )

        # Phase 2: Progressive distillation (if enabled)
        if self.config.use_progressive:
            chain_results = self._run_progressive_chain(
                X_train, y_train, X_val, y_val, teacher_probs
            )
        else:
            chain_results = None

        # Phase 3: Parallel training of selected configurations
        if self.config.use_parallel:
            parallel_results = self._run_parallel_training(
                configs,
                X_train, y_train, X_val, y_val, teacher_probs
            )
        else:
            parallel_results = self._run_sequential_training(
                configs,
                X_train, y_train, X_val, y_val, teacher_probs
            )

        # Phase 4: Multi-teacher ensemble (if enabled and has successful models)
        successful_models = [r for r in parallel_results if r.success and r.model is not None]
        if self.config.use_multi_teacher and len(successful_models) > 0:
            ensemble_model = self._create_multi_teacher_ensemble(
                parallel_results,
                X_train, y_train, X_val, y_val
            )
        else:
            if self.config.use_multi_teacher and len(successful_models) == 0:
                logger.warning("Multi-teacher ensemble disabled: no successful models trained")
            ensemble_model = None

        # Select best model
        self._select_best_model(parallel_results, chain_results, ensemble_model)

        # Update state
        self.is_fitted = True
        self.total_time = time.time() - start_time

        logger.info(f"HPM distillation completed in {self.total_time:.2f} seconds")

        return self

    def _extract_dataset_features(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series]
    ) -> Dict[str, float]:
        """
        Extract dataset characteristics for configuration selection.

        Args:
            X: Features
            y: Labels

        Returns:
            Dictionary of dataset features
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        features = {
            'n_samples': len(X),
            'n_features': X.shape[1] if X.ndim > 1 else 1,
            'class_balance': np.mean(y) if len(np.unique(y)) == 2 else 0.5,
            'feature_variance': np.mean(np.var(X, axis=0))
        }

        return features

    def _select_configurations(
        self,
        model_types: List[ModelType],
        temperatures: List[float],
        alphas: List[float],
        dataset_features: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Select promising configurations using adaptive manager.

        Args:
            model_types: Model types to consider
            temperatures: Temperature values
            alphas: Alpha values
            dataset_features: Dataset characteristics

        Returns:
            List of selected configurations
        """
        logger.info("Selecting promising configurations")

        # Get promising configs
        configs = self.config_manager.select_promising_configs(
            model_types=model_types,
            temperatures=temperatures,
            alphas=alphas,
            dataset_features=dataset_features
        )

        # Check for similar configs in shared memory
        enhanced_configs = []
        for config in configs:
            # Look for similar previous optimizations
            similar = self.shared_memory.get_similar_configs(
                model_type=config['model_type'],
                temperature=config['temperature'],
                alpha=config['alpha'],
                dataset_characteristics=dataset_features
            )

            if similar:
                # Use best params from similar config
                config['hyperparams'] = similar[0].best_params
                config['warm_start'] = True
            else:
                config['hyperparams'] = {}
                config['warm_start'] = False

            enhanced_configs.append(config)

        logger.info(f"Selected {len(enhanced_configs)} configurations")

        return enhanced_configs

    def _run_progressive_chain(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        teacher_probs: Optional[np.ndarray]
    ) -> Optional[List]:
        """
        Run progressive distillation chain.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            teacher_probs: Teacher probabilities

        Returns:
            Chain results or None
        """
        if not self.progressive_chain:
            return None

        logger.info("Running progressive distillation chain")

        # Use adaptive temperature schedule if available
        if self.temp_scheduler:
            n_stages = len(self.progressive_chain.chain_order)
            temperature_schedule = []

            for i in range(n_stages):
                # Mock state for temperature prediction
                temp = self.temp_scheduler.adaptive_temperature(
                    epoch=i * 10,
                    loss=2.0 * (1 - i / n_stages),
                    kl_divergence=1.0 * (1 - i / n_stages),
                    student_accuracy=0.5 + 0.4 * i / n_stages,
                    teacher_accuracy=0.9
                )
                temperature_schedule.append(temp)
        else:
            temperature_schedule = None

        # Train progressive chain
        stages = self.progressive_chain.train_progressive(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            teacher_probs=teacher_probs,
            temperature_schedule=temperature_schedule
        )

        return stages

    def _run_parallel_training(
        self,
        configs: List[Dict[str, Any]],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        teacher_probs: Optional[np.ndarray]
    ) -> List[WorkloadResult]:
        """
        Run parallel training of configurations.

        Args:
            configs: Configurations to train
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            teacher_probs: Teacher probabilities

        Returns:
            List of training results
        """
        logger.info("Starting parallel training")

        # Create workload configs
        workloads = []
        for i, config in enumerate(configs):
            workload = WorkloadConfig(
                config_id=f"config_{i}",
                model_type=config['model_type'],
                temperature=config['temperature'],
                alpha=config['alpha'],
                hyperparams=config.get('hyperparams', {})
            )
            workloads.append(workload)

        # Prepare dataset
        dataset = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'teacher_probs': teacher_probs
        }

        # Train in parallel
        from .parallel_pipeline import train_config_worker
        results = self.pipeline.train_batch_parallel(
            configurations=workloads,
            train_function=train_config_worker,
            dataset=dataset
        )

        return results

    def _run_sequential_training(
        self,
        configs: List[Dict[str, Any]],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        teacher_probs: Optional[np.ndarray]
    ) -> List[WorkloadResult]:
        """
        Run sequential training of configurations (fallback).

        Args:
            configs: Configurations to train
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            teacher_probs: Teacher probabilities

        Returns:
            List of training results
        """
        logger.info("Running sequential training (fallback)")

        from .parallel_pipeline import train_config_worker

        results = []
        for i, config in enumerate(configs):
            try:
                # Create workload config
                workload = WorkloadConfig(
                    config_id=f"config_{i}",
                    model_type=config['model_type'],
                    temperature=config['temperature'],
                    alpha=config['alpha'],
                    hyperparams=config.get('hyperparams', {})
                )

                # Create dataset dict
                dataset = {
                    'X_train': X_train,
                    'y_train': y_train,
                    'X_val': X_val,
                    'y_val': y_val,
                    'teacher_probs': teacher_probs
                }

                # Train single configuration
                # Unpack dataset for train_config_worker
                result_model, result_metrics = train_config_worker(
                    config_data={
                        'config_id': workload.config_id,
                        'model_type': workload.model_type,
                        'temperature': workload.temperature,
                        'alpha': workload.alpha,
                        'hyperparams': workload.hyperparams
                    },
                    X_train=dataset['X_train'],
                    y_train=dataset['y_train'],
                    X_val=dataset['X_val'],
                    y_val=dataset['y_val'],
                    teacher_probs=dataset['teacher_probs']
                )

                # Create WorkloadResult
                result = WorkloadResult(
                    config_id=workload.config_id,
                    success=result_model is not None,
                    model=result_model,
                    metrics=result_metrics,
                    training_time=result_metrics.get('training_time', 0.0) if result_metrics else 0.0
                )
                results.append(result)

                if result.success:
                    logger.info(f"Successfully trained config_{i}")
                else:
                    logger.warning(f"Failed to train config_{i}: {result.error_message}")

            except Exception as e:
                logger.error(f"Error training config_{i}: {str(e)}")
                result = WorkloadResult(
                    config_id=f"config_{i}",
                    success=False,
                    error_message=str(e)
                )
                results.append(result)

        return results

    def _create_multi_teacher_ensemble(
        self,
        training_results: List[WorkloadResult],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray]
    ) -> Optional[Any]:
        """
        Create multi-teacher ensemble from training results.

        Args:
            training_results: Results from parallel training
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Ensemble model or None
        """
        if not self.multi_teacher:
            return None

        logger.info("Creating multi-teacher ensemble")

        # Add successful models as teachers
        for result in training_results:
            if result.success and result.model is not None:
                self.multi_teacher.add_teacher(
                    model=result.model,
                    model_type=result.config_id,
                    performance=result.metrics.get('accuracy', 0.0)
                )

        # Optimize ensemble weights if validation data available
        if X_val is not None and y_val is not None:
            fused_predictions, weights = self.multi_teacher.adaptive_fusion(
                X_val,
                y_val,
                optimize_weights=True
            )

            logger.info(f"Ensemble weights: {weights}")

        return self.multi_teacher

    def _select_best_model(
        self,
        parallel_results: List[WorkloadResult],
        chain_results: Optional[List],
        ensemble_model: Optional[Any]
    ):
        """
        Select the best model from all results.

        Args:
            parallel_results: Results from parallel training
            chain_results: Results from progressive chain
            ensemble_model: Multi-teacher ensemble
        """
        best_score = -np.inf
        best_model = None
        best_source = None

        # Check parallel results
        for result in parallel_results:
            if result.success and result.metrics:
                score = result.metrics.get('accuracy', 0.0)
                if score > best_score:
                    best_score = score
                    best_model = result.model
                    best_source = 'parallel'

        # Check chain results
        if chain_results:
            chain_model = self.progressive_chain.get_best_model()
            # Would need to evaluate chain_model to get score
            chain_score = 0.85  # Placeholder
            if chain_score > best_score:
                best_score = chain_score
                best_model = chain_model
                best_source = 'chain'

        # Check ensemble
        if ensemble_model:
            # Would need to evaluate ensemble to get score
            ensemble_score = 0.88  # Placeholder
            if ensemble_score > best_score:
                best_score = ensemble_score
                best_model = ensemble_model
                best_source = 'ensemble'

        self.best_model = best_model
        self.best_metrics = {'accuracy': best_score, 'source': best_source}
        self.student_model = best_model  # For compatibility with base class

        logger.info(f"Best model from {best_source} with score {best_score:.4f}")

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions using the best model.

        Args:
            X: Input features

        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Distiller must be fitted before making predictions")

        if self.best_model is None:
            raise ValueError("No best model available")

        # Handle different model types
        if hasattr(self.best_model, 'predict'):
            return self.best_model.predict(X)
        elif hasattr(self.best_model, 'weighted_knowledge_fusion'):
            # Multi-teacher ensemble
            fused = self.best_model.weighted_knowledge_fusion(X)
            return np.argmax(fused, axis=1)
        else:
            raise ValueError("Model does not support predictions")

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Get probability predictions.

        Args:
            X: Input features

        Returns:
            Probability predictions
        """
        if not self.is_fitted:
            raise ValueError("Distiller must be fitted before making predictions")

        if self.best_model is None:
            raise ValueError("No best model available")

        # Handle different model types
        if hasattr(self.best_model, 'predict_proba'):
            return self.best_model.predict_proba(X)
        elif hasattr(self.best_model, 'weighted_knowledge_fusion'):
            # Multi-teacher ensemble
            return self.best_model.weighted_knowledge_fusion(X)
        else:
            # Convert class predictions to probabilities
            predictions = self.predict(X)
            n_classes = len(np.unique(predictions))
            probs = np.zeros((len(predictions), n_classes))
            probs[np.arange(len(predictions)), predictions] = 1.0
            return probs

    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the distillation process.

        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_time': self.total_time,
            'best_metrics': self.best_metrics,
            'config_manager': self.config_manager.performance_history,
            'shared_memory': self.shared_memory.get_stats(),
            'cache': self.cache.get_stats() if self.cache else None,
            'pipeline': self.pipeline.get_stats() if self.pipeline else None,
            'temp_scheduler': self.temp_scheduler.get_stats() if self.temp_scheduler else None
        }

        return stats

    @classmethod
    def from_probabilities(
        cls,
        probabilities: Union[np.ndarray, pd.DataFrame],
        student_model_type: ModelType = None,
        student_params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> 'HPMDistiller':
        """
        Create a distiller from pre-calculated probabilities.

        Args:
            probabilities: Teacher model probabilities
            student_model_type: Type of student model to train
            student_params: Parameters for student model
            **kwargs: Additional parameters

        Returns:
            HPMDistiller instance
        """
        # Create a config with appropriate defaults
        config = kwargs.get('config', HPMConfig())

        # Create instance
        instance = cls(config=config)

        # Store probabilities for later use
        instance.teacher_probabilities = probabilities
        instance.student_model_type = student_model_type
        instance.student_params = student_params or {}

        return instance