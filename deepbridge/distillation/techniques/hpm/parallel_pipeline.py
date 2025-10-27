"""
Parallel Distillation Pipeline for HPM-KD

This module implements efficient parallel processing for training multiple
distillation configurations simultaneously.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count, Manager
import asyncio
import time
import logging
from dataclasses import dataclass
import pickle
import traceback
from functools import partial

logger = logging.getLogger(__name__)


@dataclass
class WorkloadConfig:
    """
    Configuration for a single training workload.
    """
    config_id: str
    model_type: Any
    temperature: float
    alpha: float
    hyperparams: Dict[str, Any]
    priority: int = 1
    estimated_time: Optional[float] = None


@dataclass
class WorkloadResult:
    """
    Result from a training workload.
    """
    config_id: str
    success: bool
    model: Optional[Any] = None
    metrics: Optional[Dict[str, float]] = None
    training_time: float = 0.0
    error_message: Optional[str] = None


class ParallelDistillationPipeline:
    """
    Pipeline for parallel training of distillation configurations.

    This class manages distributed training across multiple CPU cores,
    with intelligent load balancing and progress tracking.
    """

    def __init__(
        self,
        n_workers: Optional[int] = None,
        use_processes: bool = True,
        batch_size: int = 4,
        timeout_per_config: float = 300.0,
        enable_caching: bool = True
    ):
        """
        Initialize the parallel pipeline.

        Args:
            n_workers: Number of parallel workers (None for auto)
            use_processes: Use processes (True) or threads (False)
            batch_size: Size of configuration batches
            timeout_per_config: Timeout per configuration in seconds
            enable_caching: Enable result caching
        """
        self.n_workers = n_workers or max(1, cpu_count() - 1)
        self.use_processes = use_processes
        self.batch_size = batch_size
        self.timeout_per_config = timeout_per_config
        self.enable_caching = enable_caching

        # Executor management
        self.executor = None
        self._executor_type = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

        # Results storage
        self.results_cache = {} if enable_caching else None

        # Progress tracking
        self.total_configs = 0
        self.completed_configs = 0
        self.failed_configs = 0
        self.start_time = None

        # Performance statistics
        self.timing_stats = {
            'total_time': 0.0,
            'average_time_per_config': 0.0,
            'parallel_efficiency': 0.0
        }

        logger.info(f"Initialized parallel pipeline with {self.n_workers} workers")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()

    def start(self):
        """Start the parallel pipeline."""
        if self.executor is None:
            self.executor = self._executor_type(max_workers=self.n_workers)
            self.start_time = time.time()
            logger.info("Parallel pipeline started")

    def shutdown(self):
        """Shutdown the parallel pipeline."""
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None

            # Calculate final statistics
            if self.start_time:
                self.timing_stats['total_time'] = time.time() - self.start_time
                if self.completed_configs > 0:
                    self.timing_stats['average_time_per_config'] = (
                        self.timing_stats['total_time'] / self.completed_configs
                    )

            logger.info("Parallel pipeline shutdown")

    def train_batch_parallel(
        self,
        configurations: List[WorkloadConfig],
        train_function: Callable,
        dataset: Any,
        progress_callback: Optional[Callable] = None
    ) -> List[WorkloadResult]:
        """
        Train multiple configurations in parallel.

        Args:
            configurations: List of configurations to train
            train_function: Function to train a single configuration
            dataset: Dataset to use for training
            progress_callback: Optional callback for progress updates

        Returns:
            List of training results
        """
        if not self.executor:
            self.start()

        self.total_configs = len(configurations)
        self.completed_configs = 0
        self.failed_configs = 0

        logger.info(f"Starting parallel training of {self.total_configs} configurations")

        # Balance workloads
        balanced_workloads = self._balance_workloads(configurations)

        # Submit all tasks
        futures = []
        for workload_batch in balanced_workloads:
            for config in workload_batch:
                # Check cache first
                if self.enable_caching and config.config_id in self.results_cache:
                    logger.debug(f"Using cached result for {config.config_id}")
                    continue

                # Submit training task
                future = self.executor.submit(
                    self._train_single_config,
                    config,
                    train_function,
                    dataset
                )
                futures.append((future, config))

        # Collect results as they complete
        results = []
        for future, config in futures:
            try:
                # Wait with timeout
                result = future.result(timeout=self.timeout_per_config)
                results.append(result)

                # Update progress
                self.completed_configs += 1
                if progress_callback:
                    progress_callback(self.completed_configs, self.total_configs)

                # Cache result
                if self.enable_caching:
                    self.results_cache[config.config_id] = result

                logger.debug(f"Completed {config.config_id} "
                           f"({self.completed_configs}/{self.total_configs})")

            except Exception as e:
                # Handle failed configuration
                self.failed_configs += 1
                error_msg = f"Failed to train {config.config_id}: {str(e)}"
                logger.error(error_msg)

                results.append(WorkloadResult(
                    config_id=config.config_id,
                    success=False,
                    error_message=error_msg
                ))

        # Calculate parallel efficiency
        self._calculate_efficiency(results)

        return results

    async def train_batch_async(
        self,
        configurations: List[WorkloadConfig],
        train_function: Callable,
        dataset: Any,
        progress_callback: Optional[Callable] = None
    ) -> List[WorkloadResult]:
        """
        Train configurations asynchronously.

        Args:
            configurations: List of configurations to train
            train_function: Function to train a single configuration
            dataset: Dataset to use for training
            progress_callback: Optional callback for progress updates

        Returns:
            List of training results
        """
        loop = asyncio.get_event_loop()

        # Create executor if needed
        if not self.executor:
            self.start()

        # Create async tasks
        tasks = []
        for config in configurations:
            task = loop.run_in_executor(
                self.executor,
                self._train_single_config,
                config,
                train_function,
                dataset
            )
            tasks.append(task)

        # Wait for all tasks with progress tracking
        results = []
        for i, task in enumerate(asyncio.as_completed(tasks)):
            result = await task
            results.append(result)

            # Update progress
            self.completed_configs = i + 1
            if progress_callback:
                progress_callback(self.completed_configs, len(configurations))

        return results

    def _train_single_config(
        self,
        config: WorkloadConfig,
        train_function: Callable,
        dataset: Any
    ) -> WorkloadResult:
        """
        Train a single configuration.

        Args:
            config: Configuration to train
            train_function: Training function
            dataset: Training dataset

        Returns:
            Training result
        """
        start_time = time.time()

        try:
            # Execute training
            model, metrics = train_function(
                model_type=config.model_type,
                temperature=config.temperature,
                alpha=config.alpha,
                hyperparams=config.hyperparams,
                dataset=dataset
            )

            # Calculate training time
            training_time = time.time() - start_time

            return WorkloadResult(
                config_id=config.config_id,
                success=True,
                model=model,
                metrics=metrics,
                training_time=training_time
            )

        except Exception as e:
            # Log full traceback for debugging
            logger.error(f"Error training {config.config_id}: {traceback.format_exc()}")

            return WorkloadResult(
                config_id=config.config_id,
                success=False,
                training_time=time.time() - start_time,
                error_message=str(e)
            )

    def _balance_workloads(
        self,
        configurations: List[WorkloadConfig]
    ) -> List[List[WorkloadConfig]]:
        """
        Balance workloads across workers based on estimated time.

        Args:
            configurations: List of configurations

        Returns:
            Balanced batches of configurations
        """
        # Sort by priority and estimated time
        sorted_configs = sorted(
            configurations,
            key=lambda x: (-x.priority, x.estimated_time or 0)
        )

        # Create balanced batches
        batches = [[] for _ in range(self.n_workers)]
        batch_times = [0.0] * self.n_workers

        for config in sorted_configs:
            # Find batch with minimum total time
            min_idx = np.argmin(batch_times)
            batches[min_idx].append(config)

            # Update estimated time
            batch_times[min_idx] += config.estimated_time or 60.0  # Default 60s

        # Filter empty batches
        batches = [b for b in batches if b]

        logger.debug(f"Created {len(batches)} balanced batches")

        return batches

    def _calculate_efficiency(self, results: List[WorkloadResult]):
        """
        Calculate parallel processing efficiency.

        Args:
            results: List of training results
        """
        if not results:
            return

        # Calculate total sequential time
        sequential_time = sum(r.training_time for r in results)

        # Actual parallel time
        parallel_time = self.timing_stats['total_time']

        if parallel_time > 0:
            # Efficiency = (sequential_time) / (parallel_time * n_workers)
            self.timing_stats['parallel_efficiency'] = (
                sequential_time / (parallel_time * self.n_workers)
            )

            speedup = sequential_time / parallel_time
            logger.info(f"Parallel efficiency: {self.timing_stats['parallel_efficiency']:.2%}")
            logger.info(f"Speedup: {speedup:.2f}x")

    def map_reduce(
        self,
        map_func: Callable,
        reduce_func: Callable,
        data: List[Any]
    ) -> Any:
        """
        Parallel map-reduce operation.

        Args:
            map_func: Function to map over data
            reduce_func: Function to reduce results
            data: Input data

        Returns:
            Reduced result
        """
        if not self.executor:
            self.start()

        # Map phase
        futures = [self.executor.submit(map_func, item) for item in data]

        # Collect results
        mapped_results = []
        for future in as_completed(futures):
            try:
                result = future.result(timeout=self.timeout_per_config)
                mapped_results.append(result)
            except Exception as e:
                logger.error(f"Map operation failed: {e}")

        # Reduce phase
        if mapped_results:
            return reduce_func(mapped_results)
        return None

    def parallel_evaluate(
        self,
        models: List[Any],
        X_test: Union[np.ndarray, pd.DataFrame],
        y_test: Union[np.ndarray, pd.Series],
        metrics_func: Callable
    ) -> List[Dict[str, float]]:
        """
        Evaluate multiple models in parallel.

        Args:
            models: List of models to evaluate
            X_test: Test features
            y_test: Test labels
            metrics_func: Function to calculate metrics

        Returns:
            List of metrics for each model
        """
        def evaluate_single(model):
            predictions = model.predict(X_test)
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_test)
            else:
                probabilities = None

            return metrics_func(y_test, predictions, probabilities)

        # Use map operation
        return self.map_reduce(
            evaluate_single,
            lambda x: x,  # Identity reduce
            models
        )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get pipeline statistics.

        Returns:
            Dictionary with pipeline statistics
        """
        stats = {
            'n_workers': self.n_workers,
            'total_configs': self.total_configs,
            'completed_configs': self.completed_configs,
            'failed_configs': self.failed_configs,
            'success_rate': (
                self.completed_configs / max(1, self.total_configs)
            ),
            'timing': self.timing_stats,
            'cache_size': len(self.results_cache) if self.results_cache else 0
        }

        return stats

    def clear_cache(self):
        """Clear the results cache."""
        if self.results_cache:
            self.results_cache.clear()
            logger.info("Results cache cleared")

    def save_results(self, filepath: str):
        """
        Save cached results to disk.

        Args:
            filepath: Path to save results
        """
        if not self.results_cache:
            logger.warning("No cache to save")
            return

        with open(filepath, 'wb') as f:
            pickle.dump(self.results_cache, f)

        logger.info(f"Saved {len(self.results_cache)} results to {filepath}")

    def load_results(self, filepath: str):
        """
        Load cached results from disk.

        Args:
            filepath: Path to load results from
        """
        try:
            with open(filepath, 'rb') as f:
                loaded_cache = pickle.load(f)

            if self.results_cache:
                self.results_cache.update(loaded_cache)
            else:
                self.results_cache = loaded_cache

            logger.info(f"Loaded {len(loaded_cache)} results from {filepath}")

        except FileNotFoundError:
            logger.warning(f"Results file not found: {filepath}")
        except Exception as e:
            logger.error(f"Error loading results: {e}")


# Helper function for parallel training
def train_config_worker(
    config_data: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    teacher_probs: Optional[np.ndarray]
) -> Tuple[Any, Dict[str, float]]:
    """
    Worker function for parallel configuration training.

    Args:
        config_data: Configuration dictionary
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        teacher_probs: Teacher probabilities

    Returns:
        Tuple of (trained model, metrics)
    """
    from deepbridge.utils.model_registry import ModelFactory
    from deepbridge.metrics.classification import Classification

    # Create model
    factory = ModelFactory()
    model = factory.create_model(
        model_type=config_data['model_type'],
        task_type='classification',
        **config_data.get('hyperparams', {})
    )

    # Train model (simplified - actual implementation would use distillation)
    model.fit(X_train, y_train)

    # Evaluate
    metrics_calc = Classification()
    if X_val is not None and y_val is not None:
        predictions = model.predict(X_val)
        metrics = metrics_calc.calculate_metrics(y_val, predictions)
    else:
        predictions = model.predict(X_train)
        metrics = metrics_calc.calculate_metrics(y_train, predictions)

    return model, metrics