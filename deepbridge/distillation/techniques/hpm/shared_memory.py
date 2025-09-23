"""
Shared Optimization Memory for HPM-KD

This module implements a shared memory system for hyperparameter optimization
that reduces the number of trials needed by reusing knowledge from similar configurations.
"""

import pickle
import hashlib
import numpy as np
import optuna
from collections import deque, defaultdict
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import json
import logging

from deepbridge.utils.model_registry import ModelType

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """
    Stores the result of a hyperparameter optimization trial.
    """
    model_type: ModelType
    temperature: float
    alpha: float
    best_params: Dict[str, Any]
    best_score: float
    n_trials: int
    context_hash: str
    dataset_characteristics: Dict[str, float]


class SharedOptimizationMemory:
    """
    Cache and reuse hyperparameter optimization results across similar configurations.

    This class maintains a memory of previous optimization runs and uses them to
    warm-start new optimizations, significantly reducing the number of trials needed.
    """

    def __init__(
        self,
        cache_size: int = 100,
        similarity_threshold: float = 0.8,
        min_reuse_score: float = 0.5
    ):
        """
        Initialize the shared optimization memory.

        Args:
            cache_size: Maximum number of optimization results to cache
            similarity_threshold: Minimum similarity score to reuse parameters
            min_reuse_score: Minimum performance score to consider reusing
        """
        self.cache_size = cache_size
        self.similarity_threshold = similarity_threshold
        self.min_reuse_score = min_reuse_score

        # Cache storage with LRU eviction
        self.param_cache = deque(maxlen=cache_size)

        # Index for fast lookup by model type
        self.model_type_index = defaultdict(list)

        # Performance statistics
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'trials_saved': 0
        }

    def get_similar_configs(
        self,
        model_type: ModelType,
        temperature: float,
        alpha: float,
        dataset_characteristics: Optional[Dict[str, float]] = None
    ) -> List[OptimizationResult]:
        """
        Find similar configurations in the cache.

        Args:
            model_type: Type of model
            temperature: Temperature value
            alpha: Alpha value
            dataset_characteristics: Optional dataset features

        Returns:
            List of similar optimization results
        """
        similar_configs = []

        # Get candidates of the same model type
        candidates = []
        for idx in self.model_type_index.get(model_type, []):
            if idx < len(self.param_cache):
                candidates.append(self.param_cache[idx])

        for cached_result in candidates:
            similarity = self._calculate_similarity(
                cached_result,
                model_type,
                temperature,
                alpha,
                dataset_characteristics
            )

            if similarity >= self.similarity_threshold:
                if cached_result.best_score >= self.min_reuse_score:
                    similar_configs.append(cached_result)

        # Sort by similarity (descending)
        similar_configs.sort(
            key=lambda x: self._calculate_similarity(
                x, model_type, temperature, alpha, dataset_characteristics
            ),
            reverse=True
        )

        if similar_configs:
            self.stats['cache_hits'] += 1
            logger.info(f"Found {len(similar_configs)} similar configurations in cache")
        else:
            self.stats['cache_misses'] += 1

        return similar_configs

    def warm_start_study(
        self,
        model_type: ModelType,
        temperature: float,
        alpha: float,
        similar_configs: List[OptimizationResult],
        n_trials: int = 10,
        validation_split: float = 0.2
    ) -> optuna.Study:
        """
        Create an Optuna study with warm start from similar configurations.

        Args:
            model_type: Type of model
            temperature: Temperature value
            alpha: Alpha value
            similar_configs: List of similar configurations
            n_trials: Number of trials for optimization
            validation_split: Validation split ratio

        Returns:
            Warm-started Optuna study
        """
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=max(1, n_trials // 4),
                n_ei_candidates=24
            )
        )

        # Add similar configs as initial trials
        for config in similar_configs[:3]:  # Use top 3 similar configs
            # Enqueue trial with known good parameters
            study.enqueue_trial(config.best_params)

            # Also add slight variations
            for _ in range(2):
                varied_params = self._create_variation(config.best_params, model_type)
                if varied_params:
                    study.enqueue_trial(varied_params)

        trials_saved = min(len(similar_configs) * 3, n_trials // 2)
        self.stats['trials_saved'] += trials_saved

        logger.info(f"Warm-started study with {trials_saved} trials from cache")

        return study

    def add_result(
        self,
        model_type: ModelType,
        temperature: float,
        alpha: float,
        best_params: Dict[str, Any],
        best_score: float,
        n_trials: int,
        dataset_characteristics: Optional[Dict[str, float]] = None
    ):
        """
        Add a new optimization result to the cache.

        Args:
            model_type: Type of model
            temperature: Temperature value
            alpha: Alpha value
            best_params: Best parameters found
            best_score: Best score achieved
            n_trials: Number of trials performed
            dataset_characteristics: Optional dataset features
        """
        # Create context hash
        context_hash = self._create_context_hash(
            model_type,
            temperature,
            alpha,
            dataset_characteristics
        )

        # Create result object
        result = OptimizationResult(
            model_type=model_type,
            temperature=temperature,
            alpha=alpha,
            best_params=best_params,
            best_score=best_score,
            n_trials=n_trials,
            context_hash=context_hash,
            dataset_characteristics=dataset_characteristics or {}
        )

        # Add to cache
        self.param_cache.append(result)

        # Update index
        self._rebuild_index()

        logger.info(f"Added optimization result to cache (score: {best_score:.4f})")

    def _calculate_similarity(
        self,
        cached: OptimizationResult,
        model_type: ModelType,
        temperature: float,
        alpha: float,
        dataset_characteristics: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate similarity between cached result and current configuration.

        Args:
            cached: Cached optimization result
            model_type: Current model type
            temperature: Current temperature
            alpha: Current alpha
            dataset_characteristics: Current dataset features

        Returns:
            Similarity score between 0 and 1
        """
        similarity = 0.0

        # Model type match (required)
        if cached.model_type != model_type:
            return 0.0

        similarity += 0.3  # Base similarity for same model

        # Temperature similarity (exponential decay)
        temp_diff = abs(cached.temperature - temperature)
        temp_similarity = np.exp(-temp_diff / 2.0)
        similarity += 0.2 * temp_similarity

        # Alpha similarity
        alpha_diff = abs(cached.alpha - alpha)
        alpha_similarity = 1.0 - min(alpha_diff, 1.0)
        similarity += 0.2 * alpha_similarity

        # Dataset similarity (if available)
        if dataset_characteristics and cached.dataset_characteristics:
            dataset_sim = self._dataset_similarity(
                cached.dataset_characteristics,
                dataset_characteristics
            )
            similarity += 0.3 * dataset_sim
        else:
            # No dataset info, use default similarity
            similarity += 0.15

        return min(similarity, 1.0)

    def _dataset_similarity(
        self,
        cached_features: Dict[str, float],
        current_features: Dict[str, float]
    ) -> float:
        """
        Calculate similarity between dataset characteristics.

        Args:
            cached_features: Cached dataset features
            current_features: Current dataset features

        Returns:
            Similarity score
        """
        if not cached_features or not current_features:
            return 0.5  # Default similarity

        similarity_scores = []

        # Compare common features
        common_features = set(cached_features.keys()) & set(current_features.keys())

        for feature in common_features:
            cached_val = cached_features[feature]
            current_val = current_features[feature]

            if feature == 'n_samples':
                # Log scale for sample size
                ratio = min(cached_val, current_val) / max(cached_val, current_val)
                similarity_scores.append(ratio)

            elif feature == 'n_features':
                # Linear scale for features
                diff = abs(cached_val - current_val)
                max_val = max(cached_val, current_val)
                similarity_scores.append(1.0 - min(diff / max_val, 1.0))

            elif feature in ['class_balance', 'noise_level']:
                # Direct difference for ratios
                diff = abs(cached_val - current_val)
                similarity_scores.append(1.0 - min(diff, 1.0))

        if similarity_scores:
            return np.mean(similarity_scores)
        return 0.5

    def _create_variation(
        self,
        params: Dict[str, Any],
        model_type: ModelType
    ) -> Optional[Dict[str, Any]]:
        """
        Create a slight variation of hyperparameters for exploration.

        Args:
            params: Original parameters
            model_type: Type of model

        Returns:
            Varied parameters or None if not applicable
        """
        if not params:
            return None

        varied = params.copy()

        # Model-specific variations
        if model_type in [ModelType.XGB, ModelType.GBM]:
            if 'max_depth' in varied:
                varied['max_depth'] = max(1, varied['max_depth'] + np.random.choice([-1, 1]))

            if 'n_estimators' in varied:
                factor = np.random.uniform(0.8, 1.2)
                varied['n_estimators'] = int(varied['n_estimators'] * factor)

            if 'learning_rate' in varied:
                factor = np.random.uniform(0.8, 1.2)
                varied['learning_rate'] = varied['learning_rate'] * factor

        elif model_type == ModelType.RANDOM_FOREST:
            if 'n_estimators' in varied:
                varied['n_estimators'] = max(10, varied['n_estimators'] + np.random.randint(-20, 20))

            if 'max_features' in varied and isinstance(varied['max_features'], float):
                varied['max_features'] = np.clip(
                    varied['max_features'] + np.random.uniform(-0.1, 0.1),
                    0.1,
                    1.0
                )

        elif model_type == ModelType.LOGISTIC_REGRESSION:
            if 'C' in varied:
                factor = np.random.uniform(0.5, 2.0)
                varied['C'] = varied['C'] * factor

        return varied

    def _create_context_hash(
        self,
        model_type: ModelType,
        temperature: float,
        alpha: float,
        dataset_characteristics: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Create a hash representing the optimization context.

        Args:
            model_type: Type of model
            temperature: Temperature value
            alpha: Alpha value
            dataset_characteristics: Optional dataset features

        Returns:
            Context hash string
        """
        context = {
            'model': model_type.name,
            'temp': round(temperature, 2),
            'alpha': round(alpha, 2)
        }

        if dataset_characteristics:
            # Round values to avoid minor differences
            context['dataset'] = {
                k: round(v, 3) for k, v in dataset_characteristics.items()
            }

        # Create hash
        context_str = json.dumps(context, sort_keys=True)
        return hashlib.md5(context_str.encode()).hexdigest()

    def _rebuild_index(self):
        """
        Rebuild the model type index after cache changes.
        """
        self.model_type_index.clear()

        for idx, result in enumerate(self.param_cache):
            self.model_type_index[result.model_type].append(idx)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        stats = self.stats.copy()
        stats['cache_size'] = len(self.param_cache)
        stats['hit_rate'] = (
            self.stats['cache_hits'] /
            max(1, self.stats['cache_hits'] + self.stats['cache_misses'])
        )
        return stats

    def save_to_disk(self, filepath: str):
        """
        Save cache to disk for persistence.

        Args:
            filepath: Path to save the cache
        """
        with open(filepath, 'wb') as f:
            pickle.dump({
                'cache': list(self.param_cache),
                'stats': self.stats
            }, f)

        logger.info(f"Saved cache to {filepath}")

    def load_from_disk(self, filepath: str):
        """
        Load cache from disk.

        Args:
            filepath: Path to load the cache from
        """
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.param_cache = deque(data['cache'], maxlen=self.cache_size)
                self.stats = data['stats']
                self._rebuild_index()

            logger.info(f"Loaded cache from {filepath}")
        except FileNotFoundError:
            logger.warning(f"Cache file not found: {filepath}")
        except Exception as e:
            logger.error(f"Error loading cache: {e}")

    def clear(self):
        """
        Clear the cache.
        """
        self.param_cache.clear()
        self.model_type_index.clear()
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'trials_saved': 0
        }
        logger.info("Cache cleared")