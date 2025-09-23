"""
Adaptive Configuration Manager for HPM-KD

This module implements intelligent configuration selection using Bayesian optimization
to reduce the search space from 64 to 16 most promising configurations.
"""

import numpy as np
import optuna
from typing import List, Dict, Tuple, Optional, Any
from itertools import product
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import StandardScaler
import logging

from deepbridge.utils.model_registry import ModelType

logger = logging.getLogger(__name__)


class AdaptiveConfigurationManager:
    """
    Manages intelligent selection of distillation configurations using Bayesian optimization.

    This class reduces the computational burden by selecting only the most promising
    configurations based on initial sampling and Gaussian Process predictions.
    """

    def __init__(
        self,
        max_configs: int = 16,
        initial_samples: int = 8,
        exploration_ratio: float = 0.3,
        random_state: int = 42
    ):
        """
        Initialize the adaptive configuration manager.

        Args:
            max_configs: Maximum number of configurations to select
            initial_samples: Number of initial random samples for exploration
            exploration_ratio: Ratio of exploratory vs exploitative configs
            random_state: Random seed for reproducibility
        """
        self.max_configs = max_configs
        self.initial_samples = initial_samples
        self.exploration_ratio = exploration_ratio
        self.random_state = random_state

        # Gaussian Process for performance prediction
        kernel = Matern(length_scale=1.0, nu=2.5)
        self.gp_model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            random_state=random_state
        )

        # Storage for configuration history
        self.performance_history = []
        self.config_history = []
        self.scaler = StandardScaler()

    def select_promising_configs(
        self,
        model_types: List[ModelType],
        temperatures: List[float],
        alphas: List[float],
        dataset_features: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Select the most promising configurations using Bayesian optimization.

        Args:
            model_types: List of model types to consider
            temperatures: List of temperature values
            alphas: List of alpha values
            dataset_features: Optional dataset characteristics for context

        Returns:
            List of selected configurations (max_configs items)
        """
        # Generate all possible configurations
        all_configs = list(product(model_types, temperatures, alphas))
        total_configs = len(all_configs)

        logger.info(f"Total possible configurations: {total_configs}")

        if total_configs <= self.max_configs:
            # If we have fewer configs than max, return all
            return [self._create_config_dict(c) for c in all_configs]

        # Phase 1: Initial stratified sampling
        initial_configs = self._stratified_sampling(
            all_configs,
            n_samples=min(self.initial_samples, total_configs // 2)
        )

        # Phase 2: Quick evaluation on subset of data
        if dataset_features:
            initial_scores = self._quick_evaluate(initial_configs, dataset_features)

            # Update GP model with initial results
            X_init = self._configs_to_features(initial_configs)
            self.gp_model.fit(X_init, initial_scores)

            # Phase 3: Predict performance for remaining configs
            remaining_configs = [c for c in all_configs if c not in initial_configs]
            X_remaining = self._configs_to_features(remaining_configs)

            # Get predictions with uncertainty
            mean_pred, std_pred = self.gp_model.predict(X_remaining, return_std=True)

            # Phase 4: Select configs based on upper confidence bound
            ucb_scores = mean_pred + 2.0 * std_pred  # Exploration bonus

            # Combine exploitation and exploration
            n_exploit = int(self.max_configs * (1 - self.exploration_ratio))
            n_explore = self.max_configs - n_exploit

            # Select best predicted (exploitation)
            exploit_indices = np.argsort(ucb_scores)[-n_exploit:]
            exploit_configs = [remaining_configs[i] for i in exploit_indices]

            # Select high uncertainty (exploration)
            explore_indices = np.argsort(std_pred)[-n_explore:]
            explore_configs = [remaining_configs[i] for i in explore_indices
                             if remaining_configs[i] not in exploit_configs]

            selected_configs = initial_configs + exploit_configs + explore_configs[:n_explore]
        else:
            # Without dataset features, use diversity-based selection
            selected_configs = self._diversity_based_selection(all_configs)

        # Ensure we don't exceed max_configs
        selected_configs = selected_configs[:self.max_configs]

        logger.info(f"Selected {len(selected_configs)} configurations")

        return [self._create_config_dict(c) for c in selected_configs]

    def _stratified_sampling(
        self,
        configs: List[Tuple],
        n_samples: int
    ) -> List[Tuple]:
        """
        Perform stratified sampling to ensure diversity in initial samples.

        Args:
            configs: List of all configurations
            n_samples: Number of samples to select

        Returns:
            List of selected configurations
        """
        np.random.seed(self.random_state)

        # Group configs by model type
        model_groups = {}
        for config in configs:
            model_type = config[0]
            if model_type not in model_groups:
                model_groups[model_type] = []
            model_groups[model_type].append(config)

        # Sample from each group proportionally
        samples = []
        samples_per_group = max(1, n_samples // len(model_groups))

        for model_type, group_configs in model_groups.items():
            n_group_samples = min(samples_per_group, len(group_configs))
            group_samples = np.random.choice(
                len(group_configs),
                n_group_samples,
                replace=False
            )
            samples.extend([group_configs[i] for i in group_samples])

        # Fill remaining slots randomly if needed
        remaining = n_samples - len(samples)
        if remaining > 0:
            unused_configs = [c for c in configs if c not in samples]
            if unused_configs:
                extra_samples = np.random.choice(
                    len(unused_configs),
                    min(remaining, len(unused_configs)),
                    replace=False
                )
                samples.extend([unused_configs[i] for i in extra_samples])

        return samples[:n_samples]

    def _quick_evaluate(
        self,
        configs: List[Tuple],
        dataset_features: Dict[str, float]
    ) -> np.ndarray:
        """
        Quickly evaluate configurations using heuristics.

        Args:
            configs: Configurations to evaluate
            dataset_features: Dataset characteristics

        Returns:
            Array of estimated performance scores
        """
        scores = []

        for config in configs:
            model_type, temperature, alpha = config

            # Heuristic scoring based on dataset features and config
            score = 0.5  # Base score

            # Model complexity vs dataset size
            if 'n_samples' in dataset_features:
                n_samples = dataset_features['n_samples']
                if n_samples < 1000:
                    # Small dataset - simpler models better
                    if model_type == ModelType.LOGISTIC_REGRESSION:
                        score += 0.2
                    elif model_type == ModelType.XGB:
                        score -= 0.1
                else:
                    # Large dataset - complex models better
                    if model_type in [ModelType.XGB, ModelType.GBM]:
                        score += 0.2

            # Temperature heuristics
            if temperature >= 1.0 and temperature <= 3.0:
                score += 0.1  # Moderate temperatures usually work better

            # Alpha balance
            if alpha >= 0.4 and alpha <= 0.7:
                score += 0.1  # Balanced alpha usually better

            scores.append(score)

        return np.array(scores)

    def _configs_to_features(self, configs: List[Tuple]) -> np.ndarray:
        """
        Convert configurations to numerical features for GP model.

        Args:
            configs: List of configurations

        Returns:
            Feature matrix
        """
        features = []

        for config in configs:
            model_type, temperature, alpha = config

            # Encode model type as numerical features
            model_encoding = self._encode_model_type(model_type)

            # Create feature vector
            feature_vec = [
                *model_encoding,
                temperature,
                np.log(temperature),  # Log scale for temperature
                alpha,
                alpha ** 2,  # Non-linear alpha effect
                temperature * alpha  # Interaction term
            ]

            features.append(feature_vec)

        features = np.array(features)

        # Normalize features
        if len(features) > 1:
            features = self.scaler.fit_transform(features)

        return features

    def _encode_model_type(self, model_type: ModelType) -> List[float]:
        """
        Encode model type as numerical features.

        Args:
            model_type: Model type to encode

        Returns:
            Encoded features
        """
        # Model complexity encoding
        complexity_map = {
            ModelType.LOGISTIC_REGRESSION: 1.0,
            ModelType.DECISION_TREE: 2.0,
            ModelType.RANDOM_FOREST: 3.0,
            ModelType.GBM: 4.0,
            ModelType.XGB: 5.0,
            ModelType.GAM_CLASSIFIER: 2.5
        }

        # Get complexity score
        complexity = complexity_map.get(model_type, 3.0)

        # Binary encoding for model family
        is_tree_based = int(model_type in [
            ModelType.DECISION_TREE,
            ModelType.RANDOM_FOREST,
            ModelType.GBM,
            ModelType.XGB
        ])

        is_linear = int(model_type in [
            ModelType.LOGISTIC_REGRESSION
        ])

        is_ensemble = int(model_type in [
            ModelType.RANDOM_FOREST,
            ModelType.GBM,
            ModelType.XGB
        ])

        return [complexity, is_tree_based, is_linear, is_ensemble]

    def _diversity_based_selection(
        self,
        configs: List[Tuple]
    ) -> List[Tuple]:
        """
        Select diverse configurations when no dataset features available.

        Args:
            configs: All possible configurations

        Returns:
            Selected diverse configurations
        """
        np.random.seed(self.random_state)

        # Start with stratified sampling
        selected = self._stratified_sampling(configs, self.initial_samples)

        # Add diverse configs based on distance
        while len(selected) < self.max_configs and len(selected) < len(configs):
            # Find config most distant from selected ones
            max_min_dist = -1
            best_config = None

            for config in configs:
                if config in selected:
                    continue

                # Calculate minimum distance to selected configs
                min_dist = float('inf')
                for sel_config in selected:
                    dist = self._config_distance(config, sel_config)
                    min_dist = min(min_dist, dist)

                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_config = config

            if best_config:
                selected.append(best_config)
            else:
                break

        return selected

    def _config_distance(self, config1: Tuple, config2: Tuple) -> float:
        """
        Calculate distance between two configurations.

        Args:
            config1: First configuration
            config2: Second configuration

        Returns:
            Distance score
        """
        model1, temp1, alpha1 = config1
        model2, temp2, alpha2 = config2

        # Model distance (0 if same, 1 if different)
        model_dist = 0.0 if model1 == model2 else 1.0

        # Temperature distance (normalized)
        temp_dist = abs(temp1 - temp2) / 5.0

        # Alpha distance
        alpha_dist = abs(alpha1 - alpha2)

        # Weighted combination
        return 0.4 * model_dist + 0.3 * temp_dist + 0.3 * alpha_dist

    def _create_config_dict(self, config: Tuple) -> Dict[str, Any]:
        """
        Create configuration dictionary from tuple.

        Args:
            config: Configuration tuple

        Returns:
            Configuration dictionary
        """
        return {
            'model_type': config[0],
            'temperature': config[1],
            'alpha': config[2]
        }

    def update_history(
        self,
        config: Dict[str, Any],
        performance: float
    ):
        """
        Update configuration history with results.

        Args:
            config: Configuration used
            performance: Performance achieved
        """
        self.config_history.append(config)
        self.performance_history.append(performance)

        # Retrain GP model if we have enough data
        if len(self.performance_history) >= 5:
            configs_tuple = [
                (c['model_type'], c['temperature'], c['alpha'])
                for c in self.config_history
            ]
            X = self._configs_to_features(configs_tuple)
            y = np.array(self.performance_history)
            self.gp_model.fit(X, y)