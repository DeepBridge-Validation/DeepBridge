"""
Progressive Distillation Chain for HPM-KD

This module implements a progressive distillation chain that transfers knowledge
incrementally from simple to complex models, reducing the knowledge gap.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from copy import deepcopy

from deepbridge.utils.model_registry import ModelType, ModelRegistry
from deepbridge.metrics.classification import Classification

logger = logging.getLogger(__name__)


@dataclass
class ChainStage:
    """
    Represents a stage in the progressive distillation chain.
    """
    model_type: ModelType
    model: Any
    predictions: Optional[np.ndarray] = None
    performance: Optional[float] = None
    temperature: float = 1.0
    alpha: float = 0.5


class ProgressiveDistillationChain:
    """
    Implements progressive knowledge distillation from simple to complex models.

    This approach reduces the knowledge gap by using intermediate models as
    stepping stones, where each model learns from both the original teacher
    and the previous model in the chain.
    """

    def __init__(
        self,
        chain_order: Optional[List[ModelType]] = None,
        use_adaptive_weights: bool = True,
        min_improvement: float = 0.01,
        random_state: int = 42
    ):
        """
        Initialize the progressive distillation chain.

        Args:
            chain_order: Order of models in the chain (simple to complex)
            use_adaptive_weights: Whether to adapt weights based on performance
            min_improvement: Minimum improvement to continue chain
            random_state: Random seed for reproducibility
        """
        self.chain_order = chain_order or self._get_default_chain()
        self.use_adaptive_weights = use_adaptive_weights
        self.min_improvement = min_improvement
        self.random_state = random_state

        # Storage for chain stages
        self.stages: List[ChainStage] = []
        self.model_factory = ModelRegistry()
        self.metrics_calculator = Classification()

        # Performance tracking
        self.performance_history = []
        self.best_stage_idx = -1

    def _get_default_chain(self) -> List[ModelType]:
        """
        Get default chain order from simple to complex.

        Returns:
            List of model types in increasing complexity
        """
        return [
            ModelType.LOGISTIC_REGRESSION,  # Simplest - linear
            ModelType.DECISION_TREE,        # Non-linear, interpretable
            ModelType.RANDOM_FOREST,        # Ensemble of trees
            ModelType.GBM,                   # Boosted ensemble
            ModelType.XGB                    # Most complex
        ]

    def train_progressive(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, pd.Series]] = None,
        teacher_probs: Optional[np.ndarray] = None,
        temperature_schedule: Optional[List[float]] = None,
        alpha_schedule: Optional[List[float]] = None,
        hyperparams: Optional[Dict[ModelType, Dict[str, Any]]] = None
    ) -> List[ChainStage]:
        """
        Train models progressively through the chain.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            teacher_probs: Original teacher probabilities
            temperature_schedule: Temperature values for each stage
            alpha_schedule: Alpha values for each stage
            hyperparams: Hyperparameters for each model type

        Returns:
            List of trained chain stages
        """
        logger.info(f"Starting progressive chain with {len(self.chain_order)} stages")

        # Initialize schedules
        if temperature_schedule is None:
            # Gradually increase temperature (softer targets) as models get complex
            temperature_schedule = np.linspace(0.5, 3.0, len(self.chain_order))

        if alpha_schedule is None:
            # Start with more weight on hard labels, gradually shift to soft
            alpha_schedule = np.linspace(0.7, 0.3, len(self.chain_order))

        # Clear previous stages
        self.stages = []
        self.performance_history = []

        # Progressive training
        previous_probs = teacher_probs
        previous_performance = 0.0

        for idx, model_type in enumerate(self.chain_order):
            logger.info(f"Training stage {idx+1}/{len(self.chain_order)}: {model_type.name}")

            # Get hyperparameters for this model
            model_params = {}
            if hyperparams and model_type in hyperparams:
                model_params = hyperparams[model_type]

            # Create and train model
            stage = self._train_stage(
                model_type=model_type,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                teacher_probs=teacher_probs,
                previous_probs=previous_probs,
                temperature=temperature_schedule[idx],
                alpha=alpha_schedule[idx],
                model_params=model_params,
                stage_idx=idx
            )

            self.stages.append(stage)

            # Evaluate performance
            if X_val is not None and y_val is not None:
                val_probs = stage.model.predict_proba(X_val)
                if val_probs.ndim == 1:
                    val_probs = np.column_stack([1 - val_probs, val_probs])

                val_preds = np.argmax(val_probs, axis=1)
                metrics = self.metrics_calculator.calculate_metrics(
                    y_true=y_val,
                    y_pred=val_preds,
                    y_prob=val_probs[:, 1] if val_probs.shape[1] == 2 else val_probs
                )
                stage.performance = metrics.get('accuracy', 0.0)
            else:
                # Use training performance as fallback
                train_probs = stage.model.predict_proba(X_train)
                if train_probs.ndim == 1:
                    train_probs = np.column_stack([1 - train_probs, train_probs])

                train_preds = np.argmax(train_probs, axis=1)
                metrics = self.metrics_calculator.calculate_metrics(
                    y_true=y_train,
                    y_pred=train_preds,
                    y_prob=train_probs[:, 1] if train_probs.shape[1] == 2 else train_probs
                )
                stage.performance = metrics.get('accuracy', 0.0)

            self.performance_history.append(stage.performance)
            logger.info(f"Stage {idx+1} performance: {stage.performance:.4f}")

            # Check for improvement
            improvement = stage.performance - previous_performance
            if improvement < self.min_improvement and idx > 0:
                logger.info(f"Improvement ({improvement:.4f}) below threshold, stopping chain")
                break

            # Update best stage
            if stage.performance > previous_performance:
                self.best_stage_idx = idx
                previous_performance = stage.performance

            # Use this stage's predictions as soft labels for next stage
            previous_probs = stage.predictions

        logger.info(f"Progressive chain complete. Best stage: {self.best_stage_idx+1}")
        return self.stages

    def _train_stage(
        self,
        model_type: ModelType,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        X_val: Optional[Union[np.ndarray, pd.DataFrame]],
        y_val: Optional[Union[np.ndarray, pd.Series]],
        teacher_probs: Optional[np.ndarray],
        previous_probs: Optional[np.ndarray],
        temperature: float,
        alpha: float,
        model_params: Dict[str, Any],
        stage_idx: int
    ) -> ChainStage:
        """
        Train a single stage in the progressive chain.

        Args:
            model_type: Type of model to train
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            teacher_probs: Original teacher probabilities
            previous_probs: Previous stage probabilities
            temperature: Temperature for this stage
            alpha: Alpha weight for this stage
            model_params: Model hyperparameters
            stage_idx: Index of this stage

        Returns:
            Trained chain stage
        """
        # Create model
        from deepbridge.utils.model_registry import ModelMode

        # model_type is already a ModelType enum
        model = self.model_factory.get_model(
            model_type=model_type,
            custom_params={**model_params, 'random_state': self.random_state},
            mode=ModelMode.CLASSIFICATION
        )

        # Prepare soft labels
        if teacher_probs is not None and previous_probs is not None:
            if self.use_adaptive_weights:
                # Adaptively weight teacher and previous stage
                weight = self._calculate_adaptive_weight(stage_idx)
                soft_labels = weight * teacher_probs + (1 - weight) * previous_probs
            else:
                # Equal weighting
                soft_labels = 0.5 * teacher_probs + 0.5 * previous_probs
        elif teacher_probs is not None:
            soft_labels = teacher_probs
        elif previous_probs is not None:
            soft_labels = previous_probs
        else:
            soft_labels = None

        # Apply temperature scaling if we have soft labels
        if soft_labels is not None:
            soft_labels = self._apply_temperature(soft_labels, temperature)

        # Train model with knowledge distillation
        if soft_labels is not None:
            # Create combined labels (hard + soft)
            if hasattr(model, 'fit_with_distillation'):
                # If model supports distillation directly
                model.fit_with_distillation(
                    X_train, y_train, soft_labels, alpha=alpha
                )
            else:
                # Use sample weighting as approximation
                sample_weights = self._create_sample_weights(
                    y_train, soft_labels, alpha
                )

                if hasattr(model, 'fit') and 'sample_weight' in model.fit.__code__.co_varnames:
                    model.fit(X_train, y_train, sample_weight=sample_weights)
                else:
                    # Fallback to standard training
                    model.fit(X_train, y_train)
        else:
            # Standard training without distillation
            model.fit(X_train, y_train)

        # Get predictions for next stage
        predictions = model.predict_proba(X_train)
        if predictions.ndim == 1:
            predictions = np.column_stack([1 - predictions, predictions])

        # Create stage
        stage = ChainStage(
            model_type=model_type,
            model=model,
            predictions=predictions,
            temperature=temperature,
            alpha=alpha
        )

        return stage

    def _calculate_adaptive_weight(self, stage_idx: int) -> float:
        """
        Calculate adaptive weight for combining teacher and previous stage.

        Args:
            stage_idx: Current stage index

        Returns:
            Weight for teacher probabilities (1 - weight for previous)
        """
        # Start with more weight on teacher, gradually shift to previous stage
        total_stages = len(self.chain_order)

        if total_stages == 1:
            return 1.0

        # Exponential decay of teacher influence
        teacher_weight = np.exp(-2.0 * stage_idx / total_stages)

        # Ensure reasonable bounds
        teacher_weight = np.clip(teacher_weight, 0.2, 0.8)

        return teacher_weight

    def _apply_temperature(self, probs: np.ndarray, temperature: float) -> np.ndarray:
        """
        Apply temperature scaling to probabilities.

        Args:
            probs: Probability array
            temperature: Temperature value

        Returns:
            Temperature-scaled probabilities
        """
        if temperature == 1.0:
            return probs

        # Convert to logits
        logits = np.log(probs + 1e-10)

        # Apply temperature
        # Ensure we're working with numpy arrays
        if hasattr(logits, 'values'):
            logits = logits.values

        scaled_logits = logits / temperature

        # Convert back to probabilities
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True))
        scaled_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        return scaled_probs

    def _create_sample_weights(
        self,
        y_hard: np.ndarray,
        y_soft: np.ndarray,
        alpha: float
    ) -> np.ndarray:
        """
        Create sample weights based on agreement between hard and soft labels.

        Args:
            y_hard: Hard labels
            y_soft: Soft label probabilities
            alpha: Weight for hard labels

        Returns:
            Sample weights
        """
        # Get predicted class from soft labels
        y_soft_pred = np.argmax(y_soft, axis=1)

        # Agreement between hard and soft labels
        agreement = (y_hard == y_soft_pred).astype(float)

        # Confidence of soft predictions
        confidence = np.max(y_soft, axis=1)

        # Combine into weights
        weights = alpha * np.ones_like(agreement) + (1 - alpha) * confidence * agreement

        # Normalize
        weights = weights / np.mean(weights)

        return weights

    def get_best_model(self) -> Any:
        """
        Get the best performing model from the chain.

        Returns:
            Best model
        """
        if not self.stages:
            raise ValueError("No stages trained yet")

        if self.best_stage_idx >= 0:
            return self.stages[self.best_stage_idx].model

        # Fallback to last stage
        return self.stages[-1].model

    def get_ensemble_predictions(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        weights: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Get ensemble predictions from all stages.

        Args:
            X: Input features
            weights: Optional weights for each stage

        Returns:
            Ensemble predictions
        """
        if not self.stages:
            raise ValueError("No stages trained yet")

        # Get predictions from each stage
        all_predictions = []

        for stage in self.stages:
            probs = stage.model.predict_proba(X)
            if probs.ndim == 1:
                probs = np.column_stack([1 - probs, probs])
            all_predictions.append(probs)

        # Apply weights
        if weights is None:
            # Use performance-based weights
            if all(s.performance is not None for s in self.stages):
                performances = np.array([s.performance for s in self.stages])
                weights = performances / np.sum(performances)
            else:
                # Equal weights
                weights = np.ones(len(self.stages)) / len(self.stages)

        # Weighted average
        ensemble_probs = np.zeros_like(all_predictions[0])
        for probs, weight in zip(all_predictions, weights):
            ensemble_probs += weight * probs

        return ensemble_probs

    def save_chain(self, filepath: str):
        """
        Save the progressive chain to disk.

        Args:
            filepath: Path to save the chain
        """
        import pickle

        chain_data = {
            'stages': self.stages,
            'chain_order': self.chain_order,
            'performance_history': self.performance_history,
            'best_stage_idx': self.best_stage_idx
        }

        with open(filepath, 'wb') as f:
            pickle.dump(chain_data, f)

        logger.info(f"Progressive chain saved to {filepath}")

    def load_chain(self, filepath: str):
        """
        Load a progressive chain from disk.

        Args:
            filepath: Path to load the chain from
        """
        import pickle

        with open(filepath, 'rb') as f:
            chain_data = pickle.load(f)

        self.stages = chain_data['stages']
        self.chain_order = chain_data['chain_order']
        self.performance_history = chain_data['performance_history']
        self.best_stage_idx = chain_data['best_stage_idx']

        logger.info(f"Progressive chain loaded from {filepath}")

    def get_complexity_score(self, model_type: ModelType) -> float:
        """
        Get complexity score for a model type.

        Args:
            model_type: Type of model

        Returns:
            Complexity score (0-1)
        """
        complexity_map = {
            ModelType.LOGISTIC_REGRESSION: 0.2,
            ModelType.DECISION_TREE: 0.4,
            ModelType.RANDOM_FOREST: 0.6,
            ModelType.GBM: 0.8,
            ModelType.XGB: 1.0
        }

        return complexity_map.get(model_type, 0.5)