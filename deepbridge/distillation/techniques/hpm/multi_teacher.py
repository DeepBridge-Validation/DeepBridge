"""
Attention-Weighted Multi-Teacher System for HPM-KD

This module implements a multi-teacher distillation system with attention mechanisms
that adaptively weights knowledge from multiple teacher models.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any, Union, Tuple
from dataclasses import dataclass
import logging
from sklearn.metrics import pairwise_distances
from scipy.special import softmax

logger = logging.getLogger(__name__)


@dataclass
class TeacherModel:
    """
    Represents a teacher model with its characteristics.
    """
    model: Any
    model_type: str
    performance: float
    predictions: Optional[np.ndarray] = None
    attention_weight: float = 1.0
    confidence_scores: Optional[np.ndarray] = None


class AttentionWeightedMultiTeacher:
    """
    Multi-teacher distillation system with attention-based knowledge fusion.

    This system combines knowledge from multiple teachers using learned attention
    weights that adapt based on teacher agreement, confidence, and student state.
    """

    def __init__(
        self,
        attention_type: str = 'learned',
        temperature: float = 1.0,
        agreement_threshold: float = 0.8,
        confidence_threshold: float = 0.7,
        use_uncertainty_weighting: bool = True
    ):
        """
        Initialize the multi-teacher system.

        Args:
            attention_type: Type of attention ('learned', 'performance', 'agreement', 'hybrid')
            temperature: Temperature for attention softmax
            agreement_threshold: Threshold for teacher agreement
            confidence_threshold: Threshold for prediction confidence
            use_uncertainty_weighting: Weight teachers by uncertainty estimates
        """
        self.attention_type = attention_type
        self.temperature = temperature
        self.agreement_threshold = agreement_threshold
        self.confidence_threshold = confidence_threshold
        self.use_uncertainty_weighting = use_uncertainty_weighting

        # Storage for teachers
        self.teachers: List[TeacherModel] = []

        # Learned attention parameters
        self.attention_weights = None
        self.attention_history = []

        # Performance tracking
        self.fusion_history = []

    def add_teacher(
        self,
        model: Any,
        model_type: str,
        performance: float,
        predictions: Optional[np.ndarray] = None
    ):
        """
        Add a teacher model to the ensemble.

        Args:
            model: Teacher model
            model_type: Type/name of the model
            performance: Performance metric of the teacher
            predictions: Optional pre-computed predictions
        """
        teacher = TeacherModel(
            model=model,
            model_type=model_type,
            performance=performance,
            predictions=predictions
        )

        self.teachers.append(teacher)
        logger.info(f"Added teacher: {model_type} with performance {performance:.4f}")

    def compute_attention_weights(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        student_predictions: Optional[np.ndarray] = None,
        student_state: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Compute attention weights for each teacher.

        Args:
            X: Input features
            student_predictions: Current student predictions
            student_state: Optional student learning state

        Returns:
            Attention weights for each teacher
        """
        if not self.teachers:
            raise ValueError("No teachers available")

        # Get predictions from all teachers
        teacher_predictions = self._get_all_predictions(X)

        if self.attention_type == 'learned':
            weights = self._compute_learned_attention(
                teacher_predictions,
                student_predictions,
                student_state
            )
        elif self.attention_type == 'performance':
            weights = self._compute_performance_attention()
        elif self.attention_type == 'agreement':
            weights = self._compute_agreement_attention(teacher_predictions)
        elif self.attention_type == 'hybrid':
            weights = self._compute_hybrid_attention(
                teacher_predictions,
                student_predictions,
                student_state
            )
        else:
            # Equal weights as fallback
            weights = np.ones(len(self.teachers)) / len(self.teachers)

        # Apply temperature scaling
        if self.temperature != 1.0:
            weights = softmax(np.log(weights + 1e-10) / self.temperature)

        # Store weights
        self.attention_weights = weights
        self.attention_history.append(weights.copy())

        return weights

    def _get_all_predictions(
        self,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> List[np.ndarray]:
        """
        Get predictions from all teachers.

        Args:
            X: Input features

        Returns:
            List of prediction arrays
        """
        predictions = []

        for teacher in self.teachers:
            if teacher.predictions is not None:
                # Use cached predictions
                preds = teacher.predictions
            else:
                # Compute predictions
                if hasattr(teacher.model, 'predict_proba'):
                    preds = teacher.model.predict_proba(X)
                else:
                    # For models without predict_proba
                    preds = teacher.model.predict(X)
                    # Convert to one-hot if needed
                    if preds.ndim == 1:
                        n_classes = len(np.unique(preds))
                        preds_oh = np.zeros((len(preds), n_classes))
                        preds_oh[np.arange(len(preds)), preds] = 1
                        preds = preds_oh

                # Cache predictions
                teacher.predictions = preds

            # Ensure 2D array
            if preds.ndim == 1:
                preds = np.column_stack([1 - preds, preds])

            predictions.append(preds)

            # Calculate confidence scores
            teacher.confidence_scores = np.max(preds, axis=1)

        return predictions

    def _compute_learned_attention(
        self,
        teacher_predictions: List[np.ndarray],
        student_predictions: Optional[np.ndarray],
        student_state: Optional[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Compute learned attention weights based on multiple factors.

        Args:
            teacher_predictions: Predictions from all teachers
            student_predictions: Current student predictions
            student_state: Student learning state

        Returns:
            Attention weights
        """
        n_teachers = len(self.teachers)
        weights = np.zeros(n_teachers)

        # Factor 1: Teacher performance
        performance_scores = np.array([t.performance for t in self.teachers])
        performance_weights = performance_scores / np.sum(performance_scores)

        # Factor 2: Prediction confidence
        confidence_weights = np.zeros(n_teachers)
        for i, teacher in enumerate(self.teachers):
            if teacher.confidence_scores is not None:
                # Average confidence across samples
                avg_confidence = np.mean(teacher.confidence_scores)
                confidence_weights[i] = avg_confidence

        if np.sum(confidence_weights) > 0:
            confidence_weights /= np.sum(confidence_weights)
        else:
            confidence_weights = np.ones(n_teachers) / n_teachers

        # Factor 3: Agreement with other teachers
        agreement_weights = self._compute_agreement_attention(teacher_predictions)

        # Factor 4: Complementarity (diversity)
        diversity_weights = self._compute_diversity_weights(teacher_predictions)

        # Factor 5: Student-teacher alignment (if student predictions available)
        if student_predictions is not None:
            alignment_weights = self._compute_alignment_weights(
                teacher_predictions,
                student_predictions
            )
        else:
            alignment_weights = np.ones(n_teachers) / n_teachers

        # Combine factors with learned coefficients
        if student_state and 'epoch' in student_state:
            # Adapt weights based on training progress
            epoch = student_state['epoch']
            max_epochs = student_state.get('max_epochs', 100)
            progress = epoch / max_epochs

            # Early training: focus on performance and agreement
            # Late training: focus on diversity and alignment
            weights = (
                (0.4 - 0.2 * progress) * performance_weights +
                (0.3 - 0.1 * progress) * agreement_weights +
                (0.1 + 0.2 * progress) * diversity_weights +
                (0.1 + 0.1 * progress) * alignment_weights +
                0.1 * confidence_weights
            )
        else:
            # Default combination
            weights = (
                0.3 * performance_weights +
                0.25 * agreement_weights +
                0.2 * diversity_weights +
                0.15 * alignment_weights +
                0.1 * confidence_weights
            )

        return weights

    def _compute_performance_attention(self) -> np.ndarray:
        """
        Compute attention based solely on teacher performance.

        Returns:
            Performance-based attention weights
        """
        performances = np.array([t.performance for t in self.teachers])

        # Apply softmax to convert to weights
        weights = softmax(performances / self.temperature)

        return weights

    def _compute_agreement_attention(
        self,
        teacher_predictions: List[np.ndarray]
    ) -> np.ndarray:
        """
        Compute attention based on teacher agreement.

        Args:
            teacher_predictions: Predictions from all teachers

        Returns:
            Agreement-based attention weights
        """
        n_teachers = len(teacher_predictions)
        agreement_scores = np.zeros(n_teachers)

        # Calculate pairwise agreement
        for i in range(n_teachers):
            agreements = []
            pred_i = np.argmax(teacher_predictions[i], axis=1)

            for j in range(n_teachers):
                if i != j:
                    pred_j = np.argmax(teacher_predictions[j], axis=1)
                    agreement = np.mean(pred_i == pred_j)
                    agreements.append(agreement)

            agreement_scores[i] = np.mean(agreements) if agreements else 0.5

        # Normalize
        weights = agreement_scores / np.sum(agreement_scores)

        return weights

    def _compute_diversity_weights(
        self,
        teacher_predictions: List[np.ndarray]
    ) -> np.ndarray:
        """
        Compute weights based on prediction diversity.

        Args:
            teacher_predictions: Predictions from all teachers

        Returns:
            Diversity-based weights
        """
        n_teachers = len(teacher_predictions)
        diversity_scores = np.zeros(n_teachers)

        # Calculate how different each teacher is from the mean
        mean_predictions = np.mean(teacher_predictions, axis=0)

        for i, preds in enumerate(teacher_predictions):
            # KL divergence from mean
            kl_div = self._kl_divergence(preds, mean_predictions)
            diversity_scores[i] = np.mean(kl_div)

        # Higher diversity gets more weight (encourages ensemble diversity)
        weights = diversity_scores / np.sum(diversity_scores) if np.sum(diversity_scores) > 0 else np.ones(n_teachers) / n_teachers

        return weights

    def _compute_alignment_weights(
        self,
        teacher_predictions: List[np.ndarray],
        student_predictions: np.ndarray
    ) -> np.ndarray:
        """
        Compute weights based on teacher-student alignment.

        Args:
            teacher_predictions: Predictions from all teachers
            student_predictions: Current student predictions

        Returns:
            Alignment-based weights
        """
        n_teachers = len(teacher_predictions)
        alignment_scores = np.zeros(n_teachers)

        # Ensure student predictions are in probability form
        if student_predictions.ndim == 1:
            student_predictions = np.column_stack([1 - student_predictions, student_predictions])

        for i, teacher_preds in enumerate(teacher_predictions):
            # Calculate KL divergence (lower is better alignment)
            kl_div = self._kl_divergence(student_predictions, teacher_preds)
            # Convert to similarity score
            alignment_scores[i] = np.exp(-np.mean(kl_div))

        # Normalize
        weights = alignment_scores / np.sum(alignment_scores)

        return weights

    def _compute_hybrid_attention(
        self,
        teacher_predictions: List[np.ndarray],
        student_predictions: Optional[np.ndarray],
        student_state: Optional[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Compute hybrid attention combining multiple strategies.

        Args:
            teacher_predictions: Predictions from all teachers
            student_predictions: Current student predictions
            student_state: Student learning state

        Returns:
            Hybrid attention weights
        """
        # Use learned attention as base
        return self._compute_learned_attention(
            teacher_predictions,
            student_predictions,
            student_state
        )

    def weighted_knowledge_fusion(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        attention_weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Fuse knowledge from all teachers using attention weights.

        Args:
            X: Input features
            attention_weights: Optional pre-computed attention weights

        Returns:
            Fused predictions
        """
        if not self.teachers:
            raise ValueError("No teachers available")

        # Get or compute attention weights
        if attention_weights is None:
            attention_weights = self.compute_attention_weights(X)

        # Get all predictions
        teacher_predictions = self._get_all_predictions(X)

        # Weighted fusion
        fused_predictions = np.zeros_like(teacher_predictions[0])

        for i, (preds, weight) in enumerate(zip(teacher_predictions, attention_weights)):
            fused_predictions += weight * preds

            # Update teacher attention weight
            self.teachers[i].attention_weight = weight

        # Store fusion result
        self.fusion_history.append({
            'weights': attention_weights.copy(),
            'n_samples': len(X)
        })

        return fused_predictions

    def adaptive_fusion(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y_true: Optional[np.ndarray] = None,
        optimize_weights: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform adaptive knowledge fusion with optional weight optimization.

        Args:
            X: Input features
            y_true: True labels for weight optimization
            optimize_weights: Whether to optimize weights

        Returns:
            Tuple of (fused predictions, optimized weights)
        """
        if optimize_weights and y_true is not None:
            # Optimize weights using validation performance
            best_weights = self._optimize_weights(X, y_true)
        else:
            # Use standard attention computation
            best_weights = self.compute_attention_weights(X)

        # Apply fusion with optimized weights
        fused_predictions = self.weighted_knowledge_fusion(X, best_weights)

        return fused_predictions, best_weights

    def _optimize_weights(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y_true: np.ndarray,
        n_iterations: int = 100
    ) -> np.ndarray:
        """
        Optimize attention weights using gradient-free optimization.

        Args:
            X: Input features
            y_true: True labels
            n_iterations: Number of optimization iterations

        Returns:
            Optimized weights
        """
        n_teachers = len(self.teachers)

        # Return equal weights if no teachers
        if n_teachers == 0:
            logger.warning("No teachers available for weight optimization")
            return np.array([])

        best_weights = np.ones(n_teachers) / n_teachers
        best_score = -np.inf

        # Get all teacher predictions
        teacher_predictions = self._get_all_predictions(X)

        # Check if we have predictions
        if not teacher_predictions:
            logger.warning("No teacher predictions available")
            return best_weights

        for _ in range(n_iterations):
            # Random weight perturbation
            weights = np.random.dirichlet(np.ones(n_teachers))

            # Compute fused predictions
            fused = np.zeros_like(teacher_predictions[0])
            for preds, weight in zip(teacher_predictions, weights):
                fused += weight * preds

            # Evaluate performance
            fused_class = np.argmax(fused, axis=1)
            score = np.mean(fused_class == y_true)

            if score > best_score:
                best_score = score
                best_weights = weights

        logger.info(f"Optimized weights achieved score: {best_score:.4f}")
        return best_weights

    def _kl_divergence(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """
        Calculate KL divergence between two probability distributions.

        Args:
            p: First probability distribution
            q: Second probability distribution

        Returns:
            KL divergence
        """
        # Add small epsilon to avoid log(0)
        p = np.clip(p, 1e-10, 1.0)
        q = np.clip(q, 1e-10, 1.0)

        return np.sum(p * np.log(p / q), axis=-1)

    def get_teacher_contributions(self) -> Dict[str, float]:
        """
        Get the contribution of each teacher based on attention history.

        Returns:
            Dictionary mapping teacher names to average contributions
        """
        if not self.attention_history:
            return {}

        avg_weights = np.mean(self.attention_history, axis=0)

        contributions = {}
        for i, teacher in enumerate(self.teachers):
            contributions[teacher.model_type] = float(avg_weights[i])

        return contributions

    def prune_weak_teachers(self, threshold: float = 0.05):
        """
        Remove teachers with consistently low attention weights.

        Args:
            threshold: Minimum average attention weight to keep teacher
        """
        if not self.attention_history or len(self.attention_history) < 5:
            logger.warning("Not enough history to prune teachers")
            return

        avg_weights = np.mean(self.attention_history[-10:], axis=0)

        # Identify weak teachers
        keep_indices = avg_weights >= threshold

        if np.sum(keep_indices) < 2:
            logger.warning("Would remove too many teachers, keeping top 2")
            keep_indices = np.argsort(avg_weights)[-2:]

        # Filter teachers
        self.teachers = [t for i, t in enumerate(self.teachers) if keep_indices[i]]

        logger.info(f"Pruned to {len(self.teachers)} teachers")

    def reset(self):
        """
        Reset the multi-teacher system.
        """
        self.teachers = []
        self.attention_weights = None
        self.attention_history = []
        self.fusion_history = []