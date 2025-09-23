"""
Meta-Learning Temperature Scheduler for HPM-KD

This module implements an adaptive temperature scheduling system using meta-learning
to optimize temperature values dynamically during training.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class TrainingState:
    """
    Represents the current training state for meta-learning.
    """
    epoch: int
    loss: float
    kl_divergence: float
    student_accuracy: float
    teacher_accuracy: float
    temperature: float
    gradient_norm: Optional[float] = None
    learning_rate: Optional[float] = None


class MetaTemperatureScheduler:
    """
    Adaptive temperature scheduler using meta-learning.

    This scheduler learns the optimal temperature schedule based on training
    dynamics, replacing fixed temperature values with adaptive scheduling.
    """

    def __init__(
        self,
        initial_temperature: float = 3.0,
        min_temperature: float = 0.5,
        max_temperature: float = 10.0,
        meta_learning_rate: float = 0.01,
        history_window: int = 20,
        update_frequency: int = 5,
        use_gradient_info: bool = True
    ):
        """
        Initialize the meta temperature scheduler.

        Args:
            initial_temperature: Starting temperature value
            min_temperature: Minimum allowed temperature
            max_temperature: Maximum allowed temperature
            meta_learning_rate: Learning rate for meta-model
            history_window: Size of history window for features
            update_frequency: How often to update temperature (epochs)
            use_gradient_info: Whether to use gradient information
        """
        self.initial_temperature = initial_temperature
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.meta_learning_rate = meta_learning_rate
        self.history_window = history_window
        self.update_frequency = update_frequency
        self.use_gradient_info = use_gradient_info

        # Current temperature
        self.current_temperature = initial_temperature

        # Meta-learning model
        self.meta_model = self._build_meta_model()
        self.feature_scaler = StandardScaler()

        # Training history
        self.state_history = deque(maxlen=history_window)
        self.temperature_history = []
        self.reward_history = []

        # Meta-model training data
        self.meta_features = []
        self.meta_targets = []

        # Performance tracking
        self.best_performance = 0.0
        self.best_temperature = initial_temperature

    def _build_meta_model(self) -> MLPRegressor:
        """
        Build the meta-learning model for temperature prediction.

        Returns:
            Initialized MLPRegressor
        """
        return MLPRegressor(
            hidden_layer_sizes=(64, 32, 16),
            activation='relu',
            solver='adam',
            learning_rate_init=self.meta_learning_rate,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
            warm_start=True
        )

    def adaptive_temperature(
        self,
        epoch: int,
        loss: float,
        kl_divergence: float,
        student_accuracy: float,
        teacher_accuracy: float,
        gradient_norm: Optional[float] = None,
        learning_rate: Optional[float] = None
    ) -> float:
        """
        Compute adaptive temperature based on current training state.

        Args:
            epoch: Current training epoch
            loss: Current training loss
            kl_divergence: KL divergence between student and teacher
            student_accuracy: Student model accuracy
            teacher_accuracy: Teacher model accuracy
            gradient_norm: Optional gradient norm
            learning_rate: Optional current learning rate

        Returns:
            Adaptive temperature value
        """
        # Create current state
        current_state = TrainingState(
            epoch=epoch,
            loss=loss,
            kl_divergence=kl_divergence,
            student_accuracy=student_accuracy,
            teacher_accuracy=teacher_accuracy,
            temperature=self.current_temperature,
            gradient_norm=gradient_norm,
            learning_rate=learning_rate
        )

        # Add to history
        self.state_history.append(current_state)

        # Only update temperature at specified frequency
        if epoch % self.update_frequency != 0:
            return self.current_temperature

        # Extract features from state
        features = self._extract_features(current_state)

        # Predict optimal temperature
        if len(self.meta_features) >= 10:  # Need minimum data to train
            # Update meta-model with recent data
            self._update_meta_model()

            # Predict temperature
            features_scaled = self.feature_scaler.transform(features.reshape(1, -1))
            predicted_temp = self.meta_model.predict(features_scaled)[0]

            # Apply bounds and smoothing
            new_temperature = self._apply_constraints(predicted_temp)
        else:
            # Use heuristic-based temperature
            new_temperature = self._heuristic_temperature(current_state)

        # Store for training
        self.meta_features.append(features)
        self.temperature_history.append(new_temperature)

        # Update current temperature
        self.current_temperature = new_temperature

        logger.debug(f"Epoch {epoch}: Temperature adjusted to {new_temperature:.3f}")

        return new_temperature

    def _extract_features(self, state: TrainingState) -> np.ndarray:
        """
        Extract features from training state for meta-learning.

        Args:
            state: Current training state

        Returns:
            Feature vector
        """
        features = []

        # Basic features
        features.extend([
            state.epoch / 100.0,  # Normalized epoch
            state.loss,
            state.kl_divergence,
            state.student_accuracy,
            state.teacher_accuracy,
            state.teacher_accuracy - state.student_accuracy,  # Performance gap
            state.temperature / self.max_temperature  # Normalized current temp
        ])

        # Historical features (if available)
        if len(self.state_history) >= 3:
            recent_states = list(self.state_history)[-3:]

            # Loss trend
            loss_trend = np.polyfit(
                range(len(recent_states)),
                [s.loss for s in recent_states],
                1
            )[0]
            features.append(loss_trend)

            # KL divergence trend
            kl_trend = np.polyfit(
                range(len(recent_states)),
                [s.kl_divergence for s in recent_states],
                1
            )[0]
            features.append(kl_trend)

            # Accuracy improvement rate
            acc_improvement = (
                state.student_accuracy - recent_states[0].student_accuracy
            ) / len(recent_states)
            features.append(acc_improvement)

            # Temperature stability
            if len(self.temperature_history) >= 3:
                temp_variance = np.var(self.temperature_history[-3:])
                features.append(temp_variance)
            else:
                features.append(0.0)
        else:
            # Default values for trends
            features.extend([0.0, 0.0, 0.0, 0.0])

        # Gradient information (if available)
        if self.use_gradient_info and state.gradient_norm is not None:
            features.append(np.log(state.gradient_norm + 1e-8))
        else:
            features.append(0.0)

        # Learning rate (if available)
        if state.learning_rate is not None:
            features.append(np.log(state.learning_rate + 1e-8))
        else:
            features.append(-5.0)  # Default log(lr)

        return np.array(features)

    def _heuristic_temperature(self, state: TrainingState) -> float:
        """
        Compute temperature using heuristics when meta-model is not ready.

        Args:
            state: Current training state

        Returns:
            Heuristic temperature value
        """
        # Base temperature
        temperature = self.initial_temperature

        # Adjust based on KL divergence
        if state.kl_divergence > 1.0:
            # High divergence - increase temperature (softer targets)
            temperature *= 1.2
        elif state.kl_divergence < 0.1:
            # Low divergence - decrease temperature (harder targets)
            temperature *= 0.8

        # Adjust based on performance gap
        gap = state.teacher_accuracy - state.student_accuracy
        if gap > 0.2:
            # Large gap - increase temperature
            temperature *= 1.1
        elif gap < 0.05:
            # Small gap - decrease temperature
            temperature *= 0.9

        # Adjust based on training progress
        if state.epoch > 50:
            # Later in training - gradually decrease temperature
            temperature *= (1.0 - 0.3 * min(state.epoch / 200.0, 1.0))

        # Apply constraints
        temperature = np.clip(temperature, self.min_temperature, self.max_temperature)

        return temperature

    def _apply_constraints(self, temperature: float) -> float:
        """
        Apply constraints and smoothing to predicted temperature.

        Args:
            temperature: Predicted temperature

        Returns:
            Constrained temperature
        """
        # Apply bounds
        temperature = np.clip(temperature, self.min_temperature, self.max_temperature)

        # Apply smoothing with previous temperature
        if self.temperature_history:
            # Exponential moving average
            alpha = 0.3  # Smoothing factor
            temperature = alpha * temperature + (1 - alpha) * self.current_temperature

        # Prevent too rapid changes
        if self.current_temperature > 0:
            max_change = 0.5 * self.current_temperature
            temperature = np.clip(
                temperature,
                self.current_temperature - max_change,
                self.current_temperature + max_change
            )

        return temperature

    def _update_meta_model(self):
        """
        Update the meta-learning model with collected data.
        """
        if len(self.meta_features) < 10:
            return

        # Prepare training data
        X = np.array(self.meta_features[-100:])  # Use recent data

        # Calculate targets (rewards) based on performance improvement
        y = self._calculate_rewards()

        if len(y) < len(X):
            # Pad with zeros if needed
            y = np.concatenate([np.zeros(len(X) - len(y)), y])
        elif len(y) > len(X):
            y = y[-len(X):]

        # Fit scaler if needed
        if not hasattr(self.feature_scaler, 'mean_'):
            self.feature_scaler.fit(X)

        # Scale features
        X_scaled = self.feature_scaler.transform(X)

        # Update meta-model
        try:
            self.meta_model.partial_fit(X_scaled, y)
        except:
            # First fit
            self.meta_model.fit(X_scaled, y)

    def _calculate_rewards(self) -> np.ndarray:
        """
        Calculate rewards for meta-learning based on performance.

        Returns:
            Array of reward values
        """
        rewards = []

        for i in range(1, len(self.state_history)):
            prev_state = self.state_history[i-1]
            curr_state = self.state_history[i]

            # Reward based on multiple factors
            acc_improvement = curr_state.student_accuracy - prev_state.student_accuracy
            loss_reduction = prev_state.loss - curr_state.loss
            kl_reduction = prev_state.kl_divergence - curr_state.kl_divergence

            # Weighted reward
            reward = (
                0.5 * acc_improvement +  # Accuracy improvement
                0.3 * loss_reduction +    # Loss reduction
                0.2 * kl_reduction        # KL divergence reduction
            )

            # Bonus for closing the gap with teacher
            gap_prev = prev_state.teacher_accuracy - prev_state.student_accuracy
            gap_curr = curr_state.teacher_accuracy - curr_state.student_accuracy
            if gap_curr < gap_prev:
                reward += 0.1 * (gap_prev - gap_curr)

            rewards.append(reward)

        return np.array(rewards)

    def update_with_validation(
        self,
        val_accuracy: float,
        val_loss: float
    ):
        """
        Update scheduler with validation results.

        Args:
            val_accuracy: Validation accuracy
            val_loss: Validation loss
        """
        # Calculate reward based on validation performance
        reward = val_accuracy - val_loss * 0.1

        # Store reward
        self.reward_history.append(reward)

        # Update best temperature if this is best performance
        if val_accuracy > self.best_performance:
            self.best_performance = val_accuracy
            self.best_temperature = self.current_temperature
            logger.info(f"New best temperature: {self.best_temperature:.3f} "
                       f"(accuracy: {val_accuracy:.4f})")

    def get_schedule(self, n_epochs: int) -> List[float]:
        """
        Get a temperature schedule for a given number of epochs.

        Args:
            n_epochs: Number of epochs

        Returns:
            List of temperature values
        """
        schedule = []

        # Generate mock states for prediction
        for epoch in range(n_epochs):
            # Estimate state based on typical training progression
            progress = epoch / n_epochs

            # Mock state (would be replaced with actual in training)
            mock_loss = 2.0 * (1.0 - progress)
            mock_kl = 1.0 * (1.0 - 0.8 * progress)
            mock_student_acc = 0.5 + 0.4 * progress
            mock_teacher_acc = 0.9

            temp = self.adaptive_temperature(
                epoch=epoch,
                loss=mock_loss,
                kl_divergence=mock_kl,
                student_accuracy=mock_student_acc,
                teacher_accuracy=mock_teacher_acc
            )

            schedule.append(temp)

        return schedule

    def reset(self):
        """
        Reset the scheduler to initial state.
        """
        self.current_temperature = self.initial_temperature
        self.state_history.clear()
        self.temperature_history = []
        self.reward_history = []
        self.meta_features = []
        self.meta_targets = []
        self.best_performance = 0.0
        self.best_temperature = self.initial_temperature

        # Reinitialize meta-model
        self.meta_model = self._build_meta_model()
        self.feature_scaler = StandardScaler()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get scheduler statistics.

        Returns:
            Dictionary with scheduler statistics
        """
        stats = {
            'current_temperature': self.current_temperature,
            'best_temperature': self.best_temperature,
            'best_performance': self.best_performance,
            'temperature_range': (
                min(self.temperature_history) if self.temperature_history else self.min_temperature,
                max(self.temperature_history) if self.temperature_history else self.max_temperature
            ),
            'average_temperature': np.mean(self.temperature_history) if self.temperature_history else self.initial_temperature,
            'temperature_variance': np.var(self.temperature_history) if self.temperature_history else 0.0,
            'n_updates': len(self.temperature_history)
        }

        return stats