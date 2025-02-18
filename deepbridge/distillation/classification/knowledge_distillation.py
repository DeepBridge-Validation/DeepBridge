import numpy as np
import pandas as pd
import optuna
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from typing import Dict, Any, Optional, Union, Callable, List, Tuple
import warnings

# Imports absolutos
from deepbridge.distillation.classification.model_registry import ModelRegistry, ModelType
from deepbridge.metrics.classification import Classification

class KnowledgeDistillation(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        teacher_model: Optional[BaseEstimator] = None,
        teacher_probabilities: Optional[np.ndarray] = None,
        student_model_type: ModelType = ModelType.LOGISTIC_REGRESSION,
        student_params: Dict[str, Any] = None,
        temperature: float = 1.0,
        alpha: float = 0.5,
        n_trials: int = 50,
        validation_split: float = 0.2,
        random_state: int = 42
    ):
        """
        Initialize the Knowledge Distillation model.
        
        Args:
            teacher_model: Pre-trained teacher model (optional if teacher_probabilities is provided)
            teacher_probabilities: Pre-calculated teacher probabilities (optional if teacher_model is provided)
            student_model_type: Type of student model to use
            student_params: Custom parameters for student model (if None, will be optimized)
            temperature: Temperature parameter for softening probability distributions
            alpha: Weight between teacher's loss and true label loss
            n_trials: Number of Optuna trials for hyperparameter optimization
            validation_split: Fraction of data to use for validation during optimization
            random_state: Random seed for reproducibility
        """
        if teacher_model is None and teacher_probabilities is None:
            raise ValueError("Either teacher_model or teacher_probabilities must be provided")
            
        self.teacher_model = teacher_model
        self.teacher_probabilities = teacher_probabilities
        self.student_model_type = student_model_type
        self.student_params = student_params
        self.temperature = temperature
        self.alpha = alpha
        self.n_trials = n_trials
        self.validation_split = validation_split
        self.random_state = random_state
        self.metrics_calculator = Classification()
        self.student_model = None
        self.best_params = None
        
    def _get_teacher_soft_labels(self, X: np.ndarray) -> np.ndarray:
        """
        Generate soft labels from either the teacher model or pre-calculated probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Soft labels (probabilities)
        """
        if self.teacher_probabilities is not None:
            # Use pre-calculated probabilities
            if len(self.teacher_probabilities) != len(X):
                raise ValueError(
                    f"Number of teacher probabilities ({len(self.teacher_probabilities)}) "
                    f"doesn't match number of samples ({len(X)})"
                )
            # Apply temperature scaling to probabilities
            logits = np.log(self.teacher_probabilities + 1e-7)
            return softmax(logits / self.temperature, axis=1)
            
        # Use teacher model
        try:
            # Try to get logits using decision_function
            teacher_logits = self.teacher_model.decision_function(X)
            if len(teacher_logits.shape) == 1:
                # Convert to 2D array if necessary
                teacher_logits = np.column_stack([-teacher_logits, teacher_logits])
        except (AttributeError, NotImplementedError):
            # Fallback to predict_proba
            teacher_probs = self.teacher_model.predict_proba(X)
            teacher_logits = np.log(teacher_probs + 1e-7)
        
        return softmax(teacher_logits / self.temperature, axis=1)
        
    def _get_param_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Obter o espaço de parâmetros do ModelRegistry.
        
        Args:
            trial: Optuna trial
            
        Returns:
            Dictionary of hyperparameters to try
        """
        # Utilizamos o método centralizado no ModelRegistry
        return ModelRegistry.get_param_space(self.student_model_type, trial)
        
    def _kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Calculate the Kullback-Leibler divergence between two probability distributions.
        
        Args:
            p: Target probability distribution
            q: Predicted probability distribution
            
        Returns:
            KL divergence value
        """
        # Add small value to avoid log(0)
        epsilon = 1e-10
        q = np.clip(q, epsilon, 1-epsilon)
        return np.sum(p * np.log(p / q))
        
    def _combined_loss(self, y_true: np.ndarray, soft_labels: np.ndarray, student_probs: np.ndarray) -> float:
        """
        Calculate the combined loss using both hard and soft labels.
        
        Args:
            y_true: One-hot encoded true labels
            soft_labels: Soft labels from teacher model
            student_probs: Probabilities from student model
            
        Returns:
            Combined loss value
        """
        # KL divergence for soft labels (distillation loss)
        distillation_loss = self._kl_divergence(soft_labels, student_probs)
        
        # Cross-entropy loss for hard labels
        epsilon = 1e-10
        student_probs = np.clip(student_probs, epsilon, 1-epsilon)
        hard_loss = -np.mean(np.sum(y_true * np.log(student_probs), axis=1))
        
        # Combined loss with alpha weighting
        return self.alpha * distillation_loss + (1 - self.alpha) * hard_loss
        
    def _objective(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray,
                soft_labels: np.ndarray) -> float:
        """
        Objective function for Optuna.
        
        Args:
            trial: Optuna trial
            X: Training features
            y: True labels
            soft_labels: Soft labels from teacher model
            
        Returns:
            Loss value to minimize
        """
        # Get hyperparameters for this trial
        trial_params = self._get_param_space(trial)
        
        
        # Split data for validation
        X_train, X_val, y_train, y_val, soft_train, soft_val = train_test_split(
            X, y, soft_labels, test_size=self.validation_split, random_state=self.random_state
        )
        
        # Create and train student model
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            student = ModelRegistry.get_model(self.student_model_type, trial_params)
            student.fit(X_train, y_train)
        
        # Get probabilities from student model
        student_probs = student.predict_proba(X_val)
        
        # Convert y_val to one-hot encoding for loss calculation
        n_classes = student_probs.shape[1]
        y_val_onehot = np.zeros((len(y_val), n_classes))
        y_val_onehot[np.arange(len(y_val)), y_val] = 1
        
        # Calculate combined loss
        return self._combined_loss(y_val_onehot, soft_val, student_probs)

    @classmethod
    def from_probabilities(
        cls,
        probabilities: Union[np.ndarray, pd.DataFrame],
        student_model_type: ModelType = ModelType.LOGISTIC_REGRESSION,
        student_params: Dict[str, Any] = None,
        temperature: float = 1.0,
        alpha: float = 0.5,
        n_trials: int = 50,
        validation_split: float = 0.2,
        random_state: int = 42
    ) -> 'KnowledgeDistillation':
        """
        Create a KnowledgeDistillation instance from pre-calculated probabilities.
        
        Args:
            probabilities: Array or DataFrame with shape (n_samples, 2) containing class probabilities
            student_model_type: Type of student model to use
            student_params: Custom parameters for student model (if None, will be optimized)
            temperature: Temperature parameter
            alpha: Weight parameter
            n_trials: Number of Optuna trials for hyperparameter optimization
            validation_split: Fraction of data to use for validation during optimization
            random_state: Random seed for reproducibility
            
        Returns:
            KnowledgeDistillation instance
        """
        if isinstance(probabilities, pd.DataFrame):
            probabilities = probabilities.values
            
        if probabilities.shape[1] != 2:
            raise ValueError(
                f"Probabilities must have shape (n_samples, 2), got {probabilities.shape}"
            )
            
        if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5):
            raise ValueError("Probabilities must sum to 1 for each sample")
            
        return cls(
            teacher_probabilities=probabilities,
            student_model_type=student_model_type,
            student_params=student_params,
            temperature=temperature,
            alpha=alpha,
            n_trials=n_trials,
            validation_split=validation_split,
            random_state=random_state
        )

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> 'KnowledgeDistillation':
        """
        Train the student model using Knowledge Distillation with hyperparameter optimization.
        
        Args:
            X: Training features
            y: True labels
            verbose: Whether to print optimization progress and results
            
        Returns:
            self: The trained model
        """
        # Generate soft labels
        soft_labels = self._get_teacher_soft_labels(X)
        
        if self.student_params is None:
            # Filter warnings during Optuna optimization
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                
                # Suppress Optuna logs
                import logging
                optuna_logger = logging.getLogger("optuna")
                optuna_logger_level = optuna_logger.getEffectiveLevel()
                optuna_logger.setLevel(logging.WARNING if verbose else logging.ERROR)
                
                # Optimize hyperparameters using Optuna
                study = optuna.create_study(direction="minimize")
                objective = lambda trial: self._objective(trial, X, y, soft_labels)
                study.optimize(objective, n_trials=self.n_trials)
                
                # Restore Optuna logger level
                optuna_logger.setLevel(optuna_logger_level)
                
                # Get the best hyperparameters
                self.best_params = study.best_params
                if verbose:
                    print(f"Best hyperparameters found: {self.best_params}")
                
                # Create student model with best parameters
                self.student_model = ModelRegistry.get_model(
                    model_type=self.student_model_type,
                    custom_params=self.best_params
                )
        else:
            # Use provided hyperparameters
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                self.student_model = ModelRegistry.get_model(
                    model_type=self.student_model_type,
                    custom_params=self.student_params
                )
            self.best_params = self.student_params
        
        # Train the student model, suppressing warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            self.student_model.fit(X, y)
        
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability predictions from the student model.
        
        Args:
            X: Input features
            
        Returns:
            Probability predictions
        """
        if self.student_model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        return self.student_model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Get class predictions from the student model.
        
        Args:
            X: Input features
            
        Returns:
            Class predictions
        """
        if self.student_model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        return self.student_model.predict(X)

    def evaluate(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        return_predictions: bool = False
    ) -> dict:
        """
        Evaluate the student model performance using multiple metrics.
        
        Args:
            X: Input features
            y_true: True labels
            return_predictions: Whether to include predictions in the output
            
        Returns:
            Dictionary containing evaluation metrics and optionally predictions
        """
        if self.student_model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
            
        # Get predictions
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)[:, 1]  # Probability of positive class
        
        # Calculate metrics using Classification class
        metrics = self.metrics_calculator.calculate_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob
        )
        
        # Add KL divergence to metrics
        teacher_soft_labels = self._get_teacher_soft_labels(X)
        student_probs = self.predict_proba(X)
        kl_div = self._kl_divergence(teacher_soft_labels, student_probs)
        metrics['kl_divergence'] = kl_div
        
        # Add hyperparameter info
        metrics['best_params'] = self.best_params
        
        if return_predictions:
            # Create DataFrame with predictions
            predictions_df = pd.DataFrame({
                'y_true': y_true,
                'y_pred': y_pred,
                'y_prob': y_prob
            })
            return {'metrics': metrics, 'predictions': predictions_df}
        
        return metrics

    def evaluate_from_dataframe(
        self,
        data: pd.DataFrame,
        features_columns: list,
        target_column: str,
        return_predictions: bool = False
    ) -> dict:
        """
        Evaluate model using a DataFrame as input.
        
        Args:
            data: Input DataFrame
            features_columns: List of feature column names
            target_column: Name of the target column
            return_predictions: Whether to include predictions in the output
            
        Returns:
            Dictionary containing evaluation metrics and optionally predictions
        """
        X = data[features_columns].values
        y_true = data[target_column].values
        
        return self.evaluate(X, y_true, return_predictions=return_predictions)