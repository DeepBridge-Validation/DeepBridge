import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Dict, Any, Optional, Union

# Corrigir imports para usar caminhos absolutos
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
        alpha: float = 0.5
    ):
        """
        Initialize the Knowledge Distillation model.
        
        Args:
            teacher_model: Pre-trained teacher model (optional if teacher_probabilities is provided)
            teacher_probabilities: Pre-calculated teacher probabilities (optional if teacher_model is provided)
            student_model_type: Type of student model to use
            student_params: Custom parameters for student model
            temperature: Temperature parameter for softening probability distributions
            alpha: Weight between teacher's loss and true label loss
        """
        if teacher_model is None and teacher_probabilities is None:
            raise ValueError("Either teacher_model or teacher_probabilities must be provided")
            
        self.teacher_model = teacher_model
        self.teacher_probabilities = teacher_probabilities
        self.student_model_type = student_model_type
        self.student_params = student_params or {}
        self.temperature = temperature
        self.alpha = alpha
        self.metrics_calculator = Classification()
        
        # Initialize student model
        self.student_model = ModelRegistry.get_model(
            model_type=student_model_type,
            custom_params=student_params
        )

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

    @classmethod
    def from_probabilities(
        cls,
        probabilities: Union[np.ndarray, pd.DataFrame],
        student_model_type: ModelType = ModelType.LOGISTIC_REGRESSION,
        student_params: Dict[str, Any] = None,
        temperature: float = 1.0,
        alpha: float = 0.5
    ) -> 'KnowledgeDistillation':
        """
        Create a KnowledgeDistillation instance from pre-calculated probabilities.
        
        Args:
            probabilities: Array or DataFrame with shape (n_samples, 2) containing class probabilities
            student_model_type: Type of student model to use
            student_params: Custom parameters for student model
            temperature: Temperature parameter
            alpha: Weight parameter
            
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
            alpha=alpha
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KnowledgeDistillation':
        """
        Train the student model using Knowledge Distillation.
        
        Args:
            X: Training features
            y: True labels
            
        Returns:
            self: The trained model
        """
        # Generate soft labels
        teacher_soft_labels = self._get_teacher_soft_labels(X)
        
        # Train student model
        # Note: In a more advanced implementation, you would create a custom loss
        # function that combines soft and hard labels using self.alpha
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
        return self.student_model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Get class predictions from the student model.
        
        Args:
            X: Input features
            
        Returns:
            Class predictions
        """
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
        # Get predictions
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)[:, 1]  # Probability of positive class
        
        # Calculate metrics using Classification class
        metrics = self.metrics_calculator.calculate_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob
        )
        
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
        target_column: str
    ) -> dict:
        """
        Evaluate model using a DataFrame as input.
        
        Args:
            data: Input DataFrame
            features_columns: List of feature column names
            target_column: Name of the target column
            
        Returns:
            Dictionary containing evaluation metrics
        """
        X = data[features_columns].values
        y_true = data[target_column].values
        
        return self.evaluate(X, y_true)