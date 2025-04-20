"""
Base classes for all model distillation implementations.

This module provides the abstract base classes that define the interfaces
for all distillation techniques in the DeepBridge library.
"""

import abc
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List, Tuple

from deepbridge.utils.model_registry import ModelType


class BaseDistiller(abc.ABC):
    """
    Abstract base class for all distillation techniques.
    
    This class defines the common interface that all model distillers must implement,
    ensuring consistent behavior across different distillation approaches.
    """
    
    def __init__(
        self,
        teacher_model=None, 
        student_model_type: ModelType = None,
        student_params: Optional[Dict[str, Any]] = None,
        random_state: Optional[int] = 42
    ):
        """
        Initialize the base distiller.
        
        Args:
            teacher_model: Pre-trained teacher model (optional)
            student_model_type: Type of student model to use
            student_params: Custom parameters for student model
            random_state: Random seed for reproducibility
        """
        self.teacher_model = teacher_model
        self.student_model_type = student_model_type
        self.student_params = student_params or {}
        self.random_state = random_state
        self.student_model = None
        self.is_fitted = False
        self.best_params = None
    
    @abc.abstractmethod
    def fit(
        self, 
        X: Union[np.ndarray, pd.DataFrame], 
        y: Union[np.ndarray, pd.Series],
        verbose: bool = True
    ) -> 'BaseDistiller':
        """
        Train the student model using the distillation technique.
        
        This method must be implemented by all concrete distiller classes.
        
        Args:
            X: Input features
            y: Target values
            verbose: Whether to display training progress
            
        Returns:
            self: The trained distiller instance
        """
        pass
    
    @abc.abstractmethod
    def predict(
        self, 
        X: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """
        Make predictions using the trained student model.
        
        This method must be implemented by all concrete distiller classes.
        
        Args:
            X: Input features
            
        Returns:
            Probability predictions
        """
        pass
    
    def predict_proba(
        self, 
        X: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """
        Get probability predictions from the student model.
        
        This method is optional for distillers that only provide class predictions.
        By default, it delegates to predict() method.
        
        Args:
            X: Input features
            
        Returns:
            Probability predictions for all classes
        """
        return self.predict(X)
    
    def evaluate(
        self,
        X: Union[np.ndarray, pd.DataFrame], 
        y_true: Union[np.ndarray, pd.Series],
        teacher_probs: Optional[Union[np.ndarray, pd.DataFrame]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the student model and compute comparison metrics.
        
        This is a common implementation that can be overridden by specific distillers
        if they need custom evaluation logic.
        
        Args:
            X: Input features
            y_true: True labels
            teacher_probs: Optional teacher model probabilities for comparison
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        # Use Classification metrics calculator if available
        try:
            from deepbridge.metrics.classification import Classification
            metrics_calculator = Classification()
            
            # Get student predictions
            student_probs = self.predict_proba(X)
            student_preds = (student_probs[:, 1] > 0.5).astype(int) if student_probs.shape[1] > 1 else \
                            (student_probs > 0.5).astype(int)
            
            # Calculate metrics
            metrics = metrics_calculator.calculate_metrics(
                y_true=y_true,
                y_pred=student_preds,
                y_prob=student_probs[:, 1] if student_probs.shape[1] > 1 else student_probs,
                teacher_prob=teacher_probs
            )
            
            # Add best parameters if available
            if hasattr(self, 'best_params') and self.best_params:
                metrics['best_params'] = self.best_params
                
            return metrics
            
        except ImportError:
            # Fallback to basic metrics if Classification module not available
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            # Get student predictions
            student_probs = self.predict_proba(X)
            student_preds = (student_probs[:, 1] > 0.5).astype(int) if student_probs.shape[1] > 1 else \
                            (student_probs > 0.5).astype(int)
            
            # Calculate basic metrics
            metrics = {
                'accuracy': float(accuracy_score(y_true, student_preds)),
                'precision': float(precision_score(y_true, student_preds)),
                'recall': float(recall_score(y_true, student_preds)),
                'f1_score': float(f1_score(y_true, student_preds))
            }
            
            # Add best parameters if available
            if hasattr(self, 'best_params') and self.best_params:
                metrics['best_params'] = self.best_params
                
            return metrics
    
    @classmethod
    @abc.abstractmethod
    def from_probabilities(
        cls,
        probabilities: Union[np.ndarray, pd.DataFrame],
        student_model_type: ModelType = None,
        student_params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> 'BaseDistiller':
        """
        Create a distiller from pre-calculated probabilities.
        
        This class method must be implemented by all concrete distiller classes
        to support initialization from pre-computed probability distributions.
        
        Args:
            probabilities: Array or DataFrame with teacher probabilities
            student_model_type: Type of student model to use
            student_params: Custom parameters for student model
            **kwargs: Additional parameters specific to each distillation technique
            
        Returns:
            BaseDistiller: New distiller instance
        """
        pass
    
    def __repr__(self) -> str:
        """String representation of the distiller."""
        status = "fitted" if self.is_fitted else "not fitted"
        return f"{self.__class__.__name__}(student_model_type={self.student_model_type}, {status})"