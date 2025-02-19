import typing as t
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    accuracy_score,
    recall_score,
    f1_score,
    log_loss
)
from scipy.special import kl_div
import numpy as np


class Classification:
    """
    Calculates evaluation metrics for binary classification models.
    """
    
    @staticmethod
    def calculate_metrics(
        y_true: t.Union[np.ndarray, pd.Series],
        y_pred: t.Union[np.ndarray, pd.Series],
        y_prob: t.Optional[t.Union[np.ndarray, pd.Series]] = None,
        teacher_prob: t.Optional[t.Union[np.ndarray, pd.Series]] = None
    ) -> dict:
        """
        Calculate multiple evaluation metrics.
        
        Args:
            y_true: Ground truth (correct) target values
            y_pred: Binary prediction values 
            y_prob: Predicted probabilities (required for AUC metrics)
            teacher_prob: Teacher model probabilities (required for KL divergence)
            
        Returns:
            dict: Dictionary containing calculated metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
        metrics['precision'] = float(precision_score(y_true, y_pred))
        metrics['recall'] = float(recall_score(y_true, y_pred))
        metrics['f1_score'] = float(f1_score(y_true, y_pred))
        
        # Metrics requiring probabilities
        if y_prob is not None:
            try:
                metrics['auc_roc'] = float(roc_auc_score(y_true, y_prob))
                metrics['auc_pr'] = float(average_precision_score(y_true, y_prob))
                metrics['log_loss'] = float(log_loss(y_true, y_prob))
            except ValueError as e:
                print(f"Error calculating AUC/PR/log_loss: {str(e)}")
                metrics['auc_roc'] = None
                metrics['auc_pr'] = None
                metrics['log_loss'] = None
        
        # Calculate KL divergence if teacher probabilities are provided
        if teacher_prob is not None and y_prob is not None:
            try:
                metrics['kl_divergence'] = Classification.calculate_kl_divergence(
                    teacher_prob, y_prob
                )
            except Exception as e:
                print(f"Error calculating KL divergence: {str(e)}")
                metrics['kl_divergence'] = None
                
        return metrics
    
    @staticmethod
    def calculate_metrics_from_predictions(
        data: pd.DataFrame,
        target_column: str,
        pred_column: str,
        prob_column: t.Optional[str] = None,
        teacher_prob_column: t.Optional[str] = None
    ) -> dict:
        """
        Calculates metrics using DataFrame columns.
        
        Args:
            data: DataFrame containing the predictions
            target_column: Name of the column with ground truth values
            pred_column: Name of the column with binary predictions
            prob_column: Name of the column with probabilities (optional)
            teacher_prob_column: Name of the column with teacher probabilities (optional)
            
        Returns:
            dict: Dictionary containing the calculated metrics
        """
        y_true = data[target_column]
        y_pred = data[pred_column]
        y_prob = data[prob_column] if prob_column else None
        teacher_prob = data[teacher_prob_column] if teacher_prob_column else None
        
        return Classification.calculate_metrics(y_true, y_pred, y_prob, teacher_prob)
    
    @staticmethod
    def calculate_kl_divergence(
        p: t.Union[np.ndarray, pd.Series],
        q: t.Union[np.ndarray, pd.Series]
    ) -> float:
        """
        Calculate KL divergence between two probability distributions.
        
        Args:
            p: Teacher model probabilities (reference distribution)
            q: Student model probabilities (approximating distribution)
            
        Returns:
            float: KL divergence value
        """
        # Convert inputs to numpy arrays if they're pandas Series
        if isinstance(p, pd.Series):
            p = p.values
        if isinstance(q, pd.Series):
            q = q.values
            
        # Clip probabilities to avoid log(0) errors
        epsilon = 1e-10
        p = np.clip(p, epsilon, 1.0 - epsilon)
        q = np.clip(q, epsilon, 1.0 - epsilon)
        
        # For binary classification, we need to consider both classes
        if len(p.shape) == 1:
            # Convert to two-class format
            p_two_class = np.vstack([1 - p, p]).T
            q_two_class = np.vstack([1 - q, q]).T
            
            # Calculate KL divergence
            kl = np.sum(kl_div(p_two_class, q_two_class), axis=1).mean()
        else:
            # Multi-class format is already provided
            kl = np.sum(kl_div(p, q), axis=1).mean()
            
        return float(kl)