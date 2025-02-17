import typing as t
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    accuracy_score,
    recall_score,
    f1_score
)

class Classification:
    """
    Calculates evaluation metrics for binary classification models.
    """
    
    @staticmethod
    def calculate_metrics(
        y_true: t.Union[np.ndarray, pd.Series],
        y_pred: t.Union[np.ndarray, pd.Series],
        y_prob: t.Optional[t.Union[np.ndarray, pd.Series]] = None
    ) -> dict:
        """
        Calculate multiple evaluation metrics.
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
            except ValueError as e:
                print(f"Error calculating AUC/PR: {str(e)}")
                metrics['auc_roc'] = None
                metrics['auc_pr'] = None
        
        return metrics
    
    @staticmethod
    def calculate_metrics_from_predictions(
        data: pd.DataFrame,
        target_column: str,
        pred_column: str,
        prob_column: t.Optional[str] = None
    ) -> dict:
        """
        Calculates metrics using DataFrame columns.
        
        Args:
            data: DataFrame containing the predictions
            target_column: Name of the column with ground truth values
            pred_column: Name of the column with binary predictions
            prob_column: Name of the column with probabilities (optional)
            
        Returns:
            dict: Dictionary containing the calculated metrics
        """
        y_true = data[target_column]
        y_pred = data[pred_column]
        y_prob = data[prob_column] if prob_column else None
        
        return Classification.calculate_metrics(y_true, y_pred, y_prob)