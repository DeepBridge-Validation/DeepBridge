import typing as t
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score
)


class Regression:
    """
    Calculates evaluation metrics for regression models.
    """
    
    @staticmethod
    def calculate_metrics(
        y_true: t.Union[np.ndarray, pd.Series],
        y_pred: t.Union[np.ndarray, pd.Series],
        teacher_pred: t.Optional[t.Union[np.ndarray, pd.Series]] = None
    ) -> dict:
        """
        Calculate multiple evaluation metrics for regression.
        
        Args:
            y_true: Ground truth (correct) target values
            y_pred: Predicted values 
            teacher_pred: Teacher model predictions (optional, for comparison)
            
        Returns:
            dict: Dictionary containing calculated metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['mse'] = float(mean_squared_error(y_true, y_pred))
        metrics['rmse'] = float(np.sqrt(metrics['mse']))
        metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
        metrics['r2'] = float(r2_score(y_true, y_pred))
        metrics['explained_variance'] = float(explained_variance_score(y_true, y_pred))
        
        # Calculate additional metrics if teacher model predictions are provided
        if teacher_pred is not None:
            try:
                # Ensure we're working with numpy arrays
                if isinstance(teacher_pred, pd.Series):
                    teacher_pred = teacher_pred.values
                if isinstance(y_pred, pd.Series):
                    y_pred = y_pred.values
                
                # Calculate R² between teacher and student predictions
                metrics['teacher_student_r2'] = float(r2_score(teacher_pred, y_pred))
                
                # Calculate the MSE between teacher and student predictions
                metrics['teacher_student_mse'] = float(mean_squared_error(teacher_pred, y_pred))
                
                # Calculate the correlation coefficient between teacher and student predictions
                metrics['teacher_student_corr'] = float(np.corrcoef(teacher_pred, y_pred)[0, 1])
                
            except Exception as e:
                print(f"Error calculating comparison metrics: {str(e)}")
                metrics['teacher_student_r2'] = None
                metrics['teacher_student_mse'] = None
                metrics['teacher_student_corr'] = None
                
        return metrics
    
    @staticmethod
    def calculate_metrics_from_predictions(
        data: pd.DataFrame,
        target_column: str,
        pred_column: str,
        teacher_pred_column: t.Optional[str] = None
    ) -> dict:
        """
        Calculates metrics using DataFrame columns.
        
        Args:
            data: DataFrame containing the predictions
            target_column: Name of the column with ground truth values
            pred_column: Name of the column with predictions
            teacher_pred_column: Name of the column with teacher predictions (optional)
            
        Returns:
            dict: Dictionary containing the calculated metrics
        """
        y_true = data[target_column]
        y_pred = data[pred_column]
        teacher_pred = data[teacher_pred_column] if teacher_pred_column else None
        
        return Regression.calculate_metrics(y_true, y_pred, teacher_pred)