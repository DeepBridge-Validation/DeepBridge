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
    log_loss,
    r2_score
)
from scipy.special import kl_div
from scipy import stats
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
                # Print debug information about the input shapes and types
                print(f"Teacher prob type: {type(teacher_prob)}, shape: {getattr(teacher_prob, 'shape', 'unknown')}")
                print(f"Student prob type: {type(y_prob)}, shape: {getattr(y_prob, 'shape', 'unknown')}")
                
                # Ensure we're working with numpy arrays
                if isinstance(teacher_prob, pd.Series):
                    teacher_prob = teacher_prob.values
                if isinstance(y_prob, pd.Series):
                    y_prob = y_prob.values
                    
                # Print first few values for debugging
                print(f"Teacher prob first 5 values: {teacher_prob[:5]}")
                print(f"Student prob first 5 values: {y_prob[:5]}")
                
                # Calculate KL divergence
                metrics['kl_divergence'] = Classification.calculate_kl_divergence(
                    teacher_prob, y_prob
                )
                
                # Calculate KS statistic with detailed error handling
                try:
                    ks_result = Classification.calculate_ks_statistic(teacher_prob, y_prob)
                    metrics['ks_statistic'], metrics['ks_pvalue'] = ks_result
                    print(f"KS calculation successful: {ks_result}")
                except Exception as ks_error:
                    print(f"Error in KS statistic calculation: {str(ks_error)}")
                    import traceback
                    print(traceback.format_exc())
                    metrics['ks_statistic'] = None
                    metrics['ks_pvalue'] = None
                
                # Calculate R² with detailed error handling
                try:
                    r2 = Classification.calculate_r2_score(teacher_prob, y_prob)
                    metrics['r2_score'] = r2
                    print(f"R² calculation successful: {r2}")
                except Exception as r2_error:
                    print(f"Error in R² calculation: {str(r2_error)}")
                    import traceback
                    print(traceback.format_exc())
                    metrics['r2_score'] = None
                
            except Exception as e:
                print(f"Error calculating distribution comparison metrics: {str(e)}")
                import traceback
                print(traceback.format_exc())
                metrics['kl_divergence'] = None
                metrics['ks_statistic'] = None
                metrics['ks_pvalue'] = None
                metrics['r2_score'] = None
                
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
    
    @staticmethod
    def calculate_ks_statistic(
        teacher_prob: t.Union[np.ndarray, pd.Series],
        student_prob: t.Union[np.ndarray, pd.Series]
    ) -> t.Tuple[float, float]:
        """
        Calculate Kolmogorov-Smirnov statistic between teacher and student probability distributions.
        
        Args:
            teacher_prob: Teacher model probabilities
            student_prob: Student model probabilities
            
        Returns:
            Tuple[float, float]: KS statistic and p-value
        """
        # Convert inputs to numpy arrays if they're pandas Series or other types
        if not isinstance(teacher_prob, np.ndarray):
            print(f"Converting teacher_prob from {type(teacher_prob)} to numpy array")
            teacher_prob = np.array(teacher_prob)
        if not isinstance(student_prob, np.ndarray):
            print(f"Converting student_prob from {type(student_prob)} to numpy array")
            student_prob = np.array(student_prob)
            
        # For binary classification, we only need the probability of positive class
        if len(teacher_prob.shape) > 1:
            print(f"Extracting positive class from teacher_prob with shape {teacher_prob.shape}")
            teacher_prob = teacher_prob[:, 1]  # Probability of positive class
        if len(student_prob.shape) > 1:
            print(f"Extracting positive class from student_prob with shape {student_prob.shape}")
            student_prob = student_prob[:, 1]  # Probability of positive class
        
        # Verify that we have valid input data
        if np.isnan(teacher_prob).any() or np.isnan(student_prob).any():
            print("Warning: NaN values found in probability arrays")
            # Remove NaN values
            valid_indices = ~(np.isnan(teacher_prob) | np.isnan(student_prob))
            teacher_prob = teacher_prob[valid_indices]
            student_prob = student_prob[valid_indices]
            
        if len(teacher_prob) == 0 or len(student_prob) == 0:
            print("Error: Empty arrays after cleaning")
            return 0.0, 1.0  # Return default values indicating no difference
            
        # Calculate KS statistic and p-value
        try:
            ks_stat, p_value = stats.ks_2samp(teacher_prob, student_prob)
            return float(ks_stat), float(p_value)
        except Exception as e:
            print(f"Exception in KS calculation: {str(e)}")
            print(f"teacher_prob stats: min={np.min(teacher_prob)}, max={np.max(teacher_prob)}, "
                 f"mean={np.mean(teacher_prob)}, has_nan={np.isnan(teacher_prob).any()}")
            print(f"student_prob stats: min={np.min(student_prob)}, max={np.max(student_prob)}, "
                 f"mean={np.mean(student_prob)}, has_nan={np.isnan(student_prob).any()}")
            raise
    
    @staticmethod
    def calculate_r2_score(
        teacher_prob: t.Union[np.ndarray, pd.Series],
        student_prob: t.Union[np.ndarray, pd.Series]
    ) -> float:
        """
        Calculate R² between teacher and student probability distributions.
        
        Args:
            teacher_prob: Teacher model probabilities
            student_prob: Student model probabilities
            
        Returns:
            float: R² score
        """
        # Convert inputs to numpy arrays if they're pandas Series or other types
        if not isinstance(teacher_prob, np.ndarray):
            print(f"Converting teacher_prob from {type(teacher_prob)} to numpy array for R²")
            teacher_prob = np.array(teacher_prob)
        if not isinstance(student_prob, np.ndarray):
            print(f"Converting student_prob from {type(student_prob)} to numpy array for R²")
            student_prob = np.array(student_prob)
            
        # For binary classification, we only need the probability of positive class
        if len(teacher_prob.shape) > 1:
            print(f"Extracting positive class from teacher_prob with shape {teacher_prob.shape} for R²")
            teacher_prob = teacher_prob[:, 1]  # Probability of positive class
        if len(student_prob.shape) > 1:
            print(f"Extracting positive class from student_prob with shape {student_prob.shape} for R²")
            student_prob = student_prob[:, 1]  # Probability of positive class
        
        # Verify that we have valid input data
        if np.isnan(teacher_prob).any() or np.isnan(student_prob).any():
            print("Warning: NaN values found in probability arrays for R² calculation")
            # Remove NaN values
            valid_indices = ~(np.isnan(teacher_prob) | np.isnan(student_prob))
            teacher_prob = teacher_prob[valid_indices]
            student_prob = student_prob[valid_indices]
            
        if len(teacher_prob) == 0 or len(student_prob) == 0:
            print("Error: Empty arrays after cleaning for R² calculation")
            return 0.0  # Return default value indicating no correlation
            
        try:    
            # Sort distributions to compare in a way that measures shape similarity
            teacher_sorted = np.sort(teacher_prob)
            student_sorted = np.sort(student_prob)
            
            # Print sorted distribution statistics for debugging
            print(f"Sorted teacher dist - min: {np.min(teacher_sorted)}, max: {np.max(teacher_sorted)}, "
                 f"length: {len(teacher_sorted)}")
            print(f"Sorted student dist - min: {np.min(student_sorted)}, max: {np.max(student_sorted)}, "
                 f"length: {len(student_sorted)}")
            
            # Ensure equal length by truncating the longer one
            min_len = min(len(teacher_sorted), len(student_sorted))
            teacher_sorted = teacher_sorted[:min_len]
            student_sorted = student_sorted[:min_len]
            
            # Calculate R² score
            r2 = r2_score(teacher_sorted, student_sorted)
            print(f"R² calculation result: {r2}")
            
            return float(r2)
        except Exception as e:
            print(f"Exception in R² calculation: {str(e)}")
            import traceback
            print(traceback.format_exc())
            print(f"teacher_prob stats: shape={teacher_prob.shape}, "
                 f"min={np.min(teacher_prob) if len(teacher_prob) > 0 else 'N/A'}, "
                 f"max={np.max(teacher_prob) if len(teacher_prob) > 0 else 'N/A'}")
            print(f"student_prob stats: shape={student_prob.shape}, "
                 f"min={np.min(student_prob) if len(student_prob) > 0 else 'N/A'}, "
                 f"max={np.max(student_prob) if len(student_prob) > 0 else 'N/A'}")
            return None