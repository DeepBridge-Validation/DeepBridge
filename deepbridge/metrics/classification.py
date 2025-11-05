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
        import warnings
        import logging
        logger = logging.getLogger("deepbridge.metrics")

        # Check number of unique classes
        unique_classes_true = np.unique(y_true)
        unique_classes_pred = np.unique(y_pred)

        # Accuracy can always be calculated
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))

        # Precision, recall, and F1 may be undefined for single-class cases
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress sklearn warnings

            try:
                # Use zero_division parameter to handle edge cases gracefully
                metrics['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
                metrics['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
                f1_value = float(f1_score(y_true, y_pred, zero_division=0))
                metrics['f1_score'] = f1_value
                metrics['f1-score'] = f1_value  # Also store with hyphen for compatibility

                # Log if we have single-class scenario
                if len(unique_classes_true) == 1:
                    logger.debug(f"Single class in y_true: {unique_classes_true[0]}. Metrics may be limited.")
                if len(unique_classes_pred) == 1:
                    logger.debug(f"Single class in predictions: {unique_classes_pred[0]}.")

            except Exception as e:
                logger.debug(f"Error calculating precision/recall/f1: {e}")
                metrics['precision'] = 0.0
                metrics['recall'] = 0.0
                metrics['f1_score'] = 0.0
                metrics['f1-score'] = 0.0
        
        # Metrics requiring probabilities
        if y_prob is not None:
            # Check if we have both classes in y_true
            unique_classes = np.unique(y_true)
            if len(unique_classes) < 2:
                # Single class - metrics are undefined
                import logging
                logger = logging.getLogger("deepbridge.metrics")
                logger.debug(f"Only {len(unique_classes)} class(es) found in y_true. Some metrics will be undefined.")
                metrics['roc_auc'] = None
                metrics['auc_roc'] = None
                metrics['auc_pr'] = None
                metrics['log_loss'] = None
            else:
                try:
                    # Calculate AUC-ROC and store with both naming conventions for compatibility
                    auc_value = float(roc_auc_score(y_true, y_prob))
                    metrics['roc_auc'] = auc_value
                    metrics['auc_roc'] = auc_value  # Also store with alternate name for compatibility
                    metrics['auc_pr'] = float(average_precision_score(y_true, y_prob))
                    metrics['log_loss'] = float(log_loss(y_true, y_prob))
                except ValueError as e:
                    # Log as debug instead of print to avoid cluttering output
                    import logging
                    logger = logging.getLogger("deepbridge.metrics")
                    logger.debug(f"Error calculating AUC/PR/log_loss: {str(e)}")
                    metrics['roc_auc'] = None
                    metrics['auc_roc'] = None
                    metrics['auc_pr'] = None
                    metrics['log_loss'] = None
        
        # Calculate KL divergence if teacher probabilities are provided
        if teacher_prob is not None and y_prob is not None:
            try:
                # Ensure we're working with numpy arrays
                if isinstance(teacher_prob, pd.Series):
                    teacher_prob = teacher_prob.values
                if isinstance(y_prob, pd.Series):
                    y_prob = y_prob.values
                
                # Calculate KL divergence
                metrics['kl_divergence'] = Classification.calculate_kl_divergence(
                    teacher_prob, y_prob
                )
                
                # Calculate KS statistic with error handling
                try:
                    ks_result = Classification.calculate_ks_statistic(teacher_prob, y_prob)
                    metrics['ks_statistic'], metrics['ks_pvalue'] = ks_result
                except Exception:
                    metrics['ks_statistic'] = None
                    metrics['ks_pvalue'] = None
                
                # Calculate R² with error handling
                try:
                    r2 = Classification.calculate_r2_score(teacher_prob, y_prob)
                    metrics['r2_score'] = r2
                except Exception:
                    metrics['r2_score'] = None
                
            except Exception:
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
            teacher_prob = np.array(teacher_prob)
        if not isinstance(student_prob, np.ndarray):
            student_prob = np.array(student_prob)
            
        # For binary classification, we only need the probability of positive class
        if len(teacher_prob.shape) > 1:
            teacher_prob = teacher_prob[:, 1]  # Probability of positive class
        if len(student_prob.shape) > 1:
            student_prob = student_prob[:, 1]  # Probability of positive class
        
        # Verify that we have valid input data
        if np.isnan(teacher_prob).any() or np.isnan(student_prob).any():
            # Remove NaN values
            valid_indices = ~(np.isnan(teacher_prob) | np.isnan(student_prob))
            teacher_prob = teacher_prob[valid_indices]
            student_prob = student_prob[valid_indices]
            
        if len(teacher_prob) == 0 or len(student_prob) == 0:
            return 0.0, 1.0  # Return default values indicating no difference
            
        # Calculate KS statistic and p-value
        try:
            ks_stat, p_value = stats.ks_2samp(teacher_prob, student_prob)
            return float(ks_stat), float(p_value)
        except Exception:
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
            teacher_prob = np.array(teacher_prob)
        if not isinstance(student_prob, np.ndarray):
            student_prob = np.array(student_prob)
            
        # For binary classification, we only need the probability of positive class
        if len(teacher_prob.shape) > 1:
            teacher_prob = teacher_prob[:, 1]  # Probability of positive class
        if len(student_prob.shape) > 1:
            student_prob = student_prob[:, 1]  # Probability of positive class
        
        # Verify that we have valid input data
        if np.isnan(teacher_prob).any() or np.isnan(student_prob).any():
            # Remove NaN values
            valid_indices = ~(np.isnan(teacher_prob) | np.isnan(student_prob))
            teacher_prob = teacher_prob[valid_indices]
            student_prob = student_prob[valid_indices]
            
        if len(teacher_prob) == 0 or len(student_prob) == 0:
            return 0.0  # Return default value indicating no correlation
            
        try:    
            # Sort distributions to compare in a way that measures shape similarity
            teacher_sorted = np.sort(teacher_prob)
            student_sorted = np.sort(student_prob)
            
            # Ensure equal length by truncating the longer one
            min_len = min(len(teacher_sorted), len(student_sorted))
            teacher_sorted = teacher_sorted[:min_len]
            student_sorted = student_sorted[:min_len]
            
            # Calculate R² score
            r2 = r2_score(teacher_sorted, student_sorted)
            
            return float(r2)
        except Exception:
            return None