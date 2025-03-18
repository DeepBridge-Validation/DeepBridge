import numpy as np
from typing import Callable, Dict, Union, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# Define metric functions
CLASSIFICATION_METRICS = {
    'ACC': accuracy_score,
    'ACCURACY': accuracy_score,
    'AUC': roc_auc_score,
    'PRECISION': precision_score,
    'RECALL': recall_score,
    'F1': f1_score
}

REGRESSION_METRICS = {
    'MSE': mean_squared_error,
    'MAE': mean_absolute_error,
    'R2': r2_score
}

def get_metric_function(metric_name: str, is_classification: bool) -> Callable:
    """
    Return the metric function corresponding to the given name.
    
    Args:
        metric_name: Name of the metric
        is_classification: Indicates if it's a classification problem
        
    Returns:
        Callable: Metric function
        
    Raises:
        ValueError: If metric_name is not available for the problem type
    """
    # Normalize metric name
    metric_name = metric_name.upper()
    
    if is_classification:
        if metric_name not in CLASSIFICATION_METRICS:
            raise ValueError(f"Metric '{metric_name}' not available for classification. Options: {list(CLASSIFICATION_METRICS.keys())}")
        return CLASSIFICATION_METRICS[metric_name]
    else:
        if metric_name not in REGRESSION_METRICS:
            raise ValueError(f"Metric '{metric_name}' not available for regression. Options: {list(REGRESSION_METRICS.keys())}")
        return REGRESSION_METRICS[metric_name]

def is_metric_higher_better(metric_name: str) -> bool:
    """
    Determine if higher values of the metric indicate better performance.
    
    Args:
        metric_name: Name of the metric
        
    Returns:
        bool: True if higher is better, False otherwise
    """
    # Normalize metric name
    metric_name = metric_name.upper()
    
    # Metrics where lower is better
    worse_higher = {'MSE', 'MAE'}
    
    return metric_name not in worse_higher

class RobustnessScore:
    """
    Calculate and aggregate robustness scores from perturbation test results.
    """
    
    @staticmethod
    def calculate_area_under_curve(
        perturb_sizes: np.ndarray,
        scores: np.ndarray,
        higher_is_better: bool = True
    ) -> float:
        """
        Calculate area under curve for robustness evaluation.
        
        Args:
            perturb_sizes: Array of perturbation sizes
            scores: Array of performance scores at each perturbation size
            higher_is_better: Whether higher scores indicate better performance
            
        Returns:
            float: Normalized area (0-1) where higher is always better robustness
        """
        # Sort inputs by perturbation size
        sorted_idx = np.argsort(perturb_sizes)
        x = perturb_sizes[sorted_idx]
        y = scores[sorted_idx]
        
        # Calculate area under curve using trapezoidal rule
        area = np.trapz(y, x)
        
        # Normalize by x range
        x_range = x[-1] - x[0]
        if x_range > 0:
            normalized_area = area / x_range
        else:
            normalized_area = area
        
        # For metrics where lower is better (e.g., MSE), invert the score
        # so that higher always means better robustness
        if not higher_is_better:
            # Find reasonable bounds for normalization
            y_min, y_max = np.min(y), np.max(y)
            y_range = y_max - y_min
            if y_range > 0:
                return 1.0 - ((normalized_area - y_min) / y_range)
            else:
                return 0.5  # Neutral score if all values are the same
        
        return normalized_area
    
    @staticmethod
    def calculate_robustness_index(
        results: Dict,
        metric: str,
        perturb_threshold: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate robustness index for each model from test results.
        
        Args:
            results: Results from RobustnessTest.evaluate_robustness
            metric: Name of the metric used
            perturb_threshold: Optional threshold to focus on specific perturbation range
            
        Returns:
            Dict[str, float]: Robustness index for each model (higher is better)
        """
        higher_is_better = is_metric_higher_better(metric)
        
        indices = {}
        for model_name, model_results in results.items():
            perturb_sizes = np.array(model_results['perturb_sizes'])
            mean_scores = np.array(model_results['mean_scores'])
            
            # If threshold provided, filter to perturbations <= threshold
            if perturb_threshold is not None:
                mask = perturb_sizes <= perturb_threshold
                if not np.any(mask):
                    # No sizes below threshold, use all
                    mask = np.ones_like(perturb_sizes, dtype=bool)
                perturb_sizes = perturb_sizes[mask]
                mean_scores = mean_scores[mask]
            
            # Calculate robustness index
            robustness_idx = RobustnessScore.calculate_area_under_curve(
                perturb_sizes, mean_scores, higher_is_better
            )
            
            indices[model_name] = robustness_idx
        
        return indices