import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import r2_score
from abc import ABC

class BaseVisualizer(ABC):
    """
    Base class for all visualizers with common utility functions.
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize base visualizer.
        
        Args:
            output_dir: Directory to save visualization plots
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set Seaborn styling
        sns.set_theme(style="darkgrid")
    
    def _process_probabilities(self, probs):
        """
        Process probabilities to extract positive class probabilities and ensure correct format.
        
        Args:
            probs: Input probabilities (DataFrame, Series, or ndarray)
            
        Returns:
            numpy.ndarray: Processed probability array for the positive class
        """
        # Handle pandas DataFrame
        if isinstance(probs, pd.DataFrame):
            # Check for specific probability columns
            if 'prob_class_1' in probs.columns:
                return probs['prob_class_1'].values
            elif 'prob_1' in probs.columns:
                return probs['prob_1'].values
            elif 'class_1_prob' in probs.columns:
                return probs['class_1_prob'].values
            # If no specific columns found, use the last column
            return probs.iloc[:, -1].values
        
        # Handle pandas Series
        if isinstance(probs, pd.Series):
            return probs.values
        
        # Handle numpy arrays
        if isinstance(probs, np.ndarray):
            # Extract positive class for 2D arrays
            if len(probs.shape) > 1 and probs.shape[1] > 1:
                return probs[:, 1]
            return probs
        
        # If we get here, input format is not recognized
        raise ValueError(f"Unrecognized probability format: {type(probs)}")
    
    def _calculate_metrics(self, teacher_probs, student_probs):
        """
        Calculate distribution similarity metrics.
        
        Args:
            teacher_probs: Teacher model probabilities
            student_probs: Student model probabilities
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # KL Divergence
        try:
            epsilon = 1e-10
            teacher_probs_clip = np.clip(teacher_probs, epsilon, 1-epsilon)
            student_probs_clip = np.clip(student_probs, epsilon, 1-epsilon)
            
            # Create histograms with same bins for both distributions
            bins = np.linspace(0, 1, 50)
            teacher_hist, _ = np.histogram(teacher_probs_clip, bins=bins, density=True)
            student_hist, _ = np.histogram(student_probs_clip, bins=bins, density=True)
            
            # Add small epsilon to avoid division by zero
            teacher_hist = teacher_hist + epsilon
            student_hist = student_hist + epsilon
            
            # Normalize
            teacher_hist = teacher_hist / teacher_hist.sum()
            student_hist = student_hist / student_hist.sum()
            
            # Calculate KL divergence
            kl_div = np.sum(teacher_hist * np.log(teacher_hist / student_hist))
            metrics['kl_divergence'] = float(kl_div)
            
            # Calculate Jensen-Shannon divergence (symmetric)
            m = 0.5 * (teacher_hist + student_hist)
            js_div = 0.5 * np.sum(teacher_hist * np.log(teacher_hist / m)) + \
                     0.5 * np.sum(student_hist * np.log(student_hist / m))
            metrics['jensen_shannon'] = float(js_div)
            
        except Exception as e:
            print(f"Error calculating KL divergence: {str(e)}")
            metrics['kl_divergence'] = float('nan')
            metrics['jensen_shannon'] = float('nan')
        
        # KS statistic
        try:
            ks_stat, ks_pvalue = stats.ks_2samp(teacher_probs, student_probs)
            metrics['ks_statistic'] = float(ks_stat)
            metrics['ks_pvalue'] = float(ks_pvalue)
        except Exception as e:
            print(f"Error calculating KS statistic: {str(e)}")
            metrics['ks_statistic'] = float('nan')
            metrics['ks_pvalue'] = float('nan')
        
        # R² score (using sorted distributions)
        try:
            # Sort both distributions to compare shape rather than correlation
            teacher_sorted = np.sort(teacher_probs)
            student_sorted = np.sort(student_probs)
            
            # Ensure equal length by sampling or truncating
            if len(teacher_sorted) != len(student_sorted):
                min_len = min(len(teacher_sorted), len(student_sorted))
                teacher_sorted = teacher_sorted[:min_len]
                student_sorted = student_sorted[:min_len]
                
            metrics['r2_score'] = float(r2_score(teacher_sorted, student_sorted))
        except Exception as e:
            print(f"Error calculating R² score: {str(e)}")
            metrics['r2_score'] = float('nan')
            
        return metrics