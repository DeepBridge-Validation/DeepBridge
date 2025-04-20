"""
Probability calibration tools for uncertainty estimation.

This module provides tools for calibrating model probabilities and
analyzing calibration quality.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class CalibrationAnalyzer:
    """
    Analyze and calibrate model probability predictions.
    
    This class provides methods for assessing probability calibration
    and applying calibration techniques.
    """
    
    def __init__(
        self,
        model: Any,
        method: str = 'histogram',
        n_bins: int = 10,
        verbose: bool = False
    ):
        """
        Initialize the calibration analyzer.
        
        Parameters:
        -----------
        model : Any
            Trained machine learning model
        method : str
            Method for calibration analysis:
            - 'histogram': Binning-based method
            - 'isotonic': Isotonic regression
            - 'platt': Platt scaling (logistic regression)
        n_bins : int
            Number of bins for histogram method
        verbose : bool
            Whether to print progress information
        """
        self.model = model
        self.method = method
        self.n_bins = n_bins
        self.verbose = verbose
        
        # Check if model has predict_proba method
        if not hasattr(model, 'predict_proba'):
            raise ValueError("Model must have predict_proba method")
            
        self.calibrators = {}
    
    def analyze_calibration(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        class_index: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analyze calibration of probability predictions.
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Input features
        y : array-like or Series
            Binary or multiclass targets
        class_index : int or None
            Index of class to analyze (None for binary classification)
            
        Returns:
        --------
        dict : Calibration analysis results
        """
        # Convert y to numpy array if it's a Series
        if isinstance(y, pd.Series):
            y_values = y.values
        else:
            y_values = y
            
        # Get model probabilities
        probs = self.model.predict_proba(X)
        
        # For binary classification with no class_index specified
        if class_index is None and probs.shape[1] == 2:
            class_index = 1
            
        # Get probabilities for the specified class
        if class_index is not None:
            class_probs = probs[:, class_index]
            # Convert to binary
            y_binary = (y_values == self.model.classes_[class_index]).astype(int)
        else:
            # For binary classification
            class_probs = probs[:, 1]
            y_binary = y_values
            
        # Compute metrics
        from .uncertainty_metrics import (
            expected_calibration_error,
            maximum_calibration_error,
            brier_score,
            negative_log_likelihood
        )
        
        ece = expected_calibration_error(y_binary, class_probs, n_bins=self.n_bins)
        mce = maximum_calibration_error(y_binary, class_probs, n_bins=self.n_bins)
        bs = brier_score(y_binary, class_probs)
        nll = negative_log_likelihood(y_binary, class_probs)
        
        # Calculate calibration curve
        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        bin_indices = np.digitize(class_probs, bin_edges[1:-1])
        
        bin_sums = np.bincount(bin_indices, weights=class_probs, minlength=self.n_bins)
        bin_counts = np.bincount(bin_indices, minlength=self.n_bins)
        bin_true = np.bincount(bin_indices, weights=y_binary, minlength=self.n_bins)
        
        # Calculate mean predicted probability and true fraction in each bin
        mean_pred_probs = bin_sums / (bin_counts + 1e-8)
        true_fractions = bin_true / (bin_counts + 1e-8)
        
        # Get bin centers
        bin_centers = np.linspace(0, 1, self.n_bins)
        
        return {
            'ece': ece,
            'mce': mce,
            'brier_score': bs,
            'negative_log_likelihood': nll,
            'bin_centers': bin_centers,
            'mean_pred_probs': mean_pred_probs,
            'true_fractions': true_fractions,
            'bin_counts': bin_counts,
            'class_index': class_index,
            'n_bins': self.n_bins
        }
    
    def calibrate(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        class_index: Optional[int] = None
    ) -> None:
        """
        Calibrate probability predictions.
        
        Parameters:
        -----------
        X_train : array-like or DataFrame
            Training features
        y_train : array-like or Series
            Training targets
        class_index : int or None
            Index of class to calibrate (None for all classes)
        """
        # Convert y to numpy array if it's a Series
        if isinstance(y_train, pd.Series):
            y_values = y_train.values
        else:
            y_values = y_train
            
        # Get model probabilities
        probs = self.model.predict_proba(X_train)
        
        # Get unique classes
        classes = np.unique(y_values)
        
        # Determine classes to calibrate
        if class_index is not None:
            classes_to_calibrate = [class_index]
        else:
            classes_to_calibrate = range(len(self.model.classes_))
            
        # Train calibrator for each class
        for idx in classes_to_calibrate:
            # Get probabilities for this class
            class_probs = probs[:, idx]
            
            # Create binary target
            y_binary = (y_values == self.model.classes_[idx]).astype(int)
            
            # Train calibrator
            if self.method == 'isotonic':
                # Isotonic regression
                calibrator = IsotonicRegression(out_of_bounds='clip')
                calibrator.fit(class_probs, y_binary)
            elif self.method == 'platt':
                # Platt scaling
                calibrator = LogisticRegression(C=1.0, solver='lbfgs')
                calibrator.fit(class_probs.reshape(-1, 1), y_binary)
            else:  # histogram or other methods are non-parametric
                calibrator = None
                
            # Store calibrator
            self.calibrators[idx] = calibrator
            
        if self.verbose:
            print(f"Calibrated {len(self.calibrators)} classes using {self.method} method")
    
    def calibrate_probabilities(
        self,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """
        Apply calibration to model probabilities.
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Input features
            
        Returns:
        --------
        numpy.ndarray : Calibrated probabilities
        """
        if not self.calibrators:
            raise ValueError("Model must be calibrated first")
            
        # Get model probabilities
        probs = self.model.predict_proba(X)
        
        # Initialize calibrated probabilities
        calibrated = np.zeros_like(probs)
        
        for idx, calibrator in self.calibrators.items():
            # Get probabilities for this class
            class_probs = probs[:, idx]
            
            if calibrator is not None:
                if self.method == 'isotonic':
                    # Apply isotonic regression
                    calibrated[:, idx] = calibrator.transform(class_probs)
                elif self.method == 'platt':
                    # Apply Platt scaling
                    calibrated[:, idx] = calibrator.predict_proba(
                        class_probs.reshape(-1, 1))[:, 1]
            else:
                # For non-parametric methods or no calibration
                calibrated[:, idx] = class_probs
                
        # Normalize rows to sum to 1
        row_sums = calibrated.sum(axis=1)
        calibrated = calibrated / row_sums[:, np.newaxis]
        
        return calibrated
    
    def plot_calibration(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        class_index: Optional[int] = None,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot calibration curve.
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Input features
        y : array-like or Series
            Target values
        class_index : int or None
            Index of class to plot (None for binary classification)
        figsize : tuple
            Figure size (width, height)
            
        Returns:
        --------
        matplotlib.Figure : Calibration plot
        """
        # Analyze calibration
        result = self.analyze_calibration(X, y, class_index)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot diagonal (perfect calibration)
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        
        # Plot calibration curve
        ax.plot(
            result['mean_pred_probs'],
            result['true_fractions'],
            'o-',
            label=f'Calibration Curve (ECE = {result["ece"]:.4f})'
        )
        
        # Add histogram of predictions
        twin_ax = ax.twinx()
        twin_ax.set_yticks([])
        hist_heights = result['bin_counts'] / np.max(result['bin_counts']) * 0.3
        twin_ax.bar(result['bin_centers'], hist_heights, alpha=0.3, 
                 width=1/result['n_bins'], align='center', color='gray',
                 label='Prediction Distribution')
        
        # Set labels and title
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Empirical Probability')
        ax.set_title('Calibration Curve')
        
        class_info = f" (Class {result['class_index']})" if result['class_index'] is not None else ""
        ax.set_title(f'Calibration Curve{class_info}')
        
        # Add metrics to the plot
        metrics_text = (
            f"ECE = {result['ece']:.4f}\n"
            f"MCE = {result['mce']:.4f}\n"
            f"Brier Score = {result['brier_score']:.4f}\n"
            f"NLL = {result['negative_log_likelihood']:.4f}"
        )
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
             bbox=dict(facecolor='white', alpha=0.8),
             verticalalignment='top', fontsize=10)
        
        # Set limits
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # Add legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = twin_ax.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
        
        plt.tight_layout()
        return fig


def calibrate_probabilities(
    model: Any,
    X_train: Union[np.ndarray, pd.DataFrame],
    y_train: Union[np.ndarray, pd.Series],
    X_test: Union[np.ndarray, pd.DataFrame],
    method: str = 'isotonic'
) -> np.ndarray:
    """
    Calibrate model probability predictions.
    
    Parameters:
    -----------
    model : Any
        Trained machine learning model
    X_train : array-like or DataFrame
        Training features
    y_train : array-like or Series
        Training targets
    X_test : array-like or DataFrame
        Test features
    method : str
        Calibration method to use
        
    Returns:
    --------
    numpy.ndarray : Calibrated probabilities
    """
    analyzer = CalibrationAnalyzer(model=model, method=method)
    analyzer.calibrate(X_train, y_train)
    return analyzer.calibrate_probabilities(X_test)


def isotonic_calibration(
    model: Any,
    X_train: Union[np.ndarray, pd.DataFrame],
    y_train: Union[np.ndarray, pd.Series],
    X_test: Union[np.ndarray, pd.DataFrame]
) -> np.ndarray:
    """
    Calibrate probabilities using isotonic regression.
    
    Parameters:
    -----------
    model : Any
        Trained machine learning model
    X_train : array-like or DataFrame
        Training features
    y_train : array-like or Series
        Training targets
    X_test : array-like or DataFrame
        Test features
        
    Returns:
    --------
    numpy.ndarray : Calibrated probabilities
    """
    return calibrate_probabilities(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        method='isotonic'
    )


def platt_scaling(
    model: Any,
    X_train: Union[np.ndarray, pd.DataFrame],
    y_train: Union[np.ndarray, pd.Series],
    X_test: Union[np.ndarray, pd.DataFrame]
) -> np.ndarray:
    """
    Calibrate probabilities using Platt scaling.
    
    Parameters:
    -----------
    model : Any
        Trained machine learning model
    X_train : array-like or DataFrame
        Training features
    y_train : array-like or Series
        Training targets
    X_test : array-like or DataFrame
        Test features
        
    Returns:
    --------
    numpy.ndarray : Calibrated probabilities
    """
    return calibrate_probabilities(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        method='platt'
    )