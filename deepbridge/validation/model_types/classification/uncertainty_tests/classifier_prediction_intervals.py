"""
Prediction interval and uncertainty tools for classification models.

This module provides tools for estimating and evaluating prediction
uncertainty in classification models, including calibration methods
and confidence intervals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split


class ClassificationUncertaintyTests:
    """
    Uncertainty tests for classification models.
    
    This class provides methods for evaluating and improving the
    uncertainty estimates (probability calibration) of classification models.
    """
    
    def __init__(
        self,
        model: Any,
        calibration_method: str = 'isotonic',
        n_bins: int = 10,
        random_state: Optional[int] = None,
        verbose: bool = False
    ):
        """
        Initialize the classification uncertainty tests.
        
        Parameters:
        -----------
        model : Any
            Classification model to test
        calibration_method : str
            Method for probability calibration:
            - 'isotonic': Isotonic regression (non-parametric, monotonic)
            - 'sigmoid': Platt scaling (parametric, sigmoid function)
            - 'none': No calibration
        n_bins : int
            Number of bins for calibration evaluation
        random_state : int or None
            Random seed for reproducibility
        verbose : bool
            Whether to print progress information
        """
        self.model = model
        self.calibration_method = calibration_method
        self.n_bins = n_bins
        self.random_state = random_state
        self.verbose = verbose
        
        # Check if model has predict_proba method
        if not hasattr(model, 'predict_proba'):
            raise ValueError("Model must have predict_proba method for uncertainty tests")
            
        # Initialize calibrated model
        self.calibrated_model = None
        
        # Initialize results storage
        self.results = {}
    
    def calibrate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        cv: int = 5,
        **kwargs
    ) -> Any:
        """
        Calibrate model probabilities.
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Feature data
        y : array-like or Series
            Target data
        cv : int
            Number of cross-validation folds for calibration
        **kwargs : dict
            Additional parameters
            
        Returns:
        --------
        Any : Calibrated model
        """
        # Skip calibration if method is 'none'
        if self.calibration_method == 'none':
            self.calibrated_model = self.model
            return self.model
            
        # Create calibrated model
        calibrated = CalibratedClassifierCV(
            self.model,
            method=self.calibration_method,
            cv=cv,
            n_jobs=-1,
            ensemble=True
        )
        
        # Convert y to numpy array if needed
        if isinstance(y, pd.Series):
            y_values = y.values
        else:
            y_values = y
            
        # Fit calibration model
        try:
            calibrated.fit(X, y_values)
            self.calibrated_model = calibrated
            
            if self.verbose:
                print(f"Calibrated model using {self.calibration_method} method with {cv}-fold CV")
                
            return calibrated
        except Exception as e:
            if self.verbose:
                print(f"Error calibrating model: {str(e)}")
                print("Falling back to original model")
                
            self.calibrated_model = self.model
            return self.model
    
    def evaluate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate model calibration and uncertainty estimates.
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Feature data
        y : array-like or Series
            Target data
        **kwargs : dict
            Additional parameters
            
        Returns:
        --------
        dict : Evaluation results
        """
        # Convert y to numpy array if needed
        if isinstance(y, pd.Series):
            y_values = y.values
        else:
            y_values = y
            
        # Get model to evaluate
        model_to_evaluate = self.calibrated_model if self.calibrated_model is not None else self.model
        
        # Get predictions
        try:
            y_prob = model_to_evaluate.predict_proba(X)
            
            # For binary classification, use probability of positive class
            if y_prob.shape[1] == 2:
                y_prob_pos = y_prob[:, 1]
            else:
                # For multiclass, we'll evaluate overall calibration
                # and per-class calibration separately
                y_prob_pos = None
        except Exception as e:
            if self.verbose:
                print(f"Error making predictions: {str(e)}")
            return {'error': str(e)}
        
        # Initialize results
        results = {
            'calibration_method': self.calibration_method,
            'metrics': {},
            'calibration_curve': {},
            'per_class': {}
        }
        
        # Calculate overall metrics
        # Brier score (mean squared error of probabilistic predictions)
        if y_prob.shape[1] == 2:
            # Binary case - use probability of positive class
            brier_score = brier_score_loss(y_values, y_prob[:, 1])
            results['metrics']['brier_score'] = float(brier_score)
        else:
            # Multiclass case - use one-hot encoding
            from sklearn.preprocessing import label_binarize
            classes = np.unique(y_values)
            y_bin = label_binarize(y_values, classes=classes)
            
            # Only compute if we have binary classes
            if y_bin.shape[1] > 1:
                brier_scores = []
                for i in range(y_bin.shape[1]):
                    brier_scores.append(brier_score_loss(y_bin[:, i], y_prob[:, i]))
                brier_score = np.mean(brier_scores)
                results['metrics']['brier_score'] = float(brier_score)
            
        # Log loss (negative log likelihood)
        try:
            log_loss_value = log_loss(y_values, y_prob)
            results['metrics']['log_loss'] = float(log_loss_value)
        except:
            # Skip if error occurs
            pass
            
        # Expected Calibration Error (ECE)
        if y_prob.shape[1] == 2:
            # Binary classification
            ece, calibration_curve = _compute_calibration_curve(
                y_values, y_prob[:, 1], n_bins=self.n_bins
            )
            results['metrics']['expected_calibration_error'] = float(ece)
            results['calibration_curve'] = calibration_curve
        else:
            # Multiclass classification
            # Compute per-class calibration
            classes = np.unique(y_values)
            
            all_eces = []
            for i, cls in enumerate(classes):
                # Convert to binary problem (one-vs-rest)
                y_binary = (y_values == cls).astype(int)
                
                # Get probability for this class
                y_prob_cls = y_prob[:, i]
                
                # Compute calibration curve
                ece, calibration_curve = _compute_calibration_curve(
                    y_binary, y_prob_cls, n_bins=self.n_bins
                )
                
                # Store results
                all_eces.append(ece)
                results['per_class'][str(cls)] = {
                    'expected_calibration_error': float(ece),
                    'calibration_curve': calibration_curve
                }
                
            # Average ECE across classes
            results['metrics']['expected_calibration_error'] = float(np.mean(all_eces))
            
        # Compute confidence intervals for predictions
        confidence_intervals = self._compute_confidence_intervals(X)
        results['confidence_intervals'] = confidence_intervals
        
        # Store results
        self.results = results
        
        return results
    
    def _compute_confidence_intervals(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Compute confidence intervals for predictions.
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Feature data
        alpha : float
            Confidence level (e.g., 0.05 for 95% confidence)
            
        Returns:
        --------
        dict : Confidence interval information
        """
        # Get model to evaluate
        model_to_evaluate = self.calibrated_model if self.calibrated_model is not None else self.model
        
        # Get predictions
        try:
            y_prob = model_to_evaluate.predict_proba(X)
            
            # For binary classification, focus on positive class
            if y_prob.shape[1] == 2:
                confidences = np.max(y_prob, axis=1)
                predictions = np.argmax(y_prob, axis=1)
                
                # Map back to original class labels if possible
                if hasattr(model_to_evaluate, 'classes_'):
                    predicted_classes = model_to_evaluate.classes_[predictions]
                else:
                    predicted_classes = predictions
                    
                # Simple confidence intervals:
                # Return predictions with confidence above threshold
                threshold = 1 - alpha
                
                return {
                    'predicted_classes': predicted_classes.tolist() if hasattr(predicted_classes, 'tolist') else predicted_classes,
                    'confidences': confidences.tolist(),
                    'threshold': threshold,
                    'high_confidence_mask': (confidences >= threshold).tolist()
                }
            else:
                # For multiclass, compute top classes and their probabilities
                n_classes = min(3, y_prob.shape[1])  # Top 3 or fewer classes
                
                # Get indices of top N classes for each sample
                top_indices = np.argsort(-y_prob, axis=1)[:, :n_classes]
                
                # Get probabilities of top classes
                top_probabilities = np.take_along_axis(y_prob, top_indices, axis=1)
                
                # Map indices to class labels if possible
                if hasattr(model_to_evaluate, 'classes_'):
                    classes = model_to_evaluate.classes_
                    top_classes = np.take(classes, top_indices)
                else:
                    top_classes = top_indices
                    
                # Compute prediction sets (classes with cumulative probability >= 1-alpha)
                prediction_sets = []
                for i in range(len(X)):
                    # Get probabilities in descending order
                    probs = top_probabilities[i]
                    
                    # Compute cumulative sum
                    cumsum = np.cumsum(probs)
                    
                    # Find smallest set with cumulative probability >= 1-alpha
                    set_size = np.searchsorted(cumsum, 1 - alpha) + 1
                    set_size = min(set_size, n_classes)  # Cap at n_classes
                    
                    prediction_sets.append(set_size)
                    
                return {
                    'top_classes': top_classes.tolist() if hasattr(top_classes, 'tolist') else top_classes,
                    'top_probabilities': top_probabilities.tolist(),
                    'prediction_set_sizes': prediction_sets,
                    'alpha': alpha
                }
                
        except Exception as e:
            if self.verbose:
                print(f"Error computing confidence intervals: {str(e)}")
            return {'error': str(e)}
    
    def plot_calibration_curve(
        self,
        class_index: Optional[int] = None,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot calibration curve.
        
        Parameters:
        -----------
        class_index : int or None
            Index of class to plot (None for binary classification or average)
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.Figure : Calibration curve plot
        """
        if not self.results:
            raise ValueError("No evaluation results available. Call evaluate() first.")
            
        # Get calibration curve data
        if class_index is not None:
            # Get calibration curve for specific class
            class_key = str(class_index)
            if 'per_class' not in self.results or class_key not in self.results['per_class']:
                raise ValueError(f"No results available for class {class_index}")
                
            calibration_data = self.results['per_class'][class_key]['calibration_curve']
            title_suffix = f" (Class {class_index})"
        else:
            # Get overall calibration curve
            if 'calibration_curve' not in self.results:
                raise ValueError("No calibration curve data available")
                
            calibration_data = self.results['calibration_curve']
            title_suffix = ""
            
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract data
        bin_edges = calibration_data['bin_edges']
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Plot calibration curve
        ax.plot(
            bin_centers,
            calibration_data['true_fractions'],
            's-',
            label=f'Calibration Curve (ECE={calibration_data["expected_calibration_error"]:.4f})'
        )
        
        # Plot identity line (perfect calibration)
        ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
        
        # Plot confidence histogram
        ax_twin = ax.twinx()
        bin_heights = calibration_data['bin_counts'] / np.sum(calibration_data['bin_counts'])
        bin_width = bin_edges[1] - bin_edges[0]
        
        ax_twin.bar(
            bin_centers,
            bin_heights,
            width=bin_width,
            alpha=0.3,
            color='gray',
            label='Prediction Distribution'
        )
        ax_twin.set_ylim(0, max(bin_heights) * 1.2)
        ax_twin.set_ylabel('Fraction of Samples')
        
        # Set labels and title
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('True Fraction')
        ax.set_title(f'Calibration Curve{title_suffix}')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        return fig
    
    def plot_confidence_distribution(
        self,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot distribution of prediction confidences.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.Figure : Confidence distribution plot
        """
        if not self.results or 'confidence_intervals' not in self.results:
            raise ValueError("No confidence interval data available. Call evaluate() first.")
            
        # Get confidence data
        confidence_data = self.results['confidence_intervals']
        
        # Check if we have confidence values
        if 'confidences' not in confidence_data:
            raise ValueError("No confidence values found in results")
            
        confidences = confidence_data['confidences']
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot histogram
        ax.hist(confidences, bins=20, alpha=0.7, edgecolor='black')
        
        # Add threshold line if available
        if 'threshold' in confidence_data:
            threshold = confidence_data['threshold']
            ax.axvline(x=threshold, color='red', linestyle='--', 
                     label=f'Threshold ({threshold:.2f})')
            
        # Set labels and title
        ax.set_xlabel('Prediction Confidence')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Prediction Confidences')
        
        # Add legend if we added a threshold line
        if 'threshold' in confidence_data:
            ax.legend()
            
        return fig


def calibrate_classifier_probabilities(
    model: Any,
    X_train: Union[np.ndarray, pd.DataFrame],
    y_train: Union[np.ndarray, pd.Series],
    X_test: Union[np.ndarray, pd.DataFrame],
    method: str = 'isotonic',
    cv: int = 5,
    random_state: Optional[int] = None
) -> Tuple[Any, np.ndarray]:
    """
    Calibrate classifier probabilities.
    
    Parameters:
    -----------
    model : Any
        Classification model to calibrate
    X_train : array-like or DataFrame
        Training feature data
    y_train : array-like or Series
        Training target data
    X_test : array-like or DataFrame
        Test feature data
    method : str
        Calibration method: 'isotonic', 'sigmoid', or 'none'
    cv : int
        Number of cross-validation folds
    random_state : int or None
        Random seed for reproducibility
        
    Returns:
    --------
    tuple : (calibrated_model, calibrated_probabilities)
    """
    # Check if model has predict_proba method
    if not hasattr(model, 'predict_proba'):
        raise ValueError("Model must have predict_proba method for calibration")
        
    # Skip calibration if method is 'none'
    if method == 'none':
        return model, model.predict_proba(X_test)
        
    # Create calibrated model
    calibrated = CalibratedClassifierCV(
        model,
        method=method,
        cv=cv,
        n_jobs=-1,
        ensemble=True
    )
    
    # Convert y to numpy array if needed
    if isinstance(y_train, pd.Series):
        y_train_values = y_train.values
    else:
        y_train_values = y_train
        
    # Fit calibration model
    try:
        calibrated.fit(X_train, y_train_values)
        
        # Get calibrated probabilities
        calibrated_probs = calibrated.predict_proba(X_test)
        
        return calibrated, calibrated_probs
        
    except Exception as e:
        # Fall back to original model if calibration fails
        return model, model.predict_proba(X_test)


def evaluate_calibration_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Dict[str, float]:
    """
    Evaluate calibration metrics.
    
    Parameters:
    -----------
    y_true : array-like or Series
        True target values
    y_prob : array-like
        Predicted probabilities
    n_bins : int
        Number of bins for calibration evaluation
        
    Returns:
    --------
    dict : Calibration metrics
    """
    # Convert y to numpy array if needed
    if isinstance(y_true, pd.Series):
        y_true_values = y_true.values
    else:
        y_true_values = y_true
        
    # Handle binary vs multi-class
    if y_prob.ndim == 1 or y_prob.shape[1] == 2:
        # Binary classification or probability of positive class
        if y_prob.ndim == 2:
            y_prob_pos = y_prob[:, 1]
        else:
            y_prob_pos = y_prob
            
        # Calculate metrics
        # Brier score
        brier = brier_score_loss(y_true_values, y_prob_pos)
        
        # Log loss
        try:
            if y_prob.ndim == 1:
                # Convert to 2D array for log_loss
                y_prob_2d = np.vstack([1 - y_prob_pos, y_prob_pos]).T
                ll = log_loss(y_true_values, y_prob_2d)
            else:
                ll = log_loss(y_true_values, y_prob)
        except:
            ll = np.nan
            
        # Expected Calibration Error
        ece, _ = _compute_calibration_curve(y_true_values, y_prob_pos, n_bins=n_bins)
        
        return {
            'brier_score': float(brier),
            'log_loss': float(ll),
            'expected_calibration_error': float(ece)
        }
    else:
        # Multiclass classification
        from sklearn.preprocessing import label_binarize
        
        # Get unique classes
        classes = np.unique(y_true_values)
        
        # Convert to one-hot encoding
        y_true_bin = label_binarize(y_true_values, classes=classes)
        
        # Calculate metrics
        # Brier score
        brier_scores = []
        ece_scores = []
        
        for i in range(len(classes)):
            # Brier score for this class
            brier_scores.append(brier_score_loss(y_true_bin[:, i], y_prob[:, i]))
            
            # ECE for this class
            ece, _ = _compute_calibration_curve(y_true_bin[:, i], y_prob[:, i], n_bins=n_bins)
            ece_scores.append(ece)
            
        # Average scores
        brier = np.mean(brier_scores)
        ece = np.mean(ece_scores)
        
        # Log loss
        try:
            ll = log_loss(y_true_values, y_prob)
        except:
            ll = np.nan
            
        return {
            'brier_score': float(brier),
            'log_loss': float(ll),
            'expected_calibration_error': float(ece)
        }


def compute_confidence_intervals(
    model: Any,
    X: Union[np.ndarray, pd.DataFrame],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Compute confidence intervals for classifier predictions.
    
    Parameters:
    -----------
    model : Any
        Classification model
    X : array-like or DataFrame
        Feature data
    alpha : float
        Confidence level (e.g., 0.05 for 95% confidence)
        
    Returns:
    --------
    dict : Confidence interval information
    """
    # Check if model has predict_proba method
    if not hasattr(model, 'predict_proba'):
        raise ValueError("Model must have predict_proba method for confidence intervals")
        
    # Get predictions
    y_prob = model.predict_proba(X)
    
    # For binary classification, focus on positive class
    if y_prob.shape[1] == 2:
        confidences = np.max(y_prob, axis=1)
        predictions = np.argmax(y_prob, axis=1)
        
        # Map back to original class labels if possible
        if hasattr(model, 'classes_'):
            predicted_classes = model.classes_[predictions]
        else:
            predicted_classes = predictions
            
        # Simple confidence intervals:
        # Return predictions with confidence above threshold
        threshold = 1 - alpha
        
        return {
            'predicted_classes': predicted_classes.tolist() if hasattr(predicted_classes, 'tolist') else predicted_classes,
            'confidences': confidences.tolist(),
            'threshold': threshold,
            'high_confidence_mask': (confidences >= threshold).tolist()
        }
    else:
        # For multiclass, compute top classes and their probabilities
        n_classes = min(3, y_prob.shape[1])  # Top 3 or fewer classes
        
        # Get indices of top N classes for each sample
        top_indices = np.argsort(-y_prob, axis=1)[:, :n_classes]
        
        # Get probabilities of top classes
        top_probabilities = np.take_along_axis(y_prob, top_indices, axis=1)
        
        # Map indices to class labels if possible
        if hasattr(model, 'classes_'):
            classes = model.classes_
            top_classes = np.take(classes, top_indices)
        else:
            top_classes = top_indices
            
        # Compute prediction sets (classes with cumulative probability >= 1-alpha)
        prediction_sets = []
        for i in range(len(X)):
            # Get probabilities in descending order
            probs = top_probabilities[i]
            
            # Compute cumulative sum
            cumsum = np.cumsum(probs)
            
            # Find smallest set with cumulative probability >= 1-alpha
            set_size = np.searchsorted(cumsum, 1 - alpha) + 1
            set_size = min(set_size, n_classes)  # Cap at n_classes
            
            prediction_sets.append(set_size)
            
        return {
            'top_classes': top_classes.tolist() if hasattr(top_classes, 'tolist') else top_classes,
            'top_probabilities': top_probabilities.tolist(),
            'prediction_set_sizes': prediction_sets,
            'alpha': alpha
        }


def plot_calibration_curve(
    y_true: Union[np.ndarray, pd.Series],
    y_prob: np.ndarray,
    n_bins: int = 10,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot calibration curve.
    
    Parameters:
    -----------
    y_true : array-like or Series
        True target values
    y_prob : array-like
        Predicted probabilities
    n_bins : int
        Number of bins for calibration evaluation
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.Figure : Calibration curve plot
    """
    # Convert y to numpy array if needed
    if isinstance(y_true, pd.Series):
        y_true_values = y_true.values
    else:
        y_true_values = y_true
        
    # Handle binary vs multi-class
    if y_prob.ndim == 1 or y_prob.shape[1] == 2:
        # Binary classification or probability of positive class
        if y_prob.ndim == 2:
            y_prob_pos = y_prob[:, 1]
        else:
            y_prob_pos = y_prob
            
        # Compute calibration curve
        ece, calibration_data = _compute_calibration_curve(
            y_true_values, y_prob_pos, n_bins=n_bins
        )
    else:
        # Multiclass - focus on class 0 for plotting
        # You could adapt this to plot a specific class or average
        y_prob_pos = y_prob[:, 0]
        
        # Convert to binary problem (one-vs-rest)
        if hasattr(y_true_values, 'shape') and len(y_true_values.shape) > 1:
            # Already one-hot encoded
            y_binary = y_true_values[:, 0]
        else:
            # Convert to binary
            y_binary = (y_true_values == 0).astype(int)
            
        # Compute calibration curve
        ece, calibration_data = _compute_calibration_curve(
            y_binary, y_prob_pos, n_bins=n_bins
        )
        
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract data
    bin_edges = calibration_data['bin_edges']
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Plot calibration curve
    ax.plot(
        bin_centers,
        calibration_data['true_fractions'],
        's-',
        label=f'Calibration Curve (ECE={ece:.4f})'
    )
    
    # Plot identity line (perfect calibration)
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    
    # Plot confidence histogram
    ax_twin = ax.twinx()
    bin_heights = calibration_data['bin_counts'] / np.sum(calibration_data['bin_counts'])
    bin_width = bin_edges[1] - bin_edges[0]
    
    ax_twin.bar(
        bin_centers,
        bin_heights,
        width=bin_width,
        alpha=0.3,
        color='gray',
        label='Prediction Distribution'
    )
    ax_twin.set_ylim(0, max(bin_heights) * 1.2)
    ax_twin.set_ylabel('Fraction of Samples')
    
    # Set labels and title
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('True Fraction')
    ax.set_title('Calibration Curve')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    return fig


def _compute_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute calibration curve and Expected Calibration Error (ECE).
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_prob : array-like
        Predicted probabilities
    n_bins : int
        Number of bins
        
    Returns:
    --------
    tuple : (expected_calibration_error, calibration_data)
    """
    # Create bins and digitize probabilities
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.clip(np.digitize(y_prob, bin_edges) - 1, 0, n_bins - 1)
    
    # Calculate statistics for each bin
    bin_sums = np.bincount(bin_indices, weights=y_prob, minlength=n_bins)
    bin_true_sums = np.bincount(bin_indices, weights=y_true, minlength=n_bins)
    bin_counts = np.bincount(bin_indices, minlength=n_bins)
    
    # Calculate mean predicted probability and true fraction in each bin
    bin_means = np.zeros(n_bins)
    bin_true_fractions = np.zeros(n_bins)
    
    nonzero_mask = bin_counts > 0
    bin_means[nonzero_mask] = bin_sums[nonzero_mask] / bin_counts[nonzero_mask]
    bin_true_fractions[nonzero_mask] = bin_true_sums[nonzero_mask] / bin_counts[nonzero_mask]
    
    # Calculate ECE
    ece = np.sum(bin_counts * np.abs(bin_means - bin_true_fractions)) / np.sum(bin_counts)
    
    # Format results
    calibration_data = {
        'bin_edges': bin_edges,
        'bin_counts': bin_counts,
        'mean_predicted_probs': bin_means,
        'true_fractions': bin_true_fractions,
        'expected_calibration_error': float(ece)
    }
    
    return float(ece), calibration_data