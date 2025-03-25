"""
Uncertainty validator for machine learning models.

This module provides a validator for assessing the quality of
uncertainty estimates in model predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
import matplotlib.pyplot as plt
import os
import warnings

from ..core.base_validator import BaseValidator
from ..utils.model_inspection import is_classifier, is_regressor, get_model_type
from ..utils.validation_utils import save_validation_results


class UncertaintyValidator(BaseValidator):
    """
    Validator for evaluating model uncertainty estimates.
    
    This class provides methods for assessing the quality of uncertainty
    estimates in model predictions, such as probability calibration and
    prediction intervals.
    """
    
    def __init__(
        self,
        model: Any,
        calibration_method: str = 'isotonic',
        n_bins: int = 10,
        cv_folds: int = 5,
        random_state: Optional[int] = None,
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize the uncertainty validator.
        
        Parameters:
        -----------
        model : Any
            Machine learning model to validate
        calibration_method : str
            Method for probability calibration
        n_bins : int
            Number of bins for calibration evaluation
        cv_folds : int
            Number of cross-validation folds for calibration
        random_state : int or None
            Random seed for reproducibility
        verbose : bool
            Whether to print progress information
        **kwargs : dict
            Additional parameters
        """
        super().__init__(model=model, **kwargs)
        
        self.calibration_method = calibration_method
        self.n_bins = n_bins
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.verbose = verbose
        
        # Determine model type
        self.model_type = kwargs.get('model_type', get_model_type(model))
        
        # Check if model supports uncertainty estimation
        if self.model_type == 'classifier' and not hasattr(model, 'predict_proba'):
            raise ValueError("Classifier must have predict_proba method for uncertainty validation")
            
        # Initialize results
        self.results = {}
        
        # Initialize calibrated model
        self.calibrated_model = None
    
    def validate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Validate model uncertainty estimates.
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Features to validate with
        y : array-like or Series
            Target values
        **kwargs : dict
            Additional parameters
            
        Returns:
        --------
        dict : Validation results
        """
        # Override parameters if provided
        calibration_method = kwargs.get('calibration_method', self.calibration_method)
        n_bins = kwargs.get('n_bins', self.n_bins)
        cv_folds = kwargs.get('cv_folds', self.cv_folds)
        
        # Start with clean results
        results = {
            'model_type': self.model_type,
            'calibration_method': calibration_method,
            'baseline': {},
            'calibration': {},
            'uncertainty_score': 0.0
        }
        
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X
            
        if isinstance(y, pd.Series):
            y_values = y.values
        else:
            y_values = y
            
        # Handle classification and regression differently
        if self.model_type == 'classifier':
            # Evaluate baseline calibration
            baseline_calibration = self._evaluate_calibration(X_values, y_values)
            results['baseline'] = baseline_calibration
            
            # Perform calibration
            self._calibrate_model(X_values, y_values, calibration_method, cv_folds)
            
            # Evaluate calibrated model
            if self.calibrated_model is not None:
                calibrated_metrics = self._evaluate_calibration(
                    X_values, y_values, self.calibrated_model
                )
                results['calibration'] = calibrated_metrics
                
                # Calculate improvement
                results['improvement'] = self._calculate_improvement(
                    baseline_calibration, calibrated_metrics
                )
                
            # Calculate uncertainty score
            uncertainty_score = self._calculate_uncertainty_score(results)
            results['uncertainty_score'] = uncertainty_score
            
        else:  # regressor
            # Check if model supports prediction intervals
            has_predict_interval = hasattr(self.model, 'predict_interval') or \
                                  hasattr(self.model, 'predict_quantile')
                                  
            if not has_predict_interval:
                # Try to use bootstrap for prediction intervals
                results['warning'] = "Model does not explicitly support prediction intervals. Using bootstrap approximation."
                
            # Evaluate regression uncertainty
            regression_uncertainty = self._evaluate_regression_uncertainty(X_values, y_values)
            results['baseline'] = regression_uncertainty
            
            # Calculate uncertainty score
            uncertainty_score = regression_uncertainty.get('interval_coverage', 0.0)
            results['uncertainty_score'] = uncertainty_score
        
        # Store results
        self.results = results
        
        return results
    
    def _evaluate_calibration(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model calibration.
        
        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            Target values
        model : Any or None
            Model to evaluate (None to use self.model)
            
        Returns:
        --------
        dict : Calibration metrics
        """
        # Use default model if not provided
        if model is None:
            model = self.model
            
        try:
            # Get probability predictions
            y_prob = model.predict_proba(X)
            
            # Calculate calibration metrics
            metrics = {}
            
            # Choose a scoring metric
            from sklearn.metrics import log_loss, brier_score_loss
            
            # Brier score
            if y_prob.shape[1] == 2:
                # Binary classification
                brier = brier_score_loss(y, y_prob[:, 1])
                metrics['brier_score'] = float(brier)
            else:
                # Multiclass classification - use one-vs-rest
                brier_scores = []
                for i in range(y_prob.shape[1]):
                    if i in y:  # Only consider classes that are present
                        y_binary = (y == i).astype(int)
                        brier_scores.append(brier_score_loss(y_binary, y_prob[:, i]))
                        
                if brier_scores:
                    metrics['brier_score'] = float(np.mean(brier_scores))
                    
            # Log loss
            try:
                metrics['log_loss'] = float(log_loss(y, y_prob))
            except:
                # Skip if error occurs
                pass
                
            # Expected Calibration Error
            calibration_curve = self._compute_calibration_curve(y, y_prob, self.n_bins)
            metrics['expected_calibration_error'] = float(calibration_curve['ece'])
            metrics['calibration_curve'] = calibration_curve
            
            return metrics
            
        except Exception as e:
            if self.verbose:
                print(f"Error evaluating calibration: {str(e)}")
            return {'error': str(e)}
    
    def _calibrate_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        method: str,
        cv_folds: int
    ) -> None:
        """
        Calibrate model probabilities.
        
        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            Target values
        method : str
            Calibration method
        cv_folds : int
            Number of cross-validation folds
        """
        # Skip if method is 'none'
        if method.lower() == 'none':
            self.calibrated_model = self.model
            return
            
        try:
            from sklearn.calibration import CalibratedClassifierCV
            
            # Create calibrated model
            calibrated = CalibratedClassifierCV(
                self.model,
                method=method,
                cv=cv_folds,
                n_jobs=-1,
                ensemble=True
            )
            
            # Fit calibration model
            calibrated.fit(X, y)
            
            # Store calibrated model
            self.calibrated_model = calibrated
            
            if self.verbose:
                print(f"Calibrated model using {method} method with {cv_folds}-fold CV")
                
        except Exception as e:
            if self.verbose:
                print(f"Error calibrating model: {str(e)}")
                print("Falling back to original model")
                
            self.calibrated_model = self.model
    
    def _compute_calibration_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int
    ) -> Dict[str, Any]:
        """
        Compute calibration curve and Expected Calibration Error (ECE).
        
        Parameters:
        -----------
        y_true : array-like
            True class labels
        y_prob : array-like
            Predicted probabilities
        n_bins : int
            Number of bins
            
        Returns:
        --------
        dict : Calibration curve data
        """
        # For binary classification, use probability of positive class
        if y_prob.shape[1] == 2:
            y_prob_pos = y_prob[:, 1]
            
            # Convert to binary problem if not already
            if len(np.unique(y_true)) > 2:
                # Assume class 1 is the positive class
                y_binary = (y_true == 1).astype(int)
            else:
                y_binary = y_true
                
            return self._compute_binary_calibration_curve(y_binary, y_prob_pos, n_bins)
        else:
            # For multiclass, compute average of one-vs-rest calibration curves
            ece_sum = 0.0
            bin_sums = np.zeros(n_bins)
            bin_true_sums = np.zeros(n_bins)
            bin_counts = np.zeros(n_bins)
            
            for i in range(y_prob.shape[1]):
                if i in y_true:  # Only consider classes that are present
                    # Get probability for this class
                    y_prob_class = y_prob[:, i]
                    
                    # Convert to binary problem
                    y_binary = (y_true == i).astype(int)
                    
                    # Compute calibration curve
                    cal_curve = self._compute_binary_calibration_curve(y_binary, y_prob_class, n_bins)
                    
                    # Accumulate ECE
                    ece_sum += cal_curve['ece']
                    
                    # Accumulate bin statistics
                    bin_sums += cal_curve['bin_sums']
                    bin_true_sums += cal_curve['bin_true_sums']
                    bin_counts += cal_curve['bin_counts']
                    
            # Calculate average ECE
            n_classes = len(np.unique(y_true))
            avg_ece = ece_sum / n_classes
            
            # Calculate bin statistics
            bin_probs = np.zeros(n_bins)
            bin_true_fractions = np.zeros(n_bins)
            
            mask = bin_counts > 0
            bin_probs[mask] = bin_sums[mask] / bin_counts[mask]
            bin_true_fractions[mask] = bin_true_sums[mask] / bin_counts[mask]
            
            # Create bin edges
            bin_edges = np.linspace(0, 1, n_bins + 1)
            
            return {
                'ece': avg_ece,
                'bin_edges': bin_edges.tolist(),
                'bin_probs': bin_probs.tolist(),
                'bin_true_fractions': bin_true_fractions.tolist(),
                'bin_counts': bin_counts.tolist(),
                'bin_sums': bin_sums.tolist(),
                'bin_true_sums': bin_true_sums.tolist()
            }
    
    def _compute_binary_calibration_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int
    ) -> Dict[str, Any]:
        """
        Compute calibration curve for binary classification.
        
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
        dict : Calibration curve data
        """
        # Create bins and digitize probabilities
        bin_edges = np.linspace(0, 1, n_bins + 1)
        binned = np.digitize(y_prob, bin_edges) - 1
        binned = np.clip(binned, 0, n_bins - 1)  # Clip to valid range
        
        # Calculate statistics for each bin
        bin_sums = np.bincount(binned, weights=y_prob, minlength=n_bins)
        bin_true_sums = np.bincount(binned, weights=y_true, minlength=n_bins)
        bin_counts = np.bincount(binned, minlength=n_bins)
        
        # Calculate mean predicted probability and true fraction in each bin
        bin_probs = np.zeros(n_bins)
        bin_true_fractions = np.zeros(n_bins)
        
        mask = bin_counts > 0
        bin_probs[mask] = bin_sums[mask] / bin_counts[mask]
        bin_true_fractions[mask] = bin_true_sums[mask] / bin_counts[mask]
        
        # Calculate ECE
        ece = np.sum(bin_counts * np.abs(bin_probs - bin_true_fractions)) / np.sum(bin_counts)
        
        return {
            'ece': float(ece),
            'bin_edges': bin_edges.tolist(),
            'bin_probs': bin_probs.tolist(),
            'bin_true_fractions': bin_true_fractions.tolist(),
            'bin_counts': bin_counts.tolist(),
            'bin_sums': bin_sums.tolist(),
            'bin_true_sums': bin_true_sums.tolist()
        }
    
    def _evaluate_regression_uncertainty(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate uncertainty in regression predictions.
        
        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            Target values
            
        Returns:
        --------
        dict : Regression uncertainty metrics
        """
        # Initialize results
        results = {}
        
        # Get point predictions
        y_pred = self.model.predict(X)
        
        # Calculate residuals
        residuals = y - y_pred
        
        # Calculate error metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        results['mse'] = float(mean_squared_error(y, y_pred))
        results['rmse'] = float(np.sqrt(results['mse']))
        results['mae'] = float(mean_absolute_error(y, y_pred))
        
        # Analyze residual distribution
        results['residual_mean'] = float(np.mean(residuals))
        results['residual_std'] = float(np.std(residuals))
        
        # Check for native prediction interval support
        if hasattr(self.model, 'predict_interval'):
            # Model has native prediction interval support
            try:
                intervals = self.model.predict_interval(X, alpha=0.05)
                
                # Calculate interval coverage
                coverage = np.mean((y >= intervals[:, 0]) & (y <= intervals[:, 1]))
                results['interval_coverage'] = float(coverage)
                
                # Calculate average interval width
                width = np.mean(intervals[:, 1] - intervals[:, 0])
                results['interval_width'] = float(width)
                
            except Exception as e:
                if self.verbose:
                    print(f"Error computing native prediction intervals: {str(e)}")
                    
                # Fall back to bootstrap approximation
                intervals = self._bootstrap_prediction_intervals(X, y)
                
                # Calculate metrics if bootstrap was successful
                if intervals is not None:
                    coverage = np.mean((y >= intervals[:, 0]) & (y <= intervals[:, 1]))
                    results['interval_coverage'] = float(coverage)
                    
                    width = np.mean(intervals[:, 1] - intervals[:, 0])
                    results['interval_width'] = float(width)
        else:
            # Use bootstrap approximation
            intervals = self._bootstrap_prediction_intervals(X, y)
            
            # Calculate metrics if bootstrap was successful
            if intervals is not None:
                coverage = np.mean((y >= intervals[:, 0]) & (y <= intervals[:, 1]))
                results['interval_coverage'] = float(coverage)
                
                width = np.mean(intervals[:, 1] - intervals[:, 0])
                results['interval_width'] = float(width)
                
        return results
    
    def _bootstrap_prediction_intervals(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_bootstrap: int = 100,
        alpha: float = 0.05
    ) -> Optional[np.ndarray]:
        """
        Generate bootstrap prediction intervals.
        
        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            Target values
        n_bootstrap : int
            Number of bootstrap samples
        alpha : float
            Significance level
            
        Returns:
        --------
        numpy.ndarray or None : Prediction intervals (lower, upper)
        """
        try:
            from sklearn.base import clone
            from sklearn.utils import resample
            
            # Get model class
            model_class = type(self.model)
            
            # Initialize predictions array
            n_samples = len(X)
            predictions = np.zeros((n_samples, n_bootstrap))
            
            # Generate bootstrap samples and predictions
            for i in range(n_bootstrap):
                # Resample training data
                X_boot, y_boot = resample(X, y, random_state=self.random_state + i if self.random_state else None)
                
                # Create and train bootstrap model
                boot_model = clone(self.model)
                boot_model.fit(X_boot, y_boot)
                
                # Make predictions
                predictions[:, i] = boot_model.predict(X)
                
            # Calculate prediction intervals
            lower_bound = np.percentile(predictions, alpha/2 * 100, axis=1)
            upper_bound = np.percentile(predictions, (1 - alpha/2) * 100, axis=1)
            
            # Return intervals
            return np.column_stack([lower_bound, upper_bound])
            
        except Exception as e:
            if self.verbose:
                print(f"Error in bootstrap prediction intervals: {str(e)}")
            return None
    
    def _calculate_improvement(
        self,
        baseline_metrics: Dict[str, Any],
        calibrated_metrics: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate improvement from calibration.
        
        Parameters:
        -----------
        baseline_metrics : dict
            Baseline calibration metrics
        calibrated_metrics : dict
            Calibrated model metrics
            
        Returns:
        --------
        dict : Improvement metrics
        """
        # Initialize improvement
        improvement = {}
        
        # Calculate relative improvement for each metric
        metrics_to_compare = [
            'brier_score', 'log_loss', 'expected_calibration_error'
        ]
        
        for metric in metrics_to_compare:
            if metric in baseline_metrics and metric in calibrated_metrics:
                baseline = baseline_metrics[metric]
                calibrated = calibrated_metrics[metric]
                
                if baseline != 0:
                    # For these metrics, lower is better, so negative change is improvement
                    improvement[metric] = float((baseline - calibrated) / abs(baseline))
                else:
                    improvement[metric] = 0.0
                    
        return improvement
    
    def _calculate_uncertainty_score(
        self,
        results: Dict[str, Any]
    ) -> float:
        """
        Calculate overall uncertainty score.
        
        Parameters:
        -----------
        results : dict
            Validation results
            
        Returns:
        --------
        float : Uncertainty score [0, 1]
        """
        if self.model_type == 'classifier':
            # For classifiers, use ECE, Brier score, or log loss
            if 'calibration' in results and 'expected_calibration_error' in results['calibration']:
                # Lower ECE is better, convert to [0, 1] score
                ece = results['calibration']['expected_calibration_error']
                return max(0, min(1, 1 - ece))
            elif 'baseline' in results and 'expected_calibration_error' in results['baseline']:
                # Use baseline ECE if calibration not available
                ece = results['baseline']['expected_calibration_error']
                return max(0, min(1, 1 - ece))
            else:
                # No calibration metrics available
                return 0.5
        else:
            # For regressors, use interval coverage
            if 'interval_coverage' in results['baseline']:
                # Coverage should be close to 1 - alpha (e.g., 0.95)
                coverage = results['baseline']['interval_coverage']
                
                # Ideal coverage is 0.95, penalize both under and over coverage
                target_coverage = 0.95
                coverage_error = abs(coverage - target_coverage)
                
                # Convert to [0, 1] score
                return max(0, min(1, 1 - coverage_error))
            else:
                # No interval metrics available
                return 0.5
    
    def plot_calibration_curve(
        self,
        use_calibrated: bool = True,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot calibration curve.
        
        Parameters:
        -----------
        use_calibrated : bool
            Whether to use calibrated model results
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.Figure : Calibration curve plot
        """
        if not self.results:
            raise ValueError("No results available. Run validate() first.")
            
        # Get calibration data
        if use_calibrated and 'calibration' in self.results:
            metrics = self.results['calibration']
            title_suffix = " (Calibrated)"
        else:
            metrics = self.results['baseline']
            title_suffix = " (Baseline)"
            
        if 'calibration_curve' not in metrics:
            raise ValueError("No calibration curve data available")
            
        curve_data = metrics['calibration_curve']
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot calibration curve
        bin_edges = np.array(curve_data['bin_edges'])
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_probs = np.array(curve_data['bin_probs'])
        bin_true_fractions = np.array(curve_data['bin_true_fractions'])
        bin_counts = np.array(curve_data['bin_counts'])
        
        # Plot calibration curve
        ax.plot(
            bin_probs,
            bin_true_fractions,
            's-',
            label=f'Calibration Curve (ECE={metrics["expected_calibration_error"]:.4f})'
        )
        
        # Plot ideal calibration (diagonal)
        ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
        
        # Plot confidence histogram
        ax_twin = ax.twinx()
        bin_heights = bin_counts / np.sum(bin_counts)
        
        ax_twin.bar(
            bin_centers,
            bin_heights,
            width=1/len(bin_centers),
            alpha=0.3,
            color='gray',
            label='Prediction Distribution'
        )
        
        # Set labels and title
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax_twin.set_ylabel('Fraction of Samples')
        
        ax.set_title(f'Calibration Curve{title_suffix}')
        
        # Set limits
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # Add legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        plt.tight_layout()
        return fig
    
    def plot_calibration_improvement(
        self,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot calibration improvement.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.Figure : Calibration improvement plot
        """
        if not self.results or 'calibration' not in self.results:
            raise ValueError("No calibration results available. Run validate() first.")
            
        # Get calibration data
        baseline_metrics = self.results['baseline']
        calibrated_metrics = self.results['calibration']
        
        if 'calibration_curve' not in baseline_metrics or 'calibration_curve' not in calibrated_metrics:
            raise ValueError("No calibration curve data available")
            
        baseline_curve = baseline_metrics['calibration_curve']
        calibrated_curve = calibrated_metrics['calibration_curve']
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot baseline calibration curve
        bin_edges = np.array(baseline_curve['bin_edges'])
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        ax.plot(
            np.array(baseline_curve['bin_probs']),
            np.array(baseline_curve['bin_true_fractions']),
            's-',
            color='red',
            label=f'Baseline (ECE={baseline_metrics["expected_calibration_error"]:.4f})'
        )
        
        # Plot calibrated curve
        ax.plot(
            np.array(calibrated_curve['bin_probs']),
            np.array(calibrated_curve['bin_true_fractions']),
            'o-',
            color='green',
            label=f'Calibrated (ECE={calibrated_metrics["expected_calibration_error"]:.4f})'
        )
        
        # Plot ideal calibration (diagonal)
        ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
        
        # Set labels and title
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        
        ax.set_title(f'Calibration Improvement ({self.calibration_method.title()} Method)')
        
        # Set limits
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # Add legend
        ax.legend(loc='best')
        
        plt.tight_layout()
        return fig
    
    def plot_prediction_intervals(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        n_samples: int = 100,
        alpha: float = 0.05,
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Plot prediction intervals for regression.
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Features
        y : array-like or Series
            Target values
        n_samples : int
            Number of samples to plot
        alpha : float
            Significance level
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.Figure : Prediction intervals plot
        """
        if self.model_type != 'regressor':
            raise ValueError("Prediction intervals are only applicable to regression models")
            
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X
            
        if isinstance(y, pd.Series):
            y_values = y.values
        else:
            y_values = y
            
        # Subsample if too many points
        if len(X_values) > n_samples:
            indices = np.random.choice(len(X_values), n_samples, replace=False)
            X_sample = X_values[indices]
            y_sample = y_values[indices]
        else:
            X_sample = X_values
            y_sample = y_values
            
        # Get point predictions
        y_pred = self.model.predict(X_sample)
        
        # Get prediction intervals
        if hasattr(self.model, 'predict_interval'):
            # Model has native prediction interval support
            try:
                intervals = self.model.predict_interval(X_sample, alpha=alpha)
            except:
                # Fall back to bootstrap
                intervals = self._bootstrap_prediction_intervals(X_sample, y_sample, alpha=alpha)
        else:
            # Use bootstrap
            intervals = self._bootstrap_prediction_intervals(X_sample, y_sample, alpha=alpha)
            
        if intervals is None:
            raise ValueError("Failed to compute prediction intervals")
            
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Sort by predicted value for better visualization
        sort_idx = np.argsort(y_pred)
        y_pred_sorted = y_pred[sort_idx]
        y_true_sorted = y_sample[sort_idx]
        intervals_sorted = intervals[sort_idx]
        
        # Plot point predictions vs. true values
        ax.scatter(
            range(len(y_pred_sorted)),
            y_true_sorted,
            color='blue',
            label='True Values',
            alpha=0.6
        )
        
        ax.plot(
            range(len(y_pred_sorted)),
            y_pred_sorted,
            'r-',
            label='Predictions',
            alpha=0.8
        )
        
        # Plot prediction intervals
        ax.fill_between(
            range(len(y_pred_sorted)),
            intervals_sorted[:, 0],
            intervals_sorted[:, 1],
            color='red',
            alpha=0.2,
            label=f'{int((1-alpha)*100)}% Prediction Interval'
        )
        
        # Set labels and title
        ax.set_xlabel('Sample Index (sorted by predicted value)')
        ax.set_ylabel('Target Value')
        
        # Calculate coverage
        coverage = np.mean((y_true_sorted >= intervals_sorted[:, 0]) & 
                          (y_true_sorted <= intervals_sorted[:, 1]))
        
        ax.set_title(f'Regression Prediction Intervals (Coverage: {coverage:.2%})')
        
        # Add legend
        ax.legend(loc='best')
        
        plt.tight_layout()
        return fig
    
    def save_results(
        self,
        output_dir: str,
        prefix: str = 'uncertainty',
        include_plots: bool = True
    ) -> Dict[str, str]:
        """
        Save validation results and plots.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save results in
        prefix : str
            Prefix for output files
        include_plots : bool
            Whether to save plots
            
        Returns:
        --------
        dict : Paths to saved files
        """
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Paths for saved files
        saved_files = {}
        
        # Save results as JSON
        results_path = os.path.join(output_dir, f"{prefix}_results.json")
        save_validation_results(self.results, results_path, 'json')
        saved_files['results_json'] = results_path
        
        # Save results as CSV
        csv_path = os.path.join(output_dir, f"{prefix}_results.csv")
        save_validation_results(self.results, csv_path, 'dataframe')
        saved_files['results_csv'] = csv_path
        
        # Save plots if requested
        if include_plots and self.results:
            if self.model_type == 'classifier':
                # Save calibration curve plot
                try:
                    fig = self.plot_calibration_curve(use_calibrated=False)
                    fig_path = os.path.join(output_dir, f"{prefix}_baseline_calibration.png")
                    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                    saved_files['baseline_calibration_plot'] = fig_path
                    plt.close(fig)
                except Exception as e:
                    if self.verbose:
                        print(f"Error saving baseline calibration plot: {str(e)}")
                        
                # Save calibrated curve plot if available
                if 'calibration' in self.results:
                    try:
                        fig = self.plot_calibration_curve(use_calibrated=True)
                        fig_path = os.path.join(output_dir, f"{prefix}_calibrated_curve.png")
                        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                        saved_files['calibrated_curve_plot'] = fig_path
                        plt.close(fig)
                        
                        # Also save improvement plot
                        fig = self.plot_calibration_improvement()
                        fig_path = os.path.join(output_dir, f"{prefix}_calibration_improvement.png")
                        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                        saved_files['calibration_improvement_plot'] = fig_path
                        plt.close(fig)
                    except Exception as e:
                        if self.verbose:
                            print(f"Error saving calibration improvement plot: {str(e)}")
                            
        return saved_files