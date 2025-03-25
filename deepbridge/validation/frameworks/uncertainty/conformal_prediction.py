"""
Conformal prediction for uncertainty estimation.

This module provides tools for applying conformal prediction methods
to obtain valid prediction intervals or sets.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings


class ConformalPredictor:
    """
    Conformal prediction for uncertainty estimation.
    
    This class implements conformal prediction methods to obtain
    valid prediction intervals or sets with guaranteed coverage.
    """
    
    def __init__(
        self,
        model: Any,
        problem_type: str = 'classification',
        score_type: str = 'auto',
        alpha: float = 0.1,
        calibration_fraction: float = 0.2,
        random_state: Optional[int] = None,
        verbose: bool = False
    ):
        """
        Initialize the conformal predictor.
        
        Parameters:
        -----------
        model : Any
            Trained machine learning model
        problem_type : str
            Type of problem: 'classification' or 'regression'
        score_type : str
            Type of conformity score to use:
            - For classification: 'probability', 'margin', 'auto'
            - For regression: 'absolute', 'squared', 'auto'
        alpha : float
            Significance level (1 - alpha = desired coverage)
        calibration_fraction : float
            Fraction of data to use for calibration
        random_state : int or None
            Random seed for reproducibility
        verbose : bool
            Whether to print progress information
        """
        self.model = model
        self.problem_type = problem_type
        self.alpha = alpha
        self.calibration_fraction = calibration_fraction
        self.random_state = random_state
        self.verbose = verbose
        
        # Determine score type based on problem type
        if score_type == 'auto':
            if problem_type == 'classification':
                self.score_type = 'probability'
            else:  # regression
                self.score_type = 'absolute'
        else:
            self.score_type = score_type
            
        # Check if model has required methods
        if problem_type == 'classification':
            if self.score_type == 'probability' and not hasattr(model, 'predict_proba'):
                if hasattr(model, 'predict'):
                    warnings.warn("Model does not have predict_proba method. Using 'margin' score instead.")
                    self.score_type = 'margin'
                else:
                    raise ValueError("Model must have predict or predict_proba method")
                    
        self.calibration_scores = None
        self.threshold = None
        self.classes = None
    
    def calibrate(
        self,
        X_cal: Union[np.ndarray, pd.DataFrame],
        y_cal: Union[np.ndarray, pd.Series]
    ) -> float:
        """
        Calibrate the conformal predictor.
        
        Parameters:
        -----------
        X_cal : array-like or DataFrame
            Calibration features
        y_cal : array-like or Series
            Calibration targets
            
        Returns:
        --------
        float : Conformal threshold
        """
        # Convert y to numpy array if it's a Series
        if isinstance(y_cal, pd.Series):
            y_cal_values = y_cal.values
        else:
            y_cal_values = y_cal
            
        # For classification, get unique classes
        if self.problem_type == 'classification':
            self.classes = np.unique(y_cal_values)
            
        # Compute conformity scores
        scores = self._compute_scores(X_cal, y_cal_values)
        
        # Store calibration scores
        self.calibration_scores = scores
        
        # Compute threshold
        n = len(scores)
        q = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.threshold = np.quantile(scores, q)
        
        if self.verbose:
            print(f"Calibration complete. Threshold: {self.threshold:.4f}")
            
        return self.threshold
    
    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> Union[List[List[Any]], np.ndarray]:
        """
        Generate conformal predictions.
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Input features
            
        Returns:
        --------
        list or array : Prediction sets or intervals
        """
        if self.threshold is None:
            raise ValueError("Predictor must be calibrated before prediction")
            
        if self.problem_type == 'classification':
            return self._predict_classification(X)
        else:  # regression
            return self._predict_regression(X)
    
    def _predict_classification(
        self,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> List[List[Any]]:
        """
        Generate conformal prediction sets for classification.
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Input features
            
        Returns:
        --------
        list : List of prediction sets
        """
        # Check if model has predict_proba method
        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(X)
            
            # Create prediction sets
            prediction_sets = []
            
            for i in range(len(X)):
                # Get probabilities for this sample
                sample_probs = probs[i]
                
                # Create prediction set based on threshold
                if self.score_type == 'probability':
                    # For probability score, include classes with high enough probability
                    pred_set = [self.classes[j] for j in range(len(self.classes)) 
                              if sample_probs[j] >= self.threshold]
                elif self.score_type == 'margin':
                    # For margin score, include classes with small enough margin to top class
                    top_prob = np.max(sample_probs)
                    pred_set = [self.classes[j] for j in range(len(self.classes)) 
                              if top_prob - sample_probs[j] <= self.threshold]
                
                # If prediction set is empty, include class with highest probability
                if not pred_set:
                    pred_set = [self.classes[np.argmax(sample_probs)]]
                    
                prediction_sets.append(pred_set)
        else:
            # If no predict_proba, use a simpler approach
            preds = self.model.predict(X)
            prediction_sets = [[pred] for pred in preds]  # Singleton sets
            
        return prediction_sets
    
    def _predict_regression(
        self,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """
        Generate conformal prediction intervals for regression.
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Input features
            
        Returns:
        --------
        numpy.ndarray : Prediction intervals (lower, upper)
        """
        # Get point predictions
        point_preds = self.model.predict(X)
        
        # Create prediction intervals
        intervals = np.zeros((len(X), 2))
        
        # For absolute error score
        if self.score_type == 'absolute':
            intervals[:, 0] = point_preds - self.threshold
            intervals[:, 1] = point_preds + self.threshold
        # For squared error score
        elif self.score_type == 'squared':
            error = np.sqrt(self.threshold)
            intervals[:, 0] = point_preds - error
            intervals[:, 1] = point_preds + error
            
        return intervals
    
    def _compute_scores(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray
    ) -> np.ndarray:
        """
        Compute conformity scores.
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Input features
        y : array-like
            Target values
            
        Returns:
        --------
        numpy.ndarray : Conformity scores
        """
        if self.problem_type == 'classification':
            return self._compute_classification_scores(X, y)
        else:  # regression
            return self._compute_regression_scores(X, y)
    
    def _compute_classification_scores(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray
    ) -> np.ndarray:
        """
        Compute conformity scores for classification.
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Input features
        y : array-like
            Target values
            
        Returns:
        --------
        numpy.ndarray : Conformity scores
        """
        if self.score_type == 'probability':
            # Check if model has predict_proba
            if not hasattr(self.model, 'predict_proba'):
                raise ValueError("Model must have predict_proba method for probability scores")
                
            # Get class probabilities
            probs = self.model.predict_proba(X)
            
            # Get probabilities for true classes
            scores = np.zeros(len(y))
            for i in range(len(y)):
                true_idx = np.where(self.classes == y[i])[0][0]
                scores[i] = probs[i, true_idx]
                
            # For probability score, higher is better, so we invert
            return 1 - scores
            
        elif self.score_type == 'margin':
            # Get class probabilities if available
            if hasattr(self.model, 'predict_proba'):
                probs = self.model.predict_proba(X)
                
                # Compute margin between true class and highest other class
                scores = np.zeros(len(y))
                for i in range(len(y)):
                    true_idx = np.where(self.classes == y[i])[0][0]
                    true_prob = probs[i, true_idx]
                    
                    # Get highest probability for other classes
                    other_probs = np.delete(probs[i], true_idx)
                    if len(other_probs) > 0:
                        highest_other = np.max(other_probs)
                    else:
                        highest_other = 0
                        
                    # Margin score is difference between highest other and true
                    scores[i] = highest_other - true_prob
                    
                return scores
            else:
                # If no predict_proba, use a simple score: 0 if correct, 1 if wrong
                preds = self.model.predict(X)
                return (preds != y).astype(float)
                
        else:
            raise ValueError(f"Unknown score type for classification: {self.score_type}")
    
    def _compute_regression_scores(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray
    ) -> np.ndarray:
        """
        Compute conformity scores for regression.
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Input features
        y : array-like
            Target values
            
        Returns:
        --------
        numpy.ndarray : Conformity scores
        """
        # Get predictions
        preds = self.model.predict(X)
        
        # Compute residuals
        residuals = np.abs(preds - y)
        
        # Apply score type
        if self.score_type == 'absolute':
            return residuals
        elif self.score_type == 'squared':
            return residuals ** 2
        else:
            raise ValueError(f"Unknown score type for regression: {self.score_type}")
    
    def calibrate_and_predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series]
    ) -> Union[List[List[Any]], np.ndarray]:
        """
        Calibrate and predict in one step using split-conformal.
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Input features
        y : array-like or Series
            Target values
            
        Returns:
        --------
        list or array : Prediction sets or intervals
        """
        # Split data into proper train and calibration sets
        X_train, X_cal, y_train, y_cal = train_test_split(
            X, y,
            test_size=self.calibration_fraction,
            random_state=self.random_state
        )
        
        # Fit model on proper train set
        if hasattr(self.model, 'fit'):
            self.model.fit(X_train, y_train)
            
        # Calibrate on calibration set
        self.calibrate(X_cal, y_cal)
        
        # Predict on original data
        return self.predict(X)
    
    def plot_coverage(self, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot empirical coverage.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
            
        Returns:
        --------
        matplotlib.Figure : Coverage plot
        """
        if self.calibration_scores is None:
            raise ValueError("Predictor must be calibrated before plotting")
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get sorted calibration scores
        scores = np.sort(self.calibration_scores)
        
        # Get empirical coverage for each possible threshold
        n = len(scores)
        coverage = np.arange(1, n + 1) / n
        
        # Plot empirical coverage
        ax.plot(scores, 1 - coverage, label='Empirical Coverage')
        
        # Plot desired coverage
        ax.axhline(y=1 - self.alpha, color='r', linestyle='--', 
                 label=f'Target Coverage ({1 - self.alpha:.2f})')
        
        # Add threshold line
        ax.axvline(x=self.threshold, color='g', linestyle='--',
                 label=f'Threshold ({self.threshold:.4f})')
        
        ax.set_xlabel('Conformity Score')
        ax.set_ylabel('Coverage')
        ax.set_title('Conformal Prediction Coverage')
        ax.legend()
        
        return fig


def compute_prediction_sets(
    model: Any,
    X_train: Union[np.ndarray, pd.DataFrame],
    y_train: Union[np.ndarray, pd.Series],
    X_test: Union[np.ndarray, pd.DataFrame],
    problem_type: str = 'classification',
    alpha: float = 0.1,
    score_type: str = 'auto',
    calibration_fraction: float = 0.2,
    random_state: Optional[int] = None
) -> Union[List[List[Any]], np.ndarray]:
    """
    Compute conformal prediction sets or intervals.
    
    Parameters:
    -----------
    model : Any
        Trained or untrained machine learning model
    X_train : array-like or DataFrame
        Training features
    y_train : array-like or Series
        Training targets
    X_test : array-like or DataFrame
        Test features
    problem_type : str
        Type of problem: 'classification' or 'regression'
    alpha : float
        Significance level (1 - alpha = desired coverage)
    score_type : str
        Type of conformity score to use
    calibration_fraction : float
        Fraction of training data to use for calibration
    random_state : int or None
        Random seed for reproducibility
        
    Returns:
    --------
    list or array : Prediction sets or intervals
    """
    predictor = ConformalPredictor(
        model=model,
        problem_type=problem_type,
        score_type=score_type,
        alpha=alpha,
        calibration_fraction=calibration_fraction,
        random_state=random_state
    )
    
    return predictor.calibrate_and_predict(X_train, y_train)


def get_conformal_score(
    model: Any,
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    problem_type: str = 'classification',
    score_type: str = 'auto'
) -> np.ndarray:
    """
    Compute conformity scores for a model.
    
    Parameters:
    -----------
    model : Any
        Trained machine learning model
    X : array-like or DataFrame
        Input features
    y : array-like or Series
        Target values
    problem_type : str
        Type of problem: 'classification' or 'regression'
    score_type : str
        Type of conformity score to use
        
    Returns:
    --------
    numpy.ndarray : Conformity scores
    """
    predictor = ConformalPredictor(
        model=model,
        problem_type=problem_type,
        score_type=score_type
    )
    
    # Convert y to numpy array if it's a Series
    if isinstance(y, pd.Series):
        y_values = y.values
    else:
        y_values = y
        
    # For classification, set classes
    if problem_type == 'classification':
        predictor.classes = np.unique(y_values)
        
    return predictor._compute_scores(X, y_values)