"""
Bootstrap methods for uncertainty estimation.

This module provides tools for using bootstrap resampling to 
estimate predictive uncertainty.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


class BootstrapUncertainty:
    """
    Estimate uncertainty using bootstrap resampling.
    
    This class implements various bootstrap methods for uncertainty
    estimation in machine learning predictions.
    """
    
    def __init__(
        self,
        base_model: Any,
        n_bootstrap: int = 100,
        bootstrap_type: str = 'regular',
        subsample_ratio: float = 1.0,
        random_state: Optional[int] = None,
        n_jobs: int = 1,
        verbose: bool = False
    ):
        """
        Initialize the bootstrap uncertainty estimator.
        
        Parameters:
        -----------
        base_model : Any
            Base model class (will be cloned)
        n_bootstrap : int
            Number of bootstrap samples
        bootstrap_type : str
            Type of bootstrap:
            - 'regular': Standard bootstrap
            - 'parametric': Parametric bootstrap
            - 'residual': Residual bootstrap (for regression)
        subsample_ratio : float
            Fraction of data to use in each bootstrap sample
        random_state : int or None
            Random seed for reproducibility
        n_jobs : int
            Number of parallel jobs
        verbose : bool
            Whether to print progress information
        """
        self.base_model = base_model
        self.n_bootstrap = n_bootstrap
        self.bootstrap_type = bootstrap_type
        self.subsample_ratio = subsample_ratio
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        self.bootstrap_models = []
        self.feature_names = None
    
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series]
    ) -> 'BootstrapUncertainty':
        """
        Fit bootstrap models.
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Training features
        y : array-like or Series
            Training targets
            
        Returns:
        --------
        self : BootstrapUncertainty
            Fitted estimator
        """
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns
            X_values = X.values
        else:
            X_values = X
            
        if isinstance(y, pd.Series):
            y_values = y.values
        else:
            y_values = y
            
        # Generate bootstrap indices
        n_samples = X_values.shape[0]
        bootstrap_size = int(n_samples * self.subsample_ratio)
        
        # Set random state
        np.random.seed(self.random_state)
        seeds = np.random.randint(0, 10000, size=self.n_bootstrap)
        
        if self.bootstrap_type == 'regular':
            # Standard bootstrap
            bootstrap_indices = [
                np.random.RandomState(seed).choice(
                    n_samples, size=bootstrap_size, replace=True
                ) for seed in seeds
            ]
            
            # Fit models in parallel
            self.bootstrap_models = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(self._fit_bootstrap_model)(
                    X_values[indices], y_values[indices], seed
                ) for indices, seed in zip(bootstrap_indices, seeds)
            )
            
        elif self.bootstrap_type == 'parametric':
            # Parametric bootstrap - fit model on full data first
            full_model = self._clone_model(self.base_model)
            full_model.fit(X_values, y_values)
            
            # Generate new targets from model predictions
            if hasattr(full_model, 'predict_proba'):
                # Classification - sample from predicted probabilities
                y_probs = full_model.predict_proba(X_values)
                
                # Fit models in parallel
                self.bootstrap_models = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                    delayed(self._fit_parametric_bootstrap_model)(
                        X_values, y_probs, seed, full_model.classes_
                    ) for seed in seeds
                )
                
            else:
                # Regression - sample from predicted mean and residual variance
                y_pred = full_model.predict(X_values)
                residuals = y_values - y_pred
                residual_std = np.std(residuals)
                
                # Fit models in parallel
                self.bootstrap_models = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                    delayed(self._fit_parametric_bootstrap_model_regression)(
                        X_values, y_pred, residual_std, seed
                    ) for seed in seeds
                )
                
        elif self.bootstrap_type == 'residual':
            # Residual bootstrap - fit model on full data first
            if not hasattr(self.base_model, 'predict'):
                raise ValueError("Base model must have predict method for residual bootstrap")
                
            full_model = self._clone_model(self.base_model)
            full_model.fit(X_values, y_values)
            
            # Calculate residuals
            y_pred = full_model.predict(X_values)
            residuals = y_values - y_pred
            
            # Fit models with resampled residuals
            self.bootstrap_models = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(self._fit_residual_bootstrap_model)(
                    X_values, y_pred, residuals, seed
                ) for seed in seeds
            )
            
        else:
            raise ValueError(f"Unknown bootstrap type: {self.bootstrap_type}")
            
        return self
    
    def _clone_model(self, model: Any) -> Any:
        """
        Clone a model instance.
        
        Parameters:
        -----------
        model : Any
            Model to clone
            
        Returns:
        --------
        Any : Cloned model
        """
        try:
            from sklearn.base import clone
            return clone(model)
        except (ImportError, TypeError):
            # If not a scikit-learn model, try to create a new instance
            try:
                return model.__class__(**model.get_params())
            except (AttributeError, TypeError):
                # Last resort: try to pickle and unpickle
                import pickle
                return pickle.loads(pickle.dumps(model))
    
    def _fit_bootstrap_model(
        self,
        X_bootstrap: np.ndarray,
        y_bootstrap: np.ndarray,
        random_state: int
    ) -> Any:
        """
        Fit a single bootstrap model.
        
        Parameters:
        -----------
        X_bootstrap : array-like
            Bootstrap features
        y_bootstrap : array-like
            Bootstrap targets
        random_state : int
            Random seed
            
        Returns:
        --------
        Any : Fitted model
        """
        model = self._clone_model(self.base_model)
        
        # Set random state if model supports it
        if hasattr(model, 'random_state'):
            model.random_state = random_state
            
        # Fit the model
        model.fit(X_bootstrap, y_bootstrap)
        return model
    
    def _fit_parametric_bootstrap_model(
        self,
        X: np.ndarray,
        y_probs: np.ndarray,
        random_state: int,
        classes: np.ndarray
    ) -> Any:
        """
        Fit a parametric bootstrap model for classification.
        
        Parameters:
        -----------
        X : array-like
            Features
        y_probs : array-like
            Predicted probabilities
        random_state : int
            Random seed
        classes : array-like
            Class labels
            
        Returns:
        --------
        Any : Fitted model
        """
        # Set random seed
        np.random.seed(random_state)
        
        # Sample target values from predicted probabilities
        y_bootstrap = np.zeros(len(X), dtype=classes.dtype)
        
        for i in range(len(X)):
            y_bootstrap[i] = np.random.choice(classes, p=y_probs[i])
            
        # Fit model
        return self._fit_bootstrap_model(X, y_bootstrap, random_state)
    
    def _fit_parametric_bootstrap_model_regression(
        self,
        X: np.ndarray,
        y_pred: np.ndarray,
        residual_std: float,
        random_state: int
    ) -> Any:
        """
        Fit a parametric bootstrap model for regression.
        
        Parameters:
        -----------
        X : array-like
            Features
        y_pred : array-like
            Predicted values
        residual_std : float
            Standard deviation of residuals
        random_state : int
            Random seed
            
        Returns:
        --------
        Any : Fitted model
        """
        # Set random seed
        np.random.seed(random_state)
        
        # Sample new targets from predictions and residual variance
        noise = np.random.normal(0, residual_std, size=len(X))
        y_bootstrap = y_pred + noise
        
        # Fit model
        return self._fit_bootstrap_model(X, y_bootstrap, random_state)
    
    def _fit_residual_bootstrap_model(
        self,
        X: np.ndarray,
        y_pred: np.ndarray,
        residuals: np.ndarray,
        random_state: int
    ) -> Any:
        """
        Fit a residual bootstrap model.
        
        Parameters:
        -----------
        X : array-like
            Features
        y_pred : array-like
            Predicted values
        residuals : array-like
            Residuals
        random_state : int
            Random seed
            
        Returns:
        --------
        Any : Fitted model
        """
        # Set random seed
        np.random.seed(random_state)
        
        # Resample residuals
        resampled_residuals = np.random.choice(residuals, size=len(X), replace=True)
        
        # Create new targets
        y_bootstrap = y_pred + resampled_residuals
        
        # Fit model
        return self._fit_bootstrap_model(X, y_bootstrap, random_state)
    
    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        return_individual: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate point predictions with uncertainty.
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Input features
        return_individual : bool
            Whether to return individual bootstrap predictions
            
        Returns:
        --------
        array or tuple : Mean predictions and optionally individual predictions
        """
        # Check if models are fitted
        if not self.bootstrap_models:
            raise ValueError("Models must be fitted before prediction")
            
        # Convert to numpy array if necessary
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X
            
        # Get predictions from all bootstrap models
        all_preds = []
        
        for model in self.bootstrap_models:
            try:
                preds = model.predict(X_values)
                all_preds.append(preds)
            except Exception as e:
                if self.verbose:
                    print(f"Error in bootstrap model prediction: {str(e)}")
                    
        # Convert to array
        all_preds = np.array(all_preds)
        
        # Calculate mean prediction
        mean_pred = np.mean(all_preds, axis=0)
        
        if return_individual:
            return mean_pred, all_preds
        else:
            return mean_pred
    
    def predict_proba(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        return_individual: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate probability predictions with uncertainty.
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Input features
        return_individual : bool
            Whether to return individual bootstrap predictions
            
        Returns:
        --------
        array or tuple : Mean probabilities and optionally individual probabilities
        """
        # Check if models are fitted
        if not self.bootstrap_models:
            raise ValueError("Models must be fitted before prediction")
            
        # Convert to numpy array if necessary
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X
            
        # Get predictions from all bootstrap models
        all_probs = []
        
        for model in self.bootstrap_models:
            try:
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(X_values)
                    all_probs.append(probs)
                else:
                    if self.verbose:
                        print("Model does not have predict_proba method")
            except Exception as e:
                if self.verbose:
                    print(f"Error in bootstrap model prediction: {str(e)}")
                    
        # Convert to array
        all_probs = np.array(all_probs)
        
        # Calculate mean probabilities
        mean_probs = np.mean(all_probs, axis=0)
        
        if return_individual:
            return mean_probs, all_probs
        else:
            return mean_probs
    
    def predict_interval(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        alpha: float = 0.05
    ) -> np.ndarray:
        """
        Generate prediction intervals for regression.
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Input features
        alpha : float
            Significance level (1 - alpha = confidence level)
            
        Returns:
        --------
        numpy.ndarray : Prediction intervals (lower, upper)
        """
        # Get predictions from all bootstrap models
        mean_pred, all_preds = self.predict(X, return_individual=True)
        
        # Calculate prediction intervals
        lower_quantile = alpha / 2
        upper_quantile = 1 - lower_quantile
        
        lower_bounds = np.quantile(all_preds, lower_quantile, axis=0)
        upper_bounds = np.quantile(all_preds, upper_quantile, axis=0)
        
        # Stack bounds
        intervals = np.column_stack([lower_bounds, upper_bounds])
        return intervals
    
    def predict_uncertainty(
        self,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """
        Calculate predictive uncertainty.
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Input features
            
        Returns:
        --------
        numpy.ndarray : Standard deviations of predictions
        """
        # Get predictions from all bootstrap models
        mean_pred, all_preds = self.predict(X, return_individual=True)
        
        # Calculate standard deviation
        uncertainty = np.std(all_preds, axis=0)
        return uncertainty
    
    def plot_uncertainty(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y_true: Optional[Union[np.ndarray, pd.Series]] = None,
        feature_index: Optional[int] = 0,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot predictions with uncertainty.
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Input features
        y_true : array-like or None
            True targets (optional)
        feature_index : int or None
            Index of feature to plot against (for 1D plots)
        figsize : tuple
            Figure size (width, height)
            
        Returns:
        --------
        matplotlib.Figure : Uncertainty plot
        """
        # Convert to numpy array if necessary
        if isinstance(X, pd.DataFrame):
            x_values = X.iloc[:, feature_index].values if feature_index is not None else None
            x_label = X.columns[feature_index] if feature_index is not None else None
        else:
            x_values = X[:, feature_index] if feature_index is not None else None
            x_label = f"Feature {feature_index}" if feature_index is not None else None
            
        # Get predictions and intervals
        mean_pred, all_preds = self.predict(X, return_individual=True)
        intervals = self.predict_interval(X)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if x_values is not None:
            # 1D plot
            # Sort by x values for better visualization
            sorted_indices = np.argsort(x_values)
            x_sorted = x_values[sorted_indices]
            pred_sorted = mean_pred[sorted_indices]
            lower_sorted = intervals[sorted_indices, 0]
            upper_sorted = intervals[sorted_indices, 1]
            
            # Plot predictions and intervals
            ax.plot(x_sorted, pred_sorted, 'b-', label='Mean Prediction')
            ax.fill_between(x_sorted, lower_sorted, upper_sorted, alpha=0.3, color='b', label='95% Prediction Interval')
            
            # Plot true values if provided
            if y_true is not None:
                if isinstance(y_true, pd.Series):
                    y_true = y_true.values
                    
                y_sorted = y_true[sorted_indices]
                ax.scatter(x_sorted, y_sorted, color='r', alpha=0.5, label='True Values')
                
            ax.set_xlabel(x_label)
            ax.set_ylabel('Prediction')
            ax.set_title('Predictions with Uncertainty')
            ax.legend()
            
        else:
            # If no feature index, plot distribution of predictions
            ax.boxplot(all_preds.T)
            
            # Plot true values if provided
            if y_true is not None:
                if isinstance(y_true, pd.Series):
                    y_true = y_true.values
                    
                ax.scatter(np.arange(1, len(y_true) + 1), y_true, color='r', alpha=0.5, label='True Values')
                
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Prediction')
            ax.set_title('Predictions with Uncertainty')
            
        return fig