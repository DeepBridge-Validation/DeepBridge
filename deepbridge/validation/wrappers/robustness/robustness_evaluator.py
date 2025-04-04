"""
Module for evaluating model robustness against perturbations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
import time

from deepbridge.validation.wrappers.robustness.data_perturber import DataPerturber

class RobustnessEvaluator:
    """
    Evaluates model robustness against data perturbations.
    """
    
    def __init__(self, 
                 dataset, 
                 metric: str = 'AUC', 
                 verbose: bool = False,
                 random_state: Optional[int] = None):
        """
        Initialize the robustness evaluator.
        
        Parameters:
        -----------
        dataset : DBDataset
            Dataset object containing training/test data and model
        metric : str
            Performance metric to use for evaluation ('AUC', 'accuracy', 'mse', etc.)
        verbose : bool
            Whether to print progress information
        random_state : int or None
            Random seed for reproducibility
        """
        self.dataset = dataset
        self.metric = metric
        self.verbose = verbose
        
        # Create data perturber
        self.data_perturber = DataPerturber()
        if random_state is not None:
            self.data_perturber.set_random_state(random_state)
        
        # Determine problem type based on dataset or model
        self._problem_type = self._determine_problem_type()
        
        if self.verbose:
            print(f"Problem type detected: {self._problem_type}")
            print(f"Using metric: {self.metric}")
    
    def _determine_problem_type(self) -> str:
        """Determine if the problem is classification or regression"""
        # Try to get problem type from dataset
        if hasattr(self.dataset, 'problem_type'):
            return self.dataset.problem_type
        
        # Try to infer from the model
        if hasattr(self.dataset, 'model'):
            model = self.dataset.model
            if hasattr(model, 'predict_proba'):
                return 'classification'
            else:
                return 'regression'
        
        # Default to classification
        return 'classification'
    
    def calculate_base_score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Calculate baseline score on unperturbed data.
        
        Parameters:
        -----------
        X : DataFrame
            Feature data
        y : Series
            Target variable
            
        Returns:
        --------
        float : Baseline performance score
        """
        if not hasattr(self.dataset, 'model') or self.dataset.model is None:
            raise ValueError("Dataset has no model for evaluation")
            
        model = self.dataset.model
        
        if self._problem_type == 'classification':
            # For classification, use predict_proba if available
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X)
                # Ensure y_pred_proba has correct shape
                if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
                    # Use second column for binary classification
                    y_pred_proba = y_pred_proba[:, 1]
                
                # Use appropriate metric
                if self.metric.upper() in ['AUC', 'ROC_AUC']:
                    from sklearn.metrics import roc_auc_score
                    score = roc_auc_score(y, y_pred_proba)
                else:
                    # For other metrics, convert probabilities to class labels
                    y_pred = (y_pred_proba > 0.5).astype(int)
                    score = self._get_metric_score(y, y_pred)
            else:
                # Fall back to predict for models without predict_proba
                y_pred = model.predict(X)
                score = self._get_metric_score(y, y_pred)
        else:
            # For regression, use predict
            y_pred = model.predict(X)
            score = self._get_metric_score(y, y_pred)
            
        return score
    
    def _get_metric_score(self, y_true: pd.Series, y_pred: Union[pd.Series, np.ndarray]) -> float:
        """
        Calculate score for the selected metric.
        
        Parameters:
        -----------
        y_true : Series
            True target values
        y_pred : Series or ndarray
            Predicted values
            
        Returns:
        --------
        float : Score for the selected metric
        """
        if self._problem_type == 'classification':
            metric_map = {
                'ACCURACY': 'accuracy_score',
                'F1': 'f1_score',
                'PRECISION': 'precision_score',
                'RECALL': 'recall_score'
            }
            
            if self.metric.upper() in metric_map:
                from sklearn import metrics
                metric_func = getattr(metrics, metric_map[self.metric.upper()])
                return metric_func(y_true, y_pred)
            else:
                # Default to accuracy
                from sklearn.metrics import accuracy_score
                return accuracy_score(y_true, y_pred)
        else:
            # Regression metrics
            metric_map = {
                'MSE': 'mean_squared_error',
                'MAE': 'mean_absolute_error',
                'R2': 'r2_score',
                'RMSE': 'root_mean_squared_error'
            }
            
            if self.metric.upper() in metric_map:
                from sklearn import metrics
                if self.metric.upper() == 'RMSE':
                    from sklearn.metrics import mean_squared_error
                    return np.sqrt(mean_squared_error(y_true, y_pred))
                else:
                    metric_func = getattr(metrics, metric_map[self.metric.upper()])
                    return metric_func(y_true, y_pred)
            else:
                # Default to MSE for regression
                from sklearn.metrics import mean_squared_error
                return mean_squared_error(y_true, y_pred)
    
    def evaluate_perturbation(self, 
                              X: pd.DataFrame, 
                              y: pd.Series, 
                              perturb_method: str, 
                              level: float, 
                              feature_subset: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate model performance on perturbed data.
        
        Parameters:
        -----------
        X : DataFrame
            Feature data
        y : Series
            Target variable
        perturb_method : str
            Method to use ('raw' or 'quantile')
        level : float
            Level of perturbation to apply
        feature_subset : List[str] or None
            Specific features to perturb (None for all)
            
        Returns:
        --------
        Dict[str, Any] : Results of the evaluation
        """
        # Calculate baseline score first
        base_score = self.calculate_base_score(X, y)
        
        # Perturb all features
        X_perturbed = self.data_perturber.perturb_data(
            X, 
            perturb_method, 
            level, 
            feature_subset
        )
        
        # Calculate score on perturbed data
        perturbed_score = self.calculate_score_on_perturbed_data(X_perturbed, y)
        
        # Calculate impact - for regression metrics like MSE, lower is better
        if self._problem_type == 'regression' and self.metric.upper() in ['MSE', 'MAE', 'RMSE']:
            # For these metrics, higher values mean worse performance
            impact = (perturbed_score - base_score) / max(base_score, 1e-10)
        else:
            # For classification metrics, higher values mean better performance
            impact = (base_score - perturbed_score) / max(base_score, 1e-10)
        
        # Return comprehensive results
        return {
            'base_score': base_score,
            'perturbed_score': perturbed_score,
            'impact': impact,
            'perturbation': {
                'method': perturb_method,
                'level': level,
                'features': feature_subset
            }
        }
    
    def calculate_score_on_perturbed_data(self, X_perturbed: pd.DataFrame, y: pd.Series) -> float:
        """
        Calculate score on perturbed data.
        
        Parameters:
        -----------
        X_perturbed : DataFrame
            Perturbed feature data
        y : Series
            Target variable
            
        Returns:
        --------
        float : Performance score on perturbed data
        """
        if not hasattr(self.dataset, 'model') or self.dataset.model is None:
            raise ValueError("Dataset has no model for evaluation")
            
        model = self.dataset.model
        
        if self._problem_type == 'classification':
            # For classification, use predict_proba if available
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_perturbed)
                # Ensure y_pred_proba has correct shape
                if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
                    # Use second column for binary classification
                    y_pred_proba = y_pred_proba[:, 1]
                
                # Use appropriate metric
                if self.metric.upper() in ['AUC', 'ROC_AUC']:
                    from sklearn.metrics import roc_auc_score
                    score = roc_auc_score(y, y_pred_proba)
                else:
                    # For other metrics, convert probabilities to class labels
                    y_pred = (y_pred_proba > 0.5).astype(int)
                    score = self._get_metric_score(y, y_pred)
            else:
                # Fall back to predict for models without predict_proba
                y_pred = model.predict(X_perturbed)
                score = self._get_metric_score(y, y_pred)
        else:
            # For regression, use predict
            y_pred = model.predict(X_perturbed)
            score = self._get_metric_score(y, y_pred)
            
        return score
    
    def evaluate_feature_importance(self, 
                                   X: pd.DataFrame, 
                                   y: pd.Series, 
                                   perturb_method: str, 
                                   level: float, 
                                   feature_subset: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Evaluate feature importance for robustness by perturbing each feature individually.
        
        Parameters:
        -----------
        X : DataFrame
            Feature data
        y : Series
            Target variable
        perturb_method : str
            Method to use ('raw' or 'quantile')
        level : float
            Level of perturbation to apply
        feature_subset : List[str] or None
            Specific features to evaluate (None for all features)
            
        Returns:
        --------
        Dict[str, float] : Mapping of feature names to importance scores
        """
        # Calculate baseline score
        base_score = self.calculate_base_score(X, y)
        
        # Get features to test
        feature_subset = feature_subset or X.columns.tolist()
        
        # Dictionary to store importance scores
        importance_scores = {}
        
        if self.verbose:
            print(f"Evaluating feature importance with {perturb_method} perturbation at level {level}")
        
        # Perturb each feature and evaluate
        for i, feature in enumerate(feature_subset):
            if self.verbose and (i+1) % 10 == 0:
                print(f"  - Processed {i+1}/{len(feature_subset)} features")
            
            # Create perturbed dataset with only this feature perturbed
            X_perturbed = self.data_perturber.perturb_data(X, perturb_method, level, [feature])
            
            # Calculate score on perturbed data
            perturbed_score = self.calculate_score_on_perturbed_data(X_perturbed, y)
            
            # Calculate impact - for regression metrics like MSE, lower is better
            if self._problem_type == 'regression' and self.metric.upper() in ['MSE', 'MAE', 'RMSE']:
                # For these metrics, higher values mean worse performance
                impact = (perturbed_score - base_score) / max(base_score, 1e-10)
            else:
                # For classification metrics, higher values mean better performance
                impact = (base_score - perturbed_score) / max(base_score, 1e-10)
            
            # Store importance score
            importance_scores[feature] = impact
        
        return importance_scores