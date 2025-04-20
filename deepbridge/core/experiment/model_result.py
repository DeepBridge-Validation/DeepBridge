"""
Classes for model evaluation and comparison results.
These classes implement the ModelResult interface from interfaces.py.
"""

import typing as t
from pathlib import Path
import json
import numpy as np
import pandas as pd
from datetime import datetime

from deepbridge.core.experiment.interfaces import ModelResult

class BaseModelResult(ModelResult):
    """Base implementation of the ModelResult interface"""
    
    def __init__(
        self,
        model_name: str,
        model_type: str,
        metrics: dict,
        hyperparameters: t.Optional[dict] = None,
        predictions: t.Optional[t.Dict[str, t.Any]] = None,
        metadata: t.Optional[dict] = None
    ):
        """
        Initialize with model evaluation results
        
        Args:
            model_name: Name of the model
            model_type: Type or class of the model
            metrics: Performance metrics
            hyperparameters: Model hyperparameters
            predictions: Model predictions (optional)
            metadata: Additional metadata (optional)
        """
        self._model_name = model_name
        self._model_type = model_type
        self._metrics = metrics or {}
        self._hyperparameters = hyperparameters or {}
        self._predictions = predictions or {}
        self._metadata = metadata or {}
        
    @property
    def model_name(self) -> str:
        """Get the name of the model"""
        return self._model_name
    
    @property
    def model_type(self) -> str:
        """Get the type of the model"""
        return self._model_type
    
    @property
    def metrics(self) -> dict:
        """Get performance metrics"""
        return self._metrics
    
    @property
    def hyperparameters(self) -> dict:
        """Get model hyperparameters"""
        return self._hyperparameters
    
    @property
    def predictions(self) -> dict:
        """Get model predictions"""
        return self._predictions
    
    @property
    def metadata(self) -> dict:
        """Get additional metadata"""
        return self._metadata
    
    def get_metric(self, metric_name: str, default: t.Any = None) -> t.Any:
        """Get a specific metric by name"""
        return self._metrics.get(metric_name, default)
    
    def get_hyperparameter(self, param_name: str, default: t.Any = None) -> t.Any:
        """Get a specific hyperparameter by name"""
        return self._hyperparameters.get(param_name, default)
    
    def to_dict(self) -> dict:
        """Convert the model result to a dictionary"""
        return {
            'name': self.model_name,
            'type': self.model_type,
            'metrics': self.metrics,
            'hyperparameters': self.hyperparameters,
            'metadata': self.metadata
        }
    
    # to_html method has been removed in this refactoring
    
    def compare_with(self, other: 'ModelResult', metrics: t.Optional[t.List[str]] = None) -> dict:
        """
        Compare this model result with another model result
        
        Args:
            other: Another ModelResult instance to compare with
            metrics: List of metrics to compare (if None, compare all common metrics)
            
        Returns:
            Dictionary with comparison results
        """
        if metrics is None:
            # Find common metrics
            metrics = [m for m in self.metrics if m in other.metrics]
            
        comparison = {
            'model1': self.model_name,
            'model2': other.model_name,
            'metrics_compared': {}
        }
        
        for metric in metrics:
            val1 = self.get_metric(metric)
            val2 = other.get_metric(metric)
            
            if val1 is not None and val2 is not None:
                # Calculate difference and percent change
                diff = val2 - val1
                
                if val1 != 0:
                    pct_change = (diff / val1) * 100
                else:
                    pct_change = float('inf') if diff > 0 else float('-inf') if diff < 0 else 0
                    
                comparison['metrics_compared'][metric] = {
                    'model1_value': val1,
                    'model2_value': val2,
                    'difference': diff,
                    'percent_change': pct_change
                }
                
        return comparison

class ClassificationModelResult(BaseModelResult):
    """Model result specialized for classification models"""
    
    def __init__(
        self,
        model_name: str,
        model_type: str,
        metrics: dict,
        hyperparameters: t.Optional[dict] = None,
        predictions: t.Optional[t.Dict[str, t.Any]] = None,
        metadata: t.Optional[dict] = None,
        confusion_matrix: t.Optional[t.Any] = None,
        class_names: t.Optional[t.List[str]] = None,
        auc_curve: t.Optional[t.Tuple[t.List[float], t.List[float]]] = None
    ):
        """
        Initialize with classification model evaluation results
        
        Args:
            model_name: Name of the model
            model_type: Type or class of the model
            metrics: Performance metrics
            hyperparameters: Model hyperparameters
            predictions: Model predictions (optional)
            metadata: Additional metadata (optional)
            confusion_matrix: Confusion matrix (optional)
            class_names: Names of classes (optional)
            auc_curve: ROC curve as tuple of (fpr, tpr) lists (optional)
        """
        super().__init__(model_name, model_type, metrics, hyperparameters, predictions, metadata)
        self._confusion_matrix = confusion_matrix
        self._class_names = class_names or []
        self._auc_curve = auc_curve
        
    @property
    def confusion_matrix(self) -> t.Optional[t.Any]:
        """Get the confusion matrix"""
        return self._confusion_matrix
    
    @property
    def class_names(self) -> t.List[str]:
        """Get the class names"""
        return self._class_names
    
    @property
    def auc_curve(self) -> t.Optional[t.Tuple[t.List[float], t.List[float]]]:
        """Get the ROC curve data"""
        return self._auc_curve
    
    # to_html method has been removed in this refactoring

class RegressionModelResult(BaseModelResult):
    """Model result specialized for regression models"""
    
    def __init__(
        self,
        model_name: str,
        model_type: str,
        metrics: dict,
        hyperparameters: t.Optional[dict] = None,
        predictions: t.Optional[t.Dict[str, t.Any]] = None,
        metadata: t.Optional[dict] = None,
        residuals: t.Optional[t.List[float]] = None,
        feature_importances: t.Optional[dict] = None
    ):
        """
        Initialize with regression model evaluation results
        
        Args:
            model_name: Name of the model
            model_type: Type or class of the model
            metrics: Performance metrics
            hyperparameters: Model hyperparameters
            predictions: Model predictions (optional)
            metadata: Additional metadata (optional)
            residuals: Model residuals (optional)
            feature_importances: Feature importance scores (optional)
        """
        super().__init__(model_name, model_type, metrics, hyperparameters, predictions, metadata)
        self._residuals = residuals
        self._feature_importances = feature_importances or {}
        
    @property
    def residuals(self) -> t.Optional[t.List[float]]:
        """Get the residuals"""
        return self._residuals
    
    @property
    def feature_importances(self) -> dict:
        """Get feature importance scores"""
        return self._feature_importances
    
    # to_html method has been removed in this refactoring

def create_model_result(
    model_name: str,
    model_type: str,
    metrics: dict,
    problem_type: str = 'classification',
    **kwargs
) -> ModelResult:
    """
    Factory function to create the appropriate model result object
    
    Args:
        model_name: Name of the model
        model_type: Type or class of the model
        metrics: Performance metrics
        problem_type: Type of problem ('classification', 'regression', 'forecasting')
        **kwargs: Additional parameters for specific model result types
        
    Returns:
        ModelResult instance
    """
    if problem_type.lower() == 'classification':
        return ClassificationModelResult(
            model_name=model_name,
            model_type=model_type,
            metrics=metrics,
            **kwargs
        )
    elif problem_type.lower() in ('regression', 'forecasting'):
        return RegressionModelResult(
            model_name=model_name,
            model_type=model_type,
            metrics=metrics,
            **kwargs
        )
    else:
        return BaseModelResult(
            model_name=model_name,
            model_type=model_type,
            metrics=metrics,
            **kwargs
        )