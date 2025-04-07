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
    
    def to_html(self) -> str:
        """Convert model result to HTML"""
        html = f"<h3>Model: {self.model_name} ({self.model_type})</h3>"
        
        # Metrics table
        html += "<h4>Performance Metrics</h4>"
        html += "<table border='1'><tr><th>Metric</th><th>Value</th></tr>"
        
        for metric, value in self.metrics.items():
            # Format numbers nicely
            if isinstance(value, float):
                value_str = f"{value:.4f}"
            else:
                value_str = str(value)
                
            html += f"<tr><td>{metric}</td><td>{value_str}</td></tr>"
            
        html += "</table>"
        
        # Add hyperparameters if available
        if self.hyperparameters:
            html += "<h4>Hyperparameters</h4>"
            html += "<table border='1'><tr><th>Parameter</th><th>Value</th></tr>"
            
            for param, value in self.hyperparameters.items():
                html += f"<tr><td>{param}</td><td>{value}</td></tr>"
                
            html += "</table>"
            
        return html
    
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
    
    def to_html(self) -> str:
        """Convert classification model result to HTML"""
        # Get base HTML representation
        html = super().to_html()
        
        # Add classification-specific visualizations
        if self.confusion_matrix is not None:
            html += "<h4>Confusion Matrix</h4>"
            html += "<table border='1'>"
            
            # Add header row with class names
            html += "<tr><th></th>"
            for cls in self.class_names:
                html += f"<th>Predicted {cls}</th>"
            html += "</tr>"
            
            # Add rows for each true class
            for i, cls in enumerate(self.class_names):
                html += f"<tr><th>Actual {cls}</th>"
                for j in range(len(self.class_names)):
                    html += f"<td>{self.confusion_matrix[i, j]}</td>"
                html += "</tr>"
                
            html += "</table>"
        
        return html

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
    
    def to_html(self) -> str:
        """Convert regression model result to HTML"""
        # Get base HTML representation
        html = super().to_html()
        
        # Add regression-specific visualizations
        if self.feature_importances:
            html += "<h4>Feature Importances</h4>"
            html += "<table border='1'><tr><th>Feature</th><th>Importance</th></tr>"
            
            # Sort features by importance
            sorted_features = sorted(
                self.feature_importances.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            )
            
            for feature, importance in sorted_features:
                html += f"<tr><td>{feature}</td><td>{importance:.4f}</td></tr>"
                
            html += "</table>"
        
        return html

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