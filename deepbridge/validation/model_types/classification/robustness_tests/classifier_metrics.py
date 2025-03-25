"""
Robustness metrics for classification models.

This module provides specialized metrics for evaluating the robustness
of classification models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report


class ClassificationRobustnessMetrics:
    """
    Specialized metrics for classification model robustness.
    
    This class provides methods for computing and analyzing metrics
    that specifically assess the robustness of classification models.
    """
    
    def __init__(
        self,
        model: Any,
        metrics: Optional[List[str]] = None,
        random_state: Optional[int] = None,
        verbose: bool = False
    ):
        """
        Initialize the classification robustness metrics.
        
        Parameters:
        -----------
        model : Any
            Classification model to evaluate
        metrics : list of str or None
            Metrics to compute:
            - 'accuracy': Classification accuracy
            - 'f1': F1 score
            - 'roc_auc': ROC AUC score
            - 'confusion': Confusion matrix
            - 'class_report': Classification report
        random_state : int or None
            Random seed for reproducibility
        verbose : bool
            Whether to print progress information
        """
        self.model = model
        
        # Default metrics
        self.metrics = metrics or ['accuracy', 'f1', 'roc_auc', 'confusion', 'class_report']
        
        self.random_state = random_state
        self.verbose = verbose
        
        # Set random seed
        np.random.seed(random_state)
        
        # Initialize results storage
        self.results = {}
    
    def compute(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compute robustness metrics.
        
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
        dict : Robustness metrics
        """
        # Override parameters if provided
        metrics = kwargs.get('metrics', self.metrics)
        
        # Convert y to numpy array if needed
        if isinstance(y, pd.Series):
            y_values = y.values
        else:
            y_values = y
            
        # Make predictions
        try:
            # Class predictions
            y_pred = self.model.predict(X)
            
            # Probability predictions if available
            has_proba = hasattr(self.model, 'predict_proba')
            if has_proba:
                y_proba = self.model.predict_proba(X)
                
                # Get number of classes
                if hasattr(self.model, 'classes_'):
                    classes = self.model.classes_
                    n_classes = len(classes)
                else:
                    classes = None
                    n_classes = y_proba.shape[1]
        except Exception as e:
            if self.verbose:
                print(f"Error making predictions: {str(e)}")
            return {'error': str(e)}
        
        # Initialize results
        results = {}
        
        # Compute standard metrics
        if 'accuracy' in metrics:
            results['accuracy'] = float(accuracy_score(y_values, y_pred))
            
        if 'f1' in metrics:
            if len(np.unique(y_values)) > 2:
                # Multiclass
                results['f1'] = float(f1_score(y_values, y_pred, average='weighted'))
                
                # Per-class F1
                results['f1_per_class'] = {}
                f1_per_class = f1_score(y_values, y_pred, average=None)
                
                for i, cls in enumerate(np.unique(y_values)):
                    results['f1_per_class'][str(cls)] = float(f1_per_class[i])
            else:
                # Binary
                results['f1'] = float(f1_score(y_values, y_pred))
                
        if 'roc_auc' in metrics and has_proba:
            if n_classes > 2:
                # Multiclass
                try:
                    results['roc_auc'] = float(roc_auc_score(y_values, y_proba, multi_class='ovr'))
                except:
                    results['roc_auc'] = 0.5
            else:
                # Binary
                try:
                    results['roc_auc'] = float(roc_auc_score(y_values, y_proba[:, 1]))
                except:
                    results['roc_auc'] = 0.5
                
        # Compute confusion matrix
        if 'confusion' in metrics:
            cm = confusion_matrix(y_values, y_pred)
            results['confusion_matrix'] = cm.tolist()
            
            # Compute per-class metrics from confusion matrix
            # Normalize by row (true values)
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_norm = np.nan_to_num(cm_norm)  # Handle division by zero
            
            results['class_robustness'] = {}
            for i, cls in enumerate(np.unique(y_values)):
                # Class recall (robustness of this class)
                results['class_robustness'][str(cls)] = float(cm_norm[i, i])
            
        # Generate classification report
        if 'class_report' in metrics:
            try:
                report = classification_report(y_values, y_pred, output_dict=True)
                results['classification_report'] = report
            except:
                # Skip if error occurs
                pass
                
        # Compute robustness score
        robustness_score = compute_robustness_score(results)
        results['robustness_score'] = robustness_score
        
        # Store results
        self.results = results
        
        return results
    
    def plot_confusion_matrix(
        self,
        figsize: Tuple[int, int] = (8, 6),
        normalize: bool = True
    ) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
        normalize : bool
            Whether to normalize confusion matrix
            
        Returns:
        --------
        matplotlib.Figure : Confusion matrix plot
        """
        if not self.results or 'confusion_matrix' not in self.results:
            raise ValueError("No confusion matrix available. Call compute() first.")
            
        # Get confusion matrix
        cm = np.array(self.results['confusion_matrix'])
        
        # Get class names
        if hasattr(self.model, 'classes_'):
            classes = self.model.classes_
        else:
            classes = [str(i) for i in range(cm.shape[0])]
            
        # Normalize if requested
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm = np.nan_to_num(cm)  # Handle division by zero
            
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot matrix
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        # Set labels
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=classes, yticklabels=classes,
               xlabel='Predicted label',
               ylabel='True label')
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
                
        ax.set_title('Confusion Matrix')
        fig.tight_layout()
        
        return fig
    
    def plot_class_robustness(
        self,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot robustness by class.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.Figure : Class robustness plot
        """
        if not self.results or 'class_robustness' not in self.results:
            raise ValueError("No class robustness data available. Call compute() first.")
            
        # Get class robustness
        class_robustness = self.results['class_robustness']
        
        # Sort by robustness
        classes = []
        robustness = []
        
        for cls, score in sorted(class_robustness.items(), key=lambda x: x[1]):
            classes.append(cls)
            robustness.append(score)
            
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot robustness
        ax.barh(range(len(classes)), robustness, align='center')
        
        # Set labels
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(classes)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Robustness Score')
        ax.set_title('Robustness by Class')
        
        # Add value labels
        for i, v in enumerate(robustness):
            ax.text(v + 0.01, i, f"{v:.2f}", va='center')
            
        # Add reference line at 1.0 (perfect robustness)
        ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.5)
        
        return fig


def compute_robustness_metrics(
    model: Any,
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    metrics: Optional[List[str]] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Compute robustness metrics for a classification model.
    
    Parameters:
    -----------
    model : Any
        Classification model to evaluate
    X : array-like or DataFrame
        Feature data
    y : array-like or Series
        Target data
    metrics : list of str or None
        Metrics to compute
    verbose : bool
        Whether to print progress information
        
    Returns:
    --------
    dict : Robustness metrics
    """
    metrics_calculator = ClassificationRobustnessMetrics(
        model=model,
        metrics=metrics,
        verbose=verbose
    )
    
    return metrics_calculator.compute(X, y)


def compute_robustness_score(results: Dict[str, Any]) -> float:
    """
    Compute an overall robustness score from metric results.
    
    Parameters:
    -----------
    results : dict
        Dictionary of metric results
        
    Returns:
    --------
    float : Robustness score [0, 1] where higher is more robust
    """
    # Initialize score components
    components = []
    
    # Use accuracy if available
    if 'accuracy' in results:
        components.append(results['accuracy'])
        
    # Use F1 score if available
    if 'f1' in results:
        components.append(results['f1'])
        
    # Use ROC AUC if available
    if 'roc_auc' in results:
        # Normalize ROC AUC to [0, 1] where 0.5 -> 0, 1.0 -> 1
        roc_normalized = min(1.0, max(0.0, (results['roc_auc'] - 0.5) * 2))
        components.append(roc_normalized)
        
    # Use class robustness if available
    if 'class_robustness' in results:
        # Average class robustness
        class_scores = list(results['class_robustness'].values())
        if class_scores:
            # Penalize variance between classes
            avg_class_score = np.mean(class_scores)
            class_std = np.std(class_scores)
            
            # Adjusted class score with penalty for high variance
            adjusted_class_score = avg_class_score * (1 - 0.5 * class_std)
            components.append(adjusted_class_score)
            
    # Compute final score
    if components:
        # Overall robustness score is weighted average
        # with higher weight for worst component
        min_component = min(components)
        avg_component = np.mean(components)
        
        # Weighted average (60% min, 40% avg)
        score = 0.6 * min_component + 0.4 * avg_component
    else:
        # Default score if no components
        score = 0.5
        
    return float(score)