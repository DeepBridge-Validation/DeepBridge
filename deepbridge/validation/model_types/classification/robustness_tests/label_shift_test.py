"""
Label shift robustness tests for classification models.

This module provides tools for testing the robustness of classification
models against label distribution shifts.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


class ClassificationLabelShiftTest:
    """
    Test classification model robustness against label distribution shifts.
    
    This class provides methods for evaluating how well a classification
    model maintains its performance when the distribution of class labels
    changes between training and testing.
    """
    
    def __init__(
        self,
        model: Any,
        scenarios: Optional[List[str]] = None,
        shift_ratios: Optional[List[float]] = None,
        metrics: Optional[List[str]] = None,
        random_state: Optional[int] = None,
        verbose: bool = False
    ):
        """
        Initialize the label shift test.
        
        Parameters:
        -----------
        model : Any
            Classification model to test
        scenarios : list of str or None
            Label shift scenarios to test:
            - 'imbalance': Increase/decrease class imbalance
            - 'minority': Over/under-sample minority classes
            - 'majority': Over/under-sample majority classes
            - 'reverse': Reverse class frequencies
        shift_ratios : list of float or None
            Ratios for simulating label shifts
        metrics : list of str or None
            Metrics to use for evaluation:
            - 'accuracy': Classification accuracy
            - 'f1': F1 score
            - 'roc_auc': ROC AUC score
        random_state : int or None
            Random seed for reproducibility
        verbose : bool
            Whether to print progress information
        """
        self.model = model
        
        # Default scenarios
        self.scenarios = scenarios or ['imbalance', 'minority', 'majority', 'reverse']
        
        # Default shift ratios
        self.shift_ratios = shift_ratios or [0.1, 0.25, 0.5, 0.75, 1.0]
        
        # Default metrics
        self.metrics = metrics or ['accuracy', 'f1', 'roc_auc']
        
        self.random_state = random_state
        self.verbose = verbose
        
        # Set random seed
        np.random.seed(random_state)
        
        # Initialize results storage
        self.results = {}
    
    def test(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Test model robustness against label distribution shifts.
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Feature data
        y : array-like or Series
            Target data
        **kwargs : dict
            Additional parameters that override initialization
            
        Returns:
        --------
        dict : Label shift test results
        """
        # Override parameters if provided
        scenarios = kwargs.get('scenarios', self.scenarios)
        shift_ratios = kwargs.get('shift_ratios', self.shift_ratios)
        metrics = kwargs.get('metrics', self.metrics)
        
        # Convert y to numpy array if needed
        if isinstance(y, pd.Series):
            y_values = y.values
        else:
            y_values = y
            
        # Get class distribution
        classes, class_counts = np.unique(y_values, return_counts=True)
        class_distribution = {c: count for c, count in zip(classes, class_counts)}
        
        # Compute baseline performance
        baseline_performance = self._evaluate_model(X, y_values, metrics)
        
        # Initialize results
        results = {
            'baseline': baseline_performance,
            'class_distribution': class_distribution,
            'scenarios': {},
            'robustness_score': 0.0
        }
        
        # Test each scenario
        for scenario in scenarios:
            if self.verbose:
                print(f"Testing scenario: {scenario}")
                
            scenario_results = []
            
            for ratio in shift_ratios:
                # Simulate label shift
                X_shifted, y_shifted = simulate_label_shift(X, y_values, scenario, ratio, self.random_state)
                
                # Get shifted class distribution
                shifted_classes, shifted_counts = np.unique(y_shifted, return_counts=True)
                shifted_distribution = {c: count for c, count in zip(shifted_classes, shifted_counts)}
                
                # Evaluate model
                performance = self._evaluate_model(X_shifted, y_shifted, metrics)
                
                # Calculate relative change
                relative_change = {}
                for metric in metrics:
                    baseline = baseline_performance[metric]
                    current = performance[metric]
                    if baseline != 0:
                        relative_change[metric] = (current - baseline) / baseline
                    else:
                        relative_change[metric] = 0
                        
                # Store results
                scenario_results.append({
                    'ratio': ratio,
                    'performance': performance,
                    'relative_change': relative_change,
                    'shifted_distribution': shifted_distribution
                })
                
            # Store scenario results
            results['scenarios'][scenario] = scenario_results
            
        # Compute robustness score
        robustness_score = self._compute_robustness_score(results['scenarios'], metrics)
        results['robustness_score'] = robustness_score
        
        # Store results
        self.results = results
        
        return results
    
    def _evaluate_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metrics: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate model performance on data.
        
        Parameters:
        -----------
        X : array-like
            Feature data
        y : array-like
            Target data
        metrics : list of str
            Metrics to compute
            
        Returns:
        --------
        dict : Performance metrics
        """
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
                    n_classes = len(self.model.classes_)
                else:
                    n_classes = y_proba.shape[1]
        except Exception as e:
            # If prediction fails, return zeros
            return {metric: 0.0 for metric in metrics}
            
        # Compute metrics
        results = {}
        
        for metric in metrics:
            if metric == 'accuracy':
                results[metric] = accuracy_score(y, y_pred)
            elif metric == 'f1':
                if len(np.unique(y)) > 2:
                    # Multiclass
                    results[metric] = f1_score(y, y_pred, average='weighted')
                else:
                    # Binary
                    results[metric] = f1_score(y, y_pred)
            elif metric == 'roc_auc':
                if has_proba:
                    if n_classes > 2:
                        # Multiclass
                        results[metric] = roc_auc_score(y, y_proba, multi_class='ovr')
                    else:
                        # Binary
                        results[metric] = roc_auc_score(y, y_proba[:, 1])
                else:
                    # ROC AUC requires probabilities
                    results[metric] = 0.5
            else:
                # Unknown metric
                results[metric] = 0.0
                
        return results
    
    def _compute_robustness_score(
        self,
        scenario_results: Dict[str, List[Dict[str, Any]]],
        metrics: List[str]
    ) -> float:
        """
        Compute overall robustness score.
        
        Parameters:
        -----------
        scenario_results : dict
            Results for each scenario
        metrics : list of str
            Metrics used
            
        Returns:
        --------
        float : Robustness score
        """
        # Collect all relative changes
        all_changes = []
        
        for scenario in scenario_results:
            for result in scenario_results[scenario]:
                for metric in metrics:
                    change = result['relative_change'][metric]
                    all_changes.append(change)
                    
        # Compute score
        # Lower (less negative) changes mean more robust model
        # Normalize to [0, 1] where 1 is most robust
        if not all_changes:
            return 0.0
            
        # Get mean change (typically negative for degrading performance)
        mean_change = np.mean(all_changes)
        
        # Convert to robustness score (higher is better)
        # Limit range to [0, 1]
        score = max(0, 1 + mean_change)
        score = min(1, score)
        
        return float(score)
    
    def plot_scenario_impact(
        self,
        scenario: str,
        metric: str = 'accuracy',
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot impact of a label shift scenario on model performance.
        
        Parameters:
        -----------
        scenario : str
            Scenario to plot
        metric : str
            Metric to plot
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.Figure : Impact plot
        """
        if not self.results:
            raise ValueError("No results available. Call test() first.")
            
        if 'scenarios' not in self.results:
            raise ValueError("No scenario results available.")
            
        if scenario not in self.results['scenarios']:
            raise ValueError(f"Scenario '{scenario}' not found in results.")
            
        # Get scenario results
        scenario_results = self.results['scenarios'][scenario]
        
        # Extract data
        ratios = [r['ratio'] for r in scenario_results]
        relative_changes = [r['relative_change'][metric] for r in scenario_results]
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot relative change
        ax.plot(ratios, relative_changes, 'o-', label=f'Relative change in {metric}')
        
        # Add reference line at 0
        ax.axhline(y=0, color='gray', linestyle='--')
        
        # Set labels and title
        ax.set_xlabel('Shift Ratio')
        ax.set_ylabel(f'Relative Change in {metric.upper()}')
        ax.set_title(f'Impact of {scenario.title()} Shift on Model Performance')
        
        # Add legend
        ax.legend()
        
        return fig
    
    def plot_class_distributions(
        self,
        scenario: str,
        ratio: float,
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Plot original vs. shifted class distributions.
        
        Parameters:
        -----------
        scenario : str
            Scenario to plot
        ratio : float
            Shift ratio to plot
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.Figure : Distribution plot
        """
        if not self.results:
            raise ValueError("No results available. Call test() first.")
            
        if 'scenarios' not in self.results or 'class_distribution' not in self.results:
            raise ValueError("Invalid results format.")
            
        if scenario not in self.results['scenarios']:
            raise ValueError(f"Scenario '{scenario}' not found in results.")
            
        # Get original class distribution
        original_dist = self.results['class_distribution']
        
        # Find result with matching ratio
        shifted_dist = None
        for result in self.results['scenarios'][scenario]:
            if result['ratio'] == ratio:
                shifted_dist = result['shifted_distribution']
                break
                
        if shifted_dist is None:
            raise ValueError(f"Ratio '{ratio}' not found for scenario '{scenario}'.")
            
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get classes
        all_classes = sorted(set(list(original_dist.keys()) + list(shifted_dist.keys())))
        x = np.arange(len(all_classes))
        width = 0.35
        
        # Get counts
        original_counts = [original_dist.get(c, 0) for c in all_classes]
        shifted_counts = [shifted_dist.get(c, 0) for c in all_classes]
        
        # Plot bars
        ax.bar(x - width/2, original_counts, width, label='Original Distribution')
        ax.bar(x + width/2, shifted_counts, width, label=f'{scenario.title()} Shifted Distribution')
        
        # Set labels and title
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_title(f'Original vs. Shifted Class Distributions ({scenario.title()}, ratio={ratio})')
        ax.set_xticks(x)
        ax.set_xticklabels(all_classes)
        
        # Add legend
        ax.legend()
        
        return fig


def test_label_shift_robustness(
    model: Any,
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    scenarios: Optional[List[str]] = None,
    shift_ratios: Optional[List[float]] = None,
    metrics: Optional[List[str]] = None,
    random_state: Optional[int] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Test classification model robustness against label distribution shifts.
    
    Parameters:
    -----------
    model : Any
        Classification model to test
    X : array-like or DataFrame
        Feature data
    y : array-like or Series
        Target data
    scenarios : list of str or None
        Label shift scenarios to test
    shift_ratios : list of float or None
        Ratios for simulating label shifts
    metrics : list of str or None
        Metrics to use for evaluation
    random_state : int or None
        Random seed for reproducibility
    verbose : bool
        Whether to print progress information
        
    Returns:
    --------
    dict : Label shift test results
    """
    tester = ClassificationLabelShiftTest(
        model=model,
        scenarios=scenarios,
        shift_ratios=shift_ratios,
        metrics=metrics,
        random_state=random_state,
        verbose=verbose
    )
    
    return tester.test(X, y)


def simulate_label_shift(
    X: Union[np.ndarray, pd.DataFrame],
    y: np.ndarray,
    scenario: str,
    ratio: float,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate label distribution shift.
    
    Parameters:
    -----------
    X : array-like or DataFrame
        Feature data
    y : array-like
        Target data
    scenario : str
        Label shift scenario:
        - 'imbalance': Increase/decrease class imbalance
        - 'minority': Over/under-sample minority classes
        - 'majority': Over/under-sample majority classes
        - 'reverse': Reverse class frequencies
    ratio : float
        Shift ratio (0 to 1)
    random_state : int or None
        Random seed for reproducibility
        
    Returns:
    --------
    tuple : (X_shifted, y_shifted)
    """
    # Set random seed
    np.random.seed(random_state)
    
    # Get class distribution
    classes, counts = np.unique(y, return_counts=True)
    n_classes = len(classes)
    
    # Handle DataFrame input
    is_dataframe = isinstance(X, pd.DataFrame)
    if is_dataframe:
        X_values = X.values
    else:
        X_values = X
        
    # Base case: return copy of original data
    X_shifted = X_values.copy()
    y_shifted = y.copy()
    
    if scenario == 'imbalance':
        # Increase class imbalance
        # Reduce minority classes and increase majority classes
        
        # Sort classes by frequency
        sorted_indices = np.argsort(counts)
        minority_classes = classes[sorted_indices[:n_classes//2]]
        majority_classes = classes[sorted_indices[n_classes//2:]]
        
        # New distribution - decrease minority, increase majority
        new_distribution = {}
        for i, c in enumerate(classes):
            if c in minority_classes:
                # Decrease minority by ratio
                new_distribution[c] = int(counts[i] * (1 - ratio/2))
            else:
                # Increase majority by ratio
                new_distribution[c] = int(counts[i] * (1 + ratio/2))
                
        # Create new dataset
        X_shifted, y_shifted = _resample_dataset(X_values, y, new_distribution, random_state)
        
    elif scenario == 'minority':
        # Over/under-sample minority classes
        
        # Sort classes by frequency
        sorted_indices = np.argsort(counts)
        minority_classes = classes[sorted_indices[:n_classes//2]]
        
        # New distribution - adjust minority classes
        new_distribution = {}
        for i, c in enumerate(classes):
            if c in minority_classes:
                # Adjust minority by ratio (over/under-sample)
                if ratio <= 0.5:
                    # Undersample (0 -> 0%, 0.5 -> 50%)
                    factor = 1 - ratio
                else:
                    # Oversample (0.5 -> 100%, 1.0 -> 200%)
                    factor = 1 + (ratio - 0.5) * 2
                new_distribution[c] = int(counts[i] * factor)
            else:
                # Keep majority the same
                new_distribution[c] = counts[i]
                
        # Create new dataset
        X_shifted, y_shifted = _resample_dataset(X_values, y, new_distribution, random_state)
        
    elif scenario == 'majority':
        # Over/under-sample majority classes
        
        # Sort classes by frequency
        sorted_indices = np.argsort(counts)
        majority_classes = classes[sorted_indices[n_classes//2:]]
        
        # New distribution - adjust majority classes
        new_distribution = {}
        for i, c in enumerate(classes):
            if c in majority_classes:
                # Adjust majority by ratio (over/under-sample)
                if ratio <= 0.5:
                    # Undersample (0 -> 0%, 0.5 -> 50%)
                    factor = 1 - ratio
                else:
                    # Oversample (0.5 -> 100%, 1.0 -> 200%)
                    factor = 1 + (ratio - 0.5) * 2
                new_distribution[c] = int(counts[i] * factor)
            else:
                # Keep minority the same
                new_distribution[c] = counts[i]
                
        # Create new dataset
        X_shifted, y_shifted = _resample_dataset(X_values, y, new_distribution, random_state)
        
    elif scenario == 'reverse':
        # Reverse class frequencies
        
        # Sort classes by frequency
        sorted_indices = np.argsort(counts)
        
        # New distribution - reverse frequencies
        new_distribution = {}
        for i, c in enumerate(classes):
            # Find reversed index (weighted by ratio)
            rev_idx = n_classes - 1 - i
            orig_idx = i
            
            # Interpolate between original and reversed based on ratio
            interp_idx = int(orig_idx * (1 - ratio) + rev_idx * ratio)
            interp_idx = max(0, min(interp_idx, n_classes - 1))
            
            # Use count from the interpolated position
            new_distribution[c] = counts[sorted_indices[interp_idx]]
                
        # Create new dataset
        X_shifted, y_shifted = _resample_dataset(X_values, y, new_distribution, random_state)
        
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
        
    # Convert back to DataFrame if input was DataFrame
    if is_dataframe and not isinstance(X_shifted, pd.DataFrame):
        X_shifted = pd.DataFrame(X_shifted, columns=X.columns, index=None)
        
    return X_shifted, y_shifted


def _resample_dataset(
    X: np.ndarray,
    y: np.ndarray,
    new_distribution: Dict[Any, int],
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample dataset according to new class distribution.
    
    Parameters:
    -----------
    X : array-like
        Feature data
    y : array-like
        Target data
    new_distribution : dict
        New class distribution {class: count}
    random_state : int or None
        Random seed for reproducibility
        
    Returns:
    --------
    tuple : (X_resampled, y_resampled)
    """
    # Set random seed
    rng = np.random.RandomState(random_state)
    
    # Initialize resampled datasets
    X_resampled = []
    y_resampled = []
    
    # Process each class
    for cls, count in new_distribution.items():
        # Find samples of this class
        class_mask = (y == cls)
        X_cls = X[class_mask]
        y_cls = y[class_mask]
        
        # Skip if no samples or no resampling needed
        if len(X_cls) == 0 or count == 0:
            continue
            
        # Resample
        if count <= len(X_cls):
            # Undersample - random subset
            indices = rng.choice(len(X_cls), size=count, replace=False)
            X_resampled.append(X_cls[indices])
            y_resampled.append(y_cls[indices])
        else:
            # Oversample - include all original plus random duplicates
            n_duplicates = count - len(X_cls)
            indices = rng.choice(len(X_cls), size=n_duplicates, replace=True)
            
            X_resampled.append(X_cls)  # Original samples
            X_resampled.append(X_cls[indices])  # Duplicates
            
            y_resampled.append(y_cls)  # Original samples
            y_resampled.append(y_cls[indices])  # Duplicates
    
    # Combine resampled classes
    if not X_resampled:
        # If empty, return empty arrays
        return np.empty((0, X.shape[1])), np.empty(0, dtype=y.dtype)
        
    X_combined = np.vstack(X_resampled)
    y_combined = np.concatenate(y_resampled)
    
    # Shuffle
    indices = np.arange(len(y_combined))
    rng.shuffle(indices)
    
    return X_combined[indices], y_combined[indices]