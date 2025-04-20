"""
Robustness tests for classification models.

This module provides specialized robustness testing tools for
classification models, including feature perturbation, label shift,
and specialized metrics.
"""

from .feature_perturbation_test import (
    ClassificationFeaturePerturbationTest,
    test_classifier_feature_robustness,
    plot_feature_perturbation_impact
)

from .label_shift_test import (
    ClassificationLabelShiftTest,
    test_label_shift_robustness,
    simulate_label_shift
)

from .classifier_metrics import (
    ClassificationRobustnessMetrics,
    compute_robustness_metrics,
    compute_robustness_score
)

# Combine into a single interface
class ClassificationRobustnessTests:
    """
    Comprehensive robustness tests for classification models.
    
    This class integrates various robustness testing components
    specifically designed for classification models.
    """
    
    def __init__(
        self,
        model,
        feature_perturbation_types=None,
        feature_perturbation_levels=None,
        label_shift_scenarios=None,
        **kwargs
    ):
        """
        Initialize the classification robustness tests.
        
        Parameters:
        -----------
        model : Any
            Classification model to test
        feature_perturbation_types : list or None
            Types of feature perturbations to test
        feature_perturbation_levels : list or None
            Levels of perturbation to apply
        label_shift_scenarios : list or None
            Label shift scenarios to test
        **kwargs : dict
            Additional parameters
        """
        self.model = model
        self.kwargs = kwargs
        
        # Initialize component tests
        self.feature_test = ClassificationFeaturePerturbationTest(
            model=model,
            perturbation_types=feature_perturbation_types,
            perturbation_levels=feature_perturbation_levels,
            **kwargs
        )
        
        self.label_shift_test = ClassificationLabelShiftTest(
            model=model,
            scenarios=label_shift_scenarios,
            **kwargs
        )
        
        self.metrics = ClassificationRobustnessMetrics(model=model, **kwargs)
    
    def test_feature_robustness(self, X, y, **kwargs):
        """
        Test robustness against feature perturbations.
        
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
        dict : Feature robustness test results
        """
        return self.feature_test.test(X, y, **kwargs)
    
    def test_label_shift_robustness(self, X, y, **kwargs):
        """
        Test robustness against label shift.
        
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
        dict : Label shift robustness test results
        """
        return self.label_shift_test.test(X, y, **kwargs)
    
    def compute_metrics(self, X, y, **kwargs):
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
        return self.metrics.compute(X, y, **kwargs)
    
    def test_all(self, X, y, **kwargs):
        """
        Run all robustness tests.
        
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
        dict : Comprehensive robustness test results
        """
        results = {}
        
        # Test feature robustness
        results['feature_robustness'] = self.test_feature_robustness(X, y, **kwargs)
        
        # Test label shift robustness
        results['label_shift'] = self.test_label_shift_robustness(X, y, **kwargs)
        
        # Compute metrics
        results['metrics'] = self.compute_metrics(X, y, **kwargs)
        
        # Compute overall robustness score
        feature_score = results['feature_robustness'].get('robustness_score', 0)
        label_shift_score = results['label_shift'].get('robustness_score', 0)
        
        # Weight the scores (can be adjusted)
        results['overall_robustness_score'] = 0.6 * feature_score + 0.4 * label_shift_score
        
        return results

__all__ = [
    'ClassificationRobustnessTests',
    'ClassificationFeaturePerturbationTest',
    'test_classifier_feature_robustness',
    'plot_feature_perturbation_impact',
    'ClassificationLabelShiftTest',
    'test_label_shift_robustness',
    'simulate_label_shift',
    'ClassificationRobustnessMetrics',
    'compute_robustness_metrics',
    'compute_robustness_score'
]