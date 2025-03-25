"""
Classification model validation components.

This module provides validation tools and tests specifically designed
for classification models.
"""

from .hyperparameter_tests import (
    ClassificationHyperparameterTests,
    tune_classification_hyperparameters,
    get_classification_param_grid,
    evaluate_classifier_hyperparameters
)

# Import when implemented
# from .robustness_tests import ClassificationRobustnessTests
# from .uncertainty_tests import ClassificationUncertaintyTests

class ClassificationValidator:
    """
    Comprehensive validator for classification models.
    
    This class integrates various validation components specifically
    designed for classification models.
    """
    
    def __init__(self, model, **kwargs):
        """
        Initialize the classification validator.
        
        Parameters:
        -----------
        model : Any
            Classification model to validate
        **kwargs : dict
            Additional parameters for the validator
        """
        self.model = model
        self.kwargs = kwargs
        
        # Initialize component validators
        self.hyperparameter_validator = ClassificationHyperparameterTests(model, **kwargs)
        
        # Initialize other components when implemented
        # self.robustness_validator = ClassificationRobustnessTests(model, **kwargs)
        # self.uncertainty_validator = ClassificationUncertaintyTests(model, **kwargs)
    
    def validate_hyperparameters(self, X, y, **kwargs):
        """
        Validate hyperparameters of the classification model.
        
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
        dict : Validation results
        """
        return self.hyperparameter_validator.validate(X, y, **kwargs)
    
    def validate_all(self, X, y, **kwargs):
        """
        Perform comprehensive validation of the classification model.
        
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
        dict : Validation results
        """
        results = {}
        
        # Validate hyperparameters
        results['hyperparameters'] = self.validate_hyperparameters(X, y, **kwargs)
        
        # Add other validations when implemented
        # results['robustness'] = self.robustness_validator.validate(X, y, **kwargs)
        # results['uncertainty'] = self.uncertainty_validator.validate(X, y, **kwargs)
        
        return results

__all__ = [
    'ClassificationValidator',
    'ClassificationHyperparameterTests',
    'tune_classification_hyperparameters',
    'get_classification_param_grid',
    'evaluate_classifier_hyperparameters'
]