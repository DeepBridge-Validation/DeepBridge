"""
Validators for report data.
"""

import logging
from typing import Any, Dict, List, Union, Optional

# Configure logger
logger = logging.getLogger("deepbridge.reports")

class DataValidator:
    """
    Validates data for report generation.
    """
    
    @staticmethod
    def validate_robustness_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and fix robustness data.
        
        Parameters:
        -----------
        data : Dict[str, Any]
            Robustness data to validate
            
        Returns:
        --------
        Dict[str, Any] : Validated data with defaults for missing fields
        """
        # Make a copy to avoid modifying the original
        validated = data.copy()
        
        # Required fields with defaults
        required_fields = {
            'model_name': 'Model',
            'model_type': 'Unknown Model',
            'metric': 'score',
            'base_score': 0.0,
            'robustness_score': 0.0,
            'raw_impact': 0.0,
            'quantile_impact': 0.0,
            'feature_subset': [],
            'feature_subset_display': 'All Features',
            'feature_importance': {},
            'model_feature_importance': {}
        }
        
        # Set defaults for missing fields
        for field, default in required_fields.items():
            if field not in validated or validated[field] is None:
                logger.warning(f"Missing required field '{field}', setting default value")
                validated[field] = default
        
        # Ensure raw and quantile data structures exist
        if 'raw' not in validated or not isinstance(validated['raw'], dict):
            validated['raw'] = {}
        
        if 'quantile' not in validated or not isinstance(validated['quantile'], dict):
            validated['quantile'] = {}
        
        # Convert feature subset to list if it's a string
        if isinstance(validated['feature_subset'], str):
            validated['feature_subset'] = [validated['feature_subset']]
        
        # Set feature_subset_display
        if validated['feature_subset']:
            validated['feature_subset_display'] = ", ".join(validated['feature_subset'])
        else:
            validated['feature_subset_display'] = "All Features"
        
        return validated
    
    @staticmethod
    def validate_uncertainty_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and fix uncertainty data.
        
        Parameters:
        -----------
        data : Dict[str, Any]
            Uncertainty data to validate
            
        Returns:
        --------
        Dict[str, Any] : Validated data with defaults for missing fields
        """
        # Make a copy to avoid modifying the original
        validated = data.copy()
        
        # Required fields with defaults
        required_fields = {
            'model_name': 'Model',
            'model_type': 'Unknown Model',
            'metric': 'score',
            'method': 'crqr',
            'uncertainty_score': 0.5,
            'avg_coverage': 0.0,
            'avg_width': 0.0,
            'alpha_levels': []
        }
        
        # Set defaults for missing fields
        for field, default in required_fields.items():
            if field not in validated or validated[field] is None:
                logger.warning(f"Missing required field '{field}', setting default value")
                validated[field] = default
        
        # Ensure metrics dictionary exists
        if 'metrics' not in validated or not isinstance(validated['metrics'], dict):
            validated['metrics'] = {}
        
        return validated
    
    @staticmethod
    def validate_resilience_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and fix resilience data.
        
        Parameters:
        -----------
        data : Dict[str, Any]
            Resilience data to validate
            
        Returns:
        --------
        Dict[str, Any] : Validated data with defaults for missing fields
        """
        # Make a copy to avoid modifying the original
        validated = data.copy()
        
        # Required fields with defaults
        required_fields = {
            'model_name': 'Model',
            'model_type': 'Unknown Model',
            'metric': 'score',
            'resilience_score': 0.0,
            'avg_performance_gap': 0.0,
            'distribution_shift_results': [],
            'distance_metrics': ['PSI', 'KS', 'WD1'],
            'alphas': [0.1, 0.2, 0.3]
        }
        
        # Set defaults for missing fields
        for field, default in required_fields.items():
            if field not in validated or validated[field] is None:
                logger.warning(f"Missing required field '{field}', setting default value")
                validated[field] = default
        
        # Ensure metrics dictionary exists
        if 'metrics' not in validated or not isinstance(validated['metrics'], dict):
            validated['metrics'] = {}
        
        return validated
    
    @staticmethod
    def validate_hyperparameter_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and fix hyperparameter data.
        
        Parameters:
        -----------
        data : Dict[str, Any]
            Hyperparameter data to validate
            
        Returns:
        --------
        Dict[str, Any] : Validated data with defaults for missing fields
        """
        # Make a copy to avoid modifying the original
        validated = data.copy()
        
        # Required fields with defaults
        required_fields = {
            'model_name': 'Model',
            'model_type': 'Unknown Model',
            'metric': 'score',
            'base_score': 0.0,
            'importance_scores': {},
            'tuning_order': [],
            'importance_results': [],
            'optimization_results': []
        }
        
        # Set defaults for missing fields
        for field, default in required_fields.items():
            if field not in validated or validated[field] is None:
                logger.warning(f"Missing required field '{field}', setting default value")
                validated[field] = default
        
        # Ensure metrics dictionary exists
        if 'metrics' not in validated or not isinstance(validated['metrics'], dict):
            validated['metrics'] = {}
        
        return validated