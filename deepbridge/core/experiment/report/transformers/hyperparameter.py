"""
Data transformation module for hyperparameter reports.
"""

import logging
import datetime
from typing import Dict, Any, Optional

from ..base import DataTransformer

# Configure logger
logger = logging.getLogger("deepbridge.reports")

class HyperparameterDataTransformer(DataTransformer):
    """
    Transforms hyperparameter test results data for templates.
    """
    
    def transform(self, results: Dict[str, Any], model_name: str, timestamp: Optional[str] = None) -> Dict[str, Any]:
        """
        Transform hyperparameter results data for template rendering.
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Raw hyperparameter test results
        model_name : str
            Name of the model
        timestamp : str, optional
            Timestamp for the report
            
        Returns:
        --------
        Dict[str, Any] : Transformed data for templates
        """
        logger.info("Transforming hyperparameter data structure...")
        
        if timestamp is None:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Create a deep copy of the results
        report_data = self._deep_copy(results)
        
        # Handle to_dict() method if available
        if hasattr(report_data, 'to_dict'):
            report_data = report_data.to_dict()
        
        # Handle case where results are nested under 'primary_model' key
        if 'primary_model' in report_data:
            logger.info("Found 'primary_model' key, extracting data...")
            primary_data = report_data['primary_model']
            # Copy fields from primary_model to the top level
            for key, value in primary_data.items():
                if key not in report_data:
                    report_data[key] = value
        
        # Add metadata for display
        report_data['model_name'] = report_data.get('model_name', model_name)
        report_data['timestamp'] = report_data.get('timestamp', timestamp)
        
        # Set model_type
        if 'model_type' not in report_data:
            # Try to get from primary_model if available
            if 'primary_model' in report_data and 'model_type' in report_data['primary_model']:
                report_data['model_type'] = report_data['primary_model']['model_type']
            else:
                report_data['model_type'] = "Unknown Model"
        
        # Ensure we have a proper metrics structure
        if 'metrics' not in report_data:
            report_data['metrics'] = {}
        
        # Check for alternative models in nested structure
        if 'alternative_models' not in report_data and 'results' in report_data:
            if 'hyperparameter' in report_data['results']:
                hyperparameter_results = report_data['results']['hyperparameter']
                if 'results' in hyperparameter_results and 'alternative_models' in hyperparameter_results['results']:
                    logger.info("Found alternative_models in nested structure")
                    report_data['alternative_models'] = hyperparameter_results['results']['alternative_models']
        
        # Make sure we have importance_results
        if 'importance_results' not in report_data:
            # Try to extract from other fields if possible
            if 'importance' in report_data and 'all_results' in report_data['importance']:
                report_data['importance_results'] = report_data['importance']['all_results']
            else:
                # Create empty results
                report_data['importance_results'] = []
        
        # Ensure we have importance_scores at top level
        if 'importance_scores' not in report_data:
            # Try to get from importance section if it exists
            if 'importance' in report_data and 'all_results' in report_data['importance'] and report_data['importance']['all_results']:
                first_result = report_data['importance']['all_results'][0]
                if 'normalized_importance' in first_result:
                    report_data['importance_scores'] = first_result['normalized_importance']
                elif 'raw_importance_scores' in first_result:
                    report_data['importance_scores'] = first_result['raw_importance_scores']
        
        # Ensure we have tuning_order
        if 'tuning_order' not in report_data:
            # Try to extract from results
            if 'importance' in report_data and 'all_results' in report_data['importance'] and report_data['importance']['all_results']:
                result = report_data['importance']['all_results'][0]
                if 'tuning_order' in result:
                    report_data['tuning_order'] = result['tuning_order']
                elif 'sorted_importance' in result:
                    # Use keys of sorted_importance as tuning_order
                    report_data['tuning_order'] = list(result['sorted_importance'].keys())
        
        # Convert all numpy types to Python native types
        return self.convert_numpy_types(report_data)