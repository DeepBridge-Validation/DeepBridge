"""
Data transformation module for resilience reports.
"""

import logging
import datetime
from typing import Dict, Any, Optional

from ..base import DataTransformer

# Configure logger
logger = logging.getLogger("deepbridge.reports")

class ResilienceDataTransformer(DataTransformer):
    """
    Transforms resilience test results data for templates.
    """
    
    def transform(self, results: Dict[str, Any], model_name: str, timestamp: Optional[str] = None) -> Dict[str, Any]:
        """
        Transform resilience results data for template rendering.
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Raw resilience test results
        model_name : str
            Name of the model
        timestamp : str, optional
            Timestamp for the report
            
        Returns:
        --------
        Dict[str, Any] : Transformed data for templates
        """
        logger.info("Transforming resilience data structure...")
        
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
        
        # Ensure metric name is available
        if 'metric' not in report_data:
            report_data['metric'] = next(iter(report_data.get('metrics', {}).keys()), 'score')
        
        # Check for alternative models in nested structure
        if 'alternative_models' not in report_data and 'results' in report_data:
            if 'resilience' in report_data['results']:
                resilience_results = report_data['results']['resilience']
                if 'results' in resilience_results and 'alternative_models' in resilience_results['results']:
                    logger.info("Found alternative_models in nested structure")
                    report_data['alternative_models'] = resilience_results['results']['alternative_models']
        
        # Make sure we have distribution_shift_results
        if 'distribution_shift_results' not in report_data:
            # Try to extract from other fields if possible
            if 'test_results' in report_data and isinstance(report_data['test_results'], list):
                report_data['distribution_shift_results'] = report_data['test_results']
            elif 'distribution_shift' in report_data and 'all_results' in report_data['distribution_shift']:
                # Extract results from the nested structure
                report_data['distribution_shift_results'] = report_data['distribution_shift']['all_results']
            else:
                # Create empty results
                report_data['distribution_shift_results'] = []
        
        # Ensure we have distance metrics and alphas
        if 'distance_metrics' not in report_data:
            distance_metrics = set()
            for result in report_data.get('distribution_shift_results', []):
                if 'distance_metric' in result:
                    distance_metrics.add(result['distance_metric'])
            report_data['distance_metrics'] = list(distance_metrics) if distance_metrics else ['PSI', 'KS', 'WD1']
        
        if 'alphas' not in report_data:
            alphas = set()
            for result in report_data.get('distribution_shift_results', []):
                if 'alpha' in result:
                    alphas.add(result['alpha'])
            report_data['alphas'] = sorted(list(alphas)) if alphas else [0.1, 0.2, 0.3]
        
        # Calculate average performance gap if not present
        if 'avg_performance_gap' not in report_data:
            performance_gaps = []
            for result in report_data.get('distribution_shift_results', []):
                if 'performance_gap' in result:
                    performance_gaps.append(result['performance_gap'])
            
            if performance_gaps:
                report_data['avg_performance_gap'] = sum(performance_gaps) / len(performance_gaps)
            elif 'resilience_score' in report_data:
                # If we have resilience score but no average gap, calculate gap from score
                report_data['avg_performance_gap'] = 1.0 - report_data['resilience_score']
            else:
                report_data['avg_performance_gap'] = 0.0
        
        # Process alternative models if present
        if 'alternative_models' in report_data:
            logger.info("Processing alternative models data...")
            
            # Initialize alternative models dict if needed
            if not isinstance(report_data['alternative_models'], dict):
                report_data['alternative_models'] = {}
            
            # Process each alternative model
            for alt_model_name, model_data in report_data['alternative_models'].items():
                logger.info(f"Processing alternative model: {alt_model_name}")
                
                # Ensure metrics exist
                if 'metrics' not in model_data:
                    model_data['metrics'] = {}
                    
                # Update the model data in the report
                report_data['alternative_models'][alt_model_name] = model_data
        
        # Convert all numpy types to Python native types
        return self.convert_numpy_types(report_data)