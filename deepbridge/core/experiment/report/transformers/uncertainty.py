"""
Data transformation module for uncertainty reports.
"""

import logging
import datetime
from typing import Dict, Any, Optional

from ..base import DataTransformer

# Configure logger
logger = logging.getLogger("deepbridge.reports")

class UncertaintyDataTransformer(DataTransformer):
    """
    Transforms uncertainty test results data for templates.
    """
    
    def transform(self, results: Dict[str, Any], model_name: str, timestamp: Optional[str] = None) -> Dict[str, Any]:
        """
        Transform uncertainty results data for template rendering.
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Raw uncertainty test results
        model_name : str
            Name of the model
        timestamp : str, optional
            Timestamp for the report
            
        Returns:
        --------
        Dict[str, Any] : Transformed data for templates
        """
        logger.info("Transforming uncertainty data structure...")
        
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
                if key not in report_data or key == 'crqr':
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
        
        # Set uncertainty score if not present
        if 'uncertainty_score' not in report_data:
            # Try to calculate from CRQR data
            if 'crqr' in report_data and 'by_alpha' in report_data['crqr']:
                # Average coverage quality (actual/expected ratio with penalty for over-coverage)
                coverage_ratios = []
                for alpha_key, alpha_data in report_data['crqr']['by_alpha'].items():
                    if 'overall_result' in alpha_data:
                        actual = alpha_data['overall_result'].get('coverage', 0)
                        expected = alpha_data['overall_result'].get('expected_coverage', 0)
                        if expected > 0:
                            # Penalize over-coverage less than under-coverage
                            ratio = min(actual / expected, 1.1) if actual > expected else actual / expected
                            coverage_ratios.append(ratio)
                
                if coverage_ratios:
                    report_data['uncertainty_score'] = sum(coverage_ratios) / len(coverage_ratios)
                else:
                    report_data['uncertainty_score'] = 0.5
            else:
                report_data['uncertainty_score'] = 0.5
        
        # Calculate average coverage and width if not present
        if 'avg_coverage' not in report_data and 'crqr' in report_data and 'by_alpha' in report_data['crqr']:
            coverages = []
            widths = []
            
            for alpha_key, alpha_data in report_data['crqr']['by_alpha'].items():
                if 'overall_result' in alpha_data:
                    coverages.append(alpha_data['overall_result'].get('coverage', 0))
                    widths.append(alpha_data['overall_result'].get('mean_width', 0))
            
            if coverages:
                report_data['avg_coverage'] = sum(coverages) / len(coverages)
            else:
                report_data['avg_coverage'] = 0
                
            if widths:
                report_data['avg_width'] = sum(widths) / len(widths)
            else:
                report_data['avg_width'] = 0
        
        # Ensure we have alpha levels
        if 'alpha_levels' not in report_data and 'crqr' in report_data and 'by_alpha' in report_data['crqr']:
            report_data['alpha_levels'] = list(map(float, report_data['crqr']['by_alpha'].keys()))
        
        # Set method if not present
        if 'method' not in report_data:
            report_data['method'] = 'crqr'
        
        # Check for alternative models in nested structure
        if 'alternative_models' not in report_data and 'results' in report_data:
            if 'uncertainty' in report_data['results']:
                uncertainty_results = report_data['results']['uncertainty']
                if 'results' in uncertainty_results and 'alternative_models' in uncertainty_results['results']:
                    logger.info("Found alternative_models in nested structure")
                    report_data['alternative_models'] = uncertainty_results['results']['alternative_models']
        
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