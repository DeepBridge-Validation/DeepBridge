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
        
        # Extract data from test_results structure
        if 'test_results' in report_data:
            test_results = report_data['test_results']
            logger.info(f"Found 'test_results' key with keys: {list(test_results.keys())}")

            # Extract from test_results.primary_model
            if 'primary_model' in test_results:
                primary_model = test_results['primary_model']
                logger.info(f"Found primary_model in test_results with keys: {list(primary_model.keys())}")

                # Copy all fields from test_results.primary_model to top level
                for key, value in primary_model.items():
                    if key not in report_data:
                        report_data[key] = value

                # Extract metrics for base_score
                if 'metrics' in primary_model and not report_data.get('base_score'):
                    metrics = primary_model['metrics']
                    # Try to get the main metric (accuracy, roc_auc, etc.)
                    for metric_name in ['accuracy', 'roc_auc', 'f1', 'precision']:
                        if metric_name in metrics:
                            report_data['base_score'] = metrics[metric_name]
                            report_data['metric'] = metric_name
                            logger.info(f"Set base_score to {metrics[metric_name]} from {metric_name}")
                            break

        # Extract feature_importance from initial_model_evaluation
        if 'initial_model_evaluation' in report_data:
            initial_eval = report_data['initial_model_evaluation']
            if 'models' in initial_eval and 'primary_model' in initial_eval['models']:
                initial_primary = initial_eval['models']['primary_model']
                logger.info(f"Found primary_model in initial_model_evaluation")

                # Extract feature importance
                if 'feature_importance' in initial_primary and not report_data.get('feature_importance'):
                    report_data['feature_importance'] = initial_primary['feature_importance']
                    logger.info(f"Extracted {len(initial_primary['feature_importance'])} features from initial_model_evaluation")

                # Use as model_feature_importance if not set
                if 'feature_importance' in initial_primary and not report_data.get('model_feature_importance'):
                    report_data['model_feature_importance'] = initial_primary['feature_importance']

        # Handle case where results are nested under 'primary_model' key (fallback)
        if 'primary_model' in report_data:
            logger.info("Found 'primary_model' key at root level, extracting data...")
            primary_data = report_data['primary_model']

            # Extract feature importance data
            if 'feature_importance' in primary_data and not report_data.get('feature_importance'):
                logger.info(f"Found feature_importance in primary_model with {len(primary_data['feature_importance'])} features")
                report_data['feature_importance'] = primary_data['feature_importance']

            if 'model_feature_importance' in primary_data and not report_data.get('model_feature_importance'):
                logger.info(f"Found model_feature_importance in primary_model with {len(primary_data['model_feature_importance'])} features")
                report_data['model_feature_importance'] = primary_data['model_feature_importance']

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
        
        # Check for feature importance and alternative models in nested structure
        if 'results' in report_data:
            if 'resilience' in report_data['results']:
                resilience_results = report_data['results']['resilience']
                logger.info(f"Found resilience key with keys: {list(resilience_results.keys())}")

                # Check in direct resilience object
                if 'feature_importance' in resilience_results and resilience_results['feature_importance']:
                    logger.info(f"Found feature_importance directly in results.resilience with {len(resilience_results['feature_importance'])} features")
                    report_data['feature_importance'] = resilience_results['feature_importance']

                if 'model_feature_importance' in resilience_results and resilience_results['model_feature_importance']:
                    logger.info(f"Found model_feature_importance directly in results.resilience with {len(resilience_results['model_feature_importance'])} features")
                    report_data['model_feature_importance'] = resilience_results['model_feature_importance']

                # Check in nested results
                if 'results' in resilience_results:
                    nested_results = resilience_results['results']
                    logger.info(f"Found nested results with keys: {list(nested_results.keys())}")

                    # Check for alternative models in nested structure
                    if 'alternative_models' in nested_results and 'alternative_models' not in report_data:
                        logger.info("Found alternative_models in nested structure")
                        report_data['alternative_models'] = nested_results['alternative_models']

                    # Check for feature importance in primary_model
                    if 'primary_model' in nested_results:
                        primary_model = nested_results['primary_model']
                        logger.info("Found primary_model in nested results.resilience.results")

                        if 'feature_importance' in primary_model and primary_model['feature_importance']:
                            logger.info(f"Found feature_importance in nested results with {len(primary_model['feature_importance'])} features")
                            report_data['feature_importance'] = primary_model['feature_importance']

                        if 'model_feature_importance' in primary_model and primary_model['model_feature_importance']:
                            logger.info(f"Found model_feature_importance in nested results with {len(primary_model['model_feature_importance'])} features")
                            report_data['model_feature_importance'] = primary_model['model_feature_importance']
        
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

    def _generate_boxplot_data(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate boxplot data structure from distribution shift results.

        Parameters:
        -----------
        report_data : Dict[str, Any]
            Report data containing distribution shift results

        Returns:
        --------
        Dict[str, Any] : Boxplot data structure
        """
        try:
            import math

            # Collect performance scores from shift scenarios
            scores = []

            # Try distribution_shift.all_results first (primary source)
            if 'distribution_shift' in report_data and 'all_results' in report_data['distribution_shift']:
                for result in report_data['distribution_shift']['all_results']:
                    # Try remaining_metric first, then target_performance
                    score = result.get('remaining_metric') or result.get('target_performance')

                    # Skip NaN values
                    if score is not None and not (isinstance(score, float) and math.isnan(score)):
                        scores.append(score)

            # Try distribution_shift_results as fallback
            if not scores and 'distribution_shift_results' in report_data:
                for result in report_data['distribution_shift_results']:
                    score = result.get('remaining_metric') or result.get('target_performance')

                    if score is not None and not (isinstance(score, float) and math.isnan(score)):
                        scores.append(score)

            if not scores:
                logger.warning("No performance scores found for boxplot generation")
                return {}

            # Calculate statistics
            import numpy as np
            scores_array = np.array(scores)

            base_score = report_data.get('base_score', 0)
            median = float(np.median(scores_array))
            q1 = float(np.percentile(scores_array, 25))
            q3 = float(np.percentile(scores_array, 75))
            iqr = q3 - q1
            mad = float(np.median(np.abs(scores_array - median)))
            min_score = float(np.min(scores_array))
            max_score = float(np.max(scores_array))

            # Detect outliers (values outside 1.5 * IQR from Q1/Q3)
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = [float(s) for s in scores_array if s < lower_bound or s > upper_bound]

            # Score drop from baseline
            score_drop = base_score - median if base_score else 0

            # Create boxplot data structure
            model_name = report_data.get('model_name', 'Model')
            boxplot_data = {
                'models': {
                    model_name: {
                        'base_score': base_score,
                        'median': median,
                        'mad': mad,
                        'iqr': iqr,
                        'min': min_score,
                        'max': max_score,
                        'outliers': outliers,
                        'score_drop': score_drop,
                        'q1': q1,
                        'q3': q3,
                        'scores': scores  # Keep all scores for plotting
                    }
                }
            }

            logger.info(f"Generated boxplot data: median={median:.4f}, iqr={iqr:.4f}, outliers={len(outliers)}")
            return boxplot_data

        except Exception as e:
            logger.error(f"Error generating boxplot data: {str(e)}")
            return {}