"""
Data transformer for fairness reports.

Transforms raw fairness results into a format suitable for report generation.
"""

from typing import Dict, Any, List
import logging

from .utils import (
    get_status_from_interpretation,
    get_assessment_text,
    POSTTRAIN_MAIN_METRICS,
    POSTTRAIN_COMPLEMENTARY_METRICS
)
from .chart_factory import ChartFactory

logger = logging.getLogger("deepbridge.reports")


class FairnessDataTransformer:
    """
    Transforms fairness experiment results for report generation.

    Focuses on data transformation and delegates chart creation to ChartFactory.
    """

    def __init__(self):
        """Initialize transformer with chart factory."""
        self.chart_factory = ChartFactory()

    def transform(self, results: Dict[str, Any], model_name: str = "Model") -> Dict[str, Any]:
        """
        Transform raw fairness results into report-ready format.

        Args:
            results: Dictionary containing fairness analysis results from FairnessSuite
            model_name: Name of the model

        Returns:
            Dictionary with transformed data ready for report rendering
        """
        logger.info("Transforming fairness data for report")

        # Extract main components
        protected_attrs = results.get('protected_attributes', [])
        pretrain_metrics = results.get('pretrain_metrics', {})
        posttrain_metrics = results.get('posttrain_metrics', {})
        warnings = results.get('warnings', [])
        critical_issues = results.get('critical_issues', [])
        overall_score = results.get('overall_fairness_score', 0.0)
        age_grouping_applied = results.get('age_grouping_applied', {})
        dataset_info = results.get('dataset_info', {})
        config = results.get('config', {})
        confusion_matrix = results.get('confusion_matrix', {})
        threshold_analysis = results.get('threshold_analysis', None)

        # Transform the data
        transformed = {
            'model_name': model_name,
            'model_type': 'Classification Model',

            # Summary metrics
            'summary': self._create_summary(results),

            # Protected attributes data (split into 3 categories)
            'protected_attributes': self._transform_protected_attributes(
                protected_attrs, pretrain_metrics, posttrain_metrics
            ),

            # Issues and warnings
            'issues': self._transform_issues(warnings, critical_issues),

            # Dataset information
            'dataset_info': self._transform_dataset_info(dataset_info),

            # Test configuration
            'test_config': self._transform_test_config(config, age_grouping_applied),

            # Charts data (Plotly JSON) - delegated to ChartFactory
            'charts': self.chart_factory.create_all_charts(results),

            # Metadata
            'metadata': {
                'total_attributes': len(protected_attrs),
                'total_pretrain_metrics': sum(len(m) for m in pretrain_metrics.values()),
                'total_posttrain_metrics': sum(len(m) for m in posttrain_metrics.values()),
                'has_threshold_analysis': threshold_analysis is not None,
                'has_confusion_matrix': bool(confusion_matrix),
                'age_grouping_applied': age_grouping_applied,
                'age_grouping_enabled': len(age_grouping_applied) > 0
            }
        }

        logger.info(f"Transformation complete. {len(protected_attrs)} protected attributes analyzed")
        return transformed

    def _create_summary(self, results: Dict) -> Dict[str, Any]:
        """Create summary statistics."""
        overall_score = results.get('overall_fairness_score', 0.0)
        return {
            'overall_fairness_score': float(overall_score),
            'total_warnings': len(results.get('warnings', [])),
            'total_critical': len(results.get('critical_issues', [])),
            'total_attributes': len(results.get('protected_attributes', [])),
            'config': results.get('config', 'custom'),
            'assessment': get_assessment_text(overall_score)
        }

    def _transform_protected_attributes(
        self,
        attributes: List[str],
        pretrain: Dict[str, Dict],
        posttrain: Dict[str, Dict]
    ) -> List[Dict]:
        """
        Transform protected attributes data.

        Splits post-training metrics into:
        - posttrain_main: 5 critical compliance metrics
        - posttrain_complementary: 6 additional metrics
        """
        transformed_attrs = []

        for attr in attributes:
            attr_data = {
                'name': attr,
                'pretrain_metrics': [],
                'posttrain_main': [],
                'posttrain_complementary': []
            }

            # Transform pre-training metrics
            if attr in pretrain:
                for metric_name, metric_result in pretrain[attr].items():
                    if isinstance(metric_result, dict):
                        metric_data = {
                            'name': metric_name.replace('_', ' ').title(),
                            'value': metric_result.get('value', 0.0),
                            'interpretation': metric_result.get('interpretation', ''),
                            'status': get_status_from_interpretation(
                                metric_result.get('interpretation', '')
                            )
                        }

                        # Include groups if available
                        if 'all_groups' in metric_result:
                            metric_data['groups'] = metric_result['all_groups']

                        attr_data['pretrain_metrics'].append(metric_data)

            # Transform post-training metrics (split into main and complementary)
            if attr in posttrain:
                for metric_name, metric_result in posttrain[attr].items():
                    if isinstance(metric_result, dict):
                        # Base metric data
                        metric_data = {
                            'name': metric_name.replace('_', ' ').title(),
                            'interpretation': metric_result.get('interpretation', ''),
                            'status': get_status_from_interpretation(
                                metric_result.get('interpretation', '')
                            )
                        }

                        # Add disparity and ratio if available
                        if 'disparity' in metric_result:
                            metric_data['disparity'] = metric_result.get('disparity')
                        if 'ratio' in metric_result:
                            metric_data['ratio'] = metric_result.get('ratio')
                        if 'value' in metric_result:
                            metric_data['value'] = metric_result.get('value')

                        # Add metadata (testable_groups, excluded_groups)
                        if 'testable_groups' in metric_result:
                            metric_data['testable_groups'] = metric_result['testable_groups']
                        if 'excluded_groups' in metric_result:
                            metric_data['excluded_groups'] = metric_result['excluded_groups']
                        if 'min_representation_pct' in metric_result:
                            metric_data['min_representation_pct'] = metric_result['min_representation_pct']

                        # Categorize into main or complementary
                        if metric_name in POSTTRAIN_MAIN_METRICS:
                            attr_data['posttrain_main'].append(metric_data)
                        elif metric_name in POSTTRAIN_COMPLEMENTARY_METRICS:
                            attr_data['posttrain_complementary'].append(metric_data)
                        else:
                            # Unknown metric, add to complementary by default
                            attr_data['posttrain_complementary'].append(metric_data)

            transformed_attrs.append(attr_data)

        return transformed_attrs

    def _transform_issues(self, warnings: List[str], critical: List[str]) -> Dict[str, List]:
        """Transform warnings and critical issues."""
        return {
            'warnings': warnings,
            'critical': critical,
            'total': len(warnings) + len(critical)
        }

    def _transform_dataset_info(self, dataset_info: Dict) -> Dict[str, Any]:
        """
        Transform dataset information for report display.

        Args:
            dataset_info: Dictionary containing dataset information

        Returns:
            Transformed dataset information with formatted distributions
        """
        if not dataset_info:
            return {
                'total_samples': 0,
                'target_distribution': {},
                'protected_attributes_distribution': {}
            }

        return {
            'total_samples': dataset_info.get('total_samples', 0),
            'target_distribution': dataset_info.get('target_distribution', {}),
            'protected_attributes_distribution': dataset_info.get('protected_attributes_distribution', {})
        }

    def _transform_test_config(self, config: Dict, age_grouping_applied: Dict) -> Dict[str, Any]:
        """
        Transform test configuration for report display.

        Args:
            config: Test configuration dictionary
            age_grouping_applied: Age grouping information

        Returns:
            Formatted test configuration information
        """
        if not config:
            return {}

        transformed_config = {
            'config_name': config.get('name', 'custom'),
            'metrics_tested': config.get('metrics_tested', []),
            'pretrain_enabled': config.get('include_pretrain', False),
            'confusion_matrix_enabled': config.get('include_confusion_matrix', False),
            'threshold_analysis_enabled': config.get('include_threshold_analysis', False),
            'age_grouping_enabled': config.get('age_grouping', False),
            'age_grouping_strategy': config.get('age_grouping_strategy'),
            'age_grouping_details': []
        }

        # Add age grouping details if applied
        if age_grouping_applied:
            for attr, info in age_grouping_applied.items():
                transformed_config['age_grouping_details'].append({
                    'attribute': attr,
                    'strategy': info.get('strategy'),
                    'original_range': info.get('original_range'),
                    'groups': info.get('groups', [])
                })

        return transformed_config
