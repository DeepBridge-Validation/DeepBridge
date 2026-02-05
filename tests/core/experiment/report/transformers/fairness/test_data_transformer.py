"""
Tests for FairnessDataTransformer.
"""

import pytest

from deepbridge.core.experiment.report.transformers.fairness.data_transformer import (
    FairnessDataTransformer,
)


@pytest.fixture
def full_fairness_results(
    sample_pretrain_metrics,
    sample_posttrain_metrics,
    sample_confusion_matrix,
    sample_protected_attrs,
    sample_protected_attrs_distribution,
    sample_target_distribution,
):
    """Complete fairness results for testing."""
    return {
        'protected_attributes': sample_protected_attrs,
        'pretrain_metrics': sample_pretrain_metrics,
        'posttrain_metrics': sample_posttrain_metrics,
        'confusion_matrix': sample_confusion_matrix,
        'warnings': ['Warning 1', 'Warning 2'],
        'critical_issues': ['Critical issue 1'],
        'overall_fairness_score': 0.85,
        'age_grouping_applied': {},
        'dataset_info': {
            'total_samples': 960,
            'target_distribution': sample_target_distribution,
            'protected_attributes_distribution': sample_protected_attrs_distribution,
        },
        'config': {
            'name': 'medium',
            'metrics_tested': ['statistical_parity', 'disparate_impact'],
            'include_pretrain': True,
            'include_confusion_matrix': True,
            'include_threshold_analysis': False,
            'age_grouping': False,
        },
        'threshold_analysis': None,
    }


class TestFairnessDataTransformer:
    """Tests for FairnessDataTransformer."""

    def test_transform_with_complete_data(self, full_fairness_results):
        """Test transformation with complete fairness results."""
        transformer = FairnessDataTransformer()
        result = transformer.transform(
            full_fairness_results, model_name='Test Model'
        )

        # Check top-level structure
        assert 'model_name' in result
        assert result['model_name'] == 'Test Model'
        assert 'summary' in result
        assert 'protected_attributes' in result
        assert 'issues' in result
        assert 'dataset_info' in result
        assert 'test_config' in result
        assert 'charts' in result
        assert 'metadata' in result

    def test_summary_creation(self, full_fairness_results):
        """Test that summary is created correctly."""
        transformer = FairnessDataTransformer()
        result = transformer.transform(full_fairness_results)

        summary = result['summary']
        assert summary['overall_fairness_score'] == 0.85
        assert summary['total_warnings'] == 2
        assert summary['total_critical'] == 1
        assert summary['total_attributes'] == 1
        # Config is now the full config dict, not just the name
        assert (
            isinstance(summary['config'], dict)
            or summary['config'] == 'medium'
        )
        assert 'assessment' in summary

    def test_protected_attributes_transformation(self, full_fairness_results):
        """Test that protected attributes are transformed correctly."""
        transformer = FairnessDataTransformer()
        result = transformer.transform(full_fairness_results)

        attrs = result['protected_attributes']
        assert len(attrs) == 1
        assert attrs[0]['name'] == 'gender'
        assert 'pretrain_metrics' in attrs[0]
        assert 'posttrain_main' in attrs[0]
        assert 'posttrain_complementary' in attrs[0]

    def test_posttrain_metrics_categorization(self, full_fairness_results):
        """Test that post-training metrics are categorized correctly."""
        transformer = FairnessDataTransformer()
        result = transformer.transform(full_fairness_results)

        attr = result['protected_attributes'][0]

        # Check main metrics
        main_metric_names = [m['name'] for m in attr['posttrain_main']]
        assert 'Statistical Parity' in main_metric_names
        assert 'Disparate Impact' in main_metric_names
        assert 'Equal Opportunity' in main_metric_names

        # Check complementary metrics
        comp_metric_names = [
            m['name'] for m in attr['posttrain_complementary']
        ]
        assert 'Precision Difference' in comp_metric_names
        assert 'Accuracy Difference' in comp_metric_names

    def test_issues_transformation(self, full_fairness_results):
        """Test that issues are transformed correctly."""
        transformer = FairnessDataTransformer()
        result = transformer.transform(full_fairness_results)

        issues = result['issues']
        assert len(issues['warnings']) == 2
        assert len(issues['critical']) == 1
        assert issues['total'] == 3

    def test_dataset_info_transformation(self, full_fairness_results):
        """Test that dataset info is transformed correctly."""
        transformer = FairnessDataTransformer()
        result = transformer.transform(full_fairness_results)

        dataset_info = result['dataset_info']
        assert dataset_info['total_samples'] == 960
        assert 'target_distribution' in dataset_info
        assert 'protected_attributes_distribution' in dataset_info

    def test_test_config_transformation(self, full_fairness_results):
        """Test that test config is transformed correctly."""
        transformer = FairnessDataTransformer()
        result = transformer.transform(full_fairness_results)

        test_config = result['test_config']
        assert test_config['config_name'] == 'medium'
        assert test_config['pretrain_enabled'] is True
        assert test_config['confusion_matrix_enabled'] is True
        assert test_config['threshold_analysis_enabled'] is False

    def test_charts_creation(self, full_fairness_results):
        """Test that charts are created."""
        transformer = FairnessDataTransformer()
        result = transformer.transform(full_fairness_results)

        charts = result['charts']
        assert isinstance(charts, dict)
        # Should have multiple charts
        assert len(charts) > 0

    def test_metadata_creation(self, full_fairness_results):
        """Test that metadata is created correctly."""
        transformer = FairnessDataTransformer()
        result = transformer.transform(full_fairness_results)

        metadata = result['metadata']
        assert metadata['total_attributes'] == 1
        assert metadata['has_threshold_analysis'] is False
        assert metadata['has_confusion_matrix'] is True
        assert metadata['age_grouping_enabled'] is False

    def test_transform_with_minimal_data(self):
        """Test transformation with minimal fairness results."""
        transformer = FairnessDataTransformer()

        minimal_results = {
            'protected_attributes': ['gender'],
            'pretrain_metrics': {},
            'posttrain_metrics': {},
            'warnings': [],
            'critical_issues': [],
            'overall_fairness_score': 0.0,
        }

        result = transformer.transform(minimal_results)

        # Should still return valid structure
        assert 'model_name' in result
        assert 'summary' in result
        assert 'protected_attributes' in result

    def test_handles_missing_optional_fields(self):
        """Test that transformer handles missing optional fields gracefully."""
        transformer = FairnessDataTransformer()

        results = {
            'protected_attributes': ['gender'],
            'pretrain_metrics': {},
            'posttrain_metrics': {},
            'warnings': [],
            'critical_issues': [],
            'overall_fairness_score': 0.5
            # Missing: confusion_matrix, threshold_analysis, dataset_info, etc.
        }

        result = transformer.transform(results)

        # Should not raise exceptions
        assert result is not None
        assert 'charts' in result
