"""Tests for robustness data structures and transformer."""

import pytest
from datetime import datetime
from deepbridge.core.experiment.report.data.robustness import (
    RobustnessReportData,
    RobustnessDataTransformer,
    RobustnessMetrics,
    PerturbationResult,
    FeatureImportance
)


class TestPerturbationResult:
    """Test PerturbationResult dataclass."""

    def test_creation(self):
        """Test creating perturbation result."""
        result = PerturbationResult(
            level=0.1,
            mean_score=0.85,
            worst_score=0.80,
            impact=0.05,
            num_samples=100
        )

        assert result.level == 0.1
        assert result.mean_score == 0.85
        assert result.worst_score == 0.80
        assert result.impact == 0.05
        assert result.num_samples == 100

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = PerturbationResult(
            level=0.2,
            mean_score=0.80
        )

        data = result.to_dict()

        assert isinstance(data, dict)
        assert data['level'] == 0.2
        assert data['mean_score'] == 0.80


class TestFeatureImportance:
    """Test FeatureImportance dataclass."""

    def test_creation(self):
        """Test creating feature importance."""
        feature = FeatureImportance(
            feature_name="feature_1",
            importance=0.25,
            rank=1
        )

        assert feature.feature_name == "feature_1"
        assert feature.importance == 0.25
        assert feature.rank == 1

    def test_to_dict(self):
        """Test conversion to dictionary."""
        feature = FeatureImportance(
            feature_name="feature_2",
            importance=0.15,
            rank=2
        )

        data = feature.to_dict()

        assert isinstance(data, dict)
        assert data['feature_name'] == "feature_2"
        assert data['importance'] == 0.15


class TestRobustnessMetrics:
    """Test RobustnessMetrics dataclass."""

    def test_creation(self):
        """Test creating robustness metrics."""
        metrics = RobustnessMetrics(
            robustness_score=0.85,
            base_score=0.90,
            avg_raw_impact=0.05,
            avg_quantile_impact=0.04
        )

        assert metrics.robustness_score == 0.85
        assert metrics.base_score == 0.90
        assert metrics.avg_raw_impact == 0.05
        assert metrics.avg_quantile_impact == 0.04

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = RobustnessMetrics(
            robustness_score=0.80,
            base_score=0.85
        )

        data = metrics.to_dict()

        assert isinstance(data, dict)
        assert 'robustness_score' in data
        assert 'base_score' in data


class TestRobustnessReportData:
    """Test RobustnessReportData dataclass."""

    def test_creation(self):
        """Test creating robustness report data."""
        metrics = RobustnessMetrics(
            robustness_score=0.85,
            base_score=0.90
        )

        data = RobustnessReportData(
            generated_at=datetime.now(),
            report_type="robustness",
            model_name="Test Model",
            model_type="RandomForest",
            metrics=metrics
        )

        assert data.model_name == "Test Model"
        assert data.model_type == "RandomForest"
        assert data.metrics.robustness_score == 0.85
        assert data.report_type == "robustness"

    def test_validate_valid_data(self):
        """Test validation with valid data."""
        metrics = RobustnessMetrics(
            robustness_score=0.75,
            base_score=0.85
        )

        data = RobustnessReportData(
            generated_at=datetime.now(),
            report_type="robustness",
            model_name="Test Model",
            model_type="XGBoost",
            metrics=metrics
        )

        assert data.validate() is True

    def test_validate_missing_model_name(self):
        """Test validation with missing model name."""
        metrics = RobustnessMetrics(
            robustness_score=0.75,
            base_score=0.85
        )

        data = RobustnessReportData(
            generated_at=datetime.now(),
            report_type="robustness",
            model_name="",
            model_type="XGBoost",
            metrics=metrics
        )

        with pytest.raises(ValueError, match="model_name is required"):
            data.validate()

    def test_validate_invalid_robustness_score(self):
        """Test validation with invalid robustness score."""
        metrics = RobustnessMetrics(
            robustness_score=1.5,  # Invalid: > 1
            base_score=0.85
        )

        data = RobustnessReportData(
            generated_at=datetime.now(),
            report_type="robustness",
            model_name="Test Model",
            model_type="XGBoost",
            metrics=metrics
        )

        with pytest.raises(ValueError, match="robustness_score must be between 0 and 1"):
            data.validate()

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = RobustnessMetrics(
            robustness_score=0.85,
            base_score=0.90
        )

        data = RobustnessReportData(
            generated_at=datetime.now(),
            report_type="robustness",
            model_name="Test Model",
            model_type="RandomForest",
            metrics=metrics
        )

        result = data.to_dict()

        assert isinstance(result, dict)
        assert result['model_name'] == "Test Model"
        assert result['model_type'] == "RandomForest"
        assert 'metrics' in result
        assert result['robustness_score'] == 0.85  # Backward compatibility


class TestRobustnessDataTransformer:
    """Test RobustnessDataTransformer."""

    @pytest.fixture
    def sample_raw_data(self):
        """Sample raw robustness data."""
        return {
            'model_name': 'Test Model',
            'model_type': 'RandomForest',
            'base_score': 0.90,
            'robustness_score': 0.85,
            'avg_raw_impact': 0.05,
            'avg_quantile_impact': 0.04,
            'metric': 'accuracy',
            'raw': {
                'by_level': {
                    '0.1': {
                        'overall_result': {
                            'all_features': {
                                'mean_score': 0.88,
                                'worst_score': 0.85,
                                'impact': 0.02
                            }
                        }
                    },
                    '0.2': {
                        'overall_result': {
                            'all_features': {
                                'mean_score': 0.85,
                                'worst_score': 0.80,
                                'impact': 0.05
                            }
                        }
                    }
                }
            },
            'quantile': {
                'by_level': {
                    '0.1': {
                        'overall_result': {
                            'all_features': {
                                'mean_score': 0.89,
                                'worst_score': 0.87
                            }
                        }
                    }
                }
            },
            'feature_importance': {
                'feature_1': 0.25,
                'feature_2': 0.20,
                'feature_3': 0.15
            }
        }

    def test_transform_basic(self, sample_raw_data):
        """Test basic transformation."""
        transformer = RobustnessDataTransformer()
        result = transformer.transform(sample_raw_data)

        assert isinstance(result, RobustnessReportData)
        assert result.model_name == 'Test Model'
        assert result.model_type == 'RandomForest'
        assert result.metrics.robustness_score == 0.85

    def test_transform_perturbation_results(self, sample_raw_data):
        """Test perturbation results extraction."""
        transformer = RobustnessDataTransformer()
        result = transformer.transform(sample_raw_data)

        assert len(result.perturbation_results_raw) == 2
        assert result.perturbation_results_raw[0].level == 0.1
        assert result.perturbation_results_raw[0].mean_score == 0.88

        assert len(result.perturbation_results_quantile) == 1
        assert result.perturbation_results_quantile[0].level == 0.1

    def test_transform_feature_importance(self, sample_raw_data):
        """Test feature importance extraction."""
        transformer = RobustnessDataTransformer()
        result = transformer.transform(sample_raw_data)

        assert len(result.feature_importance) == 3
        assert result.feature_importance[0].feature_name == 'feature_1'
        assert result.feature_importance[0].importance == 0.25
        assert result.feature_importance[0].rank == 1

    def test_validate_raw_data_invalid(self):
        """Test validation of invalid raw data."""
        transformer = RobustnessDataTransformer()

        with pytest.raises(ValueError, match="Expected dict"):
            transformer.transform("not a dict")

        with pytest.raises(ValueError, match="cannot be empty"):
            transformer.transform({})

    def test_extract_model_type_from_nested(self):
        """Test extracting model type from nested structure."""
        transformer = RobustnessDataTransformer()

        raw_data = {
            'model_name': 'Test',
            'primary_model': {
                'model_type': 'XGBoost'
            },
            'base_score': 0.85,
            'robustness_score': 0.80
        }

        result = transformer.transform(raw_data)
        assert result.model_type == 'XGBoost'

    def test_compute_robustness_score_from_impact(self):
        """Test computing robustness score from impact."""
        transformer = RobustnessDataTransformer()

        raw_data = {
            'model_name': 'Test',
            'model_type': 'SVM',
            'base_score': 0.90,
            'avg_overall_impact': 0.15  # No robustness_score provided
        }

        result = transformer.transform(raw_data)
        assert result.metrics.robustness_score == pytest.approx(0.85)  # 1 - 0.15
