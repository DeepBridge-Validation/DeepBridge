"""
Pytest fixtures for fairness chart tests.

Provides common test data and mock objects.
"""

import json

import pytest


@pytest.fixture
def sample_posttrain_metrics():
    """Sample post-training metrics data."""
    return {
        'gender': {
            'statistical_parity': {
                'disparity': 0.05,
                'interpretation': '✓ GOOD - Low disparity',
                'testable_groups': ['Male', 'Female'],
                'excluded_groups': [],
            },
            'disparate_impact': {
                'ratio': 0.85,
                'passes_80_rule': True,
                'interpretation': '✓ GOOD - Passes 80% rule',
            },
            'equal_opportunity': {
                'disparity': 0.03,
                'interpretation': '✓ GOOD - Equal opportunity maintained',
            },
            'equalized_odds': {
                'disparity': 0.04,
                'interpretation': '✓ GOOD - Balanced error rates',
            },
            'false_negative_rate_difference': {
                'disparity': 0.02,
                'interpretation': '✓ GOOD - Low FNR difference',
            },
            'precision_difference': {
                'disparity': 0.06,
                'interpretation': '✓ GOOD - Similar precision',
            },
            'accuracy_difference': {
                'disparity': 0.01,
                'interpretation': '✓ GOOD - Similar accuracy',
            },
        }
    }


@pytest.fixture
def sample_pretrain_metrics():
    """Sample pre-training metrics data."""
    return {
        'gender': {
            'class_balance': {
                'value': 0.02,
                'interpretation': '✓ GOOD - Balanced classes',
                'all_groups': {'Male': 0.51, 'Female': 0.49},
            },
            'concept_balance': {
                'value': 0.03,
                'group_a': 'Male',
                'group_b': 'Female',
                'group_a_positive_rate': 0.52,
                'group_b_positive_rate': 0.49,
                'interpretation': '✓ GOOD - Similar concept distribution',
            },
            'kl_divergence': {
                'value': 0.015,
                'interpretation': '✓ GOOD - Low KL divergence',
            },
            'js_divergence': {
                'value': 0.008,
                'interpretation': '✓ GOOD - Low JS divergence',
            },
        }
    }


@pytest.fixture
def sample_confusion_matrix():
    """Sample confusion matrix data."""
    return {
        'gender': {
            'Male': {'TP': 45, 'TN': 40, 'FP': 5, 'FN': 10},
            'Female': {'TP': 42, 'TN': 38, 'FP': 7, 'FN': 13},
        }
    }


@pytest.fixture
def sample_protected_attrs():
    """Sample protected attributes list."""
    return ['gender']


@pytest.fixture
def sample_protected_attrs_distribution():
    """Sample protected attributes distribution data."""
    return {
        'gender': {
            'distribution': {
                'Male': {'count': 500, 'percentage': 52.0},
                'Female': {'count': 460, 'percentage': 48.0},
            },
            'total_samples': 960,
        }
    }


@pytest.fixture
def sample_target_distribution():
    """Sample target distribution data."""
    return {
        '0': {'count': 400, 'percentage': 41.7},
        '1': {'count': 560, 'percentage': 58.3},
    }


@pytest.fixture
def sample_threshold_analysis():
    """Sample threshold analysis data."""
    return {
        'threshold_curve': [
            {
                'threshold': 0.1,
                'disparate_impact_ratio': 0.65,
                'f1_score': 0.75,
            },
            {
                'threshold': 0.3,
                'disparate_impact_ratio': 0.75,
                'f1_score': 0.82,
            },
            {
                'threshold': 0.5,
                'disparate_impact_ratio': 0.85,
                'f1_score': 0.88,
            },
            {
                'threshold': 0.7,
                'disparate_impact_ratio': 0.82,
                'f1_score': 0.85,
            },
            {
                'threshold': 0.9,
                'disparate_impact_ratio': 0.78,
                'f1_score': 0.78,
            },
        ],
        'optimal_threshold': 0.5,
    }


def validate_plotly_json(json_str):
    """
    Validate that a string is valid Plotly JSON.

    Args:
        json_str: JSON string to validate

    Returns:
        bool: True if valid, False otherwise
    """
    if not json_str or json_str == '{}':
        return False

    try:
        data = json.loads(json_str)
        # Check for basic Plotly structure
        return 'data' in data or 'layout' in data
    except (json.JSONDecodeError, TypeError):
        return False


@pytest.fixture
def plotly_validator():
    """Fixture that returns the Plotly JSON validator function."""
    return validate_plotly_json
