"""
Utility functions for fairness report transformers.

Contains helper functions for status extraction, metric categorization, and common operations.
"""

from typing import List


def get_status_from_interpretation(interpretation: str) -> str:
    """
    Extract status classification from interpretation string.

    Args:
        interpretation: Interpretation text from metric result

    Returns:
        Status string: 'critical', 'warning', 'success', or 'ok'
    """
    interp_upper = interpretation.upper()

    if ('✗' in interpretation or 'CRÍTICO' in interp_upper or 'CRITICAL' in interp_upper or
        'VERMELHO' in interp_upper or 'RED' in interp_upper or 'HIGH LEGAL RISK' in interp_upper):
        return 'critical'
    elif ('⚠' in interpretation or 'MODERADO' in interp_upper or 'MODERATE' in interp_upper or
          'AMARELO' in interp_upper or 'YELLOW' in interp_upper or 'WARNING' in interp_upper):
        return 'warning'
    elif ('✓' in interpretation or 'EXCELENTE' in interp_upper or 'EXCELLENT' in interp_upper or
          'BOM' in interp_upper or 'GOOD' in interp_upper or 'VERDE' in interp_upper or 'GREEN' in interp_upper):
        return 'success'
    else:
        return 'ok'


def get_assessment_text(score: float) -> str:
    """
    Get textual assessment based on overall fairness score.

    Args:
        score: Overall fairness score (0.0 to 1.0)

    Returns:
        Assessment text describing the fairness level
    """
    if score >= 0.9:
        return "EXCELLENT - Very high fairness"
    elif score >= 0.8:
        return "GOOD - Adequate fairness for production"
    elif score >= 0.6:
        return "MODERATE - Requires improvements before production"
    else:
        return "CRITICAL - Not recommended for production"


# Metric categorization constants
POSTTRAIN_MAIN_METRICS: List[str] = [
    'statistical_parity',
    'equal_opportunity',
    'equalized_odds',
    'disparate_impact',
    'false_negative_rate_difference'
]

POSTTRAIN_COMPLEMENTARY_METRICS: List[str] = [
    'conditional_acceptance',
    'conditional_rejection',
    'precision_difference',
    'accuracy_difference',
    'treatment_equality',
    'entropy_index'
]

PRETRAIN_METRICS: List[str] = [
    'class_balance',
    'concept_balance',
    'kl_divergence',
    'js_divergence'
]

# Display labels for metrics
METRIC_LABELS = {
    # Post-training main
    'statistical_parity': 'Statistical Parity',
    'equal_opportunity': 'Equal Opportunity',
    'equalized_odds': 'Equalized Odds',
    'disparate_impact': 'Disparate Impact',
    'false_negative_rate_difference': 'FNR Difference',

    # Post-training complementary
    'conditional_acceptance': 'Conditional Acceptance',
    'conditional_rejection': 'Conditional Rejection',
    'precision_difference': 'Precision Difference',
    'accuracy_difference': 'Accuracy Difference',
    'treatment_equality': 'Treatment Equality',
    'entropy_index': 'Entropy Index',

    # Pre-training
    'class_balance': 'Class Balance (BCL)',
    'concept_balance': 'Concept Balance (BCO)',
    'kl_divergence': 'KL Divergence',
    'js_divergence': 'JS Divergence'
}

# Short labels for compact displays (e.g., heatmaps)
METRIC_SHORT_LABELS = {
    'statistical_parity': 'Statistical<br>Parity',
    'equal_opportunity': 'Equal<br>Opportunity',
    'equalized_odds': 'Equalized<br>Odds',
    'disparate_impact': 'Disparate<br>Impact ⚖️',
    'false_negative_rate_difference': 'FNR<br>Difference',
    'conditional_acceptance': 'Conditional<br>Acceptance',
    'conditional_rejection': 'Conditional<br>Rejection',
    'precision_difference': 'Precision<br>Difference',
    'accuracy_difference': 'Accuracy<br>Difference',
    'treatment_equality': 'Treatment<br>Equality',
    'entropy_index': 'Entropy<br>Index'
}


def format_metric_name(metric_name: str) -> str:
    """
    Format metric name for display.

    Args:
        metric_name: Raw metric name (e.g., 'statistical_parity')

    Returns:
        Formatted display name
    """
    return METRIC_LABELS.get(metric_name, metric_name.replace('_', ' ').title())


def format_attribute_name(attr_name: str) -> str:
    """
    Format protected attribute name for display.

    Args:
        attr_name: Raw attribute name (e.g., 'gender_protected')

    Returns:
        Formatted display name
    """
    return attr_name.replace('_', ' ').title()
