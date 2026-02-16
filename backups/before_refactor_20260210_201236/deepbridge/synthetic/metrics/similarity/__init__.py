"""
Similarity analysis and quality improvement for synthetic data.

This module provides functions for analyzing and improving the quality of synthetic data
by measuring similarity to original data and performing various enhancements.

Main components:
- Core functions for calculating similarity between datasets
- Enhancement techniques for improving synthetic data quality
- Privacy assessment tools
- Utility functions for data analysis and visualization

Usage example:
    from synthetic.metrics.similarity import calculate_similarity, filter_by_similarity
    from synthetic.metrics.similarity import enhance_synthetic_data_quality

    # Calculate similarity between original and synthetic data
    similarity_scores = calculate_similarity(
        original_data=original_df,
        synthetic_data=synthetic_df
    )

    # Filter out samples that are too similar to original data
    filtered_data = filter_by_similarity(
        original_data=original_df,
        synthetic_data=synthetic_df,
        threshold=0.9
    )

    # Enhance synthetic data quality
    enhanced_data = enhance_synthetic_data_quality(
        original_data=original_df,
        synthetic_data=synthetic_df,
        target_column='target'
    )
"""

# Data enhancement functionality
from .data_enhancement import detect_duplicates, enhance_synthetic_data_quality

# Core similarity functions
from .similarity_core import calculate_similarity, filter_by_similarity

# Utils for similarity calculations
from .similarity_utils import (
    calculate_distribution_divergence,
    calculate_diversity,
    evaluate_pairwise_correlations,
)

# Import privacy assessment if available - with fallback
try:
    from .privacy_assessment import (
        assess_k_anonymity,
        assess_l_diversity,
        calculate_privacy_risk,
    )
except ImportError:
    # Define placeholders if privacy_assessment module is not available
    def calculate_privacy_risk(*args, **kwargs):
        raise NotImplementedError('Privacy risk calculation not available')

    def assess_k_anonymity(*args, **kwargs):
        raise NotImplementedError('K-anonymity assessment not available')

    def assess_l_diversity(*args, **kwargs):
        raise NotImplementedError('L-diversity assessment not available')


# Import visualization utilities if available - with fallback
try:
    from .visualization_utils import (
        plot_attribute_distributions,
        plot_correlation_comparison,
        plot_distribution_comparison,
        plot_privacy_risk,
        visualize_data_comparison,
    )
except ImportError:
    # Define placeholders if visualization module is not available
    def visualize_data_comparison(*args, **kwargs):
        raise NotImplementedError(
            'Data comparison visualization not available'
        )

    def plot_distribution_comparison(*args, **kwargs):
        raise NotImplementedError('Distribution comparison plot not available')

    def plot_correlation_comparison(*args, **kwargs):
        raise NotImplementedError('Correlation comparison plot not available')

    def plot_privacy_risk(*args, **kwargs):
        raise NotImplementedError('Privacy risk plot not available')

    def plot_attribute_distributions(*args, **kwargs):
        raise NotImplementedError('Attribute distribution plot not available')


# Export all functions
__all__ = [
    # Core functions
    'calculate_similarity',
    'filter_by_similarity',
    # Enhancement
    'enhance_synthetic_data_quality',
    'detect_duplicates',
    # Diversity
    'calculate_diversity',
    # Privacy assessment
    'calculate_privacy_risk',
    'assess_k_anonymity',
    'assess_l_diversity',
    # Visualization
    'visualize_data_comparison',
    'plot_distribution_comparison',
    'plot_correlation_comparison',
    'plot_privacy_risk',
    'plot_attribute_distributions',
    # Utils
    'calculate_distribution_divergence',
    'evaluate_pairwise_correlations',
]
