"""
Visualization tools for robustness validation results.

This module provides specialized visualization functions for
robustness validation results, including perturbation impact,
adversarial attacks, and distribution shift resilience.
"""

from .perturbation_plots import (
    plot_feature_perturbation_impact,
    plot_perturbation_heatmap,
    plot_aggregate_perturbation_impact,
    plot_relative_performance_change
)

from .adversarial_plots import (
    plot_adversarial_examples,
    plot_attack_success_rates,
    plot_perturbation_magnitude,
    plot_adversarial_robustness_comparison
)

from .resilience_plots import (
    plot_resilience_scores,
    plot_distribution_shift_impact,
    plot_feature_robustness_ranking,
    plot_robustness_score_comparison
)

__all__ = [
    # Perturbation plots
    'plot_feature_perturbation_impact',
    'plot_perturbation_heatmap',
    'plot_aggregate_perturbation_impact',
    'plot_relative_performance_change',
    
    # Adversarial plots
    'plot_adversarial_examples',
    'plot_attack_success_rates',
    'plot_perturbation_magnitude',
    'plot_adversarial_robustness_comparison',
    
    # Resilience plots
    'plot_resilience_scores',
    'plot_distribution_shift_impact',
    'plot_feature_robustness_ranking',
    'plot_robustness_score_comparison'
]