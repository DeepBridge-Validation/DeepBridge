"""
Visualization tools for validation results.

This module provides specialized visualization functions for
different types of validation results.
"""

# Import visualization modules
from .robustness import perturbation_plots, adversarial_plots, resilience_plots
from .uncertainty import calibration_plots, interval_plots
from .hyperparameters import importance_plots, tuning_comparison_plots

# Convenience imports for commonly used functions
from .robustness.perturbation_plots import (
    plot_feature_perturbation_impact,
    plot_perturbation_heatmap
)
from .robustness.adversarial_plots import (
    plot_adversarial_examples,
    plot_attack_success_rates
)
from .robustness.resilience_plots import (
    plot_resilience_scores,
    plot_distribution_shift_impact
)
from .uncertainty.calibration_plots import (
    plot_calibration_curve,
    plot_reliability_diagram
)
from .uncertainty.interval_plots import (
    plot_prediction_intervals,
    plot_confidence_distribution
)
from .hyperparameters.importance_plots import (
    plot_hyperparameter_importance,
    plot_param_influence
)
from .hyperparameters.tuning_comparison_plots import (
    plot_optimization_history,
    plot_tuning_comparison
)

__all__ = [
    # Submodules
    'perturbation_plots',
    'adversarial_plots',
    'resilience_plots',
    'calibration_plots',
    'interval_plots',
    'importance_plots',
    'tuning_comparison_plots',
    
    # Robustness plots
    'plot_feature_perturbation_impact',
    'plot_perturbation_heatmap',
    'plot_adversarial_examples',
    'plot_attack_success_rates',
    'plot_resilience_scores',
    'plot_distribution_shift_impact',
    
    # Uncertainty plots
    'plot_calibration_curve',
    'plot_reliability_diagram',
    'plot_prediction_intervals',
    'plot_confidence_distribution',
    
    # Hyperparameters plots
    'plot_hyperparameter_importance',
    'plot_param_influence',
    'plot_optimization_history',
    'plot_tuning_comparison'
]