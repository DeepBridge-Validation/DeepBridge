"""
Robustness testing framework.

This module provides tools for analyzing model robustness against
various types of perturbations, adversarial attacks, and outliers.
"""

from .robustness_tester import RobustnessTester, run_robustness_test_suite
from .outlier_tester import OutlierRobustnessTester, detect_outliers
from .resilience_analyzer import ResilienceAnalyzer, compute_resilience_score

# Import from subdirectories
from .adversarial import (
    FGSMAttacker, generate_fgsm_attack,
    PGDAttacker, generate_pgd_attack,
    BlackboxAttacker, generate_blackbox_attack
)
from .perturbation import (
    apply_gaussian_noise,
    apply_salt_pepper_noise,
    apply_feature_perturbation,
    get_perturbation_function
)

__all__ = [
    'RobustnessTester',
    'run_robustness_test_suite',
    'OutlierRobustnessTester',
    'detect_outliers',
    'ResilienceAnalyzer',
    'compute_resilience_score',
    # Adversarial classes and functions
    'FGSMAttacker', 
    'generate_fgsm_attack',
    'PGDAttacker', 
    'generate_pgd_attack',
    'BlackboxAttacker', 
    'generate_blackbox_attack',
    # Perturbation functions
    'apply_gaussian_noise',
    'apply_salt_pepper_noise',
    'apply_feature_perturbation',
    'get_perturbation_function'
]