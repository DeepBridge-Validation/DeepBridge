"""
Fairness testing module for DeepBridge.

This module provides comprehensive fairness testing capabilities for ML models,
following industry standards (EEOC, ECOA) and best practices from libraries
like AI Fairness 360, Fairlearn, and Aequitas.

Main components:
- FairnessMetrics: 15 fairness metrics (4 pre-training + 11 post-training)
- FairnessSuite: Complete fairness testing suite integrated with DeepBridge
- FairnessVisualizer: Visualization tools for fairness analysis

Available Metrics:

PRE-TRAINING (Model-Independent):
    1. class_balance - Sample size balance between groups
    2. concept_balance - Positive class rate balance
    3. kl_divergence - Distribution similarity (KL)
    4. js_divergence - Distribution similarity (JS)

POST-TRAINING (Model-Dependent):
    5. statistical_parity - Equal positive prediction rate
    6. equal_opportunity - Equal TPR across groups
    7. equalized_odds - Equal TPR and FPR
    8. disparate_impact - EEOC 80% rule compliance
    9. false_negative_rate_difference - Equal miss rate
    10. conditional_acceptance - Equal precision/PPV
    11. conditional_rejection - Equal NPV
    12. precision_difference - Equal precision
    13. accuracy_difference - Equal overall accuracy
    14. treatment_equality - Equal FN/FP ratio
    15. entropy_index - Individual fairness

Available Visualizations:
    1. plot_distribution_by_group - Target distribution across protected groups
    2. plot_metrics_comparison - Bar chart comparing all fairness metrics
    3. plot_threshold_impact - Line charts showing fairness vs threshold
    4. plot_confusion_matrices - Side-by-side confusion matrices by group
    5. plot_fairness_radar - Radar chart with all fairness dimensions
    6. plot_group_comparison - Detailed comparison of model performance by group

Usage:
    from deepbridge.validation.fairness import FairnessMetrics, FairnessVisualizer
    from deepbridge.validation.wrappers import FairnessSuite

    # Pre-training metrics (no model needed)
    bcl = FairnessMetrics.class_balance(y_true, sensitive_feature)
    bco = FairnessMetrics.concept_balance(y_true, sensitive_feature)

    # Post-training metrics (model required)
    sp = FairnessMetrics.statistical_parity(y_pred, sensitive_feature)
    eo = FairnessMetrics.equal_opportunity(y_true, y_pred, sensitive_feature)

    # Run complete fairness analysis
    fairness = FairnessSuite(dataset, protected_attributes=['gender', 'race'])
    results = fairness.config('full').run()

    # Visualize results
    FairnessVisualizer.plot_metrics_comparison(
        results['posttrain_metrics'],
        protected_attrs=['gender', 'race'],
        output_path='fairness_metrics.png'
    )
"""

from deepbridge.validation.fairness.metrics import FairnessMetrics
from deepbridge.validation.fairness.visualizations import FairnessVisualizer

__all__ = ['FairnessMetrics', 'FairnessVisualizer']
