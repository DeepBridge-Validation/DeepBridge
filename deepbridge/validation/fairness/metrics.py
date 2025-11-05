"""
Fairness metrics for machine learning models.

This module implements industry-standard fairness metrics following best practices
from AI Fairness 360 (IBM), Fairlearn (Microsoft), and regulatory frameworks
(EEOC, ECOA, Fair Lending Act).

Key Concepts:
- Protected Attributes: Features like race, gender, age that should not influence decisions
- Privileged/Unprivileged Groups: Groups with historically different treatment
- Disparate Impact: Adverse impact on protected groups (legal threshold: 80% rule)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union, List


class FairnessMetrics:
    """
    Comprehensive fairness metrics for ML model evaluation.

    PRE-TRAINING METRICS (Model-Independent):
    1. Class Balance (BCL) - Sample size balance
    2. Concept Balance (BCO) - Positive class rate balance
    3. KL Divergence - Distribution similarity
    4. JS Divergence - Symmetric distribution similarity

    POST-TRAINING METRICS (Model-Dependent):
    5. Statistical Parity - Equal positive prediction rate
    6. Equal Opportunity - Equal TPR across groups
    7. Equalized Odds - Equal TPR and FPR
    8. Disparate Impact - EEOC 80% rule compliance
    9. False Negative Rate Difference - Equal miss rate
    10. Conditional Acceptance - Equal precision/PPV
    11. Conditional Rejection - Equal NPV
    12. Precision Difference - Equal precision
    13. Accuracy Difference - Equal overall accuracy
    14. Treatment Equality - Equal FN/FP ratio
    15. Entropy Index - Individual fairness

    All metrics return structured dictionaries with:
    - Metric values
    - Group-level breakdowns
    - Pass/fail indicators (✓ Green / ⚠ Yellow / ✗ Red)
    - Human-readable interpretations

    REPRESENTATIVENESS THRESHOLD:
    - MIN_REPRESENTATION_PCT: Minimum percentage of samples for a group to be
      included in fairness analysis (default: 2.0%)
    - Based on EEOC Uniform Guidelines Question 21 "Flip-Flop Rule"
    - Groups below this threshold are excluded to avoid statistical instability
    """

    # EEOC Compliance: Minimum representativeness threshold
    MIN_REPRESENTATION_PCT = 2.0

    @staticmethod
    def statistical_parity(y_pred: Union[np.ndarray, pd.Series],
                          sensitive_feature: Union[np.ndarray, pd.Series],
                          min_representation_pct: float = None) -> Dict[str, Any]:
        """
        Statistical Parity (Demographic Parity)

        Measures if the positive prediction rate is equal across groups.

        Formula:
            P(Y_hat=1 | A=a) = P(Y_hat=1 | A=b) for all groups a, b

        Parameters:
        -----------
        y_pred : array-like
            Binary predictions (0 or 1)
        sensitive_feature : array-like
            Protected attribute values
        min_representation_pct : float, optional
            Minimum percentage of samples for a group to be included in analysis.
            Groups below this threshold are excluded per EEOC Question 21.
            If None, uses FairnessMetrics.MIN_REPRESENTATION_PCT (default: 2.0%)

        Returns:
        --------
        dict : {
            'metric_name': str,
            'all_groups': Dict[str, Dict],  # Info about ALL groups (testable + excluded)
            'testable_groups': Dict[str, Dict],  # Groups meeting threshold
            'excluded_groups': Dict[str, Dict],  # Groups below threshold
            'disparity': float,  # Max difference between testable groups
            'ratio': float,  # min_rate / max_rate for testable groups
            'passes_80_rule': bool,  # ratio >= 0.8 (EEOC standard)
            'min_representation_pct': float,  # Threshold used
            'interpretation': str
        }

        Example:
        --------
        >>> y_pred = np.array([1, 0, 1, 1, 0, 1])
        >>> gender = np.array(['M', 'M', 'F', 'F', 'M', 'F'])
        >>> result = FairnessMetrics.statistical_parity(y_pred, gender)
        >>> print(result['passes_80_rule'])
        True
        """
        # Use default threshold if not specified
        if min_representation_pct is None:
            min_representation_pct = FairnessMetrics.MIN_REPRESENTATION_PCT

        # Convert to numpy arrays
        y_pred = np.asarray(y_pred)
        sensitive_feature = np.asarray(sensitive_feature)

        # Get unique groups
        groups = np.unique(sensitive_feature)
        total_samples = len(sensitive_feature)
        all_groups = {}

        # Calculate info for each group
        for group in groups:
            mask = sensitive_feature == group
            group_size = np.sum(mask)
            group_pct = (group_size / total_samples) * 100
            meets_threshold = group_pct >= min_representation_pct

            if group_size > 0:
                positive_rate = np.mean(y_pred[mask] == 1)
            else:
                positive_rate = 0.0

            all_groups[str(group)] = {
                'rate': float(positive_rate),
                'count': int(group_size),
                'percentage': float(group_pct),
                'meets_threshold': meets_threshold
            }

        # Separate testable vs excluded groups
        testable_groups = {g: info for g, info in all_groups.items() if info['meets_threshold']}
        excluded_groups = {g: info for g, info in all_groups.items() if not info['meets_threshold']}

        # Calculate metrics ONLY for testable groups
        if len(testable_groups) < 2:
            return {
                'metric_name': 'statistical_parity',
                'all_groups': all_groups,
                'testable_groups': testable_groups,
                'excluded_groups': excluded_groups,
                'disparity': None,
                'ratio': None,
                'passes_80_rule': None,
                'min_representation_pct': min_representation_pct,
                'interpretation': f"INSUFFICIENT: Less than 2 groups with ≥{min_representation_pct}% representation for analysis"
            }

        testable_rates = [info['rate'] for info in testable_groups.values()]
        max_rate = max(testable_rates)
        min_rate = min(testable_rates)

        disparity = max_rate - min_rate
        ratio = min_rate / max_rate if max_rate > 0 else 0.0
        passes_80_rule = ratio >= 0.8

        return {
            'metric_name': 'statistical_parity',
            'all_groups': all_groups,
            'testable_groups': testable_groups,
            'excluded_groups': excluded_groups,
            'disparity': float(disparity),
            'ratio': float(ratio),
            'passes_80_rule': passes_80_rule,
            'min_representation_pct': min_representation_pct,
            'interpretation': _interpret_statistical_parity(disparity, ratio)
        }

    @staticmethod
    def equal_opportunity(y_true: Union[np.ndarray, pd.Series],
                         y_pred: Union[np.ndarray, pd.Series],
                         sensitive_feature: Union[np.ndarray, pd.Series],
                         min_representation_pct: float = None) -> Dict[str, Any]:
        """
        Equal Opportunity

        Measures if the True Positive Rate (TPR/Recall) is equal across groups.
        Focuses on ensuring the model identifies positive outcomes equally.

        Formula:
            P(Y_hat=1 | Y=1, A=a) = P(Y_hat=1 | Y=1, A=b) for all groups a, b

        Parameters:
        -----------
        y_true : array-like
            True binary labels (0 or 1)
        y_pred : array-like
            Predicted binary labels (0 or 1)
        sensitive_feature : array-like
            Protected attribute values
        min_representation_pct : float, optional
            Minimum percentage of samples for a group to be included in analysis.
            If None, uses FairnessMetrics.MIN_REPRESENTATION_PCT (default: 2.0%)

        Returns:
        --------
        dict : {
            'metric_name': str,
            'all_groups': Dict[str, Dict],  # Info about ALL groups
            'testable_groups': Dict[str, Dict],  # Groups meeting threshold
            'excluded_groups': Dict[str, Dict],  # Groups below threshold
            'disparity': float,  # Max difference in TPR for testable groups
            'ratio': float,  # min_tpr / max_tpr for testable groups
            'min_representation_pct': float,
            'interpretation': str
        }

        Note:
        -----
        Equal Opportunity is less strict than Equalized Odds (only requires
        equal TPR, not equal FPR). Suitable when false positives are less
        concerning than false negatives.
        """
        # Use default threshold if not specified
        if min_representation_pct is None:
            min_representation_pct = FairnessMetrics.MIN_REPRESENTATION_PCT

        # Convert to numpy arrays
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        sensitive_feature = np.asarray(sensitive_feature)

        groups = np.unique(sensitive_feature)
        total_samples = len(sensitive_feature)
        all_groups = {}

        # Calculate info for each group
        for group in groups:
            mask_all = sensitive_feature == group
            group_size = np.sum(mask_all)
            group_pct = (group_size / total_samples) * 100
            meets_threshold = group_pct >= min_representation_pct

            # Mask for: this group AND positive label
            mask_pos = (sensitive_feature == group) & (y_true == 1)
            n_positives = np.sum(mask_pos)

            if n_positives > 0:
                tpr = np.mean(y_pred[mask_pos] == 1)
            else:
                tpr = np.nan

            all_groups[str(group)] = {
                'tpr': float(tpr) if not np.isnan(tpr) else tpr,
                'count': int(group_size),
                'percentage': float(group_pct),
                'meets_threshold': meets_threshold
            }

        # Separate testable vs excluded groups
        testable_groups = {g: info for g, info in all_groups.items() if info['meets_threshold']}
        excluded_groups = {g: info for g, info in all_groups.items() if not info['meets_threshold']}

        # Calculate metrics ONLY for testable groups with valid TPR
        testable_tprs = {g: info['tpr'] for g, info in testable_groups.items() if not np.isnan(info['tpr'])}

        if len(testable_tprs) < 2:
            return {
                'metric_name': 'equal_opportunity',
                'all_groups': all_groups,
                'testable_groups': testable_groups,
                'excluded_groups': excluded_groups,
                'disparity': None,
                'ratio': None,
                'min_representation_pct': min_representation_pct,
                'interpretation': f"INSUFFICIENT: Less than 2 groups with ≥{min_representation_pct}% representation and valid TPR"
            }

        valid_tprs = list(testable_tprs.values())
        max_tpr = max(valid_tprs)
        min_tpr = min(valid_tprs)
        disparity = max_tpr - min_tpr
        ratio = min_tpr / max_tpr if max_tpr > 0 else 0.0

        return {
            'metric_name': 'equal_opportunity',
            'all_groups': all_groups,
            'testable_groups': testable_groups,
            'excluded_groups': excluded_groups,
            'disparity': float(disparity),
            'ratio': float(ratio),
            'min_representation_pct': min_representation_pct,
            'interpretation': _interpret_equal_opportunity(disparity)
        }

    @staticmethod
    def equalized_odds(y_true: Union[np.ndarray, pd.Series],
                      y_pred: Union[np.ndarray, pd.Series],
                      sensitive_feature: Union[np.ndarray, pd.Series],
                      min_representation_pct: float = None) -> Dict[str, Any]:
        """
        Equalized Odds

        Measures if BOTH TPR and FPR are equal across groups.
        More strict than Equal Opportunity.

        Formula:
            P(Y_hat=1 | Y=y, A=a) = P(Y_hat=1 | Y=y, A=b)
            for all groups a, b and y ∈ {0,1}

        Parameters:
        -----------
        y_true : array-like
            True binary labels (0 or 1)
        y_pred : array-like
            Predicted binary labels (0 or 1)
        sensitive_feature : array-like
            Protected attribute values
        min_representation_pct : float, optional
            Minimum percentage of samples for a group to be included in analysis.
            If None, uses FairnessMetrics.MIN_REPRESENTATION_PCT (default: 2.0%)

        Returns:
        --------
        dict : {
            'metric_name': str,
            'all_groups': Dict[str, Dict],  # Info about ALL groups
            'testable_groups': Dict[str, Dict],  # Groups meeting threshold
            'excluded_groups': Dict[str, Dict],  # Groups below threshold
            'tpr_disparity': float,
            'fpr_disparity': float,
            'combined_disparity': float,  # max(tpr_disp, fpr_disp)
            'min_representation_pct': float,
            'interpretation': str
        }

        Note:
        -----
        Equalized Odds is considered the strictest fairness criterion.
        It ensures both benefits (high TPR) and harms (low FPR) are
        distributed equally across groups.
        """
        # Use default threshold if not specified
        if min_representation_pct is None:
            min_representation_pct = FairnessMetrics.MIN_REPRESENTATION_PCT

        # Convert to numpy arrays
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        sensitive_feature = np.asarray(sensitive_feature)

        groups = np.unique(sensitive_feature)
        total_samples = len(sensitive_feature)
        all_groups = {}

        for group in groups:
            mask_all = sensitive_feature == group
            group_size = np.sum(mask_all)
            group_pct = (group_size / total_samples) * 100
            meets_threshold = group_pct >= min_representation_pct

            # TPR calculation
            mask_pos = (sensitive_feature == group) & (y_true == 1)
            n_positives = np.sum(mask_pos)

            if n_positives > 0:
                tpr = np.mean(y_pred[mask_pos] == 1)
            else:
                tpr = np.nan

            # FPR calculation
            mask_neg = (sensitive_feature == group) & (y_true == 0)
            n_negatives = np.sum(mask_neg)

            if n_negatives > 0:
                fpr = np.mean(y_pred[mask_neg] == 1)
            else:
                fpr = np.nan

            all_groups[str(group)] = {
                'tpr': float(tpr) if not np.isnan(tpr) else tpr,
                'fpr': float(fpr) if not np.isnan(fpr) else fpr,
                'count': int(group_size),
                'percentage': float(group_pct),
                'meets_threshold': meets_threshold
            }

        # Separate testable vs excluded groups
        testable_groups = {g: info for g, info in all_groups.items() if info['meets_threshold']}
        excluded_groups = {g: info for g, info in all_groups.items() if not info['meets_threshold']}

        # Calculate disparities ONLY for testable groups
        testable_tprs = [info['tpr'] for info in testable_groups.values() if not np.isnan(info['tpr'])]
        testable_fprs = [info['fpr'] for info in testable_groups.values() if not np.isnan(info['fpr'])]

        if len(testable_tprs) < 2 or len(testable_fprs) < 2:
            return {
                'metric_name': 'equalized_odds',
                'all_groups': all_groups,
                'testable_groups': testable_groups,
                'excluded_groups': excluded_groups,
                'tpr_disparity': None,
                'fpr_disparity': None,
                'combined_disparity': None,
                'min_representation_pct': min_representation_pct,
                'interpretation': f"INSUFFICIENT: Less than 2 groups with ≥{min_representation_pct}% representation and valid TPR/FPR"
            }

        tpr_disparity = max(testable_tprs) - min(testable_tprs)
        fpr_disparity = max(testable_fprs) - min(testable_fprs)
        combined_disparity = max(tpr_disparity, fpr_disparity)

        return {
            'metric_name': 'equalized_odds',
            'all_groups': all_groups,
            'testable_groups': testable_groups,
            'excluded_groups': excluded_groups,
            'tpr_disparity': float(tpr_disparity),
            'fpr_disparity': float(fpr_disparity),
            'combined_disparity': float(combined_disparity),
            'min_representation_pct': min_representation_pct,
            'interpretation': _interpret_equalized_odds(tpr_disparity, fpr_disparity)
        }

    @staticmethod
    def disparate_impact(y_pred: Union[np.ndarray, pd.Series],
                        sensitive_feature: Union[np.ndarray, pd.Series],
                        threshold: float = 0.8,
                        min_representation_pct: float = None) -> Dict[str, Any]:
        """
        Disparate Impact Ratio

        Ratio between selection rate of unprivileged and privileged groups.

        Legal Standard (EEOC):
            Ratio < 0.8 is considered evidence of adverse impact

        Formula:
            DI = P(Y_hat=1 | A=unprivileged) / P(Y_hat=1 | A=privileged)

        Parameters:
        -----------
        y_pred : array-like
            Binary predictions (0 or 1)
        sensitive_feature : array-like
            Protected attribute values
        threshold : float, default=0.8
            Threshold for passing (EEOC standard is 0.8)
        min_representation_pct : float, optional
            Minimum percentage of samples for a group to be included in analysis.
            If None, uses FairnessMetrics.MIN_REPRESENTATION_PCT (default: 2.0%)

        Returns:
        --------
        dict : {
            'metric_name': str,
            'all_groups': Dict[str, Dict],  # Info about ALL groups
            'testable_groups': Dict[str, Dict],  # Groups meeting threshold
            'excluded_groups': Dict[str, Dict],  # Groups below threshold
            'ratio': float,  # Ideal: 1.0 (for testable groups)
            'threshold': float,
            'passes_threshold': bool,  # >= threshold
            'unprivileged_rate': float,
            'privileged_rate': float,
            'min_representation_pct': float,
            'interpretation': str
        }

        References:
        -----------
        - EEOC Uniform Guidelines on Employee Selection (1978)
        - Federal court precedent (Griggs v. Duke Power Co., 1971)
        """
        # Use default threshold if not specified
        if min_representation_pct is None:
            min_representation_pct = FairnessMetrics.MIN_REPRESENTATION_PCT

        # Convert to numpy arrays
        y_pred = np.asarray(y_pred)
        sensitive_feature = np.asarray(sensitive_feature)

        groups = np.unique(sensitive_feature)
        total_samples = len(sensitive_feature)
        all_groups = {}

        # Calculate info for each group
        for group in groups:
            mask = sensitive_feature == group
            group_size = np.sum(mask)
            group_pct = (group_size / total_samples) * 100
            meets_threshold = group_pct >= min_representation_pct

            if group_size > 0:
                positive_rate = np.mean(y_pred[mask] == 1)
            else:
                positive_rate = 0.0

            all_groups[str(group)] = {
                'rate': float(positive_rate),
                'count': int(group_size),
                'percentage': float(group_pct),
                'meets_threshold': meets_threshold
            }

        # Separate testable vs excluded groups
        testable_groups = {g: info for g, info in all_groups.items() if info['meets_threshold']}
        excluded_groups = {g: info for g, info in all_groups.items() if not info['meets_threshold']}

        # Calculate metrics ONLY for testable groups
        if len(testable_groups) < 2:
            return {
                'metric_name': 'disparate_impact',
                'all_groups': all_groups,
                'testable_groups': testable_groups,
                'excluded_groups': excluded_groups,
                'ratio': None,
                'threshold': threshold,
                'passes_threshold': None,
                'unprivileged_rate': None,
                'privileged_rate': None,
                'min_representation_pct': min_representation_pct,
                'interpretation': f"INSUFFICIENT: Less than 2 groups with ≥{min_representation_pct}% representation for analysis"
            }

        testable_rates = [info['rate'] for info in testable_groups.values()]
        min_rate = min(testable_rates)
        max_rate = max(testable_rates)

        ratio = min_rate / max_rate if max_rate > 0 else 0.0
        passes_threshold = ratio >= threshold

        return {
            'metric_name': 'disparate_impact',
            'all_groups': all_groups,
            'testable_groups': testable_groups,
            'excluded_groups': excluded_groups,
            'ratio': float(ratio),
            'threshold': threshold,
            'passes_threshold': passes_threshold,
            'unprivileged_rate': float(min_rate),
            'privileged_rate': float(max_rate),
            'min_representation_pct': min_representation_pct,
            'interpretation': _interpret_disparate_impact(ratio, threshold)
        }

    # ==================== PRE-TRAINING METRICS ====================
    # These metrics are model-independent and evaluate dataset bias

    @staticmethod
    def class_balance(y_true: Union[np.ndarray, pd.Series],
                     sensitive_feature: Union[np.ndarray, pd.Series]) -> Dict[str, Any]:
        """
        Class Balance (BCL)

        Measures the balance of sample sizes between groups.
        Detects if one group is underrepresented in the dataset.

        Formula:
            BCL = (n_a - n_b) / n_total
            where n_a, n_b are group sizes

        Parameters:
        -----------
        y_true : array-like
            True labels (not used, but kept for API consistency)
        sensitive_feature : array-like
            Protected attribute values

        Returns:
        --------
        dict : {
            'metric_name': str,
            'value': float,  # BCL value (-1 to 1, ideal: 0)
            'group_a': str,
            'group_b': str,
            'group_a_size': int,
            'group_b_size': int,
            'total_size': int,
            'interpretation': str
        }
        """
        sensitive_feature = np.asarray(sensitive_feature)

        # Get unique groups and their sizes
        groups = np.unique(sensitive_feature)
        if len(groups) < 2:
            return {
                'metric_name': 'class_balance',
                'value': 0.0,
                'group_a': str(groups[0]) if len(groups) > 0 else 'N/A',
                'group_b': 'N/A',
                'group_a_size': len(sensitive_feature),
                'group_b_size': 0,
                'total_size': len(sensitive_feature),
                'interpretation': 'Apenas um grupo presente'
            }

        # Use top 2 most frequent groups
        group_counts = pd.Series(sensitive_feature).value_counts()
        group_a, group_b = group_counts.index[:2]
        n_a = group_counts.iloc[0]
        n_b = group_counts.iloc[1]
        n_total = len(sensitive_feature)

        # Calculate BCL
        bcl = (n_a - n_b) / n_total

        return {
            'metric_name': 'class_balance',
            'value': float(bcl),
            'group_a': str(group_a),
            'group_b': str(group_b),
            'group_a_size': int(n_a),
            'group_b_size': int(n_b),
            'total_size': int(n_total),
            'interpretation': _interpret_class_balance(bcl)
        }

    @staticmethod
    def concept_balance(y_true: Union[np.ndarray, pd.Series],
                       sensitive_feature: Union[np.ndarray, pd.Series]) -> Dict[str, Any]:
        """
        Concept Balance (BCO)

        Measures if the positive class rate differs between groups.
        Detects if one group has inherently more positive outcomes.

        Formula:
            BCO = P(Y=1 | A=a) - P(Y=1 | A=b)

        Parameters:
        -----------
        y_true : array-like
            True binary labels (0 or 1)
        sensitive_feature : array-like
            Protected attribute values

        Returns:
        --------
        dict : {
            'metric_name': str,
            'value': float,  # BCO value (ideal: 0)
            'group_a': str,
            'group_b': str,
            'group_a_positive_rate': float,
            'group_b_positive_rate': float,
            'interpretation': str
        }
        """
        y_true = np.asarray(y_true)
        sensitive_feature = np.asarray(sensitive_feature)

        groups = np.unique(sensitive_feature)
        if len(groups) < 2:
            return {
                'metric_name': 'concept_balance',
                'value': 0.0,
                'group_a': str(groups[0]) if len(groups) > 0 else 'N/A',
                'group_b': 'N/A',
                'group_a_positive_rate': 0.0,
                'group_b_positive_rate': 0.0,
                'interpretation': 'Apenas um grupo presente'
            }

        # Use top 2 most frequent groups
        group_counts = pd.Series(sensitive_feature).value_counts()
        group_a, group_b = group_counts.index[:2]

        # Calculate positive rates
        mask_a = sensitive_feature == group_a
        mask_b = sensitive_feature == group_b

        rate_a = np.mean(y_true[mask_a] == 1) if np.sum(mask_a) > 0 else 0.0
        rate_b = np.mean(y_true[mask_b] == 1) if np.sum(mask_b) > 0 else 0.0

        bco = rate_a - rate_b

        return {
            'metric_name': 'concept_balance',
            'value': float(bco),
            'group_a': str(group_a),
            'group_b': str(group_b),
            'group_a_positive_rate': float(rate_a),
            'group_b_positive_rate': float(rate_b),
            'interpretation': _interpret_concept_balance(bco)
        }

    @staticmethod
    def kl_divergence(y_true: Union[np.ndarray, pd.Series],
                     sensitive_feature: Union[np.ndarray, pd.Series]) -> Dict[str, Any]:
        """
        Kullback-Leibler Divergence (KL)

        Measures the difference in label distributions between groups.
        Asymmetric measure (KL(P||Q) ≠ KL(Q||P)).

        Formula:
            KL(P||Q) = Σ P(x) * log(P(x) / Q(x))

        Parameters:
        -----------
        y_true : array-like
            True binary labels (0 or 1)
        sensitive_feature : array-like
            Protected attribute values

        Returns:
        --------
        dict : {
            'metric_name': str,
            'value': float,  # KL divergence (>= 0, ideal: 0)
            'group_a': str,
            'group_b': str,
            'interpretation': str
        }
        """
        from scipy.stats import entropy

        y_true = np.asarray(y_true)
        sensitive_feature = np.asarray(sensitive_feature)

        groups = np.unique(sensitive_feature)
        if len(groups) < 2:
            return {
                'metric_name': 'kl_divergence',
                'value': 0.0,
                'group_a': str(groups[0]) if len(groups) > 0 else 'N/A',
                'group_b': 'N/A',
                'interpretation': 'Apenas um grupo presente'
            }

        # Use top 2 most frequent groups
        group_counts = pd.Series(sensitive_feature).value_counts()
        group_a, group_b = group_counts.index[:2]

        # Get label distributions
        mask_a = sensitive_feature == group_a
        mask_b = sensitive_feature == group_b

        dist_a = pd.Series(y_true[mask_a]).value_counts(normalize=True).sort_index()
        dist_b = pd.Series(y_true[mask_b]).value_counts(normalize=True).sort_index()

        # Ensure same categories
        all_cats = sorted(set(dist_a.index) | set(dist_b.index))
        dist_a = dist_a.reindex(all_cats, fill_value=1e-10)
        dist_b = dist_b.reindex(all_cats, fill_value=1e-10)

        # Calculate KL divergence
        kl = entropy(dist_a, dist_b)

        return {
            'metric_name': 'kl_divergence',
            'value': float(kl),
            'group_a': str(group_a),
            'group_b': str(group_b),
            'interpretation': _interpret_kl_divergence(kl)
        }

    @staticmethod
    def js_divergence(y_true: Union[np.ndarray, pd.Series],
                     sensitive_feature: Union[np.ndarray, pd.Series]) -> Dict[str, Any]:
        """
        Jensen-Shannon Divergence (JS)

        Symmetric version of KL divergence.
        Measures distribution similarity (bounded, symmetric).

        Formula:
            JS(P||Q) = 0.5 * [KL(P||M) + KL(Q||M)]
            where M = 0.5 * (P + Q)

        Parameters:
        -----------
        y_true : array-like
            True binary labels (0 or 1)
        sensitive_feature : array-like
            Protected attribute values

        Returns:
        --------
        dict : {
            'metric_name': str,
            'value': float,  # JS divergence (0 to 1, ideal: 0)
            'group_a': str,
            'group_b': str,
            'interpretation': str
        }
        """
        from scipy.stats import entropy

        y_true = np.asarray(y_true)
        sensitive_feature = np.asarray(sensitive_feature)

        groups = np.unique(sensitive_feature)
        if len(groups) < 2:
            return {
                'metric_name': 'js_divergence',
                'value': 0.0,
                'group_a': str(groups[0]) if len(groups) > 0 else 'N/A',
                'group_b': 'N/A',
                'interpretation': 'Apenas um grupo presente'
            }

        # Use top 2 most frequent groups
        group_counts = pd.Series(sensitive_feature).value_counts()
        group_a, group_b = group_counts.index[:2]

        # Get label distributions
        mask_a = sensitive_feature == group_a
        mask_b = sensitive_feature == group_b

        dist_a = pd.Series(y_true[mask_a]).value_counts(normalize=True).sort_index()
        dist_b = pd.Series(y_true[mask_b]).value_counts(normalize=True).sort_index()

        # Ensure same categories
        all_cats = sorted(set(dist_a.index) | set(dist_b.index))
        dist_a = dist_a.reindex(all_cats, fill_value=1e-10)
        dist_b = dist_b.reindex(all_cats, fill_value=1e-10)

        # Calculate JS divergence
        dist_m = (dist_a + dist_b) / 2
        js = 0.5 * (entropy(dist_a, dist_m) + entropy(dist_b, dist_m))

        return {
            'metric_name': 'js_divergence',
            'value': float(js),
            'group_a': str(group_a),
            'group_b': str(group_b),
            'interpretation': _interpret_js_divergence(js)
        }

    # ==================== POST-TRAINING METRICS ====================
    # These metrics evaluate model predictions for fairness

    @staticmethod
    def false_negative_rate_difference(y_true: Union[np.ndarray, pd.Series],
                                       y_pred: Union[np.ndarray, pd.Series],
                                       sensitive_feature: Union[np.ndarray, pd.Series],
                                       min_representation_pct: float = None) -> Dict[str, Any]:
        """
        False Negative Rate Difference (TFN)

        Measures the difference in False Negative Rate (Miss Rate) between groups.
        Important when missing positive cases has severe consequences.

        Formula:
            TFN = FNR_a - FNR_b
            where FNR = FN / (FN + TP)

        Parameters:
        -----------
        y_true : array-like
            True binary labels (0 or 1)
        y_pred : array-like
            Predicted binary labels (0 or 1)
        sensitive_feature : array-like
            Protected attribute values
        min_representation_pct : float, optional
            Minimum percentage of samples for a group to be included in analysis.
            If None, uses FairnessMetrics.MIN_REPRESENTATION_PCT (default: 2.0%)

        Returns:
        --------
        dict : {
            'metric_name': str,
            'all_groups': Dict[str, Dict],  # Info about ALL groups
            'testable_groups': Dict[str, Dict],  # Groups meeting threshold
            'excluded_groups': Dict[str, Dict],  # Groups below threshold
            'value': float,  # FNR difference (ideal: 0) for testable groups
            'group_a': str,
            'group_b': str,
            'group_a_fnr': float,
            'group_b_fnr': float,
            'min_representation_pct': float,
            'interpretation': str
        }
        """
        from sklearn.metrics import confusion_matrix

        # Use default threshold if not specified
        if min_representation_pct is None:
            min_representation_pct = FairnessMetrics.MIN_REPRESENTATION_PCT

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        sensitive_feature = np.asarray(sensitive_feature)

        groups = np.unique(sensitive_feature)
        total_samples = len(sensitive_feature)
        all_groups = {}

        # Calculate info for each group
        for group in groups:
            mask = sensitive_feature == group
            group_size = np.sum(mask)
            group_pct = (group_size / total_samples) * 100
            meets_threshold = group_pct >= min_representation_pct

            if group_size > 0:
                cm = confusion_matrix(y_true[mask], y_pred[mask], labels=[0, 1])
                fn = cm[1, 0]  # False Negatives
                tp = cm[1, 1]  # True Positives
                fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            else:
                fn, tp, fnr = 0, 0, 0.0

            all_groups[str(group)] = {
                'fnr': float(fnr),
                'fn': int(fn),
                'tp': int(tp),
                'count': int(group_size),
                'percentage': float(group_pct),
                'meets_threshold': meets_threshold
            }

        # Separate testable vs excluded groups
        testable_groups = {g: info for g, info in all_groups.items() if info['meets_threshold']}
        excluded_groups = {g: info for g, info in all_groups.items() if not info['meets_threshold']}

        # Calculate metrics for top 2 testable groups
        if len(testable_groups) < 2:
            return {
                'metric_name': 'false_negative_rate_difference',
                'all_groups': all_groups,
                'testable_groups': testable_groups,
                'excluded_groups': excluded_groups,
                'value': None,
                'group_a': 'N/A',
                'group_b': 'N/A',
                'group_a_fnr': None,
                'group_b_fnr': None,
                'min_representation_pct': min_representation_pct,
                'interpretation': f"INSUFFICIENT: Less than 2 groups with ≥{min_representation_pct}% representation"
            }

        # Get top 2 testable groups by count
        sorted_testable = sorted(testable_groups.items(), key=lambda x: x[1]['count'], reverse=True)
        group_a, info_a = sorted_testable[0]
        group_b, info_b = sorted_testable[1]

        fnr_a = info_a['fnr']
        fnr_b = info_b['fnr']
        tfn = fnr_a - fnr_b

        return {
            'metric_name': 'false_negative_rate_difference',
            'all_groups': all_groups,
            'testable_groups': testable_groups,
            'excluded_groups': excluded_groups,
            'value': float(tfn),
            'group_a': str(group_a),
            'group_b': str(group_b),
            'group_a_fnr': float(fnr_a),
            'group_b_fnr': float(fnr_b),
            'min_representation_pct': min_representation_pct,
            'interpretation': _interpret_fnr_difference(tfn)
        }

    @staticmethod
    def conditional_acceptance(y_true: Union[np.ndarray, pd.Series],
                              y_pred: Union[np.ndarray, pd.Series],
                              sensitive_feature: Union[np.ndarray, pd.Series]) -> Dict[str, Any]:
        """
        Conditional Acceptance (AC)

        Measures if the proportion of true positives among predicted positives
        is equal across groups.

        Formula:
            AC = P(Y=1 | Y_hat=1, A=a) - P(Y=1 | Y_hat=1, A=b)
            This is related to Precision/PPV

        Parameters:
        -----------
        y_true : array-like
            True binary labels (0 or 1)
        y_pred : array-like
            Predicted binary labels (0 or 1)
        sensitive_feature : array-like
            Protected attribute values

        Returns:
        --------
        dict : {
            'metric_name': str,
            'value': float,  # AC difference (ideal: 0)
            'group_a': str,
            'group_b': str,
            'group_a_rate': float,
            'group_b_rate': float,
            'interpretation': str
        }
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        sensitive_feature = np.asarray(sensitive_feature)

        groups = np.unique(sensitive_feature)
        if len(groups) < 2:
            return {
                'metric_name': 'conditional_acceptance',
                'value': 0.0,
                'group_a': 'N/A',
                'group_b': 'N/A',
                'group_a_rate': 0.0,
                'group_b_rate': 0.0,
                'interpretation': 'Apenas um grupo presente'
            }

        # Use top 2 most frequent groups
        group_counts = pd.Series(sensitive_feature).value_counts()
        group_a, group_b = group_counts.index[:2]

        # Calculate conditional acceptance for each group
        mask_a = sensitive_feature == group_a
        mask_b = sensitive_feature == group_b

        # P(Y=1 | Y_hat=1, A=a)
        pred_pos_a = y_pred[mask_a] == 1
        pred_pos_b = y_pred[mask_b] == 1

        n_pred_pos_a = np.sum(pred_pos_a)
        n_pred_pos_b = np.sum(pred_pos_b)

        if n_pred_pos_a > 0:
            rate_a = np.mean(y_true[mask_a][pred_pos_a] == 1)
        else:
            rate_a = 0.0

        if n_pred_pos_b > 0:
            rate_b = np.mean(y_true[mask_b][pred_pos_b] == 1)
        else:
            rate_b = 0.0

        ac = rate_a - rate_b

        return {
            'metric_name': 'conditional_acceptance',
            'value': float(ac),
            'group_a': str(group_a),
            'group_b': str(group_b),
            'group_a_rate': float(rate_a),
            'group_b_rate': float(rate_b),
            'interpretation': _interpret_conditional_acceptance(ac)
        }

    @staticmethod
    def conditional_rejection(y_true: Union[np.ndarray, pd.Series],
                             y_pred: Union[np.ndarray, pd.Series],
                             sensitive_feature: Union[np.ndarray, pd.Series]) -> Dict[str, Any]:
        """
        Conditional Rejection (RC)

        Measures if the proportion of true negatives among predicted negatives
        is equal across groups.

        Formula:
            RC = P(Y=0 | Y_hat=0, A=a) - P(Y=0 | Y_hat=0, A=b)
            This is related to NPV (Negative Predictive Value)

        Parameters:
        -----------
        y_true : array-like
            True binary labels (0 or 1)
        y_pred : array-like
            Predicted binary labels (0 or 1)
        sensitive_feature : array-like
            Protected attribute values

        Returns:
        --------
        dict : {
            'metric_name': str,
            'value': float,  # RC difference (ideal: 0)
            'group_a': str,
            'group_b': str,
            'group_a_rate': float,
            'group_b_rate': float,
            'interpretation': str
        }
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        sensitive_feature = np.asarray(sensitive_feature)

        groups = np.unique(sensitive_feature)
        if len(groups) < 2:
            return {
                'metric_name': 'conditional_rejection',
                'value': 0.0,
                'group_a': 'N/A',
                'group_b': 'N/A',
                'group_a_rate': 0.0,
                'group_b_rate': 0.0,
                'interpretation': 'Apenas um grupo presente'
            }

        # Use top 2 most frequent groups
        group_counts = pd.Series(sensitive_feature).value_counts()
        group_a, group_b = group_counts.index[:2]

        # Calculate conditional rejection for each group
        mask_a = sensitive_feature == group_a
        mask_b = sensitive_feature == group_b

        # P(Y=0 | Y_hat=0, A=a)
        pred_neg_a = y_pred[mask_a] == 0
        pred_neg_b = y_pred[mask_b] == 0

        n_pred_neg_a = np.sum(pred_neg_a)
        n_pred_neg_b = np.sum(pred_neg_b)

        if n_pred_neg_a > 0:
            rate_a = np.mean(y_true[mask_a][pred_neg_a] == 0)
        else:
            rate_a = 0.0

        if n_pred_neg_b > 0:
            rate_b = np.mean(y_true[mask_b][pred_neg_b] == 0)
        else:
            rate_b = 0.0

        rc = rate_a - rate_b

        return {
            'metric_name': 'conditional_rejection',
            'value': float(rc),
            'group_a': str(group_a),
            'group_b': str(group_b),
            'group_a_rate': float(rate_a),
            'group_b_rate': float(rate_b),
            'interpretation': _interpret_conditional_rejection(rc)
        }

    @staticmethod
    def precision_difference(y_true: Union[np.ndarray, pd.Series],
                            y_pred: Union[np.ndarray, pd.Series],
                            sensitive_feature: Union[np.ndarray, pd.Series]) -> Dict[str, Any]:
        """
        Precision Difference (DP)

        Measures the difference in Precision (PPV) between groups.

        Formula:
            DP = Precision_a - Precision_b
            where Precision = TP / (TP + FP)

        Parameters:
        -----------
        y_true : array-like
            True binary labels (0 or 1)
        y_pred : array-like
            Predicted binary labels (0 or 1)
        sensitive_feature : array-like
            Protected attribute values

        Returns:
        --------
        dict : {
            'metric_name': str,
            'value': float,  # Precision difference (ideal: 0)
            'group_a': str,
            'group_b': str,
            'group_a_precision': float,
            'group_b_precision': float,
            'interpretation': str
        }
        """
        from sklearn.metrics import precision_score

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        sensitive_feature = np.asarray(sensitive_feature)

        groups = np.unique(sensitive_feature)
        if len(groups) < 2:
            return {
                'metric_name': 'precision_difference',
                'value': 0.0,
                'group_a': 'N/A',
                'group_b': 'N/A',
                'group_a_precision': 0.0,
                'group_b_precision': 0.0,
                'interpretation': 'Apenas um grupo presente'
            }

        # Use top 2 most frequent groups
        group_counts = pd.Series(sensitive_feature).value_counts()
        group_a, group_b = group_counts.index[:2]

        # Calculate precision for each group
        mask_a = sensitive_feature == group_a
        mask_b = sensitive_feature == group_b

        prec_a = precision_score(y_true[mask_a], y_pred[mask_a], zero_division=0)
        prec_b = precision_score(y_true[mask_b], y_pred[mask_b], zero_division=0)

        dp = prec_a - prec_b

        return {
            'metric_name': 'precision_difference',
            'value': float(dp),
            'group_a': str(group_a),
            'group_b': str(group_b),
            'group_a_precision': float(prec_a),
            'group_b_precision': float(prec_b),
            'interpretation': _interpret_precision_difference(dp)
        }

    @staticmethod
    def accuracy_difference(y_true: Union[np.ndarray, pd.Series],
                           y_pred: Union[np.ndarray, pd.Series],
                           sensitive_feature: Union[np.ndarray, pd.Series]) -> Dict[str, Any]:
        """
        Accuracy Difference (DA)

        Measures the difference in overall Accuracy between groups.

        Formula:
            DA = Accuracy_a - Accuracy_b
            where Accuracy = (TP + TN) / (TP + TN + FP + FN)

        Parameters:
        -----------
        y_true : array-like
            True binary labels (0 or 1)
        y_pred : array-like
            Predicted binary labels (0 or 1)
        sensitive_feature : array-like
            Protected attribute values

        Returns:
        --------
        dict : {
            'metric_name': str,
            'value': float,  # Accuracy difference (ideal: 0)
            'group_a': str,
            'group_b': str,
            'group_a_accuracy': float,
            'group_b_accuracy': float,
            'interpretation': str
        }
        """
        from sklearn.metrics import accuracy_score

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        sensitive_feature = np.asarray(sensitive_feature)

        groups = np.unique(sensitive_feature)
        if len(groups) < 2:
            return {
                'metric_name': 'accuracy_difference',
                'value': 0.0,
                'group_a': 'N/A',
                'group_b': 'N/A',
                'group_a_accuracy': 0.0,
                'group_b_accuracy': 0.0,
                'interpretation': 'Apenas um grupo presente'
            }

        # Use top 2 most frequent groups
        group_counts = pd.Series(sensitive_feature).value_counts()
        group_a, group_b = group_counts.index[:2]

        # Calculate accuracy for each group
        mask_a = sensitive_feature == group_a
        mask_b = sensitive_feature == group_b

        acc_a = accuracy_score(y_true[mask_a], y_pred[mask_a])
        acc_b = accuracy_score(y_true[mask_b], y_pred[mask_b])

        da = acc_a - acc_b

        return {
            'metric_name': 'accuracy_difference',
            'value': float(da),
            'group_a': str(group_a),
            'group_b': str(group_b),
            'group_a_accuracy': float(acc_a),
            'group_b_accuracy': float(acc_b),
            'interpretation': _interpret_accuracy_difference(da)
        }

    @staticmethod
    def treatment_equality(y_true: Union[np.ndarray, pd.Series],
                          y_pred: Union[np.ndarray, pd.Series],
                          sensitive_feature: Union[np.ndarray, pd.Series]) -> Dict[str, Any]:
        """
        Treatment Equality (IT)

        Measures the ratio of errors (FN/FP) between groups.
        Ensures that the balance between missing positives and false alarms
        is similar across groups.

        Formula:
            IT = (FN_a / FP_a) - (FN_b / FP_b)

        Parameters:
        -----------
        y_true : array-like
            True binary labels (0 or 1)
        y_pred : array-like
            Predicted binary labels (0 or 1)
        sensitive_feature : array-like
            Protected attribute values

        Returns:
        --------
        dict : {
            'metric_name': str,
            'value': float,  # Treatment equality difference (ideal: 0)
            'group_a': str,
            'group_b': str,
            'group_a_ratio': float,
            'group_b_ratio': float,
            'interpretation': str
        }
        """
        from sklearn.metrics import confusion_matrix

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        sensitive_feature = np.asarray(sensitive_feature)

        groups = np.unique(sensitive_feature)
        if len(groups) < 2:
            return {
                'metric_name': 'treatment_equality',
                'value': 0.0,
                'group_a': 'N/A',
                'group_b': 'N/A',
                'group_a_ratio': 0.0,
                'group_b_ratio': 0.0,
                'interpretation': 'Apenas um grupo presente'
            }

        # Use top 2 most frequent groups
        group_counts = pd.Series(sensitive_feature).value_counts()
        group_a, group_b = group_counts.index[:2]

        # Calculate FN/FP ratio for each group
        mask_a = sensitive_feature == group_a
        mask_b = sensitive_feature == group_b

        cm_a = confusion_matrix(y_true[mask_a], y_pred[mask_a], labels=[0, 1])
        cm_b = confusion_matrix(y_true[mask_b], y_pred[mask_b], labels=[0, 1])

        # Extract FN and FP
        fp_a = cm_a[0, 1]  # False Positives
        fn_a = cm_a[1, 0]  # False Negatives
        fp_b = cm_b[0, 1]
        fn_b = cm_b[1, 0]

        # Calculate ratios
        ratio_a = fn_a / fp_a if fp_a > 0 else 0.0
        ratio_b = fn_b / fp_b if fp_b > 0 else 0.0

        it = ratio_a - ratio_b

        return {
            'metric_name': 'treatment_equality',
            'value': float(it),
            'group_a': str(group_a),
            'group_b': str(group_b),
            'group_a_ratio': float(ratio_a),
            'group_b_ratio': float(ratio_b),
            'interpretation': _interpret_treatment_equality(it)
        }

    @staticmethod
    def entropy_index(y_true: Union[np.ndarray, pd.Series],
                     y_pred: Union[np.ndarray, pd.Series],
                     alpha: float = 2.0) -> Dict[str, Any]:
        """
        Entropy Index (IE) - Individual Fairness

        Measures individual-level fairness using generalized entropy.
        Unlike group fairness metrics, this evaluates fairness at the individual level.

        Formula:
            IE = (1 / (n * α * (α-1))) * Σ[(b_i / μ)^α - 1]
            where b_i = |y_pred - y_true| + 1

        Parameters:
        -----------
        y_true : array-like
            True binary labels (0 or 1)
        y_pred : array-like
            Predicted binary labels (0 or 1)
        alpha : float, default=2.0
            Entropy parameter (typically 0, 1, or 2)

        Returns:
        --------
        dict : {
            'metric_name': str,
            'value': float,  # Entropy index (>= 0, ideal: 0)
            'alpha': float,
            'interpretation': str
        }

        Note:
        -----
        This metric does not use sensitive features, making it a measure
        of individual fairness rather than group fairness.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Calculate benefit (error + 1)
        b_i = np.abs(y_pred - y_true) + 1
        mu = np.mean(b_i)
        n = len(b_i)

        # Calculate entropy index
        if alpha == 0:
            # L'Hopital's rule for alpha -> 0
            ie = -np.mean(np.log(b_i / mu)) / n
        elif alpha == 1:
            # L'Hopital's rule for alpha -> 1
            ie = np.mean((b_i / mu) * np.log(b_i / mu)) / n
        else:
            ie = np.sum((b_i / mu) ** alpha - 1) / (n * alpha * (alpha - 1))

        return {
            'metric_name': 'entropy_index',
            'value': float(ie),
            'alpha': alpha,
            'interpretation': _interpret_entropy_index(ie)
        }


# ==================== Interpretation Helpers ====================

def _interpret_statistical_parity(disparity: float, ratio: float) -> str:
    """Generate human-readable interpretation for statistical parity"""
    if disparity < 0.01:
        return "EXCELLENT: Near perfect statistical parity across groups"
    elif ratio >= 0.8:
        return "GOOD: Passes EEOC 80% rule (compliant with regulations)"
    elif ratio >= 0.6:
        return "MODERATE: Some disparity present - requires investigation"
    else:
        return "CRITICAL: Significant disparity detected - high discrimination risk"


def _interpret_equal_opportunity(disparity: float) -> str:
    """Generate human-readable interpretation for equal opportunity"""
    if disparity < 0.05:
        return "EXCELLENT: True Positive Rate balanced across groups"
    elif disparity < 0.1:
        return "GOOD: Small difference in TPR (acceptable for most cases)"
    elif disparity < 0.2:
        return "MODERATE: Notable difference in TPR - some groups disadvantaged"
    else:
        return "CRITICAL: Significant difference in TPR - groups clearly harmed"


def _interpret_equalized_odds(tpr_disp: float, fpr_disp: float) -> str:
    """Generate human-readable interpretation for equalized odds"""
    max_disp = max(tpr_disp, fpr_disp)

    if max_disp < 0.05:
        return "EXCELLENT: TPR and FPR balanced across all groups"
    elif max_disp < 0.1:
        return "GOOD: Small differences in TPR/FPR (within acceptable limits)"
    elif max_disp < 0.2:
        return "MODERATE: Notable differences in TPR or FPR - requires attention"
    else:
        return "CRITICAL: Significant differences in TPR/FPR - equalized odds violation"


def _interpret_disparate_impact(ratio: float, threshold: float) -> str:
    """Generate human-readable interpretation for disparate impact"""
    if ratio >= 0.95:
        return "EXCELLENT: Nearly equal impact across groups (no evidence of discrimination)"
    elif ratio >= threshold:
        return f"GOOD: Passes EEOC threshold {threshold} (compliant with regulations)"
    elif ratio >= 0.6:
        return f"MODERATE: Below threshold {threshold} - attention required"
    else:
        return "CRITICAL: Significant disparate impact - HIGH LEGAL RISK of discrimination"


# ==================== PRE-TRAINING INTERPRETATIONS ====================

def _interpret_class_balance(bcl: float) -> str:
    """Generate human-readable interpretation for class balance"""
    abs_bcl = abs(bcl)
    if abs_bcl <= 0.1:
        return "✓ Green: Adequate balance between groups"
    elif abs_bcl <= 0.3:
        return "⚠ Yellow: Moderate imbalance - consider oversampling/undersampling"
    else:
        return "✗ Red: Critical imbalance - risk of model bias"


def _interpret_concept_balance(bco: float) -> str:
    """Generate human-readable interpretation for concept balance"""
    abs_bco = abs(bco)
    if abs_bco <= 0.05:
        return "✓ Green: Concept balanced across groups"
    elif abs_bco <= 0.15:
        return "⚠ Yellow: Moderate concept imbalance"
    else:
        return "✗ Red: Critical concept imbalance - possible structural bias"


def _interpret_kl_divergence(kl: float) -> str:
    """Generate human-readable interpretation for KL divergence"""
    if kl < 0.1:
        return "✓ Green: Very similar distributions"
    elif kl < 0.5:
        return "⚠ Yellow: Moderately different distributions"
    else:
        return "✗ Red: Very different distributions - high bias risk"


def _interpret_js_divergence(js: float) -> str:
    """Generate human-readable interpretation for JS divergence"""
    if js < 0.05:
        return "✓ Green: Very similar distributions"
    elif js < 0.2:
        return "⚠ Yellow: Moderately different distributions"
    else:
        return "✗ Red: Very different distributions - high bias risk"


# ==================== POST-TRAINING INTERPRETATIONS ====================

def _interpret_fnr_difference(tfn: float) -> str:
    """Generate human-readable interpretation for FNR difference"""
    abs_tfn = abs(tfn)
    if abs_tfn <= 0.05:
        return "✓ Green: FN rate balanced across groups"
    elif abs_tfn <= 0.15:
        return "⚠ Yellow: Moderate difference in FN - some groups lose opportunities"
    else:
        return "✗ Red: Critical difference in FN - groups significantly harmed"


def _interpret_conditional_acceptance(ac: float) -> str:
    """Generate human-readable interpretation for conditional acceptance"""
    abs_ac = abs(ac)
    if abs_ac <= 0.05:
        return "✓ Green: Adequate conditional acceptance"
    elif abs_ac <= 0.15:
        return "⚠ Yellow: Possible bias in conditional acceptance"
    else:
        return "✗ Red: Critical bias in acceptance - different precision patterns"


def _interpret_conditional_rejection(rc: float) -> str:
    """Generate human-readable interpretation for conditional rejection"""
    abs_rc = abs(rc)
    if abs_rc <= 0.05:
        return "✓ Green: Adequate conditional rejection"
    elif abs_rc <= 0.15:
        return "⚠ Yellow: Possible bias in conditional rejection"
    else:
        return "✗ Red: Critical bias in rejection - different NPV patterns"


def _interpret_precision_difference(dp: float) -> str:
    """Generate human-readable interpretation for precision difference"""
    abs_dp = abs(dp)
    if abs_dp <= 0.05:
        return "✓ Green: Precision balanced across groups"
    elif abs_dp <= 0.15:
        return "⚠ Yellow: Moderate difference in precision"
    else:
        return "✗ Red: Critical difference in precision - unequal reliability"


def _interpret_accuracy_difference(da: float) -> str:
    """Generate human-readable interpretation for accuracy difference"""
    abs_da = abs(da)
    if abs_da <= 0.05:
        return "✓ Green: Accuracy balanced across groups"
    elif abs_da <= 0.15:
        return "⚠ Yellow: Moderate difference in accuracy"
    else:
        return "✗ Red: Critical difference in accuracy - unequal performance"


def _interpret_treatment_equality(it: float) -> str:
    """Generate human-readable interpretation for treatment equality"""
    abs_it = abs(it)
    if abs_it < 0.5:
        return "✓ Green: Balanced treatment between FN and FP"
    elif abs_it < 1.5:
        return "⚠ Yellow: Moderate imbalance between error types"
    else:
        return "✗ Red: Critical imbalance - one group suffers more FN or FP"


def _interpret_entropy_index(ie: float) -> str:
    """Generate human-readable interpretation for entropy index"""
    abs_ie = abs(ie)
    if abs_ie < 0.1:
        return "✓ Green: Low individual inequality"
    elif abs_ie < 0.3:
        return "⚠ Yellow: Moderate inequality at individual level"
    else:
        return "✗ Red: High individual inequality - fairness compromised"
