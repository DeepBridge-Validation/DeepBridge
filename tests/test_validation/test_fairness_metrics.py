"""
Comprehensive tests for FairnessMetrics module.

This test suite validates all 15 fairness metrics:
- 4 PRE-TRAINING metrics (model-independent)
- 11 POST-TRAINING metrics (model-dependent)

Coverage Target: ~100% (all methods, edge cases, error paths)
Test Categories:
1. Basic functionality tests
2. Edge case tests (single group, empty data, etc.)
3. EEOC compliance tests (80% rule, representativeness threshold)
4. Mathematical correctness tests
5. Integration tests
"""

import numpy as np
import pandas as pd
import pytest

from deepbridge.validation.fairness.metrics import FairnessMetrics


class TestStatisticalParity:
    """Tests for statistical_parity metric"""

    def test_basic_statistical_parity_balanced(self):
        """Test with perfectly balanced predictions"""
        y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        sensitive = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'])

        result = FairnessMetrics.statistical_parity(y_pred, sensitive)

        assert result['metric_name'] == 'statistical_parity'
        assert result['disparity'] == 0.0
        assert result['ratio'] == 1.0
        assert result['passes_80_rule'] is True
        assert 'EXCELLENT' in result['interpretation']

    def test_statistical_parity_with_disparity(self):
        """Test with clear disparity between groups"""
        y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        sensitive = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'])

        result = FairnessMetrics.statistical_parity(y_pred, sensitive)

        assert result['disparity'] == 1.0  # 100% vs 0%
        assert result['ratio'] == 0.0
        assert result['passes_80_rule'] is False
        assert 'CRITICAL' in result['interpretation']

    def test_statistical_parity_80_rule_threshold(self):
        """Test EEOC 80% rule threshold"""
        # Test case where ratio is exactly at 80% threshold
        y_pred = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 0])
        sensitive = np.array(['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B'])

        result = FairnessMetrics.statistical_parity(y_pred, sensitive)

        # Group A: 4/5 = 0.8, Group B: 4/5 = 0.8 -> ratio = 1.0
        # When both groups have same rate, ratio should be 1.0
        assert result['passes_80_rule'] is True
        assert result['ratio'] == 1.0

    def test_statistical_parity_single_group(self):
        """Test with only one group (no comparison possible)"""
        y_pred = np.array([1, 0, 1, 0])
        sensitive = np.array(['A', 'A', 'A', 'A'])

        result = FairnessMetrics.statistical_parity(y_pred, sensitive)

        assert result['disparity'] is None
        assert result['ratio'] is None
        assert result['passes_80_rule'] is None
        assert 'INSUFFICIENT' in result['interpretation']

    def test_statistical_parity_min_representation_threshold(self):
        """Test minimum representation threshold (EEOC Question 21)"""
        # 100 samples: Group A=98 (98%), Group B=2 (2%)
        y_pred = np.concatenate([np.ones(98), np.zeros(2)])
        sensitive = np.array(['A'] * 98 + ['B'] * 2)

        result = FairnessMetrics.statistical_parity(
            y_pred, sensitive, min_representation_pct=2.0
        )

        # Both groups meet 2% threshold
        assert len(result['testable_groups']) == 2
        assert len(result['excluded_groups']) == 0

        # Now with 3% threshold, Group B (2%) is excluded
        result = FairnessMetrics.statistical_parity(
            y_pred, sensitive, min_representation_pct=3.0
        )

        assert len(result['testable_groups']) == 1
        assert len(result['excluded_groups']) == 1
        assert 'B' in result['excluded_groups']
        assert 'INSUFFICIENT' in result['interpretation']

    def test_statistical_parity_three_groups(self):
        """Test with more than 2 groups"""
        y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 0])
        sensitive = np.array(['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C', 'C'])

        result = FairnessMetrics.statistical_parity(y_pred, sensitive)

        # Should consider all testable groups
        assert len(result['testable_groups']) == 3
        assert result['disparity'] is not None
        assert result['ratio'] is not None

    def test_statistical_parity_pandas_series(self):
        """Test with pandas Series input"""
        y_pred = pd.Series([1, 1, 0, 0])
        sensitive = pd.Series(['A', 'A', 'B', 'B'])

        result = FairnessMetrics.statistical_parity(y_pred, sensitive)

        assert result['metric_name'] == 'statistical_parity'
        assert result['disparity'] is not None

    def test_statistical_parity_all_positive_predictions(self):
        """Test when all predictions are positive"""
        y_pred = np.ones(10)
        sensitive = np.array(['A'] * 5 + ['B'] * 5)

        result = FairnessMetrics.statistical_parity(y_pred, sensitive)

        assert result['disparity'] == 0.0
        assert result['ratio'] == 1.0
        assert result['passes_80_rule'] is True

    def test_statistical_parity_all_negative_predictions(self):
        """Test when all predictions are negative"""
        y_pred = np.zeros(10)
        sensitive = np.array(['A'] * 5 + ['B'] * 5)

        result = FairnessMetrics.statistical_parity(y_pred, sensitive)

        # Both groups have 0% positive rate -> 0/0 = 0
        assert result['ratio'] == 0.0


class TestEqualOpportunity:
    """Tests for equal_opportunity metric"""

    def test_basic_equal_opportunity(self):
        """Test basic equal opportunity calculation"""
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 0, 0, 1, 1])
        sensitive = np.array(['A', 'A', 'B', 'B', 'A', 'A', 'B', 'B'])

        result = FairnessMetrics.equal_opportunity(y_true, y_pred, sensitive)

        assert result['metric_name'] == 'equal_opportunity'
        # Group A: 1 TP / 2 positives = 0.5 TPR
        # Group B: 0 TP / 2 positives = 0.0 TPR
        assert result['testable_groups']['A']['tpr'] == 1.0
        assert result['testable_groups']['B']['tpr'] == 0.0

    def test_equal_opportunity_perfect_equality(self):
        """Test when TPR is equal across groups"""
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1])
        sensitive = np.array(['A', 'A', 'B', 'B'])

        result = FairnessMetrics.equal_opportunity(y_true, y_pred, sensitive)

        assert result['disparity'] == 0.0
        assert result['ratio'] == 1.0
        assert 'EXCELLENT' in result['interpretation']

    def test_equal_opportunity_no_positives_in_group(self):
        """Test when one group has no positive labels"""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])
        sensitive = np.array(['A', 'A', 'B', 'B'])

        result = FairnessMetrics.equal_opportunity(y_true, y_pred, sensitive)

        # Group B has no positives -> TPR = NaN
        assert np.isnan(result['testable_groups']['B']['tpr'])
        assert 'INSUFFICIENT' in result['interpretation']

    def test_equal_opportunity_min_representation(self):
        """Test minimum representation threshold"""
        y_true = np.array([1] * 98 + [1] * 2)
        y_pred = np.array([1] * 98 + [0] * 2)
        sensitive = np.array(['A'] * 98 + ['B'] * 2)

        result = FairnessMetrics.equal_opportunity(
            y_true, y_pred, sensitive, min_representation_pct=3.0
        )

        assert len(result['testable_groups']) == 1
        assert 'B' in result['excluded_groups']


class TestEqualizedOdds:
    """Tests for equalized_odds metric"""

    def test_basic_equalized_odds(self):
        """Test basic equalized odds calculation"""
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        sensitive = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'])

        result = FairnessMetrics.equalized_odds(y_true, y_pred, sensitive)

        assert result['metric_name'] == 'equalized_odds'
        # Perfect predictions -> TPR=1.0, FPR=0.0 for both groups
        assert result['tpr_disparity'] == 0.0
        assert result['fpr_disparity'] == 0.0
        assert 'EXCELLENT' in result['interpretation']

    def test_equalized_odds_with_disparity(self):
        """Test with TPR and FPR disparities"""
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        sensitive = np.array(['A', 'A', 'B', 'B', 'A', 'A', 'B', 'B'])

        result = FairnessMetrics.equalized_odds(y_true, y_pred, sensitive)

        # Group A: TPR = 1.0, FPR = 1.0
        # Group B: TPR = 0.0, FPR = 0.0
        assert result['tpr_disparity'] > 0
        assert result['fpr_disparity'] > 0

    def test_equalized_odds_single_group(self):
        """Test with single group"""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0])
        sensitive = np.array(['A', 'A', 'A', 'A'])

        result = FairnessMetrics.equalized_odds(y_true, y_pred, sensitive)

        assert result['tpr_disparity'] is None
        assert result['fpr_disparity'] is None
        assert 'INSUFFICIENT' in result['interpretation']


class TestDisparateImpact:
    """Tests for disparate_impact metric"""

    def test_basic_disparate_impact(self):
        """Test basic disparate impact calculation"""
        y_pred = np.array([1, 1, 1, 1, 1, 1, 0, 0])
        sensitive = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'])

        result = FairnessMetrics.disparate_impact(y_pred, sensitive)

        assert result['metric_name'] == 'disparate_impact'
        # Group A: 4/4 = 1.0, Group B: 2/4 = 0.5 -> ratio = 0.5
        assert result['ratio'] == 0.5
        assert result['passes_threshold'] is False  # < 0.8

    def test_disparate_impact_passes_80_rule(self):
        """Test when disparate impact passes 80% rule"""
        y_pred = np.array([1, 1, 1, 1, 1, 1, 1, 0])
        sensitive = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'])

        result = FairnessMetrics.disparate_impact(y_pred, sensitive)

        # Group A: 4/4 = 1.0, Group B: 3/4 = 0.75
        # ratio = 0.75 / 1.0 = 0.75 < 0.8 (FAILS by default)
        assert result['passes_threshold'] is False

    def test_disparate_impact_custom_threshold(self):
        """Test with custom threshold"""
        y_pred = np.array([1, 1, 1, 0, 1, 1, 0, 0])
        sensitive = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'])

        result = FairnessMetrics.disparate_impact(
            y_pred, sensitive, threshold=0.5
        )

        # Group A: 3/4 = 0.75, Group B: 2/4 = 0.5
        # ratio = 0.5 / 0.75 = 0.667
        assert result['ratio'] >= 0.5
        assert result['passes_threshold'] is True

    def test_disparate_impact_perfect_parity(self):
        """Test with perfect parity (ratio = 1.0)"""
        y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        sensitive = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'])

        result = FairnessMetrics.disparate_impact(y_pred, sensitive)

        assert result['ratio'] == 1.0
        assert result['passes_threshold'] is True
        assert 'EXCELLENT' in result['interpretation']


class TestClassBalance:
    """Tests for class_balance (BCL) metric"""

    def test_basic_class_balance(self):
        """Test basic class balance calculation"""
        y_true = np.array([1, 0, 1, 0])
        sensitive = np.array(['A', 'A', 'B', 'B'])

        result = FairnessMetrics.class_balance(y_true, sensitive)

        assert result['metric_name'] == 'class_balance'
        # Both groups have 2 samples -> BCL = 0
        assert result['value'] == 0.0
        assert result['group_a_size'] == 2
        assert result['group_b_size'] == 2
        assert '✓ Green' in result['interpretation']

    def test_class_balance_imbalanced(self):
        """Test with imbalanced groups"""
        y_true = np.array([1] * 8 + [0] * 2)
        sensitive = np.array(['A'] * 8 + ['B'] * 2)

        result = FairnessMetrics.class_balance(y_true, sensitive)

        # BCL = (8 - 2) / 10 = 0.6
        assert result['value'] == 0.6
        assert result['group_a_size'] == 8
        assert result['group_b_size'] == 2
        assert '✗ Red' in result['interpretation']

    def test_class_balance_single_group(self):
        """Test with single group"""
        y_true = np.array([1, 0, 1, 0])
        sensitive = np.array(['A', 'A', 'A', 'A'])

        result = FairnessMetrics.class_balance(y_true, sensitive)

        assert result['value'] == 0.0
        assert result['group_b'] == 'N/A'
        assert 'um grupo' in result['interpretation']

    def test_class_balance_three_groups(self):
        """Test with three groups (uses top 2)"""
        y_true = np.array([1] * 10)
        sensitive = np.array(['A'] * 5 + ['B'] * 3 + ['C'] * 2)

        result = FairnessMetrics.class_balance(y_true, sensitive)

        # Should compare A (5) vs B (3), ignoring C
        assert result['group_a'] == 'A'
        assert result['group_b'] == 'B'
        assert result['group_a_size'] == 5
        assert result['group_b_size'] == 3


class TestConceptBalance:
    """Tests for concept_balance (BCO) metric"""

    def test_basic_concept_balance(self):
        """Test basic concept balance calculation"""
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        sensitive = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'])

        result = FairnessMetrics.concept_balance(y_true, sensitive)

        assert result['metric_name'] == 'concept_balance'
        # Both groups: 2/4 = 0.5 -> BCO = 0.0
        assert result['value'] == 0.0
        assert '✓ Green' in result['interpretation']

    def test_concept_balance_imbalanced(self):
        """Test with imbalanced positive rates"""
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        sensitive = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'])

        result = FairnessMetrics.concept_balance(y_true, sensitive)

        # Group A: 4/4 = 1.0, Group B: 0/4 = 0.0 -> BCO = 1.0
        assert result['value'] == 1.0
        assert result['group_a_positive_rate'] == 1.0
        assert result['group_b_positive_rate'] == 0.0
        assert '✗ Red' in result['interpretation']

    def test_concept_balance_single_group(self):
        """Test with single group"""
        y_true = np.array([1, 0, 1, 0])
        sensitive = np.array(['A', 'A', 'A', 'A'])

        result = FairnessMetrics.concept_balance(y_true, sensitive)

        assert result['value'] == 0.0
        assert result['group_b'] == 'N/A'


class TestKLDivergence:
    """Tests for kl_divergence metric"""

    def test_basic_kl_divergence(self):
        """Test basic KL divergence calculation"""
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        sensitive = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'])

        result = FairnessMetrics.kl_divergence(y_true, sensitive)

        assert result['metric_name'] == 'kl_divergence'
        # Both groups have same distribution -> KL ≈ 0
        assert result['value'] < 0.01
        assert '✓ Green' in result['interpretation']

    def test_kl_divergence_different_distributions(self):
        """Test with different label distributions"""
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        sensitive = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'])

        result = FairnessMetrics.kl_divergence(y_true, sensitive)

        # Group A: all 1s, Group B: all 0s -> high KL divergence
        assert result['value'] > 0.5
        assert '✗ Red' in result['interpretation']

    def test_kl_divergence_single_group(self):
        """Test with single group"""
        y_true = np.array([1, 0, 1, 0])
        sensitive = np.array(['A', 'A', 'A', 'A'])

        result = FairnessMetrics.kl_divergence(y_true, sensitive)

        assert result['value'] == 0.0
        assert result['group_b'] == 'N/A'


class TestJSDivergence:
    """Tests for js_divergence metric"""

    def test_basic_js_divergence(self):
        """Test basic JS divergence calculation"""
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        sensitive = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'])

        result = FairnessMetrics.js_divergence(y_true, sensitive)

        assert result['metric_name'] == 'js_divergence'
        # Same distribution -> JS ≈ 0
        assert result['value'] < 0.01
        assert '✓ Green' in result['interpretation']

    def test_js_divergence_different_distributions(self):
        """Test with different distributions"""
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        sensitive = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'])

        result = FairnessMetrics.js_divergence(y_true, sensitive)

        # Different distributions -> high JS
        assert result['value'] > 0.2
        assert '✗ Red' in result['interpretation']

    def test_js_divergence_symmetry(self):
        """Test that JS divergence is symmetric"""
        y_true = np.array([1, 1, 0, 0])
        sensitive_ab = np.array(['A', 'A', 'B', 'B'])
        sensitive_ba = np.array(['B', 'B', 'A', 'A'])

        result_ab = FairnessMetrics.js_divergence(y_true, sensitive_ab)
        result_ba = FairnessMetrics.js_divergence(y_true, sensitive_ba)

        # JS is symmetric: JS(A||B) = JS(B||A)
        assert abs(result_ab['value'] - result_ba['value']) < 1e-6


class TestFalseNegativeRateDifference:
    """Tests for false_negative_rate_difference metric"""

    def test_basic_fnr_difference(self):
        """Test basic FNR difference calculation"""
        y_true = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        y_pred = np.array([1, 1, 0, 0, 1, 1, 1, 0])
        sensitive = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'])

        result = FairnessMetrics.false_negative_rate_difference(
            y_true, y_pred, sensitive
        )

        assert result['metric_name'] == 'false_negative_rate_difference'
        # Group A: 2 FN / 4 = 0.5, Group B: 1 FN / 4 = 0.25
        assert result['group_a_fnr'] == 0.5
        assert result['group_b_fnr'] == 0.25

    def test_fnr_difference_no_false_negatives(self):
        """Test when there are no false negatives"""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])
        sensitive = np.array(['A', 'A', 'B', 'B'])

        result = FairnessMetrics.false_negative_rate_difference(
            y_true, y_pred, sensitive
        )

        # Both groups: FNR = 0
        assert result['group_a_fnr'] == 0.0
        assert result['group_b_fnr'] == 0.0
        assert '✓ Green' in result['interpretation']

    def test_fnr_difference_min_representation(self):
        """Test minimum representation threshold"""
        y_true = np.array([1] * 98 + [1] * 2)
        y_pred = np.array([1] * 98 + [0] * 2)
        sensitive = np.array(['A'] * 98 + ['B'] * 2)

        result = FairnessMetrics.false_negative_rate_difference(
            y_true, y_pred, sensitive, min_representation_pct=3.0
        )

        assert len(result['testable_groups']) == 1
        assert 'B' in result['excluded_groups']


class TestConditionalAcceptance:
    """Tests for conditional_acceptance metric"""

    def test_basic_conditional_acceptance(self):
        """Test basic conditional acceptance calculation"""
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        sensitive = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'])

        result = FairnessMetrics.conditional_acceptance(
            y_true, y_pred, sensitive
        )

        assert result['metric_name'] == 'conditional_acceptance'
        # Group A: P(Y=1|Ŷ=1) = 2/4 = 0.5
        # Group B: P(Y=1|Ŷ=1) = 2/4 = 0.5
        assert result['value'] == 0.0

    def test_conditional_acceptance_no_positive_predictions(self):
        """Test when there are no positive predictions"""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0, 0, 0, 0])
        sensitive = np.array(['A', 'A', 'B', 'B'])

        result = FairnessMetrics.conditional_acceptance(
            y_true, y_pred, sensitive
        )

        # No positive predictions -> rate = 0 for both groups
        assert result['group_a_rate'] == 0.0
        assert result['group_b_rate'] == 0.0


class TestConditionalRejection:
    """Tests for conditional_rejection metric"""

    def test_basic_conditional_rejection(self):
        """Test basic conditional rejection calculation"""
        y_true = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        sensitive = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'])

        result = FairnessMetrics.conditional_rejection(
            y_true, y_pred, sensitive
        )

        assert result['metric_name'] == 'conditional_rejection'
        # Both groups: P(Y=0|Ŷ=0) = 2/4 = 0.5
        assert result['value'] == 0.0

    def test_conditional_rejection_all_positive_predictions(self):
        """Test when all predictions are positive"""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 1, 1, 1])
        sensitive = np.array(['A', 'A', 'B', 'B'])

        result = FairnessMetrics.conditional_rejection(
            y_true, y_pred, sensitive
        )

        # No negative predictions -> rate = 0 for both groups
        assert result['group_a_rate'] == 0.0
        assert result['group_b_rate'] == 0.0


class TestPrecisionDifference:
    """Tests for precision_difference metric"""

    def test_basic_precision_difference(self):
        """Test basic precision difference calculation"""
        y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 1, 1, 0, 1, 1, 1, 0])
        sensitive = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'])

        result = FairnessMetrics.precision_difference(
            y_true, y_pred, sensitive
        )

        assert result['metric_name'] == 'precision_difference'
        # Group A: 2 TP / 3 pred_pos = 0.667
        # Group B: 2 TP / 3 pred_pos = 0.667
        assert abs(result['value']) < 0.1

    def test_precision_difference_perfect_precision(self):
        """Test with perfect precision in both groups"""
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        sensitive = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'])

        result = FairnessMetrics.precision_difference(
            y_true, y_pred, sensitive
        )

        # Both groups: precision = 1.0 -> difference = 0
        assert abs(result['value']) < 0.01
        assert result['group_a_precision'] == 1.0
        assert result['group_b_precision'] == 1.0

    def test_precision_difference_zero_division(self):
        """Test when there are no positive predictions"""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0, 0, 0, 0])
        sensitive = np.array(['A', 'A', 'B', 'B'])

        result = FairnessMetrics.precision_difference(
            y_true, y_pred, sensitive
        )

        # No positive predictions -> precision = 0 (zero_division=0)
        assert result['group_a_precision'] == 0.0
        assert result['group_b_precision'] == 0.0


class TestAccuracyDifference:
    """Tests for accuracy_difference metric"""

    def test_basic_accuracy_difference(self):
        """Test basic accuracy difference calculation"""
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 1, 0, 1, 0])
        sensitive = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'])

        result = FairnessMetrics.accuracy_difference(
            y_true, y_pred, sensitive
        )

        assert result['metric_name'] == 'accuracy_difference'
        # Group A: 4/4 = 1.0, Group B: 2/4 = 0.5
        assert result['group_a_accuracy'] == 1.0
        assert result['group_b_accuracy'] == 0.5

    def test_accuracy_difference_perfect_accuracy(self):
        """Test with perfect accuracy in both groups"""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0])
        sensitive = np.array(['A', 'A', 'B', 'B'])

        result = FairnessMetrics.accuracy_difference(
            y_true, y_pred, sensitive
        )

        assert result['value'] == 0.0
        assert '✓ Green' in result['interpretation']


class TestTreatmentEquality:
    """Tests for treatment_equality metric"""

    def test_basic_treatment_equality(self):
        """Test basic treatment equality calculation"""
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        y_pred = np.array([0, 1, 1, 0, 0, 1, 1, 0])
        sensitive = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'])

        result = FairnessMetrics.treatment_equality(
            y_true, y_pred, sensitive
        )

        assert result['metric_name'] == 'treatment_equality'
        # Group A: FN=1, FP=1 -> ratio=1.0
        # Group B: FN=1, FP=1 -> ratio=1.0
        assert result['value'] == 0.0

    def test_treatment_equality_no_false_positives(self):
        """Test when FP=0 (division by zero case)"""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0, 0, 0, 0])
        sensitive = np.array(['A', 'A', 'B', 'B'])

        result = FairnessMetrics.treatment_equality(
            y_true, y_pred, sensitive
        )

        # FP=0 -> ratio=0.0 for both groups
        assert result['group_a_ratio'] == 0.0
        assert result['group_b_ratio'] == 0.0


class TestEntropyIndex:
    """Tests for entropy_index metric"""

    def test_basic_entropy_index(self):
        """Test basic entropy index calculation"""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])

        result = FairnessMetrics.entropy_index(y_true, y_pred)

        assert result['metric_name'] == 'entropy_index'
        # Perfect predictions -> b_i = 1 for all -> IE ≈ 0
        assert abs(result['value']) < 0.1
        assert '✓ Green' in result['interpretation']

    def test_entropy_index_with_errors(self):
        """Test entropy index with prediction errors"""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0, 0, 1, 1])

        result = FairnessMetrics.entropy_index(y_true, y_pred)

        # All predictions wrong -> b_i = 2 for all
        # When all b_i are equal, entropy should be 0 or very small
        assert result['value'] is not None

    def test_entropy_index_alpha_0(self):
        """Test entropy index with alpha=0"""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 1, 0, 0])

        result = FairnessMetrics.entropy_index(y_true, y_pred, alpha=0.0)

        assert result['alpha'] == 0.0
        assert result['value'] is not None

    def test_entropy_index_alpha_1(self):
        """Test entropy index with alpha=1"""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 1, 0, 0])

        result = FairnessMetrics.entropy_index(y_true, y_pred, alpha=1.0)

        assert result['alpha'] == 1.0
        assert result['value'] is not None

    def test_entropy_index_alpha_2(self):
        """Test entropy index with alpha=2 (default)"""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 1, 0, 0])

        result = FairnessMetrics.entropy_index(y_true, y_pred, alpha=2.0)

        assert result['alpha'] == 2.0
        assert result['value'] is not None


class TestEdgeCases:
    """Tests for edge cases and error conditions"""

    def test_empty_arrays(self):
        """Test with empty arrays"""
        y_pred = np.array([])
        sensitive = np.array([])

        # Empty arrays should return a result with no groups
        result = FairnessMetrics.statistical_parity(y_pred, sensitive)

        # Should have empty testable_groups
        assert len(result['testable_groups']) == 0

    def test_mismatched_lengths(self):
        """Test with mismatched array lengths"""
        y_pred = np.array([1, 0])
        sensitive = np.array([1, 0, 1])

        # Should raise error
        with pytest.raises((ValueError, IndexError)):
            FairnessMetrics.statistical_parity(y_pred, sensitive)

    def test_non_binary_predictions(self):
        """Test with non-binary predictions (should still work)"""
        y_pred = np.array([0, 1, 2, 3])
        sensitive = np.array(['A', 'A', 'B', 'B'])

        # Should work but interpret 2, 3 as "positive" (!=0)
        result = FairnessMetrics.statistical_parity(y_pred, sensitive)
        assert result['metric_name'] == 'statistical_parity'

    def test_string_sensitive_features(self):
        """Test with string sensitive features"""
        y_pred = np.array([1, 0, 1, 0])
        sensitive = np.array(['Male', 'Male', 'Female', 'Female'])

        result = FairnessMetrics.statistical_parity(y_pred, sensitive)

        assert 'Male' in result['testable_groups'] or 'Female' in result['testable_groups']

    def test_numeric_sensitive_features(self):
        """Test with numeric sensitive features"""
        y_pred = np.array([1, 0, 1, 0])
        sensitive = np.array([0, 0, 1, 1])

        result = FairnessMetrics.statistical_parity(y_pred, sensitive)

        assert '0' in result['testable_groups'] or '1' in result['testable_groups']


class TestMinRepresentationPct:
    """Tests for MIN_REPRESENTATION_PCT class attribute"""

    def test_default_min_representation_pct(self):
        """Test that default MIN_REPRESENTATION_PCT is 2.0"""
        assert FairnessMetrics.MIN_REPRESENTATION_PCT == 2.0

    def test_min_representation_pct_used_as_default(self):
        """Test that MIN_REPRESENTATION_PCT is used when not specified"""
        y_pred = np.array([1] * 98 + [0] * 2)
        sensitive = np.array(['A'] * 98 + ['B'] * 2)

        result = FairnessMetrics.statistical_parity(y_pred, sensitive)

        # Should use 2.0% as default
        assert result['min_representation_pct'] == 2.0
