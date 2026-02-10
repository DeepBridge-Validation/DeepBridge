"""
Tests for deepbridge.synthetic.metrics.similarity.__init__.py

Tests that all functions are successfully imported from similarity module.
"""

import pytest


class TestSuccessfulImports:
    """Test that functions exist when modules are available (normal case)."""

    def test_privacy_functions_exist(self):
        """Test that privacy functions are successfully imported."""
        import deepbridge.synthetic.metrics.similarity as sim

        # Should have the actual functions, not fallbacks
        assert hasattr(sim, 'calculate_privacy_risk')
        assert hasattr(sim, 'assess_k_anonymity')
        assert hasattr(sim, 'assess_l_diversity')

    def test_visualization_functions_exist(self):
        """Test that visualization functions are successfully imported."""
        import deepbridge.synthetic.metrics.similarity as sim

        # Should have the actual functions, not fallbacks
        assert hasattr(sim, 'visualize_data_comparison')
        assert hasattr(sim, 'plot_distribution_comparison')
        assert hasattr(sim, 'plot_correlation_comparison')
        assert hasattr(sim, 'plot_privacy_risk')
        assert hasattr(sim, 'plot_attribute_distributions')

    def test_core_functions_exist(self):
        """Test that core similarity functions exist."""
        import deepbridge.synthetic.metrics.similarity as sim

        assert hasattr(sim, 'calculate_similarity')
        assert hasattr(sim, 'filter_by_similarity')
        assert hasattr(sim, 'enhance_synthetic_data_quality')
        assert hasattr(sim, 'detect_duplicates')
        assert hasattr(sim, 'calculate_diversity')
        assert hasattr(sim, 'calculate_distribution_divergence')
        assert hasattr(sim, 'evaluate_pairwise_correlations')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
