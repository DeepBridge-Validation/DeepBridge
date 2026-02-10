"""
Comprehensive tests for chart_mapper utility.

This test suite validates:
1. ensure_chart_mappings - chart alias mapping functionality
2. Bidirectional mappings for reliability charts
3. Bidirectional mappings for bandwidth charts
4. Bidirectional mappings for model comparison charts
5. Edge cases and error handling

Coverage Target: ~95%+
"""

import pytest
from unittest.mock import patch, MagicMock

from deepbridge.core.experiment.report.chart_mapper import ensure_chart_mappings


# ==================== Basic Functionality Tests ====================


class TestEnsureChartMappings:
    """Tests for ensure_chart_mappings function"""

    def test_empty_dict_returns_empty_dict(self):
        """Test with empty dictionary"""
        result = ensure_chart_mappings({})

        assert result == {}

    def test_preserves_original_charts(self):
        """Test that original charts are preserved"""
        charts = {'chart1': 'content1', 'chart2': 'content2'}

        result = ensure_chart_mappings(charts)

        assert 'chart1' in result
        assert 'chart2' in result
        assert result['chart1'] == 'content1'
        assert result['chart2'] == 'content2'

    def test_does_not_modify_input_dict(self):
        """Test that input dictionary is not modified"""
        charts = {'reliability_distribution': 'content'}
        original_len = len(charts)

        result = ensure_chart_mappings(charts)

        # Original should be unchanged
        assert len(charts) == original_len
        # Result should have aliases added
        assert len(result) > len(charts)


# ==================== Reliability Chart Mappings ====================


class TestReliabilityChartMappings:
    """Tests for reliability chart mappings"""

    def test_reliability_distribution_adds_aliases(self):
        """Test reliability_distribution gets feature_reliability and reliability_analysis aliases"""
        charts = {'reliability_distribution': 'content'}

        result = ensure_chart_mappings(charts)

        assert 'feature_reliability' in result
        assert 'reliability_analysis' in result
        assert result['feature_reliability'] == 'content'
        assert result['reliability_analysis'] == 'content'

    def test_feature_reliability_adds_aliases(self):
        """Test feature_reliability gets reliability_distribution and reliability_analysis aliases"""
        charts = {'feature_reliability': 'content'}

        result = ensure_chart_mappings(charts)

        assert 'reliability_distribution' in result
        assert 'reliability_analysis' in result

    def test_reliability_analysis_adds_aliases(self):
        """Test reliability_analysis gets other reliability aliases"""
        charts = {'reliability_analysis': 'content'}

        result = ensure_chart_mappings(charts)

        assert 'feature_reliability' in result
        assert 'reliability_distribution' in result


# ==================== Bandwidth Chart Mappings ====================


class TestBandwidthChartMappings:
    """Tests for bandwidth chart mappings"""

    def test_marginal_bandwidth_adds_aliases(self):
        """Test marginal_bandwidth gets interval width aliases"""
        charts = {'marginal_bandwidth': 'content'}

        result = ensure_chart_mappings(charts)

        assert 'interval_widths_comparison' in result
        assert 'width_distribution' in result
        assert 'interval_widths_boxplot' in result

    def test_interval_widths_comparison_adds_aliases(self):
        """Test interval_widths_comparison gets bandwidth aliases"""
        charts = {'interval_widths_comparison': 'content'}

        result = ensure_chart_mappings(charts)

        assert 'marginal_bandwidth' in result
        assert 'width_distribution' in result
        assert 'interval_widths_boxplot' in result

    def test_width_distribution_adds_aliases(self):
        """Test width_distribution gets interval width aliases"""
        charts = {'width_distribution': 'content'}

        result = ensure_chart_mappings(charts)

        assert 'marginal_bandwidth' in result
        assert 'interval_widths_comparison' in result

    def test_interval_widths_boxplot_adds_aliases(self):
        """Test interval_widths_boxplot gets bandwidth aliases"""
        charts = {'interval_widths_boxplot': 'content'}

        result = ensure_chart_mappings(charts)

        assert 'marginal_bandwidth' in result
        assert 'interval_widths_comparison' in result
        assert 'width_distribution' in result


# ==================== Model Comparison Mappings ====================


class TestModelComparisonMappings:
    """Tests for model comparison chart mappings"""

    def test_model_comparison_adds_aliases(self):
        """Test model_comparison gets metrics comparison aliases"""
        charts = {'model_comparison': 'content'}

        result = ensure_chart_mappings(charts)

        assert 'model_metrics_comparison' in result
        assert 'model_comparison_chart' in result
        assert 'model_metrics' in result

    def test_model_metrics_comparison_adds_aliases(self):
        """Test model_metrics_comparison gets model comparison aliases"""
        charts = {'model_metrics_comparison': 'content'}

        result = ensure_chart_mappings(charts)

        assert 'model_comparison' in result
        assert 'model_comparison_chart' in result
        assert 'model_metrics' in result

    def test_model_comparison_chart_adds_aliases(self):
        """Test model_comparison_chart gets model metrics aliases"""
        charts = {'model_comparison_chart': 'content'}

        result = ensure_chart_mappings(charts)

        assert 'model_comparison' in result
        assert 'model_metrics_comparison' in result

    def test_model_metrics_adds_aliases(self):
        """Test model_metrics gets model comparison aliases"""
        charts = {'model_metrics': 'content'}

        result = ensure_chart_mappings(charts)

        assert 'model_comparison' in result
        assert 'model_metrics_comparison' in result
        assert 'model_comparison_chart' in result


# ==================== Other Charts ====================


class TestOtherCharts:
    """Tests for other charts without aliases"""

    def test_performance_gap_by_alpha_no_aliases(self):
        """Test performance_gap_by_alpha has no aliases"""
        charts = {'performance_gap_by_alpha': 'content'}

        result = ensure_chart_mappings(charts)

        # Should only have the original chart
        assert len(result) == 1
        assert 'performance_gap_by_alpha' in result

    def test_coverage_vs_expected_no_aliases(self):
        """Test coverage_vs_expected has no aliases"""
        charts = {'coverage_vs_expected': 'content'}

        result = ensure_chart_mappings(charts)

        assert len(result) == 1

    def test_width_vs_coverage_no_aliases(self):
        """Test width_vs_coverage has no aliases"""
        charts = {'width_vs_coverage': 'content'}

        result = ensure_chart_mappings(charts)

        assert len(result) == 1

    def test_uncertainty_metrics_no_aliases(self):
        """Test uncertainty_metrics has no aliases"""
        charts = {'uncertainty_metrics': 'content'}

        result = ensure_chart_mappings(charts)

        assert len(result) == 1


# ==================== Multiple Charts ====================


class TestMultipleCharts:
    """Tests with multiple charts"""

    def test_multiple_charts_with_different_aliases(self):
        """Test multiple charts each get their own aliases"""
        charts = {
            'reliability_distribution': 'rel_content',
            'marginal_bandwidth': 'bw_content'
        }

        result = ensure_chart_mappings(charts)

        # Should have both original charts
        assert result['reliability_distribution'] == 'rel_content'
        assert result['marginal_bandwidth'] == 'bw_content'

        # Should have reliability aliases
        assert result['feature_reliability'] == 'rel_content'

        # Should have bandwidth aliases
        assert result['interval_widths_comparison'] == 'bw_content'

    def test_mixed_charts_with_and_without_aliases(self):
        """Test mix of charts with and without aliases"""
        charts = {
            'reliability_distribution': 'content1',
            'performance_gap_by_alpha': 'content2'
        }

        result = ensure_chart_mappings(charts)

        # reliability_distribution should have aliases
        assert 'feature_reliability' in result

        # performance_gap_by_alpha should not have aliases
        assert len([k for k in result.keys() if 'performance' in k]) == 1


# ==================== Existing Alias Handling ====================


class TestExistingAliasHandling:
    """Tests for handling existing aliases"""

    def test_does_not_overwrite_existing_alias(self):
        """Test that existing alias is not overwritten"""
        charts = {
            'reliability_distribution': 'content1',
            'feature_reliability': 'existing_content'
        }

        result = ensure_chart_mappings(charts)

        # Existing alias should not be overwritten
        assert result['feature_reliability'] == 'existing_content'

    def test_both_original_and_alias_present(self):
        """Test when both original chart and alias already exist"""
        charts = {
            'reliability_distribution': 'content1',
            'feature_reliability': 'content2',
            'reliability_analysis': 'content3'
        }

        result = ensure_chart_mappings(charts)

        # All should be preserved
        assert result['reliability_distribution'] == 'content1'
        assert result['feature_reliability'] == 'content2'
        assert result['reliability_analysis'] == 'content3'


# ==================== Logging Tests ====================


class TestLogging:
    """Tests for logging behavior"""

    @patch('deepbridge.core.experiment.report.chart_mapper.logger')
    def test_logs_alias_additions(self, mock_logger):
        """Test that alias additions are logged"""
        charts = {'reliability_distribution': 'content'}

        ensure_chart_mappings(charts)

        # Should log info about added aliases
        assert mock_logger.info.called

    @patch('deepbridge.core.experiment.report.chart_mapper.logger')
    def test_logs_chart_names_after_mapping(self, mock_logger):
        """Test that final chart names are logged"""
        charts = {'reliability_distribution': 'content'}

        ensure_chart_mappings(charts)

        # Should log chart names after mapping
        calls = [str(call) for call in mock_logger.info.call_args_list]
        assert any('Chart names after mapping' in call for call in calls)

    @patch('deepbridge.core.experiment.report.chart_mapper.logger')
    def test_no_logging_for_empty_charts(self, mock_logger):
        """Test no chart names logged for empty input"""
        ensure_chart_mappings({})

        # Should not log chart names for empty dict
        calls = [str(call) for call in mock_logger.info.call_args_list]
        chart_name_logs = [c for c in calls if 'Chart names after mapping' in c]
        assert len(chart_name_logs) == 0


# ==================== Integration Tests ====================


class TestIntegration:
    """Integration tests for complete workflows"""

    def test_full_workflow_all_chart_types(self):
        """Test complete workflow with all chart types"""
        charts = {
            'reliability_distribution': 'rel',
            'marginal_bandwidth': 'bw',
            'model_comparison': 'comp',
            'performance_gap_by_alpha': 'perf'
        }

        result = ensure_chart_mappings(charts)

        # Verify all originals are present
        assert all(k in result for k in charts.keys())

        # Verify reliability aliases
        assert 'feature_reliability' in result
        assert 'reliability_analysis' in result

        # Verify bandwidth aliases
        assert 'interval_widths_comparison' in result
        assert 'width_distribution' in result

        # Verify model comparison aliases
        assert 'model_metrics_comparison' in result
        assert 'model_comparison_chart' in result

        # Verify chart without aliases is unchanged
        assert result['performance_gap_by_alpha'] == 'perf'


# ==================== Edge Cases ====================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_chart_with_special_characters(self):
        """Test chart names with special characters"""
        charts = {'chart-name_123': 'content'}

        result = ensure_chart_mappings(charts)

        assert 'chart-name_123' in result

    def test_chart_content_types(self):
        """Test various chart content types"""
        charts = {
            'reliability_distribution': {'type': 'dict'},
            'marginal_bandwidth': ['list', 'content'],
            'performance_gap_by_alpha': 123
        }

        result = ensure_chart_mappings(charts)

        # Content should be preserved regardless of type
        assert isinstance(result['reliability_distribution'], dict)
        assert isinstance(result['marginal_bandwidth'], list)
        assert isinstance(result['performance_gap_by_alpha'], int)

    def test_large_number_of_charts(self):
        """Test with large number of charts"""
        charts = {f'chart_{i}': f'content_{i}' for i in range(100)}

        result = ensure_chart_mappings(charts)

        # All charts should be preserved
        assert len(result) >= len(charts)
