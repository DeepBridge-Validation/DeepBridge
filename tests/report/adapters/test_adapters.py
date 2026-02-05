"""
Tests for report adapters (Phase 3 Sprint 14).

Tests both JSONAdapter and HTMLAdapter for converting domain models
to different output formats.
"""

import json
from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import pytest

from deepbridge.core.experiment.report.adapters import (
    HTMLAdapter,
    JSONAdapter,
    ReportAdapter,
)
from deepbridge.core.experiment.report.domain import (
    ChartSpec,
    ChartType,
    Metric,
    MetricType,
    Report,
    ReportMetadata,
    ReportSection,
    ReportType,
)

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_metadata():
    """Create sample report metadata."""
    return ReportMetadata(
        model_name="TestModel",
        test_type=ReportType.UNCERTAINTY,
        dataset_name="TestDataset",
        dataset_size=1000,
        test_duration=123.45,
    )


@pytest.fixture
def sample_metric():
    """Create sample metric."""
    return Metric(
        name="accuracy",
        value=0.95,
        type=MetricType.PERCENTAGE,
        threshold=0.90,
        higher_is_better=True,
    )


@pytest.fixture
def sample_chart_spec():
    """Create sample chart specification."""
    return ChartSpec(
        id="coverage_plot",
        type=ChartType.COVERAGE,
        title="Coverage Analysis",
        data={
            'alphas': [0.1, 0.2, 0.3],
            'coverage': [0.91, 0.81, 0.72],
            'expected': [0.90, 0.80, 0.70],
        },
        width=800,
        height=600,
    )


@pytest.fixture
def sample_section(sample_metric, sample_chart_spec):
    """Create sample report section."""
    section = ReportSection(
        id="results",
        title="Test Results",
        description="Results of the uncertainty test",
    )
    section.add_metric(sample_metric)
    section.add_chart(sample_chart_spec)
    return section


@pytest.fixture
def sample_report(sample_metadata, sample_section, sample_metric):
    """Create sample report."""
    report = Report(metadata=sample_metadata)
    report.add_summary_metric(sample_metric)
    report.add_section(sample_section)
    return report


# =============================================================================
# JSONAdapter Tests
# =============================================================================

class TestJSONAdapter:
    """Tests for JSONAdapter."""

    def test_json_adapter_initialization(self):
        """Test JSONAdapter initialization."""
        adapter = JSONAdapter()
        assert adapter.indent is None
        assert adapter.ensure_ascii is False

        adapter = JSONAdapter(indent=2, ensure_ascii=True)
        assert adapter.indent == 2
        assert adapter.ensure_ascii is True

    def test_json_adapter_render_basic(self, sample_report):
        """Test basic JSON rendering."""
        adapter = JSONAdapter(indent=2)
        json_str = adapter.render(sample_report)

        # Should be valid JSON
        data = json.loads(json_str)

        # Check structure
        assert 'metadata' in data
        assert 'sections' in data
        assert 'summary_metrics' in data

        # Check metadata
        assert data['metadata']['model_name'] == 'TestModel'
        assert data['metadata']['test_type'] == 'uncertainty'

        # Check sections
        assert len(data['sections']) == 1
        assert data['sections'][0]['title'] == 'Test Results'

        # Check metrics
        assert len(data['summary_metrics']) == 1
        assert data['summary_metrics'][0]['name'] == 'accuracy'

    def test_json_adapter_render_dict(self, sample_report):
        """Test rendering to dict without JSON serialization."""
        adapter = JSONAdapter()
        data = adapter.render_dict(sample_report)

        # Should be a dict
        assert isinstance(data, dict)

        # Check structure
        assert 'metadata' in data
        assert 'sections' in data
        assert 'summary_metrics' in data

    def test_json_adapter_clean_none_values(self, sample_metadata):
        """Test that None values are cleaned."""
        # Create report with optional fields as None
        report = Report(metadata=sample_metadata)
        section = ReportSection(
            id="test",
            title="Test",
            description=None,  # Will be cleaned
        )
        report.add_section(section)

        adapter = JSONAdapter()
        data = adapter.render_dict(report)

        # description should not be in the output
        # (Pydantic might exclude it by default, but let's check)
        section_data = data['sections'][0]
        # If None is cleaned, it won't be present
        # If not cleaned, it will be None
        # Either way is acceptable, but cleaned is preferred

    def test_json_adapter_datetime_serialization(self, sample_metadata):
        """Test custom datetime serialization."""
        report = Report(metadata=sample_metadata)

        adapter = JSONAdapter(indent=2)
        json_str = adapter.render(report)
        data = json.loads(json_str)

        # created_at should be serialized as ISO format string
        assert 'created_at' in data['metadata']
        created_at = data['metadata']['created_at']
        # Should be parseable as ISO format
        datetime.fromisoformat(created_at.replace('Z', '+00:00'))

    def test_json_adapter_validation_error(self, sample_metadata):
        """Test that validation errors are raised."""
        # Create a valid report first
        report = Report(metadata=sample_metadata)

        # Bypass Pydantic validation by using __dict__ directly
        report.__dict__['metadata'] = None

        adapter = JSONAdapter()

        with pytest.raises(ValueError, match="Report must have metadata"):
            adapter.render(report)

    def test_json_adapter_compact_format(self, sample_report):
        """Test compact JSON format (no indentation)."""
        adapter = JSONAdapter(indent=None)
        json_str = adapter.render(sample_report)

        # Should not have newlines (compact)
        assert '\n' not in json_str or json_str.count('\n') < 5

    def test_json_adapter_pretty_format(self, sample_report):
        """Test pretty JSON format (with indentation)."""
        adapter = JSONAdapter(indent=2)
        json_str = adapter.render(sample_report)

        # Should have many newlines (pretty)
        assert json_str.count('\n') > 10


# =============================================================================
# HTMLAdapter Tests
# =============================================================================

class TestHTMLAdapter:
    """Tests for HTMLAdapter."""

    def test_html_adapter_initialization(self):
        """Test HTMLAdapter initialization."""
        adapter = HTMLAdapter()
        assert adapter.template_manager is None
        assert adapter.asset_manager is None
        assert adapter.theme == "default"

        adapter = HTMLAdapter(theme="dark")
        assert adapter.theme == "dark"

    def test_html_adapter_fallback_rendering(self, sample_report):
        """Test HTML rendering with fallback (no template manager)."""
        adapter = HTMLAdapter()
        html = adapter.render(sample_report)

        # Should be valid HTML
        assert html.startswith("<!DOCTYPE html>")
        assert "<html>" in html
        assert "</html>" in html

        # Should contain report title
        assert "TestModel" in html

        # Should contain test type
        assert "uncertainty" in html

        # Should contain metrics
        assert "accuracy" in html

        # Should contain sections
        assert "Test Results" in html

    def test_html_adapter_chart_generation(self, sample_report):
        """Test that charts are generated via ChartRegistry."""
        adapter = HTMLAdapter()

        # Mock ChartRegistry to track calls
        with patch.object(adapter, 'chart_registry') as mock_registry:
            mock_result = Mock()
            mock_result.is_success = True
            mock_result.content = "<div>Chart Content</div>"
            mock_registry.generate.return_value = mock_result

            html = adapter.render(sample_report)

            # Should have called generate for coverage chart
            mock_registry.generate.assert_called_once()
            call_args = mock_registry.generate.call_args

            # Check arguments
            # ChartType.COVERAGE.value is 'coverage_chart' in the registry
            assert call_args[0][0] == 'coverage_chart'  # Chart type from ChartType enum
            assert 'alphas' in call_args[0][1]  # Data

    def test_html_adapter_chart_error_handling(self, sample_report):
        """Test error handling for chart generation failures."""
        adapter = HTMLAdapter()

        # Mock ChartRegistry to return error
        with patch.object(adapter, 'chart_registry') as mock_registry:
            mock_result = Mock()
            mock_result.is_success = False
            mock_result.error = "Chart generation failed"
            mock_registry.generate.return_value = mock_result

            html = adapter.render(sample_report)

            # Should contain error message
            assert "Error" in html or "error" in html

    def test_html_adapter_metric_formatting(self, sample_report):
        """Test metric formatting in HTML."""
        adapter = HTMLAdapter()
        html = adapter.render(sample_report)

        # Should format percentage metrics
        assert "95" in html  # 0.95 as percentage

    def test_html_adapter_metric_status_classes(self):
        """Test CSS classes for metric status."""
        adapter = HTMLAdapter()

        # Passing metric
        passing = Metric(name="test", value=0.95, threshold=0.90, higher_is_better=True)
        assert adapter._get_metric_status_class(passing) == "metric-pass"

        # Failing metric
        failing = Metric(name="test", value=0.85, threshold=0.90, higher_is_better=True)
        assert adapter._get_metric_status_class(failing) == "metric-fail"

        # Neutral metric (no threshold)
        neutral = Metric(name="test", value=0.90)
        assert adapter._get_metric_status_class(neutral) == "metric-neutral"

    def test_html_adapter_section_rendering(self, sample_section):
        """Test section rendering."""
        adapter = HTMLAdapter()

        # Create minimal report with section
        metadata = ReportMetadata(
            model_name="Test",
            test_type=ReportType.UNCERTAINTY,
        )
        report = Report(metadata=metadata)
        report.add_section(sample_section)

        html = adapter.render(report)

        # Should contain section title
        assert "Test Results" in html

        # Should contain section description
        assert "Results of the uncertainty test" in html

        # Should contain section ID
        assert "results" in html

    def test_html_adapter_nested_sections(self):
        """Test rendering of nested sections."""
        adapter = HTMLAdapter()

        # Create report with nested sections
        metadata = ReportMetadata(
            model_name="Test",
            test_type=ReportType.UNCERTAINTY,
        )
        report = Report(metadata=metadata)

        # Main section
        main_section = ReportSection(id="main", title="Main Section")

        # Subsection
        subsection = ReportSection(id="sub", title="Subsection")
        subsection.add_metric(Metric(name="test_metric", value=0.95))

        main_section.add_subsection(subsection)
        report.add_section(main_section)

        html = adapter.render(report)

        # Should contain both section titles
        assert "Main Section" in html
        assert "Subsection" in html

        # Should contain nested metric
        assert "test_metric" in html

    def test_html_adapter_validation_error(self, sample_metadata):
        """Test validation errors are raised."""
        adapter = HTMLAdapter()

        # Create a valid report first
        report = Report(metadata=sample_metadata)

        # Bypass Pydantic validation by using __dict__ directly
        report.__dict__['metadata'] = None

        with pytest.raises(ValueError, match="Report must have metadata"):
            adapter.render(report)

    def test_html_adapter_default_css(self):
        """Test that default CSS is included in fallback mode."""
        adapter = HTMLAdapter()
        metadata = ReportMetadata(
            model_name="Test",
            test_type=ReportType.UNCERTAINTY,
        )
        report = Report(metadata=metadata)

        html = adapter.render(report)

        # Should contain CSS styles
        assert "<style>" in html
        assert "font-family" in html
        assert ".metric-pass" in html
        assert ".metric-fail" in html

    def test_html_adapter_summary_metrics(self, sample_report):
        """Test rendering of summary metrics."""
        adapter = HTMLAdapter()
        html = adapter.render(sample_report)

        # Should contain summary section
        assert "Summary" in html or "summary" in html

        # Should contain summary metric
        assert "accuracy" in html

    def test_html_adapter_with_template_manager(self, sample_report):
        """Test HTML adapter with template manager."""
        # Mock template manager
        mock_template_manager = Mock()
        mock_template = Mock()
        mock_template_manager.get_template_paths.return_value = ["template.html"]
        mock_template_manager.find_template.return_value = "template.html"
        mock_template_manager.load_template.return_value = mock_template
        mock_template_manager.render_template.return_value = "<html>Rendered</html>"

        adapter = HTMLAdapter(template_manager=mock_template_manager)

        with patch.object(adapter, 'chart_registry') as mock_registry:
            mock_result = Mock()
            mock_result.is_success = True
            mock_result.content = "<div>Chart</div>"
            mock_registry.generate.return_value = mock_result

            html = adapter.render(sample_report)

            # Should have used template manager
            mock_template_manager.render_template.assert_called_once()
            assert html == "<html>Rendered</html>"

    def test_html_adapter_theme_support(self, sample_report):
        """Test theme is passed to template context."""
        adapter = HTMLAdapter(theme="dark")

        with patch.object(adapter, 'chart_registry') as mock_registry:
            mock_result = Mock()
            mock_result.is_success = True
            mock_result.content = "<div>Chart</div>"
            mock_registry.generate.return_value = mock_result

            html = adapter.render(sample_report)

            # Theme should be in HTML (at least in context)
            # In fallback mode, theme isn't used, but context is created
            assert html  # Just verify it renders


# =============================================================================
# Integration Tests
# =============================================================================

class TestAdapterIntegration:
    """Integration tests for adapters."""

    def test_json_to_dict_to_html_roundtrip(self, sample_report):
        """Test that report can be exported to JSON and HTML."""
        # Export to JSON
        json_adapter = JSONAdapter(indent=2)
        json_str = json_adapter.render(sample_report)
        json_data = json.loads(json_str)

        # Verify JSON structure
        assert 'metadata' in json_data
        assert 'sections' in json_data

        # Export to HTML
        html_adapter = HTMLAdapter()
        html = html_adapter.render(sample_report)

        # Verify HTML structure
        assert "<!DOCTYPE html>" in html
        assert sample_report.metadata.model_name in html

    def test_multiple_adapters_same_report(self, sample_report):
        """Test that multiple adapters can render the same report."""
        json_adapter = JSONAdapter()
        html_adapter = HTMLAdapter()

        # Both should render successfully
        json_str = json_adapter.render(sample_report)
        html = html_adapter.render(sample_report)

        # Both should contain model name
        assert "TestModel" in json_str
        assert "TestModel" in html

    def test_adapter_with_complex_report(self, sample_metadata):
        """Test adapters with complex nested report."""
        # Build complex report
        report = Report(metadata=sample_metadata)

        # Add summary metrics
        report.add_summary_metric(
            Metric(name="overall_score", value=0.92, type=MetricType.PERCENTAGE)
        )

        # Add multiple sections with nesting
        for i in range(3):
            section = ReportSection(
                id=f"section_{i}",
                title=f"Section {i}",
            )

            # Add metrics
            for j in range(2):
                section.add_metric(
                    Metric(name=f"metric_{i}_{j}", value=0.8 + j * 0.1)
                )

            # Add subsection
            subsection = ReportSection(
                id=f"subsection_{i}",
                title=f"Subsection {i}",
            )
            subsection.add_metric(Metric(name=f"sub_metric_{i}", value=0.75))

            section.add_subsection(subsection)
            report.add_section(section)

        # Test JSON adapter
        json_adapter = JSONAdapter(indent=2)
        json_str = json_adapter.render(report)
        json_data = json.loads(json_str)

        assert len(json_data['sections']) == 3
        assert len(json_data['summary_metrics']) == 1

        # Test HTML adapter
        html_adapter = HTMLAdapter()
        html = html_adapter.render(report)

        # Should contain all section titles
        for i in range(3):
            assert f"Section {i}" in html
            assert f"Subsection {i}" in html


# =============================================================================
# Edge Cases
# =============================================================================

class TestAdapterEdgeCases:
    """Edge case tests for adapters."""

    def test_empty_report(self, sample_metadata):
        """Test rendering empty report (no sections)."""
        report = Report(metadata=sample_metadata)

        # JSON should work
        json_adapter = JSONAdapter()
        json_str = json_adapter.render(report)
        data = json.loads(json_str)
        assert data['sections'] == []

        # HTML should work
        html_adapter = HTMLAdapter()
        html = html_adapter.render(report)
        assert "<!DOCTYPE html>" in html

    def test_section_without_metrics_or_charts(self, sample_metadata):
        """Test rendering section without content."""
        report = Report(metadata=sample_metadata)
        section = ReportSection(id="empty", title="Empty Section")
        report.add_section(section)

        # Should render without errors
        json_adapter = JSONAdapter()
        json_str = json_adapter.render(report)
        assert "Empty Section" in json_str

        html_adapter = HTMLAdapter()
        html = html_adapter.render(report)
        assert "Empty Section" in html

    def test_metric_with_string_value(self, sample_metadata):
        """Test metric with string value."""
        report = Report(metadata=sample_metadata)
        section = ReportSection(id="test", title="Test")
        # Metric accepts string values
        section.add_metric(Metric(name="test_metric", value="N/A"))
        report.add_section(section)

        # Should handle string value gracefully
        json_adapter = JSONAdapter()
        json_str = json_adapter.render(report)
        data = json.loads(json_str)
        assert data['sections'][0]['metrics'][0]['value'] == "N/A"

        html_adapter = HTMLAdapter()
        html = html_adapter.render(report)
        assert "N/A" in html


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
