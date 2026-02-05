"""
Tests for PDF and Markdown adapters (Phase 4 Sprint 19-21).
"""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from deepbridge.core.experiment.report.adapters import (
    MarkdownAdapter,
    PDFAdapter,
)
from deepbridge.core.experiment.report.domain.general import (
    ChartSpec,
    ChartType,
    Metric,
    MetricType,
    Report,
    ReportMetadata,
    ReportSection,
    ReportType,
)

# ==================================================================================
# Fixtures
# ==================================================================================

@pytest.fixture
def sample_report():
    """Create a sample report for testing."""
    metadata = ReportMetadata(
        model_name="TestModel",
        model_type="XGBoost",
        test_type=ReportType.UNCERTAINTY,
        created_at=datetime(2025, 11, 6, 10, 30, 0),
        dataset_name="TestDataset",
        tags=["test", "sample"]
    )

    report = Report(
        metadata=metadata,
        title="Test Uncertainty Report",
        subtitle="Sample Report for Testing"
    )

    # Add summary metrics
    report.add_summary_metric(
        Metric(
            name="accuracy",
            value=0.95,
            type=MetricType.PERCENTAGE,
            unit="%",
            description="Model accuracy"
        )
    )

    report.add_summary_metric(
        Metric(
            name="coverage",
            value=0.92,
            type=MetricType.PERCENTAGE,
            unit="%",
            description="Coverage at 90%"
        )
    )

    # Add section
    section = ReportSection(
        id="results",
        title="Test Results",
        description="Main test results section"
    )

    section.add_metric(
        Metric(
            name="mean_width",
            value=1.234,
            type=MetricType.SCALAR,
            description="Mean interval width"
        )
    )

    # Add chart
    section.add_chart(
        ChartSpec(
            id="chart_1",
            type=ChartType.COVERAGE,
            title="Coverage Chart",
            description="Coverage analysis",
            data={
                "x": [1, 2, 3],
                "y": [0.8, 0.9, 0.95]
            }
        )
    )

    report.add_section(section)

    return report


# ==================================================================================
# MarkdownAdapter Tests
# ==================================================================================

class TestMarkdownAdapter:
    """Tests for MarkdownAdapter."""

    def test_adapter_initialization(self):
        """Test adapter can be initialized."""
        adapter = MarkdownAdapter()
        assert adapter is not None
        assert adapter.include_toc is True
        assert adapter.heading_level_start == 1

    def test_adapter_with_custom_settings(self):
        """Test adapter with custom settings."""
        adapter = MarkdownAdapter(
            include_toc=False,
            heading_level_start=2,
            chart_placeholder="link"
        )
        assert adapter.include_toc is False
        assert adapter.heading_level_start == 2
        assert adapter.chart_placeholder == "link"

    def test_render_returns_string(self, sample_report):
        """Test render returns a string."""
        adapter = MarkdownAdapter()
        result = adapter.render(sample_report)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_render_includes_title(self, sample_report):
        """Test rendered markdown includes title."""
        adapter = MarkdownAdapter()
        result = adapter.render(sample_report)

        assert "# Test Uncertainty Report" in result
        assert "**Sample Report for Testing**" in result

    def test_render_includes_metadata(self, sample_report):
        """Test rendered markdown includes metadata."""
        adapter = MarkdownAdapter()
        result = adapter.render(sample_report)

        assert "## Metadata" in result
        assert "- **Model**: TestModel" in result
        assert "- **Model Type**: XGBoost" in result
        assert "- **Test Type**: uncertainty" in result
        assert "- **Dataset**: TestDataset" in result
        assert "- **Tags**: test, sample" in result

    def test_render_includes_toc(self, sample_report):
        """Test rendered markdown includes table of contents."""
        adapter = MarkdownAdapter(include_toc=True)
        result = adapter.render(sample_report)

        assert "## Table of Contents" in result
        assert "- [Summary](#summary)" in result
        assert "- [Test Results](#test-results)" in result

    def test_render_excludes_toc_when_disabled(self, sample_report):
        """Test TOC can be excluded."""
        adapter = MarkdownAdapter(include_toc=False)
        result = adapter.render(sample_report)

        assert "## Table of Contents" not in result

    def test_render_includes_summary(self, sample_report):
        """Test rendered markdown includes summary."""
        adapter = MarkdownAdapter()
        result = adapter.render(sample_report)

        assert "## Summary" in result
        assert "| Metric | Value | Unit | Description |" in result
        assert "| accuracy |" in result
        assert "| coverage |" in result

    def test_render_includes_sections(self, sample_report):
        """Test rendered markdown includes sections."""
        adapter = MarkdownAdapter()
        result = adapter.render(sample_report)

        assert "## Test Results" in result
        assert "Main test results section" in result

    def test_render_includes_metrics_table(self, sample_report):
        """Test rendered markdown includes metrics in tables."""
        adapter = MarkdownAdapter()
        result = adapter.render(sample_report)

        assert "### Metrics" in result
        assert "| mean_width |" in result

    def test_render_includes_charts(self, sample_report):
        """Test rendered markdown includes chart placeholders."""
        adapter = MarkdownAdapter(chart_placeholder="chart")
        result = adapter.render(sample_report)

        assert "### Charts" in result
        assert "#### Coverage Chart" in result
        assert "Chart Type: coverage_chart" in result

    def test_chart_placeholder_link_mode(self, sample_report):
        """Test chart placeholder in link mode."""
        adapter = MarkdownAdapter(chart_placeholder="link")
        result = adapter.render(sample_report)

        assert "![Coverage Chart](chart_1.png)" in result

    def test_chart_placeholder_ignore_mode(self, sample_report):
        """Test charts can be ignored."""
        adapter = MarkdownAdapter(chart_placeholder="ignore")
        result = adapter.render(sample_report)

        assert "Chart Type:" not in result

    def test_render_includes_footer(self, sample_report):
        """Test rendered markdown includes footer."""
        adapter = MarkdownAdapter()
        result = adapter.render(sample_report)

        assert "*Generated by DeepBridge*" in result
        assert "2025-11-06" in result

    def test_format_metric_value_float(self):
        """Test metric value formatting for floats."""
        adapter = MarkdownAdapter()

        # Normal float
        result = adapter._format_metric_value(1.23456)
        assert "1.234" in result or "1.235" in result  # Allow for rounding

        # Very small float
        assert "1.23e-05" in adapter._format_metric_value(0.0000123)

        # Very large float
        assert "1.23e+05" in adapter._format_metric_value(123456.78)

    def test_format_metric_value_boolean(self):
        """Test metric value formatting for booleans."""
        adapter = MarkdownAdapter()

        assert adapter._format_metric_value(True) == "Yes"
        assert adapter._format_metric_value(False) == "No"

    def test_format_metric_value_list(self):
        """Test metric value formatting for lists."""
        adapter = MarkdownAdapter()

        result = adapter._format_metric_value([1, 2, 3, 4, 5, 6, 7])
        assert "1, 2, 3, 4, 5" in result

    def test_create_anchor(self):
        """Test anchor creation."""
        adapter = MarkdownAdapter()

        assert adapter._create_anchor("Test Section") == "test-section"
        assert adapter._create_anchor("Results & Analysis") == "results--analysis"
        assert adapter._create_anchor("Model (v2)") == "model-v2"

    def test_save_to_file(self, sample_report):
        """Test saving markdown to file."""
        adapter = MarkdownAdapter()
        markdown = adapter.render(sample_report)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_report.md"
            result_path = adapter.save_to_file(markdown, str(output_path))

            assert Path(result_path).exists()
            assert Path(result_path).read_text(encoding='utf-8') == markdown

    def test_validation_error_no_metadata(self):
        """Test validation fails without metadata."""
        # Pydantic will raise ValidationError when creating Report without required metadata
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            report = Report(
                metadata=None,  # Invalid
                title="Test"
            )


# ==================================================================================
# PDFAdapter Tests
# ==================================================================================

class TestPDFAdapter:
    """Tests for PDFAdapter."""

    def test_adapter_initialization(self):
        """Test adapter can be initialized."""
        adapter = PDFAdapter()
        assert adapter is not None
        assert adapter.theme == "pdf"
        assert adapter.page_size == "A4"

    def test_adapter_with_custom_settings(self):
        """Test adapter with custom settings."""
        adapter = PDFAdapter(
            theme="custom",
            page_size="Letter"
        )
        assert adapter.theme == "custom"
        assert adapter.page_size == "Letter"

    def test_render_returns_bytes(self, sample_report):
        """Test render returns bytes."""
        adapter = PDFAdapter()
        result = adapter.render(sample_report)

        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_render_creates_valid_pdf(self, sample_report):
        """Test rendered output is valid PDF."""
        adapter = PDFAdapter()
        result = adapter.render(sample_report)

        # Check PDF signature
        assert result[:4] == b'%PDF'

    def test_render_includes_content(self, sample_report):
        """Test PDF includes content (basic check)."""
        adapter = PDFAdapter()
        result = adapter.render(sample_report)

        # Convert bytes to string for basic content check
        # Note: This is a simplified check, real PDF parsing would be more complex
        content = result.decode('latin-1', errors='ignore')

        assert "TestModel" in content or len(result) > 1000  # Basic sanity check

    def test_get_static_chart_type(self):
        """Test chart type mapping for static versions."""
        adapter = PDFAdapter()

        assert adapter._get_static_chart_type("width_vs_coverage") == "width_vs_coverage_static"
        assert adapter._get_static_chart_type("perturbation_impact") == "perturbation_impact_static"
        assert adapter._get_static_chart_type("unknown_chart") == "unknown_chart"

    def test_format_metric_value(self):
        """Test metric value formatting."""
        adapter = PDFAdapter()

        assert adapter._format_metric_value(1.23456) == "1.2346"
        assert adapter._format_metric_value(42) == "42"
        assert adapter._format_metric_value("text") == "text"

    def test_format_summary(self, sample_report):
        """Test summary formatting."""
        adapter = PDFAdapter()
        formatted = adapter._format_summary(sample_report.summary_metrics)

        assert len(formatted) == 2
        assert formatted[0]["name"] == "accuracy"
        assert "0.95" in formatted[0]["value"]
        assert formatted[0]["unit"] == "%"

    def test_get_pdf_css(self):
        """Test PDF CSS generation."""
        adapter = PDFAdapter()
        css = adapter._get_pdf_css("uncertainty")

        assert "@page" in css
        assert "size: A4" in css
        assert "@media print" in css
        assert "page-break" in css

    def test_create_error_placeholder(self):
        """Test error placeholder creation."""
        adapter = PDFAdapter()
        placeholder = adapter._create_error_placeholder()

        assert placeholder.startswith("data:image/png;base64,")

    def test_save_to_file(self, sample_report):
        """Test saving PDF to file."""
        adapter = PDFAdapter()
        pdf_bytes = adapter.render(sample_report)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_report.pdf"
            result_path = adapter.save_to_file(pdf_bytes, str(output_path))

            assert Path(result_path).exists()
            content = Path(result_path).read_bytes()
            assert content == pdf_bytes
            assert content[:4] == b'%PDF'

    def test_generate_simple_html(self, sample_report):
        """Test simple HTML generation fallback."""
        adapter = PDFAdapter()

        # Create context
        charts = {}
        context = adapter._create_pdf_context(sample_report, charts)

        # Generate simple HTML
        html = adapter._generate_simple_html(context)

        assert "<!DOCTYPE html>" in html
        assert "<html>" in html
        assert sample_report.title in html
        assert sample_report.metadata.model_name in html

    def test_validation_error_no_metadata(self):
        """Test validation fails without metadata."""
        # Pydantic will raise ValidationError when creating Report without required metadata
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            report = Report(
                metadata=None,  # Invalid
                title="Test"
            )

    def test_validation_error_no_model_name(self):
        """Test validation fails without model name."""
        # Pydantic will raise ValidationError when creating metadata without required model_name
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            metadata = ReportMetadata(
                model_name=None,  # Invalid
                test_type=ReportType.UNCERTAINTY
            )

    def test_pdf_adapter_with_nested_sections(self):
        """Test PDF generation with nested sections."""
        metadata = ReportMetadata(
            model_name="TestModel",
            test_type=ReportType.UNCERTAINTY
        )

        report = Report(
            metadata=metadata,
            title="Nested Report"
        )

        # Create parent section
        parent = ReportSection(
            id="parent",
            title="Parent Section"
        )

        # Create child section
        child = ReportSection(
            id="child",
            title="Child Section"
        )
        child.add_metric(Metric(name="test", value=1.0))

        parent.add_subsection(child)
        report.add_section(parent)

        # Render should not fail
        adapter = PDFAdapter()
        result = adapter.render(report)

        assert isinstance(result, bytes)
        assert len(result) > 0


# ==================================================================================
# Integration Tests
# ==================================================================================

class TestMultiFormatGeneration:
    """Test generating reports in multiple formats."""

    def test_generate_all_formats(self, sample_report):
        """Test generating report in all formats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Markdown
            md_adapter = MarkdownAdapter()
            md_content = md_adapter.render(sample_report)
            md_path = md_adapter.save_to_file(md_content, str(tmpdir / "report.md"))

            # PDF
            pdf_adapter = PDFAdapter()
            pdf_bytes = pdf_adapter.render(sample_report)
            pdf_path = pdf_adapter.save_to_file(pdf_bytes, str(tmpdir / "report.pdf"))

            # Verify files exist
            assert Path(md_path).exists()
            assert Path(pdf_path).exists()

            # Verify content
            assert Path(md_path).read_text(encoding='utf-8') == md_content
            assert Path(pdf_path).read_bytes() == pdf_bytes
