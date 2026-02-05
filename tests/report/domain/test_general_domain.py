"""
Tests for general presentation-agnostic domain models (Phase 3 Sprint 13).

Tests all core domain classes:
- ReportMetadata
- Metric
- ChartSpec
- ReportSection
- Report
"""

from datetime import datetime

import pytest

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

# ==================================================================================
# ReportMetadata Tests
# ==================================================================================


class TestReportMetadata:
    """Test ReportMetadata class."""

    def test_basic_creation(self):
        """Test creating basic metadata."""
        metadata = ReportMetadata(
            model_name='TestModel', test_type=ReportType.UNCERTAINTY
        )

        assert metadata.model_name == 'TestModel'
        assert metadata.test_type == ReportType.UNCERTAINTY
        assert metadata.version == '1.0'
        assert isinstance(metadata.created_at, datetime)

    def test_with_optional_fields(self):
        """Test metadata with optional fields."""
        metadata = ReportMetadata(
            model_name='TestModel',
            test_type=ReportType.ROBUSTNESS,
            dataset_name='MNIST',
            dataset_size=10000,
            test_duration=123.45,
            model_type='classification',
            tags=['production', 'v2'],
        )

        assert metadata.dataset_name == 'MNIST'
        assert metadata.dataset_size == 10000
        assert metadata.test_duration == 123.45
        assert metadata.model_type == 'classification'
        assert len(metadata.tags) == 2

    def test_extra_metadata(self):
        """Test extra metadata dictionary."""
        metadata = ReportMetadata(
            model_name='Test',
            test_type=ReportType.RESILIENCE,
            extra={'custom_field': 'value', 'number': 42},
        )

        assert metadata.extra['custom_field'] == 'value'
        assert metadata.extra['number'] == 42


# ==================================================================================
# Metric Tests
# ==================================================================================


class TestMetric:
    """Test Metric class."""

    def test_scalar_metric(self):
        """Test creating scalar metric."""
        metric = Metric(name='accuracy', value=0.95, type=MetricType.SCALAR)

        assert metric.name == 'accuracy'
        assert metric.value == 0.95
        assert metric.type == MetricType.SCALAR

    def test_percentage_metric(self):
        """Test percentage metric."""
        metric = Metric(
            name='coverage', value=0.92, type=MetricType.PERCENTAGE
        )

        assert metric.formatted_value == '92.0%'

    def test_metric_with_threshold(self):
        """Test metric with pass/fail threshold."""
        # Passing metric
        passing = Metric(
            name='score', value=0.95, threshold=0.90, higher_is_better=True
        )

        assert passing.is_passing is True

        # Failing metric
        failing = Metric(
            name='score', value=0.85, threshold=0.90, higher_is_better=True
        )

        assert failing.is_passing is False

    def test_metric_lower_is_better(self):
        """Test metric where lower is better."""
        metric = Metric(
            name='error_rate',
            value=0.05,
            threshold=0.10,
            higher_is_better=False,
        )

        assert metric.is_passing is True

    def test_metric_no_threshold(self):
        """Test metric without threshold."""
        metric = Metric(name='value', value=0.5)

        assert metric.is_passing is None

    def test_formatted_value_custom_format(self):
        """Test custom format string."""
        metric = Metric(name='score', value=0.123456, format_string='.2f')

        assert metric.formatted_value == '0.12'

    def test_formatted_value_count(self):
        """Test count type formatting."""
        metric = Metric(name='count', value=123.7, type=MetricType.COUNT)

        assert metric.formatted_value == '123'

    def test_metric_with_unit(self):
        """Test metric with unit."""
        metric = Metric(
            name='duration',
            value=45.2,
            type=MetricType.DURATION,
            unit='seconds',
        )

        assert metric.unit == 'seconds'


# ==================================================================================
# ChartSpec Tests
# ==================================================================================


class TestChartSpec:
    """Test ChartSpec class."""

    def test_basic_chart_spec(self):
        """Test creating basic chart specification."""
        chart = ChartSpec(
            id='coverage_plot',
            type=ChartType.COVERAGE,
            title='Coverage Analysis',
            data={
                'alphas': [0.1, 0.2, 0.3],
                'coverage': [0.91, 0.81, 0.72],
                'expected': [0.90, 0.80, 0.70],
            },
        )

        assert chart.id == 'coverage_plot'
        assert chart.type == ChartType.COVERAGE
        assert chart.title == 'Coverage Analysis'
        assert len(chart.data['alphas']) == 3

    def test_chart_with_options(self):
        """Test chart with custom options."""
        chart = ChartSpec(
            id='line_plot',
            type=ChartType.LINE,
            title='Line Chart',
            data={'x': [1, 2], 'y': [3, 4]},
            width=800,
            height=600,
            options={'color': 'blue', 'line_width': 2},
        )

        assert chart.width == 800
        assert chart.height == 600
        assert chart.options['color'] == 'blue'

    def test_primary_chart(self):
        """Test primary chart flag."""
        chart = ChartSpec(
            id='main_chart',
            type=ChartType.BAR,
            title='Main Chart',
            data={},
            is_primary=True,
        )

        assert chart.is_primary is True


# ==================================================================================
# ReportSection Tests
# ==================================================================================


class TestReportSection:
    """Test ReportSection class."""

    def test_basic_section(self):
        """Test creating basic section."""
        section = ReportSection(
            id='results', title='Test Results', description='Main test results'
        )

        assert section.id == 'results'
        assert section.title == 'Test Results'
        assert section.description == 'Main test results'
        assert len(section.metrics) == 0
        assert len(section.charts) == 0

    def test_add_metric(self):
        """Test adding metrics to section."""
        section = ReportSection(id='sec1', title='Section 1')

        metric1 = Metric(name='metric1', value=0.5)
        metric2 = Metric(name='metric2', value=0.8)

        section.add_metric(metric1).add_metric(metric2)

        assert len(section.metrics) == 2
        assert section.metrics[0].name == 'metric1'
        assert section.metrics[1].name == 'metric2'

    def test_add_chart(self):
        """Test adding charts to section."""
        section = ReportSection(id='sec1', title='Section 1')

        chart = ChartSpec(
            id='chart1', type=ChartType.LINE, title='Chart 1', data={}
        )

        section.add_chart(chart)

        assert len(section.charts) == 1
        assert section.charts[0].id == 'chart1'

    def test_add_subsection(self):
        """Test adding subsections."""
        parent = ReportSection(id='parent', title='Parent')
        child = ReportSection(id='child', title='Child')

        parent.add_subsection(child)

        assert len(parent.subsections) == 1
        assert parent.subsections[0].id == 'child'

    def test_primary_metrics_property(self):
        """Test filtering primary metrics."""
        section = ReportSection(id='sec', title='Section')

        section.add_metric(Metric(name='m1', value=1, is_primary=True))
        section.add_metric(Metric(name='m2', value=2, is_primary=False))
        section.add_metric(Metric(name='m3', value=3, is_primary=True))

        primary = section.primary_metrics

        assert len(primary) == 2
        assert primary[0].name == 'm1'
        assert primary[1].name == 'm3'

    def test_primary_charts_property(self):
        """Test filtering primary charts."""
        section = ReportSection(id='sec', title='Section')

        section.add_chart(
            ChartSpec(
                id='c1',
                type=ChartType.LINE,
                title='C1',
                data={},
                is_primary=True,
            )
        )
        section.add_chart(
            ChartSpec(
                id='c2',
                type=ChartType.BAR,
                title='C2',
                data={},
                is_primary=False,
            )
        )

        primary = section.primary_charts

        assert len(primary) == 1
        assert primary[0].id == 'c1'


# ==================================================================================
# Report Tests
# ==================================================================================


class TestReport:
    """Test Report class."""

    def test_basic_report(self):
        """Test creating basic report."""
        metadata = ReportMetadata(
            model_name='TestModel', test_type=ReportType.UNCERTAINTY
        )

        report = Report(metadata=metadata)

        assert report.metadata.model_name == 'TestModel'
        assert report.metadata.test_type == ReportType.UNCERTAINTY
        assert len(report.sections) == 0

    def test_add_section(self):
        """Test adding sections to report."""
        metadata = ReportMetadata(
            model_name='Test', test_type=ReportType.ROBUSTNESS
        )
        report = Report(metadata=metadata)

        section1 = ReportSection(id='sec1', title='Section 1')
        section2 = ReportSection(id='sec2', title='Section 2')

        report.add_section(section1).add_section(section2)

        assert len(report.sections) == 2
        assert report.sections[0].id == 'sec1'

    def test_add_summary_metric(self):
        """Test adding summary metrics."""
        metadata = ReportMetadata(
            model_name='Test', test_type=ReportType.UNCERTAINTY
        )
        report = Report(metadata=metadata)

        metric = Metric(name='overall_score', value=0.92, is_primary=True)
        report.add_summary_metric(metric)

        assert len(report.summary_metrics) == 1
        assert report.summary_metrics[0].name == 'overall_score'

    def test_get_section(self):
        """Test getting section by ID."""
        metadata = ReportMetadata(
            model_name='Test', test_type=ReportType.RESILIENCE
        )
        report = Report(metadata=metadata)

        section = ReportSection(id='results', title='Results')
        report.add_section(section)

        found = report.get_section('results')

        assert found is not None
        assert found.id == 'results'

        not_found = report.get_section('nonexistent')
        assert not_found is None

    def test_get_all_metrics(self):
        """Test getting all metrics from all sections."""
        metadata = ReportMetadata(
            model_name='Test', test_type=ReportType.UNCERTAINTY
        )
        report = Report(metadata=metadata)

        # Add summary metric
        report.add_summary_metric(Metric(name='summary', value=1))

        # Add section with metrics
        section = ReportSection(id='sec1', title='Section 1')
        section.add_metric(Metric(name='metric1', value=2))
        section.add_metric(Metric(name='metric2', value=3))

        # Add subsection with metric
        subsection = ReportSection(id='sub1', title='Subsection')
        subsection.add_metric(Metric(name='metric3', value=4))
        section.add_subsection(subsection)

        report.add_section(section)

        all_metrics = report.get_all_metrics()

        assert len(all_metrics) == 4
        assert all_metrics[0].name == 'summary'
        assert all_metrics[1].name == 'metric1'
        assert all_metrics[3].name == 'metric3'

    def test_get_all_charts(self):
        """Test getting all charts from all sections."""
        metadata = ReportMetadata(
            model_name='Test', test_type=ReportType.ROBUSTNESS
        )
        report = Report(metadata=metadata)

        section = ReportSection(id='sec1', title='Section 1')
        section.add_chart(
            ChartSpec(id='c1', type=ChartType.LINE, title='C1', data={})
        )
        section.add_chart(
            ChartSpec(id='c2', type=ChartType.BAR, title='C2', data={})
        )

        subsection = ReportSection(id='sub1', title='Subsection')
        subsection.add_chart(
            ChartSpec(id='c3', type=ChartType.COVERAGE, title='C3', data={})
        )
        section.add_subsection(subsection)

        report.add_section(section)

        all_charts = report.get_all_charts()

        assert len(all_charts) == 3
        assert all_charts[0].id == 'c1'
        assert all_charts[2].id == 'c3'

    def test_display_title_explicit(self):
        """Test display title with explicit title."""
        metadata = ReportMetadata(
            model_name='Test', test_type=ReportType.UNCERTAINTY
        )
        report = Report(metadata=metadata, title='Custom Title')

        assert report.display_title == 'Custom Title'

    def test_display_title_auto(self):
        """Test display title auto-generated from metadata."""
        metadata = ReportMetadata(
            model_name='MyModel', test_type=ReportType.ROBUSTNESS
        )
        report = Report(metadata=metadata)

        assert report.display_title == 'Robustness Report - MyModel'


# ==================================================================================
# Integration Tests
# ==================================================================================


class TestDomainModelIntegration:
    """Test building complete reports using domain models."""

    def test_build_complete_report(self):
        """Test building a complete report with all components."""
        # Create metadata
        metadata = ReportMetadata(
            model_name='ResNet50',
            test_type=ReportType.UNCERTAINTY,
            dataset_name='ImageNet',
            dataset_size=50000,
        )

        # Create report
        report = Report(
            metadata=metadata,
            title='Uncertainty Analysis Report',
            introduction='This report analyzes uncertainty quantification.',
        )

        # Add summary metrics
        report.add_summary_metric(
            Metric(
                name='overall_uncertainty',
                value=0.85,
                is_primary=True,
                type=MetricType.PERCENTAGE,
            )
        )

        # Create main section
        results_section = ReportSection(
            id='main_results',
            title='Main Results',
            description='Primary test results',
        )

        # Add metrics to section
        results_section.add_metric(
            Metric(
                name='coverage_90', value=0.92, description='Coverage at 90%'
            )
        )
        results_section.add_metric(
            Metric(
                name='mean_width',
                value=0.15,
                description='Mean interval width',
            )
        )

        # Add chart to section
        results_section.add_chart(
            ChartSpec(
                id='coverage_plot',
                type=ChartType.COVERAGE,
                title='Coverage vs Expected',
                data={
                    'alphas': [0.1, 0.2, 0.3],
                    'coverage': [0.91, 0.81, 0.72],
                    'expected': [0.90, 0.80, 0.70],
                },
                is_primary=True,
            )
        )

        # Create subsection
        calibration_section = ReportSection(
            id='calibration', title='Calibration Analysis'
        )
        calibration_section.add_metric(
            Metric(name='calibration_error', value=0.02)
        )

        results_section.add_subsection(calibration_section)

        # Add section to report
        report.add_section(results_section)

        # Verify structure
        assert len(report.sections) == 1
        assert len(report.summary_metrics) == 1
        assert (
            len(report.get_all_metrics()) == 4
        )  # 1 summary + 2 in main + 1 in sub
        assert len(report.get_all_charts()) == 1

        # Verify data integrity
        assert report.metadata.model_name == 'ResNet50'
        assert report.sections[0].metrics[0].name == 'coverage_90'
        assert report.sections[0].charts[0].type == ChartType.COVERAGE
        assert report.sections[0].subsections[0].id == 'calibration'

    def test_json_serialization(self):
        """Test that domain models can be serialized to JSON."""
        metadata = ReportMetadata(
            model_name='Test', test_type=ReportType.UNCERTAINTY
        )
        report = Report(metadata=metadata)

        section = ReportSection(id='sec1', title='Section 1')
        section.add_metric(Metric(name='metric1', value=0.95))
        report.add_section(section)

        # Convert to dict (JSON-safe)
        data = report.model_dump_json_safe()

        assert isinstance(data, dict)
        assert data['metadata']['model_name'] == 'Test'
        assert data['sections'][0]['metrics'][0]['value'] == 0.95
