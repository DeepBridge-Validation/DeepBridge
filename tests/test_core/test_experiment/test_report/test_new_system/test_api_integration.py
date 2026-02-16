"""Integration tests for ReportGenerator API."""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from deepbridge.core.experiment.report.api import (
    ReportGenerator,
    generate_robustness_report
)
from deepbridge.core.experiment.report.config import (
    RenderConfig,
    ReportStyle,
    OutputFormat
)


class TestReportGeneratorIntegration:
    """Integration tests for ReportGenerator."""

    @pytest.fixture
    def temp_template_dir(self):
        """Create temporary template directory with mock templates."""
        temp_dir = tempfile.mkdtemp()
        template_dir = Path(temp_dir)

        # Create robustness templates
        robustness_dir = template_dir / "robustness"
        robustness_dir.mkdir()

        (robustness_dir / "full.html").write_text("""
<!DOCTYPE html>
<html>
<head><title>Robustness Report - {{ data.model_name }}</title></head>
<body>
    <h1>Robustness Report</h1>
    <p>Model: {{ data.model_name }}</p>
    <p>Score: {{ data.robustness_score }}</p>
</body>
</html>
        """)

        (robustness_dir / "simple.html").write_text("""
<html>
<body>
    <h2>{{ data.model_name }}</h2>
    <p>Score: {{ data.robustness_score }}</p>
</body>
</html>
        """)

        (robustness_dir / "static.html").write_text("""
<html><body>{{ data.model_name }}: {{ data.robustness_score }}</body></html>
        """)

        yield template_dir

        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_robustness_data(self):
        """Sample robustness experiment results."""
        return {
            'model_name': 'TestModel',
            'model_type': 'RandomForest',
            'base_score': 0.90,
            'robustness_score': 0.85,
            'avg_raw_impact': 0.05,
            'avg_quantile_impact': 0.04,
            'metric': 'accuracy',
            'raw': {
                'by_level': {
                    '0.1': {
                        'overall_result': {
                            'all_features': {
                                'mean_score': 0.88,
                                'worst_score': 0.85,
                                'impact': 0.02
                            }
                        }
                    }
                }
            }
        }

    def test_generator_initialization(self, temp_template_dir):
        """Test ReportGenerator initialization."""
        generator = ReportGenerator(
            template_dir=temp_template_dir,
            asset_manager=None
        )

        assert generator.template_engine is not None
        assert generator.renderers is not None
        assert 'robustness' in generator.transformers

    def test_generate_robustness_html_full(self, temp_template_dir, sample_robustness_data, tmp_path):
        """Test generating full HTML robustness report."""
        generator = ReportGenerator(
            template_dir=temp_template_dir,
            asset_manager=None
        )

        output_path = tmp_path / "robustness_full.html"
        config = RenderConfig(
            style=ReportStyle.FULL,
            format=OutputFormat.HTML
        )

        result_path = generator.generate_robustness_report(
            results=sample_robustness_data,
            output_path=output_path,
            config=config
        )

        assert result_path == output_path
        assert result_path.exists()

        content = result_path.read_text()
        assert '<html>' in content
        assert 'TestModel' in content
        assert '0.85' in content

    def test_generate_robustness_html_simple(self, temp_template_dir, sample_robustness_data, tmp_path):
        """Test generating simple HTML robustness report."""
        generator = ReportGenerator(
            template_dir=temp_template_dir,
            asset_manager=None
        )

        output_path = tmp_path / "robustness_simple.html"
        config = RenderConfig(
            style=ReportStyle.SIMPLE,
            format=OutputFormat.HTML,
            include_charts=False
        )

        result_path = generator.generate_robustness_report(
            results=sample_robustness_data,
            output_path=output_path,
            config=config
        )

        assert result_path.exists()
        content = result_path.read_text()
        assert 'TestModel' in content

    def test_generate_robustness_json(self, temp_template_dir, sample_robustness_data, tmp_path):
        """Test generating JSON robustness report."""
        generator = ReportGenerator(
            template_dir=temp_template_dir,
            asset_manager=None
        )

        output_path = tmp_path / "robustness.json"
        config = RenderConfig(format=OutputFormat.JSON)

        result_path = generator.generate_robustness_report(
            results=sample_robustness_data,
            output_path=output_path,
            config=config
        )

        assert result_path.exists()

        with open(result_path) as f:
            data = json.load(f)

        assert data['report_type'] == 'robustness'
        assert data['model_name'] == 'TestModel'
        assert 'robustness_score' in data

    def test_generate_robustness_default_config(self, temp_template_dir, sample_robustness_data, tmp_path):
        """Test generating report with default config."""
        generator = ReportGenerator(
            template_dir=temp_template_dir,
            asset_manager=None
        )

        output_path = tmp_path / "robustness_default.html"

        # No config provided - should use defaults
        result_path = generator.generate_robustness_report(
            results=sample_robustness_data,
            output_path=output_path
        )

        assert result_path.exists()
        content = result_path.read_text()
        assert '<html>' in content

    def test_generate_invalid_report_type(self, temp_template_dir, tmp_path):
        """Test that unsupported report type raises error."""
        generator = ReportGenerator(
            template_dir=temp_template_dir,
            asset_manager=None
        )

        output_path = tmp_path / "report.html"
        config = RenderConfig()

        # Try to generate unsupported report type
        with pytest.raises(ValueError, match="Unsupported report type"):
            generator._generate_report(
                report_type='unsupported_type',
                results={},
                output_path=output_path,
                config=config
            )

    def test_add_custom_transformer(self, temp_template_dir):
        """Test adding custom transformer."""
        generator = ReportGenerator(
            template_dir=temp_template_dir,
            asset_manager=None
        )

        # Mock custom transformer
        class MockTransformer:
            def transform(self, raw_data):
                pass

        custom_transformer = MockTransformer()
        generator.add_transformer('custom', custom_transformer)

        assert 'custom' in generator.transformers
        assert generator.transformers['custom'] is custom_transformer

    def test_add_custom_renderer(self, temp_template_dir):
        """Test adding custom renderer."""
        generator = ReportGenerator(
            template_dir=temp_template_dir,
            asset_manager=None
        )

        # Mock custom renderer
        class MockRenderer:
            def render(self, data, config):
                return "mock output"

        custom_renderer = MockRenderer()
        generator.add_renderer(OutputFormat.PDF, custom_renderer)

        assert OutputFormat.PDF in generator.renderers
        assert generator.renderers[OutputFormat.PDF] is custom_renderer

    def test_multiple_reports_same_generator(self, temp_template_dir, sample_robustness_data, tmp_path):
        """Test generating multiple reports with same generator."""
        generator = ReportGenerator(
            template_dir=temp_template_dir,
            asset_manager=None
        )

        # Generate multiple reports
        path1 = generator.generate_robustness_report(
            results=sample_robustness_data,
            output_path=tmp_path / "report1.html",
            config=RenderConfig(style=ReportStyle.FULL)
        )

        path2 = generator.generate_robustness_report(
            results=sample_robustness_data,
            output_path=tmp_path / "report2.html",
            config=RenderConfig(style=ReportStyle.SIMPLE, include_charts=False)
        )

        path3 = generator.generate_robustness_report(
            results=sample_robustness_data,
            output_path=tmp_path / "report3.json",
            config=RenderConfig(format=OutputFormat.JSON)
        )

        assert path1.exists() and path1.name == "report1.html"
        assert path2.exists() and path2.name == "report2.html"
        assert path3.exists() and path3.name == "report3.json"


class TestConvenienceFunctions:
    """Test convenience functions."""

    @pytest.fixture
    def temp_template_dir(self):
        """Create temporary template directory."""
        temp_dir = tempfile.mkdtemp()
        template_dir = Path(temp_dir)

        robustness_dir = template_dir / "robustness"
        robustness_dir.mkdir()
        (robustness_dir / "full.html").write_text(
            "<html>{{ data.model_name }}</html>"
        )

        yield template_dir

        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_data(self):
        """Sample experiment data."""
        return {
            'model_name': 'TestModel',
            'model_type': 'RF',
            'base_score': 0.90,
            'robustness_score': 0.85,
            'raw': {'by_level': {}}
        }

    def test_generate_robustness_report_function(self, temp_template_dir, sample_data, tmp_path):
        """Test generate_robustness_report convenience function."""
        output_path = tmp_path / "report.html"

        result_path = generate_robustness_report(
            results=sample_data,
            output_path=output_path,
            template_dir=temp_template_dir
        )

        assert result_path.exists()
        content = result_path.read_text()
        assert 'TestModel' in content

    def test_convenience_function_with_config(self, temp_template_dir, sample_data, tmp_path):
        """Test convenience function with custom config."""
        output_path = tmp_path / "report.json"
        config = RenderConfig(format=OutputFormat.JSON)

        result_path = generate_robustness_report(
            results=sample_data,
            output_path=output_path,
            config=config,
            template_dir=temp_template_dir
        )

        assert result_path.exists()

        with open(result_path) as f:
            data = json.load(f)

        assert data['report_type'] == 'robustness'


class TestErrorHandling:
    """Test error handling in ReportGenerator."""

    @pytest.fixture
    def temp_template_dir(self):
        """Create minimal template directory."""
        temp_dir = tempfile.mkdtemp()
        template_dir = Path(temp_dir)

        robustness_dir = template_dir / "robustness"
        robustness_dir.mkdir()
        (robustness_dir / "full.html").write_text("<html>test</html>")

        yield template_dir

        shutil.rmtree(temp_dir)

    def test_invalid_results_data(self, temp_template_dir, tmp_path):
        """Test that invalid results data raises error."""
        generator = ReportGenerator(
            template_dir=temp_template_dir,
            asset_manager=None
        )

        output_path = tmp_path / "report.html"

        # Empty results should raise error
        with pytest.raises(RuntimeError):
            generator.generate_robustness_report(
                results={},
                output_path=output_path
            )

    def test_missing_template(self, temp_template_dir, tmp_path):
        """Test error when template is missing."""
        generator = ReportGenerator(
            template_dir=temp_template_dir,
            asset_manager=None
        )

        output_path = tmp_path / "report.html"
        sample_data = {
            'model_name': 'Test',
            'model_type': 'RF',
            'base_score': 0.85,
            'robustness_score': 0.80
        }

        # Try to use a style that doesn't have a template
        config = RenderConfig(style=ReportStyle.INTERACTIVE)

        with pytest.raises(RuntimeError):
            generator.generate_robustness_report(
                results=sample_data,
                output_path=output_path,
                config=config
            )

    def test_output_directory_created(self, temp_template_dir, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        generator = ReportGenerator(
            template_dir=temp_template_dir,
            asset_manager=None
        )

        # Output path in non-existent subdirectory
        output_path = tmp_path / "subdir" / "nested" / "report.html"

        sample_data = {
            'model_name': 'Test',
            'model_type': 'RF',
            'base_score': 0.85,
            'robustness_score': 0.80
        }

        result_path = generator.generate_robustness_report(
            results=sample_data,
            output_path=output_path
        )

        assert result_path.exists()
        assert result_path.parent.exists()
