"""Tests for HTML and JSON renderers."""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any

from deepbridge.core.experiment.report.renderers.html import (
    HTMLRenderer,
    HTMLRendererWithAssets
)
from deepbridge.core.experiment.report.renderers.json import (
    JSONRenderer,
    JSONLinesRenderer
)
from deepbridge.core.experiment.report.config import (
    RenderConfig,
    ReportStyle,
    OutputFormat
)
from deepbridge.core.experiment.report.data.base import ReportData
from deepbridge.core.experiment.report.templates.engine import TemplateEngine


# Mock ReportData for testing
@dataclass
class MockReportData(ReportData):
    """Mock report data for testing."""
    test_value: str = "test"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'generated_at': self.generated_at.isoformat(),
            'report_type': self.report_type,
            'version': self.version,
            'test_value': self.test_value,
            'metadata': self.metadata
        }

    def validate(self) -> bool:
        return True


class TestHTMLRenderer:
    """Test HTMLRenderer class."""

    @pytest.fixture
    def temp_template_dir(self):
        """Create temporary template directory."""
        temp_dir = tempfile.mkdtemp()
        template_dir = Path(temp_dir)

        # Create mock templates for testing
        (template_dir / "mock").mkdir()
        (template_dir / "mock" / "full.html").write_text(
            "<html><body><h1>{{ data.test_value }}</h1></body></html>"
        )
        (template_dir / "mock" / "simple.html").write_text(
            "<html><body><p>{{ data.test_value }}</p></body></html>"
        )
        (template_dir / "mock" / "static.html").write_text(
            "<html><body><div>{{ data.test_value }}</div></body></html>"
        )

        yield template_dir

        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_data(self):
        """Create mock report data."""
        return MockReportData(
            generated_at=datetime.now(),
            report_type='mock',
            test_value='Hello World'
        )

    @pytest.fixture
    def template_engine(self, temp_template_dir):
        """Create template engine with mock templates."""
        return TemplateEngine(temp_template_dir)

    @pytest.fixture
    def html_renderer(self, template_engine):
        """Create HTML renderer."""
        return HTMLRenderer(template_engine)

    def test_initialization(self, template_engine):
        """Test HTMLRenderer initialization."""
        renderer = HTMLRenderer(template_engine)

        assert renderer.template_engine is template_engine

    def test_render_full_style(self, html_renderer, mock_data):
        """Test rendering with FULL style."""
        config = RenderConfig(
            style=ReportStyle.FULL,
            format=OutputFormat.HTML
        )

        html = html_renderer.render(mock_data, config)

        assert '<html>' in html
        assert '<h1>Hello World</h1>' in html

    def test_render_simple_style(self, html_renderer, mock_data):
        """Test rendering with SIMPLE style."""
        config = RenderConfig(
            style=ReportStyle.SIMPLE,
            format=OutputFormat.HTML,
            include_charts=False
        )

        html = html_renderer.render(mock_data, config)

        assert '<html>' in html
        assert '<p>Hello World</p>' in html

    def test_render_static_style(self, html_renderer, mock_data):
        """Test rendering with STATIC style."""
        config = RenderConfig(
            style=ReportStyle.STATIC,
            format=OutputFormat.HTML
        )

        html = html_renderer.render(mock_data, config)

        assert '<html>' in html
        assert '<div>Hello World</div>' in html

    def test_render_invalid_format(self, html_renderer, mock_data):
        """Test that non-HTML format raises error."""
        config = RenderConfig(format=OutputFormat.JSON)

        with pytest.raises(ValueError, match="only supports HTML format"):
            html_renderer.render(mock_data, config)

    def test_render_to_file(self, html_renderer, mock_data, tmp_path):
        """Test rendering to file."""
        config = RenderConfig(format=OutputFormat.HTML)
        output_path = tmp_path / "report.html"

        result_path = html_renderer.render_to_file(mock_data, config, output_path)

        assert result_path == output_path
        assert result_path.exists()
        content = result_path.read_text()
        assert '<html>' in content
        assert 'Hello World' in content

    def test_get_template_name(self, html_renderer):
        """Test _get_template_name method."""
        assert html_renderer._get_template_name('robustness', ReportStyle.FULL) == 'robustness/full.html'
        assert html_renderer._get_template_name('resilience', ReportStyle.SIMPLE) == 'resilience/simple.html'
        assert html_renderer._get_template_name('uncertainty', ReportStyle.STATIC) == 'uncertainty/static.html'

    def test_prepare_context(self, html_renderer, mock_data):
        """Test _prepare_context method."""
        config = RenderConfig(
            style=ReportStyle.FULL,
            theme='dark',
            metadata={'custom': 'value'}
        )

        context = html_renderer._prepare_context(mock_data, config)

        assert 'data' in context
        assert 'config' in context
        assert context['report_type'] == 'mock'
        assert context['style'] == 'full'
        assert context['theme'] == 'dark'
        assert 'metadata' in context
        assert context['metadata']['custom'] == 'value'


class TestHTMLRendererWithAssets:
    """Test HTMLRendererWithAssets class."""

    @pytest.fixture
    def temp_template_dir(self):
        """Create temporary template directory."""
        temp_dir = tempfile.mkdtemp()
        template_dir = Path(temp_dir)

        (template_dir / "mock").mkdir()
        (template_dir / "mock" / "full.html").write_text(
            "<html><body>{{ data.test_value }}</body></html>"
        )

        yield template_dir

        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_data(self):
        """Create mock report data."""
        return MockReportData(
            generated_at=datetime.now(),
            report_type='mock'
        )

    def test_initialization_without_asset_manager(self, temp_template_dir):
        """Test initialization without asset manager."""
        engine = TemplateEngine(temp_template_dir)
        renderer = HTMLRendererWithAssets(engine, asset_manager=None)

        # Should initialize even without asset manager
        assert renderer.template_engine is engine

    def test_prepare_context_with_embed_assets(self, temp_template_dir, mock_data):
        """Test context preparation with embed_assets."""
        engine = TemplateEngine(temp_template_dir)

        # Mock asset manager
        class MockAssetManager:
            def get_css_content(self):
                return "body { color: black; }"

            def get_js_content(self):
                return "console.log('test');"

        renderer = HTMLRendererWithAssets(engine, MockAssetManager())
        config = RenderConfig(embed_assets=True)

        context = renderer._prepare_context(mock_data, config)

        assert 'css_content' in context
        assert 'js_content' in context
        assert context['embed_assets'] is True


class TestJSONRenderer:
    """Test JSONRenderer class."""

    @pytest.fixture
    def mock_data(self):
        """Create mock report data."""
        return MockReportData(
            generated_at=datetime(2024, 1, 1, 12, 0, 0),
            report_type='mock',
            test_value='Test Value',
            metadata={'key': 'value'}
        )

    def test_initialization(self):
        """Test JSONRenderer initialization."""
        renderer = JSONRenderer(indent=2, sort_keys=True)

        assert renderer.indent == 2
        assert renderer.sort_keys is True

    def test_render_basic(self, mock_data):
        """Test basic JSON rendering."""
        renderer = JSONRenderer()
        config = RenderConfig(format=OutputFormat.JSON)

        json_str = renderer.render(mock_data, config)

        assert json_str is not None
        data = json.loads(json_str)
        assert data['report_type'] == 'mock'
        assert data['test_value'] == 'Test Value'

    def test_render_with_config(self, mock_data):
        """Test rendering with config included."""
        renderer = JSONRenderer()
        config = RenderConfig(
            format=OutputFormat.JSON,
            include_raw_data=True
        )

        json_str = renderer.render(mock_data, config)
        data = json.loads(json_str)

        assert 'report' in data
        assert 'config' in data
        assert data['report']['report_type'] == 'mock'

    def test_render_invalid_format(self, mock_data):
        """Test that non-JSON format raises error."""
        renderer = JSONRenderer()
        config = RenderConfig(format=OutputFormat.HTML)

        with pytest.raises(ValueError, match="only supports JSON format"):
            renderer.render(mock_data, config)

    def test_render_compact(self, mock_data):
        """Test rendering compact JSON."""
        renderer = JSONRenderer(indent=2)
        config = RenderConfig(format=OutputFormat.JSON)

        compact = renderer.render_compact(mock_data, config)

        # Compact should have no newlines (except potentially at end)
        assert compact.count('\n') <= 1

    def test_render_pretty(self, mock_data):
        """Test rendering pretty-printed JSON."""
        renderer = JSONRenderer(indent=None)
        config = RenderConfig(format=OutputFormat.JSON)

        pretty = renderer.render_pretty(mock_data, config)

        # Pretty should have multiple newlines for indentation
        assert pretty.count('\n') > 5

    def test_render_to_file(self, mock_data, tmp_path):
        """Test rendering to file."""
        renderer = JSONRenderer()
        config = RenderConfig(format=OutputFormat.JSON)
        output_path = tmp_path / "report.json"

        result_path = renderer.render_to_file(mock_data, config, output_path)

        assert result_path == output_path
        assert result_path.exists()

        with open(result_path) as f:
            data = json.load(f)

        assert data['report_type'] == 'mock'

    def test_indent_formatting(self, mock_data):
        """Test that indent parameter affects formatting."""
        config = RenderConfig(format=OutputFormat.JSON)

        renderer_no_indent = JSONRenderer(indent=None)
        renderer_indent_2 = JSONRenderer(indent=2)
        renderer_indent_4 = JSONRenderer(indent=4)

        json_no_indent = renderer_no_indent.render(mock_data, config)
        json_indent_2 = renderer_indent_2.render(mock_data, config)
        json_indent_4 = renderer_indent_4.render(mock_data, config)

        # More indentation -> longer output
        assert len(json_no_indent) < len(json_indent_2) < len(json_indent_4)

    def test_sort_keys(self, mock_data):
        """Test sort_keys parameter."""
        config = RenderConfig(format=OutputFormat.JSON)

        renderer = JSONRenderer(sort_keys=True)
        json_str = renderer.render(mock_data, config)

        # Check that keys appear in alphabetical order
        data = json.loads(json_str)
        keys = list(data.keys())
        assert keys == sorted(keys)


class TestJSONLinesRenderer:
    """Test JSONLinesRenderer class."""

    @pytest.fixture
    def mock_data(self):
        """Create mock report data."""
        return MockReportData(
            generated_at=datetime(2024, 1, 1, 12, 0, 0),
            report_type='mock',
            test_value='Test'
        )

    def test_initialization(self):
        """Test JSONLinesRenderer initialization."""
        renderer = JSONLinesRenderer()

        # Should be initialized with compact format
        assert renderer.indent is None
        assert renderer.sort_keys is False

    def test_render_line(self, mock_data):
        """Test rendering single line."""
        renderer = JSONLinesRenderer()
        config = RenderConfig(format=OutputFormat.JSON)

        line = renderer.render_line(mock_data, config)

        assert line.endswith('\n')
        assert line.count('\n') == 1

        data = json.loads(line.strip())
        assert data['report_type'] == 'mock'

    def test_render_to_file_new(self, mock_data, tmp_path):
        """Test rendering to new file."""
        renderer = JSONLinesRenderer()
        config = RenderConfig(format=OutputFormat.JSON)
        output_path = tmp_path / "log.jsonl"

        result_path = renderer.render_to_file(mock_data, config, output_path, append=False)

        assert result_path.exists()
        lines = result_path.read_text().strip().split('\n')
        assert len(lines) == 1

    def test_render_to_file_append(self, mock_data, tmp_path):
        """Test appending to existing file."""
        renderer = JSONLinesRenderer()
        config = RenderConfig(format=OutputFormat.JSON)
        output_path = tmp_path / "log.jsonl"

        # Write first line
        renderer.render_to_file(mock_data, config, output_path, append=False)

        # Append second line
        mock_data2 = MockReportData(
            generated_at=datetime.now(),
            report_type='mock',
            test_value='Test2'
        )
        renderer.render_to_file(mock_data2, config, output_path, append=True)

        # Should have 2 lines
        lines = output_path.read_text().strip().split('\n')
        assert len(lines) == 2

        data1 = json.loads(lines[0])
        data2 = json.loads(lines[1])

        assert data1['test_value'] == 'Test'
        assert data2['test_value'] == 'Test2'


class TestRendererIntegration:
    """Integration tests for renderers."""

    @pytest.fixture
    def mock_data(self):
        """Create mock report data."""
        return MockReportData(
            generated_at=datetime.now(),
            report_type='mock',
            test_value='Integration Test'
        )

    def test_html_and_json_render_same_data(self, mock_data, tmp_path):
        """Test that HTML and JSON renderers handle same data."""
        # Create temporary templates
        template_dir = tmp_path / "templates"
        template_dir.mkdir()
        (template_dir / "mock").mkdir()
        (template_dir / "mock" / "full.html").write_text(
            "<html>{{ data.test_value }}</html>"
        )

        # HTML renderer
        engine = TemplateEngine(template_dir)
        html_renderer = HTMLRenderer(engine)
        html_config = RenderConfig(format=OutputFormat.HTML)
        html = html_renderer.render(mock_data, html_config)

        # JSON renderer
        json_renderer = JSONRenderer()
        json_config = RenderConfig(format=OutputFormat.JSON)
        json_str = json_renderer.render(mock_data, json_config)

        # Both should process the same data
        assert 'Integration Test' in html

        json_data = json.loads(json_str)
        assert json_data['test_value'] == 'Integration Test'
