"""Tests for configuration system."""

import pytest
from deepbridge.core.experiment.report.config import (
    RenderConfig,
    ReportStyle,
    OutputFormat,
    get_preset_config,
    PRESET_CONFIGS
)


class TestReportStyle:
    """Test ReportStyle enum."""

    def test_styles_defined(self):
        """Test that all expected styles are defined."""
        assert ReportStyle.FULL.value == "full"
        assert ReportStyle.SIMPLE.value == "simple"
        assert ReportStyle.STATIC.value == "static"
        assert ReportStyle.INTERACTIVE.value == "interactive"


class TestOutputFormat:
    """Test OutputFormat enum."""

    def test_formats_defined(self):
        """Test that all expected formats are defined."""
        assert OutputFormat.HTML.value == "html"
        assert OutputFormat.JSON.value == "json"
        assert OutputFormat.PDF.value == "pdf"
        assert OutputFormat.MARKDOWN.value == "markdown"


class TestRenderConfig:
    """Test RenderConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RenderConfig()

        assert config.style == ReportStyle.FULL
        assert config.format == OutputFormat.HTML
        assert config.include_charts is True
        assert config.interactive_charts is False
        assert config.embed_assets is True
        assert config.include_raw_data is False
        assert config.theme == "light"

    def test_custom_config(self):
        """Test custom configuration."""
        config = RenderConfig(
            style=ReportStyle.SIMPLE,
            format=OutputFormat.JSON,
            include_charts=False,
            theme="dark"
        )

        assert config.style == ReportStyle.SIMPLE
        assert config.format == OutputFormat.JSON
        assert config.include_charts is False
        assert config.theme == "dark"

    def test_invalid_theme_raises_error(self):
        """Test that invalid theme raises ValueError."""
        with pytest.raises(ValueError, match="Invalid theme"):
            RenderConfig(theme="invalid")

    def test_static_interactive_conflict_raises_error(self):
        """Test that STATIC style with interactive charts raises error."""
        with pytest.raises(ValueError, match="STATIC style cannot have interactive_charts"):
            RenderConfig(
                style=ReportStyle.STATIC,
                interactive_charts=True
            )

    def test_simple_with_charts_warns(self):
        """Test that SIMPLE style with charts shows warning."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RenderConfig(
                style=ReportStyle.SIMPLE,
                include_charts=True
            )

            assert len(w) == 1
            assert "SIMPLE style" in str(w[0].message)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = RenderConfig(
            style=ReportStyle.FULL,
            format=OutputFormat.JSON
        )

        data = config.to_dict()

        assert isinstance(data, dict)
        assert data['style'] == 'full'
        assert data['format'] == 'json'
        assert 'metadata' in data

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            'style': 'simple',
            'format': 'html',
            'include_charts': False,
            'theme': 'dark'
        }

        config = RenderConfig.from_dict(data)

        assert config.style == ReportStyle.SIMPLE
        assert config.format == OutputFormat.HTML
        assert config.include_charts is False
        assert config.theme == "dark"


class TestPresetConfigs:
    """Test preset configurations."""

    def test_all_presets_defined(self):
        """Test that all expected presets exist."""
        assert 'full_interactive' in PRESET_CONFIGS
        assert 'simple_static' in PRESET_CONFIGS
        assert 'static_embedded' in PRESET_CONFIGS
        assert 'json_api' in PRESET_CONFIGS

    def test_get_preset_config(self):
        """Test getting preset configuration."""
        config = get_preset_config('full_interactive')

        assert isinstance(config, RenderConfig)
        assert config.style == ReportStyle.FULL
        assert config.interactive_charts is True

    def test_invalid_preset_raises_error(self):
        """Test that invalid preset name raises error."""
        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset_config('invalid_preset')

    def test_full_interactive_preset(self):
        """Test full_interactive preset."""
        config = get_preset_config('full_interactive')

        assert config.style == ReportStyle.FULL
        assert config.format == OutputFormat.HTML
        assert config.include_charts is True
        assert config.interactive_charts is True
        assert config.embed_assets is True

    def test_simple_static_preset(self):
        """Test simple_static preset."""
        config = get_preset_config('simple_static')

        assert config.style == ReportStyle.SIMPLE
        assert config.format == OutputFormat.HTML
        assert config.include_charts is False
        assert config.interactive_charts is False

    def test_json_api_preset(self):
        """Test json_api preset."""
        config = get_preset_config('json_api')

        assert config.style == ReportStyle.FULL
        assert config.format == OutputFormat.JSON
        assert config.include_charts is False
        assert config.include_raw_data is True
