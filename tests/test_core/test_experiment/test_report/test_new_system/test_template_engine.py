"""Tests for template engine."""

import pytest
from pathlib import Path
import tempfile
import shutil
from jinja2 import TemplateNotFound

from deepbridge.core.experiment.report.templates.engine import (
    TemplateEngine,
    get_default_engine,
    DEFAULT_TEMPLATE_DIR
)


class TestTemplateEngine:
    """Test TemplateEngine class."""

    @pytest.fixture
    def temp_template_dir(self):
        """Create temporary template directory."""
        temp_dir = tempfile.mkdtemp()
        template_dir = Path(temp_dir)

        # Create sample templates
        (template_dir / "test.html").write_text(
            "<h1>{{ title }}</h1>"
        )
        (template_dir / "subdir").mkdir()
        (template_dir / "subdir" / "nested.html").write_text(
            "<p>{{ content }}</p>"
        )

        yield template_dir

        # Cleanup
        shutil.rmtree(temp_dir)

    def test_initialization_with_default_dir(self):
        """Test initialization with default template directory."""
        engine = TemplateEngine()

        assert engine.template_dir == DEFAULT_TEMPLATE_DIR
        assert engine.env is not None

    def test_initialization_with_custom_dir(self, temp_template_dir):
        """Test initialization with custom template directory."""
        engine = TemplateEngine(temp_template_dir)

        assert engine.template_dir == temp_template_dir
        assert engine.env is not None

    def test_initialization_with_nonexistent_dir(self):
        """Test initialization with non-existent directory raises error."""
        with pytest.raises(ValueError, match="Template directory does not exist"):
            TemplateEngine(Path("/nonexistent/path"))

    def test_get_template_success(self, temp_template_dir):
        """Test getting existing template."""
        engine = TemplateEngine(temp_template_dir)

        template = engine.get_template("test.html")

        assert template is not None
        assert hasattr(template, 'render')

    def test_get_template_nested(self, temp_template_dir):
        """Test getting template in subdirectory."""
        engine = TemplateEngine(temp_template_dir)

        template = engine.get_template("subdir/nested.html")

        assert template is not None

    def test_get_template_not_found(self, temp_template_dir):
        """Test getting non-existent template raises error."""
        engine = TemplateEngine(temp_template_dir)

        with pytest.raises(TemplateNotFound):
            engine.get_template("nonexistent.html")

    def test_render_simple(self, temp_template_dir):
        """Test rendering simple template."""
        engine = TemplateEngine(temp_template_dir)

        html = engine.render("test.html", {"title": "Test Title"})

        assert "<h1>Test Title</h1>" in html

    def test_render_nested(self, temp_template_dir):
        """Test rendering nested template."""
        engine = TemplateEngine(temp_template_dir)

        html = engine.render("subdir/nested.html", {"content": "Test Content"})

        assert "<p>Test Content</p>" in html

    def test_render_with_missing_variable(self, temp_template_dir):
        """Test rendering with missing variable (should not raise)."""
        engine = TemplateEngine(temp_template_dir)

        # Jinja2 by default doesn't raise on missing variables
        html = engine.render("test.html", {})

        assert "<h1></h1>" in html

    def test_render_string(self, temp_template_dir):
        """Test rendering template from string."""
        engine = TemplateEngine(temp_template_dir)

        template_str = "<div>{{ name }}</div>"
        html = engine.render_string(template_str, {"name": "Test"})

        assert "<div>Test</div>" in html

    def test_render_string_complex(self, temp_template_dir):
        """Test rendering complex template string."""
        engine = TemplateEngine(temp_template_dir)

        template_str = """
        {% for item in items %}
        <li>{{ item }}</li>
        {% endfor %}
        """
        html = engine.render_string(template_str, {"items": ["A", "B", "C"]})

        assert "<li>A</li>" in html
        assert "<li>B</li>" in html
        assert "<li>C</li>" in html

    def test_register_filter(self, temp_template_dir):
        """Test registering custom filter."""
        engine = TemplateEngine(temp_template_dir)

        def reverse_string(s):
            return s[::-1]

        engine.register_filter('reverse', reverse_string)

        html = engine.render_string("{{ text | reverse }}", {"text": "hello"})

        assert "olleh" in html

    def test_register_global(self, temp_template_dir):
        """Test registering global variable."""
        engine = TemplateEngine(temp_template_dir)

        engine.register_global('app_name', 'DeepBridge')

        html = engine.render_string("{{ app_name }}", {})

        assert "DeepBridge" in html

    def test_register_global_function(self, temp_template_dir):
        """Test registering global function."""
        engine = TemplateEngine(temp_template_dir)

        def get_greeting(name):
            return f"Hello, {name}!"

        engine.register_global('greet', get_greeting)

        html = engine.render_string("{{ greet('World') }}", {})

        assert "Hello, World!" in html

    def test_list_templates(self, temp_template_dir):
        """Test listing all templates."""
        engine = TemplateEngine(temp_template_dir)

        templates = engine.list_templates()

        assert len(templates) >= 2
        assert "test.html" in templates
        assert "subdir/nested.html" in templates

    def test_list_templates_with_pattern(self, temp_template_dir):
        """Test listing templates with pattern."""
        engine = TemplateEngine(temp_template_dir)

        templates = engine.list_templates("*.html")

        assert "test.html" in templates
        # Nested template should not match top-level pattern
        assert "subdir/nested.html" not in templates

    def test_list_templates_with_glob_pattern(self, temp_template_dir):
        """Test listing templates with glob pattern."""
        engine = TemplateEngine(temp_template_dir)

        templates = engine.list_templates("subdir/*.html")

        assert "subdir/nested.html" in templates
        assert "test.html" not in templates

    def test_autoescape_enabled(self, temp_template_dir):
        """Test that autoescape is enabled by default for HTML."""
        engine = TemplateEngine(temp_template_dir, autoescape=True)

        html = engine.render_string(
            "{{ content }}",
            {"content": "<script>alert('xss')</script>"}
        )

        # Script tags should be escaped
        assert "&lt;script&gt;" in html
        assert "<script>" not in html

    def test_autoescape_disabled(self, temp_template_dir):
        """Test that autoescape can be disabled."""
        engine = TemplateEngine(temp_template_dir, autoescape=False)

        html = engine.render_string(
            "{{ content }}",
            {"content": "<script>alert('xss')</script>"}
        )

        # Script tags should NOT be escaped
        assert "<script>" in html

    def test_trim_blocks_enabled(self, temp_template_dir):
        """Test that trim_blocks is enabled."""
        engine = TemplateEngine(temp_template_dir)

        template_str = """
        {% if true %}
        content
        {% endif %}
        """
        html = engine.render_string(template_str, {})

        # Should not have excessive whitespace
        assert html.strip() == "content"

    def test_builtin_global_now(self, temp_template_dir):
        """Test built-in 'now' global function."""
        engine = TemplateEngine(temp_template_dir)

        html = engine.render_string("{{ now() }}", {})

        # Should contain ISO format datetime
        assert len(html) > 0
        assert "T" in html  # ISO format has 'T' separator

    def test_builtin_global_version(self, temp_template_dir):
        """Test built-in 'version' global function."""
        engine = TemplateEngine(temp_template_dir)

        html = engine.render_string("{{ version() }}", {})

        # Should return version or "unknown"
        assert len(html) > 0


class TestGetDefaultEngine:
    """Test get_default_engine function."""

    def test_returns_engine_instance(self):
        """Test that it returns a TemplateEngine instance."""
        engine = get_default_engine()

        assert isinstance(engine, TemplateEngine)

    def test_uses_default_directory(self):
        """Test that it uses the default template directory."""
        engine = get_default_engine()

        assert engine.template_dir == DEFAULT_TEMPLATE_DIR

    def test_engine_is_functional(self):
        """Test that the returned engine can load templates."""
        engine = get_default_engine()

        # Should be able to list templates from actual template directory
        templates = engine.list_templates()

        # At minimum, should have some templates (if directory structure exists)
        assert isinstance(templates, list)


class TestTemplateEngineIntegration:
    """Integration tests for TemplateEngine."""

    @pytest.fixture
    def realistic_template_dir(self):
        """Create realistic template structure."""
        temp_dir = tempfile.mkdtemp()
        template_dir = Path(temp_dir)

        # Create base template
        (template_dir / "base.html").write_text("""
<!DOCTYPE html>
<html>
<head>
    <title>{% block title %}Default Title{% endblock %}</title>
</head>
<body>
    {% block content %}{% endblock %}
</body>
</html>
        """)

        # Create child template using inheritance
        (template_dir / "report.html").write_text("""
{% extends "base.html" %}

{% block title %}{{ report_title }}{% endblock %}

{% block content %}
<h1>{{ report_title }}</h1>
<p>{{ description }}</p>
{% endblock %}
        """)

        yield template_dir

        shutil.rmtree(temp_dir)

    def test_template_inheritance(self, realistic_template_dir):
        """Test that template inheritance works."""
        engine = TemplateEngine(realistic_template_dir)

        html = engine.render("report.html", {
            "report_title": "Robustness Report",
            "description": "Test description"
        })

        assert "<!DOCTYPE html>" in html
        assert "<title>Robustness Report</title>" in html
        assert "<h1>Robustness Report</h1>" in html
        assert "<p>Test description</p>" in html

    def test_multiple_renders(self, realistic_template_dir):
        """Test multiple renders with different contexts."""
        engine = TemplateEngine(realistic_template_dir)

        html1 = engine.render("report.html", {
            "report_title": "Report 1",
            "description": "Description 1"
        })

        html2 = engine.render("report.html", {
            "report_title": "Report 2",
            "description": "Description 2"
        })

        assert "Report 1" in html1
        assert "Report 2" in html2
        assert "Report 2" not in html1
        assert "Report 1" not in html2
