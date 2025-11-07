"""
HTML adapter for reports (Phase 3 Sprint 14).

Converts domain models to HTML using templates and ChartRegistry.
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
from .base import ReportAdapter
from ..domain.general import Report, ReportSection, Metric, ChartSpec

logger = logging.getLogger("deepbridge.reports")


class HTMLAdapter(ReportAdapter):
    """
    Adapter to convert Report domain model to HTML.

    Features:
    - Uses Jinja2 templates
    - Generates charts via ChartRegistry
    - Injects CSS/JS assets
    - Responsive design
    - Supports custom themes

    Example:
        >>> adapter = HTMLAdapter(template_manager, asset_manager)
        >>> html = adapter.render(report)
        >>> with open('report.html', 'w') as f:
        ...     f.write(html)
    """

    def __init__(
        self,
        template_manager=None,
        asset_manager=None,
        theme: str = "default",
        cache_manager=None
    ):
        """
        Initialize HTML adapter.

        Args:
            template_manager: Template manager for loading templates
            asset_manager: Asset manager for CSS/JS
            theme: Visual theme to use
            cache_manager: Optional cache manager for caching charts
        """
        self.template_manager = template_manager
        self.asset_manager = asset_manager
        self.theme = theme
        self.cache_manager = cache_manager

        # Import ChartRegistry
        from ..charts import ChartRegistry
        self.chart_registry = ChartRegistry

    def render(self, report: Report) -> str:
        """
        Render report to HTML string.

        Args:
            report: Report domain model

        Returns:
            HTML string

        Example:
            >>> html = adapter.render(report)
            >>> # Complete HTML document with styles and scripts
        """
        self._validate_report(report)

        # Generate charts from ChartSpecs
        charts = self._generate_charts(report)

        # Create template context
        context = self._create_context(report, charts)

        # Render template
        html = self._render_template(context, report)

        return html

    def _generate_charts(self, report: Report) -> Dict[str, str]:
        """
        Generate all charts from ChartSpecs using ChartRegistry.

        Args:
            report: Report with ChartSpecs

        Returns:
            Dictionary mapping chart IDs to chart content (base64 or Plotly JSON)
        """
        charts = {}

        for chart_spec in report.get_all_charts():
            try:
                # Try cache first if available
                if self.cache_manager:
                    cache_key = self.cache_manager.make_chart_key(
                        chart_spec.type.value,
                        chart_spec.data
                    )
                    cached_result = self.cache_manager.get_chart(cache_key)

                    if cached_result is not None:
                        charts[chart_spec.id] = cached_result
                        logger.info(f"Using cached chart: {chart_spec.id}")
                        continue

                # Generate chart using ChartRegistry
                result = self.chart_registry.generate(
                    chart_spec.type.value,  # ChartType enum value
                    chart_spec.data,
                    title=chart_spec.title,
                    **chart_spec.options
                )

                if result.is_success:
                    charts[chart_spec.id] = result.content
                    logger.info(f"Generated chart: {chart_spec.id}")

                    # Cache the result if cache_manager available
                    if self.cache_manager:
                        self.cache_manager.cache_chart(cache_key, result.content)
                else:
                    logger.error(f"Failed to generate chart {chart_spec.id}: {result.error}")
                    charts[chart_spec.id] = self._create_error_chart(chart_spec.title, result.error)

            except Exception as e:
                logger.error(f"Error generating chart {chart_spec.id}: {str(e)}")
                charts[chart_spec.id] = self._create_error_chart(chart_spec.title, str(e))

        return charts

    def _create_context(self, report: Report, charts: Dict[str, str]) -> Dict[str, Any]:
        """
        Create template context from report and charts.

        Args:
            report: Report domain model
            charts: Generated charts

        Returns:
            Template context dictionary
        """
        context = {
            # Report structure
            'report': report,
            'metadata': report.metadata,
            'sections': report.sections,
            'summary_metrics': report.summary_metrics,

            # Charts
            'charts': charts,

            # Metadata
            'title': report.display_title,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'theme': self.theme,

            # Helper functions
            'format_metric': self._format_metric_html,
            'get_chart': lambda chart_id: charts.get(chart_id, ''),
            'metric_status_class': self._get_metric_status_class,
        }

        # Add assets if available
        if self.asset_manager:
            context.update(self._get_assets(report))

        return context

    def _render_template(self, context: Dict[str, Any], report: Report) -> str:
        """
        Render HTML using template.

        Args:
            context: Template context
            report: Report (for finding template)

        Returns:
            Rendered HTML
        """
        # If template manager available, use it
        if self.template_manager:
            try:
                # Find appropriate template based on test type
                test_type = report.metadata.test_type.value
                template_paths = self.template_manager.get_template_paths(test_type, "static")
                template_path = self.template_manager.find_template(template_paths)

                if template_path:
                    template = self.template_manager.load_template(template_path)
                    return self.template_manager.render_template(template, context)
            except Exception as e:
                logger.warning(f"Could not use template manager: {e}")

        # Fallback: Generate simple HTML
        return self._generate_simple_html(context)

    def _generate_simple_html(self, context: Dict[str, Any]) -> str:
        """
        Generate simple HTML without templates (fallback).

        Args:
            context: Template context

        Returns:
            HTML string
        """
        report = context['report']

        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"<title>{context['title']}</title>",
            "<style>",
            self._get_default_css(),
            "</style>",
            "</head>",
            "<body>",
            f"<h1>{context['title']}</h1>",
            f"<p>Model: {report.metadata.model_name}</p>",
            f"<p>Test Type: {report.metadata.test_type.value}</p>",
            f"<p>Generated: {context['timestamp']}</p>",
        ]

        # Summary metrics
        if report.summary_metrics:
            html_parts.append("<h2>Summary</h2>")
            html_parts.append("<table>")
            for metric in report.summary_metrics:
                html_parts.append(
                    f"<tr><td>{metric.name}</td><td>{metric.formatted_value}</td></tr>"
                )
            html_parts.append("</table>")

        # Sections
        for section in report.sections:
            html_parts.append(self._render_section_html(section, context['charts']))

        html_parts.extend(["</body>", "</html>"])

        return "\n".join(html_parts)

    def _render_section_html(self, section: ReportSection, charts: Dict[str, str]) -> str:
        """Render a section as HTML."""
        html_parts = [
            f"<div class='section' id='{section.id}'>",
            f"<h2>{section.title}</h2>"
        ]

        if section.description:
            html_parts.append(f"<p>{section.description}</p>")

        # Metrics
        if section.metrics:
            html_parts.append("<h3>Metrics</h3>")
            html_parts.append("<table>")
            for metric in section.metrics:
                status_class = self._get_metric_status_class(metric)
                html_parts.append(
                    f"<tr class='{status_class}'>"
                    f"<td>{metric.name}</td>"
                    f"<td>{metric.formatted_value}</td>"
                    "</tr>"
                )
            html_parts.append("</table>")

        # Charts
        for chart_spec in section.charts:
            chart_content = charts.get(chart_spec.id, '')
            if chart_content:
                html_parts.append(f"<div class='chart'>")
                html_parts.append(f"<h3>{chart_spec.title}</h3>")
                html_parts.append(f"<div id='{chart_spec.id}'>{chart_content}</div>")
                html_parts.append("</div>")

        # Subsections
        for subsection in section.subsections:
            html_parts.append(self._render_section_html(subsection, charts))

        html_parts.append("</div>")

        return "\n".join(html_parts)

    def _format_metric_html(self, metric: Metric) -> str:
        """Format metric as HTML."""
        status_class = self._get_metric_status_class(metric)
        return f"<span class='{status_class}'>{metric.formatted_value}</span>"

    def _get_metric_status_class(self, metric: Metric) -> str:
        """Get CSS class based on metric status."""
        if metric.is_passing is True:
            return "metric-pass"
        elif metric.is_passing is False:
            return "metric-fail"
        else:
            return "metric-neutral"

    def _create_error_chart(self, title: str, error: str) -> str:
        """Create HTML for error chart."""
        return f"<div class='chart-error'><h4>{title}</h4><p>Error: {error}</p></div>"

    def _get_assets(self, report: Report) -> Dict[str, Any]:
        """Get CSS/JS assets from asset manager."""
        if not self.asset_manager:
            return {}

        try:
            test_type = report.metadata.test_type.value
            # This would use the actual asset manager methods
            return {
                'css_content': '',  # asset_manager would provide this
                'js_content': ''
            }
        except Exception as e:
            logger.warning(f"Could not load assets: {e}")
            return {}

    def _get_default_css(self) -> str:
        """Get default CSS styles."""
        return """
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        h1 { color: #2c3e50; }
        h2 { color: #34495e; margin-top: 30px; }
        h3 { color: #7f8c8d; }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        .metric-pass { color: #27ae60; font-weight: bold; }
        .metric-fail { color: #e74c3c; font-weight: bold; }
        .metric-neutral { color: #34495e; }
        .section {
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .chart {
            margin: 30px 0;
        }
        .chart-error {
            padding: 20px;
            background: #fee;
            border: 1px solid #fcc;
            border-radius: 4px;
            color: #c00;
        }
        """
