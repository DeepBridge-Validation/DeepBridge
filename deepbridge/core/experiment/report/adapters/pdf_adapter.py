"""
PDF adapter for reports (Phase 4 Sprint 19-21).

Converts domain models to PDF format using WeasyPrint.
Generates PDF from HTML with optimized styling for print.
"""

from typing import Dict, Any, Optional
import logging
from pathlib import Path
from .base import ReportAdapter
from ..domain.general import Report

logger = logging.getLogger("deepbridge.reports")


class PDFAdapter(ReportAdapter):
    """
    Adapter to convert Report domain model to PDF.

    Features:
    - Uses WeasyPrint for HTML to PDF conversion
    - Print-optimized CSS (page breaks, fixed dimensions)
    - Static images for charts
    - Professional styling
    - A4 page format

    Example:
        >>> adapter = PDFAdapter(template_manager, asset_manager)
        >>> pdf_bytes = adapter.render(report)
        >>> # Write to file
        >>> with open('report.pdf', 'wb') as f:
        ...     f.write(pdf_bytes)
    """

    def __init__(
        self,
        template_manager=None,
        asset_manager=None,
        theme: str = "pdf",
        page_size: str = "A4",
        cache_manager=None
    ):
        """
        Initialize PDF adapter.

        Args:
            template_manager: Template manager for loading templates
            asset_manager: Asset manager for CSS/images
            theme: Visual theme to use (default: "pdf")
            page_size: Page size (A4, Letter, etc.)
            cache_manager: Optional cache manager for caching charts
        """
        self.template_manager = template_manager
        self.asset_manager = asset_manager
        self.theme = theme
        self.page_size = page_size
        self.cache_manager = cache_manager

        # Import ChartRegistry
        from ..charts import ChartRegistry
        self.chart_registry = ChartRegistry

    def render(self, report: Report) -> bytes:
        """
        Render report to PDF bytes.

        Args:
            report: Report domain model

        Returns:
            PDF document as bytes

        Example:
            >>> pdf_bytes = adapter.render(report)
            >>> len(pdf_bytes)  # Size in bytes
            125847
        """
        self._validate_report(report)

        # Step 1: Generate HTML optimized for PDF
        html_content = self._generate_pdf_html(report)

        # Step 2: Convert HTML to PDF
        pdf_bytes = self._html_to_pdf(html_content)

        logger.info(f"PDF generated successfully ({len(pdf_bytes)} bytes)")
        return pdf_bytes

    def _generate_pdf_html(self, report: Report) -> str:
        """
        Generate HTML optimized for PDF conversion.

        Key differences from web HTML:
        - Static images instead of interactive charts
        - Print-optimized CSS
        - Page break hints
        - Fixed dimensions
        - No JavaScript

        Args:
            report: Report domain model

        Returns:
            HTML string optimized for PDF
        """
        # Generate static charts (PNG/base64)
        charts = self._generate_static_charts(report)

        # Create PDF-specific context
        context = self._create_pdf_context(report, charts)

        # Render template
        html = self._render_pdf_template(context, report)

        return html

    def _generate_static_charts(self, report: Report) -> Dict[str, str]:
        """
        Generate all charts as static images (PNG) for PDF.

        Args:
            report: Report with ChartSpecs

        Returns:
            Dictionary mapping chart IDs to base64-encoded PNG images
        """
        charts = {}

        for chart_spec in report.get_all_charts():
            try:
                # Try cache first if available
                if self.cache_manager:
                    cache_key = self.cache_manager.make_chart_key(
                        f"{chart_spec.type.value}_static",
                        chart_spec.data
                    )
                    cached_result = self.cache_manager.get_chart(cache_key)

                    if cached_result is not None:
                        charts[chart_spec.id] = cached_result
                        logger.info(f"Using cached chart: {chart_spec.id}")
                        continue

                # Use static version of chart if available
                chart_type = self._get_static_chart_type(chart_spec.type.value)

                # Generate static chart
                result = self.chart_registry.generate(
                    chart_type,
                    chart_spec.data,
                    title=chart_spec.title,
                    **chart_spec.options
                )

                if result.is_success:
                    charts[chart_spec.id] = result.content
                    logger.info(f"Generated static chart: {chart_spec.id}")

                    # Cache the result
                    if self.cache_manager:
                        self.cache_manager.cache_chart(cache_key, result.content)
                else:
                    logger.error(f"Failed to generate chart {chart_spec.id}: {result.error}")
                    charts[chart_spec.id] = self._create_error_placeholder()

            except Exception as e:
                logger.error(f"Error generating chart {chart_spec.id}: {str(e)}")
                charts[chart_spec.id] = self._create_error_placeholder()

        return charts

    def _get_static_chart_type(self, chart_type: str) -> str:
        """
        Get static version of chart type for PDF.

        Args:
            chart_type: Original chart type

        Returns:
            Static chart type name
        """
        # Map interactive charts to static versions
        static_mapping = {
            "width_vs_coverage": "width_vs_coverage_static",
            "perturbation_impact": "perturbation_impact_static",
            # Add more mappings as needed
        }

        return static_mapping.get(chart_type, chart_type)

    def _create_pdf_context(self, report: Report, charts: Dict[str, str]) -> Dict[str, Any]:
        """
        Create template context for PDF generation.

        Args:
            report: Report domain model
            charts: Generated static charts

        Returns:
            Template context dictionary
        """
        context = {
            # Report structure
            "title": report.title,
            "subtitle": report.subtitle,
            "model_name": report.metadata.model_name,
            "test_type": report.metadata.test_type.value,
            "created_at": report.metadata.created_at,

            # Content
            "summary": self._format_summary(report.summary_metrics),
            "sections": self._format_sections(report.sections, charts),

            # Charts
            "charts": charts,

            # PDF-specific settings
            "pdf_mode": True,
            "page_size": self.page_size,
            "css_content": self._get_pdf_css(report.metadata.test_type.value),

            # Assets
            "logo": self._get_logo_data_uri(),
        }

        return context

    def _format_summary(self, summary_metrics: list) -> list:
        """Format summary metrics for template."""
        formatted = []
        for metric in summary_metrics:
            formatted.append({
                "name": metric.name,
                "value": self._format_metric_value(metric.value),
                "unit": metric.unit or "",
                "description": metric.description or ""
            })
        return formatted

    def _format_sections(self, sections: list, charts: Dict[str, str]) -> list:
        """Format sections with charts for template."""
        formatted = []

        for section in sections:
            section_data = {
                "id": section.id,
                "title": section.title,
                "description": section.description or "",
                "metrics": self._format_metrics(section.metrics),
                "charts": self._format_section_charts(section.charts, charts),
                "subsections": self._format_sections(section.subsections, charts) if section.subsections else []
            }
            formatted.append(section_data)

        return formatted

    def _format_metrics(self, metrics: list) -> list:
        """Format metrics for template."""
        return [{
            "name": m.name,
            "value": self._format_metric_value(m.value),
            "unit": m.unit or "",
            "description": m.description or ""
        } for m in metrics]

    def _format_section_charts(self, chart_specs: list, charts: Dict[str, str]) -> list:
        """Format charts for section."""
        formatted = []
        for chart_spec in chart_specs:
            if chart_spec.id in charts:
                formatted.append({
                    "id": chart_spec.id,
                    "title": chart_spec.title,
                    "image": charts[chart_spec.id],
                    "description": chart_spec.description or ""
                })
        return formatted

    def _format_metric_value(self, value) -> str:
        """Format metric value for display."""
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    def _get_pdf_css(self, test_type: str) -> str:
        """
        Get CSS optimized for PDF rendering.

        Includes:
        - Page size and margins
        - Page break rules
        - Print-safe colors
        - Fixed dimensions

        Args:
            test_type: Type of test report

        Returns:
            CSS string
        """
        # Get base CSS from CSSManager if available
        base_css = ""
        if hasattr(self, 'css_manager') and self.css_manager:
            try:
                from ..css_manager import CSSManager
                css_manager = CSSManager()
                base_css = css_manager.get_compiled_css(test_type)
            except Exception as e:
                logger.warning(f"Could not load base CSS: {e}")

        # PDF-specific CSS
        pdf_css = """
        /* PDF Page Settings */
        @page {
            size: A4;
            margin: 2cm;

            @top-right {
                content: "Page " counter(page) " of " counter(pages);
                font-size: 10pt;
                color: #666;
            }
        }

        /* Print Media Queries */
        @media print {
            .no-print {
                display: none !important;
            }

            .page-break {
                page-break-after: always;
            }

            .page-break-before {
                page-break-before: always;
            }

            .avoid-break {
                page-break-inside: avoid;
            }

            table {
                page-break-inside: avoid;
            }

            img {
                page-break-inside: avoid;
            }
        }

        /* Body and Typography */
        body {
            font-family: 'Helvetica', 'Arial', sans-serif;
            font-size: 11pt;
            line-height: 1.6;
            color: #333;
        }

        h1 {
            font-size: 24pt;
            margin-bottom: 10pt;
            color: #1a1a1a;
        }

        h2 {
            font-size: 18pt;
            margin-top: 15pt;
            margin-bottom: 8pt;
            color: #2a2a2a;
            border-bottom: 2pt solid #e0e0e0;
            padding-bottom: 5pt;
        }

        h3 {
            font-size: 14pt;
            margin-top: 12pt;
            margin-bottom: 6pt;
            color: #3a3a3a;
        }

        /* Charts */
        .chart {
            max-width: 100%;
            page-break-inside: avoid;
            margin: 15pt 0;
        }

        .chart img {
            max-width: 100%;
            height: auto;
        }

        /* Tables */
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 10pt 0;
            font-size: 10pt;
        }

        th {
            background-color: #f5f5f5;
            font-weight: bold;
            padding: 8pt;
            text-align: left;
            border-bottom: 2pt solid #ddd;
        }

        td {
            padding: 6pt 8pt;
            border-bottom: 1pt solid #e0e0e0;
        }

        /* Header/Footer */
        .pdf-header {
            margin-bottom: 20pt;
            padding-bottom: 10pt;
            border-bottom: 2pt solid #333;
        }

        .pdf-footer {
            margin-top: 20pt;
            padding-top: 10pt;
            border-top: 1pt solid #ddd;
            font-size: 9pt;
            color: #666;
        }

        /* Metrics Summary */
        .summary {
            background-color: #f9f9f9;
            padding: 15pt;
            margin: 15pt 0;
            border-left: 4pt solid #4CAF50;
        }

        /* Sections */
        section {
            margin: 20pt 0;
        }
        """

        return base_css + "\n\n" + pdf_css

    def _get_logo_data_uri(self) -> str:
        """
        Get logo as data URI for embedding in PDF.

        Returns:
            Data URI string or empty string if logo not found
        """
        # Try to get logo from asset_manager
        if self.asset_manager:
            try:
                logo_path = self.asset_manager.get_asset_path("logo.png")
                if logo_path and Path(logo_path).exists():
                    import base64
                    with open(logo_path, "rb") as f:
                        logo_data = base64.b64encode(f.read()).decode()
                    return f"data:image/png;base64,{logo_data}"
            except Exception as e:
                logger.warning(f"Could not load logo: {e}")

        return ""

    def _render_pdf_template(self, context: Dict[str, Any], report: Report) -> str:
        """
        Render PDF template with context.

        Args:
            context: Template context
            report: Report model

        Returns:
            Rendered HTML string
        """
        if not self.template_manager:
            # Fallback to simple HTML generation
            return self._generate_simple_html(context)

        # Try to load PDF-specific template
        template_paths = [
            f"report_types/{report.metadata.test_type.value}/pdf/index.html",
            f"report_types/{report.metadata.test_type.value}/static/index.html",
            "pdf/base.html"
        ]

        for template_path in template_paths:
            try:
                template = self.template_manager.get_template(template_path)
                return template.render(context)
            except Exception as e:
                logger.debug(f"Template {template_path} not found: {e}")
                continue

        # Fallback to simple HTML
        return self._generate_simple_html(context)

    def _generate_simple_html(self, context: Dict[str, Any]) -> str:
        """
        Generate simple HTML when templates are not available.

        Args:
            context: Template context

        Returns:
            HTML string
        """
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{context['title']}</title>
    <style>
        {context['css_content']}
    </style>
</head>
<body>
    <div class="pdf-header">
        {f'<img src="{context["logo"]}" alt="Logo" style="height: 40px;">' if context['logo'] else ''}
        <h1>{context['title']}</h1>
        <p><strong>{context['subtitle']}</strong></p>
        <p>Model: {context['model_name']} | Generated: {context['created_at'].strftime('%Y-%m-%d %H:%M')}</p>
    </div>

    <div class="summary">
        <h2>Summary</h2>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Unit</th>
                </tr>
            </thead>
            <tbody>
"""

        for metric in context['summary']:
            html += f"""
                <tr>
                    <td>{metric['name']}</td>
                    <td>{metric['value']}</td>
                    <td>{metric['unit']}</td>
                </tr>
"""

        html += """
            </tbody>
        </table>
    </div>
"""

        # Sections
        for section in context['sections']:
            html += f"""
    <section class="avoid-break">
        <h2>{section['title']}</h2>
        <p>{section['description']}</p>
"""

            # Charts
            for chart in section['charts']:
                html += f"""
        <div class="chart avoid-break">
            <h3>{chart['title']}</h3>
            <img src="{chart['image']}" alt="{chart['title']}">
            {f'<p>{chart["description"]}</p>' if chart.get('description') else ''}
        </div>
"""

            # Metrics
            if section['metrics']:
                html += """
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Unit</th>
                </tr>
            </thead>
            <tbody>
"""
                for metric in section['metrics']:
                    html += f"""
                <tr>
                    <td>{metric['name']}</td>
                    <td>{metric['value']}</td>
                    <td>{metric['unit']}</td>
                </tr>
"""
                html += """
            </tbody>
        </table>
"""

            html += """
    </section>
"""

        html += f"""
    <div class="pdf-footer">
        <p>Generated: {context['created_at'].strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
</body>
</html>
"""

        return html

    def _html_to_pdf(self, html_content: str) -> bytes:
        """
        Convert HTML to PDF using WeasyPrint.

        Args:
            html_content: HTML string to convert

        Returns:
            PDF bytes

        Raises:
            Exception: If PDF generation fails
        """
        try:
            from weasyprint import HTML, CSS
            from weasyprint.text.fonts import FontConfiguration

            # Create font configuration
            font_config = FontConfiguration()

            # Create HTML object
            html_doc = HTML(string=html_content)

            # Generate PDF
            pdf_bytes = html_doc.write_pdf(font_config=font_config)

            return pdf_bytes

        except Exception as e:
            logger.error(f"Failed to generate PDF: {str(e)}")
            raise

    def _create_error_placeholder(self) -> str:
        """
        Create error placeholder for charts that failed to generate.

        Returns:
            Base64-encoded error image
        """
        # Simple 1x1 transparent PNG
        return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

    def save_to_file(self, pdf_bytes: bytes, file_path: str) -> str:
        """
        Save PDF bytes to file.

        Args:
            pdf_bytes: PDF document bytes
            file_path: Output file path

        Returns:
            Absolute path to saved file
        """
        # Ensure directory exists
        output_path = Path(file_path).absolute()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write PDF
        with open(output_path, 'wb') as f:
            f.write(pdf_bytes)

        logger.info(f"PDF saved to: {output_path}")
        return str(output_path)
