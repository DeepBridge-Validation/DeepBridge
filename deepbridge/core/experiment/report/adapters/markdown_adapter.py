"""
Markdown adapter for reports (Phase 4 Sprint 20-21).

Converts domain models to Markdown format for documentation and notebooks.
"""

from typing import Dict, Any, Optional, List
import logging
from pathlib import Path
from .base import ReportAdapter
from ..domain.general import Report, ReportSection, Metric, ChartSpec

logger = logging.getLogger("deepbridge.reports")


class MarkdownAdapter(ReportAdapter):
    """
    Adapter to convert Report domain model to Markdown.

    Features:
    - Clean markdown formatting
    - GitHub/GitLab compatible
    - Tables for metrics
    - Links to charts (if exported separately)
    - Hierarchical sections
    - Compatible with static site generators

    Use cases:
    - Documentation
    - Jupyter notebooks
    - GitHub/GitLab wikis
    - Static site generators (Hugo, Jekyll, etc.)
    - README files

    Example:
        >>> adapter = MarkdownAdapter()
        >>> markdown = adapter.render(report)
        >>> with open('report.md', 'w') as f:
        ...     f.write(markdown)
    """

    def __init__(
        self,
        include_toc: bool = True,
        heading_level_start: int = 1,
        chart_placeholder: str = "chart"
    ):
        """
        Initialize Markdown adapter.

        Args:
            include_toc: Include table of contents (default: True)
            heading_level_start: Starting heading level (default: 1 for #)
            chart_placeholder: How to handle charts ("chart", "link", "ignore")
        """
        self.include_toc = include_toc
        self.heading_level_start = heading_level_start
        self.chart_placeholder = chart_placeholder

    def render(self, report: Report) -> str:
        """
        Render report to Markdown string.

        Args:
            report: Report domain model

        Returns:
            Markdown string

        Example:
            >>> markdown = adapter.render(report)
            >>> print(markdown[:100])
            # Uncertainty Analysis Report
            **XGBoost Model**
            ...
        """
        self._validate_report(report)

        md = []

        # Title and metadata
        md.extend(self._render_header(report))

        # Table of contents
        if self.include_toc:
            md.extend(self._render_toc(report))

        # Summary
        md.extend(self._render_summary(report))

        # Sections
        md.extend(self._render_sections(report.sections, level=self.heading_level_start + 1))

        # Footer
        md.extend(self._render_footer(report))

        return "\n".join(md)

    def _render_header(self, report: Report) -> List[str]:
        """
        Render markdown header with title and metadata.

        Args:
            report: Report model

        Returns:
            List of markdown lines
        """
        md = []

        # Title
        md.append(f"{'#' * self.heading_level_start} {report.title}")
        md.append("")

        # Subtitle
        if report.subtitle:
            md.append(f"**{report.subtitle}**")
            md.append("")

        # Metadata
        md.append("## Metadata")
        md.append("")
        md.append(f"- **Model**: {report.metadata.model_name}")
        md.append(f"- **Model Type**: {report.metadata.model_type or 'N/A'}")
        md.append(f"- **Test Type**: {report.metadata.test_type.value}")
        md.append(f"- **Generated**: {report.metadata.created_at.strftime('%Y-%m-%d %H:%M:%S')}")

        if report.metadata.dataset_name:
            md.append(f"- **Dataset**: {report.metadata.dataset_name}")

        if report.metadata.tags:
            md.append(f"- **Tags**: {', '.join(report.metadata.tags)}")

        md.append("")
        md.append("---")
        md.append("")

        return md

    def _render_toc(self, report: Report) -> List[str]:
        """
        Render table of contents.

        Args:
            report: Report model

        Returns:
            List of markdown lines
        """
        md = []

        md.append("## Table of Contents")
        md.append("")

        # Summary
        md.append("- [Summary](#summary)")

        # Sections
        for section in report.sections:
            md.extend(self._render_toc_section(section, level=1))

        md.append("")
        md.append("---")
        md.append("")

        return md

    def _render_toc_section(self, section: ReportSection, level: int) -> List[str]:
        """
        Render TOC entry for section.

        Args:
            section: Section to render
            level: Indentation level

        Returns:
            List of markdown lines
        """
        md = []

        indent = "  " * level
        anchor = self._create_anchor(section.title)
        md.append(f"{indent}- [{section.title}](#{anchor})")

        # Subsections
        for subsection in section.subsections:
            md.extend(self._render_toc_section(subsection, level + 1))

        return md

    def _render_summary(self, report: Report) -> List[str]:
        """
        Render summary section.

        Args:
            report: Report model

        Returns:
            List of markdown lines
        """
        md = []

        md.append("## Summary")
        md.append("")

        if report.summary_metrics:
            md.extend(self._render_metrics_table(report.summary_metrics))
        else:
            md.append("*No summary metrics available.*")

        md.append("")

        return md

    def _render_sections(self, sections: List[ReportSection], level: int) -> List[str]:
        """
        Render all sections recursively.

        Args:
            sections: List of sections
            level: Heading level

        Returns:
            List of markdown lines
        """
        md = []

        for section in sections:
            md.extend(self._render_section(section, level))

        return md

    def _render_section(self, section: ReportSection, level: int) -> List[str]:
        """
        Render a single section.

        Args:
            section: Section to render
            level: Heading level

        Returns:
            List of markdown lines
        """
        md = []

        # Section title
        md.append(f"{'#' * level} {section.title}")
        md.append("")

        # Description
        if section.description:
            md.append(section.description)
            md.append("")

        # Metrics
        if section.metrics:
            md.append("### Metrics")
            md.append("")
            md.extend(self._render_metrics_table(section.metrics))
            md.append("")

        # Charts
        if section.charts:
            md.append("### Charts")
            md.append("")
            md.extend(self._render_charts(section.charts))
            md.append("")

        # Subsections
        if section.subsections:
            md.extend(self._render_sections(section.subsections, level + 1))

        return md

    def _render_metrics_table(self, metrics: List[Metric]) -> List[str]:
        """
        Render metrics as markdown table.

        Args:
            metrics: List of metrics

        Returns:
            List of markdown lines
        """
        if not metrics:
            return ["*No metrics available.*"]

        md = []

        # Table header
        md.append("| Metric | Value | Unit | Description |")
        md.append("|--------|-------|------|-------------|")

        # Table rows
        for metric in metrics:
            value = self._format_metric_value(metric.value)
            unit = metric.unit or "-"
            description = metric.description or "-"
            md.append(f"| {metric.name} | {value} | {unit} | {description} |")

        return md

    def _render_charts(self, charts: List[ChartSpec]) -> List[str]:
        """
        Render charts section.

        Args:
            charts: List of chart specifications

        Returns:
            List of markdown lines
        """
        md = []

        for chart in charts:
            md.append(f"#### {chart.title}")
            md.append("")

            if chart.description:
                md.append(chart.description)
                md.append("")

            # Chart placeholder
            if self.chart_placeholder == "chart":
                md.append(f"*Chart: {chart.type.value}*")
                md.append("")
                md.append("```")
                md.append(f"Chart Type: {chart.type.value}")
                md.append(f"Chart ID: {chart.id}")
                md.append("Note: Chart data available in JSON export or HTML report")
                md.append("```")
            elif self.chart_placeholder == "link":
                chart_filename = f"{chart.id}.png"
                md.append(f"![{chart.title}]({chart_filename})")
            elif self.chart_placeholder == "ignore":
                pass  # Don't include charts

            md.append("")

        return md

    def _render_footer(self, report: Report) -> List[str]:
        """
        Render footer.

        Args:
            report: Report model

        Returns:
            List of markdown lines
        """
        md = []

        md.append("---")
        md.append("")
        md.append("*Generated by DeepBridge*")
        md.append(f"*{report.metadata.created_at.strftime('%Y-%m-%d %H:%M:%S')}*")
        md.append("")

        return md

    def _format_metric_value(self, value: Any) -> str:
        """
        Format metric value for display.

        Args:
            value: Metric value

        Returns:
            Formatted string
        """
        if isinstance(value, float):
            # Format float with appropriate precision
            if abs(value) < 0.01 or abs(value) > 10000:
                return f"{value:.2e}"
            else:
                return f"{value:.4f}"
        elif isinstance(value, bool):
            return "Yes" if value else "No"
        elif isinstance(value, list):
            return ", ".join(str(v) for v in value[:5])  # Limit to 5 items
        else:
            return str(value)

    def _create_anchor(self, text: str) -> str:
        """
        Create anchor link from text (GitHub style).

        Args:
            text: Text to convert

        Returns:
            Anchor string
        """
        # Convert to lowercase
        anchor = text.lower()

        # Replace spaces with hyphens
        anchor = anchor.replace(" ", "-")

        # Remove special characters
        anchor = "".join(c for c in anchor if c.isalnum() or c == "-")

        return anchor

    def save_to_file(self, markdown: str, file_path: str) -> str:
        """
        Save markdown to file.

        Args:
            markdown: Markdown content
            file_path: Output file path

        Returns:
            Absolute path to saved file
        """
        # Ensure directory exists
        output_path = Path(file_path).absolute()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write markdown
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown)

        logger.info(f"Markdown saved to: {output_path}")
        return str(output_path)
