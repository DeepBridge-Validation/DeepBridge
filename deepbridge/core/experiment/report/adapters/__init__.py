"""
Adapters for converting domain models to different output formats.

The Adapter pattern separates the domain model (WHAT to display) from the
rendering logic (HOW to display). This allows the same domain model to be
rendered in multiple formats without coupling the model to any specific format.

Architecture:
    Domain Model (Report) → Adapter → Output Format

Available Adapters:
    - HTMLAdapter: Converts Report to HTML using templates and ChartRegistry (Phase 3 Sprint 14)
    - JSONAdapter: Converts Report to JSON for APIs and storage (Phase 3 Sprint 14)
    - PDFAdapter: Converts Report to PDF using WeasyPrint (Phase 4 Sprint 19-21)
    - MarkdownAdapter: Converts Report to Markdown for documentation (Phase 4 Sprint 20-21)

Example Usage:
    >>> from deepbridge.core.experiment.report.domain import Report, ReportMetadata
    >>> from deepbridge.core.experiment.report.adapters import HTMLAdapter, JSONAdapter, PDFAdapter, MarkdownAdapter
    >>>
    >>> # Create a report using domain model
    >>> report = Report(metadata=ReportMetadata(...))
    >>>
    >>> # Render to HTML
    >>> html_adapter = HTMLAdapter(template_manager=tm, asset_manager=am)
    >>> html = html_adapter.render(report)
    >>>
    >>> # Export to JSON
    >>> json_adapter = JSONAdapter(indent=2)
    >>> json_str = json_adapter.render(report)
    >>>
    >>> # Generate PDF
    >>> pdf_adapter = PDFAdapter(template_manager=tm, asset_manager=am)
    >>> pdf_bytes = pdf_adapter.render(report)
    >>>
    >>> # Create Markdown
    >>> md_adapter = MarkdownAdapter()
    >>> markdown = md_adapter.render(report)

Benefits:
    - Separation of concerns (domain vs rendering)
    - Multiple output formats from same model
    - Easy to add new adapters
    - Type-safe with domain model validation
"""

from .base import ReportAdapter
from .html_adapter import HTMLAdapter
from .json_adapter import JSONAdapter
from .pdf_adapter import PDFAdapter
from .markdown_adapter import MarkdownAdapter

__all__ = [
    'ReportAdapter',
    'HTMLAdapter',
    'JSONAdapter',
    'PDFAdapter',
    'MarkdownAdapter',
]
