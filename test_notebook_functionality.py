#!/usr/bin/env python3
"""
Quick test script to verify Phase 4 notebook functionality.
Tests the core features demonstrated in the notebooks.
"""

import sys
from pathlib import Path
from datetime import datetime

# Test imports
print("=" * 80)
print("Testing Phase 4 Imports...")
print("=" * 80)

try:
    # Test adapter imports
    from deepbridge.core.experiment.report.adapters import (
        PDFAdapter,
        MarkdownAdapter,
        JSONAdapter
    )
    print("✅ Adapters imported successfully")

    # Test domain model imports
    from deepbridge.core.experiment.report.domain import (
        Report,
        ReportMetadata,
        ReportType,
        ReportSection,
        Metric,
        MetricType
    )
    print("✅ Domain models imported successfully")

    # Test async generator imports
    from deepbridge.core.experiment.report import (
        AsyncReportGenerator,
        ReportTask,
        generate_report_async,
        generate_reports_async
    )
    print("✅ Async generators imported successfully")

except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test creating a simple report
print("\n" + "=" * 80)
print("Testing Report Creation...")
print("=" * 80)

try:
    metadata = ReportMetadata(
        model_name="Test Model",
        model_type="binary_classification",
        test_type=ReportType.UNCERTAINTY,
        created_at=datetime.now(),
        dataset_name="Test Dataset",
        dataset_size=100
    )
    print("✅ Metadata created")

    report = Report(
        metadata=metadata,
        title="Test Report",
        subtitle="Functionality Test"
    )
    print("✅ Report created")

    # Add metric
    report.add_summary_metric(
        Metric(
            name="Test Accuracy",
            value=0.95,
            type=MetricType.PERCENTAGE,
            is_primary=True
        )
    )
    print("✅ Metric added")

    # Add section
    section = ReportSection(
        id="results",
        title="Test Results"
    )
    section.add_metric(
        Metric(name="Score", value=95.0, type=MetricType.SCALAR)
    )
    report.add_section(section)
    print("✅ Section added")

except Exception as e:
    print(f"❌ Report creation error: {e}")
    sys.exit(1)

# Test PDF generation
print("\n" + "=" * 80)
print("Testing PDF Generation...")
print("=" * 80)

try:
    pdf_adapter = PDFAdapter(theme="professional", page_size="A4")
    pdf_bytes = pdf_adapter.render(report)
    print(f"✅ PDF rendered ({len(pdf_bytes)} bytes)")

    # Save to temp file
    output_dir = Path("/tmp/deepbridge_test")
    output_dir.mkdir(exist_ok=True)
    pdf_path = output_dir / "test_report.pdf"
    pdf_adapter.save_to_file(pdf_bytes, str(pdf_path))
    print(f"✅ PDF saved to {pdf_path}")

except Exception as e:
    print(f"❌ PDF generation error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test Markdown generation
print("\n" + "=" * 80)
print("Testing Markdown Generation...")
print("=" * 80)

try:
    md_adapter = MarkdownAdapter(include_toc=True)
    markdown = md_adapter.render(report)
    print(f"✅ Markdown rendered ({len(markdown)} characters)")

    md_path = output_dir / "test_report.md"
    md_adapter.save_to_file(markdown, str(md_path))
    print(f"✅ Markdown saved to {md_path}")

    # Show preview
    print("\n📄 Markdown Preview (first 300 chars):")
    print("-" * 80)
    print(markdown[:300])
    print("...")
    print("-" * 80)

except Exception as e:
    print(f"❌ Markdown generation error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test JSON generation
print("\n" + "=" * 80)
print("Testing JSON Generation...")
print("=" * 80)

try:
    json_adapter = JSONAdapter(indent=2)
    json_str = json_adapter.render(report)
    print(f"✅ JSON rendered ({len(json_str)} characters)")

    json_path = output_dir / "test_report.json"
    with open(json_path, 'w') as f:
        f.write(json_str)
    print(f"✅ JSON saved to {json_path}")

except Exception as e:
    print(f"❌ JSON generation error: {e}")
    sys.exit(1)

# Test async generation
print("\n" + "=" * 80)
print("Testing Async Generation...")
print("=" * 80)

try:
    import asyncio

    async def test_async():
        result = await generate_report_async(
            adapter=MarkdownAdapter(),
            report=report,
            output_path=str(output_dir / "test_async_report.md")
        )
        return result

    # Run async test
    result_path = asyncio.run(test_async())
    print(f"✅ Async generation successful: {result_path}")

except Exception as e:
    print(f"❌ Async generation error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 80)
print("🎉 All Tests Passed!")
print("=" * 80)
print("\n✨ Phase 4 functionality verified:")
print("   • PDF generation ✅")
print("   • Markdown generation ✅")
print("   • JSON generation ✅")
print("   • Async generation ✅")
print("   • Domain model ✅")
print(f"\n📁 Test outputs in: {output_dir}")
print("\n💡 Notebooks are ready to use!")
