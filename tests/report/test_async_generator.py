"""
Tests for async report generation (Phase 4 Sprint 25-26).
"""

import asyncio
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from deepbridge.core.experiment.report.adapters import (
    MarkdownAdapter,
    PDFAdapter,
)
from deepbridge.core.experiment.report.async_generator import (
    AsyncReportGenerator,
    ExecutorType,
    ProgressTracker,
    ReportTask,
    TaskStatus,
    generate_report_async,
    generate_reports_async,
)
from deepbridge.core.experiment.report.domain.general import (
    Metric,
    MetricType,
    Report,
    ReportMetadata,
    ReportSection,
    ReportType,
)

# ==================================================================================
# Fixtures
# ==================================================================================


@pytest.fixture
def sample_report():
    """Create a sample report for testing."""
    metadata = ReportMetadata(
        model_name='TestModel',
        test_type=ReportType.UNCERTAINTY,
        created_at=datetime(2025, 11, 6, 12, 0, 0),
    )

    report = Report(
        metadata=metadata, title='Test Report', subtitle='Async Test'
    )

    report.add_summary_metric(
        Metric(name='accuracy', value=0.95, type=MetricType.PERCENTAGE)
    )

    section = ReportSection(id='test', title='Test Section')
    section.add_metric(Metric(name='metric1', value=1.0))
    report.add_section(section)

    return report


# ==================================================================================
# ReportTask Tests
# ==================================================================================


class TestReportTask:
    """Tests for ReportTask."""

    def test_task_creation(self, sample_report):
        """Test task can be created."""
        adapter = MarkdownAdapter()
        task = ReportTask('task1', adapter, sample_report, 'output.md')

        assert task.task_id == 'task1'
        assert task.adapter == adapter
        assert task.report == sample_report
        assert task.output_path == 'output.md'
        assert task.status == TaskStatus.PENDING
        assert task.result is None
        assert task.error is None

    def test_task_repr(self, sample_report):
        """Test task string representation."""
        task = ReportTask('task1', MarkdownAdapter(), sample_report)
        assert 'task1' in repr(task)
        assert 'pending' in repr(task).lower()


# ==================================================================================
# ProgressTracker Tests
# ==================================================================================


class TestProgressTracker:
    """Tests for ProgressTracker."""

    def test_tracker_creation(self):
        """Test tracker can be created."""
        tracker = ProgressTracker(total=10)

        assert tracker.total == 10
        assert tracker.completed == 0
        assert tracker.failed == 0
        assert tracker.cancelled == 0

    def test_tracker_update_completed(self, sample_report):
        """Test updating tracker with completed task."""
        tracker = ProgressTracker(total=5)

        task = ReportTask('task1', MarkdownAdapter(), sample_report)
        task.status = TaskStatus.COMPLETED
        tracker.register_task(task)
        tracker.update(task)

        assert tracker.completed == 1
        assert tracker.failed == 0

    def test_tracker_update_failed(self, sample_report):
        """Test updating tracker with failed task."""
        tracker = ProgressTracker(total=5)

        task = ReportTask('task1', MarkdownAdapter(), sample_report)
        task.status = TaskStatus.FAILED
        tracker.register_task(task)
        tracker.update(task)

        assert tracker.completed == 0
        assert tracker.failed == 1

    def test_tracker_percentage(self):
        """Test percentage calculation."""
        tracker = ProgressTracker(total=10)

        assert tracker.percentage() == 0.0

        tracker.completed = 5
        assert tracker.percentage() == 50.0

        tracker.completed = 10
        assert tracker.percentage() == 100.0

    def test_tracker_summary(self):
        """Test summary statistics."""
        tracker = ProgressTracker(total=10)
        tracker.completed = 7
        tracker.failed = 2
        tracker.cancelled = 1

        summary = tracker.summary()

        assert summary['total'] == 10
        assert summary['completed'] == 7
        assert summary['failed'] == 2
        assert summary['cancelled'] == 1
        assert summary['percentage'] == 70.0
        assert summary['pending'] == 0

    def test_tracker_with_callback(self, sample_report):
        """Test tracker with callback."""
        calls = []

        def callback(completed, total, task):
            calls.append((completed, total, task.task_id))

        tracker = ProgressTracker(total=3, callback=callback)

        task1 = ReportTask('task1', MarkdownAdapter(), sample_report)
        task1.status = TaskStatus.COMPLETED
        tracker.update(task1)

        assert len(calls) == 1
        assert calls[0] == (1, 3, 'task1')


# ==================================================================================
# AsyncReportGenerator Tests
# ==================================================================================


class TestAsyncReportGenerator:
    """Tests for AsyncReportGenerator."""

    def test_generator_creation(self):
        """Test generator can be created."""
        generator = AsyncReportGenerator(max_workers=2)

        assert generator.max_workers == 2
        assert generator.executor_type == ExecutorType.THREAD
        assert generator.executor is not None

    def test_generator_with_process_executor(self):
        """Test generator with process executor."""
        generator = AsyncReportGenerator(
            max_workers=2, executor_type=ExecutorType.PROCESS
        )

        assert generator.executor_type == ExecutorType.PROCESS

    @pytest.mark.asyncio
    async def test_generate_single_markdown(self, sample_report):
        """Test generating single markdown report."""
        generator = AsyncReportGenerator()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'test.md'

            task = ReportTask(
                'task1', MarkdownAdapter(), sample_report, str(output_path)
            )

            result_task = await generator.generate_single(task)

            assert result_task.status == TaskStatus.COMPLETED
            assert result_task.result is not None
            assert result_task.error is None
            assert Path(result_task.result).exists()

        generator.shutdown()

    @pytest.mark.asyncio
    async def test_generate_single_pdf(self, sample_report):
        """Test generating single PDF report."""
        generator = AsyncReportGenerator()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'test.pdf'

            task = ReportTask(
                'task1', PDFAdapter(), sample_report, str(output_path)
            )

            result_task = await generator.generate_single(task)

            assert result_task.status == TaskStatus.COMPLETED
            assert result_task.result is not None
            assert Path(result_task.result).exists()
            assert Path(result_task.result).read_bytes()[:4] == b'%PDF'

        generator.shutdown()

    @pytest.mark.asyncio
    async def test_generate_batch(self, sample_report):
        """Test batch generation."""
        generator = AsyncReportGenerator(max_workers=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            tasks = [
                ReportTask(
                    'task1',
                    MarkdownAdapter(),
                    sample_report,
                    str(Path(tmpdir) / 'r1.md'),
                ),
                ReportTask(
                    'task2',
                    MarkdownAdapter(),
                    sample_report,
                    str(Path(tmpdir) / 'r2.md'),
                ),
                ReportTask(
                    'task3',
                    PDFAdapter(),
                    sample_report,
                    str(Path(tmpdir) / 'r3.pdf'),
                ),
            ]

            completed_tasks = await generator.generate_batch(tasks)

            assert len(completed_tasks) == 3
            assert all(
                t.status == TaskStatus.COMPLETED for t in completed_tasks
            )
            assert all(Path(t.result).exists() for t in completed_tasks)

        generator.shutdown()

    @pytest.mark.asyncio
    async def test_generate_batch_with_progress(self, sample_report):
        """Test batch generation with progress callback."""
        generator = AsyncReportGenerator(max_workers=2)

        progress_updates = []

        def progress_callback(completed, total, task):
            progress_updates.append((completed, total))

        with tempfile.TemporaryDirectory() as tmpdir:
            tasks = [
                ReportTask(
                    'task1',
                    MarkdownAdapter(),
                    sample_report,
                    str(Path(tmpdir) / f'r{i}.md'),
                )
                for i in range(5)
            ]

            completed_tasks = await generator.generate_batch(
                tasks, progress_callback
            )

            assert len(completed_tasks) == 5
            assert len(progress_updates) == 5
            assert progress_updates[-1] == (5, 5)

        generator.shutdown()

    @pytest.mark.asyncio
    async def test_generate_with_limit(self, sample_report):
        """Test generation with concurrency limit."""
        generator = AsyncReportGenerator(max_workers=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            tasks = [
                ReportTask(
                    'task1',
                    MarkdownAdapter(),
                    sample_report,
                    str(Path(tmpdir) / f'r{i}.md'),
                )
                for i in range(10)
            ]

            completed_tasks = await generator.generate_with_limit(
                tasks, limit=2
            )

            assert len(completed_tasks) == 10
            assert all(
                t.status == TaskStatus.COMPLETED for t in completed_tasks
            )

        generator.shutdown()

    @pytest.mark.asyncio
    async def test_task_timing(self, sample_report):
        """Test task timing information."""
        generator = AsyncReportGenerator()

        task = ReportTask('task1', MarkdownAdapter(), sample_report)
        result_task = await generator.generate_single(task)

        assert result_task.start_time is not None
        assert result_task.end_time is not None
        assert result_task.end_time > result_task.start_time

        generator.shutdown()


# ==================================================================================
# Convenience Function Tests
# ==================================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.asyncio
    async def test_generate_report_async(self, sample_report):
        """Test generate_report_async function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'test.md'

            result = await generate_report_async(
                MarkdownAdapter(), sample_report, str(output_path)
            )

            assert result is not None
            assert Path(result).exists()

    @pytest.mark.asyncio
    async def test_generate_reports_async(self, sample_report):
        """Test generate_reports_async function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tasks = [
                {
                    'adapter': MarkdownAdapter(),
                    'report': sample_report,
                    'output_path': str(Path(tmpdir) / f'r{i}.md'),
                }
                for i in range(3)
            ]

            results = await generate_reports_async(tasks, max_workers=2)

            assert len(results) == 3
            assert all(r['status'] == 'completed' for r in results)
            assert all(r['result'] is not None for r in results)
            assert all(r['duration'] is not None for r in results)

    @pytest.mark.asyncio
    async def test_generate_reports_async_with_progress(self, sample_report):
        """Test generate_reports_async with progress callback."""
        progress_calls = []

        def progress(completed, total, task):
            progress_calls.append(completed)

        with tempfile.TemporaryDirectory() as tmpdir:
            tasks = [
                {
                    'adapter': MarkdownAdapter(),
                    'report': sample_report,
                    'output_path': str(Path(tmpdir) / f'r{i}.md'),
                }
                for i in range(5)
            ]

            results = await generate_reports_async(
                tasks, max_workers=2, progress_callback=progress
            )

            assert len(results) == 5
            assert len(progress_calls) == 5


# ==================================================================================
# Integration Tests
# ==================================================================================


class TestAsyncIntegration:
    """Integration tests for async generation."""

    @pytest.mark.asyncio
    async def test_mixed_format_generation(self, sample_report):
        """Test generating multiple formats asynchronously."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tasks = [
                {
                    'adapter': MarkdownAdapter(),
                    'report': sample_report,
                    'output_path': str(Path(tmpdir) / 'report.md'),
                },
                {
                    'adapter': PDFAdapter(),
                    'report': sample_report,
                    'output_path': str(Path(tmpdir) / 'report.pdf'),
                },
            ]

            results = await generate_reports_async(tasks)

            assert len(results) == 2
            assert all(r['status'] == 'completed' for r in results)

            # Verify files exist
            assert Path(tmpdir, 'report.md').exists()
            assert Path(tmpdir, 'report.pdf').exists()

            # Verify content
            md_content = Path(tmpdir, 'report.md').read_text()
            assert 'Test Report' in md_content

            pdf_content = Path(tmpdir, 'report.pdf').read_bytes()
            assert pdf_content[:4] == b'%PDF'
