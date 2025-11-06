"""
Async report generation (Phase 4 Sprint 25-26).

Provides asynchronous report generation for handling multiple reports
or long-running report generation tasks.

Features:
- Thread-based or process-based parallelism
- Progress tracking
- Batch generation
- Error handling
- Cancellation support
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, List, Any, Callable, Optional
import logging
from pathlib import Path
from datetime import datetime
from enum import Enum

logger = logging.getLogger("deepbridge.reports")


class ExecutorType(str, Enum):
    """Executor type enumeration."""
    THREAD = "thread"
    PROCESS = "process"


class TaskStatus(str, Enum):
    """Task status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ReportTask:
    """
    A single report generation task.

    Encapsulates all information needed to generate one report.
    """

    def __init__(
        self,
        task_id: str,
        adapter,
        report,
        output_path: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize report task.

        Args:
            task_id: Unique task identifier
            adapter: Report adapter instance (PDFAdapter, MarkdownAdapter, etc.)
            report: Report domain model to render
            output_path: Optional output path (for adapters that save to file)
            **kwargs: Additional arguments to pass to adapter
        """
        self.task_id = task_id
        self.adapter = adapter
        self.report = report
        self.output_path = output_path
        self.kwargs = kwargs

        self.status = TaskStatus.PENDING
        self.result = None
        self.error = None
        self.start_time = None
        self.end_time = None

    def __repr__(self):
        return f"ReportTask(id={self.task_id}, status={self.status})"


class ProgressTracker:
    """
    Track progress of async report generation.

    Provides hooks for monitoring progress and calling callbacks.
    """

    def __init__(self, total: int, callback: Optional[Callable] = None):
        """
        Initialize progress tracker.

        Args:
            total: Total number of tasks
            callback: Optional callback function(completed, total, task)
        """
        self.total = total
        self.completed = 0
        self.failed = 0
        self.cancelled = 0
        self.callback = callback
        self.tasks: Dict[str, ReportTask] = {}

    def register_task(self, task: ReportTask):
        """Register a task for tracking."""
        self.tasks[task.task_id] = task

    def update(self, task: ReportTask):
        """
        Update progress for a task.

        Args:
            task: Completed task
        """
        if task.status == TaskStatus.COMPLETED:
            self.completed += 1
        elif task.status == TaskStatus.FAILED:
            self.failed += 1
        elif task.status == TaskStatus.CANCELLED:
            self.cancelled += 1

        # Call callback if provided
        if self.callback:
            self.callback(self.completed, self.total, task)

    def percentage(self) -> float:
        """Get completion percentage."""
        if self.total == 0:
            return 100.0
        return (self.completed / self.total) * 100

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "total": self.total,
            "completed": self.completed,
            "failed": self.failed,
            "cancelled": self.cancelled,
            "percentage": self.percentage(),
            "pending": self.total - self.completed - self.failed - self.cancelled
        }


class AsyncReportGenerator:
    """
    Asynchronous report generator.

    Supports:
    - Thread-based concurrency (I/O bound tasks)
    - Process-based parallelism (CPU bound tasks)
    - Progress tracking
    - Batch generation
    - Error handling

    Example:
        >>> # Create generator
        >>> generator = AsyncReportGenerator(max_workers=4)
        >>>
        >>> # Create tasks
        >>> task1 = ReportTask("task1", PDFAdapter(), report1, "report1.pdf")
        >>> task2 = ReportTask("task2", MarkdownAdapter(), report2, "report2.md")
        >>>
        >>> # Generate reports
        >>> results = await generator.generate_batch([task1, task2])
    """

    def __init__(
        self,
        max_workers: int = 4,
        executor_type: ExecutorType = ExecutorType.THREAD
    ):
        """
        Initialize async generator.

        Args:
            max_workers: Maximum concurrent workers
            executor_type: Type of executor (THREAD or PROCESS)
        """
        self.max_workers = max_workers
        self.executor_type = executor_type

        # Create executor
        if executor_type == ExecutorType.PROCESS:
            self.executor = ProcessPoolExecutor(max_workers=max_workers)
            logger.info(f"Using ProcessPoolExecutor with {max_workers} workers")
        else:
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
            logger.info(f"Using ThreadPoolExecutor with {max_workers} workers")

    async def generate_single(self, task: ReportTask) -> ReportTask:
        """
        Generate single report asynchronously.

        Args:
            task: Report task to execute

        Returns:
            Completed task with result or error

        Example:
            >>> task = ReportTask("pdf1", PDFAdapter(), report, "output.pdf")
            >>> completed_task = await generator.generate_single(task)
            >>> if completed_task.status == TaskStatus.COMPLETED:
            ...     print(f"Report saved to: {completed_task.result}")
        """
        loop = asyncio.get_event_loop()

        task.status = TaskStatus.RUNNING
        task.start_time = datetime.now()

        try:
            # Run task in executor (thread/process pool)
            result = await loop.run_in_executor(
                self.executor,
                self._execute_task,
                task
            )

            task.result = result
            task.status = TaskStatus.COMPLETED
            logger.info(f"Task {task.task_id} completed successfully")

        except Exception as e:
            task.error = str(e)
            task.status = TaskStatus.FAILED
            logger.error(f"Task {task.task_id} failed: {str(e)}")

        finally:
            task.end_time = datetime.now()

        return task

    def _execute_task(self, task: ReportTask) -> Any:
        """
        Execute a report generation task.

        This runs in the executor (thread/process pool).

        Args:
            task: Task to execute

        Returns:
            Task result (PDF bytes, markdown string, file path, etc.)
        """
        logger.debug(f"Executing task {task.task_id}")

        # Render report
        result = task.adapter.render(task.report)

        # Save to file if output_path provided
        if task.output_path:
            if hasattr(task.adapter, 'save_to_file'):
                result = task.adapter.save_to_file(result, task.output_path)
            else:
                # Manual save for adapters without save_to_file
                output_path = Path(task.output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                if isinstance(result, bytes):
                    output_path.write_bytes(result)
                else:
                    output_path.write_text(str(result), encoding='utf-8')

                result = str(output_path.absolute())

        return result

    async def generate_batch(
        self,
        tasks: List[ReportTask],
        progress_callback: Optional[Callable] = None
    ) -> List[ReportTask]:
        """
        Generate multiple reports in parallel.

        Args:
            tasks: List of report tasks
            progress_callback: Optional callback(completed, total, task)

        Returns:
            List of completed tasks

        Example:
            >>> tasks = [
            ...     ReportTask("pdf1", PDFAdapter(), report1, "report1.pdf"),
            ...     ReportTask("pdf2", PDFAdapter(), report2, "report2.pdf"),
            ...     ReportTask("md1", MarkdownAdapter(), report3, "report3.md"),
            ... ]
            >>>
            >>> completed = await generator.generate_batch(
            ...     tasks,
            ...     progress_callback=lambda c, t, task: print(f"{c}/{t} complete")
            ... )
        """
        logger.info(f"Starting batch of {len(tasks)} reports")

        # Create progress tracker
        tracker = ProgressTracker(len(tasks), progress_callback)
        for task in tasks:
            tracker.register_task(task)

        # Create coroutines
        coroutines = [self.generate_single(task) for task in tasks]

        # Run in parallel
        completed_tasks = []
        for coro in asyncio.as_completed(coroutines):
            task = await coro
            tracker.update(task)
            completed_tasks.append(task)

        logger.info(f"Batch complete: {tracker.summary()}")
        return completed_tasks

    async def generate_with_limit(
        self,
        tasks: List[ReportTask],
        limit: int,
        progress_callback: Optional[Callable] = None
    ) -> List[ReportTask]:
        """
        Generate reports with concurrency limit.

        Useful for controlling memory usage or API rate limits.

        Args:
            tasks: List of report tasks
            limit: Maximum concurrent tasks
            progress_callback: Optional callback

        Returns:
            List of completed tasks
        """
        logger.info(f"Generating {len(tasks)} reports with limit={limit}")

        semaphore = asyncio.Semaphore(limit)
        tracker = ProgressTracker(len(tasks), progress_callback)

        async def limited_task(task: ReportTask) -> ReportTask:
            async with semaphore:
                tracker.register_task(task)
                result = await self.generate_single(task)
                tracker.update(result)
                return result

        completed = await asyncio.gather(*[limited_task(t) for t in tasks])
        logger.info(f"Limited batch complete: {tracker.summary()}")
        return list(completed)

    def shutdown(self, wait: bool = True):
        """
        Shutdown executor.

        Args:
            wait: Wait for pending tasks to complete
        """
        logger.info("Shutting down executor")
        self.executor.shutdown(wait=wait)

    def __del__(self):
        """Cleanup executor."""
        self.shutdown(wait=False)


# ==================================================================================
# Convenience Functions
# ==================================================================================

async def generate_report_async(
    adapter,
    report,
    output_path: Optional[str] = None
) -> Any:
    """
    Generate a single report asynchronously.

    Convenience function for generating one report.

    Args:
        adapter: Report adapter instance
        report: Report domain model
        output_path: Optional output file path

    Returns:
        Report result (bytes, string, or file path)

    Example:
        >>> from deepbridge.core.experiment.report.adapters import PDFAdapter
        >>> pdf_bytes = await generate_report_async(
        ...     PDFAdapter(),
        ...     my_report,
        ...     "output.pdf"
        ... )
    """
    generator = AsyncReportGenerator()
    task = ReportTask("single", adapter, report, output_path)
    result_task = await generator.generate_single(task)
    generator.shutdown()

    if result_task.status == TaskStatus.FAILED:
        raise Exception(f"Report generation failed: {result_task.error}")

    return result_task.result


async def generate_reports_async(
    tasks: List[Dict[str, Any]],
    max_workers: int = 4,
    progress_callback: Optional[Callable] = None
) -> List[Dict[str, Any]]:
    """
    Generate multiple reports asynchronously.

    Convenience function for batch generation.

    Args:
        tasks: List of task dicts with keys: adapter, report, output_path
        max_workers: Maximum concurrent workers
        progress_callback: Optional progress callback

    Returns:
        List of results

    Example:
        >>> tasks = [
        ...     {"adapter": PDFAdapter(), "report": report1, "output_path": "r1.pdf"},
        ...     {"adapter": MarkdownAdapter(), "report": report2, "output_path": "r2.md"},
        ... ]
        >>>
        >>> results = await generate_reports_async(tasks, max_workers=4)
    """
    generator = AsyncReportGenerator(max_workers=max_workers)

    # Create ReportTask objects
    report_tasks = []
    for i, task_dict in enumerate(tasks):
        task = ReportTask(
            task_id=f"task_{i}",
            adapter=task_dict["adapter"],
            report=task_dict["report"],
            output_path=task_dict.get("output_path")
        )
        report_tasks.append(task)

    # Generate
    completed_tasks = await generator.generate_batch(report_tasks, progress_callback)
    generator.shutdown()

    # Return results
    return [
        {
            "task_id": task.task_id,
            "status": task.status.value,
            "result": task.result,
            "error": task.error,
            "duration": (task.end_time - task.start_time).total_seconds() if task.end_time else None
        }
        for task in completed_tasks
    ]
