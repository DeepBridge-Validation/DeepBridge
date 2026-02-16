# 📚 Exemplos de Uso - Fase 4: Multi-Formato e Async

**Versão:** 2.0
**Data:** 06/11/2025

---

## 🎯 Visão Geral

Este documento fornece exemplos práticos de uso dos novos recursos da Fase 4:

- **PDFAdapter**: Geração de reports em PDF
- **MarkdownAdapter**: Geração de reports em Markdown
- **AsyncReportGenerator**: Geração assíncrona de reports

---

## 📄 Exemplo 1: Gerando PDF

### Básico

```python
from deepbridge.core.experiment.report.adapters import PDFAdapter
from deepbridge.core.experiment.report.domain import Report, ReportMetadata, ReportType

# Create report
metadata = ReportMetadata(
    model_name="XGBoost Classifier",
    model_type="classification",
    test_type=ReportType.UNCERTAINTY,
    dataset_name="MNIST"
)

report = Report(
    metadata=metadata,
    title="Uncertainty Analysis Report",
    subtitle="XGBoost Model on MNIST Dataset"
)

# Add metrics and sections
# ... (adicionar métricas e seções)

# Generate PDF
adapter = PDFAdapter()
pdf_bytes = adapter.render(report)

# Save to file
pdf_path = adapter.save_to_file(pdf_bytes, "outputs/uncertainty_report.pdf")
print(f"PDF saved to: {pdf_path}")
```

### Personalizado

```python
from deepbridge.core.experiment.report.adapters import PDFAdapter

# Create adapter with custom settings
adapter = PDFAdapter(
    theme="dark",
    page_size="Letter",  # or "A4", "A3", etc.
    cache_manager=my_cache_manager  # Optional cache
)

# Generate PDF
pdf_bytes = adapter.render(report)
adapter.save_to_file(pdf_bytes, "report_custom.pdf")
```

---

## 📝 Exemplo 2: Gerando Markdown

### Básico com TOC

```python
from deepbridge.core.experiment.report.adapters import MarkdownAdapter

# Create adapter with TOC
adapter = MarkdownAdapter(
    include_toc=True,
    heading_level_start=1
)

# Generate markdown
markdown = adapter.render(report)

# Save to file
md_path = adapter.save_to_file(markdown, "docs/report.md")
print(f"Markdown saved to: {md_path}")
```

### Com Links para Charts

```python
from deepbridge.core.experiment.report.adapters import MarkdownAdapter

# Adapter com links para charts
adapter = MarkdownAdapter(
    include_toc=True,
    chart_placeholder="link"  # Gera ![title](chart_id.png)
)

markdown = adapter.render(report)
adapter.save_to_file(markdown, "report_with_charts.md")
```

### Sem TOC para Notebooks

```python
from deepbridge.core.experiment.report.adapters import MarkdownAdapter

# Para uso em Jupyter notebooks
adapter = MarkdownAdapter(
    include_toc=False,
    heading_level_start=2,  # Start with ##
    chart_placeholder="ignore"  # Ignore charts
)

markdown = adapter.render(report)
print(markdown)
```

---

## ⚡ Exemplo 3: Geração Assíncrona - Single Report

### Método 1: Usando AsyncReportGenerator

```python
import asyncio
from deepbridge.core.experiment.report.async_generator import (
    AsyncReportGenerator,
    ReportTask
)
from deepbridge.core.experiment.report.adapters import PDFAdapter

async def generate_single_report():
    # Create generator
    generator = AsyncReportGenerator(max_workers=4)

    # Create task
    task = ReportTask(
        task_id="uncertainty_pdf",
        adapter=PDFAdapter(),
        report=my_report,
        output_path="outputs/report.pdf"
    )

    # Generate
    completed_task = await generator.generate_single(task)

    if completed_task.status == "completed":
        print(f"Report saved to: {completed_task.result}")
        print(f"Duration: {completed_task.end_time - completed_task.start_time}")
    else:
        print(f"Failed: {completed_task.error}")

    generator.shutdown()

# Run
asyncio.run(generate_single_report())
```

### Método 2: Usando Convenience Function

```python
import asyncio
from deepbridge.core.experiment.report.async_generator import generate_report_async
from deepbridge.core.experiment.report.adapters import PDFAdapter

async def quick_generate():
    # One-liner async generation
    result = await generate_report_async(
        adapter=PDFAdapter(),
        report=my_report,
        output_path="report.pdf"
    )
    print(f"PDF saved to: {result}")

asyncio.run(quick_generate())
```

---

## ⚡ Exemplo 4: Geração Assíncrona - Batch

### Gerando Múltiplos Reports em Paralelo

```python
import asyncio
from deepbridge.core.experiment.report.async_generator import (
    AsyncReportGenerator,
    ReportTask,
    ExecutorType
)
from deepbridge.core.experiment.report.adapters import (
    PDFAdapter,
    MarkdownAdapter,
    HTMLAdapter,
    JSONAdapter
)

async def generate_batch():
    # Create generator
    generator = AsyncReportGenerator(
        max_workers=4,
        executor_type=ExecutorType.THREAD  # or PROCESS for CPU-bound
    )

    # Create tasks for different formats
    tasks = [
        # PDF
        ReportTask("pdf", PDFAdapter(), report, "outputs/report.pdf"),

        # Markdown
        ReportTask("markdown", MarkdownAdapter(), report, "outputs/report.md"),

        # HTML
        ReportTask("html", HTMLAdapter(), report, "outputs/report.html"),

        # JSON
        ReportTask("json", JSONAdapter(), report, "outputs/report.json"),
    ]

    # Generate all in parallel
    completed = await generator.generate_batch(tasks)

    # Check results
    for task in completed:
        if task.status == "completed":
            print(f"✅ {task.task_id}: {task.result}")
        else:
            print(f"❌ {task.task_id}: {task.error}")

    generator.shutdown()

asyncio.run(generate_batch())
```

### Com Progress Tracking

```python
import asyncio
from deepbridge.core.experiment.report.async_generator import (
    AsyncReportGenerator,
    ReportTask
)

def progress_callback(completed, total, task):
    """Progress callback."""
    percentage = (completed / total) * 100
    print(f"Progress: {completed}/{total} ({percentage:.1f}%) - {task.task_id}")

async def generate_with_progress():
    generator = AsyncReportGenerator(max_workers=4)

    # Create 10 tasks
    tasks = [
        ReportTask(f"report_{i}", PDFAdapter(), reports[i], f"output_{i}.pdf")
        for i in range(10)
    ]

    # Generate with progress tracking
    completed = await generator.generate_batch(
        tasks,
        progress_callback=progress_callback
    )

    print(f"✅ All {len(completed)} reports generated!")
    generator.shutdown()

asyncio.run(generate_with_progress())
```

---

## ⚡ Exemplo 5: Batch com Convenience Function

### Mais Simples e Direto

```python
import asyncio
from deepbridge.core.experiment.report.async_generator import generate_reports_async
from deepbridge.core.experiment.report.adapters import PDFAdapter, MarkdownAdapter

async def batch_generate():
    # Define tasks as dicts
    tasks = [
        {
            "adapter": PDFAdapter(),
            "report": uncertainty_report,
            "output_path": "outputs/uncertainty.pdf"
        },
        {
            "adapter": MarkdownAdapter(),
            "report": robustness_report,
            "output_path": "outputs/robustness.md"
        },
        {
            "adapter": PDFAdapter(),
            "report": resilience_report,
            "output_path": "outputs/resilience.pdf"
        }
    ]

    # Generate all
    results = await generate_reports_async(
        tasks,
        max_workers=4,
        progress_callback=lambda c, t, task: print(f"{c}/{t} complete")
    )

    # Print results
    for result in results:
        print(f"{result['task_id']}: {result['status']} - {result['result']}")
        print(f"  Duration: {result['duration']:.2f}s")

asyncio.run(batch_generate())
```

---

## 🔄 Exemplo 6: Multi-Formato do Mesmo Report

### Gerando Todos os Formatos

```python
import asyncio
from deepbridge.core.experiment.report.async_generator import generate_reports_async
from deepbridge.core.experiment.report.adapters import (
    PDFAdapter,
    MarkdownAdapter,
    HTMLAdapter,
    JSONAdapter
)

async def export_all_formats(report, output_dir):
    """Export report to all available formats."""

    tasks = [
        # PDF for printing
        {
            "adapter": PDFAdapter(),
            "report": report,
            "output_path": f"{output_dir}/report.pdf"
        },

        # Markdown for documentation
        {
            "adapter": MarkdownAdapter(include_toc=True),
            "report": report,
            "output_path": f"{output_dir}/report.md"
        },

        # HTML for web viewing
        {
            "adapter": HTMLAdapter(),
            "report": report,
            "output_path": f"{output_dir}/report.html"
        },

        # JSON for API/storage
        {
            "adapter": JSONAdapter(indent=2),
            "report": report,
            "output_path": f"{output_dir}/report.json"
        }
    ]

    results = await generate_reports_async(tasks, max_workers=4)

    print(f"✅ Exported {len(results)} formats:")
    for r in results:
        print(f"  - {r['result']}")

    return results

# Use
asyncio.run(export_all_formats(my_report, "outputs"))
```

---

## 🎛️ Exemplo 7: Controle de Concorrência

### Limitando Workers para Controle de Memória

```python
import asyncio
from deepbridge.core.experiment.report.async_generator import AsyncReportGenerator, ReportTask
from deepbridge.core.experiment.report.adapters import PDFAdapter

async def generate_with_limit():
    generator = AsyncReportGenerator(max_workers=8)

    # 100 reports
    tasks = [
        ReportTask(f"report_{i}", PDFAdapter(), reports[i], f"output_{i}.pdf")
        for i in range(100)
    ]

    # Generate with max 5 concurrent tasks
    completed = await generator.generate_with_limit(
        tasks,
        limit=5,
        progress_callback=lambda c, t, _: print(f"Progress: {c}/{t}")
    )

    print(f"✅ Generated {len(completed)} reports with limit=5")
    generator.shutdown()

asyncio.run(generate_with_limit())
```

---

## 🛠️ Exemplo 8: Error Handling

### Tratando Erros em Batch

```python
import asyncio
from deepbridge.core.experiment.report.async_generator import (
    generate_reports_async,
    TaskStatus
)

async def safe_batch_generate():
    tasks = [
        {"adapter": PDFAdapter(), "report": report1, "output_path": "r1.pdf"},
        {"adapter": PDFAdapter(), "report": report2, "output_path": "r2.pdf"},
        {"adapter": PDFAdapter(), "report": report3, "output_path": "r3.pdf"},
    ]

    results = await generate_reports_async(tasks)

    # Separate successful and failed
    successful = [r for r in results if r["status"] == "completed"]
    failed = [r for r in results if r["status"] == "failed"]

    print(f"✅ Successful: {len(successful)}")
    print(f"❌ Failed: {len(failed)}")

    # Handle failures
    for failure in failed:
        print(f"Failed task {failure['task_id']}: {failure['error']}")

    return successful

asyncio.run(safe_batch_generate())
```

---

## 📊 Exemplo 9: Uso Completo - Pipeline de Reports

### Pipeline Completo: Teste → Report → Multi-Formato

```python
import asyncio
from deepbridge.core.experiment.report.domain import (
    Report, ReportMetadata, ReportType, ReportSection, Metric
)
from deepbridge.core.experiment.report.adapters import (
    PDFAdapter, MarkdownAdapter, HTMLAdapter
)
from deepbridge.core.experiment.report.async_generator import generate_reports_async

async def complete_pipeline():
    # 1. Run tests (simulado)
    test_results = run_uncertainty_test(model, X_test, y_test)

    # 2. Create domain model
    metadata = ReportMetadata(
        model_name="XGBoost",
        test_type=ReportType.UNCERTAINTY,
        dataset_name="MNIST"
    )

    report = Report(
        metadata=metadata,
        title="Uncertainty Analysis",
        subtitle="XGBoost on MNIST"
    )

    # Add summary metrics
    report.add_summary_metric(Metric(name="coverage", value=0.92))
    report.add_summary_metric(Metric(name="mean_width", value=1.23))

    # Add sections
    section = ReportSection(id="results", title="Test Results")
    section.add_metric(Metric(name="accuracy", value=0.95))
    report.add_section(section)

    # 3. Generate multiple formats asynchronously
    tasks = [
        # PDF for stakeholders
        {
            "adapter": PDFAdapter(),
            "report": report,
            "output_path": "outputs/stakeholders/uncertainty_report.pdf"
        },

        # Markdown for documentation
        {
            "adapter": MarkdownAdapter(include_toc=True),
            "report": report,
            "output_path": "docs/reports/uncertainty.md"
        },

        # HTML for interactive viewing
        {
            "adapter": HTMLAdapter(),
            "report": report,
            "output_path": "outputs/web/uncertainty.html"
        }
    ]

    print("Generating reports in parallel...")
    results = await generate_reports_async(
        tasks,
        max_workers=3,
        progress_callback=lambda c, t, _: print(f"  {c}/{t} complete")
    )

    print("\n✅ Pipeline complete!")
    for r in results:
        print(f"  - {r['result']} ({r['duration']:.2f}s)")

# Run pipeline
asyncio.run(complete_pipeline())
```

---

## 🎓 Melhores Práticas

### 1. Escolha do Executor

```python
# Para I/O bound (file writes, network):
generator = AsyncReportGenerator(
    max_workers=10,
    executor_type=ExecutorType.THREAD
)

# Para CPU bound (heavy computations):
generator = AsyncReportGenerator(
    max_workers=4,  # ~= CPU cores
    executor_type=ExecutorType.PROCESS
)
```

### 2. Controle de Memória

```python
# Gerar muitos PDFs? Limite concorrência:
await generator.generate_with_limit(tasks, limit=3)
```

### 3. Progress Tracking

```python
# Sempre use callback para batches grandes:
def progress(completed, total, task):
    print(f"[{datetime.now()}] {completed}/{total} - {task.task_id}")

await generator.generate_batch(tasks, progress_callback=progress)
```

### 4. Error Handling

```python
# Sempre verifique status:
for task in completed:
    if task.status != TaskStatus.COMPLETED:
        logger.error(f"Task {task.task_id} failed: {task.error}")
```

---

## 📚 Recursos Adicionais

- **Documentação API:** `FASE_4_EXTENSAO.md`
- **Testes:** `tests/report/adapters/test_pdf_markdown_adapters.py`
- **Async Tests:** `tests/report/test_async_generator.py`
- **Domain Model:** `deepbridge/core/experiment/report/domain/general.py`

---

**Última atualização:** 06/11/2025
**Versão:** 2.0
