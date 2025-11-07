# 📊 Reports Testing Notebooks

This directory contains comprehensive notebooks for testing and demonstrating the new **Phase 4** report generation features.

## 📚 Available Notebooks

### 01. 📊 Uncertainty Report Generation
**File:** `01_uncertainty_report.ipynb`
**Level:** Intermediate
**Time:** ~15 minutes

**Topics Covered:**
- Running uncertainty quantification tests
- Generating HTML reports (traditional method)
- Introduction to Phase 4 adapters (PDF, Markdown)
- Comparing different report formats
- Understanding when to use each format

**What You'll Learn:**
- How to run uncertainty tests
- Generate reports in multiple formats
- Understand the new adapter pattern
- Choose the right format for your use case

---

### 02. 🛡️ Robustness Report Generation
**File:** `04_robustness_report.ipynb`
**Level:** Intermediate
**Time:** ~15 minutes

**Topics Covered:**
- Running robustness tests (feature perturbation)
- Evaluating model stability
- Generating HTML reports (traditional method)
- Introduction to Phase 4 adapters (PDF, Markdown)
- Understanding feature sensitivity

**What You'll Learn:**
- How to run robustness tests
- Measure prediction stability under perturbations
- Identify sensitive features
- Generate robustness reports in multiple formats
- Choose the right format for your use case

---

### 03. 🔄 Resilience Report Generation
**File:** `05_resilience_report.ipynb`
**Level:** Intermediate
**Time:** ~15 minutes

**Topics Covered:**
- Running resilience tests (distribution shift)
- Evaluating worst-case performance
- Generating HTML reports (traditional method)
- Introduction to Phase 4 adapters (PDF, Markdown)
- Understanding distribution shift impact

**What You'll Learn:**
- How to run resilience tests
- Evaluate model under distribution shifts
- Measure worst-case scenarios
- Generate resilience reports in multiple formats
- Monitor model performance degradation

---

### 04. 🎨 Multi-Format Report Generation
**File:** `02_multi_format_generation.ipynb`
**Level:** Advanced
**Time:** ~20 minutes

**Topics Covered:**
- Using Phase 4 adapters (PDF, Markdown, JSON, HTML)
- Creating presentation-agnostic domain models
- Generating the same report in multiple formats
- Customizing each format appropriately
- Format comparison and recommendations

**What You'll Learn:**
- Work with the new domain model
- Generate PDFs with WeasyPrint
- Create Markdown for documentation
- Export JSON for APIs
- Compare and choose formats

**Key Features Demonstrated:**
- ✅ PDFAdapter - Print-optimized reports
- ✅ MarkdownAdapter - GitHub/GitLab compatible
- ✅ JSONAdapter - Machine-readable export
- ✅ HTMLAdapter - Interactive reports

---

### 05. ⚡ Async Batch Report Generation
**File:** `03_async_batch_generation.ipynb`
**Level:** Advanced
**Time:** ~25 minutes

**Topics Covered:**
- Asynchronous report generation
- Batch processing with AsyncReportGenerator
- Progress tracking with real-time callbacks
- Error handling in async context
- Performance optimization
- Production-ready pipelines

**What You'll Learn:**
- Generate reports asynchronously (non-blocking)
- Process multiple reports in parallel (3-5x faster!)
- Track progress with callbacks
- Handle errors gracefully
- Build production pipelines
- Optimize for throughput

**Key Features Demonstrated:**
- ✅ AsyncReportGenerator - Parallel generation
- ✅ ReportTask - Task encapsulation
- ✅ ProgressTracker - Real-time monitoring
- ✅ Batch generation - Multiple reports at once
- ✅ Mixed formats - Different formats in same batch
- ✅ Error handling - Robust failure recovery

---

## 🚀 Getting Started

### Prerequisites

```bash
# Install DeepBridge with Phase 4 features
pip install deepbridge

# Additional dependencies for Phase 4
pip install weasyprint  # For PDF generation
```

### Running the Notebooks

1. **Start Jupyter:**
   ```bash
   jupyter notebook examples/notebooks/07_reports/
   ```

2. **Or use JupyterLab:**
   ```bash
   jupyter lab examples/notebooks/07_reports/
   ```

3. **Select a notebook** and run cells sequentially

### Recommended Order

**For Beginners - Test-Specific Reports:**
1. Start with **01_uncertainty_report.ipynb** - Learn uncertainty quantification
2. Try **04_robustness_report.ipynb** - Explore feature perturbation analysis
3. Then **05_resilience_report.ipynb** - Understand distribution shift handling

**For Advanced Users - Multi-Format & Async:**
4. Move to **02_multi_format_generation.ipynb** - Master the adapter pattern
5. Finish with **03_async_batch_generation.ipynb** - Production-ready batch processing

**Quick Path (Core Phase 4 Features):**
- **01** → **02** → **03** (Uncertainty basics → Multi-format → Async)

---

## 📋 Features Covered

### Phase 4 Innovations

#### 1. Multi-Format Support
```python
# Same report, multiple formats!
pdf_adapter = PDFAdapter()
md_adapter = MarkdownAdapter()
json_adapter = JSONAdapter()

pdf_bytes = pdf_adapter.render(report)  # PDF
markdown = md_adapter.render(report)     # Markdown
json_str = json_adapter.render(report)   # JSON
```

#### 2. Async Generation
```python
# Generate multiple reports in parallel
tasks = [
    {"adapter": PDFAdapter(), "report": report1, "output_path": "r1.pdf"},
    {"adapter": MarkdownAdapter(), "report": report2, "output_path": "r2.md"},
]

results = await generate_reports_async(tasks, max_workers=4)
# 3-5x faster than sequential!
```

#### 3. Presentation-Agnostic Domain Model
```python
# Define WHAT to display, not HOW
report = Report(metadata=ReportMetadata(...))
report.add_summary_metric(Metric(name="accuracy", value=0.95))
report.add_section(section)

# Render to any format
pdf_adapter.render(report)  # PDF
md_adapter.render(report)   # Markdown
html_adapter.render(report) # HTML
```

---

## 🎯 Learning Path

### Beginner → Intermediate (Test-Specific Reports)
1. **01_uncertainty_report.ipynb** - Start here!
   - Learn basic report generation
   - Understand different formats
   - See Phase 4 innovations
   - Uncertainty quantification basics

2. **04_robustness_report.ipynb** - Model stability
   - Feature perturbation analysis
   - Prediction stability metrics
   - Same structure as notebook 01

3. **05_resilience_report.ipynb** - Distribution shift
   - Distribution shift handling
   - Worst-case performance analysis
   - Same structure as notebook 01

### Intermediate → Advanced (Multi-Format Mastery)
4. **02_multi_format_generation.ipynb**
   - Work with adapters
   - Create domain models
   - Generate multiple formats
   - Advanced Phase 4 features

### Advanced → Production (Batch Processing)
5. **03_async_batch_generation.ipynb**
   - Async generation
   - Performance optimization
   - Production pipelines
   - 3-5x speedup with parallelization

---

## 📊 Format Comparison

| Format | Best For | Interactive | Printable | File Size | Phase |
|--------|----------|-------------|-----------|-----------|-------|
| **HTML** | Web viewing, exploration | ✅ | ❌ | Large | Legacy |
| **PDF** | Print, archival, regulatory | ❌ | ✅ | Medium | 4 |
| **Markdown** | Documentation, wikis | ❌ | ⚠️ | Small | 4 |
| **JSON** | APIs, programmatic access | ❌ | ❌ | Small | 3/4 |

---

## 💡 Tips & Best Practices

### 1. Choosing Formats

**For Stakeholders:**
- PDF for executives and regulatory
- HTML for technical team review

**For Documentation:**
- Markdown for GitHub/GitLab
- PDF for formal records

**For Automation:**
- JSON for APIs and pipelines
- Async generation for batch processing

### 2. Performance Optimization

**Sequential (Slow):**
```python
for report in reports:
    generate_report(report)  # One at a time
# Total: 10 reports × 2s = 20s
```

**Async Parallel (Fast):**
```python
await generate_reports_async(tasks, max_workers=4)
# Total: ~5s (4x speedup!)
```

### 3. Production Patterns

```python
# Production-ready pipeline
async def generate_daily_reports():
    models = load_models()
    tasks = create_tasks(models)

    results = await generate_reports_async(
        tasks,
        max_workers=6,
        progress_callback=log_progress
    )

    archive_reports(results)
    notify_stakeholders(results)
```

---

## 🆘 Troubleshooting

### Issue: WeasyPrint not installed
```bash
# Solution: Install WeasyPrint
pip install weasyprint

# On Linux, may need system dependencies:
sudo apt-get install libpango-1.0-0 libharfbuzz0b libpangoft2-1.0-0
```

### Issue: Async not working in Jupyter
```python
# Solution: Use await directly in Jupyter
# Jupyter notebooks support top-level await
result = await generate_report_async(...)
```

### Issue: Memory errors with large batches
```python
# Solution: Limit concurrent workers
generator = AsyncReportGenerator(max_workers=2)  # Reduce from 4
```

---

## 📚 Additional Resources

### Documentation
- [Phase 4 Features](../../../SPRINT19_26_PHASE4_COMPLETE.md)
- [Examples](../../../deepbridge/core/experiment/report/EXAMPLES_PHASE4.md)
- [Project Summary](../../../PROJECT_COMPLETE_SUMMARY.md)

### API Reference
- [PDFAdapter](../../../deepbridge/core/experiment/report/adapters/pdf_adapter.py)
- [MarkdownAdapter](../../../deepbridge/core/experiment/report/adapters/markdown_adapter.py)
- [AsyncReportGenerator](../../../deepbridge/core/experiment/report/async_generator.py)

---

## 🎉 What's New in Phase 4

### Multi-Format Adapters
- ✅ **PDF** - Professional print output with WeasyPrint
- ✅ **Markdown** - GitHub/GitLab compatible documentation
- ✅ **JSON** - Machine-readable API export
- ✅ **HTML** - Enhanced interactive reports

### Async Generation
- ✅ **Parallel processing** - 3-5x faster
- ✅ **Progress tracking** - Real-time callbacks
- ✅ **Batch generation** - Multiple reports at once
- ✅ **Error handling** - Graceful failure recovery

### Domain Model
- ✅ **Type-safe** - Pydantic validation
- ✅ **Presentation-agnostic** - Separate WHAT from HOW
- ✅ **Extensible** - Easy to add new formats
- ✅ **Testable** - Clean separation of concerns

---

## 🤝 Contributing

Found an issue or have suggestions? Please:
1. Check existing notebooks for examples
2. Review Phase 4 documentation
3. Open an issue with details
4. Submit a PR with improvements

---

## 📝 License

These notebooks are part of the DeepBridge project.

---

**Happy Testing! 🚀**

For questions or feedback, please refer to the main DeepBridge documentation.
