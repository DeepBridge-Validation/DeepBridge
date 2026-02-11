# Report Generation System

**Status:** ✅ **Production Ready** (Refactored - Phase 4)

Unified API for generating experiment reports in multiple formats (HTML, JSON).

---

## 🚀 Quick Start

```python
from deepbridge.core.experiment.report import ReportGenerator, RenderConfig, ReportStyle
from pathlib import Path

# Create generator (uses default templates)
generator = ReportGenerator()

# Generate full interactive HTML report
generator.generate_robustness_report(
    results=experiment.results,
    output_path=Path("reports/robustness.html")
)

# Generate simple static HTML report
generator.generate_robustness_report(
    results=experiment.results,
    output_path=Path("reports/robustness_simple.html"),
    config=RenderConfig(style=ReportStyle.SIMPLE)
)

# Generate JSON for API
generator.generate_robustness_report(
    results=experiment.results,
    output_path=Path("reports/robustness.json"),
    config=RenderConfig(format=OutputFormat.JSON)
)
```

---

## 📋 Supported Report Types

| Report Type | Method | Description |
|-------------|--------|-------------|
| **Robustness** | `generate_robustness_report()` | Model robustness to perturbations |
| **Resilience** | `generate_resilience_report()` | Model resilience metrics |
| **Uncertainty** | `generate_uncertainty_report()` | Uncertainty quantification |
| **Fairness** | `generate_fairness_report()` | Fairness and bias analysis |

---

## 🎨 Report Styles

### Full Interactive (Default)
- Interactive charts (Plotly)
- All metrics and details
- JavaScript enabled
```python
config = RenderConfig(style=ReportStyle.FULL, interactive_charts=True)
```

### Simple Static
- Static charts only
- Key metrics only
- Lightweight HTML
```python
config = RenderConfig(style=ReportStyle.SIMPLE)
```

### Static
- No JavaScript dependencies
- Self-contained HTML
- For offline/embedded use
```python
config = RenderConfig(style=ReportStyle.STATIC, interactive_charts=False)
```

---

## 📊 Output Formats

### HTML (Default)
```python
from deepbridge.core.experiment.report import OutputFormat

config = RenderConfig(format=OutputFormat.HTML)
generator.generate_robustness_report(results, Path("report.html"), config)
```

### JSON
```python
config = RenderConfig(format=OutputFormat.JSON)
generator.generate_robustness_report(results, Path("report.json"), config)
```

---

## 🔧 Configuration

### Using RenderConfig

```python
from deepbridge.core.experiment.report import RenderConfig, ReportStyle, OutputFormat

config = RenderConfig(
    style=ReportStyle.FULL,          # Report style
    format=OutputFormat.HTML,         # Output format
    include_charts=True,              # Include charts
    interactive_charts=True,          # Interactive (Plotly) vs static (images)
    embed_assets=True,                # Embed CSS/JS vs external links
    theme="default"                   # Color theme
)

generator.generate_robustness_report(results, output_path, config)
```

### Using Presets

```python
from deepbridge.core.experiment.report import get_preset_config

# Full interactive HTML
config = get_preset_config("full_interactive")

# Simple static HTML
config = get_preset_config("simple_static")

# JSON for API
config = get_preset_config("json_api")

generator.generate_robustness_report(results, output_path, config)
```

---

## 📂 Module Structure

```
deepbridge/core/experiment/report/
├── api.py                          # 🔵 ReportGenerator (main API)
├── config.py                       # 🔵 RenderConfig, enums
│
├── data/                           # 🔵 Data Layer (transformers)
│   ├── base.py                     # Base classes (ReportData, etc.)
│   ├── robustness.py               # Robustness data & transformer
│   ├── resilience.py               # Resilience data & transformer
│   ├── uncertainty.py              # Uncertainty data & transformer
│   └── fairness.py                 # Fairness data & transformer
│
├── renderers/                      # 🔵 Renderers
│   ├── base.py                     # ReportRenderer protocol
│   ├── html.py                     # HTMLRenderer
│   └── json.py                     # JSONRenderer
│
├── templates/                      # 🔵 Template engine
│   ├── engine.py                   # TemplateEngine (Jinja2)
│   ├── filters.py                  # Custom Jinja2 filters
│   └── html/                       # HTML templates
│       ├── base.html
│       ├── robustness/             # (full.html, simple.html, static.html)
│       ├── resilience/
│       ├── uncertainty/
│       ├── fairness/
│       └── shared/                 # (header.html, footer.html, charts.html)
│
├── assets/                         # Static assets
│   ├── manager.py                  # AssetManager
│   └── static/
│       ├── css/
│       ├── js/                     # External JavaScript
│       └── images/
│
└── utils/                          # Utilities
    ├── sanitizers.py
    └── validators.py
```

**Legend:**
- 🔵 = New unified system (recommended)
- ⚠️ = Old system (deprecated, see Migration Guide)

---

## 🧪 Testing

### Run All Tests
```bash
poetry run pytest tests/test_core/test_experiment/test_report/test_new_system/ -v
```

**Expected:** 121/121 tests passing (100% coverage)

### Coverage
```bash
poetry run pytest tests/test_core/test_experiment/test_report/test_new_system/ --cov=deepbridge.core.experiment.report --cov-report=html
```

---

## 🔄 Migration from Old System

If you're using the old renderer classes (`RobustnessRenderer`, `ResilienceRenderer`, etc.), see:

📖 **[Migration Guide](../../../../../desenvolvimento/refatorar/MIGRATION_GUIDE_REPORT_GENERATION.md)**

### Quick Migration Example

**Before (❌ Deprecated):**
```python
from deepbridge.core.experiment.report.renderers import RobustnessRenderer

renderer = RobustnessRenderer(template_manager, asset_manager)
renderer.render(results, "report.html", model_name="MyModel")
```

**After (✅ Recommended):**
```python
from deepbridge.core.experiment.report import ReportGenerator

generator = ReportGenerator()
generator.generate_robustness_report(results, Path("report.html"))
```

---

## 📚 Advanced Usage

### Custom Template Directory

```python
from pathlib import Path

generator = ReportGenerator(template_dir=Path("custom/templates"))
```

### Custom Transformers

```python
from deepbridge.core.experiment.report.data.base import DataTransformer

class MyCustomTransformer(DataTransformer):
    def transform(self, raw_data):
        # Custom transformation logic
        return custom_data

generator = ReportGenerator()
generator.add_transformer("custom", MyCustomTransformer())
generator.generate_report("custom", results, output_path)
```

### Custom Renderers

```python
from deepbridge.core.experiment.report.renderers.base import ReportRenderer

class MyCustomRenderer(ReportRenderer):
    def render(self, data, config):
        # Custom rendering logic
        return content

generator = ReportGenerator()
generator.add_renderer(OutputFormat.CUSTOM, MyCustomRenderer(template_engine))
```

---

## 🎯 Key Benefits

| Aspect | Old System | New System | Improvement |
|--------|-----------|------------|-------------|
| **Code Size** | ~14,000 lines | ~8,000 lines | -43% |
| **Code Duplication** | 20-30% | < 5% | -80% |
| **Largest File** | 2,538 lines | < 500 lines | -80% |
| **Test Coverage** | ~10% | 100% | +900% |
| **Type Safety** | None (dicts) | Full (dataclasses) | ✅ |
| **External JS** | Embedded | Separate files | ✅ |
| **JSON Output** | No | Yes | ✅ |
| **Configuration** | Hard-coded | Config-based | ✅ |

---

## 📖 Related Documentation

- **Refactoring Document**: [`01_REPORT_GENERATION.md`](../../../../../desenvolvimento/refatorar/01_REPORT_GENERATION.md)
- **Migration Guide**: [`MIGRATION_GUIDE_REPORT_GENERATION.md`](../../../../../desenvolvimento/refatorar/MIGRATION_GUIDE_REPORT_GENERATION.md)
- **Test Examples**: [`tests/test_core/test_experiment/test_report/test_new_system/`](../../../../../tests/test_core/test_experiment/test_report/test_new_system/)

---

## 📞 Support

### Issues
If you encounter issues with the new system:
1. Check the Migration Guide
2. Review test examples in `test_new_system/`
3. Check deprecation warnings in logs
4. Consult the refactoring document

### Contributing
When adding new report types:
1. Create dataclass in `data/<type>.py`
2. Create transformer class
3. Create templates in `templates/html/<type>/`
4. Add method to `ReportGenerator` in `api.py`
5. Write tests in `test_new_system/`

---

**Version:** 1.0.0 (Refactored)
**Last Updated:** 2026-02-10
**Status:** ✅ Production Ready
