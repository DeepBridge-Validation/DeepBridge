# Migration Guide: Report Generation System

## 🎯 Overview

This guide helps you migrate from the **old report generation system** to the **new unified API**.

### What Changed?

**Before (❌ Deprecated):**
- Multiple separate renderer classes: `RobustnessRenderer`, `RobustnessRendererSimple`, `StaticRobustnessRenderer`, etc.
- Separate transformer modules for each variant
- No type safety (nested dictionaries)
- JavaScript embedded in Python code
- ~14,000 lines of code with 20-30% duplication

**After (✅ New System):**
- Single unified API: `ReportGenerator`
- Configuration-based style selection: `RenderConfig`
- Type-safe dataclasses
- External templates and JavaScript
- ~8,000 lines (-43% code reduction)
- 121/121 tests passing (100% coverage)

---

## 🚀 Quick Migration

### 1. Basic Report Generation

#### Old Code (Deprecated)
```python
from deepbridge.core.experiment.report.renderers import RobustnessRenderer

renderer = RobustnessRenderer(template_manager, asset_manager)
renderer.render(results, "report.html", model_name="MyModel")
```

#### New Code (✅ Recommended)
```python
from deepbridge.core.experiment.report import ReportGenerator
from pathlib import Path

generator = ReportGenerator()
generator.generate_robustness_report(
    results=results,
    output_path=Path("report.html")
)
```

---

### 2. Simple Reports

#### Old Code (Deprecated)
```python
from deepbridge.core.experiment.report.renderers import RobustnessRendererSimple

renderer = RobustnessRendererSimple(template_manager, asset_manager)
renderer.render(results, "report_simple.html", model_name="MyModel")
```

#### New Code (✅ Recommended)
```python
from deepbridge.core.experiment.report import ReportGenerator, RenderConfig, ReportStyle

generator = ReportGenerator()
generator.generate_robustness_report(
    results=results,
    output_path=Path("report_simple.html"),
    config=RenderConfig(style=ReportStyle.SIMPLE)
)
```

---

### 3. Static Reports

#### Old Code (Deprecated)
```python
from deepbridge.core.experiment.report.renderers.static import StaticRobustnessRenderer

renderer = StaticRobustnessRenderer(template_manager, asset_manager)
renderer.render(results, "report_static.html", model_name="MyModel")
```

#### New Code (✅ Recommended)
```python
from deepbridge.core.experiment.report import ReportGenerator, RenderConfig, ReportStyle

generator = ReportGenerator()
generator.generate_robustness_report(
    results=results,
    output_path=Path("report_static.html"),
    config=RenderConfig(style=ReportStyle.STATIC, interactive_charts=False)
)
```

---

### 4. JSON Output

#### Old Code (No equivalent)
```python
# JSON output was not supported in old system
```

#### New Code (✅ New Feature!)
```python
from deepbridge.core.experiment.report import ReportGenerator, RenderConfig, OutputFormat

generator = ReportGenerator()
generator.generate_robustness_report(
    results=results,
    output_path=Path("report.json"),
    config=RenderConfig(format=OutputFormat.JSON)
)
```

---

## 📋 Migration by Report Type

### Robustness Reports

| Old Class | New API Method | Config |
|-----------|----------------|--------|
| `RobustnessRenderer` | `generate_robustness_report()` | `RenderConfig(style=ReportStyle.FULL)` |
| `RobustnessRendererSimple` | `generate_robustness_report()` | `RenderConfig(style=ReportStyle.SIMPLE)` |
| `StaticRobustnessRenderer` | `generate_robustness_report()` | `RenderConfig(style=ReportStyle.STATIC)` |

**Example:**
```python
from deepbridge.core.experiment.report import ReportGenerator, RenderConfig, ReportStyle

generator = ReportGenerator()

# Full interactive report
generator.generate_robustness_report(
    results=experiment.results,
    output_path=Path("reports/robustness_full.html"),
    config=RenderConfig(style=ReportStyle.FULL, interactive_charts=True)
)

# Simple report
generator.generate_robustness_report(
    results=experiment.results,
    output_path=Path("reports/robustness_simple.html"),
    config=RenderConfig(style=ReportStyle.SIMPLE)
)

# Static report
generator.generate_robustness_report(
    results=experiment.results,
    output_path=Path("reports/robustness_static.html"),
    config=RenderConfig(style=ReportStyle.STATIC)
)
```

---

### Resilience Reports

| Old Class | New API Method | Config |
|-----------|----------------|--------|
| `ResilienceRenderer` | `generate_resilience_report()` | `RenderConfig(style=ReportStyle.FULL)` |
| `ResilienceRendererSimple` | `generate_resilience_report()` | `RenderConfig(style=ReportStyle.SIMPLE)` |
| `StaticResilienceRenderer` | `generate_resilience_report()` | `RenderConfig(style=ReportStyle.STATIC)` |

**Example:**
```python
generator.generate_resilience_report(
    results=experiment.results,
    output_path=Path("reports/resilience.html"),
    config=RenderConfig(style=ReportStyle.FULL)
)
```

---

### Uncertainty Reports

| Old Class | New API Method | Config |
|-----------|----------------|--------|
| `UncertaintyRenderer` | `generate_uncertainty_report()` | `RenderConfig(style=ReportStyle.FULL)` |
| `UncertaintyRendererSimple` | `generate_uncertainty_report()` | `RenderConfig(style=ReportStyle.SIMPLE)` |
| `StaticUncertaintyRenderer` | `generate_uncertainty_report()` | `RenderConfig(style=ReportStyle.STATIC)` |

**Example:**
```python
generator.generate_uncertainty_report(
    results=experiment.results,
    output_path=Path("reports/uncertainty.html"),
    config=RenderConfig(style=ReportStyle.FULL)
)
```

---

### Fairness Reports

| Old Class | New API Method | Config |
|-----------|----------------|--------|
| `FairnessRendererSimple` | `generate_fairness_report()` | `RenderConfig(style=ReportStyle.SIMPLE)` |

**Example:**
```python
generator.generate_fairness_report(
    results=experiment.results,
    output_path=Path("reports/fairness.html"),
    config=RenderConfig(style=ReportStyle.SIMPLE)
)
```

---

## 🔧 Configuration Options

### RenderConfig Parameters

```python
from deepbridge.core.experiment.report import RenderConfig, ReportStyle, OutputFormat

config = RenderConfig(
    style=ReportStyle.FULL,           # FULL, SIMPLE, STATIC, INTERACTIVE
    format=OutputFormat.HTML,          # HTML, JSON
    include_charts=True,               # Include charts in report
    interactive_charts=True,           # Make charts interactive (requires JavaScript)
    embed_assets=True,                 # Embed CSS/JS or link to external files
    theme="default"                    # Color theme: default, dark, light
)
```

### Preset Configurations

Use preset configurations for common scenarios:

```python
from deepbridge.core.experiment.report import get_preset_config

# Full interactive HTML report
config = get_preset_config("full_interactive")

# Simple static HTML report
config = get_preset_config("simple_static")

# JSON for API
config = get_preset_config("json_api")
```

---

## 🧪 Testing Your Migration

### 1. Run Regression Tests

```bash
# Test new system
poetry run pytest tests/test_core/test_experiment/test_report/test_new_system/ -v

# Expected: 121/121 tests passing
```

### 2. Compare Output

Generate reports with both old and new systems and compare:

```python
# Generate with old system (for comparison only)
from deepbridge.core.experiment.report.renderers import RobustnessRenderer
old_renderer = RobustnessRenderer(template_manager, asset_manager)
old_renderer.render(results, "report_old.html")

# Generate with new system
from deepbridge.core.experiment.report import ReportGenerator
generator = ReportGenerator()
generator.generate_robustness_report(results, Path("report_new.html"))

# Compare visually or with diff tools
```

---

## 🚨 Breaking Changes

### 1. API Signature Changes

**Old:**
```python
renderer.render(results, file_path, model_name, report_type, save_chart)
```

**New:**
```python
generator.generate_robustness_report(results, output_path, config)
```

**Migration:**
- `file_path` → `output_path` (now a `Path` object)
- `model_name` → included in `results` dict
- `report_type` → `config.style`
- `save_chart` → removed (not needed with new template system)

---

### 2. Import Changes

**Old:**
```python
from deepbridge.core.experiment.report.renderers import (
    RobustnessRenderer,
    RobustnessRendererSimple,
    ResilienceRenderer,
    UncertaintyRenderer,
)
```

**New:**
```python
from deepbridge.core.experiment.report import (
    ReportGenerator,
    RenderConfig,
    ReportStyle,
    OutputFormat,
)
```

---

### 3. Template Manager Not Required

**Old:**
```python
from deepbridge.core.experiment.report.template_manager import TemplateManager
from deepbridge.core.experiment.report.asset_manager import AssetManager

template_manager = TemplateManager(templates_dir)
asset_manager = AssetManager(templates_dir)
renderer = RobustnessRenderer(template_manager, asset_manager)
```

**New:**
```python
# Template and asset management handled internally
generator = ReportGenerator()  # Uses default templates

# Or specify custom template directory
generator = ReportGenerator(template_dir=Path("custom/templates"))
```

---

## 📚 Advanced Usage

### Custom Transformers

```python
from deepbridge.core.experiment.report import ReportGenerator
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
        return html_content

generator = ReportGenerator()
generator.add_renderer(OutputFormat.HTML, MyCustomRenderer(template_engine))
```

---

## 🐛 Troubleshooting

### Issue: DeprecationWarning in Logs

**Cause:** You're still using old renderers.

**Solution:** Migrate to new `ReportGenerator` API (see Quick Migration above).

---

### Issue: Template Not Found

**Cause:** Custom template directory doesn't have required templates.

**Solution:**
```python
# Use default templates
generator = ReportGenerator()  # Don't specify template_dir

# Or ensure your custom directory has the structure:
# custom/templates/html/
#   ├── robustness/
#   │   ├── full.html
#   │   ├── simple.html
#   │   └── static.html
#   ├── resilience/...
#   └── shared/...
```

---

### Issue: Missing AssetManager

**Cause:** AssetManager not found or not imported.

**Solution:**
```python
# AssetManager is optional and loaded automatically
generator = ReportGenerator()  # Works without explicit AssetManager

# To use custom AssetManager:
from deepbridge.core.experiment.report.asset_manager import AssetManager
asset_manager = AssetManager(templates_dir)
generator = ReportGenerator(asset_manager=asset_manager)
```

---

## 📞 Support

### Documentation
- **New System Documentation**: `/desenvolvimento/refatorar/01_REPORT_GENERATION.md`
- **API Reference**: See docstrings in `deepbridge/core/experiment/report/api.py`

### Tests
- **Test Examples**: `tests/test_core/test_experiment/test_report/test_new_system/`
- **Run Tests**: `poetry run pytest tests/test_core/test_experiment/test_report/test_new_system/`

### Getting Help
If you encounter issues during migration:
1. Check this migration guide
2. Review test examples in `test_new_system/`
3. Check deprecation warnings in logs
4. Consult the refactoring document: `01_REPORT_GENERATION.md`

---

## ✅ Migration Checklist

- [ ] Replace all `RobustnessRenderer` imports with `ReportGenerator`
- [ ] Replace all `ResilienceRenderer` imports with `ReportGenerator`
- [ ] Replace all `UncertaintyRenderer` imports with `ReportGenerator`
- [ ] Replace all `FairnessRendererSimple` imports with `ReportGenerator`
- [ ] Remove explicit `TemplateManager` and `AssetManager` initialization
- [ ] Update method calls from `.render()` to `.generate_*_report()`
- [ ] Convert file paths from strings to `Path` objects
- [ ] Replace `report_type` parameter with `RenderConfig(style=...)`
- [ ] Update imports to use new API
- [ ] Run regression tests to verify output
- [ ] Remove deprecated imports from codebase
- [ ] Update internal documentation/README

---

## 📊 Benefits Summary

| Aspect | Old System | New System | Improvement |
|--------|-----------|------------|-------------|
| **Code Lines** | ~14,000 | ~8,000 | -43% |
| **Files > 1000 lines** | 5 | 0 | -100% |
| **Code Duplication** | 20-30% | < 5% | -80% |
| **Test Coverage** | ~10% | 100% | +900% |
| **Largest Method** | 486 lines | < 50 lines | -90% |
| **Type Safety** | None | Full | ✅ |
| **JavaScript Testing** | Impossible | Possible | ✅ |
| **JSON Output** | No | Yes | ✅ |

---

**Last Updated:** 2026-02-10
**DeepBridge Version:** 1.0.0+
**Refactoring Phase:** 4 (Cleanup)
