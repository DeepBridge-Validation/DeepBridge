# DeepBridge Renderers - Quick Reference Guide

## File Locations

```
/home/guhaase/projetos/DeepBridge/deepbridge/core/experiment/report/renderers/
├── base_renderer.py                    (678 LOC - Base class)
├── uncertainty_renderer_simple.py       (113 LOC - Inherits)
├── robustness_renderer_simple.py        (121 LOC - Inherits)
├── resilience_renderer_simple.py        (112 LOC - Inherits)
├── fairness_renderer_simple.py          (145 LOC - Inherits + custom filter)
├── hyperparameter_renderer.py           (111 LOC - Composition, NOT inheriting)
└── template_manager.py, asset_manager.py (supporting files)
```

## Renderer Summary

### 1. UncertaintyRendererSimple (113 LOC)
- **Location:** `/deepbridge/core/experiment/report/renderers/uncertainty_renderer_simple.py`
- **Status:** Excellent (inherits from BaseRenderer)
- **Key Context Fields:** uncertainty_score, total_alphas, avg_coverage, avg_width
- **Transformer:** UncertaintyDataTransformerSimple
- **Eliminated Duplication:** 180 LOC

### 2. RobustnessRendererSimple (121 LOC)
- **Location:** `/deepbridge/core/experiment/report/renderers/robustness_renderer_simple.py`
- **Status:** Excellent (inherits from BaseRenderer)
- **Key Context Fields:** robustness_score, base_score, avg_impact, metric
- **Advanced Features:** WeakSpot analysis, Overfitting analysis
- **Transformer:** RobustnessDataTransformerSimple
- **Eliminated Duplication:** 130 LOC

### 3. ResilienceRendererSimple (112 LOC)
- **Location:** `/deepbridge/core/experiment/report/renderers/resilience_renderer_simple.py`
- **Status:** Excellent (inherits from BaseRenderer)
- **Key Context Fields:** resilience_score, total_scenarios, valid_scenarios
- **Transformer:** ResilienceDataTransformerSimple
- **Eliminated Duplication:** 115 LOC

### 4. FairnessRendererSimple (145 LOC)
- **Location:** `/deepbridge/core/experiment/report/renderers/fairness_renderer_simple.py`
- **Status:** Good (inherits from BaseRenderer)
- **Key Context Fields:** overall_fairness_score, protected_attributes, warnings, critical_issues
- **Custom Method:** `_format_number()` - Jinja2 filter for number formatting
- **Transformer:** FairnessDataTransformerSimple
- **Eliminated Duplication:** 105 LOC

### 5. HyperparameterRenderer (111 LOC)
- **Location:** `/deepbridge/core/experiment/report/renderers/hyperparameter_renderer.py`
- **Status:** NEEDS REFACTORING (composition pattern, NOT inheriting)
- **Key Context Fields:** importance_scores, tuning_order, optimization_results
- **Transformer:** HyperparameterDataTransformer
- **Issues:**
  - Uses composition instead of inheritance
  - Duplicates template/asset loading logic
  - Uses legacy `_create_context()` method
  - Uses `_write_report()` instead of `_write_html()`
- **Potential Savings:** ~50 LOC

## BaseRenderer Core Methods (678 LOC)

### Template Method Pattern
- `_load_template(test_type, report_type)` - Load Jinja2 template
- `_get_assets(test_type)` - Get CSS, JS, logo, favicon
- `_get_css_content(test_type)` - Get compiled CSS
- `_get_js_content(test_type)` - Get JS (overridable)
- `_render_template(template, context)` - Render with Jinja2
- `_create_base_context(report_data, test_type, assets)` - Create template context

### Output Methods
- `_write_html(html, file_path)` - Write HTML to file
- `_write_report(html, file_path)` - Write HTML (legacy alias)
- `_ensure_output_dir(file_path)` - Create output directory
- `_fix_html_entities(html)` - Fix HTML entity escaping

### Data Serialization Methods (To Be Cleaned Up)
- `_json_serializer(obj)` - Handle special types in JSON
- `_safe_json_dumps(data)` - Safe JSON serialization
- `_create_serializable_data(data)` - LEGACY - Remove
- `_process_alternative_models(alt_models)` - LEGACY - Remove
- `_create_context(...)` - LEGACY - Remove (74 LOC)

## Common Pattern in All Renderers

```python
def render(self, results, file_path, model_name="Model", 
           report_type="interactive", save_chart=False):
    # 1. Transform data
    report_data = self.data_transformer.transform(results, model_name=model_name)
    
    # 2. Load template
    template = self._load_template('test_type', report_type)
    
    # 3. Get assets
    assets = self._get_assets('test_type')
    
    # 4. Create base context
    context = self._create_base_context(report_data, 'test_type', assets)
    
    # 5. Add test-type-specific context
    context.update({'custom_field_1': ..., 'custom_field_2': ...})
    
    # 6. Render & write
    html = self._render_template(template, context)
    return self._write_html(html, file_path)
```

## Transformer Output Structure

All transformers produce normalized data:
```python
{
    'summary': {
        'score': float,
        'metric': str,
        # test-type-specific fields
    },
    'features': {
        'total': int,
        'importance': dict,
        # feature-specific data
    },
    'metadata': {
        # test-specific metadata
    },
    'charts': {
        # Plotly chart definitions
    },
    # additional test-type-specific sections
}
```

## Issues & Refactoring Opportunities

### Critical Issue: HyperparameterRenderer
- **Problem:** Does NOT inherit from BaseRenderer (breaks pattern)
- **Duplication:** ~25 LOC of template/asset loading
- **Wrong Method Calls:** Uses `_create_context()` instead of `_create_base_context()`
- **Inconsistent Naming:** Uses `_write_report()` instead of `_write_html()`
- **Fix:** Convert to inherit from BaseRenderer (PRIORITY 1)

### Legacy Code in BaseRenderer
- **`_create_context()` method (74 LOC):** Only used by HyperparameterRenderer (which shouldn't use it)
- **Data serialization methods (170 LOC):** Redundant with JsonFormatter
- **Fix:** Remove legacy methods once HyperparameterRenderer is refactored

### Naming Inconsistencies
- `_write_report()` vs `_write_html()` - Choose one standard
- Some methods use `_write_report()`, others use `_write_html()`

## LOC Breakdown

| Category | LOC | Notes |
|----------|-----|-------|
| BaseRenderer base functionality | 200 | Core patterns, kept |
| BaseRenderer legacy code | 244 | `_create_context()`, data serialization - REMOVE |
| BaseRenderer CSS/JS/Template | 234 | Template method pattern - KEEP |
| Simple renderers (4) | 458 | Clean, no duplication |
| HyperparameterRenderer | 111 | Has duplication - REFACTOR |
| **TOTAL** | **1280** | |

## Refactoring Savings Potential

| Phase | Action | Savings | Priority |
|-------|--------|---------|----------|
| Phase 2 (Completed) | Convert 4 simple renderers to inherit | 530 LOC | ✓ Done |
| Phase 3.1 | Fix HyperparameterRenderer inheritance | 50 LOC | HIGH |
| Phase 3.2 | Remove legacy BaseRenderer methods | 244 LOC | MEDIUM |
| Phase 3.3 | Standardize method naming | 2-5 LOC | LOW |
| **Total Phase 3** | | **296+ LOC** | |

## What's Working Well

1. ✓ BaseRenderer abstraction is solid
2. ✓ Template method pattern well-implemented
3. ✓ Transformer pattern is clean and consistent
4. ✓ 4 out of 5 renderers follow best practices
5. ✓ 530 LOC already eliminated in Phase 2

## What Needs Work

1. ✗ HyperparameterRenderer doesn't inherit
2. ✗ Legacy methods still in BaseRenderer
3. ✗ Inconsistent method naming
4. ✗ Multiple data serialization paths

