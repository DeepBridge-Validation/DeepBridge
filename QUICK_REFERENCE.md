# DeepBridge Codebase Quick Reference

## Directory Map

```
deepbridge/
├── cli/                    Command-line interface
├── config/                 Configuration management  
├── core/experiment/        Main framework (14K+ lines) ⚠️ LARGEST MODULE
│   ├── managers/          Test managers
│   ├── report/            Report generation (8K+ lines)
│   └── ...
├── distillation/           Model distillation
├── metrics/                Performance metrics
├── models/                 Model wrappers
├── synthetic/              Synthetic data generation
├── templates/              HTML/JS templates
├── utils/                  Utilities
└── validation/             Validation wrappers ⚠️ HIGH COUPLING
    ├── fairness/
    ├── robustness/
    └── wrappers/
```

## Files by Size Category

### CRITICAL (>2000 lines) - REFACTOR NEEDED
- `seaborn_utils.py` (2,713) - 37 methods, god class
- `static_uncertainty_renderer.py` (2,538) - monolithic
- `resilience_suite.py` (2,369) - feature god class
- `static_resilience.py` (2,226) - procedural code
- `robustness_renderer.py` (2,220) - complex renderer

### HIGH (1000-1999 lines) - REVIEW NEEDED
- `fairness_metrics.py` (1,712) - 30 methods
- `resilience_simple.py` (1,563) - duplicate logic
- `results.py` (1,515) - 37 methods
- `uncertainty_suite.py` (1,478)
- `fairness_suite.py` (1,348)
- `auto_distiller.py` (1,295)
- `gaussian_copula.py` (1,156)

### MODERATE (500-999 lines) - MONITOR
- 32 additional files in this range

## Key Metrics At-A-Glance

| Metric | Value | Status |
|--------|-------|--------|
| Total lines | 92,503 | Growing |
| Total files | 232 | -/+ |
| Files >500L | 45 | ⚠️ HIGH |
| Functions >100L | 12+ | ⚠️ HIGH |
| God classes (15+ methods) | 37+ | ⚠️ HIGH |
| Code duplication | 20-30% | ⚠️ HIGH |
| Module coupling (0-100) | 65-85 | ⚠️ HIGH |
| Circular deps (potential) | 3 paths | ⚠️ RISK |
| Late imports | 65 instances | ⚠️ RISK |
| Test coverage | ~40% | ⚠️ LOW |

## Architecture Issues

### Design Smells
1. **God Classes** - Too many responsibilities
   - SeabornChartGenerator (37 methods)
   - Experiment (32 methods)
   - FairnessMetrics (30 methods)
   - ExperimentResult (37 methods)

2. **Variant Proliferation** - Multiple similar implementations
   - Simple vs. Full vs. Static variants
   - Example: 3 robustness renderers + 3 transformers

3. **Large Functions** - Hard to understand and test
   - _prepare_chart_data: 486 lines
   - render: 300 lines
   - _load_js_content: 299 lines

4. **Circular Dependencies** - Fragile architecture
   - core/experiment ↔ validation
   - distillation ↔ core
   - Mitigated by 65+ late imports

### High-Risk Components
```
REPORT GENERATION (2.5K+ lines)
├── Renderers (Interactive, Static, Simple)
├── Transformers (Data transformation)
├── Templates (HTML/JS)
└── Assets (CSS, JS, images)
Status: ⚠️ MULTIPLE INCOMPATIBLE IMPLEMENTATIONS

VALIDATION SUITES (1000-2300L each)
├── ResilienceSuite
├── RobustnessSuite
├── UncertaintySuite
└── FairnessSuite
Status: ⚠️ MONOLITHIC, NO CLEAR PATTERNS

FAIRNESS METRICS (1,712 lines)
└── 30 static methods
Status: ⚠️ DIFFICULT TO EXTEND, TEST

CONFIGURATION (4 systems)
├── Parameter standards
├── Settings module
├── Hard-coded defaults
└── Runtime arguments
Status: ⚠️ NO UNIFIED SYSTEM
```

## Dependency Coupling

### Dependency Direction (should be one-way)
```
SHOULD BE:
core → utils → validation → distillation

ACTUALLY IS:
core ←→ validation  (BIDIRECTIONAL ⚠️)
distillation ↔ core (BIDIRECTIONAL ⚠️)
report ↔ results (TIGHTLY COUPLED ⚠️)
renderers ↔ transformers (TIGHTLY COUPLED ⚠️)
```

### Coupling Scores (0-100, higher = worse)
| Module Pair | Score | Issue |
|-------------|-------|-------|
| core/experiment ↔ core/report | 85 | Report embedded in results |
| distillation ↔ core/experiment | 80 | Creates experiment objects |
| validation ↔ fairness_metrics | 75 | Direct dependencies |
| renderers ↔ transformers | 70 | Each imports specific transformer |
| core/experiment ↔ validation | 65 | Experiment uses suites |

## Code Duplication Patterns

### Rendering (appears 8+ times)
```python
# Load template → insert data → embed assets → save file
# Repeated in every renderer
```

### Chart Generation (appears 10+ times)
```python
# Data aggregation → setup → encoding → base64
# Repeated in seaborn_utils, renderers, transformers
```

### Group Metrics (appears 5+ times)
```python
# Group filtering → calculation → aggregation → thresholding
# Repeated in fairness_suite, metrics, transformers
```

## Inconsistencies

### Naming Conventions (No standard)
```
robustness_renderer.py
robustness_renderer_simple.py
static_robustness_renderer.py
robustness_domain.py
robustness.py (transformer)
robustness_simple.py (transformer)
static_robustness.py (transformer)
```
**Problem:** Unclear which variant to use

### Class Hierarchies (Mixed strategies)
```python
# Type A: Inheritance
class ResilienceRenderer(BaseRenderer):
    pass

# Type B: Composition
class RobustnessRenderer:
    def __init__(self):
        self.base_renderer = BaseRenderer(...)

# Type C: Static methods
class FairnessMetrics:
    @staticmethod
    def statistical_parity(...): pass
```
**Problem:** Inconsistent patterns

### Configuration (4 different systems)
```python
# System 1: Centralized standards
get_test_config(TestType.RESILIENCE, ConfigName.FULL)

# System 2: Settings module
from deepbridge.config.settings import DistillationConfig

# System 3: Hard-coded
DEFAULT_PARAMS = {'alpha': 0.1}

# System 4: Runtime arguments
def __init__(self, config_dict):
    self.config = config_dict
```
**Problem:** No unified approach

## Risk Assessment

### CRITICAL RISKS
- [ ] Circular dependencies cause import failures
- [ ] God classes have too many responsibilities
- [ ] Report rendering has silent failures
- [ ] Validation suites may have incorrect logic
- [ ] 20% code is duplicated (maintenance burden)

### MEDIUM RISKS  
- [ ] Limited type hints (30% coverage)
- [ ] Inconsistent patterns across codebase
- [ ] Poor test coverage (40% estimated)
- [ ] Configuration chaos (4 systems)
- [ ] Variant proliferation (maintenance nightmare)

### ACTIONABLE ITEMS

**Week 1:** Break circular dependencies
**Month 1:** Consolidate rendering variants
**Month 2:** Refactor god classes
**Month 3:** Standardize patterns
**Month 4-6:** Improve test coverage

## Performance Notes

### Likely Bottlenecks
1. Report generation (complex transformations)
2. Chart rendering (matplotlib + base64 encoding)
3. Data transformations (deep nesting)
4. Import time (65 late imports, circular deps)

### Optimization Opportunities
1. Template caching
2. Chart rendering optimization
3. Data structure simplification
4. Lazy loading for heavy modules

## Testing Strategy

### Coverage Gaps
- [ ] Fairness transformers - NO TESTS
- [ ] Report renderers - MINIMAL TESTS
- [ ] Distillation techniques - NO TESTS
- [ ] Metrics calculation - SPARSE TESTS

### Recommendation
- Aim for 80%+ coverage
- Focus on: core logic, metrics, renderers
- Add fixtures for complex test data
- Test circular dependency handling

## Refactoring Roadmap

### Month 1-2: Break Variants
- Consolidate simple/static/full
- Single rendering pipeline
- Template-based approach

### Month 2-3: Split God Classes
- FairnessMetrics → 4 metric classes
- SeabornChartGenerator → ChartFactory
- Experiment → orchestrator components
- ResilienceSuite → test factories

### Month 3: Standardize
- One inheritance/composition strategy
- Unified configuration
- Consistent naming
- Standard error handling

### Month 4-6: Test & Polish
- Add comprehensive tests
- Add type hints
- Profile performance
- Document architecture

## Files to Watch

**Most Problematic:**
- `core/experiment/report/utils/seaborn_utils.py` - 37 methods
- `core/experiment/report/renderers/robustness_renderer.py` - 486-line method
- `core/experiment/report/renderers/static/static_uncertainty_renderer.py` - monolithic
- `validation/wrappers/resilience_suite.py` - 2,369 lines, 20 methods

**High Risk:**
- `core/experiment/experiment.py` - 32 methods
- `core/experiment/results.py` - 37 methods, mixed concerns
- `validation/fairness/metrics.py` - 30 static methods
- `distillation/auto_distiller.py` - orchestration complexity

**Need Tests:**
- All transformers (fairness, resilience, robustness, uncertainty)
- All renderers (especially static variants)
- All metrics calculations
- Distillation techniques

## Quick Commands

```bash
# Find files over 1000 lines
find . -name "*.py" ! -path "*/__pycache__/*" -exec wc -l {} \; | sort -rn | head -20

# Find functions over 100 lines
grep -r "^    def \|^def " --include="*.py" | grep -v test

# Find circular import risks
grep -r "^from deepbridge\|^import deepbridge" --include="*.py" | sort | uniq -c | sort -rn

# Find late imports (inside functions)
grep -r "^    from deepbridge\|^    import deepbridge" --include="*.py" | wc -l

# Find star imports
grep -r "import \*" --include="*.py"

# Test coverage estimate
python -m pytest --cov=deepbridge tests/ | grep "TOTAL"
```

## References

- Full Analysis: `CODEBASE_ANALYSIS_REPORT.md`
- Executive Summary: `ANALYSIS_EXECUTIVE_SUMMARY.txt`
- Quick Reference: This file

---

**Last Updated:** 2026-02-10
**Analysis Depth:** Very Thorough
**Status:** Technical Debt: HIGH - Action Required
