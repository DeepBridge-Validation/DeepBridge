# DeepBridge CSS System - Quick Reference

**Version**: 1.0
**Date**: 2025-10-29
**Status**: Production Ready ✅

---

## 📚 Overview

This directory contains the **three-layer CSS architecture** for all DeepBridge reports:

```
Layer 1: base_styles.css          (Design tokens, reset, typography, utilities)
Layer 2: report_components.css    (Shared UI components for all reports)
Layer 3: {report}_custom.css      (Report-specific overrides)
```

---

## 📁 File Structure

```
/deepbridge/templates/
├── base_styles.css                    # Layer 1: Foundation (11 KB)
├── report_components.css              # Layer 2: Components (17 KB)
└── report_types/
    ├── robustness/interactive/
    │   └── (uses base + components only)
    ├── resilience/interactive/
    │   └── css/
    │       └── resilience_custom.css  # Layer 3: Custom (2.5 KB)
    └── uncertainty/interactive/
        └── css/
            └── uncertainty_custom.css # Layer 3: Custom (2.3 KB)
```

---

## 🚀 Quick Start

### Using in Your Renderer

```python
from ..css_manager import CSSManager

class YourRenderer:
    def __init__(self, template_manager, asset_manager):
        self.css_manager = CSSManager()

    def _get_css_content(self) -> str:
        try:
            # Automatically compiles: base + components + custom
            return self.css_manager.get_compiled_css('your_report_name')
        except Exception as e:
            logger.error(f"CSS error: {e}")
            return fallback_css
```

That's it! Your report now has all base styles, all components, and custom overrides.

---

## 🎨 Design System

### Colors

```css
--primary-color: #1b78de;      /* Bright Blue */
--secondary-color: #2c3e50;    /* Dark Slate */
--success-color: #28a745;      /* Green */
--danger-color: #dc3545;       /* Red */
--warning-color: #f39c12;      /* Orange */
```

### Spacing

```css
--spacing-xs: 4px
--spacing-sm: 8px
--spacing-md: 16px
--spacing-lg: 24px
--spacing-xl: 32px
```

### Typography

```css
--font-size-xs: 12px
--font-size-sm: 14px
--font-size-base: 16px
--font-size-lg: 18px
--font-size-xl: 20px
--font-size-2xl: 24px
--font-size-3xl: 30px
--font-size-4xl: 36px
```

---

## 🧩 Available Components

All reports automatically get these components:

1. **`.report-container`** - Main container (max-width: 1200px)
2. **`.report-header`** - Header with gradient
3. **`.metrics-grid`** - Grid for metric cards
4. **`.metric-card`** - Individual metric display
5. **`.tab-navigation`** - Tab system
6. **`.tab-button`** - Individual tabs
7. **`.section`** - Content sections
8. **`.chart-container`** - Chart wrapper (with loading/error states)
9. **`.data-table`** - Data tables (with sticky headers)
10. **`.btn`** - Buttons (6 variants, 3 sizes)
11. **`.badge`** - Badges and labels
12. **`.alert`** - Alert messages

Plus **100+ utility classes** for spacing, text, display, etc.

---

## ✏️ Common Tasks

### Change Primary Color for ALL Reports

```bash
# Edit base_styles.css
vim /deepbridge/templates/base_styles.css

# Find line 11:
--primary-color: #1b78de;

# Change to your color:
--primary-color: #YOUR_COLOR;

# Save and regenerate reports
# All reports automatically use new color!
```

### Add New Component for ALL Reports

```bash
# Edit report_components.css
vim /deepbridge/templates/report_components.css

# Add your component CSS:
.your-component {
    /* styles */
}

# Save
# All reports can now use .your-component
```

### Customize ONE Report

```bash
# Create or edit custom CSS for that report
vim /deepbridge/templates/report_types/your_report/interactive/css/your_report_custom.css

# Add overrides:
.report-container {
    max-width: 1600px;  /* Override default 1200px */
}

body {
    background: linear-gradient(135deg, #color1, #color2);
}

# Save
# Only your_report uses these styles
```

### Create New Report Type

**Step 1**: Add CSSManager to your renderer

```python
from ..css_manager import CSSManager

class NewReportRenderer:
    def __init__(self, template_manager, asset_manager):
        self.css_manager = CSSManager()

    def _get_css_content(self):
        return self.css_manager.get_compiled_css('new_report')
```

**Step 2** (Optional): Create custom CSS

```bash
mkdir -p /deepbridge/templates/report_types/new_report/interactive/css
vim /deepbridge/templates/report_types/new_report/interactive/css/new_report_custom.css

# Add ONLY report-specific styles
```

**Done!** Your new report has:
- ✅ All design tokens
- ✅ All 12+ components
- ✅ Your custom styles

---

## 📖 Documentation

**Full documentation**: `/analise_v2/CSS_STANDARDIZATION_COMPLETE.md`

**Quick summary**: `/analise_v2/LAYOUT_STANDARDIZATION_SUMMARY.md`

**Phase reports**:
- Phase 1: `/analise_v2/PHASE1_IMPLEMENTATION_COMPLETE.md`
- Phase 2-3: `/analise_v2/PHASE2_3_COMPLETE.md`

**Index**: `/analise_v2/INDEX_DOCUMENTATION.md`

---

## 🔍 Troubleshooting

### CSS not compiling?

```python
# Check CSSManager validation
from deepbridge.core.experiment.report.css_manager import CSSManager

manager = CSSManager()
validation = manager.validate_css_files()
print(validation)
# Should show: {'base_styles': True, 'components': True, 'errors': []}
```

### Custom CSS not loading?

```python
# Check custom CSS info
info = manager.get_custom_css_info('your_report')
print(info)
# Should show: {'exists': True, 'path': '...', 'size': ...}
```

### Report looks wrong?

1. Check console for CSS errors
2. Verify CSSManager is initialized in renderer
3. Check custom CSS isn't overriding too much
4. Compare with working report (resilience or uncertainty)

---

## 📊 System Status

- ✅ **Renderers migrated**: 3/3 (100%)
- ✅ **Code duplication**: 0% (was 70%)
- ✅ **Color consistency**: 100%
- ✅ **Tests passing**: 8/8 (100%)
- ✅ **Production status**: Ready

---

## 🎯 Best Practices

### DO ✅

- Use CSS variables for colors, spacing, typography
- Use existing components when possible
- Add custom CSS only for unique styles
- Follow naming conventions (BEM-like)
- Test in multiple browsers

### DON'T ❌

- Don't inline CSS in renderers
- Don't duplicate base styles
- Don't override base CSS variables in custom CSS (extend instead)
- Don't forget to use `|safe` filter in Jinja2 for CSS
- Don't create custom components that could be shared

---

## 🆘 Need Help?

**See full documentation**: `/analise_v2/CSS_STANDARDIZATION_COMPLETE.md`

**Migration guide for new reports**: Section "Migration Guide for Future Developers"

**Examples**: See `resilience_renderer_simple.py` or `uncertainty_renderer_simple.py`

---

**Created**: 2025-10-29
**Version**: 1.0
**Status**: Production Ready ✅
