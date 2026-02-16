# DeepBridge v2.0.0 - Release Notes

## 🎉 DeepBridge v2.0.0 - Major Release

**DeepBridge v2.0 focuses exclusively on Model Validation.**

This is a major release that restructures DeepBridge into a modular ecosystem. The library is now split into three focused packages:

- **deepbridge** (this package) - Core model validation framework
- **deepbridge-distillation** - Model distillation module (separate package)
- **deepbridge-synthetic** - Synthetic data generation (standalone package)

---

## 🚨 Breaking Changes

### Package Split

Modules moved to separate repositories:
- `deepbridge.distillation` → [`deepbridge-distillation`](https://github.com/DeepBridge-Validation/deepbridge-distillation)
- `deepbridge.synthetic` → [`deepbridge-synthetic`](https://github.com/DeepBridge-Validation/deepbridge-synthetic)

### Installation Changes

**Before (v1.x):**
```bash
pip install deepbridge  # All-in-one package
```

**Now (v2.0):**
```bash
# For validation only
pip install deepbridge

# For validation + distillation
pip install deepbridge deepbridge-distillation

# For synthetic data (standalone)
pip install deepbridge-synthetic
```

### Import Changes

**Before (v1.x):**
```python
from deepbridge.distillation import AutoDistiller
from deepbridge.synthetic import Synthesize
```

**Now (v2.0):**
```python
from deepbridge_distillation import AutoDistiller
from deepbridge_synthetic import Synthesize
```

---

## ✨ What's New

### Removed
- `deepbridge/distillation/` module (now in separate package)
- `deepbridge/synthetic/` module (now in separate package)
- Related tests for distillation and synthetic modules
- Torch and dask from core dependencies (lighter installation)

### Changed
- Focused library on model validation (core competency)
- Reduced core dependencies for lighter installation
- Updated API to v2.0
- Improved type hints across codebase
- Enhanced documentation and examples

### Added
- New examples: `robustness_example.py` and `fairness_example.py`
- Comprehensive migration guide
- Updated README with v2.0 information and links to new packages
- Clear documentation on package split strategy

---

## 📖 Migration Guide

See [Migration Guide](https://github.com/DeepBridge-Validation/DeepBridge/blob/feat/split-repos-v2/desenvolvimento/refatoracao/GUIA_RAPIDO_MIGRACAO.md) for detailed instructions.

### Quick Migration Summary

1. **If you use only validation:**
   ```bash
   pip install --upgrade deepbridge
   # No code changes needed
   ```

2. **If you use distillation:**
   ```bash
   pip install deepbridge deepbridge-distillation
   ```
   Update imports:
   ```python
   # Old
   from deepbridge.distillation import AutoDistiller

   # New
   from deepbridge_distillation import AutoDistiller
   ```

3. **If you use synthetic data:**
   ```bash
   pip install deepbridge-synthetic
   ```
   Update imports:
   ```python
   # Old
   from deepbridge.synthetic import Synthesize

   # New
   from deepbridge_synthetic import Synthesize
   ```

---

## 🔗 Related Packages

- [deepbridge-distillation v2.0.0](https://github.com/DeepBridge-Validation/deepbridge-distillation/releases/tag/v2.0.0)
- [deepbridge-synthetic v2.0.0](https://github.com/DeepBridge-Validation/deepbridge-synthetic/releases/tag/v2.0.0)

---

## 📦 Installation

```bash
pip install deepbridge
```

For development:
```bash
git clone https://github.com/DeepBridge-Validation/DeepBridge.git
cd DeepBridge
poetry install
```

---

## 🐛 Bug Reports & Support

- **GitHub Issues**: https://github.com/DeepBridge-Validation/DeepBridge/issues
- **Migration Help**: See [Migration Guide](https://github.com/DeepBridge-Validation/DeepBridge/blob/feat/split-repos-v2/desenvolvimento/refatoracao/GUIA_RAPIDO_MIGRACAO.md)

---

## 📅 v1.x Support Timeline

- **Support until**: 2026-12-31
- **Security fixes**: Yes (critical only)
- **Bug fixes**: Yes (critical only)
- **New features**: No (v2.x only)

A deprecation warning has been added to v1.63.0 to guide users to migrate.

---

**Full Changelog**: https://github.com/DeepBridge-Validation/DeepBridge/blob/feat/split-repos-v2/CHANGELOG.md
