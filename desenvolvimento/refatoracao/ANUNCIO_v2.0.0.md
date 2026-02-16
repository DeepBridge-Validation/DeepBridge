# DeepBridge v2.0.0 Released - Package Split & Focus on Validation

We're excited to announce **DeepBridge v2.0.0** - a major release that restructures DeepBridge into a focused, modular ecosystem!

## What's New?

DeepBridge v2.0 focuses exclusively on **Model Validation**, with distillation and synthetic data modules split into separate packages:

### 📦 Three Focused Packages

1. **`deepbridge`** - Core model validation framework (lighter, faster)
2. **`deepbridge-distillation`** - Model distillation module
3. **`deepbridge-synthetic`** - Synthetic data generation (standalone)

### ✨ Key Improvements

- **Lighter Installation**: Core package no longer requires PyTorch/Dask
- **Modular Design**: Install only what you need
- **Better Focus**: Each package does one thing well
- **Improved API**: Enhanced type hints and documentation
- **New Examples**: Robustness and fairness validation examples

## Installation

### For Validation Only
```bash
pip install deepbridge
```

### For Validation + Distillation
```bash
pip install deepbridge deepbridge-distillation
```

### For Synthetic Data (Standalone)
```bash
pip install deepbridge-synthetic
```

## Migration from v1.x

**Good news!** Migration is straightforward:

### If you only use validation
```bash
pip install --upgrade deepbridge
# No code changes needed!
```

### If you use distillation
```bash
pip install deepbridge deepbridge-distillation
```

Update imports:
```python
# Old (v1.x)
from deepbridge.distillation import AutoDistiller

# New (v2.0)
from deepbridge_distillation import AutoDistiller
```

### If you use synthetic data
```bash
pip install deepbridge-synthetic
```

Update imports:
```python
# Old (v1.x)
from deepbridge.synthetic import Synthesize

# New (v2.0)
from deepbridge_synthetic import Synthesize
```

**Full Migration Guide**: [GUIA_RAPIDO_MIGRACAO.md](https://github.com/DeepBridge-Validation/DeepBridge/blob/feat/split-repos-v2/desenvolvimento/refatoracao/GUIA_RAPIDO_MIGRACAO.md)

## Breaking Changes

- `deepbridge.distillation` → `deepbridge-distillation` (separate package)
- `deepbridge.synthetic` → `deepbridge-synthetic` (separate package)
- Imports changed from `deepbridge.module` to `deepbridge_module`

## v1.x Support

- **Support until**: December 31, 2026
- **Security fixes**: Yes (critical only)
- **Bug fixes**: Yes (critical only)
- **New features**: No (v2.x only)

v1.63.0 has been released with a deprecation warning to guide migration.

## New Repositories

- **deepbridge**: https://github.com/DeepBridge-Validation/DeepBridge
- **deepbridge-distillation**: https://github.com/DeepBridge-Validation/deepbridge-distillation
- **deepbridge-synthetic**: https://github.com/DeepBridge-Validation/deepbridge-synthetic

## Resources

- **Release Notes**: [v2.0.0 Release](https://github.com/DeepBridge-Validation/DeepBridge/releases/tag/v2.0.0)
- **Migration Guide**: [GUIA_RAPIDO_MIGRACAO.md](https://github.com/DeepBridge-Validation/DeepBridge/blob/feat/split-repos-v2/desenvolvimento/refatoracao/GUIA_RAPIDO_MIGRACAO.md)
- **CHANGELOG**: [CHANGELOG.md](https://github.com/DeepBridge-Validation/DeepBridge/blob/feat/split-repos-v2/CHANGELOG.md)

## Getting Help

- **Questions**: [GitHub Discussions](https://github.com/DeepBridge-Validation/DeepBridge/discussions)
- **Issues**: [GitHub Issues](https://github.com/DeepBridge-Validation/DeepBridge/issues)
- **Migration Support**: See [Migration Guide](https://github.com/DeepBridge-Validation/DeepBridge/blob/feat/split-repos-v2/desenvolvimento/refatoracao/GUIA_RAPIDO_MIGRACAO.md)

## Thank You!

Special thanks to all contributors and users who helped shape DeepBridge v2.0. Your feedback and support made this possible!

---

**Try it now**: `pip install deepbridge`

**Maintainers**: Gustavo Haase, Paulo Dourado
**License**: MIT

---

## Social Media Version (Short)

**Twitter/LinkedIn**:

🎉 DeepBridge v2.0.0 is here!

✨ What's new:
- Split into 3 focused packages
- Lighter installation (no PyTorch in core)
- Better modularity
- Improved API

📦 Install only what you need:
• deepbridge - Model validation
• deepbridge-distillation - Model distillation
• deepbridge-synthetic - Synthetic data (standalone)

Migration guide: [link]

#Python #MachineLearning #MLOps #DataScience

---

## Reddit r/MachineLearning Version

**Title**: [P] DeepBridge v2.0 - Model Validation Framework Now Modular

**Body**:

I'm excited to share DeepBridge v2.0, a major update to our model validation framework!

**What is DeepBridge?**

DeepBridge is a Python library for comprehensive ML model validation, including:
- Performance metrics
- Robustness testing
- Fairness evaluation
- Model distillation
- Synthetic data generation

**What's New in v2.0?**

We've restructured the library into three focused packages:

1. `deepbridge` - Core validation (lighter, no PyTorch dependency)
2. `deepbridge-distillation` - Model distillation with HPO
3. `deepbridge-synthetic` - Synthetic data generation (fully standalone)

**Why the Split?**

The monolithic v1.x had heavyweight dependencies (PyTorch, Dask) even for users who only needed basic validation. v2.0 lets you install only what you need.

**Installation**:
```bash
# Validation only (lightweight)
pip install deepbridge

# Validation + distillation
pip install deepbridge deepbridge-distillation

# Synthetic data (standalone)
pip install deepbridge-synthetic
```

**Migration from v1.x**: Straightforward! See our [migration guide](link).

**Links**:
- GitHub: https://github.com/DeepBridge-Validation/DeepBridge
- PyPI: https://pypi.org/project/deepbridge/
- Docs: [link]

**License**: MIT

Happy to answer any questions!
