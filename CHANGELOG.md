# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0-alpha.1] - 2026-02-16

### Breaking Changes

**DeepBridge v2.0 focuses exclusively on Model Validation.**

Modules moved to separate repositories:
- `deepbridge.distillation` → [`deepbridge-distillation`](https://github.com/DeepBridge-Validation/deepbridge-distillation)
- `deepbridge.synthetic` → [`deepbridge-synthetic`](https://github.com/DeepBridge-Validation/deepbridge-synthetic)

### Removed

- Removed `deepbridge/distillation/` module (now in separate package)
- Removed `deepbridge/synthetic/` module (now in separate package)
- Removed related tests for distillation and synthetic modules
- Removed torch and dask from core dependencies (lighter installation)

### Changed

- Focused library on model validation (core competency)
- Reduced core dependencies for lighter installation
- Updated API to v2.0 (see Migration Guide)
- Improved type hints across codebase
- Enhanced documentation and examples
- Moved distillation-specific code to deepbridge-distillation package
- Moved synthetic data generation to deepbridge-synthetic package (standalone)

### Added

- New examples: `robustness_example.py` and `fairness_example.py`
- Comprehensive migration guide ([GUIA_RAPIDO_MIGRACAO.md](desenvolvimento/refatoracao/GUIA_RAPIDO_MIGRACAO.md))
- Updated README with v2.0 information and links to new packages
- Clear documentation on package split strategy

### Migration

See [Migration Guide](desenvolvimento/refatoracao/GUIA_RAPIDO_MIGRACAO.md) for detailed instructions on migrating from v1.x to v2.0.

**Quick Summary:**
- If you use only validation: `pip install --upgrade deepbridge`
- If you use distillation: `pip install deepbridge deepbridge-distillation`
- If you use synthetic data: `pip install deepbridge-synthetic` (standalone)

---

## [1.62.0] - 2025-11-03 (Last v1.x release)

### Added
- Complete fairness testing framework
- 15 fairness metrics (pre-training and post-training)
- Auto-detection of sensitive attributes
- EEOC compliance verification (80% rule)
- Threshold analysis for fairness optimization
- Interactive HTML reports with visualizations
- Comprehensive fairness documentation

### Changed
- Various bug fixes and improvements
- Enhanced documentation

### Deprecated
- Monolithic structure (to be split in v2.0)
- Warning: `deepbridge.distillation` and `deepbridge.synthetic` will move to separate packages in v2.0

---

## Previous Versions

For complete v1.x changelog history, see the [v1.x releases](https://github.com/DeepBridge-Validation/DeepBridge/releases?q=v1) on GitHub.

---

## Migration Support

### v1.x → v2.0 Migration Resources

- **Migration Guide**: [GUIA_RAPIDO_MIGRACAO.md](desenvolvimento/refatoracao/GUIA_RAPIDO_MIGRACAO.md)
- **New Packages**:
  - [deepbridge-distillation](https://github.com/DeepBridge-Validation/deepbridge-distillation)
  - [deepbridge-synthetic](https://github.com/DeepBridge-Validation/deepbridge-synthetic)
- **Support**: [GitHub Issues](https://github.com/DeepBridge-Validation/DeepBridge/issues)

### v1.x Support Timeline

- **Support until**: 2026-12-31
- **Security fixes**: Yes (critical only)
- **Bug fixes**: Yes (critical only)
- **New features**: No (v2.x only)

---

**Maintainers**: Gustavo Haase, Paulo Dourado
**License**: MIT
