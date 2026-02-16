# deepbridge-distillation v2.0.0 - Release Notes

## 🎉 deepbridge-distillation v2.0.0 - Initial Release

**Model Distillation module extracted from DeepBridge v1.x**

This is the first standalone release of the deepbridge-distillation package, extracted from the monolithic DeepBridge v1.x library.

---

## 📦 About

`deepbridge-distillation` provides advanced model distillation techniques for knowledge transfer from complex models to simpler ones.

### Key Features

- **AutoDistiller**: Automatic hyperparameter optimization for distillation
- **Multiple Distillation Techniques**:
  - Knowledge Distillation (KD)
  - Hint-based distillation
  - Attention transfer
  - Feature matching
- **Hyperparameter Optimization**: Built-in HPO with Optuna
- **Production-ready**: Includes saving/loading trained models
- **Framework Support**: PyTorch integration

---

## 🚀 Installation

```bash
# Install distillation module (includes deepbridge as dependency)
pip install deepbridge-distillation

# Or with all dependencies
pip install deepbridge deepbridge-distillation
```

---

## 📖 Quick Start

```python
from deepbridge_distillation import AutoDistiller
from deepbridge import DBDataset

# Prepare data
dataset = DBDataset(X, y)

# Auto-distillation with HPO
distiller = AutoDistiller(
    teacher_model=large_model,
    student_model=small_model,
    dataset=dataset,
    n_trials=50
)

# Run distillation
distiller.fit()

# Get optimized student model
optimized_student = distiller.best_student_
```

---

## 🔄 Migration from DeepBridge v1.x

### Import Changes

**Before (v1.x):**
```python
from deepbridge.distillation import AutoDistiller
```

**Now (v2.0):**
```python
from deepbridge_distillation import AutoDistiller
```

### Installation Changes

**Before (v1.x):**
```bash
pip install deepbridge  # Included distillation
```

**Now (v2.0):**
```bash
pip install deepbridge-distillation  # Separate package
```

---

## 📚 Documentation

- **GitHub Repository**: https://github.com/DeepBridge-Validation/deepbridge-distillation
- **Main DeepBridge Docs**: https://github.com/DeepBridge-Validation/DeepBridge
- **Migration Guide**: [DeepBridge Migration Guide](https://github.com/DeepBridge-Validation/DeepBridge/blob/feat/split-repos-v2/desenvolvimento/refatoracao/GUIA_RAPIDO_MIGRACAO.md)

---

## 🔗 Related Packages

- [deepbridge v2.0.0](https://github.com/DeepBridge-Validation/DeepBridge/releases/tag/v2.0.0) - Core validation framework
- [deepbridge-synthetic v2.0.0](https://github.com/DeepBridge-Validation/deepbridge-synthetic/releases/tag/v2.0.0) - Synthetic data generation

---

## 🐛 Bug Reports & Support

- **GitHub Issues**: https://github.com/DeepBridge-Validation/deepbridge-distillation/issues
- **Discussions**: https://github.com/DeepBridge-Validation/deepbridge-distillation/discussions

---

## 📋 Dependencies

- `deepbridge >= 2.0.0` - Core validation framework
- `torch >= 1.9.0` - PyTorch for neural networks
- `optuna >= 2.10.0` - Hyperparameter optimization
- `numpy >= 1.21.0`
- `scikit-learn >= 0.24.0`

---

**Full Changelog**: https://github.com/DeepBridge-Validation/deepbridge-distillation/commits/main
