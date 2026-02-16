# deepbridge-synthetic v2.0.0 - Release Notes

## 🎉 deepbridge-synthetic v2.0.0 - Initial Standalone Release

**Synthetic Data Generation - Now Fully Standalone**

This is the first standalone release of the deepbridge-synthetic package, extracted from the monolithic DeepBridge v1.x library. Unlike other DeepBridge modules, this package is **completely independent** and doesn't require the core deepbridge package.

---

## 📦 About

`deepbridge-synthetic` provides advanced synthetic data generation techniques for creating realistic datasets while preserving statistical properties and privacy.

### Key Features

- **Multiple Generation Methods**:
  - SMOTE (Synthetic Minority Over-sampling Technique)
  - ADASYN (Adaptive Synthetic Sampling)
  - Gaussian Copula
  - CTGAN (Conditional Tabular GAN)
  - TVAE (Tabular VAE)
- **Privacy-Preserving**: Differential privacy support
- **Quality Metrics**: Built-in similarity and quality assessment
- **Standalone**: No dependencies on deepbridge core
- **Production-ready**: Easy-to-use API

---

## 🚀 Installation

```bash
# Standalone installation (no deepbridge required)
pip install deepbridge-synthetic
```

---

## 📖 Quick Start

```python
from deepbridge_synthetic import Synthesize
import pandas as pd

# Load your data
data = pd.read_csv('your_data.csv')

# Generate synthetic data
synthesizer = Synthesize(method='ctgan')
synthetic_data = synthesizer.fit_generate(
    data,
    n_samples=1000
)

# Evaluate quality
from deepbridge_synthetic.metrics import calculate_similarity

similarity_score = calculate_similarity(data, synthetic_data)
print(f"Similarity: {similarity_score:.2%}")
```

---

## 🔄 Migration from DeepBridge v1.x

### Import Changes

**Before (v1.x):**
```python
from deepbridge.synthetic import Synthesize
```

**Now (v2.0):**
```python
from deepbridge_synthetic import Synthesize
```

### Installation Changes

**Before (v1.x):**
```bash
pip install deepbridge  # Included synthetic module
```

**Now (v2.0):**
```bash
pip install deepbridge-synthetic  # Standalone package
```

### Key Difference

**v2.0 is completely standalone** - you don't need to install `deepbridge` to use synthetic data generation!

---

## 📚 Available Methods

### Classical Methods
- **SMOTE**: Best for imbalanced classification datasets
- **ADASYN**: Adaptive density-based oversampling
- **Gaussian Copula**: Preserves correlations

### Deep Learning Methods
- **CTGAN**: Conditional GAN for tabular data
- **TVAE**: Variational autoencoder for tabular data

### Privacy-Preserving
- Differential privacy integration
- k-anonymity support

---

## 📖 Documentation

- **GitHub Repository**: https://github.com/DeepBridge-Validation/deepbridge-synthetic
- **Main DeepBridge Docs**: https://github.com/DeepBridge-Validation/DeepBridge
- **Migration Guide**: [DeepBridge Migration Guide](https://github.com/DeepBridge-Validation/DeepBridge/blob/feat/split-repos-v2/desenvolvimento/refatoracao/GUIA_RAPIDO_MIGRACAO.md)

---

## 🔗 Related Packages

- [deepbridge v2.0.0](https://github.com/DeepBridge-Validation/DeepBridge/releases/tag/v2.0.0) - Core validation framework (optional)
- [deepbridge-distillation v2.0.0](https://github.com/DeepBridge-Validation/deepbridge-distillation/releases/tag/v2.0.0) - Model distillation

---

## 🐛 Bug Reports & Support

- **GitHub Issues**: https://github.com/DeepBridge-Validation/deepbridge-synthetic/issues
- **Discussions**: https://github.com/DeepBridge-Validation/deepbridge-synthetic/discussions

---

## 📋 Dependencies

This package is **standalone** and has minimal dependencies:

- `numpy >= 1.21.0`
- `pandas >= 1.3.0`
- `scikit-learn >= 0.24.0`
- `sdv >= 0.18.0` - For CTGAN/TVAE
- `imbalanced-learn >= 0.8.0` - For SMOTE/ADASYN

**No deepbridge dependency required!**

---

## 🎯 Use Cases

- **Data Augmentation**: Expand training datasets
- **Privacy Protection**: Generate synthetic data for sharing
- **Imbalanced Data**: Create balanced datasets for ML
- **Testing**: Generate realistic test data
- **Research**: Experiment with synthetic populations

---

**Full Changelog**: https://github.com/DeepBridge-Validation/deepbridge-synthetic/commits/main
