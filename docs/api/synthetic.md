# Synthetic Data Generation API

## Overview

The Synthetic Data Generation API provides comprehensive tools for creating high-quality synthetic datasets that maintain the statistical properties of original data while addressing privacy concerns. This API is designed to be flexible, allowing for different generation methods suitable for various data types and use cases.

## Core Components

### Generator Classes

The API includes several generator classes for different synthetic data generation methods:

#### UltraLightGenerator

```python
from deepbridge.synthetic import UltraLightGenerator

generator = UltraLightGenerator(
    categorical_columns=None,
    continuous_columns=None,
    verbose=False
)
```

A lightweight generator that efficiently handles mixed data types with minimal computational overhead.

| Method | Description |
| ------ | ----------- |
| `fit(data)` | Fit the generator to the provided DataFrame |
| `generate(n_samples)` | Generate synthetic samples |
| `fit_generate(data, n_samples)` | Convenience method to fit and generate in one step |
| `save(path)` | Save the fitted generator to a file |
| `load(path)` | Load a previously saved generator |

[Detailed documentation](../concepts/synthetic_UltraLightGenerator.md)

#### GaussianCopulaGenerator

```python
from deepbridge.synthetic import GaussianCopulaGenerator

generator = GaussianCopulaGenerator(
    categorical_columns=None,
    continuous_columns=None,
    correlation_method='pearson'
)
```

Uses Gaussian copulas to capture correlations between features while preserving original marginal distributions.

| Method | Description |
| ------ | ----------- |
| `fit(data)` | Fit the generator to the provided DataFrame |
| `generate(n_samples)` | Generate synthetic samples |
| `fit_generate(data, n_samples)` | Convenience method to fit and generate in one step |
| `save(path)` | Save the fitted generator to a file |
| `load(path)` | Load a previously saved generator |

[Detailed documentation](../concepts/synthetic_gaussian_copula.md)

#### CTGANGenerator

```python
from deepbridge.synthetic import CTGANGenerator

generator = CTGANGenerator(
    embedding_dim=128,
    generator_dim=(256, 256),
    discriminator_dim=(256, 256),
    batch_size=500,
    epochs=300
)
```

A GAN-based approach specifically designed for tabular data with mixed data types.

| Method | Description |
| ------ | ----------- |
| `fit(data)` | Fit the generator to the provided DataFrame |
| `generate(n_samples)` | Generate synthetic samples |
| `fit_generate(data, n_samples)` | Convenience method to fit and generate in one step |
| `save(path)` | Save the fitted generator to a file |
| `load(path)` | Load a previously saved generator |

[Detailed documentation](../concepts/synthetic_ctgan.md)

### Evaluation Tools

#### SyntheticDataEvaluator

```python
from deepbridge.synthetic import SyntheticDataEvaluator

evaluator = SyntheticDataEvaluator(
    original_data,
    synthetic_data,
    categorical_columns=None,
    continuous_columns=None
)
```

Comprehensive evaluation of synthetic data quality.

| Method | Description |
| ------ | ----------- |
| `evaluate()` | Calculate comprehensive metrics to assess synthetic data quality |
| `statistical_similarity()` | Measure statistical similarity between original and synthetic data |
| `privacy_metrics()` | Evaluate privacy preservation of synthetic data |
| `utility_metrics()` | Assess utility of synthetic data for machine learning tasks |
| `plot_distributions()` | Visualize feature distributions comparison |
| `plot_correlation_comparison()` | Compare correlation matrices between datasets |
| `plot_pca_comparison()` | Compare PCA projections of both datasets |

### Utility Functions

#### Column Type Detection

```python
from deepbridge.synthetic.utils import detect_column_types

categorical, continuous = detect_column_types(dataframe, categorical_threshold=10)
```

Automatically identifies categorical and continuous columns in a DataFrame.

#### Data Transformation

```python
from deepbridge.synthetic.utils import transform_data, inverse_transform_data

# Transform data for better generation
transformed_data = transform_data(dataframe, continuous_columns)

# Reverse transformation after generation
original_scale_data = inverse_transform_data(
    synthetic_data,
    transformation_params,
    continuous_columns
)
```

Utilities for data transformation and normalization to improve generation quality.

## Integration with Experiment Framework

### Using Synthetic Data in Experiments

```python
from deepbridge.core.experiment import Experiment
from deepbridge.core.db_data import DBDataset
from deepbridge.synthetic import UltraLightGenerator

# Create original dataset
original_dataset = DBDataset(original_df, target_column="target")

# Generate synthetic data
generator = UltraLightGenerator()
synthetic_df = generator.fit_generate(original_df, n_samples=1000)
synthetic_dataset = DBDataset(synthetic_df, target_column="target")

# Create experiment with both datasets
experiment = Experiment(original_dataset)
experiment.add_dataset(synthetic_dataset, name="synthetic")

# Run validation on both datasets
results = experiment.validate()
```

## Examples

### Basic Synthetic Data Generation

```python
import pandas as pd
from deepbridge.synthetic import UltraLightGenerator

# Load original data
data = pd.read_csv("customer_data.csv")

# Initialize generator
generator = UltraLightGenerator()

# Fit and generate
synthetic_data = generator.fit_generate(data, n_samples=5000)

# Save synthetic data
synthetic_data.to_csv("synthetic_customer_data.csv", index=False)
```

### Comparing Multiple Generation Methods

```python
import pandas as pd
from deepbridge.synthetic import (
    UltraLightGenerator,
    GaussianCopulaGenerator,
    CTGANGenerator,
    SyntheticDataEvaluator
)
import matplotlib.pyplot as plt

# Load original data
data = pd.read_csv("financial_data.csv")

# Initialize generators
generators = {
    "UltraLight": UltraLightGenerator(),
    "GaussianCopula": GaussianCopulaGenerator(),
    "CTGAN": CTGANGenerator(epochs=100)
}

# Generate synthetic data using each method
synthetic_datasets = {}
for name, generator in generators.items():
    synthetic_datasets[name] = generator.fit_generate(data, n_samples=1000)

# Evaluate each synthetic dataset
metrics = {}
for name, synthetic_data in synthetic_datasets.items():
    evaluator = SyntheticDataEvaluator(data, synthetic_data)
    metrics[name] = evaluator.evaluate()

# Compare metrics
for metric_name in ['statistical_similarity', 'privacy_score', 'utility_score']:
    plt.figure(figsize=(10, 6))
    plt.bar(metrics.keys(), [m[metric_name] for m in metrics.values()])
    plt.title(f"Comparison of {metric_name}")
    plt.ylabel("Score")
    plt.savefig(f"{metric_name}_comparison.png")
```

### Privacy-Focused Generation

```python
from deepbridge.synthetic import GaussianCopulaGenerator
from deepbridge.synthetic.privacy import PrivacyEnhancer

# Load sensitive data
data = pd.read_csv("patient_records.csv")

# Apply privacy enhancement techniques
privacy_enhancer = PrivacyEnhancer(epsilon=1.0)  # ε-differential privacy
enhanced_data = privacy_enhancer.transform(data)

# Generate synthetic data with enhanced privacy
generator = GaussianCopulaGenerator()
private_synthetic_data = generator.fit_generate(enhanced_data, n_samples=2000)

# Evaluate privacy metrics
from deepbridge.synthetic import SyntheticDataEvaluator
evaluator = SyntheticDataEvaluator(data, private_synthetic_data)
privacy_metrics = evaluator.privacy_metrics()
print(f"Re-identification risk: {privacy_metrics['reidentification_risk']}")
```

## Best Practices

1. **Column Type Detection**: While automatic detection works well in most cases, explicitly specifying categorical and continuous columns is recommended for optimal results.

2. **Sampling Considerations**: For large datasets, consider using a representative sample for fitting generators to improve performance.

3. **Evaluation**: Always evaluate synthetic data quality before using it in downstream tasks.

4. **Generation Size**: Start with generating a dataset of similar size to the original, then scale up or down as needed.

5. **Method Selection**:
   - Use UltraLightGenerator for quick experiments and datasets with simple relationships
   - Use GaussianCopulaGenerator for datasets where maintaining correlations is important
   - Use CTGANGenerator for complex relationships and when computational resources allow

## Advanced Topics

- [Custom generation models](../advanced/custom_models.md)
- [Optimizing generation parameters](../advanced/optimization.md)
- [Deployment considerations](../advanced/deployment.md)