# Synthetic Data Generation Guide

## Overview

This guide introduces DeepBridge's synthetic data generation capabilities, which allow you to create high-quality synthetic datasets that preserve the statistical properties of your original data while addressing privacy concerns.

## Why Use Synthetic Data?

Synthetic data offers several advantages in machine learning workflows:

- **Privacy Protection**: Generate data without exposing sensitive information
- **Data Augmentation**: Increase dataset size for improved model training
- **Scenario Testing**: Create specific data distributions for testing edge cases
- **Balance Representation**: Address class imbalance issues in your original dataset
- **Development Without Real Data**: Enable development and testing when real data is limited

## Available Generation Methods

DeepBridge provides multiple synthetic data generation methods to suit different needs:

### UltraLightGenerator

A lightweight generative model that efficiently creates synthetic data with low computational overhead, suitable for tabular data with mixed data types.

```python
from deepbridge.synthetic import UltraLightGenerator

# Initialize generator
generator = UltraLightGenerator()

# Fit to your dataset
generator.fit(your_dataframe)

# Generate synthetic samples
synthetic_data = generator.generate(n_samples=1000)
```

[Learn more about UltraLightGenerator](../concepts/synthetic_UltraLightGenerator.md)

### Gaussian Copula

A statistical approach that captures correlations between features using Gaussian copulas, preserving the original feature distributions.

```python
from deepbridge.synthetic import GaussianCopulaGenerator

# Initialize generator
generator = GaussianCopulaGenerator()

# Fit and generate data
synthetic_data = generator.fit_generate(your_dataframe, n_samples=1000)
```

[Learn more about Gaussian Copula](../concepts/synthetic_gaussian_copula.md)

### CTGAN

A GAN-based approach specifically designed for tabular data, capturing complex relationships between continuous and categorical variables.

```python
from deepbridge.synthetic import CTGANGenerator

# Initialize generator
generator = CTGANGenerator(epochs=100)

# Fit to your dataset
generator.fit(your_dataframe)

# Generate synthetic samples
synthetic_data = generator.generate(n_samples=1000)
```

[Learn more about CTGAN](../concepts/synthetic_ctgan.md)

## Quality Assessment

DeepBridge provides tools to evaluate the quality of your synthetic data:

```python
from deepbridge.synthetic import SyntheticDataEvaluator

# Initialize evaluator
evaluator = SyntheticDataEvaluator(original_data, synthetic_data)

# Get comprehensive quality metrics
metrics = evaluator.evaluate()

# Visualize comparison
evaluator.plot_distributions()
evaluator.plot_correlation_comparison()
```

The evaluation includes:

- **Statistical Similarity**: How well the synthetic data matches the statistical properties of the original data
- **Privacy Metrics**: Assessment of re-identification risk
- **Utility Metrics**: How well models trained on synthetic data perform compared to those trained on real data

## Integration with Experiments

You can easily incorporate synthetic data generation into your DeepBridge experiments:

```python
from deepbridge.core.experiment import Experiment
from deepbridge.core.db_data import DBDataset
from deepbridge.synthetic import UltraLightGenerator

# Create experiment with original dataset
dataset = DBDataset(original_df, target_column="target")
experiment = Experiment(dataset)

# Generate synthetic data and add to experiment
generator = UltraLightGenerator()
synthetic_data = generator.fit_generate(original_df, n_samples=5000)
synthetic_dataset = DBDataset(synthetic_data, target_column="target")

# Add synthetic dataset to experiment for comparison
experiment.add_dataset(synthetic_dataset, name="synthetic")

# Run validation on both datasets
results = experiment.validate()
```

## Best Practices

1. **Start Simple**: Begin with UltraLightGenerator for quick results before trying more complex methods
2. **Evaluate Thoroughly**: Always assess the quality of synthetic data before using it
3. **Domain-Specific Validation**: Consider domain-specific metrics relevant to your use case
4. **Privacy Considerations**: For sensitive data, use additional privacy-preserving techniques
5. **Incremental Generation**: Generate data incrementally to assess quality at each step

## Next Steps

- Explore the technical documentation on [synthetic data generation methods](../concepts/synthetic_data.md)
- Learn about the [API reference](../api/synthetic.md) for programmatic control
- Check out advanced topics in [custom data generation models](../advanced/custom_models.md)