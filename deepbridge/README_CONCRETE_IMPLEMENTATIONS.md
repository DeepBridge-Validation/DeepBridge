# Concrete Implementations for DeepBridge Interfaces

This document describes the concrete implementations of the abstract interfaces defined in the DeepBridge library.

## Overview

As part of the refactoring effort to improve code organization and maintainability, we've created abstract base classes (interfaces) for key components in the codebase. These interfaces define the contract that concrete implementations must follow.

This document covers the following concrete implementations:

1. `StandardReporter` - Implementation of `BaseReporter` interface
2. `StandardProcessor` - Implementation of `BaseProcessor` interface
3. `StandardGenerator` - Implementation of `BaseGenerator` interface

## StandardReporter

The `StandardReporter` class provides a standard implementation of the `BaseReporter` interface for generating HTML reports from data.

### Features

- Template-based report generation using Jinja2
- Fallback to basic HTML generation if template is missing
- Support for converting various data types (including Pandas DataFrames and NumPy arrays) to HTML
- Automatic formatting of values for display
- Structured data visualization with collapsible sections

### Usage Example

```python
from deepbridge.reporting import StandardReporter

# Create reporter
reporter = StandardReporter(template_dir='/path/to/templates', template_name='report.html')

# Prepare data for report
report_data = {
    'metrics': {
        'accuracy': 0.95,
        'precision': 0.92,
        'recall': 0.91
    },
    'model_info': {
        'name': 'RandomForest',
        'params': {'n_estimators': 100}
    }
}

# Generate and save report
report_path = reporter.save_report(
    'experiment_report.html',
    report_data,
    report_title='Model Evaluation Report',
    create_basic_html=True  # Fallback if template is missing
)
```

## StandardProcessor

The `StandardProcessor` class provides a standard implementation of the `BaseProcessor` interface for processing data for machine learning tasks.

### Features

- Automatic data type inference (numerical vs categorical)
- Support for various scaling options (standard, minmax, robust)
- Missing value handling
- Outlier detection and capping
- Categorical data encoding (one-hot, label)
- Preservation of data types

### Usage Example

```python
from deepbridge.core import StandardProcessor
import pandas as pd

# Create processor
processor = StandardProcessor(
    verbose=True,
    scaler_type='standard',
    handle_missing=True,
    handle_outliers=True,
    categorical_encoding='onehot'
)

# Load data
data = pd.read_csv('data.csv')

# Process data with automatic column type inference
processed_data = processor.process(
    data,
    fit=True,
    target_column='target'
)

# Process new data using the fitted processor
new_data = pd.read_csv('new_data.csv')
processed_new = processor.process(new_data, target_column='target')
```

## StandardGenerator

The `StandardGenerator` class provides a standard implementation of the `BaseGenerator` interface for generating synthetic data.

### Features

- Multiple generation methods (Gaussian, GMM, KDE, Bootstrap)
- Preservation of feature correlations
- Support for both numerical and categorical features
- Outlier generation for robust testing
- Control over noise levels
- Preservation of original data types

### Usage Example

```python
from deepbridge.synthetic import StandardGenerator
import pandas as pd

# Create generator
generator = StandardGenerator(
    random_state=42,
    verbose=True,
    method='gmm',  # Gaussian Mixture Model
    n_components=3,
    preserve_correlations=True,
    outlier_rate=0.05
)

# Load data
data = pd.read_csv('data.csv')

# Fit generator to the data
generator.fit(
    data,
    target_column='target'
)

# Generate synthetic data
synthetic_data = generator.generate(
    num_samples=1000,
    noise_level=0.1
)
```

## Combined Example

See the `examples/standard_implementations_example.py` file for a complete example that demonstrates using all three implementations together.