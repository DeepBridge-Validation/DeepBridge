# Troubleshooting Guide

This guide helps you resolve common issues when using DeepBridge.

## Installation Problems

### ImportError: No module named 'deepbridge'

**Problem**: Python can't find DeepBridge after installation.

**Solutions**:
1. Verify installation:
   ```bash
   pip list | grep deepbridge
   ```

2. Reinstall in the correct environment:
   ```bash
   pip uninstall deepbridge
   pip install deepbridge
   ```

3. Check Python path:
   ```python
   import sys
   print(sys.path)
   # Ensure your environment's site-packages is listed
   ```

### Dependency Conflicts

**Problem**: Installation fails due to conflicting package versions.

**Solutions**:
1. Use a clean virtual environment:
   ```bash
   python -m venv clean_env
   source clean_env/bin/activate
   pip install deepbridge
   ```

2. Force reinstall dependencies:
   ```bash
   pip install --force-reinstall deepbridge
   ```

3. Install specific versions:
   ```bash
   pip install "deepbridge==0.1.39"
   ```

### Build Errors on Windows

**Problem**: Compilation errors for dependencies like XGBoost.

**Solutions**:
1. Install Visual C++ Build Tools
2. Use conda instead of pip:
   ```bash
   conda install -c conda-forge xgboost
   pip install deepbridge
   ```

## Runtime Errors

### "Model is not fitted" Error

**Problem**: Trying to validate an untrained model.

**Solution**:
```python
# Wrong
model = RandomForestClassifier()
experiment = Experiment('test', dataset, {'model': model})

# Correct
model = RandomForestClassifier()
model.fit(X_train, y_train)  # Train first!
experiment = Experiment('test', dataset, {'model': model})
```

### AttributeError: 'NoneType' object has no attribute

**Problem**: Missing required parameters or data.

**Solution**:
```python
# Check your dataset
print(dataset.features)  # Should not be None
print(dataset.target)    # Should not be None

# Verify model has required methods
print(hasattr(model, 'predict'))  # Should be True
print(hasattr(model, 'fit'))      # Should be True
```

### ValueError: Input contains NaN

**Problem**: Dataset contains missing values.

**Solution**:
```python
# Check for NaN values
print(df.isnull().sum())

# Handle missing values
df_clean = df.dropna()  # or df.fillna(method='mean')

# Create dataset with clean data
dataset = DBDataset(df_clean, target_column='target')
```

## Memory Issues

### MemoryError During Testing

**Problem**: Large datasets causing out-of-memory errors.

**Solutions**:

1. **Sample your data**:
   ```python
   # Use a subset for validation
   sample_df = df.sample(n=10000, random_state=42)
   dataset = DBDataset(sample_df, target_column='target')
   ```

2. **Use quick configuration**:
   ```python
   # Reduces memory usage
   results = experiment.run_test('robustness', config='quick')
   ```

3. **Enable Dask for large data**:
   ```python
   import dask.dataframe as dd
   
   # Convert to Dask DataFrame
   dask_df = dd.from_pandas(df, npartitions=10)
   ```

4. **Process in batches**:
   ```python
   # Custom batch processing
   batch_size = 1000
   results = []
   
   for i in range(0, len(df), batch_size):
       batch = df.iloc[i:i+batch_size]
       batch_dataset = DBDataset(batch, target_column='target')
       result = suite.run()
       results.append(result)
   ```

### Jupyter Kernel Dying

**Problem**: Kernel crashes during large computations.

**Solutions**:
1. Increase memory limit:
   ```bash
   jupyter notebook --NotebookApp.max_buffer_size=1000000000
   ```

2. Clear outputs regularly:
   ```python
   import gc
   gc.collect()
   ```

## Report Generation Issues

### Blank or Missing Charts

**Problem**: Reports generate but charts don't display.

**Solutions**:

1. **Check browser console** (F12) for JavaScript errors

2. **Use static reports**:
   ```python
   experiment.generate_report(
       test_type='robustness',
       format='static'
   )
   ```

3. **Verify data format**:
   ```python
   # Ensure results contain expected keys
   print(results.keys())
   # Should include: 'base_score', 'raw', 'feature_importance', etc.
   ```

### Large Report Files

**Problem**: HTML reports are too large (>100MB).

**Solutions**:

1. **Enable data sampling**:
   ```python
   # Limit data points in visualizations
   experiment.generate_report(
       test_type='robustness',
       output_dir='./reports',
       max_chart_points=1000
   )
   ```

2. **Use static format with separate files**:
   ```python
   experiment.generate_report(
       test_type='robustness',
       format='static',
       static_options={'save_charts': True}
   )
   ```

### Report Generation Hangs

**Problem**: Report generation never completes.

**Solutions**:

1. **Check disk space**:
   ```bash
   df -h  # Linux/macOS
   ```

2. **Enable verbose logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

3. **Generate reports sequentially**:
   ```python
   # Instead of all at once
   for test_type in ['robustness', 'uncertainty']:
       experiment.generate_report(test_type, f'./reports/{test_type}')
   ```

## Model-Specific Issues

### XGBoost Installation Failed

**Problem**: XGBoost won't install or import.

**Solutions**:

1. **Use pre-built wheels**:
   ```bash
   pip install xgboost --no-build-isolation
   ```

2. **Install system dependencies** (Ubuntu/Debian):
   ```bash
   sudo apt-get install build-essential
   ```

3. **Use conda**:
   ```bash
   conda install -c conda-forge py-xgboost
   ```

### Scikit-learn Compatibility

**Problem**: "This model is not supported" errors.

**Solution**: Ensure your model follows scikit-learn API:
```python
from sklearn.base import BaseEstimator, ClassifierMixin

class MyModel(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        # Your implementation
        return self
    
    def predict(self, X):
        # Your implementation
        return predictions
    
    def predict_proba(self, X):
        # For classification
        return probabilities
```

## Performance Issues

### Slow Test Execution

**Problem**: Tests take too long to complete.

**Solutions**:

1. **Enable parallel processing**:
   ```python
   import os
   os.environ['DEEPBRIDGE_N_JOBS'] = '-1'  # Use all cores
   ```

2. **Use quick configuration first**:
   ```python
   # Start with quick
   quick_results = experiment.run_test('robustness', config='quick')
   
   # Only run full if needed
   if quick_results['avg_impact'] > 0.1:
       full_results = experiment.run_test('robustness', config='full')
   ```

3. **Profile your code**:
   ```python
   import cProfile
   import pstats
   
   profiler = cProfile.Profile()
   profiler.enable()
   
   # Your DeepBridge code
   results = experiment.run_test('robustness')
   
   profiler.disable()
   stats = pstats.Stats(profiler).sort_stats('cumtime')
   stats.print_stats(10)  # Top 10 time-consuming functions
   ```

### High CPU Usage

**Problem**: DeepBridge uses all CPU cores.

**Solution**: Limit parallel jobs:
```python
# Use only 2 cores
os.environ['DEEPBRIDGE_N_JOBS'] = '2'

# Or in configuration
config = {
    'n_jobs': 2,
    'n_trials': 5
}
results = experiment.run_test('robustness', config=config)
```

## Data Issues

### Inconsistent Results

**Problem**: Results vary between runs.

**Solution**: Set all random seeds:
```python
import random
import numpy as np

# Set seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Use in models
model = RandomForestClassifier(random_state=SEED)

# Use in data splitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)
```

### Feature Mismatch Errors

**Problem**: "Feature names mismatch" between training and testing.

**Solution**:
```python
# Ensure consistent feature order
feature_names = ['feat1', 'feat2', 'feat3']

# Create datasets with explicit features
train_dataset = DBDataset(
    train_df,
    target_column='target',
    features=feature_names
)

test_dataset = DBDataset(
    test_df,
    target_column='target',
    features=feature_names  # Same order!
)
```

## Common Warning Messages

### "ConvergenceWarning: Liblinear failed to converge"

**Meaning**: Model optimization didn't fully converge.

**Solution**:
```python
# Increase iterations
model = LogisticRegression(max_iter=1000)  # Default is 100
```

### "DataConversionWarning: A column-vector y was passed"

**Meaning**: Target variable shape issue.

**Solution**:
```python
# Flatten target array
y = y.ravel()  # or y.reshape(-1)
```

### "UserWarning: No data for colormapping provided"

**Meaning**: Empty data in visualization.

**Solution**: Check that your results contain data:
```python
if len(results['plot_data']) > 0:
    experiment.generate_report('robustness')
else:
    print("No data to plot")
```

## Getting Additional Help

### Enable Debug Mode

```python
# Set up detailed logging
import logging

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deepbridge_debug.log'),
        logging.StreamHandler()
    ]
)

# Now run your code
logger = logging.getLogger(__name__)
logger.debug("Starting validation...")
```

### Collect System Information

```python
import sys
import platform
import deepbridge

print(f"Python version: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"DeepBridge version: {deepbridge.__version__}")

# List all installed packages
import pkg_resources
installed_packages = [d.project_name for d in pkg_resources.working_set]
print(f"Installed packages: {sorted(installed_packages)}")
```

### Create Minimal Reproducible Example

When reporting issues:

```python
# Minimal example that reproduces the issue
import numpy as np
import pandas as pd
from deepbridge import DBDataset
from deepbridge.validation.wrappers import RobustnessSuite
from sklearn.ensemble import RandomForestClassifier

# Small synthetic data
X = np.random.rand(100, 5)
y = np.random.randint(0, 2, 100)
df = pd.DataFrame(X, columns=[f'f{i}' for i in range(5)])
df['target'] = y

# Your issue here
dataset = DBDataset(df, 'target')
model = RandomForestClassifier().fit(X, y)
suite = RobustnessSuite(dataset, model)
results = suite.run()  # <-- Error occurs here
```

### Contact Support

If you still need help:

1. Check [FAQ](faq.md) first
2. Search [GitHub Issues](https://github.com/DeepBridge-Validation/DeepBridge/issues)
3. Create new issue with:
   - DeepBridge version
   - Full error traceback
   - Minimal code example
   - System information

For commercial support: gustavo.haase@gmail.com