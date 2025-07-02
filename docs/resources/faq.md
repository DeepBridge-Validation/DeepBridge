# Frequently Asked Questions (FAQ)

## General Questions

### What is DeepBridge?

DeepBridge is a comprehensive Python library for advanced machine learning model validation, distillation, and performance analysis. It helps you:
- Validate model robustness and reliability
- Create efficient distilled models
- Generate synthetic data
- Produce detailed validation reports

### Who should use DeepBridge?

DeepBridge is designed for:
- Data scientists validating production models
- ML engineers optimizing model deployment
- Researchers testing model reliability
- Teams needing comprehensive model documentation

### What makes DeepBridge different?

DeepBridge provides:
1. **Comprehensive Testing**: Four types of validation tests in one framework
2. **Automated Reporting**: Professional HTML reports with no coding required
3. **Model Distillation**: Easy creation of efficient production models
4. **Privacy-Preserving**: Built-in synthetic data generation

## Installation Issues

### Q: I'm getting a "No module named 'deepbridge'" error

**A:** Make sure you've installed DeepBridge correctly:
```bash
pip install deepbridge
```

If using a virtual environment, ensure it's activated:
```bash
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows
```

### Q: Installation fails with dependency conflicts

**A:** Create a clean virtual environment:
```bash
python -m venv deepbridge-env
source deepbridge-env/bin/activate
pip install deepbridge
```

### Q: Which Python versions are supported?

**A:** DeepBridge supports Python 3.10, 3.11, and 3.12. Check your version:
```bash
python --version
```

## Usage Questions

### Q: How do I start with DeepBridge?

**A:** Follow this simple workflow:
```python
from deepbridge import DBDataset
from deepbridge.core.experiment import Experiment

# 1. Create dataset
dataset = DBDataset(df, target_column='target')

# 2. Create experiment
experiment = Experiment('my_test', dataset, {'model': my_model})

# 3. Run tests
results = experiment.run_test('robustness', config='quick')

# 4. Generate report
experiment.generate_report('robustness', './reports')
```

### Q: What's the difference between 'quick', 'medium', and 'full' configs?

**A:** Configuration levels control test thoroughness:
- **quick**: Fast tests (3-5 iterations) for initial assessment
- **medium**: Balanced tests (5-10 iterations) for standard validation
- **full**: Comprehensive tests (10-20 iterations) for final validation

### Q: Can I use DeepBridge with my custom model?

**A:** Yes! DeepBridge works with any scikit-learn compatible model:
```python
class MyCustomModel:
    def fit(self, X, y):
        # Your implementation
        pass
    
    def predict(self, X):
        # Your implementation
        pass
    
    def predict_proba(self, X):  # For classification
        # Your implementation
        pass
```

## Model Validation

### Q: What tests should I run?

**A:** It depends on your use case:
- **Always run**: Robustness testing (essential for any model)
- **For critical decisions**: Add uncertainty quantification
- **For production**: Include resilience testing
- **During development**: Use hyperparameter importance

### Q: How do I interpret robustness scores?

**A:** Robustness scores indicate model stability:
- **> 0.9**: Excellent robustness
- **0.7-0.9**: Good robustness
- **0.5-0.7**: Moderate vulnerability
- **< 0.5**: High vulnerability

### Q: My model fails uncertainty tests. What should I do?

**A:** Poor uncertainty calibration can be improved by:
1. Using temperature scaling
2. Applying model distillation
3. Ensemble methods
4. Recalibrating with more data

## Model Distillation

### Q: When should I use model distillation?

**A:** Use distillation when you need:
- Faster inference (mobile/edge deployment)
- Smaller model size
- Better uncertainty estimates
- Simplified model interpretation

### Q: Which student model type should I choose?

**A:** Depends on your constraints:
- **MLP**: Fast inference, small size
- **GBM**: Good performance, interpretable
- **XGBoost**: Best performance, moderate size

### Q: Can I distill ensemble models?

**A:** Yes! DeepBridge supports distilling from any model that provides predictions:
```python
# Ensemble predictions
predictions = ensemble_model.predict_proba(X)

# Distill to single model
distiller = AutoDistiller(dataset)
student = distiller.run(use_probabilities=True)
```

## Synthetic Data

### Q: Is synthetic data safe for production testing?

**A:** Yes, when generated properly:
- Preserves statistical properties
- Removes personally identifiable information
- Maintains relationships between features
- Always validate quality metrics before use

### Q: Which synthetic data method should I use?

**A:** 
- **Gaussian Copula**: Best for mixed data types, preserves correlations
- **CTGAN**: Good for complex distributions (coming soon)
- **Standard Generator**: Fast, simple data

### Q: How many synthetic samples should I generate?

**A:** Generally:
- Same size as original: For testing
- 2-5x original: For augmentation
- 10x original: For privacy-critical applications

## Reports

### Q: How do I customize report appearance?

**A:** Add custom CSS:
```python
experiment.generate_report(
    test_type='robustness',
    output_dir='./reports',
    custom_css="""
    .metric-card {
        background: #custom-color;
    }
    """
)
```

### Q: Can I generate PDF reports?

**A:** Use static reports and convert:
```python
# Generate static report
experiment.generate_report(
    test_type='robustness',
    format='static',
    static_options={'save_charts': True}
)
# Then use wkhtmltopdf or similar tool
```

### Q: Reports are too large. How can I reduce size?

**A:** Several options:
1. Use static format with separate chart files
2. Enable data sampling for large datasets
3. Limit the number of features analyzed
4. Use 'quick' configuration

## Performance

### Q: DeepBridge is slow on large datasets. What can I do?

**A:** Optimize performance:
```python
# 1. Enable parallel processing
import os
os.environ['DEEPBRIDGE_N_JOBS'] = '-1'

# 2. Use Dask for large data
dataset = DBDataset(df, use_dask=True)

# 3. Sample data for testing
sampled_df = df.sample(n=10000)
```

### Q: How much memory does DeepBridge need?

**A:** Memory usage depends on:
- Dataset size: ~3-5x dataset memory
- Test configuration: 'full' uses more memory
- Number of models: Each model adds overhead

### Q: Can I run DeepBridge on GPU?

**A:** DeepBridge itself doesn't use GPU, but your models can:
```python
# Use GPU-enabled models
import xgboost as xgb
model = xgb.XGBClassifier(tree_method='gpu_hist')
```

## Integration

### Q: How do I integrate DeepBridge with MLflow?

**A:** See [complete example](../tutorials/complete_workflow.md)

### Q: Can I use DeepBridge in CI/CD pipelines?

**A:** Yes! DeepBridge has full CLI support:
```yaml
# .github/workflows/validate.yml
- name: Validate Model
  run: |
    deepbridge validate \
      --dataset data.csv \
      --model model.pkl \
      --tests all
```

### Q: Does DeepBridge work with cloud services?

**A:** Yes, DeepBridge works with:
- AWS SageMaker
- Google Cloud AI Platform
- Azure ML
- Any service supporting Python libraries

## Troubleshooting

### Q: I'm getting "Model not fitted" errors

**A:** Ensure your model is trained before validation:
```python
model.fit(X_train, y_train)  # Train first
experiment = Experiment('test', dataset, {'model': model})
```

### Q: Charts aren't displaying in reports

**A:** Check:
1. Using a modern browser (Chrome, Firefox, Safari)
2. JavaScript is enabled
3. No browser extensions blocking scripts
4. Try static format as alternative

### Q: Results vary between runs

**A:** Set random seeds for reproducibility:
```python
import numpy as np
np.random.seed(42)

# In your models
model = RandomForestClassifier(random_state=42)
```

## Best Practices

### Q: What's a good validation workflow?

**A:** Follow this approach:
1. Start with 'quick' tests
2. If issues found, run 'medium' tests
3. For production, use 'full' tests
4. Always generate reports for documentation

### Q: Should I test on training or test data?

**A:** Always use test data:
```python
X_train, X_test, y_train, y_test = dataset.train_test_split()
model.fit(X_train, y_train)
# Create test dataset for validation
test_dataset = DBDataset(
    pd.DataFrame(X_test, columns=dataset.features),
    target_column='target'
)
```

### Q: How often should I re-validate models?

**A:** Validate when:
- Before deployment
- After retraining
- When data distribution changes
- Periodically in production (monthly/quarterly)

## Getting Help

### Q: Where can I find more examples?

**A:** Check these resources:
- [Basic Examples](../tutorials/basic_examples.md)
- [Complete Workflow](../tutorials/complete_workflow.md)
- [GitHub Examples](https://github.com/DeepBridge-Validation/DeepBridge/tree/main/examples)

### Q: How do I report bugs?

**A:** Report issues on [GitHub](https://github.com/DeepBridge-Validation/DeepBridge/issues) with:
- DeepBridge version
- Python version
- Minimal code example
- Error traceback

### Q: Can I contribute to DeepBridge?

**A:** Yes! See our [Contributing Guide](../contributing.md) for:
- Code style guidelines
- Testing requirements
- Pull request process

### Q: Is commercial support available?

**A:** For commercial support, consulting, or custom features, contact: gustavo.haase@gmail.com