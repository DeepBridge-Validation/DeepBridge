# Basic Examples

This guide provides simple examples to get you started with DeepBridge's main features.

## 1. Basic Model Validation

### Simple Robustness Test

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from deepbridge import DBDataset
from deepbridge.validation.wrappers import RobustnessSuite

# Create sample data
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5)
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
df['target'] = y

# Train a model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Create dataset
dataset = DBDataset(
    data=df,
    target_column='target',
    features=[f'feature_{i}' for i in range(10)]
)

# Run robustness test
suite = RobustnessSuite(dataset=dataset, model=model, config='quick')
results = suite.run()

print(f"Model robustness score: {1 - results['avg_impact']:.2f}")
print(f"Most sensitive features: {list(results['feature_importance'].keys())[:3]}")
```

### Quick Uncertainty Check

```python
from deepbridge.validation.wrappers import UncertaintySuite

# Run uncertainty quantification
suite = UncertaintySuite(dataset=dataset, model=model, config='quick')
results = suite.run()

print(f"Uncertainty quality score: {results['uncertainty_quality_score']:.2f}")
print(f"Average coverage error: {results['coverage_error']:.3f}")
```

## 2. Model Distillation Example

### Basic Knowledge Distillation

```python
from deepbridge.distillation import KnowledgeDistillation
from sklearn.neural_network import MLPClassifier

# Assume we have a complex teacher model
teacher_model = RandomForestClassifier(n_estimators=500, max_depth=20)
teacher_model.fit(X, y)

# Create a simpler student model
distiller = KnowledgeDistillation(
    teacher_model=teacher_model,
    student_type='mlp',
    temperature=3.0
)

# Train student
student_model = distiller.fit(X, y)

# Compare performance
teacher_score = teacher_model.score(X, y)
student_score = student_model.score(X, y)

print(f"Teacher accuracy: {teacher_score:.3f}")
print(f"Student accuracy: {student_score:.3f}")
print(f"Compression ratio: {distiller.get_compression_ratio():.1f}x")
```

## 3. Running a Complete Experiment

### End-to-End Validation Workflow

```python
from deepbridge.core.experiment import Experiment

# Create experiment with multiple models
experiment = Experiment(
    name='model_comparison',
    dataset=dataset,
    models={
        'rf_baseline': RandomForestClassifier(n_estimators=100),
        'rf_tuned': RandomForestClassifier(n_estimators=200, max_depth=10),
    }
)

# Run multiple tests
test_types = ['robustness', 'uncertainty', 'resilience']
all_results = {}

for test in test_types:
    print(f"Running {test} test...")
    results = experiment.run_test(test, config='quick')
    all_results[test] = results
    
# Generate comprehensive report
experiment.generate_report(
    test_type='robustness',
    output_dir='./my_first_report'
)

print("Report generated successfully!")
```

## 4. Synthetic Data Generation

### Generate Privacy-Preserving Data

```python
from deepbridge.synthetic import StandardGenerator

# Load sensitive data
sensitive_df = pd.read_csv('sensitive_data.csv')

# Generate synthetic version
generator = StandardGenerator(method='gaussian_copula')
synthetic_df = generator.fit_generate(sensitive_df, n_samples=1000)

# Evaluate quality
from deepbridge.synthetic.metrics import SyntheticMetrics

metrics = SyntheticMetrics()
quality_report = metrics.evaluate(
    real_data=sensitive_df,
    synthetic_data=synthetic_df
)

print(f"Statistical similarity: {quality_report['statistical']['overall']:.2f}")
print(f"Privacy score: {quality_report['privacy']['score']:.2f}")
```

## 5. Command Line Examples

### Basic CLI Usage

```bash
# Validate a model
deepbridge validate \
    --dataset my_data.csv \
    --model my_model.pkl \
    --tests robustness \
    --config quick

# Generate synthetic data
deepbridge synthetic generate \
    --data original.csv \
    --output synthetic.csv \
    --method gaussian_copula \
    --samples 5000

# Create a distilled model
deepbridge distill train \
    --teacher complex_model.pkl \
    --data training_data.csv \
    --student-type mlp \
    --output simple_model.pkl
```

## 6. Working with Different Data Types

### Classification Example

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

# Create DataFrame
df_train = pd.DataFrame(X_train, columns=iris.feature_names)
df_train['species'] = y_train

# Create dataset
dataset = DBDataset(
    data=df_train,
    target_column='species',
    task='classification'
)

print(f"Dataset shape: {dataset.data.shape}")
print(f"Number of classes: {dataset.n_classes}")
```

### Regression Example

```python
from sklearn.datasets import make_regression

# Generate regression data
X, y = make_regression(n_samples=1000, n_features=5, noise=10)
df = pd.DataFrame(X, columns=[f'var_{i}' for i in range(5)])
df['target'] = y

# Create dataset
dataset = DBDataset(
    data=df,
    target_column='target',
    task='regression'
)

# Train and validate a regression model
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(dataset.get_features(), dataset.get_target())

# Run validation
experiment = Experiment('regression_test', dataset, {'linear': model})
results = experiment.run_test('robustness', config='quick')
```

## 7. Feature Importance Analysis

### Understanding Model Decisions

```python
from deepbridge.utils import FeatureManager

# Analyze features
manager = FeatureManager()
feature_stats = manager.get_feature_stats(dataset.data)

print("Feature Statistics:")
for feat, stats in feature_stats.items():
    print(f"{feat}: mean={stats['mean']:.2f}, std={stats['std']:.2f}")

# Get feature importance from robustness test
robustness_suite = RobustnessSuite(dataset, model)
results = robustness_suite.run()

# Plot feature importance
import matplotlib.pyplot as plt

features = list(results['feature_importance'].keys())
importances = list(results['feature_importance'].values())

plt.figure(figsize=(10, 6))
plt.bar(features, importances)
plt.xlabel('Features')
plt.ylabel('Robustness Impact')
plt.title('Feature Sensitivity Analysis')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

## Next Steps

Now that you've seen basic examples:

1. **Dive Deeper**: Check the [User Guide](../guides/validation.md) for detailed explanations
2. **Advanced Features**: Explore [Technical Documentation](../technical/implementation_guide.md)
3. **Real Projects**: See [Complete Workflow Example](complete_workflow.md)
4. **CLI Mastery**: Learn more in [CLI Usage Guide](../guides/cli.md)

## Tips for Beginners

1. **Start Small**: Use 'quick' configuration for initial tests
2. **Understand Your Data**: Use `DBDataset` to ensure proper data formatting
3. **Compare Models**: Always test multiple models for benchmarking
4. **Save Results**: Use experiment tracking to save all results
5. **Visualize**: Generate reports to better understand model behavior

## Common Patterns

### Pattern 1: Quick Model Assessment
```python
# Quick 3-step validation
dataset = DBDataset(df, 'target')
suite = RobustnessSuite(dataset, model, 'quick')
results = suite.run()
```

### Pattern 2: Full Pipeline
```python
# Complete validation pipeline
experiment = Experiment('full_validation', dataset, models)
experiment.run_all_tests('medium')
experiment.generate_report('all', './reports')
```

### Pattern 3: Model Comparison
```python
# Compare multiple models
models = {
    'simple': LogisticRegression(),
    'complex': RandomForestClassifier(),
    'distilled': distilled_model
}
experiment = Experiment('comparison', dataset, models)
```