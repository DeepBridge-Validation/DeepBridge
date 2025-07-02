# Complete Workflow Example

This tutorial demonstrates a complete end-to-end workflow using DeepBridge for a real-world scenario: validating and optimizing a credit risk model.

## Scenario Overview

We'll work through:
1. Loading and preparing credit risk data
2. Training multiple models
3. Running comprehensive validation tests
4. Creating a distilled model for production
5. Generating synthetic data for testing
6. Creating detailed reports

## Step 1: Data Preparation

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from deepbridge import DBDataset

# Load credit risk dataset
# For this example, we'll create a synthetic credit dataset
np.random.seed(42)
n_samples = 10000

# Generate features
data = {
    'age': np.random.randint(18, 70, n_samples),
    'income': np.random.lognormal(10.5, 0.5, n_samples),
    'credit_score': np.random.randint(300, 850, n_samples),
    'num_credit_lines': np.random.poisson(3, n_samples),
    'debt_to_income': np.random.beta(2, 5, n_samples),
    'employment_years': np.random.exponential(5, n_samples),
    'previous_defaults': np.random.binomial(2, 0.1, n_samples),
    'loan_amount': np.random.lognormal(9.5, 0.8, n_samples)
}

df = pd.DataFrame(data)

# Create target based on features (simplified risk model)
risk_score = (
    - 0.01 * df['age']
    - 0.00001 * df['income']
    + 0.002 * (850 - df['credit_score'])
    + 0.1 * df['num_credit_lines']
    + 2 * df['debt_to_income']
    - 0.05 * df['employment_years']
    + 0.5 * df['previous_defaults']
    + 0.00001 * df['loan_amount']
    + np.random.normal(0, 0.5, n_samples)
)

df['default'] = (risk_score > np.percentile(risk_score, 80)).astype(int)

print(f"Dataset shape: {df.shape}")
print(f"Default rate: {df['default'].mean():.2%}")
print("\nFeature summary:")
print(df.describe())
```

## Step 2: Create Dataset and Split

```python
# Define features
features = ['age', 'income', 'credit_score', 'num_credit_lines', 
            'debt_to_income', 'employment_years', 'previous_defaults', 
            'loan_amount']

# Create DeepBridge dataset
dataset = DBDataset(
    data=df,
    target_column='default',
    features=features,
    task='classification'
)

# Split data
X_train, X_test, y_train, y_test = dataset.train_test_split(
    test_size=0.2,
    random_state=42,
    stratify=dataset.get_target()
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
```

## Step 3: Train Multiple Models

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Scale features for some models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train different models
models = {}

# 1. Logistic Regression (baseline)
print("Training Logistic Regression...")
models['logistic'] = LogisticRegression(max_iter=1000, random_state=42)
models['logistic'].fit(X_train_scaled, y_train)

# 2. Random Forest
print("Training Random Forest...")
models['random_forest'] = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=20,
    random_state=42
)
models['random_forest'].fit(X_train, y_train)

# 3. XGBoost
print("Training XGBoost...")
models['xgboost'] = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
models['xgboost'].fit(X_train, y_train)

# 4. Neural Network
print("Training Neural Network...")
models['neural_net'] = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    max_iter=1000,
    random_state=42
)
models['neural_net'].fit(X_train_scaled, y_train)

# Evaluate base performance
from sklearn.metrics import classification_report

print("\nBase Model Performance:")
for name, model in models.items():
    if name in ['logistic', 'neural_net']:
        score = model.score(X_test_scaled, y_test)
    else:
        score = model.score(X_test, y_test)
    print(f"{name}: {score:.3f}")
```

## Step 4: Run Comprehensive Validation

```python
from deepbridge.core.experiment import Experiment

# Create experiment
experiment = Experiment(
    name='credit_risk_validation',
    dataset=dataset,
    models=models,
    output_dir='./credit_risk_analysis'
)

# Run all validation tests
print("\nRunning validation tests...")
validation_results = {}

# 1. Robustness Testing
print("Testing robustness...")
robustness_results = experiment.run_test('robustness', config='medium')
validation_results['robustness'] = robustness_results

# 2. Uncertainty Quantification
print("Testing uncertainty...")
uncertainty_results = experiment.run_test('uncertainty', config='medium')
validation_results['uncertainty'] = uncertainty_results

# 3. Resilience Testing
print("Testing resilience...")
resilience_results = experiment.run_test('resilience', config='medium')
validation_results['resilience'] = resilience_results

# 4. Hyperparameter Importance
print("Testing hyperparameters...")
hyperparam_results = experiment.run_test('hyperparameter', config='quick')
validation_results['hyperparameter'] = hyperparam_results

# Summarize results
print("\n=== Validation Summary ===")
for test, results in validation_results.items():
    if test == 'robustness':
        print(f"\nRobustness Test:")
        for model_name in models.keys():
            if model_name in results:
                score = 1 - results[model_name]['avg_impact']
                print(f"  {model_name}: {score:.3f}")
```

## Step 5: Model Distillation

```python
from deepbridge.distillation import AutoDistiller

# Select best performing model as teacher
teacher_model = models['xgboost']  # Assuming XGBoost performed best

# Create predictions for distillation
teacher_predictions = teacher_model.predict_proba(X_train)

# Add predictions to dataset
distillation_df = pd.DataFrame(X_train, columns=features)
distillation_df['target'] = y_train
distillation_df['prob_0'] = teacher_predictions[:, 0]
distillation_df['prob_1'] = teacher_predictions[:, 1]

# Create distillation dataset
distillation_dataset = DBDataset(
    data=distillation_df,
    target_column='target',
    features=features,
    prob_cols=['prob_0', 'prob_1']
)

# Run automated distillation
print("\nRunning model distillation...")
distiller = AutoDistiller(
    dataset=distillation_dataset,
    output_dir='./distillation_results',
    model_types=['mlp', 'gbm'],
    n_trials=20,
    test_size=0.2
)

distillation_results = distiller.run(use_probabilities=True)
best_student = distiller.get_best_model()

print(f"\nBest student model: {distillation_results['best_model_type']}")
print(f"Student performance: {distillation_results['best_score']:.3f}")
```

## Step 6: Generate Synthetic Data

```python
from deepbridge.synthetic import StandardGenerator
from deepbridge.synthetic.metrics import SyntheticMetrics

# Generate synthetic data for safe testing
print("\nGenerating synthetic data...")
generator = StandardGenerator(method='gaussian_copula')

# Fit on real data and generate synthetic samples
synthetic_df = generator.fit_generate(
    df[features + ['default']], 
    n_samples=5000
)

# Evaluate synthetic data quality
metrics = SyntheticMetrics()
quality_report = metrics.evaluate(
    real_data=df[features + ['default']],
    synthetic_data=synthetic_df
)

print(f"Statistical similarity: {quality_report['statistical']['overall']:.3f}")
print(f"Privacy score: {quality_report['privacy']['score']:.3f}")
print(f"Utility score: {quality_report['utility']['ml_efficacy']:.3f}")

# Test models on synthetic data
print("\nTesting models on synthetic data:")
X_synthetic = synthetic_df[features]
y_synthetic = synthetic_df['default']

for name, model in models.items():
    if name in ['logistic', 'neural_net']:
        X_test_data = scaler.transform(X_synthetic)
    else:
        X_test_data = X_synthetic
    
    score = model.score(X_test_data, y_synthetic)
    print(f"{name}: {score:.3f}")
```

## Step 7: Generate Comprehensive Reports

```python
# Generate reports for all tests
print("\nGenerating reports...")

# 1. Individual test reports
for test_type in ['robustness', 'uncertainty', 'resilience']:
    experiment.generate_report(
        test_type=test_type,
        output_dir=f'./reports/{test_type}',
        format='interactive'
    )

# 2. Generate static report for documentation
experiment.generate_report(
    test_type='robustness',
    output_dir='./reports/static',
    format='static',
    static_options={
        'save_charts': True,
        'dpi': 300,
        'figure_size': (10, 6)
    }
)

print("Reports generated successfully!")
```

## Step 8: Production Deployment Preparation

```python
# Prepare final model for deployment
from deepbridge.utils import ModelRegistry
import joblib

# Select final model (distilled version)
production_model = best_student

# Save model and metadata
model_metadata = {
    'model_type': type(production_model).__name__,
    'training_date': pd.Timestamp.now().isoformat(),
    'validation_scores': {
        'accuracy': distillation_results['best_score'],
        'robustness': 1 - robustness_results[distillation_results['best_model_type']]['avg_impact'],
        'uncertainty_quality': uncertainty_results[distillation_results['best_model_type']]['uncertainty_quality_score']
    },
    'features': features,
    'scaler_required': distillation_results['best_model_type'] in ['mlp']
}

# Save artifacts
joblib.dump(production_model, './models/credit_risk_model.pkl')
joblib.dump(model_metadata, './models/credit_risk_metadata.pkl')
if model_metadata['scaler_required']:
    joblib.dump(scaler, './models/credit_risk_scaler.pkl')

print("\nModel saved for production deployment")
print(f"Model type: {model_metadata['model_type']}")
print(f"Validation scores: {model_metadata['validation_scores']}")
```

## Step 9: Create Model Card

```python
# Generate model card for documentation
model_card = f"""
# Credit Risk Model Card

## Model Details
- **Model Type**: {model_metadata['model_type']}
- **Training Date**: {model_metadata['training_date']}
- **Purpose**: Predict credit default risk for loan applications

## Performance Metrics
- **Accuracy**: {model_metadata['validation_scores']['accuracy']:.3f}
- **Robustness Score**: {model_metadata['validation_scores']['robustness']:.3f}
- **Uncertainty Quality**: {model_metadata['validation_scores']['uncertainty_quality']:.3f}

## Training Data
- **Samples**: {len(df)}
- **Features**: {', '.join(features)}
- **Default Rate**: {df['default'].mean():.2%}

## Validation Tests Performed
1. **Robustness Testing**: Model maintains {model_metadata['validation_scores']['robustness']:.1%} performance under perturbations
2. **Uncertainty Quantification**: Well-calibrated prediction intervals
3. **Resilience Testing**: Stable under distribution shifts
4. **Synthetic Data Testing**: Consistent performance on privacy-preserving synthetic data

## Limitations
- Model trained on simplified synthetic data
- May require retraining with real production data
- Performance may degrade with significant distribution shifts

## Deployment Considerations
- Requires feature scaling: {model_metadata['scaler_required']}
- Input features must match training schema
- Monitor for data drift in production
"""

with open('./models/MODEL_CARD.md', 'w') as f:
    f.write(model_card)

print("Model card created successfully!")
```

## Step 10: Command Line Automation

```python
# Create automation script
automation_script = """#!/bin/bash
# Credit Risk Model Validation Pipeline

echo "Starting credit risk model validation pipeline..."

# 1. Validate models
deepbridge validate \\
    --dataset credit_data.csv \\
    --models models/ \\
    --tests all \\
    --config medium \\
    --output validation_results/

# 2. Generate synthetic data
deepbridge synthetic generate \\
    --data credit_data.csv \\
    --method gaussian_copula \\
    --samples 10000 \\
    --output synthetic_credit_data.csv

# 3. Create distilled model
deepbridge distill auto \\
    --dataset credit_data.csv \\
    --teacher best_model.pkl \\
    --trials 50 \\
    --output distilled_model.pkl

# 4. Generate reports
deepbridge report \\
    --results validation_results/ \\
    --output reports/ \\
    --format interactive

echo "Pipeline completed successfully!"
"""

with open('./run_validation.sh', 'w') as f:
    f.write(automation_script)

print("Automation script created: run_validation.sh")
```

## Summary and Next Steps

This complete workflow demonstrated:

1. **Data Preparation**: Loading and preparing real-world style data
2. **Model Training**: Training multiple model types for comparison
3. **Comprehensive Validation**: Running all DeepBridge validation tests
4. **Model Distillation**: Creating efficient production models
5. **Synthetic Data**: Generating privacy-preserving test data
6. **Report Generation**: Creating detailed validation reports
7. **Production Prep**: Saving models with metadata
8. **Documentation**: Creating model cards
9. **Automation**: Setting up repeatable pipelines

### Key Takeaways

- Always validate models comprehensively before deployment
- Use distillation to create efficient production models
- Generate synthetic data for safe testing
- Document everything with model cards
- Automate validation pipelines for consistency

### Next Steps

1. **Customize for Your Use Case**: Adapt this workflow to your specific domain
2. **Add Monitoring**: Implement production monitoring using validation metrics
3. **Continuous Validation**: Set up automated validation in CI/CD
4. **Explore Advanced Features**: Try custom test configurations and renderers

For more information, see:
- [Technical Documentation](../technical/implementation_guide.md)
- [API Reference](../api/complete_reference.md)
- [Advanced Topics](../advanced/custom_models.md)