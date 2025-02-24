# DeepBridge Quick Start Guide

## Introduction

DeepBridge is a powerful Python library for machine learning model validation and distillation. This guide will help you quickly get started, covering different scenarios and use cases.

## Installation

Install DeepBridge using pip:

```bash
pip install deepbridge
```

## Use Cases and Examples


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DeepBridge-Validation/DeepBridge/blob/master/examples/quickstart.ipynb)



### 1. Basic Model Validation

#### When to Use
- For binary classification projects
- When you need to compare different models
- To validate model performance before deployment

```python
import pandas as pd
import json
from deepbridge.model_validation import ModelValidation
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def robust_model_validation(data, target_column):
    """
    Validates models with comprehensive error handling.
    
    Args:
        data (pd.DataFrame): Dataset
        target_column (str): Name of the target column
    
    Returns:
        dict: Model validation results
    
    Raises:
        ValueError: For invalid data
        Exception: For validation process failures
    """
    try:
        # Initial validations
        if data is None or len(data) == 0:
            raise ValueError("Dataset cannot be empty")
        
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        # Prepare data
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create validation experiment
        experiment = ModelValidation(
            experiment_name="customer_churn_validation"
        )
        
        # Add data to experiment
        experiment.add_data(
            X_train=X_train, 
            y_train=y_train, 
            X_test=X_test, 
            y_test=y_test
        )
        
        # Define models for comparison
        models = {
            'Random Forest': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
            ]),
            'Logistic Regression': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(max_iter=1000, random_state=42))
            ])
        }
        
        # Model results
        results = {}
        
        # Train and evaluate each model
        for name, model in models.items():
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Add model to experiment
                experiment.add_model(model, name)
                
                # Evaluate model
                results[name] = {
                    'train_accuracy': model.score(X_train, y_train),
                    'test_accuracy': model.score(X_test, y_test)
                }
            
            except Exception as model_error:
                print(f"Error processing model {name}: {model_error}")
                results[name] = {'error': str(model_error)}
        
        return results
    
    except Exception as e:
        print(f"Validation process error: {e}")
        raise

# Usage example
try:
    # Load data (replace with your dataset)
    data = pd.read_csv('churn_data.csv')
    
    # Validate models
    validation_results = robust_model_validation(data, 'churn')
    
    # Print results
    for model, metrics in validation_results.items():
        print(f"\nModel: {model}")
        print(json.dumps(metrics, indent=2))

except Exception as e:
    print(f"General error: {e}")
```

### 2. Model Distillation

#### When to Use
- To reduce computational complexity
- When you need lighter models for production
- To improve inference time

```python
from deepbridge.model_distiller import ModelDistiller
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def robust_model_distillation(
    X_train, 
    y_train, 
    teacher_model, 
    student_model_type='gbm'
):
    """
    Performs knowledge distillation with error handling.
    
    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        teacher_model: High-complexity model
        student_model_type (str): Student model type
    
    Returns:
        ModelDistiller: Distilled model
    
    Raises:
        ValueError: For invalid inputs
        Exception: For distillation process failures
    """
    try:
        # Initial validations
        if X_train is None or y_train is None:
            raise ValueError("Training data cannot be None")
        
        if len(X_train) != len(y_train):
            raise ValueError("Features and labels must have equal length")
        
        # Get teacher model probabilities
        try:
            teacher_probas = teacher_model.predict_proba(X_train)
        except Exception as e:
            raise ValueError(f"Failed to get probabilities from teacher model: {e}")
        
        # Create distiller
        try:
            distiller = ModelDistiller(
                model_type=student_model_type,
                model_params={
                    'n_estimators': 50,
                    'learning_rate': 0.1
                }
            )
            
            # Train distilled model
            distiller.fit(
                X=X_train, 
                probas=teacher_probas,
                test_size=0.2,
                verbose=True
            )
            
            return distiller
        
        except Exception as distillation_error:
            raise ValueError(f"Error in distillation process: {distillation_error}")
    
    except Exception as e:
        print(f"Model distillation error: {e}")
        raise

# Usage example
try:
    # Prepare data (replace with your data)
    X_train = np.random.rand(1000, 10)
    y_train = np.random.randint(0, 2, 1000)
    
    # Train teacher model (e.g., a complex model)
    teacher_model = RandomForestClassifier(n_estimators=200)
    teacher_model.fit(X_train, y_train)
    
    # Perform distillation
    distilled_model = robust_model_distillation(
        X_train, 
        y_train, 
        teacher_model, 
        student_model_type='gbm'
    )
    
    # Compare performance
    teacher_metrics = teacher_model.score(X_train, y_train)
    distilled_metrics = distilled_model.evaluate(X_train, y_train)
    
    print("Teacher Model Metrics:", teacher_metrics)
    print("Distilled Model Metrics:", distilled_metrics)

except Exception as e:
    print(f"General process error: {e}")
```

### 3. Complex Use Case: Experiment Management

#### When to Use
- Machine learning projects with multiple models
- Algorithm comparative research
- Systematic experiment tracking

```python
import os
from deepbridge.auto_distiller import AutoDistiller
from deepbridge.db_data import DBDataset
import pandas as pd

def advanced_experiment(
    data_path, 
    target_column, 
    output_path='experiment_results'
):
    """
    Performs advanced distillation experiment with multiple models.
    
    Args:
        data_path (str): Path to the dataset
        target_column (str): Name of the target column
        output_path (str): Directory to save results
    
    Returns:
        dict: Consolidated experiment results
    """
    try:
        # Validate data path
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Load data
        data = pd.read_csv(data_path)
        
        # Create dataset
        dataset = DBDataset(
            data=data,
            target_column=target_column,
            synthetic=True  # Generate synthetic data for augmentation
        )
        
        # Configure auto distiller
        auto_distiller = AutoDistiller(
            dataset=dataset,
            output_dir=output_path,
            n_trials=20,  # More iterations for optimization
            random_state=42
        )
        
        # Customize model configuration
        auto_distiller.customize_config(
            model_types=['gbm', 'xgb', 'random_forest'],
            temperatures=[0.5, 1.0, 2.0],
            alphas=[0.3, 0.5, 0.7]
        )
        
        # Run experiments
        results = auto_distiller.run()
        
        # Generate report
        report = auto_distiller.generate_report()
        
        # Save best model
        best_model_path = auto_distiller.save_best_model()
        
        return {
            'results': results,
            'report': report,
            'best_model': best_model_path
        }
    
    except Exception as e:
        print(f"Advanced experiment error: {e}")
        raise

# Usage example
try:
    experiment_results = advanced_experiment(
        data_path='project_data.csv', 
        target_column='classification_target'
    )
    
    print("Experiment completed successfully!")
    print("Report Summary:")
    print(experiment_results['report'])

except Exception as e:
    print(f"Experiment failed: {e}")
```

## Next Steps

- Explore the [Advanced Guides](advanced/index.md)
- Consult the [API Reference](api/index.md)
- Join the [DeepBridge Community](community.md)

## Final Tips

1. Always validate your data before creating experiments
2. Use robust error handling
3. Document each step of your machine learning process
4. Experiment with different model configurations
5. Compare multiple models before making your final choice

## Troubleshooting

- **Installation Error**: Check Python version compatibility
- **Performance Issues**: Adjust hyperparameters and dataset size
- **Model Errors**: Verify data quality and preprocessing