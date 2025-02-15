# Using the DeepBridge CLI

This guide will walk you through using the DeepBridge Command Line Interface (CLI) for managing your machine learning experiments and model distillation tasks.

## Installation

The CLI is automatically installed with DeepBridge:

```bash
pip install deepb
```

## Basic Commands

Check the installation and version:
```bash
deepbridge --version
```

Get help on available commands:
```bash
deepbridge --help
```

## Working with Experiments

### Creating and Managing Experiments

1. Create a new experiment:
```bash
# Basic experiment
deepbridge validation create my_first_experiment

# With custom path
deepbridge validation create churn_prediction --path ./projects/churn
```

2. Add data to your experiment:
```bash
# Add training data with target column
deepbridge validation add-data \
    ./projects/churn \
    data/train.csv \
    --target-column churn

# Add both training and test data
deepbridge validation add-data \
    ./projects/churn \
    data/train.csv \
    --test-data data/test.csv \
    --target-column churn
```

3. Check experiment status:
```bash
# View experiment information
deepbridge validation info ./projects/churn

# Get JSON output for scripting
deepbridge validation info ./projects/churn --format json
```

### Real-World Example: Customer Churn Prediction

```bash
# 1. Create project structure
mkdir -p churn_project/{data,models,results}

# 2. Create experiment
deepbridge validation create churn_model --path ./churn_project

# 3. Add data
deepbridge validation add-data \
    ./churn_project \
    ./churn_project/data/train.csv \
    --test-data ./churn_project/data/test.csv \
    --target-column customer_churned

# 4. Check setup
deepbridge validation info ./churn_project
```

## Model Distillation

### Training Distilled Models

1. Basic model training:
```bash
# Train GBM model
deepbridge distill train gbm \
    original_predictions.csv \
    features.csv \
    --save-path ./models/gbm_v1

# Train XGBoost with custom parameters
deepbridge distill train xgb \
    predictions.csv \
    features.csv \
    --save-path ./models/xgb_v1 \
    --params params.json
```

Example `params.json`:
```json
{
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1
}
```

2. Making predictions:
```bash
# Save predictions to file
deepbridge distill predict \
    ./models/gbm_v1/model.joblib \
    new_data.csv \
    --output predictions.csv

# View predictions directly
deepbridge distill predict \
    ./models/gbm_v1/model.joblib \
    new_data.csv
```

3. Evaluating models:
```bash
# Compare model performances
deepbridge distill evaluate \
    ./models/gbm_v1/model.joblib \
    true_labels.csv \
    original_preds.csv \
    distilled_preds.csv
```

### Real-World Example: Model Optimization

```bash
# 1. Train different model types
for model in gbm xgb mlp; do
    deepbridge distill train $model \
        teacher_predictions.csv \
        features.csv \
        --save-path ./models/${model}_v1
done

# 2. Generate predictions
for model in gbm xgb mlp; do
    deepbridge distill predict \
        ./models/${model}_v1/model.joblib \
        test_data.csv \
        --output ./results/${model}_predictions.csv
done

# 3. Evaluate each model
for model in gbm xgb mlp; do
    deepbridge distill evaluate \
        ./models/${model}_v1/model.joblib \
        true_labels.csv \
        teacher_preds.csv \
        ./results/${model}_predictions.csv \
        --format json > ./results/${model}_evaluation.json
done
```

## Common Workflows

### 1. Experiment Iteration

```bash
# Iterate through model versions
for version in {1..3}; do
    # Create experiment version
    deepbridge validation create \
        experiment_v${version} \
        --path ./experiments/v${version}
    
    # Add data
    deepbridge validation add-data \
        ./experiments/v${version} \
        train_v${version}.csv \
        --test-data test_v${version}.csv \
        --target-column target
    
    # Train distilled model
    deepbridge distill train gbm \
        original_preds_v${version}.csv \
        features_v${version}.csv \
        --save-path ./models/v${version}
done
```

### 2. Model Comparison

```bash
# Create results directory
mkdir -p results

# Train and evaluate multiple models
for model_type in gbm xgb mlp; do
    # Train model
    deepbridge distill train ${model_type} \
        predictions.csv \
        features.csv \
        --save-path ./models/${model_type}
    
    # Generate predictions
    deepbridge distill predict \
        ./models/${model_type}/model.joblib \
        test_features.csv \
        --output ./results/${model_type}_preds.csv
    
    # Evaluate model
    deepbridge distill evaluate \
        ./models/${model_type}/model.joblib \
        true_labels.csv \
        teacher_preds.csv \
        ./results/${model_type}_preds.csv \
        --format json > ./results/${model_type}_eval.json
done
```

## Tips and Best Practices

1. **Organize Your Projects**
   ```bash
   project/
   ├── data/
   │   ├── train.csv
   │   └── test.csv
   ├── models/
   │   ├── gbm/
   │   ├── xgb/
   │   └── mlp/
   └── results/
       ├── predictions/
       └── evaluations/
   ```

2. **Use Meaningful Names**
   ```bash
   # Good
   deepbridge validation create customer_churn_rf_v1
   
   # Not so good
   deepbridge validation create exp1
   ```

3. **Document Your Experiments**
   ```bash
   # Save experiment info
   deepbridge validation info ./experiments/exp1 \
       --format json > experiment_metadata.json
   ```

4. **Automate Common Tasks**
   Create shell scripts for repeated workflows:
   ```bash
   #!/bin/bash
   # train_models.sh
   
   MODEL_TYPES="gbm xgb mlp"
   
   for model in $MODEL_TYPES; do
       echo "Training $model model..."
       deepbridge distill train $model \
           data/predictions.csv \
           data/features.csv \
           --save-path models/$model
   done
   ```

## Troubleshooting

Common issues and solutions:

1. **File Not Found**
   ```bash
   # Check file paths
   ls -l data/train.csv
   
   # Use absolute paths if needed
   deepbridge validation add-data \
       $(pwd)/experiment \
       $(pwd)/data/train.csv
   ```

2. **Invalid Data Format**
   ```bash
   # Check CSV format
   head -n 5 data/train.csv
   
   # Verify column names
   deepbridge validation info ./experiment
   ```

3. **Model Training Failures**
   ```bash
   # Enable debug logging
   export DEEPBRIDGE_LOG_LEVEL=DEBUG
   deepbridge distill train gbm predictions.csv features.csv
   ```

## Next Steps

- Check the [API Reference](../api/cli.md) for detailed command documentation
- Learn about [Model Validation](validation.md)
- Explore [Model Distillation](distillation.md)