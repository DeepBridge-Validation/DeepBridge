# Command Line Interface (CLI) Reference

DeepBridge provides a comprehensive command-line interface for managing experiments, model validation, and distillation tasks.

## Global Options

```bash
deepbridge [OPTIONS] COMMAND [ARGS]...
```

Global options available for all commands:

- `--version, -v`: Show the version and exit
- `--help`: Show help message and exit

## Command Groups

The CLI is organized into two main command groups:

- `validation`: Commands for model validation experiments
- `distill`: Commands for model distillation

## Validation Commands

### Create Experiment

Creates a new validation experiment.

```bash
deepbridge validation create NAME [OPTIONS]
```

**Arguments:**
- `NAME`: Name of the experiment

**Options:**
- `--path, -p TEXT`: Path to save experiment files
  - Default: `./experiments/<NAME>`
  - Optional

**Example:**
```bash
# Create with default path
deepbridge validation create my_experiment

# Create with custom path
deepbridge validation create my_experiment -p /path/to/experiments
```

### Add Data

Adds training and test data to an existing experiment.

```bash
deepbridge validation add-data EXPERIMENT_PATH TRAIN_DATA [OPTIONS]
```

**Arguments:**
- `EXPERIMENT_PATH`: Path to experiment directory
- `TRAIN_DATA`: Path to training data CSV file

**Options:**
- `--test-data, -t TEXT`: Path to test data CSV file
- `--target-column, -y TEXT`: Name of target column [required]

**Example:**
```bash
# Add training and test data
deepbridge validation add-data \
    ./experiments/my_experiment \
    train.csv \
    -t test.csv \
    -y target
```

### Get Info

Retrieves information about an experiment.

```bash
deepbridge validation info EXPERIMENT_PATH [OPTIONS]
```

**Arguments:**
- `EXPERIMENT_PATH`: Path to experiment directory

**Options:**
- `--format, -f TEXT`: Output format (table or json)
  - Default: "table"
  - Choices: ["table", "json"]

**Example:**
```bash
# Get info in table format
deepbridge validation info ./experiments/my_experiment

# Get info in JSON format
deepbridge validation info ./experiments/my_experiment -f json
```

## Distillation Commands

### Train Model

Trains a distilled model using data from an original model.

```bash
deepbridge distill train MODEL_TYPE PREDICTIONS FEATURES [OPTIONS]
```

**Arguments:**
- `MODEL_TYPE`: Type of model to use
  - Choices: ["gbm", "xgb", "mlp"]
- `PREDICTIONS`: Path to original model predictions CSV
- `FEATURES`: Path to features data CSV

**Options:**
- `--save-path, -s TEXT`: Path to save the model
- `--params, -p TEXT`: JSON file with model parameters
- `--test-size, -t FLOAT`: Test set size for validation
  - Default: 0.2
  - Range: 0.0 < x < 1.0

**Example:**
```bash
# Basic training
deepbridge distill train gbm predictions.csv features.csv -s ./models

# Training with custom parameters
deepbridge distill train xgb \
    predictions.csv \
    features.csv \
    -s ./models \
    -p params.json \
    -t 0.3
```

### Make Predictions

Makes predictions using a trained distilled model.

```bash
deepbridge distill predict MODEL_PATH INPUT_DATA [OPTIONS]
```

**Arguments:**
- `MODEL_PATH`: Path to saved model
- `INPUT_DATA`: Path to input data CSV

**Options:**
- `--output, -o TEXT`: Path to save predictions
  - If not provided, prints to console

**Example:**
```bash
# Save predictions to file
deepbridge distill predict \
    ./models/model.joblib \
    new_data.csv \
    -o predictions.csv

# Print predictions to console
deepbridge distill predict ./models/model.joblib new_data.csv
```

### Evaluate Model

Evaluates performance of a distilled model.

```bash
deepbridge distill evaluate MODEL_PATH TRUE_LABELS ORIGINAL_PREDS DISTILLED_PREDS [OPTIONS]
```

**Arguments:**
- `MODEL_PATH`: Path to saved model
- `TRUE_LABELS`: Path to true labels CSV
- `ORIGINAL_PREDS`: Path to original model predictions CSV
- `DISTILLED_PREDS`: Path to distilled model predictions CSV

**Options:**
- `--format, -f TEXT`: Output format (table or json)
  - Default: "table"
  - Choices: ["table", "json"]

**Example:**
```bash
# Evaluate with table output
deepbridge distill evaluate \
    ./models/model.joblib \
    true_labels.csv \
    original_preds.csv \
    distilled_preds.csv

# Evaluate with JSON output
deepbridge distill evaluate \
    ./models/model.joblib \
    true_labels.csv \
    original_preds.csv \
    distilled_preds.csv \
    -f json
```

## Error Handling

The CLI provides informative error messages when:
- Required arguments are missing
- Files don't exist or are inaccessible
- Data format is incorrect
- Model training fails
- Predictions cannot be generated

Example error messages:
```bash
Error: File 'data.csv' not found
Error: Invalid model type. Choose from: gbm, xgb, mlp
Error: Target column 'target' not found in data
```

## Environment Variables

The CLI respects the following environment variables:
- `DEEPBRIDGE_HOME`: Base directory for experiments and models
- `DEEPBRIDGE_LOG_LEVEL`: Logging level (INFO, DEBUG, etc.)

Example:
```bash
export DEEPBRIDGE_HOME=/path/to/experiments
export DEEPBRIDGE_LOG_LEVEL=DEBUG
```

## Best Practices

1. **Organization**
   ```bash
   # Use descriptive experiment names
   deepbridge validation create customer_churn_v1

   # Organize models in subdirectories
   deepbridge distill train gbm preds.csv features.csv -s ./models/churn/v1
   ```

2. **Data Management**
   ```bash
   # Use consistent file naming
   deepbridge validation add-data \
       ./experiments/exp1 \
       data_2024_02_train.csv \
       -t data_2024_02_test.csv
   ```

3. **Model Evaluation**
   ```bash
   # Save evaluation results
   deepbridge distill evaluate \
       ./models/model.joblib \
       labels.csv preds_orig.csv preds_dist.csv \
       -f json > evaluation_results.json
   ```

## See Also

- [Model Validation Guide](../guides/validation.md)
- [Model Distillation Guide](../guides/distillation.md)
- [API Reference](../api/model_validation.md)