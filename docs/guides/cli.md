# DeepBridge CLI User Guide

## Table of Contents
1. [Installation](#installation)
2. [Basic Usage](#basic-usage)
3. [Command Reference](#command-reference)
4. [Advanced Usage Patterns](#advanced-usage-patterns)
5. [Best Practices](#best-practices)

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Installation Methods

#### Using pip
```bash
# Install latest stable version
pip install deepbridge

# Install with optional dependencies
pip install deepbridge[full]
```

#### Using pipx (Recommended)
```bash
# Install pipx
python3 -m pip install --user pipx
python3 -m pipx ensurepath

# Install DeepBridge
pipx install deepbridge
```

## Basic Usage

### Quick Start

```bash
# Show available commands
deepbridge --help

# Check version
deepbridge --version

# Get documentation for a specific command
deepbridge validation --help
```

## Command Reference

### Experiment Commands

The DeepBridge CLI provides access to the component-based architecture through convenient commands:

```bash
# Create new experiment with specific test types
deepbridge experiment create \
    --name my_experiment \
    --path ./experiments \
    --data-file data.csv \
    --target-column target \
    --tests robustness uncertainty \
    --experiment-type binary_classification

# Run tests with specific configuration
deepbridge experiment run-tests \
    --path ./experiments/my_experiment \
    --config medium

# Generate report from experiment
deepbridge experiment report \
    --path ./experiments/my_experiment \
    --output ./reports/experiment_report.html
```

### Model Validation Commands

```bash
# Create a new validation experiment
deepbridge validation create experiment_name \
    --path /path/to/experiments \
    [--description "Optional experiment description"]

# Add data to an experiment
deepbridge validation add-data \
    /path/to/experiment \
    train_data.csv \
    --target-column target_variable \
    [--test-data test_data.csv]

# Run validation tests
deepbridge validation test \
    --path /path/to/experiment \
    --tests robustness uncertainty \
    --config medium
```

### Model Distillation Commands

```bash
# Train a distilled model
deepbridge distill train model_type \
    teacher_predictions.csv \
    features.csv \
    [--save-path /path/to/save] \
    [--params params.json] \
    [--test-size 0.2] \
    [--temperature 1.0] \
    [--alpha 0.5]

# Make predictions with a distilled model
deepbridge distill predict \
    /path/to/model.pkl \
    input_data.csv \
    [--output predictions.csv]

# Compare teacher and student models
deepbridge distill compare \
    --teacher-model /path/to/teacher.pkl \
    --student-model /path/to/student.pkl \
    --test-data test_data.csv \
    --output comparison_report.html
```

### Robustness Testing Commands

```bash
# Run robustness tests on a model
deepbridge robustness test \
    --model-path /path/to/model.pkl \
    --data-file data.csv \
    --target-column target \
    --config full \
    --output robustness_report.html

# Compare multiple models for robustness
deepbridge robustness compare \
    --models-dir /path/to/models \
    --data-file data.csv \
    --target-column target \
    --output comparison.html
```

### Visualization Commands

```bash
# Generate robustness visualizations
deepbridge visualize robustness \
    --results-file /path/to/results.json \
    --output-dir /path/to/visualizations

# Create comparison plots
deepbridge visualize comparison \
    --results-dir /path/to/results \
    --output comparison_plot.html \
    --plot-type bar
```

## Advanced Usage Patterns

### Experiment Pipeline

Create an end-to-end experiment script:

```bash
#!/bin/bash
# Complete experiment pipeline

# Create experiment directory
mkdir -p experiments/my_experiment

# Create experiment
deepbridge experiment create \
    --name my_experiment \
    --path ./experiments/my_experiment \
    --data-file data.csv \
    --target-column target \
    --tests robustness uncertainty resilience \
    --experiment-type binary_classification

# Train distilled model
deepbridge distill train gbm \
    --experiment-path ./experiments/my_experiment \
    --temperature 2.0 \
    --alpha 0.5 \
    --n-trials 20

# Run comprehensive tests
deepbridge experiment run-tests \
    --path ./experiments/my_experiment \
    --config full

# Generate report
deepbridge experiment report \
    --path ./experiments/my_experiment \
    --output ./reports/comprehensive_report.html

echo "Experiment pipeline completed!"
```

### Complex Model Comparison

Compare multiple model types with different configurations:

```bash
#!/bin/bash
# Advanced model comparison script

MODEL_TYPES=("gbm" "xgb" "random_forest" "logistic_regression")
TEMPS=(1.0 2.0 5.0)
ALPHAS=(0.3 0.5 0.7)
CONFIG="medium"

# Create experiment
deepbridge experiment create \
    --name model_comparison \
    --path ./experiments/comparison \
    --data-file data.csv \
    --target-column target \
    --test robustness \
    --experiment-type binary_classification

# Train all model combinations
for model in "${MODEL_TYPES[@]}"; do
    for temp in "${TEMPS[@]}"; do
        for alpha in "${ALPHAS[@]}"; do
            MODEL_NAME="${model}_t${temp}_a${alpha}"
            echo "Training model: $MODEL_NAME"
            
            deepbridge distill train "$model" \
                --experiment-path ./experiments/comparison \
                --model-name "$MODEL_NAME" \
                --temperature "$temp" \
                --alpha "$alpha" \
                --n-trials 10
        done
    done
done

# Run robustness tests on all models
deepbridge robustness compare \
    --experiment-path ./experiments/comparison \
    --config "$CONFIG" \
    --output ./reports/model_comparison.html

echo "Model comparison completed!"
```

## Best Practices

### 1. Consistent Naming Convention

Use consistent naming conventions for experiments and outputs:

```bash
# Naming pattern: <project>_<model-type>_<date>
EXPERIMENT_NAME="customer_churn_gbm_$(date +%Y%m%d)"

deepbridge experiment create \
    --name "$EXPERIMENT_NAME" \
    --path "./experiments/$EXPERIMENT_NAME" \
    --data-file data.csv \
    --target-column target
```

### 2. Configuration Files

Use JSON configuration files for reproducible experiments:

```json
{
  "experiment": {
    "name": "production_model_v2",
    "type": "binary_classification",
    "tests": ["robustness", "uncertainty", "resilience"],
    "random_state": 42
  },
  "data": {
    "file": "data/processed/training_data.csv",
    "target_column": "churn",
    "test_size": 0.2
  },
  "distillation": {
    "model_type": "gbm",
    "temperature": 2.0,
    "alpha": 0.5,
    "n_trials": 50
  }
}
```

Then use it with the CLI:

```bash
deepbridge experiment create --config experiment_config.json
```

### 3. Output Management

Organize outputs in a structured way:

```bash
# Create directory structure
mkdir -p experiments/{data,models,reports,visualizations}

# Run experiment with structured output
deepbridge experiment create \
    --name my_experiment \
    --path ./experiments/models/my_experiment \
    --data-file ./experiments/data/processed_data.csv \
    --target-column target

# Generate report to specific location
deepbridge experiment report \
    --path ./experiments/models/my_experiment \
    --output ./experiments/reports/my_experiment_report.html
```

### 4. Automation and CI/CD Integration

Integrate DeepBridge with CI/CD pipelines:

```bash
#!/bin/bash
# CI/CD script for model validation

# Set environment variables
export EXPERIMENT_NAME="model_validation_$(date +%Y%m%d)"
export DATA_PATH="./data/latest.csv"
export REPORT_PATH="./reports/$EXPERIMENT_NAME.html"

# Run validation
deepbridge validation create "$EXPERIMENT_NAME" \
    --data-file "$DATA_PATH" \
    --target-column target \
    --tests robustness

# Run tests
deepbridge validation test \
    --path "$EXPERIMENT_NAME" \
    --config medium

# Generate report
deepbridge experiment report \
    --path "$EXPERIMENT_NAME" \
    --output "$REPORT_PATH"

# Check for performance thresholds
PERFORMANCE=$(jq '.overall_score' "$EXPERIMENT_NAME/results.json")
THRESHOLD=0.85

if (( $(echo "$PERFORMANCE < $THRESHOLD" | bc -l) )); then
    echo "Model validation failed: performance $PERFORMANCE below threshold $THRESHOLD"
    exit 1
else
    echo "Model validation passed: performance $PERFORMANCE above threshold $THRESHOLD"
    exit 0
fi
```

## CLI Architecture

The DeepBridge CLI follows the same component-based architecture as the core library:

```
┌───────────────────┐
│      CLI App      │
└─────────┬─────────┘
          │
          ▼
┌───────────────────────────────────────────────────────┐
│                                                       │
│  ┌───────────┐  ┌────────────┐  ┌──────────────┐      │
│  │Experiment │  │Validation  │  │Distillation  │      │
│  │Commands   │  │Commands    │  │Commands      │      │
│  └───────────┘  └────────────┘  └──────────────┘      │
│                                                       │
│  ┌───────────┐  ┌────────────┐  ┌──────────────┐      │
│  │Robustness │  │Visualization│  │Utility      │      │
│  │Commands   │  │Commands     │  │Commands     │      │
│  └───────────┘  └────────────┘  └──────────────┘      │
│                                                       │
└───────────────────────────────────────────────────────┘
```

Each command group interacts with the corresponding components in the core library, providing a seamless command-line interface to the entire framework.

## Conclusion

The DeepBridge CLI provides a powerful interface to the component-based architecture of the framework. By using these commands, you can automate complex model validation and distillation workflows, integrate them into your development pipelines, and ensure reproducible machine learning experiments.