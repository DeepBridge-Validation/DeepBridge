# Command Line Interface (CLI) Reference

## Overview

DeepBridge provides a powerful command-line interface (CLI) that allows you to run experiments, generate reports, and perform various operations without writing code. This document provides a complete reference for all available CLI commands and options.

## Global Options

The following options apply to all DeepBridge CLI commands:

| Option | Description |
| ------ | ----------- |
| `--verbose, -v` | Enable verbose output |
| `--config PATH` | Path to configuration file |
| `--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}` | Set logging level |
| `--version` | Show version and exit |
| `--help, -h` | Show help message and exit |

## Commands

### `deepbridge validate`

Validate a model using various testing methods.

```bash
deepbridge validate --data-path data.csv --model-path model.pkl --output results/
```

#### Options

| Option | Description |
| ------ | ----------- |
| `--data-path PATH` | Path to the dataset file |
| `--target-column NAME` | Name of the target column |
| `--model-path PATH` | Path to the model file |
| `--output PATH` | Directory to save validation results |
| `--robustness` | Run robustness tests |
| `--uncertainty` | Run uncertainty tests |
| `--resilience` | Run resilience tests |
| `--test-size FLOAT` | Test set size (default: 0.2) |
| `--cv-folds INT` | Number of cross-validation folds (default: 5) |
| `--processors [standard,custom]` | Preprocessing methods to use |

### `deepbridge distill`

Distill knowledge from a teacher model to a student model.

```bash
deepbridge distill --teacher-path teacher.pkl --student-config config.json --data-path data.csv --output student.pkl
```

#### Options

| Option | Description |
| ------ | ----------- |
| `--teacher-path PATH` | Path to the teacher model file |
| `--student-config PATH` | Configuration file for the student model |
| `--data-path PATH` | Path to the dataset file |
| `--target-column NAME` | Name of the target column |
| `--output PATH` | Path to save the distilled student model |
| `--temperature FLOAT` | Temperature parameter for distillation (default: 1.0) |
| `--alpha FLOAT` | Weight for distillation loss (default: 0.5) |
| `--epochs INT` | Number of training epochs (default: 100) |
| `--batch-size INT` | Training batch size (default: 32) |
| `--auto` | Use AutoDistiller to automatically tune parameters |

### `deepbridge generate`

Generate synthetic data from an existing dataset.

```bash
deepbridge generate --data-path data.csv --output synthetic.csv --method gaussian-copula --samples 1000
```

#### Options

| Option | Description |
| ------ | ----------- |
| `--data-path PATH` | Path to the source dataset file |
| `--output PATH` | Path to save the generated data |
| `--method {ultra-light,gaussian-copula,ctgan}` | Generation method to use |
| `--samples INT` | Number of samples to generate |
| `--categorical NAMES` | Comma-separated list of categorical columns |
| `--continuous NAMES` | Comma-separated list of continuous columns |
| `--evaluate` | Evaluate quality of generated data |
| `--save-metrics PATH` | Path to save evaluation metrics |

### `deepbridge report`

Generate a comprehensive report from experiment results.

```bash
deepbridge report --results-dir results/ --output report.html --format html
```

#### Options

| Option | Description |
| ------ | ----------- |
| `--results-dir PATH` | Directory containing experiment results |
| `--output PATH` | Path to save the generated report |
| `--format {html,pdf,markdown}` | Report format (default: html) |
| `--title TEXT` | Report title |
| `--include-plots` | Include visualization plots |
| `--include-tables` | Include detailed metric tables |
| `--template PATH` | Custom report template path |

### `deepbridge optimize`

Optimize model performance through pruning, quantization, or other techniques.

```bash
deepbridge optimize --model-path model.pkl --data-path data.csv --method pruning --output optimized.pkl
```

#### Options

| Option | Description |
| ------ | ----------- |
| `--model-path PATH` | Path to the model file |
| `--data-path PATH` | Path to the dataset file |
| `--target-column NAME` | Name of the target column |
| `--output PATH` | Path to save the optimized model |
| `--method {pruning,quantization,both}` | Optimization method |
| `--pruning-rate FLOAT` | Rate of parameters to prune (default: 0.3) |
| `--quantization-bits INT` | Bit precision for quantization (default: 8) |
| `--evaluate` | Evaluate performance after optimization |

### `deepbridge experiment`

Run a complete experiment from a configuration file.

```bash
deepbridge experiment --config experiment.yaml --output experiment_results/
```

#### Options

| Option | Description |
| ------ | ----------- |
| `--config PATH` | Path to experiment configuration file |
| `--output PATH` | Directory to save experiment results |
| `--resume` | Resume a previously started experiment |
| `--override KEY=VALUE` | Override configuration parameters |

## Configuration Files

DeepBridge CLI commands can use configuration files to specify complex parameters. These files can be in YAML or JSON format.

### Example Validation Configuration

```yaml
data:
  path: data/training.csv
  target_column: target
  test_size: 0.2
  cv_folds: 5

model:
  path: models/random_forest.pkl
  type: classifier

validation:
  robustness: true
  uncertainty: true
  resilience: false
  methods:
    - noise_test:
        noise_level: 0.1
    - missing_value_test:
        missing_ratio: 0.2
    - perturbation_test:
        perturbation_type: gaussian

output:
  path: results/validation_output
  format: json
```

### Example Distillation Configuration

```yaml
teacher:
  path: models/complex_model.pkl
  type: classifier

student:
  architecture: mlp
  hidden_layers: [64, 32]
  activation: relu
  dropout: 0.2

data:
  path: data/training.csv
  target_column: target
  validation_split: 0.2

distillation:
  temperature: 2.0
  alpha: 0.7
  epochs: 200
  batch_size: 64
  optimizer:
    name: adam
    learning_rate: 0.001

output:
  path: models/distilled_model.pkl
  save_history: true
```

## Environment Variables

DeepBridge CLI recognizes the following environment variables:

| Variable | Description |
| -------- | ----------- |
| `DEEPBRIDGE_CONFIG` | Default config file path |
| `DEEPBRIDGE_OUTPUT_DIR` | Default output directory |
| `DEEPBRIDGE_LOG_LEVEL` | Default logging level |
| `DEEPBRIDGE_CACHE_DIR` | Directory for caching intermediate results |

## Exit Codes

| Code | Description |
| ---- | ----------- |
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | File not found |
| 4 | Permission denied |
| 5 | Validation error |
| 6 | Model error |

## Examples

### Basic Model Validation

```bash
deepbridge validate --data-path data.csv --model-path random_forest.pkl --robustness --output results/
```

### Cross-Validation with Custom Preprocessing

```bash
deepbridge validate --data-path data.csv --model-path model.pkl --cv-folds 10 --processors standard --output results/
```

### Automated Distillation

```bash
deepbridge distill --teacher-path complex_model.pkl --data-path data.csv --auto --output distilled_model.pkl
```

### Generate Synthetic Data with Quality Evaluation

```bash
deepbridge generate --data-path sensitive_data.csv --method ctgan --samples 5000 --evaluate --output synthetic_data.csv
```

### End-to-End Experiment from Configuration

```bash
deepbridge experiment --config experiments/full_workflow.yaml --output project_results/
```