# DeepBridge CLI User Guide

## Table of Contents
1. [Installation](#installation)
2. [Basic Usage](#basic-usage)
3. [Command Reference](#command-reference)
4. [Advanced Usage Patterns](#advanced-usage-patterns)
5. [Environment Configuration](#environment-configuration)
6. [Troubleshooting](#troubleshooting)
7. [Best Practices](#best-practices)

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
```

## Command Reference

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

# View experiment information
deepbridge validation info \
    /path/to/experiment \
    [--format json|table]
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

# Evaluate distilled model performance
deepbridge distill evaluate \
    /path/to/model.pkl \
    true_labels.csv \
    original_predictions.csv \
    distilled_predictions.csv \
    [--format json|table]
```

## Advanced Usage Patterns

### Complex Experiment Workflows

```bash
#!/bin/bash
# Advanced experiment automation script

# Set up experiment directories
mkdir -p experiments/{v1,v2,v3}

# Iterate through multiple configurations
MODEL_TYPES=("gbm" "xgb" "random_forest")
TEMPERATURES=(0.5 1.0 2.0)
ALPHAS=(0.3 0.5 0.7)

# Loop through configurations
for model in "${MODEL_TYPES[@]}"; do
    for temp in "${TEMPERATURES[@]}"; do
        for alpha in "${ALPHAS[@]}"; do
            # Create unique experiment name
            exp_name="experiment_${model}_t${temp}_a${alpha}"
            
            # Run distillation
            deepbridge distill train "$model" \
                teacher_predictions.csv \
                features.csv \
                --save-path "experiments/${exp_name}" \
                --temperature "$temp" \
                --alpha "$alpha"
            
            # Evaluate model
            deepbridge distill evaluate \
                "experiments/${exp_name}/model.pkl" \
                true_labels.csv \
                teacher_predictions.csv \
                "experiments/${exp_name}/predictions.csv" \
                --format json > "experiments/${exp_name}/metrics.json"
        done
    done
done
```

### Parallel Experiment Execution

```python
# parallel_experiments.py
import subprocess
import multiprocessing
from itertools import product

def run_experiment(params):
    """
    Run a single experiment configuration
    
    Args:
        params (tuple): Experiment configuration parameters
    """
    model, temp, alpha = params
    exp_name = f"experiment_{model}_t{temp}_a{alpha}"
    
    try:
        # Construct CLI command
        cmd = [
            "deepbridge", "distill", "train", model,
            "teacher_predictions.csv",
            "features.csv",
            "--save-path", f"experiments/{exp_name}",
            "--temperature", str(temp),
            "--alpha", str(alpha)
        ]
        
        # Run command
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True
        )
        
        # Log results
        with open(f"experiments/{exp_name}/log.txt", "w") as log_file:
            log_file.write(result.stdout)
            log_file.write(result.stderr)
        
        return exp_name
    except Exception as e:
        print(f"Error in experiment {exp_name}: {e}")
        return None

def main():
    # Define experiment configurations
    model_types = ["gbm", "xgb", "random_forest"]
    temperatures = [0.5, 1.0, 2.0]
    alphas = [0.3, 0.5, 0.7]
    
    # Generate all possible configurations
    experiments = list(product(model_types, temperatures, alphas))
    
    # Use multiprocessing to run experiments in parallel
    with multiprocessing.Pool() as pool:
        completed_experiments = pool.map(run_experiment, experiments)
    
    print("Completed experiments:", completed_experiments)

if __name__ == "__main__":
    main()
```

## Environment Configuration

### Virtual Environment Setup

```bash
# Create virtual environment
python3 -m venv deepbridge_env

# Activate virtual environment
# On Unix/macOS
source deepbridge_env/bin/activate
# On Windows
deepbridge_env\Scripts\activate

# Install DeepBridge
pip install deepbridge

# Create requirements file
pip freeze > requirements.txt
```

### Configuration File

Create a `deepbridge.yaml` in your project root:

```yaml
# DeepBridge Configuration
project:
  name: my_ml_project
  version: 1.0.0

experiments:
  default_path: ./experiments
  random_seed: 42

distillation:
  default_model_type: gbm
  temperatures: [0.5, 1.0, 2.0]
  alphas: [0.3, 0.5, 0.7]

logging:
  level: INFO
  path: ./logs
```

## Troubleshooting

### Common Issues and Solutions

1. **Installation Problems**
   - Ensure Python version compatibility (3.8+)
   - Update pip: `python -m pip install --upgrade pip`
   - Check system dependencies

2. **Command Execution Errors**

   ```bash
   # Verbose error logging
   DEEPBRIDGE_DEBUG=1 deepbridge distill train ...
   ```

3. **Dependency Conflicts**

   ```bash
   # Create isolated environment
   pipx install deepbridge

   # Or use explicit version
   pip install deepbridge==X.Y.Z
   ```

### Debugging Commands

```bash
# Show detailed version information
deepbridge --version --verbose

# Validate installation
deepbridge doctor

# Check system compatibility
deepbridge system-check
```

## Best Practices

1. **Environment Management**
   - Always use virtual environments
   - Pin dependencies
   - Use consistent Python versions

2. **Experiment Organization**
   - Use descriptive experiment names
   - Store metadata with experiments
   - Version control your configurations

3. **Security**
   - Avoid committing sensitive data
   - Use environment variables for credentials
   - Implement access controls

4. **Performance**
   - Monitor resource usage
   - Use parallel processing for large experiments
   - Optimize model and data preprocessing

## Extending CLI Functionality

### Custom CLI Plugins

```python
# custom_plugin.py
import typer

# Create a custom CLI extension
custom_app = typer.Typer()

@custom_app.command()
def analyze(
    experiment_path: str,
    output_path: str = None
):
    """
    Custom analysis command for DeepBridge experiments
    """
    # Implement your custom analysis logic
    print(f"Analyzing experiment: {experiment_path}")

# Integrate with main DeepBridge CLI
from deepbridge.cli import app
app.add_typer(custom_app, name="custom")
```

## Community and Support

- **Documentation**: [DeepBridge Docs](https://deepbridge.readthedocs.io/)
- **Issue Tracker**: [GitHub Issues](https://github.com/deepbridge/deepbridge/issues)
- **Community Chat**: [Slack Channel](https://deepbridge-community.slack.com)

## Conclusion

The DeepBridge CLI provides a powerful, flexible interface for machine learning experiment management. By following these guidelines, you can streamline your model validation and distillation workflows.