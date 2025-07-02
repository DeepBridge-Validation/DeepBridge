# Installation Guide

## Requirements

DeepBridge requires Python 3.10-3.12 and works on Linux, macOS, and Windows.

## Installation Methods

### Using pip (Recommended)

The easiest way to install DeepBridge is using pip:

```bash
pip install deepbridge
```

### Install from Source

For the latest development version:

```bash
git clone https://github.com/DeepBridge-Validation/DeepBridge.git
cd DeepBridge
pip install -e .
```

### Using Poetry

If you prefer using Poetry for dependency management:

```bash
git clone https://github.com/DeepBridge-Validation/DeepBridge.git
cd DeepBridge
poetry install
```

## Dependencies

DeepBridge automatically installs all required dependencies:

### Core Dependencies
- **numpy** (≥2.2.3): Numerical computations
- **pandas** (≥2.2.3): Data manipulation
- **scikit-learn** (≥1.6.1): Machine learning algorithms
- **xgboost** (≥2.1.4): Gradient boosting
- **scipy** (≥1.15.1): Scientific computing

### Visualization
- **matplotlib** (≥3.10.0): Static plotting
- **seaborn** (≥0.13.2): Statistical visualizations
- **plotly** (≥6.0.0): Interactive visualizations

### Additional Libraries
- **optuna** (≥4.2.1): Hyperparameter optimization
- **jinja2** (≥3.1.5): Template engine for reports
- **typer**: Command-line interface
- **rich**: Beautiful terminal output
- **dask**: Distributed computing (optional)

## Verification

Verify your installation:

```bash
# Check version
python -c "import deepbridge; print(deepbridge.__version__)"

# Run basic test
python -c "from deepbridge import DBDataset; print('Installation successful!')"

# Check CLI
deepbridge --version
```

## Environment Setup

### Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv deepbridge-env

# Activate (Linux/macOS)
source deepbridge-env/bin/activate

# Activate (Windows)
deepbridge-env\Scripts\activate

# Install DeepBridge
pip install deepbridge
```

### Conda Environment

```bash
# Create conda environment
conda create -n deepbridge python=3.11

# Activate environment
conda activate deepbridge

# Install DeepBridge
pip install deepbridge
```

## Platform-Specific Notes

### macOS
- Ensure Xcode Command Line Tools are installed:
  ```bash
  xcode-select --install
  ```

### Windows
- Visual C++ Build Tools may be required for some dependencies
- Use PowerShell or Git Bash for better command-line experience

### Linux
- Most distributions work out of the box
- For Ubuntu/Debian, you might need:
  ```bash
  sudo apt-get update
  sudo apt-get install python3-dev python3-pip
  ```

## Optional Components

### For Development

```bash
# Install development dependencies
pip install deepbridge[dev]

# Or using poetry
poetry install --with dev
```

### For Documentation

```bash
# Install documentation dependencies
pip install deepbridge[docs]

# Or using poetry
poetry install --with docs
```

## Troubleshooting Installation

### Common Issues

1. **Import Error**: Ensure you're using Python 3.10-3.12
   ```bash
   python --version
   ```

2. **Permission Denied**: Use `--user` flag or virtual environment
   ```bash
   pip install --user deepbridge
   ```

3. **Dependency Conflicts**: Create a clean virtual environment
   ```bash
   python -m venv clean-env
   source clean-env/bin/activate  # or clean-env\Scripts\activate on Windows
   pip install deepbridge
   ```

4. **Memory Issues**: For large datasets, install Dask
   ```bash
   pip install dask[distributed]
   ```

## Next Steps

After installation, proceed to:
- [Quick Start Guide](quickstart.md) - Get started with basic usage
- [Basic Examples](basic_examples.md) - See DeepBridge in action
- [CLI Usage](../guides/cli.md) - Learn about command-line tools

## Getting Help

If you encounter issues:
1. Check the [FAQ](../resources/faq.md)
2. Visit [Troubleshooting Guide](../resources/troubleshooting.md)
3. Open an issue on [GitHub](https://github.com/DeepBridge-Validation/DeepBridge/issues)