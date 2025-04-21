# Claude Instructions

## Project Overview
DeepBridge is a framework for evaluating, testing, and enhancing machine learning models with focuses on robustness, uncertainty quantification, resilience, hyperparameter optimization, and synthetic data generation.

## Common Commands
- Run linting: `flake8`
- Run type checking: `mypy .`
- Run tests: `pytest`

## Important Directories
- `core/experiment/`: Contains experiment management and execution
- `validation/wrappers/`: Contains test suites for different model properties
- `templates/`: HTML templates for report generation
- `synthetic/`: Synthetic data generation capabilities
- `metrics/`: Evaluation metrics for different model types
- `distillation/`: Model distillation and optimization techniques

## Coding Style
- Follow PEP 8 conventions
- Use type hints
- Write docstrings in Google style format
- Keep methods focused and single-purpose
- Use descriptive variable names