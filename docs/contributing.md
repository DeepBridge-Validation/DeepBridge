# Contributing to DeepBridge

Thank you for your interest in contributing to DeepBridge! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct:
- Be respectful and inclusive
- Exercise consideration and empathy
- Focus on what is best for the community
- Give and gracefully accept constructive feedback

## Getting Started

1. **Fork the Repository**
   ```bash
   git clone https://github.com/DeepBridge-Validation/DeepBridge.git
   cd deepbridge
   ```

2. **Set Up Development Environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows

   # Install dependencies
   pip install -r requirements-dev.txt
   ```

3. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Guidelines

### Code Style

We follow PEP 8 guidelines. Use the provided pre-commit hooks:

```bash
pre-commit install
```

### Testing

All new code should include tests:

```bash
# Run tests
pytest tests/

# Check coverage
pytest --cov=deepbridge tests/
```

### Documentation

- Add docstrings to all public functions and classes
- Update relevant documentation files
- Include examples when appropriate

### Commit Messages

Follow the conventional commits specification:

```
feat: add new feature X
fix: correct bug in Y
docs: update installation instructions
test: add tests for feature Z
```

## Pull Request Process

1. **Update Documentation**
   - Add or update docstrings
   - Update README if needed
   - Add to CHANGELOG.md

2. **Run Tests**
   ```bash
   # Run full test suite
   pytest
   
   # Run linting
   flake8 deepbridge
   ```

3. **Create Pull Request**
   - Provide clear description
   - Link related issues
   - Include any necessary screenshots

4. **Code Review**
   - Address review comments
   - Keep discussions focused and professional
   - Update PR as needed

## Development Setup

### Required Tools

- Python 3.12+
- Git
- virtualenv or conda

### Optional Tools

- Docker for container testing
- make for build automation
- pre-commit for code quality

### IDE Configuration

VS Code settings:
```json
{
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black"
}
```

## Project Structure

```
deepbridge/
├── __init__.py
├── auto
│   ├── __init__.py
│   ├── config.py
│   ├── experiment_runner.py
│   ├── metrics.py
│   ├── reporting.py
│   └── visualization.py
├── auto_distiller.py
├── cli.py
├── db_data.py
├── distillation
│   ├── __init__.py
│   └── classification
│       ├── __init__.py
│       ├── __pycache__
│       │   ├── __init__.cpython-312.pyc
│       │   ├── knowledge_distillation.cpython-312.pyc
│       │   └── model_registry.cpython-312.pyc
│       ├── ensambledistillation.py
│       ├── knowledge_distillation.py
│       ├── model_registry.py
│       ├── pruning.py
│       ├── quantization.py
│       └── temperature_scaling.py
├── experiment.py
├── metrics
│   ├── __init__.py
│   └── classification.py
├── processing
│   ├── __init__.py
│   ├── data_validator.py
│   ├── feature_manager.py
│   ├── model_handler.py
│   └── probability_manager.py
├── results
│   ├── __init__.py
│   ├── dataset_formatter.py
│   └── html.py
└── visualizer
    ├── __init__.py
    └── distribution_visualizer.py
```

## Common Tasks

### Adding New Features

1. Create feature branch
2. Implement feature
3. Add tests
4. Update documentation
5. Submit PR

### Fixing Bugs

1. Create bug fix branch
2. Add test to reproduce bug
3. Fix bug
4. Verify fix
5. Submit PR

### Updating Documentation

1. Make changes in docs/
2. Build documentation locally
3. Preview changes
4. Submit PR

## Getting Help

- Create an issue for questions
- Join our community discussions
- Check existing documentation

## Release Process

1. Update version in setup.py
2. Update CHANGELOG.md
3. Create release branch
4. Run tests and checks
5. Create GitHub release
6. Deploy to PyPI

Thank you for contributing to DeepBridge!