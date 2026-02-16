# Centralized Configuration System

This document explains DeepBridge's configuration system for test parameters and standardization.

## Overview

DeepBridge's testing framework uses standardized configurations across various test types (uncertainty, robustness, resilience, and hyperparameter tuning). These configurations are defined as `quick`, `medium`, or `full` test suites with varying levels of thoroughness.

Previously, configuration parameters were duplicated across different modules. The new centralized configuration system addresses this by:

1. Centralizing all configuration parameters in one location
2. Providing standardized access methods
3. Reducing duplication and maintenance issues
4. Adding validation for test types and configuration levels

## Implementation

### Core Module: `parameter_standards.py`

The centralized configuration system is implemented in `deepbridge/core/experiment/parameter_standards.py` which:

1. Defines standard parameter names as constants
2. Provides enums for test types and configuration levels
3. Centralizes configuration templates for all test types
4. Offers utility functions to validate and access configurations

### Key Components

#### Enums

```python
class TestType(Enum):
    """Enum for standardized test types"""
    ROBUSTNESS = "robustness"
    UNCERTAINTY = "uncertainty"
    RESILIENCE = "resilience"
    HYPERPARAMETERS = "hyperparameters"

class ConfigName(Enum):
    """Enum for standardized configuration names"""
    QUICK = "quick"
    MEDIUM = "medium"
    FULL = "full"
```

#### Centralized Configuration Templates

```python
# Configuration templates for test types
ROBUSTNESS_CONFIGS = {
    ConfigName.QUICK.value: {...},
    ConfigName.MEDIUM.value: {...},
    ConfigName.FULL.value: {...}
}

# Similar structures for UNCERTAINTY_CONFIGS, RESILIENCE_CONFIGS, and HYPERPARAMETER_CONFIGS
```

#### Master Configuration Dictionary

```python
TEST_CONFIGS = {
    TestType.ROBUSTNESS.value: ROBUSTNESS_CONFIGS,
    TestType.UNCERTAINTY.value: UNCERTAINTY_CONFIGS,
    TestType.RESILIENCE.value: RESILIENCE_CONFIGS,
    TestType.HYPERPARAMETERS.value: HYPERPARAMETER_CONFIGS
}
```

#### Utility Functions

- `get_test_types()`: Returns all valid test types
- `get_config_names()`: Returns all valid configuration names
- `is_valid_test_type(test_type)`: Validates a test type
- `is_valid_config_name(config_name)`: Validates a configuration name
- `get_test_config(test_type, config_name)`: Gets configuration options for a specific test type and level

## Usage

### In Test Runner

```python
from deepbridge.core.experiment.parameter_standards import (
    get_test_config, TestType, ConfigName
)

# Get configuration for a test type
config = get_test_config(TestType.UNCERTAINTY.value, ConfigName.MEDIUM.value)

# Access specific parameters
alpha_levels = config[0]['params']['alpha']  # For uncertainty tests
```

### In Test Suites

```python
from deepbridge.core.experiment.parameter_standards import (
    get_test_config, TestType, ConfigName
)

# Load configurations from central system
def _get_config_templates(self):
    return {
        config_name: get_test_config(TestType.UNCERTAINTY.value, config_name)
        for config_name in [ConfigName.QUICK.value, ConfigName.MEDIUM.value, ConfigName.FULL.value]
    }
```

### In Managers

```python
from deepbridge.core.experiment.parameter_standards import (
    is_valid_config_name, ConfigName
)

# Validate configuration name
if not is_valid_config_name(config_name):
    self.log(f"Warning: Invalid configuration name '{config_name}'. Using 'quick' instead.")
    config_name = ConfigName.QUICK.value
```

## Benefits

1. **Consistency**: Ensures all components use the same configuration parameters
2. **Maintainability**: Single location for parameter updates
3. **Extensibility**: Easy to add new test types or configuration levels
4. **Validation**: Built-in validation for test types and configuration names
5. **Documentation**: Centralized documentation of available configurations

## Extending the System

To add a new test type or configuration level:

1. Add the new type to the appropriate enum in `parameter_standards.py`
2. Create a configuration template for the new type
3. Add it to the master `TEST_CONFIGS` dictionary

## Future Enhancements

- Support for custom configuration profiles
- Versioning of configuration templates
- Integration with a configuration file system
- UI for configuration management