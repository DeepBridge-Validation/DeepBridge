# JSON Formatter Utility

## Overview

The `JsonFormatter` class provides utilities for safely formatting and sanitizing JSON data for use in JavaScript contexts, particularly in HTML reports. It handles common issues like trailing commas that can cause JavaScript syntax errors.

## Features

- Removes trailing commas from JSON objects and arrays
- Replaces NaN and Infinity values with null
- Validates and formats JSON strings or Python data structures
- Provides HTML embedding for JSON data in script tags

## Usage

### Basic Sanitization

```python
from core.experiment.report.utils.json_formatter import JsonFormatter

# Sanitize a JSON string with trailing commas
json_str = '{"name": "test", "value": 123,}'
sanitized = JsonFormatter.sanitize_json_string(json_str)
# Result: '{"name": "test", "value": 123}'
```

### Converting Python Data to JSON

```python
# Format Python data for JavaScript
data = {"name": "test", "values": [1, 2, 3], "special": float('nan')}
formatted = JsonFormatter.format_for_javascript(data)
# Result: '{"name": "test", "values": [1, 2, 3], "special": null}'
```

### Embedding in HTML

```python
# Generate a script tag with JSON data
html = JsonFormatter.embed_in_html(data, "reportData")
# Result: A script tag with the JSON data assigned to reportData variable
```

### In Renderers

The formatter is already integrated with the report rendering system. Custom renderers can use it:

```python
def render_template(self, data):
    # Format data as safe JSON
    json_data = JsonFormatter.format_for_javascript(data)
    
    # Include in template context
    context = {
        'data_json': json_data
    }
    
    return self.template_engine.render('template.html', context)
```

## Why Use This Formatter?

JavaScript is more lenient than JSON in what it accepts. For example, JavaScript allows:
- Trailing commas in objects and arrays
- Unquoted property names
- Comments
- Special values like NaN and Infinity

These features are not valid in JSON and can cause errors when embedding data in HTML reports. This formatter ensures compatibility by:

1. Cleaning up invalid JSON syntax that would cause errors
2. Converting special values to null
3. Ensuring that the output is both valid JSON and valid JavaScript

## Integration

The formatter is automatically used by the base renderer class and has been integrated with:
- RobustnessRenderer
- BaseRenderer 
- JavaScriptSyntaxFixer

No additional configuration is required to benefit from it in report generation.