"""
JSON utilities for safe serialization in reports.

This module provides utilities for safe JSON serialization,
handling special cases like NaN, Infinity, datetime, and numpy types.

Version: 1.0
Date: 2025-11-05
"""

import json
import math
import datetime
import logging
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)

# Try to import numpy if available
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    logger.debug("NumPy not available, numpy type handling disabled")


class SafeJSONEncoder(json.JSONEncoder):
    """
    JSON encoder that handles special types safely.

    Handles:
    - NaN and Infinity (converted to None)
    - datetime objects (converted to ISO format)
    - numpy types (if numpy is available)
    - Other non-serializable objects (converted to string)

    Example:
        >>> encoder = SafeJSONEncoder()
        >>> json.dumps({'value': float('nan')}, cls=SafeJSONEncoder)
        '{"value": null}'
    """

    def default(self, obj: Any) -> Any:
        """
        Convert non-serializable objects to JSON-compatible types.

        Parameters:
        -----------
        obj : Any
            Object to serialize

        Returns:
        --------
        Any : JSON-compatible representation
        """
        # Handle floats with special values
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None

        # Handle dates and datetimes
        if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
            return obj.isoformat()

        # Handle numpy types if numpy is available
        if HAS_NUMPY:
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                # Check for NaN/Inf in numpy floats
                if np.isnan(obj) or np.isinf(obj):
                    return None
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.bool_):
                return bool(obj)

        # Default: convert to string
        try:
            return str(obj)
        except Exception as e:
            logger.warning(f"Could not serialize object of type {type(obj)}: {e}")
            return None


def safe_json_dumps(data: Union[Dict, List, Any], indent: int = None, **kwargs) -> str:
    """
    Safely serialize data to JSON string.

    This function handles special cases that would normally cause json.dumps to fail:
    - NaN and Infinity values (converted to null)
    - datetime objects (converted to ISO format strings)
    - numpy types (if numpy is available)
    - Other non-serializable objects (converted to strings)

    Parameters:
    -----------
    data : Union[Dict, List, Any]
        Data to serialize
    indent : int, optional
        Number of spaces for indentation (None for compact output)
    **kwargs : dict
        Additional arguments passed to json.dumps

    Returns:
    --------
    str : JSON string

    Example:
        >>> data = {'value': float('nan'), 'date': datetime.now()}
        >>> json_str = safe_json_dumps(data, indent=2)
    """
    try:
        # First clean the data to handle NaN/Inf values
        cleaned_data = clean_for_json(data)

        # Then serialize with safe encoder for other special types
        return json.dumps(
            cleaned_data,
            cls=SafeJSONEncoder,
            ensure_ascii=False,
            indent=indent,
            **kwargs
        )
    except Exception as e:
        logger.error(f"Error serializing data to JSON: {str(e)}")
        # Return empty JSON object as fallback
        return "{}"


def safe_json_loads(json_str: str) -> Union[Dict, List, Any]:
    """
    Safely load JSON string.

    Parameters:
    -----------
    json_str : str
        JSON string to parse

    Returns:
    --------
    Union[Dict, List, Any] : Parsed data

    Raises:
    -------
    json.JSONDecodeError : If JSON string is invalid

    Example:
        >>> data = safe_json_loads('{"key": "value"}')
    """
    return json.loads(json_str)


def clean_for_json(data: Any) -> Any:
    """
    Recursively clean data structure for JSON serialization.

    This function walks through nested dictionaries and lists,
    replacing NaN, Infinity, and other problematic values with None.

    Parameters:
    -----------
    data : Any
        Data to clean (can be dict, list, or scalar)

    Returns:
    --------
    Any : Cleaned data structure

    Example:
        >>> data = {'scores': [1.0, float('nan'), 3.0], 'nested': {'value': float('inf')}}
        >>> clean_data = clean_for_json(data)
        >>> clean_data
        {'scores': [1.0, None, 3.0], 'nested': {'value': None}}
    """
    if isinstance(data, dict):
        return {key: clean_for_json(value) for key, value in data.items()}

    elif isinstance(data, list):
        return [clean_for_json(item) for item in data]

    elif isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return None
        return data

    elif HAS_NUMPY and isinstance(data, (np.floating, np.integer)):
        if isinstance(data, np.floating) and (np.isnan(data) or np.isinf(data)):
            return None
        return float(data) if isinstance(data, np.floating) else int(data)

    else:
        return data


def format_for_javascript(data: Union[Dict, List, Any]) -> str:
    """
    Format data for safe embedding in JavaScript code.

    This is an alias for safe_json_dumps that emphasizes the use case
    of embedding JSON in HTML/JavaScript.

    Parameters:
    -----------
    data : Union[Dict, List, Any]
        Data to format

    Returns:
    --------
    str : JSON string safe for JavaScript

    Example:
        >>> data = {'value': float('nan')}
        >>> js_data = format_for_javascript(data)
        >>> print(f"const data = {js_data};")
        const data = {"value": null};
    """
    # Clean data first to handle nested structures
    cleaned_data = clean_for_json(data)

    # Then serialize with safe encoder
    return safe_json_dumps(cleaned_data)


# Convenience function for backwards compatibility with existing code
def json_serializer(obj: Any) -> Any:
    """
    JSON serializer for objects not serializable by default json code.

    This is a simplified version for use with json.dumps(default=...) parameter.

    Parameters:
    -----------
    obj : Any
        Object to serialize

    Returns:
    --------
    Any : Serialized object

    Example:
        >>> json.dumps({'date': datetime.now()}, default=json_serializer)
    """
    encoder = SafeJSONEncoder()
    return encoder.default(obj)


# Example usage and testing
if __name__ == '__main__':
    # Set up logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 80)
    print("JSON Utils Test")
    print("=" * 80)

    # Test 1: NaN and Infinity handling
    print("\n1. Testing NaN and Infinity handling...")
    test_data = {
        'nan_value': float('nan'),
        'inf_value': float('inf'),
        'neg_inf_value': float('-inf'),
        'normal_value': 3.14
    }
    json_str = safe_json_dumps(test_data, indent=2)
    print(f"Input: {test_data}")
    print(f"JSON output:\n{json_str}")

    # Test 2: Datetime handling
    print("\n2. Testing datetime handling...")
    test_data = {
        'timestamp': datetime.datetime(2025, 11, 5, 12, 0, 0),
        'date': datetime.date(2025, 11, 5)
    }
    json_str = safe_json_dumps(test_data, indent=2)
    print(f"JSON output:\n{json_str}")

    # Test 3: Nested structures
    print("\n3. Testing nested structures...")
    test_data = {
        'metrics': {
            'accuracy': 0.95,
            'loss': float('nan')
        },
        'scores': [1.0, 2.0, float('inf'), 4.0]
    }
    json_str = safe_json_dumps(test_data, indent=2)
    print(f"JSON output:\n{json_str}")

    # Test 4: Format for JavaScript
    print("\n4. Testing format_for_javascript...")
    test_data = {'value': float('nan'), 'count': 42}
    js_str = format_for_javascript(test_data)
    print(f"JavaScript ready: const data = {js_str};")

    # Test 5: NumPy types (if available)
    if HAS_NUMPY:
        print("\n5. Testing NumPy types...")
        test_data = {
            'np_int': np.int64(42),
            'np_float': np.float64(3.14),
            'np_nan': np.float64(np.nan),
            'np_array': np.array([1, 2, 3])
        }
        json_str = safe_json_dumps(test_data, indent=2)
        print(f"JSON output:\n{json_str}")
    else:
        print("\n5. NumPy not available, skipping NumPy tests")

    print("\n" + "=" * 80)
    print("Test Complete")
    print("=" * 80)


# ==================================================================================
# Data Preparation for Templates (Phase 2 Sprint 5-6)
# ==================================================================================

def prepare_data_for_template(data: Dict, test_type: str) -> Dict:
    """
    Prepare data for template rendering.

    Combines raw data with JSON-safe serialization, replacing
    DataIntegrationManager (Phase 2 simplification).

    Args:
        data: Raw report data dictionary
        test_type: Type of test ('uncertainty', 'robustness', etc.)

    Returns:
        Dictionary ready for template rendering with:
        - 'data': Original data
        - 'data_json': JSON-safe string for JavaScript
        - 'test_type': Test type identifier

    Example:
        >>> report_data = {'model_name': 'MyModel', 'metrics': {...}}
        >>> template_data = prepare_data_for_template(report_data, 'uncertainty')
        >>> # Use in template: {{ data_json }} for JS access
    """
    return {
        'data': data,
        'data_json': format_for_javascript(data),
        'test_type': test_type
    }
