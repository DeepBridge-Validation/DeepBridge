"""
Custom Jinja2 filters for report templates.

This module provides custom filters used in report templates for
formatting numbers, dates, and other values.
"""

import json
from datetime import datetime
from typing import Any, Optional
from jinja2 import Environment


def format_number(value: float, decimals: int = 2) -> str:
    """Format number with specified decimal places.

    Args:
        value: Number to format
        decimals: Number of decimal places

    Returns:
        Formatted number string

    Example:
        >>> format_number(3.14159, 2)
        '3.14'
    """
    if value is None:
        return "N/A"

    try:
        return f"{float(value):.{decimals}f}"
    except (ValueError, TypeError):
        return str(value)


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format number as percentage.

    Args:
        value: Number to format (0-1 range)
        decimals: Number of decimal places

    Returns:
        Formatted percentage string

    Example:
        >>> format_percentage(0.8547)
        '85.5%'
    """
    if value is None:
        return "N/A"

    try:
        percentage = float(value) * 100
        return f"{percentage:.{decimals}f}%"
    except (ValueError, TypeError):
        return str(value)


def format_datetime(value: datetime, format_string: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format datetime object.

    Args:
        value: Datetime to format
        format_string: strftime format string

    Returns:
        Formatted datetime string

    Example:
        >>> format_datetime(datetime.now())
        '2026-02-10 12:30:45'
    """
    if value is None:
        return "N/A"

    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return value

    if isinstance(value, datetime):
        return value.strftime(format_string)

    return str(value)


def to_json(value: Any, indent: Optional[int] = None) -> str:
    """Convert value to JSON string.

    Args:
        value: Value to convert
        indent: Indentation level (None for compact)

    Returns:
        JSON string

    Example:
        >>> to_json({'key': 'value'}, indent=2)
        '{\\n  "key": "value"\\n}'
    """
    try:
        return json.dumps(value, indent=indent, default=str)
    except (TypeError, ValueError) as e:
        return f"Error serializing to JSON: {e}"


def format_metric_name(name: str) -> str:
    """Format metric name for display.

    Converts snake_case to Title Case.

    Args:
        name: Metric name

    Returns:
        Formatted name

    Example:
        >>> format_metric_name('mean_absolute_error')
        'Mean Absolute Error'
    """
    if not name:
        return ""

    return name.replace('_', ' ').title()


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted size string

    Example:
        >>> format_file_size(1536)
        '1.5 KB'
    """
    if size_bytes is None or size_bytes < 0:
        return "N/A"

    units = ['B', 'KB', 'MB', 'GB', 'TB']
    size = float(size_bytes)
    unit_index = 0

    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1

    if unit_index == 0:
        return f"{int(size)} {units[unit_index]}"
    else:
        return f"{size:.1f} {units[unit_index]}"


def truncate_text(text: str, max_length: int = 50, suffix: str = "...") -> str:
    """Truncate text to maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to append if truncated

    Returns:
        Truncated text

    Example:
        >>> truncate_text("This is a very long text", 10)
        'This is a...'
    """
    if not text or len(text) <= max_length:
        return text

    return text[:max_length - len(suffix)] + suffix


def colorize_metric(value: float, threshold: float = 0.5, inverse: bool = False) -> str:
    """Return CSS class for colorizing metric based on threshold.

    Args:
        value: Metric value
        threshold: Threshold for good/bad
        inverse: If True, lower is better

    Returns:
        CSS class name

    Example:
        >>> colorize_metric(0.8, 0.5)
        'metric-good'
        >>> colorize_metric(0.3, 0.5)
        'metric-bad'
    """
    if value is None:
        return 'metric-neutral'

    try:
        value = float(value)
        threshold = float(threshold)

        if inverse:
            return 'metric-good' if value <= threshold else 'metric-bad'
        else:
            return 'metric-good' if value >= threshold else 'metric-bad'
    except (ValueError, TypeError):
        return 'metric-neutral'


def pluralize(count: int, singular: str, plural: Optional[str] = None) -> str:
    """Return singular or plural form based on count.

    Args:
        count: Number of items
        singular: Singular form
        plural: Plural form (if None, adds 's' to singular)

    Returns:
        Appropriate form

    Example:
        >>> pluralize(1, 'model')
        'model'
        >>> pluralize(2, 'model')
        'models'
        >>> pluralize(2, 'matrix', 'matrices')
        'matrices'
    """
    if count == 1:
        return singular

    if plural is None:
        return f"{singular}s"

    return plural


def safe_divide(numerator: float, denominator: float, default: Any = 0) -> float:
    """Safely divide two numbers, handling division by zero.

    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division fails

    Returns:
        Result of division or default

    Example:
        >>> safe_divide(10, 2)
        5.0
        >>> safe_divide(10, 0, default='N/A')
        'N/A'
    """
    try:
        if denominator == 0:
            return default
        return float(numerator) / float(denominator)
    except (ValueError, TypeError, ZeroDivisionError):
        return default


def register_custom_filters(env: Environment) -> None:
    """Register all custom filters with Jinja2 environment.

    Args:
        env: Jinja2 Environment instance

    Example:
        >>> from jinja2 import Environment
        >>> env = Environment()
        >>> register_custom_filters(env)
        >>> # Now filters are available in templates
    """
    filters = {
        'format_number': format_number,
        'format_percentage': format_percentage,
        'format_datetime': format_datetime,
        'to_json': to_json,
        'format_metric_name': format_metric_name,
        'format_file_size': format_file_size,
        'truncate_text': truncate_text,
        'colorize_metric': colorize_metric,
        'pluralize': pluralize,
        'safe_divide': safe_divide,
    }

    env.filters.update(filters)
