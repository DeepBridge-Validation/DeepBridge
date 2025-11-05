"""
Base classes for domain models (Phase 3 Sprint 10).

Provides common configuration and validators for all report domain models.
"""

from typing import Any
from pydantic import BaseModel, field_validator


class ReportBaseModel(BaseModel):
    """
    Base class for all report domain models.

    Features:
    - Validates data on assignment
    - Allows arbitrary types (numpy arrays, etc.)
    - Coerces None to field defaults
    - Rounds floats to 4 decimal places
    - Uses snake_case for JSON serialization

    Example:
        class MyReportData(ReportBaseModel):
            model_name: str
            accuracy: float = 0.0

        report = MyReportData(model_name="Test", accuracy=None)
        print(report.accuracy)  # 0.0 (None coerced to default)
    """

    model_config = {
        'validate_assignment': True,  # Validate on attribute assignment
        'arbitrary_types_allowed': True,  # Allow numpy arrays, etc.
        'populate_by_name': True,  # Allow both alias and field name
        'str_strip_whitespace': True,  # Strip whitespace from strings
    }

    @field_validator('*', mode='before')
    @classmethod
    def coerce_none_to_default(cls, v: Any, info) -> Any:
        """
        Coerce None to field default value.

        This prevents None from propagating through the system and
        eliminates the need for .get(key, default) patterns.

        Example:
            >>> data = MyModel(score=None)  # score has default=0.0
            >>> data.score  # 0.0, not None
        """
        # Get field info
        field = cls.model_fields.get(info.field_name)

        # If value is None and field has a default, use the default
        if v is None and field and field.default is not None:
            return field.default

        return v

    @field_validator('*', mode='after')
    @classmethod
    def round_floats(cls, v: Any) -> Any:
        """
        Round float values to 4 decimal places for consistency.

        Example:
            >>> data = MyModel(score=0.123456789)
            >>> data.score  # 0.1235
        """
        if isinstance(v, float):
            return round(v, 4)
        return v

    def model_dump_json_safe(self) -> dict:
        """
        Dump model to dict with JSON-safe values.

        Handles:
        - NaN/Inf values
        - Numpy types
        - Datetime objects

        Returns:
            Dictionary safe for JSON serialization
        """
        return self.model_dump(mode='json')

    def __repr__(self) -> str:
        """Concise representation for debugging."""
        fields = ', '.join(
            f"{k}={repr(v)[:50]}"
            for k, v in self.model_dump().items()
            if v is not None
        )
        return f"{self.__class__.__name__}({fields})"
