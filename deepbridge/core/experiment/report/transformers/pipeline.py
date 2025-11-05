"""
Transform pipeline for report data processing.

Provides modular pipeline for validation, transformation, and enrichment
of report data (Phase 2 Sprint 7-8).

Example Usage:
    >>> from .pipeline import TransformPipeline, Validator, Transformer, Enricher
    >>>
    >>> # Create custom stages
    >>> class MyValidator(Validator):
    ...     def validate(self, data):
    ...         errors = []
    ...         if 'required_field' not in data:
    ...             errors.append("Missing required_field")
    ...         return errors
    >>>
    >>> class MyTransformer(Transformer):
    ...     def transform(self, data):
    ...         return {'processed': data}
    >>>
    >>> # Build and execute pipeline
    >>> pipeline = (TransformPipeline()
    ...             .add_stage(MyValidator())
    ...             .add_stage(MyTransformer()))
    >>>
    >>> result = pipeline.execute(raw_data)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import logging

logger = logging.getLogger("deepbridge.reports")


# ==================================================================================
# Pipeline Stages - Abstract Base Classes
# ==================================================================================

class PipelineStage(ABC):
    """
    Base class for all pipeline stages.

    All stages must implement the process() method which takes data
    and returns transformed data.
    """

    @abstractmethod
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process data and return transformed result.

        Args:
            data: Input data dictionary

        Returns:
            Transformed data dictionary

        Raises:
            Exception: If processing fails
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Validator(PipelineStage):
    """
    Validates input data.

    Subclasses must implement validate() which returns a list of error messages.
    Empty list means validation passed.
    """

    @abstractmethod
    def validate(self, data: Dict[str, Any]) -> List[str]:
        """
        Validate data and return list of errors.

        Args:
            data: Data to validate

        Returns:
            List of error messages (empty if valid)

        Example:
            >>> def validate(self, data):
            ...     errors = []
            ...     if 'model_name' not in data:
            ...         errors.append("Missing model_name")
            ...     return errors
        """
        pass

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate data and return unchanged if valid.

        Raises:
            ValueError: If validation fails
        """
        errors = self.validate(data)
        if errors:
            error_msg = f"Validation errors in {self.__class__.__name__}: {', '.join(errors)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.debug(f"{self.__class__.__name__}: Validation passed")
        return data


class Transformer(PipelineStage):
    """
    Transforms data structure.

    Subclasses must implement transform() which converts data
    from one structure to another.
    """

    @abstractmethod
    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform data structure.

        Args:
            data: Input data

        Returns:
            Transformed data with new structure

        Example:
            >>> def transform(self, data):
            ...     return {
            ...         'model': data['model_name'],
            ...         'scores': self._extract_scores(data)
            ...     }
        """
        pass

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform and return new structure."""
        result = self.transform(data)
        logger.debug(f"{self.__class__.__name__}: Transformation complete")
        return result


class Enricher(PipelineStage):
    """
    Enriches data with derived metrics and calculations.

    Subclasses must implement enrich() which adds computed fields
    to existing data.
    """

    @abstractmethod
    def enrich(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich data with derived metrics.

        Args:
            data: Input data

        Returns:
            Data with additional computed fields

        Example:
            >>> def enrich(self, data):
            ...     data['summary'] = self._calculate_summary(data['metrics'])
            ...     data['quality_score'] = self._calculate_quality(data)
            ...     return data
        """
        pass

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich and return enhanced data."""
        result = self.enrich(data)
        logger.debug(f"{self.__class__.__name__}: Enrichment complete")
        return result


# ==================================================================================
# Transform Pipeline
# ==================================================================================

class TransformPipeline:
    """
    Pipeline for sequential data transformation.

    Executes stages in order: Validation → Transformation → Enrichment

    Features:
    - Fluent interface for adding stages
    - Detailed logging of each stage
    - Proper error propagation
    - Extensible via custom stages

    Example:
        >>> pipeline = TransformPipeline()
        >>> pipeline.add_stage(UncertaintyValidator())
        >>> pipeline.add_stage(UncertaintyTransformer())
        >>> pipeline.add_stage(UncertaintyEnricher())
        >>>
        >>> result = pipeline.execute(raw_experiment_data)

    Or with fluent interface:
        >>> result = (TransformPipeline()
        ...           .add_stage(UncertaintyValidator())
        ...           .add_stage(UncertaintyTransformer())
        ...           .add_stage(UncertaintyEnricher())
        ...           .execute(raw_data))
    """

    def __init__(self):
        """Initialize empty pipeline."""
        self.stages: List[PipelineStage] = []
        logger.debug("TransformPipeline initialized")

    def add_stage(self, stage: PipelineStage) -> 'TransformPipeline':
        """
        Add a stage to the pipeline.

        Args:
            stage: PipelineStage instance (Validator, Transformer, or Enricher)

        Returns:
            Self for fluent interface

        Example:
            >>> pipeline.add_stage(MyValidator()).add_stage(MyTransformer())
        """
        if not isinstance(stage, PipelineStage):
            raise TypeError(f"Stage must be PipelineStage, got {type(stage)}")

        self.stages.append(stage)
        logger.debug(f"Added stage: {stage}")
        return self  # Fluent interface

    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute all pipeline stages in order.

        Args:
            data: Input data dictionary

        Returns:
            Fully processed data

        Raises:
            ValueError: If validation fails
            Exception: If any stage fails

        Example:
            >>> result = pipeline.execute({'model_name': 'MyModel', ...})
        """
        if not self.stages:
            logger.warning("Pipeline has no stages, returning data unchanged")
            return data

        logger.info(f"Executing pipeline with {len(self.stages)} stages")

        result = data
        for i, stage in enumerate(self.stages, 1):
            logger.debug(f"Stage {i}/{len(self.stages)}: {stage.__class__.__name__}")

            try:
                result = stage.process(result)
            except Exception as e:
                logger.error(f"Pipeline failed at stage {i} ({stage.__class__.__name__}): {e}")
                raise

        logger.info("Pipeline execution complete")
        return result

    def clear(self) -> 'TransformPipeline':
        """
        Remove all stages from pipeline.

        Returns:
            Self for fluent interface
        """
        self.stages.clear()
        logger.debug("Pipeline cleared")
        return self

    def __len__(self) -> int:
        """Return number of stages."""
        return len(self.stages)

    def __repr__(self) -> str:
        stage_names = [s.__class__.__name__ for s in self.stages]
        return f"TransformPipeline({stage_names})"


# ==================================================================================
# Example Implementation (for documentation purposes)
# ==================================================================================

class ExampleValidator(Validator):
    """Example validator showing the pattern."""

    def validate(self, data: Dict[str, Any]) -> List[str]:
        """Validate that data has required fields."""
        errors = []
        required_fields = ['model_name', 'test_results']

        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")

        return errors


class ExampleTransformer(Transformer):
    """Example transformer showing the pattern."""

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform to standardized structure."""
        return {
            'model': data.get('model_name', 'Unknown'),
            'results': data.get('test_results', {}),
            'metadata': {
                'source': 'experiment',
                'processed': True
            }
        }


class ExampleEnricher(Enricher):
    """Example enricher showing the pattern."""

    def enrich(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Add computed summary metrics."""
        # Add summary
        data['summary'] = {
            'total_tests': len(data.get('results', {})),
            'model_name': data.get('model', 'Unknown')
        }

        # Add quality score (dummy calculation)
        data['quality_score'] = 0.85

        return data


# ==================================================================================
# Main - Example Usage
# ==================================================================================

if __name__ == "__main__":
    """
    Demonstrate pipeline usage.
    """
    print("=" * 80)
    print("Transform Pipeline Example")
    print("=" * 80)

    # Example data
    example_data = {
        'model_name': 'TestModel',
        'test_results': {
            'accuracy': 0.92,
            'precision': 0.88
        }
    }

    # Create pipeline
    pipeline = (TransformPipeline()
                .add_stage(ExampleValidator())
                .add_stage(ExampleTransformer())
                .add_stage(ExampleEnricher()))

    print(f"\nPipeline: {pipeline}")
    print(f"Stages: {len(pipeline)}")

    # Execute
    print("\nExecuting pipeline...")
    result = pipeline.execute(example_data)

    # Show result
    print("\nResult:")
    import json
    print(json.dumps(result, indent=2))

    print("\n" + "=" * 80)
    print("Pipeline Example Complete")
    print("=" * 80)
