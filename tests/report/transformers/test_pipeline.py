"""
Tests for Transform Pipeline.

Tests the modular transformation pipeline created in Phase 2 Sprint 7-8.
"""

import pytest

from deepbridge.core.experiment.report.transformers.pipeline import (
    Enricher,
    PipelineStage,
    Transformer,
    TransformPipeline,
    Validator,
)


# Test Implementations
class MockValidator(Validator):
    """Mock validator for testing."""
    def validate(self, data):
        errors = []
        if 'required_field' not in data:
            errors.append("Missing required_field")
        return errors


class FailingValidator(Validator):
    """Validator that always fails."""
    def validate(self, data):
        return ["Always fails"]


class MockTransformer(Transformer):
    """Mock transformer for testing."""
    def transform(self, data):
        return {
            'transformed': True,
            'original_keys': list(data.keys())
        }


class MockEnricher(Enricher):
    """Mock enricher for testing."""
    def enrich(self, data):
        data['enriched'] = True
        data['summary'] = {'total_keys': len(data)}
        return data


class TrackingStage(PipelineStage):
    """Stage that tracks execution order."""
    def __init__(self, name, execution_list):
        self.name = name
        self.execution_list = execution_list

    def process(self, data):
        self.execution_list.append(self.name)
        return data


# ==================================================================================
# Tests
# ==================================================================================

class TestPipelineStage:
    """Tests for PipelineStage base class."""

    def test_stage_repr(self):
        """Test string representation of stages."""
        validator = MockValidator()
        assert 'MockValidator' in repr(validator)


class TestValidator:
    """Tests for Validator base class."""

    def test_validator_success(self):
        """Test validator with valid data."""
        validator = MockValidator()
        data = {'required_field': 'value'}

        result = validator.process(data)

        assert result == data  # Data unchanged

    def test_validator_failure(self):
        """Test validator with invalid data."""
        validator = MockValidator()
        data = {}  # Missing required_field

        with pytest.raises(ValueError) as exc_info:
            validator.process(data)

        assert 'required_field' in str(exc_info.value)

    def test_failing_validator(self):
        """Test validator that always fails."""
        validator = FailingValidator()

        with pytest.raises(ValueError) as exc_info:
            validator.process({'any': 'data'})

        assert 'Always fails' in str(exc_info.value)


class TestTransformer:
    """Tests for Transformer base class."""

    def test_transformer_transforms_data(self):
        """Test transformer modifies data structure."""
        transformer = MockTransformer()
        data = {'key1': 'value1', 'key2': 'value2'}

        result = transformer.process(data)

        assert result['transformed'] is True
        assert 'original_keys' in result
        assert set(result['original_keys']) == {'key1', 'key2'}


class TestEnricher:
    """Tests for Enricher base class."""

    def test_enricher_adds_fields(self):
        """Test enricher adds derived fields."""
        enricher = MockEnricher()
        data = {'existing': 'data'}

        result = enricher.process(data)

        assert result['enriched'] is True
        assert 'summary' in result
        assert result['summary']['total_keys'] > 0


class TestTransformPipeline:
    """Tests for TransformPipeline."""

    def test_empty_pipeline(self):
        """Test pipeline with no stages."""
        pipeline = TransformPipeline()
        data = {'test': 'data'}

        result = pipeline.execute(data)

        assert result == data

    def test_single_stage_pipeline(self):
        """Test pipeline with one stage."""
        pipeline = TransformPipeline()
        pipeline.add_stage(MockValidator())

        data = {'required_field': 'value'}
        result = pipeline.execute(data)

        assert result == data

    def test_multi_stage_pipeline(self):
        """Test pipeline with multiple stages."""
        pipeline = TransformPipeline()
        pipeline.add_stage(MockValidator())
        pipeline.add_stage(MockTransformer())
        pipeline.add_stage(MockEnricher())

        data = {'required_field': 'value', 'other': 'data'}
        result = pipeline.execute(data)

        assert result['transformed'] is True
        assert result['enriched'] is True
        assert 'summary' in result

    def test_fluent_interface(self):
        """Test fluent interface for adding stages."""
        pipeline = (TransformPipeline()
                    .add_stage(MockValidator())
                    .add_stage(MockTransformer())
                    .add_stage(MockEnricher()))

        assert len(pipeline) == 3

    def test_execution_order(self):
        """Test stages execute in correct order."""
        execution_order = []
        pipeline = TransformPipeline()

        pipeline.add_stage(TrackingStage('first', execution_order))
        pipeline.add_stage(TrackingStage('second', execution_order))
        pipeline.add_stage(TrackingStage('third', execution_order))

        pipeline.execute({})

        assert execution_order == ['first', 'second', 'third']

    def test_pipeline_failure_propagates(self):
        """Test that stage failures propagate."""
        pipeline = TransformPipeline()
        pipeline.add_stage(FailingValidator())  # This will fail

        with pytest.raises(ValueError):
            pipeline.execute({'any': 'data'})

    def test_pipeline_clear(self):
        """Test clearing pipeline stages."""
        pipeline = TransformPipeline()
        pipeline.add_stage(MockValidator())
        pipeline.add_stage(MockTransformer())

        assert len(pipeline) == 2

        pipeline.clear()

        assert len(pipeline) == 0

    def test_pipeline_repr(self):
        """Test string representation of pipeline."""
        pipeline = TransformPipeline()
        pipeline.add_stage(MockValidator())
        pipeline.add_stage(MockTransformer())

        repr_str = repr(pipeline)

        assert 'TransformPipeline' in repr_str
        assert 'MockValidator' in repr_str
        assert 'MockTransformer' in repr_str

    def test_invalid_stage_type(self):
        """Test adding non-PipelineStage raises error."""
        pipeline = TransformPipeline()

        with pytest.raises(TypeError):
            pipeline.add_stage("not a stage")


class TestPipelineIntegration:
    """Integration tests for complete pipelines."""

    def test_complete_data_transformation(self):
        """Test complete transformation pipeline."""
        # Setup pipeline
        pipeline = (TransformPipeline()
                    .add_stage(MockValidator())
                    .add_stage(MockTransformer())
                    .add_stage(MockEnricher()))

        # Input data
        input_data = {
            'required_field': 'present',
            'metric1': 0.92,
            'metric2': 0.88
        }

        # Execute
        result = pipeline.execute(input_data)

        # Verify transformations
        assert result['transformed'] is True
        assert result['enriched'] is True
        assert 'summary' in result
        assert result['summary']['total_keys'] > 0

    def test_validation_prevents_processing(self):
        """Test validation failure prevents downstream processing."""
        execution_order = []

        pipeline = TransformPipeline()
        pipeline.add_stage(FailingValidator())
        pipeline.add_stage(TrackingStage('should_not_run', execution_order))

        with pytest.raises(ValueError):
            pipeline.execute({})

        # Verify downstream stage never executed
        assert 'should_not_run' not in execution_order


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
