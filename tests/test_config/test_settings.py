"""
Tests for DistillationConfig settings.

Coverage Target: Cover all configuration management.
"""

import pytest
import os
import tempfile
import shutil
from deepbridge.config.settings import DistillationConfig
from deepbridge.utils.model_registry import ModelType


class TestDistillationConfigInitialization:
    """Tests for DistillationConfig initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        config = DistillationConfig()

        assert config.output_dir == 'distillation_results'
        assert config.test_size == 0.2
        assert config.random_state == 42
        assert config.n_trials == 10
        assert config.validation_split == 0.2
        assert config.verbose is True
        assert config.distillation_method == 'surrogate'

    def test_init_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        config = DistillationConfig(
            output_dir='custom_output',
            test_size=0.3,
            random_state=123,
            n_trials=20,
            validation_split=0.25,
            verbose=False,
            distillation_method='knowledge_distillation'
        )

        assert config.output_dir == 'custom_output'
        assert config.test_size == 0.3
        assert config.random_state == 123
        assert config.n_trials == 20
        assert config.validation_split == 0.25
        assert config.verbose is False
        assert config.distillation_method == 'knowledge_distillation'

    def test_init_with_hpm_parameters(self):
        """Test initialization with HPM-specific parameters."""
        config = DistillationConfig(
            use_hpm=True,
            max_configs=32,
            parallel_workers=4,
            use_cache=False,
            use_progressive=False,
            use_multi_teacher=False,
            use_adaptive_temperature=False,
            cache_memory_gb=4.0
        )

        assert config.use_hpm is True
        assert config.max_configs == 32
        assert config.parallel_workers == 4
        assert config.use_cache is False
        assert config.use_progressive is False
        assert config.use_multi_teacher is False
        assert config.use_adaptive_temperature is False
        assert config.cache_memory_gb == 4.0

    def test_default_model_types(self):
        """Test that default model types are set correctly."""
        config = DistillationConfig()

        assert len(config.model_types) == 4
        assert ModelType.LOGISTIC_REGRESSION in config.model_types
        assert ModelType.DECISION_TREE in config.model_types
        assert ModelType.GBM in config.model_types
        assert ModelType.XGB in config.model_types

    def test_default_temperatures(self):
        """Test that default temperatures are set correctly."""
        config = DistillationConfig()

        assert config.temperatures == [0.5, 1.0, 2.0, 3.0]

    def test_default_alphas(self):
        """Test that default alphas are set correctly."""
        config = DistillationConfig()

        assert config.alphas == [0.3, 0.5, 0.7, 0.9]

    def test_output_directory_creation(self):
        """Test that output directory is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test_output')
            config = DistillationConfig(output_dir=output_path)

            assert os.path.exists(output_path)


class TestDistillationConfigCustomize:
    """Tests for customization methods."""

    def test_customize_model_types(self):
        """Test customizing model types."""
        config = DistillationConfig()
        new_model_types = [ModelType.LOGISTIC_REGRESSION, ModelType.DECISION_TREE]

        config.customize(model_types=new_model_types)

        assert config.model_types == new_model_types
        assert len(config.model_types) == 2

    def test_customize_temperatures(self):
        """Test customizing temperatures."""
        config = DistillationConfig()
        new_temperatures = [1.0, 2.0]

        config.customize(temperatures=new_temperatures)

        assert config.temperatures == new_temperatures

    def test_customize_alphas(self):
        """Test customizing alphas."""
        config = DistillationConfig()
        new_alphas = [0.5, 0.7]

        config.customize(alphas=new_alphas)

        assert config.alphas == new_alphas

    def test_customize_distillation_method(self):
        """Test customizing distillation method."""
        config = DistillationConfig()

        config.customize(distillation_method='knowledge_distillation')

        assert config.distillation_method == 'knowledge_distillation'

    def test_customize_multiple_parameters(self):
        """Test customizing multiple parameters at once."""
        config = DistillationConfig()
        new_model_types = [ModelType.GBM]
        new_temperatures = [1.5]
        new_alphas = [0.6]

        config.customize(
            model_types=new_model_types,
            temperatures=new_temperatures,
            alphas=new_alphas,
            distillation_method='hpm'
        )

        assert config.model_types == new_model_types
        assert config.temperatures == new_temperatures
        assert config.alphas == new_alphas
        assert config.distillation_method == 'hpm'

    def test_customize_with_none_values_keeps_defaults(self):
        """Test that None values don't override existing settings."""
        config = DistillationConfig()
        original_model_types = config.model_types.copy()

        config.customize(model_types=None, temperatures=[1.0])

        assert config.model_types == original_model_types
        assert config.temperatures == [1.0]


class TestDistillationConfigHelperMethods:
    """Tests for helper methods."""

    def test_get_total_configurations(self):
        """Test calculating total configurations."""
        config = DistillationConfig()

        # Default: 4 model_types * 4 temperatures * 4 alphas = 64
        total = config.get_total_configurations()
        assert total == 64

    def test_get_total_configurations_after_customize(self):
        """Test total configurations after customization."""
        config = DistillationConfig()
        config.customize(
            model_types=[ModelType.LOGISTIC_REGRESSION],
            temperatures=[1.0, 2.0],
            alphas=[0.5]
        )

        # 1 model_type * 2 temperatures * 1 alpha = 2
        total = config.get_total_configurations()
        assert total == 2

    def test_log_info_when_verbose(self, capsys):
        """Test that log_info prints when verbose is True."""
        config = DistillationConfig(verbose=True)

        config.log_info("Test message")

        captured = capsys.readouterr()
        assert "Test message" in captured.out

    def test_log_info_when_not_verbose(self, capsys):
        """Test that log_info doesn't print when verbose is False."""
        config = DistillationConfig(verbose=False)

        config.log_info("Test message")

        captured = capsys.readouterr()
        assert "Test message" not in captured.out


class TestDistillationConfigEdgeCases:
    """Tests for edge cases."""

    def test_empty_model_types_list(self):
        """Test with empty model types after customization."""
        config = DistillationConfig()
        config.customize(model_types=[])

        total = config.get_total_configurations()
        assert total == 0

    def test_single_values_for_all_parameters(self):
        """Test with single values for all parameters."""
        config = DistillationConfig()
        config.customize(
            model_types=[ModelType.LOGISTIC_REGRESSION],
            temperatures=[1.0],
            alphas=[0.5]
        )

        total = config.get_total_configurations()
        assert total == 1

    def test_large_configuration_space(self):
        """Test with large configuration space."""
        config = DistillationConfig()
        config.customize(
            temperatures=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            alphas=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        )

        # 4 model_types * 8 temperatures * 9 alphas = 288
        total = config.get_total_configurations()
        assert total == 288


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
