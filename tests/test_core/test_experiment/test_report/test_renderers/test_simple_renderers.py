"""
Tests for Simple Renderers (Phase 2).

Tests the template method pattern implementation for all simple renderers:
- UncertaintyRendererSimple
- RobustnessRendererSimple
- ResilienceRendererSimple
- FairnessRendererSimple
- HyperparameterRenderer

These renderers were refactored in Phase 2 to inherit from BaseRenderer.
Focus: Verify pattern implementation, not full integration testing.
"""

from pathlib import Path
from unittest.mock import Mock

import pytest

# ==================================================================================
# Fixtures
# ==================================================================================


@pytest.fixture
def mock_template_manager():
    """Create mock TemplateManager."""
    manager = Mock()
    template = Mock()
    manager.load_template = Mock(return_value=template)
    manager.render_template = Mock(return_value='<html>Mock Report</html>')
    manager.find_template = Mock(return_value='mock_template_path')
    return manager


@pytest.fixture
def mock_asset_manager(tmp_path):
    """Create mock AssetManager."""
    manager = Mock()
    manager.get_css_content = Mock(return_value='body { margin: 0; }')
    manager.get_js_content = Mock(return_value="console.log('test');")
    manager.get_logo_base64 = Mock(
        return_value='data:image/png;base64,mocklogo'
    )
    return manager


# ==================================================================================
# Tests for UncertaintyRendererSimple
# ==================================================================================


class TestUncertaintyRendererSimple:
    """Tests for UncertaintyRendererSimple."""

    def test_initialization(self, mock_template_manager, mock_asset_manager):
        """Test renderer initialization."""
        from deepbridge.core.experiment.report.renderers.uncertainty_renderer_simple import (
            UncertaintyRendererSimple,
        )

        renderer = UncertaintyRendererSimple(
            mock_template_manager, mock_asset_manager
        )

        assert renderer.template_manager is mock_template_manager
        assert renderer.asset_manager is mock_asset_manager
        assert hasattr(renderer, 'data_transformer')

    def test_inherits_from_base_renderer(
        self, mock_template_manager, mock_asset_manager
    ):
        """Test that renderer inherits from BaseRenderer."""
        from deepbridge.core.experiment.report.renderers.base_renderer import (
            BaseRenderer,
        )
        from deepbridge.core.experiment.report.renderers.uncertainty_renderer_simple import (
            UncertaintyRendererSimple,
        )

        renderer = UncertaintyRendererSimple(
            mock_template_manager, mock_asset_manager
        )
        assert isinstance(renderer, BaseRenderer)

    def test_has_required_methods(
        self, mock_template_manager, mock_asset_manager
    ):
        """Test that renderer has required methods from BaseRenderer."""
        from deepbridge.core.experiment.report.renderers.uncertainty_renderer_simple import (
            UncertaintyRendererSimple,
        )

        renderer = UncertaintyRendererSimple(
            mock_template_manager, mock_asset_manager
        )

        # Renderer's own method
        assert hasattr(renderer, 'render')
        assert callable(renderer.render)

        # Inherited BaseRenderer methods (Phase 2 pattern)
        assert hasattr(renderer, '_load_template')
        assert hasattr(renderer, '_get_assets')
        assert hasattr(renderer, '_create_base_context')
        assert hasattr(renderer, '_render_template')
        assert hasattr(renderer, '_write_html')


# ==================================================================================
# Tests for RobustnessRendererSimple
# ==================================================================================


class TestRobustnessRendererSimple:
    """Tests for RobustnessRendererSimple."""

    def test_initialization(self, mock_template_manager, mock_asset_manager):
        """Test renderer initialization."""
        from deepbridge.core.experiment.report.renderers.robustness_renderer_simple import (
            RobustnessRendererSimple,
        )

        renderer = RobustnessRendererSimple(
            mock_template_manager, mock_asset_manager
        )

        assert renderer.template_manager is mock_template_manager
        assert renderer.asset_manager is mock_asset_manager
        assert hasattr(renderer, 'data_transformer')

    def test_inherits_from_base_renderer(
        self, mock_template_manager, mock_asset_manager
    ):
        """Test that renderer inherits from BaseRenderer."""
        from deepbridge.core.experiment.report.renderers.base_renderer import (
            BaseRenderer,
        )
        from deepbridge.core.experiment.report.renderers.robustness_renderer_simple import (
            RobustnessRendererSimple,
        )

        renderer = RobustnessRendererSimple(
            mock_template_manager, mock_asset_manager
        )
        assert isinstance(renderer, BaseRenderer)


# ==================================================================================
# Tests for ResilienceRendererSimple
# ==================================================================================


class TestResilienceRendererSimple:
    """Tests for ResilienceRendererSimple."""

    def test_initialization(self, mock_template_manager, mock_asset_manager):
        """Test renderer initialization."""
        from deepbridge.core.experiment.report.renderers.resilience_renderer_simple import (
            ResilienceRendererSimple,
        )

        renderer = ResilienceRendererSimple(
            mock_template_manager, mock_asset_manager
        )

        assert renderer.template_manager is mock_template_manager
        assert hasattr(renderer, 'data_transformer')

    def test_inherits_from_base_renderer(
        self, mock_template_manager, mock_asset_manager
    ):
        """Test that renderer inherits from BaseRenderer."""
        from deepbridge.core.experiment.report.renderers.base_renderer import (
            BaseRenderer,
        )
        from deepbridge.core.experiment.report.renderers.resilience_renderer_simple import (
            ResilienceRendererSimple,
        )

        renderer = ResilienceRendererSimple(
            mock_template_manager, mock_asset_manager
        )
        assert isinstance(renderer, BaseRenderer)


# ==================================================================================
# Tests for FairnessRendererSimple
# ==================================================================================


class TestFairnessRendererSimple:
    """Tests for FairnessRendererSimple."""

    def test_initialization(self, mock_template_manager, mock_asset_manager):
        """Test renderer initialization."""
        from deepbridge.core.experiment.report.renderers.fairness_renderer_simple import (
            FairnessRendererSimple,
        )

        renderer = FairnessRendererSimple(
            mock_template_manager, mock_asset_manager
        )

        assert renderer.template_manager is mock_template_manager
        assert hasattr(renderer, 'data_transformer')

    def test_inherits_from_base_renderer(
        self, mock_template_manager, mock_asset_manager
    ):
        """Test that renderer inherits from BaseRenderer."""
        from deepbridge.core.experiment.report.renderers.base_renderer import (
            BaseRenderer,
        )
        from deepbridge.core.experiment.report.renderers.fairness_renderer_simple import (
            FairnessRendererSimple,
        )

        renderer = FairnessRendererSimple(
            mock_template_manager, mock_asset_manager
        )
        assert isinstance(renderer, BaseRenderer)


# ==================================================================================
# Tests for HyperparameterRenderer
# ==================================================================================


class TestHyperparameterRenderer:
    """Tests for HyperparameterRenderer."""

    def test_initialization(self, mock_template_manager, mock_asset_manager):
        """Test renderer initialization."""
        from deepbridge.core.experiment.report.renderers.hyperparameter_renderer import (
            HyperparameterRenderer,
        )

        renderer = HyperparameterRenderer(
            mock_template_manager, mock_asset_manager
        )

        assert renderer.template_manager is mock_template_manager
        assert hasattr(renderer, 'data_transformer')

    def test_inherits_from_base_renderer(
        self, mock_template_manager, mock_asset_manager
    ):
        """Test that renderer inherits from BaseRenderer."""
        from deepbridge.core.experiment.report.renderers.base_renderer import (
            BaseRenderer,
        )
        from deepbridge.core.experiment.report.renderers.hyperparameter_renderer import (
            HyperparameterRenderer,
        )

        renderer = HyperparameterRenderer(
            mock_template_manager, mock_asset_manager
        )
        assert isinstance(renderer, BaseRenderer)


# ==================================================================================
# Integration Tests - Phase 2 Pattern Verification
# ==================================================================================


class TestSimpleRendererPhase2Pattern:
    """Integration tests verifying Phase 2 refactoring pattern."""

    def test_all_renderers_inherit_from_base_renderer(
        self, mock_template_manager, mock_asset_manager
    ):
        """Verify all simple renderers inherit from BaseRenderer (Phase 2 goal)."""
        from deepbridge.core.experiment.report.renderers.base_renderer import (
            BaseRenderer,
        )
        from deepbridge.core.experiment.report.renderers.fairness_renderer_simple import (
            FairnessRendererSimple,
        )
        from deepbridge.core.experiment.report.renderers.hyperparameter_renderer import (
            HyperparameterRenderer,
        )
        from deepbridge.core.experiment.report.renderers.resilience_renderer_simple import (
            ResilienceRendererSimple,
        )
        from deepbridge.core.experiment.report.renderers.robustness_renderer_simple import (
            RobustnessRendererSimple,
        )
        from deepbridge.core.experiment.report.renderers.uncertainty_renderer_simple import (
            UncertaintyRendererSimple,
        )

        renderers = [
            UncertaintyRendererSimple(
                mock_template_manager, mock_asset_manager
            ),
            RobustnessRendererSimple(
                mock_template_manager, mock_asset_manager
            ),
            ResilienceRendererSimple(
                mock_template_manager, mock_asset_manager
            ),
            FairnessRendererSimple(mock_template_manager, mock_asset_manager),
            HyperparameterRenderer(mock_template_manager, mock_asset_manager),
        ]

        for renderer in renderers:
            assert isinstance(
                renderer, BaseRenderer
            ), f'{renderer.__class__.__name__} does not inherit from BaseRenderer'

    def test_all_renderers_have_consistent_interface(
        self, mock_template_manager, mock_asset_manager
    ):
        """Verify all simple renderers have consistent render() interface."""
        from deepbridge.core.experiment.report.renderers.fairness_renderer_simple import (
            FairnessRendererSimple,
        )
        from deepbridge.core.experiment.report.renderers.hyperparameter_renderer import (
            HyperparameterRenderer,
        )
        from deepbridge.core.experiment.report.renderers.resilience_renderer_simple import (
            ResilienceRendererSimple,
        )
        from deepbridge.core.experiment.report.renderers.robustness_renderer_simple import (
            RobustnessRendererSimple,
        )
        from deepbridge.core.experiment.report.renderers.uncertainty_renderer_simple import (
            UncertaintyRendererSimple,
        )

        renderers = [
            UncertaintyRendererSimple(
                mock_template_manager, mock_asset_manager
            ),
            RobustnessRendererSimple(
                mock_template_manager, mock_asset_manager
            ),
            ResilienceRendererSimple(
                mock_template_manager, mock_asset_manager
            ),
            FairnessRendererSimple(mock_template_manager, mock_asset_manager),
            HyperparameterRenderer(mock_template_manager, mock_asset_manager),
        ]

        for renderer in renderers:
            # All should have render method
            assert hasattr(
                renderer, 'render'
            ), f'{renderer.__class__.__name__} missing render() method'
            assert callable(
                renderer.render
            ), f'{renderer.__class__.__name__}.render is not callable'

            # All should have data_transformer
            assert hasattr(
                renderer, 'data_transformer'
            ), f'{renderer.__class__.__name__} missing data_transformer'

            # All should inherit BaseRenderer template methods (Phase 2 pattern)
            for method_name in [
                '_load_template',
                '_get_assets',
                '_create_base_context',
                '_render_template',
                '_write_html',
            ]:
                assert hasattr(
                    renderer, method_name
                ), f'{renderer.__class__.__name__} missing {method_name}'

    def test_template_method_pattern_eliminates_duplication(
        self, mock_template_manager, mock_asset_manager
    ):
        """
        Verify Phase 2 benefit: Template method pattern eliminates code duplication.

        Before Phase 2: Each renderer had ~180 lines of duplicate helper methods
        After Phase 2: All renderers inherit these methods from BaseRenderer
        """
        from deepbridge.core.experiment.report.renderers.uncertainty_renderer_simple import (
            UncertaintyRendererSimple,
        )

        renderer = UncertaintyRendererSimple(
            mock_template_manager, mock_asset_manager
        )

        # These methods should be inherited, not redefined
        import inspect

        # _load_template should be defined in BaseRenderer, not in UncertaintyRendererSimple
        method = renderer._load_template
        defining_class = method.__qualname__.split('.')[0]

        assert (
            defining_class == 'BaseRenderer'
        ), 'Method should be inherited from BaseRenderer, not redefined'

    def test_renderers_use_composition_for_transformers(
        self, mock_template_manager, mock_asset_manager
    ):
        """Verify renderers use composition pattern for data transformers."""
        from deepbridge.core.experiment.report.renderers.robustness_renderer_simple import (
            RobustnessRendererSimple,
        )
        from deepbridge.core.experiment.report.renderers.uncertainty_renderer_simple import (
            UncertaintyRendererSimple,
        )

        uncertainty_renderer = UncertaintyRendererSimple(
            mock_template_manager, mock_asset_manager
        )
        robustness_renderer = RobustnessRendererSimple(
            mock_template_manager, mock_asset_manager
        )

        # Each renderer should have its own transformer type
        assert (
            type(uncertainty_renderer.data_transformer).__name__
            == 'UncertaintyDataTransformerSimple'
        )
        assert (
            type(robustness_renderer.data_transformer).__name__
            == 'RobustnessDataTransformerSimple'
        )

        # Transformers are different instances
        assert (
            uncertainty_renderer.data_transformer
            is not robustness_renderer.data_transformer
        )


class TestPhase2Metrics:
    """Tests documenting Phase 2 improvements."""

    def test_code_reduction_metric(
        self, mock_template_manager, mock_asset_manager
    ):
        """
        Phase 2 Goal: Reduce code duplication.

        Metric: Each simple renderer eliminated ~180 lines of duplicate code
        by inheriting from BaseRenderer instead of defining helper methods.
        """
        import inspect

        from deepbridge.core.experiment.report.renderers.uncertainty_renderer_simple import (
            UncertaintyRendererSimple,
        )

        # Get source code
        source = inspect.getsource(UncertaintyRendererSimple)
        lines = source.split('\n')

        # Count non-empty, non-comment lines
        code_lines = [
            line
            for line in lines
            if line.strip() and not line.strip().startswith('#')
        ]

        # After Phase 2, simple renderers should be < 150 lines
        # (Before Phase 2: ~280 lines with duplicated helpers)
        assert (
            len(code_lines) < 150
        ), f'Renderer has {len(code_lines)} lines, expected < 150 after Phase 2 refactoring'

    def test_five_renderers_refactored(
        self, mock_template_manager, mock_asset_manager
    ):
        """Verify all 5 simple renderers were refactored in Phase 2."""
        from deepbridge.core.experiment.report.renderers.base_renderer import (
            BaseRenderer,
        )

        renderer_classes = [
            'UncertaintyRendererSimple',
            'RobustnessRendererSimple',
            'ResilienceRendererSimple',
            'FairnessRendererSimple',
            'HyperparameterRenderer',
        ]

        for class_name in renderer_classes:
            if class_name == 'UncertaintyRendererSimple':
                from deepbridge.core.experiment.report.renderers.uncertainty_renderer_simple import (
                    UncertaintyRendererSimple,
                )

                cls = UncertaintyRendererSimple
            elif class_name == 'RobustnessRendererSimple':
                from deepbridge.core.experiment.report.renderers.robustness_renderer_simple import (
                    RobustnessRendererSimple,
                )

                cls = RobustnessRendererSimple
            elif class_name == 'ResilienceRendererSimple':
                from deepbridge.core.experiment.report.renderers.resilience_renderer_simple import (
                    ResilienceRendererSimple,
                )

                cls = ResilienceRendererSimple
            elif class_name == 'FairnessRendererSimple':
                from deepbridge.core.experiment.report.renderers.fairness_renderer_simple import (
                    FairnessRendererSimple,
                )

                cls = FairnessRendererSimple
            else:  # HyperparameterRenderer
                from deepbridge.core.experiment.report.renderers.hyperparameter_renderer import (
                    HyperparameterRenderer,
                )

                cls = HyperparameterRenderer

            # Verify inheritance
            assert issubclass(
                cls, BaseRenderer
            ), f'{class_name} does not inherit from BaseRenderer'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
