"""
Public API for report generation.

This module provides the unified ReportGenerator API that replaces
all the old separate renderer classes.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

from .config import RenderConfig, OutputFormat, ReportStyle, get_preset_config
from .data.base import ReportData
from .data.robustness import RobustnessDataTransformer
from .renderers.html import HTMLRenderer, HTMLRendererWithAssets
from .renderers.json import JSONRenderer
from .templates import TemplateEngine


logger = logging.getLogger(__name__)


class ReportGenerator:
    """Unified API for generating all types of reports.

    This single class replaces all the old *_renderer.py variants:
    - robustness_renderer.py, robustness_renderer_simple.py, static_robustness_renderer.py
    - resilience_renderer.py, resilience_renderer_simple.py, static_resilience_renderer.py
    - uncertainty_renderer.py, uncertainty_renderer_simple.py, static_uncertainty_renderer.py
    - fairness_renderer.py, fairness_renderer_simple.py
    - distillation_renderer.py, hyperparameter_renderer.py

    All report types and styles are now handled through configuration.

    Example:
        >>> from deepbridge.core.experiment.report import ReportGenerator, RenderConfig
        >>> from pathlib import Path
        >>>
        >>> generator = ReportGenerator()
        >>>
        >>> # Generate full interactive report
        >>> generator.generate_robustness_report(
        ...     results=experiment.results,
        ...     output_path=Path("reports/robustness_full.html")
        ... )
        >>>
        >>> # Generate simple static report
        >>> generator.generate_robustness_report(
        ...     results=experiment.results,
        ...     output_path=Path("reports/robustness_simple.html"),
        ...     config=RenderConfig(style=ReportStyle.SIMPLE)
        ... )
        >>>
        >>> # Generate JSON for API
        >>> generator.generate_robustness_report(
        ...     results=experiment.results,
        ...     output_path=Path("reports/robustness.json"),
        ...     config=RenderConfig(format=OutputFormat.JSON)
        ... )
    """

    def __init__(
        self,
        template_dir: Optional[Path] = None,
        asset_manager=None,
    ):
        """Initialize report generator.

        Args:
            template_dir: Directory containing templates (None for default)
            asset_manager: Asset manager instance (None for default)
        """
        # Initialize template engine
        self.template_engine = TemplateEngine(template_dir)

        # Get the actual template directory being used
        actual_template_dir = self.template_engine.template_dir

        # AssetManager expects the parent directory that contains assets/common folders
        # If template_dir points to templates/html, we need templates/ parent
        if actual_template_dir.name == 'html':
            asset_manager_dir = actual_template_dir.parent
        else:
            asset_manager_dir = actual_template_dir

        # Initialize asset manager (if available)
        self.asset_manager = asset_manager
        if asset_manager is None:
            try:
                from .asset_manager import AssetManager
                self.asset_manager = AssetManager(str(asset_manager_dir))
            except (ImportError, FileNotFoundError) as e:
                logger.warning(f"AssetManager not available: {e}")
                self.asset_manager = None

        # Initialize renderers
        self.renderers = {
            OutputFormat.HTML: HTMLRendererWithAssets(
                self.template_engine,
                self.asset_manager
            ),
            OutputFormat.JSON: JSONRenderer(indent=2),
        }

        # Initialize transformers
        self.transformers = {
            'robustness': RobustnessDataTransformer(),
            # Others will be added as they are implemented
            # 'resilience': ResilienceDataTransformer(),
            # 'uncertainty': UncertaintyDataTransformer(),
            # 'fairness': FairnessDataTransformer(),
            # 'distillation': DistillationDataTransformer(),
        }

        logger.info("ReportGenerator initialized")

    def generate_robustness_report(
        self,
        results: Dict[str, Any],
        output_path: Union[str, Path],
        config: Optional[RenderConfig] = None,
    ) -> Path:
        """Generate robustness report.

        Replaces:
        - RobustnessRenderer (FULL style)
        - RobustnessRendererSimple (SIMPLE style)
        - StaticRobustnessRenderer (STATIC style)

        Args:
            results: Raw experiment results
            output_path: Path where to save the report
            config: Rendering configuration (None for defaults)

        Returns:
            Path to the generated report file

        Raises:
            ValueError: If results are invalid
            RuntimeError: If generation fails

        Example:
            >>> generator = ReportGenerator()
            >>>
            >>> # Full interactive HTML report
            >>> generator.generate_robustness_report(
            ...     results=experiment.results,
            ...     output_path="robustness.html"
            ... )
            >>>
            >>> # Simple static report
            >>> from deepbridge.core.experiment.report import RenderConfig, ReportStyle
            >>> generator.generate_robustness_report(
            ...     results=experiment.results,
            ...     output_path="robustness_simple.html",
            ...     config=RenderConfig(style=ReportStyle.SIMPLE)
            ... )
            >>>
            >>> # JSON output
            >>> from deepbridge.core.experiment.report import OutputFormat
            >>> generator.generate_robustness_report(
            ...     results=experiment.results,
            ...     output_path="robustness.json",
            ...     config=RenderConfig(format=OutputFormat.JSON)
            ... )
        """
        return self._generate_report(
            report_type='robustness',
            results=results,
            output_path=output_path,
            config=config
        )

    def generate_resilience_report(
        self,
        results: Dict[str, Any],
        output_path: Union[str, Path],
        config: Optional[RenderConfig] = None,
    ) -> Path:
        """Generate resilience report.

        Replaces:
        - ResilienceRenderer
        - ResilienceRendererSimple
        - StaticResilienceRenderer

        Args:
            results: Raw experiment results
            output_path: Path where to save the report
            config: Rendering configuration

        Returns:
            Path to the generated report file
        """
        return self._generate_report(
            report_type='resilience',
            results=results,
            output_path=output_path,
            config=config
        )

    def generate_uncertainty_report(
        self,
        results: Dict[str, Any],
        output_path: Union[str, Path],
        config: Optional[RenderConfig] = None,
    ) -> Path:
        """Generate uncertainty report.

        Replaces:
        - UncertaintyRenderer
        - UncertaintyRendererSimple
        - StaticUncertaintyRenderer

        Args:
            results: Raw experiment results
            output_path: Path where to save the report
            config: Rendering configuration

        Returns:
            Path to the generated report file
        """
        return self._generate_report(
            report_type='uncertainty',
            results=results,
            output_path=output_path,
            config=config
        )

    def generate_fairness_report(
        self,
        results: Dict[str, Any],
        output_path: Union[str, Path],
        config: Optional[RenderConfig] = None,
    ) -> Path:
        """Generate fairness report.

        Replaces:
        - FairnessRenderer
        - FairnessRendererSimple

        Args:
            results: Raw experiment results
            output_path: Path where to save the report
            config: Rendering configuration

        Returns:
            Path to the generated report file
        """
        return self._generate_report(
            report_type='fairness',
            results=results,
            output_path=output_path,
            config=config
        )

    def generate_distillation_report(
        self,
        results: Dict[str, Any],
        output_path: Union[str, Path],
        config: Optional[RenderConfig] = None,
    ) -> Path:
        """Generate distillation report.

        Replaces:
        - DistillationRenderer

        Args:
            results: Raw experiment results
            output_path: Path where to save the report
            config: Rendering configuration

        Returns:
            Path to the generated report file
        """
        return self._generate_report(
            report_type='distillation',
            results=results,
            output_path=output_path,
            config=config
        )

    def _generate_report(
        self,
        report_type: str,
        results: Dict[str, Any],
        output_path: Union[str, Path],
        config: Optional[RenderConfig] = None,
    ) -> Path:
        """Internal method to generate any type of report.

        Args:
            report_type: Type of report ('robustness', 'resilience', etc.)
            results: Raw experiment results
            output_path: Output file path
            config: Rendering configuration

        Returns:
            Path to generated file

        Raises:
            ValueError: If report_type is not supported or data is invalid
            RuntimeError: If generation fails
        """
        # Use default config if none provided
        if config is None:
            config = RenderConfig()

        output_path = Path(output_path)

        logger.info(f"Generating {report_type} report...")
        logger.info(f"Output: {output_path}")
        logger.info(f"Style: {config.style.value}, Format: {config.format.value}")

        try:
            # Step 1: Get appropriate transformer
            if report_type not in self.transformers:
                raise ValueError(
                    f"Unsupported report type: {report_type}. "
                    f"Supported types: {list(self.transformers.keys())}"
                )

            transformer = self.transformers[report_type]

            # Step 2: Transform raw data to typed structure
            logger.debug("Transforming raw data...")
            typed_data = transformer.transform(results)

            # Step 3: Validate data
            logger.debug("Validating data...")
            typed_data.validate()

            # Step 4: Get appropriate renderer
            if config.format not in self.renderers:
                raise ValueError(
                    f"Unsupported output format: {config.format}. "
                    f"Supported formats: {list(self.renderers.keys())}"
                )

            renderer = self.renderers[config.format]

            # Step 5: Render and save
            logger.debug("Rendering report...")
            output_file = renderer.render_to_file(typed_data, config, output_path)

            logger.info(f"Report generated successfully: {output_file}")
            return output_file

        except ValueError as e:
            # Check if this is a configuration error (re-raise as-is)
            # or a data validation error (wrap in RuntimeError)
            error_msg = str(e)
            if "Unsupported report type" in error_msg or "Unsupported output format" in error_msg:
                # Configuration error - re-raise as-is
                raise
            else:
                # Data validation error - wrap in RuntimeError
                logger.error(f"Failed to generate {report_type} report: {e}")
                raise RuntimeError(
                    f"Report generation failed for {report_type}: {e}"
                ) from e
        except Exception as e:
            logger.error(f"Failed to generate {report_type} report: {e}", exc_info=True)
            raise RuntimeError(
                f"Report generation failed for {report_type}: {e}"
            ) from e

    def add_transformer(self, report_type: str, transformer) -> None:
        """Register a custom transformer for a report type.

        Args:
            report_type: Report type identifier
            transformer: DataTransformer instance

        Example:
            >>> from .data.custom import CustomTransformer
            >>> generator.add_transformer('custom', CustomTransformer())
        """
        self.transformers[report_type] = transformer
        logger.info(f"Registered transformer for report type: {report_type}")

    def add_renderer(self, format: OutputFormat, renderer) -> None:
        """Register a custom renderer for an output format.

        Args:
            format: Output format
            renderer: ReportRenderer instance

        Example:
            >>> from .renderers.pdf import PDFRenderer
            >>> generator.add_renderer(OutputFormat.PDF, PDFRenderer())
        """
        self.renderers[format] = renderer
        logger.info(f"Registered renderer for format: {format.value}")


def generate_robustness_report(
    results: Dict[str, Any],
    output_path: Union[str, Path],
    config: Optional[RenderConfig] = None,
    **kwargs
) -> Path:
    """Convenience function to generate robustness report.

    Creates a ReportGenerator instance and generates the report.

    Args:
        results: Raw experiment results
        output_path: Path where to save the report
        config: Rendering configuration
        **kwargs: Additional arguments passed to ReportGenerator

    Returns:
        Path to the generated report

    Example:
        >>> from deepbridge.core.experiment.report import generate_robustness_report
        >>> generate_robustness_report(
        ...     results=experiment.results,
        ...     output_path="robustness.html"
        ... )
    """
    generator = ReportGenerator(**kwargs)
    return generator.generate_robustness_report(results, output_path, config)


# Convenience functions for other report types
def generate_resilience_report(
    results: Dict[str, Any],
    output_path: Union[str, Path],
    config: Optional[RenderConfig] = None,
    **kwargs
) -> Path:
    """Convenience function to generate resilience report."""
    generator = ReportGenerator(**kwargs)
    return generator.generate_resilience_report(results, output_path, config)


def generate_uncertainty_report(
    results: Dict[str, Any],
    output_path: Union[str, Path],
    config: Optional[RenderConfig] = None,
    **kwargs
) -> Path:
    """Convenience function to generate uncertainty report."""
    generator = ReportGenerator(**kwargs)
    return generator.generate_uncertainty_report(results, output_path, config)


def generate_fairness_report(
    results: Dict[str, Any],
    output_path: Union[str, Path],
    config: Optional[RenderConfig] = None,
    **kwargs
) -> Path:
    """Convenience function to generate fairness report."""
    generator = ReportGenerator(**kwargs)
    return generator.generate_fairness_report(results, output_path, config)
