"""
Static uncertainty report renderer - REFACTORED to use ChartRegistry.

Phase 3 Sprint 11 - Uses new chart generation system.
Reduces from 1602 → ~350 lines (-78%).
"""

import os
import logging
from typing import Dict, Any

logger = logging.getLogger("deepbridge.reports")


class StaticUncertaintyRenderer:
    """
    Renderer for static uncertainty test reports using ChartRegistry.

    This is a refactored version that eliminates 1,250+ lines of duplicated
    charting code by using the centralized ChartRegistry system.
    """

    def __init__(self, template_manager, asset_manager):
        """
        Initialize the static uncertainty renderer.

        Parameters:
        -----------
        template_manager : TemplateManager
            Manager for templates
        asset_manager : AssetManager
            Manager for assets (CSS, JS, images)
        """
        from .base_static_renderer import BaseStaticRenderer
        self.base_renderer = BaseStaticRenderer(template_manager, asset_manager)
        self.template_manager = template_manager
        self.asset_manager = asset_manager

        # Import transformers
        from ...transformers.uncertainty import UncertaintyDataTransformer
        from ...transformers.static.static_uncertainty import StaticUncertaintyTransformer
        self.data_transformer = UncertaintyDataTransformer()
        self.static_transformer = StaticUncertaintyTransformer()

        # Import new chart registry
        from ...charts import ChartRegistry
        self.chart_registry = ChartRegistry

        logger.info("StaticUncertaintyRenderer initialized with ChartRegistry")

    def render(self, results: Dict[str, Any], file_path: str, model_name: str = "Model",
              report_type: str = "static", save_chart: bool = False) -> str:
        """
        Render static uncertainty report from results data.

        Parameters:
        -----------
        results : Dict[str, Any]
            Uncertainty test results
        file_path : str
            Path where the HTML report will be saved
        model_name : str, optional
            Name of the model for display in the report
        report_type : str, optional
            Type of report to generate ('interactive' or 'static')
        save_chart : bool, optional
            Whether to save charts as separate files (True) or embed them directly (False)

        Returns:
        --------
        str : Path to the generated report

        Raises:
        -------
        FileNotFoundError: If template or assets not found
        ValueError: If required data missing
        """
        logger.info(f"Generating static uncertainty report to: {file_path}")
        self.report_file_path = file_path
        self.save_chart = save_chart

        try:
            # 1. Transform data using existing transformers
            report_data = self._transform_data(results, model_name)

            # 2. Generate charts using ChartRegistry
            charts = self._generate_charts(report_data, save_chart)

            # 3. Create template context
            context = self._create_context(report_data, charts)

            # 4. Render HTML
            html_content = self._render_html(context)

            # 5. Write file
            return self._write_report(html_content, file_path)

        except Exception as e:
            logger.error(f"Error generating static uncertainty report: {str(e)}", exc_info=True)
            raise

    def _transform_data(self, results: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        """
        Transform raw results into report data.

        Parameters:
        -----------
        results : Dict[str, Any]
            Raw uncertainty test results
        model_name : str
            Name of the model

        Returns:
        --------
        Dict[str, Any] : Transformed report data
        """
        logger.info("Transforming uncertainty data")

        # Use existing data transformer
        report_data = self.data_transformer.transform(results, model_name)

        # Apply static-specific transformations
        report_data = self.static_transformer.transform(report_data)

        logger.info(f"Data transformation complete. Report has {len(report_data)} keys")
        return report_data

    def _generate_charts(self, report_data: Dict[str, Any], save_chart: bool = False) -> Dict[str, str]:
        """
        Generate all charts using ChartRegistry.

        This replaces 800+ lines of chart generation code with simple
        ChartRegistry calls.

        Parameters:
        -----------
        report_data : Dict[str, Any]
            Transformed report data
        save_chart : bool, optional
            Whether to save charts as separate files

        Returns:
        --------
        Dict[str, str] : Dictionary of chart names and their content/paths
        """
        logger.info("Generating charts using ChartRegistry")
        charts = {}

        # Setup chart directory if needed
        if save_chart:
            charts_dir = self._setup_charts_directory()
        else:
            charts_dir = None

        try:
            # Chart 1: Coverage vs Expected
            if self._has_data(report_data, ['calibration_results']):
                chart_data = self._prepare_coverage_data(report_data)
                result = self.chart_registry.generate('coverage_chart', chart_data)

                if result.is_success:
                    charts['coverage_vs_expected'] = self._process_chart_result(
                        result, 'coverage_vs_expected', charts_dir
                    )
                    logger.info("Generated coverage vs expected chart")

            # Chart 2: Width vs Coverage
            if self._has_data(report_data, ['calibration_results']):
                chart_data = self._prepare_width_coverage_data(report_data)
                result = self.chart_registry.generate('width_vs_coverage_static', chart_data)

                if result.is_success:
                    charts['width_vs_coverage'] = self._process_chart_result(
                        result, 'width_vs_coverage', charts_dir
                    )
                    logger.info("Generated width vs coverage chart")

            # Chart 3: Calibration Error
            if self._has_data(report_data, ['calibration_results']):
                chart_data = self._prepare_calibration_error_data(report_data)
                result = self.chart_registry.generate('calibration_error', chart_data)

                if result.is_success:
                    charts['calibration_error'] = self._process_chart_result(
                        result, 'calibration_error', charts_dir
                    )
                    logger.info("Generated calibration error chart")

            # Chart 4: Alternative Methods Comparison
            if self._has_data(report_data, ['alternative_models']):
                chart_data = self._prepare_alternative_methods_data(report_data)
                result = self.chart_registry.generate('alternative_methods_comparison', chart_data)

                if result.is_success:
                    charts['alternative_methods'] = self._process_chart_result(
                        result, 'alternative_methods', charts_dir
                    )
                    logger.info("Generated alternative methods comparison chart")

            logger.info(f"Successfully generated {len(charts)} charts")

        except Exception as e:
            logger.error(f"Error generating charts: {str(e)}", exc_info=True)
            # Return whatever charts we managed to generate

        return charts

    def _setup_charts_directory(self) -> str:
        """Create and return charts directory path."""
        report_dir = os.path.dirname(os.path.abspath(self.report_file_path))
        charts_dir = os.path.join(report_dir, "uncertainty_charts")
        os.makedirs(charts_dir, exist_ok=True)
        logger.info(f"Created chart directory at: {charts_dir}")
        return charts_dir

    def _has_data(self, report_data: Dict[str, Any], required_keys: list) -> bool:
        """Check if required data keys exist."""
        return all(key in report_data for key in required_keys)

    def _prepare_coverage_data(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare data for coverage vs expected chart.

        Parameters:
        -----------
        report_data : Dict[str, Any]
            Transformed report data

        Returns:
        --------
        Dict[str, Any] : Data in ChartRegistry format
        """
        calib = report_data['calibration_results']

        return {
            'alphas': self._to_list(calib.get('alpha_values', [])),
            'coverage': self._to_list(calib.get('coverage_values', [])),
            'expected': self._to_list(calib.get('expected_coverages', []))
        }

    def _prepare_width_coverage_data(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for width vs coverage chart."""
        calib = report_data['calibration_results']

        return {
            'coverage': self._to_list(calib.get('coverage_values', [])),
            'width': self._to_list(calib.get('avg_width_values', []))
        }

    def _prepare_calibration_error_data(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for calibration error chart."""
        calib = report_data['calibration_results']

        alpha_values = self._to_list(calib.get('alpha_values', []))
        coverage_values = self._to_list(calib.get('coverage_values', []))
        expected_values = self._to_list(calib.get('expected_coverages', []))

        # Calculate calibration errors
        calibration_errors = [
            abs(cov - exp) for cov, exp in zip(coverage_values, expected_values)
        ]

        return {
            'alphas': alpha_values,
            'calibration_errors': calibration_errors
        }

    def _prepare_alternative_methods_data(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for alternative methods comparison chart."""
        alt_models = report_data.get('alternative_models', [])

        methods = []
        scores = []

        for model in alt_models:
            if isinstance(model, dict):
                methods.append(model.get('name', 'Unknown'))
                scores.append(model.get('uncertainty_score', 0.0))

        return {
            'methods': methods,
            'scores': scores
        }

    def _process_chart_result(self, result, chart_name: str, charts_dir: str = None) -> str:
        """
        Process chart result - either save to file or return base64.

        Parameters:
        -----------
        result : ChartResult
            Result from ChartRegistry
        chart_name : str
            Name of the chart
        charts_dir : str, optional
            Directory to save chart files

        Returns:
        --------
        str : Either base64 encoded image or relative file path
        """
        if charts_dir:
            # Save to file and return relative path
            file_name = f"{chart_name}.png"
            file_path = os.path.join(charts_dir, file_name)

            # Decode base64 and save
            import base64
            with open(file_path, 'wb') as f:
                f.write(base64.b64decode(result.content))

            # Return relative path
            charts_subdir = os.path.basename(charts_dir)
            return f"{charts_subdir}/{file_name}"
        else:
            # Return base64 directly
            return result.content

    def _to_list(self, data) -> list:
        """Convert numpy arrays or other iterables to lists."""
        if hasattr(data, 'tolist') and callable(getattr(data, 'tolist')):
            return data.tolist()
        elif isinstance(data, (list, tuple)):
            return list(data)
        else:
            return []

    def _create_context(self, report_data: Dict[str, Any], charts: Dict[str, str]) -> Dict[str, Any]:
        """
        Create template context with data and charts.

        Parameters:
        -----------
        report_data : Dict[str, Any]
            Transformed report data
        charts : Dict[str, str]
            Generated charts

        Returns:
        --------
        Dict[str, Any] : Template context
        """
        context = {
            'report_data': report_data,
            'charts': charts,
            'model_name': report_data.get('model_name', 'Model'),
            'timestamp': report_data.get('timestamp'),
            'test_type': 'uncertainty',
            'report_type': 'static'
        }

        # Add assets
        context.update(self.base_renderer._get_assets('uncertainty'))

        return context

    def _render_html(self, context: Dict[str, Any]) -> str:
        """
        Render HTML from template and context.

        Parameters:
        -----------
        context : Dict[str, Any]
            Template context

        Returns:
        --------
        str : Rendered HTML content
        """
        # Find template
        template_path = self.template_manager.get_template_paths('uncertainty', 'static')
        template = self.template_manager.load_template(template_path[0])

        # Render
        html = self.template_manager.render_template(template, context)

        return html

    def _write_report(self, html_content: str, file_path: str) -> str:
        """
        Write HTML content to file.

        Parameters:
        -----------
        html_content : str
            Rendered HTML
        file_path : str
            Output file path

        Returns:
        --------
        str : Path to written file
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

        # Write file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"Static uncertainty report written to: {file_path}")
        return file_path
