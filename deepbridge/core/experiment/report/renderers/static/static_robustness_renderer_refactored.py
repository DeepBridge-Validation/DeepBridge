"""
Static robustness report renderer - REFACTORED to use ChartRegistry.

Phase 3 Sprint 11 - Uses new chart generation system.
Reduces from 546 → ~180 lines (-67%).
"""

import os
import logging
from typing import Dict, Any

logger = logging.getLogger("deepbridge.reports")


class StaticRobustnessRenderer:
    """
    Renderer for static robustness test reports using ChartRegistry.

    This is a refactored version that eliminates 350+ lines of duplicated
    charting code by using the centralized ChartRegistry system.
    """

    def __init__(self, template_manager, asset_manager):
        """
        Initialize the static robustness renderer.

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
        from ...transformers.robustness import RobustnessDataTransformer
        self.data_transformer = RobustnessDataTransformer()

        # Import static transformer if available
        try:
            from ...transformers.static import StaticRobustnessTransformer
            self.static_transformer = StaticRobustnessTransformer()
        except ImportError:
            self.static_transformer = None
            logger.warning("Static transformer not available")

        # Import new chart registry
        from ...charts import ChartRegistry
        self.chart_registry = ChartRegistry

        logger.info("StaticRobustnessRenderer initialized with ChartRegistry")

    def render(self, results: Dict[str, Any], file_path: str, model_name: str = "Model",
              report_type: str = "static", save_chart: bool = False) -> str:
        """
        Render static robustness report from results data.

        Parameters:
        -----------
        results : Dict[str, Any]
            Robustness test results
        file_path : str
            Path where the HTML report will be saved
        model_name : str, optional
            Name of the model for display in the report
        report_type : str, optional
            Type of report to generate ('interactive' or 'static')
        save_chart : bool, optional
            Whether to save charts as separate files

        Returns:
        --------
        str : Path to the generated report
        """
        logger.info(f"Generating static robustness report to: {file_path}")
        self.report_file_path = file_path
        self.save_chart = save_chart

        try:
            # 1. Transform data
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
            logger.error(f"Error generating static robustness report: {str(e)}", exc_info=True)
            raise

    def _transform_data(self, results: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        """Transform raw results into report data."""
        logger.info("Transforming robustness data")

        # Use existing data transformer
        report_data = self.data_transformer.transform(results, model_name)

        # Apply static-specific transformations if available
        if self.static_transformer:
            report_data = self.static_transformer.transform(results, model_name)
            logger.info("Applied static transformations")

        return report_data

    def _generate_charts(self, report_data: Dict[str, Any], save_chart: bool = False) -> Dict[str, str]:
        """
        Generate all charts using ChartRegistry.

        Replaces 200+ lines of chart generation code.

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
            # Chart 1: Perturbation Impact (Overview)
            if self._has_data(report_data, ['raw']):
                chart_data = self._prepare_perturbation_data(report_data)

                if chart_data['perturbation_levels']:
                    result = self.chart_registry.generate('perturbation_impact_static', chart_data)

                    if result.is_success:
                        charts['overview_chart'] = self._process_chart_result(
                            result, 'overview_chart', charts_dir
                        )
                        logger.info("Generated perturbation impact chart")

            # Chart 2: Feature Robustness
            if self._has_data(report_data, ['feature_impacts']):
                chart_data = self._prepare_feature_robustness_data(report_data)

                if chart_data['features']:
                    result = self.chart_registry.generate('feature_robustness', chart_data)

                    if result.is_success:
                        charts['feature_robustness'] = self._process_chart_result(
                            result, 'feature_robustness', charts_dir
                        )
                        logger.info("Generated feature robustness chart")

            # Chart 3: Model Comparison
            if self._has_data(report_data, ['alternative_models']) and report_data.get('alternative_models'):
                chart_data = self._prepare_model_comparison_data(report_data)

                if len(chart_data['models']) > 1:
                    result = self.chart_registry.generate('model_comparison', chart_data)

                    if result.is_success:
                        charts['comparison_chart'] = self._process_chart_result(
                            result, 'comparison_chart', charts_dir
                        )
                        logger.info("Generated model comparison chart")

            logger.info(f"Successfully generated {len(charts)} charts")

        except Exception as e:
            logger.error(f"Error generating charts: {str(e)}", exc_info=True)

        return charts

    def _setup_charts_directory(self) -> str:
        """Create and return charts directory path."""
        report_dir = os.path.dirname(os.path.abspath(self.report_file_path))
        charts_dir = os.path.join(report_dir, "robustness_charts")
        os.makedirs(charts_dir, exist_ok=True)
        logger.info(f"Created chart directory at: {charts_dir}")
        return charts_dir

    def _has_data(self, report_data: Dict[str, Any], required_keys: list) -> bool:
        """Check if required data keys exist."""
        return all(key in report_data for key in required_keys)

    def _prepare_perturbation_data(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare data for perturbation impact chart.

        Parameters:
        -----------
        report_data : Dict[str, Any]
            Transformed report data

        Returns:
        --------
        Dict[str, Any] : Data in ChartRegistry format
        """
        perturbation_levels = []
        mean_scores = []
        std_scores = []

        if 'raw' in report_data and 'by_level' in report_data['raw']:
            raw_data = report_data['raw']['by_level']

            # Sort levels numerically
            levels = sorted([float(level) for level in raw_data.keys()])

            for level in levels:
                level_str = str(level)
                if level_str in raw_data and 'overall_result' in raw_data[level_str]:
                    result = raw_data[level_str]['overall_result']

                    if 'all_features' in result:
                        perturbation_levels.append(level)
                        mean_scores.append(result['all_features'].get('mean_score', 0))
                        std_scores.append(result['all_features'].get('std_score', 0))

        return {
            'perturbation_levels': perturbation_levels,
            'mean_scores': mean_scores,
            'std_scores': std_scores
        }

    def _prepare_feature_robustness_data(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for feature robustness chart."""
        features = []
        robustness_scores = []

        feature_impacts = report_data.get('feature_impacts', {})

        for feature, impact_data in feature_impacts.items():
            if isinstance(impact_data, dict):
                # Robustness score = 1 - impact (higher is better)
                impact = impact_data.get('impact', 0)
                robustness = max(0, 1 - impact)

                features.append(feature)
                robustness_scores.append(robustness)

        return {
            'features': features,
            'robustness_scores': robustness_scores
        }

    def _prepare_model_comparison_data(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for model comparison chart."""
        models = [report_data.get('model_name', 'Primary Model')]
        metrics = {
            'robustness': [report_data.get('robustness_score', 0)]
        }

        # Add alternative models
        for model_name, model_data in report_data.get('alternative_models', {}).items():
            models.append(model_name)
            metrics['robustness'].append(model_data.get('robustness_score', 0))

        return {
            'models': models,
            'metrics': metrics
        }

    def _process_chart_result(self, result, chart_name: str, charts_dir: str = None) -> str:
        """Process chart result - either save to file or return base64."""
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

    def _create_context(self, report_data: Dict[str, Any], charts: Dict[str, str]) -> Dict[str, Any]:
        """Create template context with data and charts."""
        context = {
            'report_data': report_data,
            'charts': charts,
            'model_name': report_data.get('model_name', 'Model'),
            'timestamp': report_data.get('timestamp'),
            'test_type': 'robustness',
            'report_type': 'static'
        }

        # Add assets
        context.update(self.base_renderer._get_assets('robustness'))

        return context

    def _render_html(self, context: Dict[str, Any]) -> str:
        """Render HTML from template and context."""
        # Find template
        template_paths = self.template_manager.get_template_paths('robustness', 'static')
        template_path = self.template_manager.find_template(template_paths)

        if not template_path:
            raise FileNotFoundError(f"No static template found for robustness report")

        template = self.template_manager.load_template(template_path)

        # Render
        html = self.template_manager.render_template(template, context)

        return html

    def _write_report(self, html_content: str, file_path: str) -> str:
        """Write HTML content to file."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

        # Write file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"Static robustness report written to: {file_path}")
        return file_path
