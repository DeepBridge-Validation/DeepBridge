"""
Static resilience report renderer - REFACTORED to use ChartRegistry.

Phase 3 Sprint 11 - Uses new chart generation system.
Reduces from 1,226 → ~380 lines (-69%).
"""

import os
import logging
from typing import Dict, Any, List

logger = logging.getLogger("deepbridge.reports")


class StaticResilienceRenderer:
    """
    Renderer for static resilience test reports using ChartRegistry.

    This is a refactored version that eliminates 800+ lines of duplicated
    charting code by using the centralized ChartRegistry system.
    """

    def __init__(self, template_manager, asset_manager):
        """
        Initialize the static resilience renderer.

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
        from ...transformers.resilience import ResilienceDataTransformer
        self.data_transformer = ResilienceDataTransformer()

        # Import static transformer if available
        try:
            from ...transformers.static import StaticResilienceTransformer
            self.static_transformer = StaticResilienceTransformer()
        except ImportError:
            self.static_transformer = None
            logger.warning("Static transformer not available")

        # Import new chart registry
        from ...charts import ChartRegistry
        self.chart_registry = ChartRegistry

        logger.info("StaticResilienceRenderer initialized with ChartRegistry")

    def render(self, results: Dict[str, Any], file_path: str, model_name: str = "Model",
              report_type: str = "static", save_chart: bool = False) -> str:
        """
        Render static resilience report from results data.

        Parameters:
        -----------
        results : Dict[str, Any]
            Resilience test results
        file_path : str
            Path where the HTML report will be saved
        model_name : str, optional
            Name of the model for display in the report
        report_type : str, optional
            Type of report to generate
        save_chart : bool, optional
            Whether to save charts as separate files

        Returns:
        --------
        str : Path to the generated report
        """
        logger.info(f"Generating static resilience report to: {file_path}")
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
            logger.error(f"Error generating static resilience report: {str(e)}", exc_info=True)
            raise

    def _transform_data(self, results: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        """Transform raw results into report data."""
        logger.info("Transforming resilience data")

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

        Replaces 500+ lines of chart generation code.

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
            # Chart 1: Test Type Comparison (Radar)
            test_type_data = self._prepare_test_type_data(report_data)
            if test_type_data['test_types']:
                result = self.chart_registry.generate('test_type_comparison', test_type_data)

                if result.is_success:
                    charts['test_type_comparison'] = self._process_chart_result(
                        result, 'test_type_comparison', charts_dir
                    )
                    logger.info("Generated test type comparison chart")

            # Chart 2: Scenario Degradation (PSI)
            if self._has_data(report_data, ['scenarios']):
                scenario_data = self._prepare_scenario_degradation_data(report_data)

                if scenario_data['scenarios']:
                    result = self.chart_registry.generate('scenario_degradation', scenario_data)

                    if result.is_success:
                        charts['scenario_degradation'] = self._process_chart_result(
                            result, 'scenario_degradation', charts_dir
                        )
                        logger.info("Generated scenario degradation chart")

            # Chart 3: Feature Distribution Shift
            if self._has_data(report_data, ['feature_distances']):
                feature_data = self._prepare_feature_shift_data(report_data)

                if feature_data['features']:
                    # Use feature_robustness chart (adapted for distribution shift)
                    result = self.chart_registry.generate('feature_robustness', feature_data)

                    if result.is_success:
                        charts['feature_distribution_shift'] = self._process_chart_result(
                            result, 'feature_distribution_shift', charts_dir
                        )
                        logger.info("Generated feature distribution shift chart")

            # Chart 4: Model Comparison
            if self._has_data(report_data, ['alternative_models']) and report_data.get('alternative_models'):
                model_data = self._prepare_model_comparison_data(report_data)

                if len(model_data['models']) > 1:
                    result = self.chart_registry.generate('model_comparison', model_data)

                    if result.is_success:
                        charts['model_comparison'] = self._process_chart_result(
                            result, 'model_comparison', charts_dir
                        )
                        logger.info("Generated model comparison chart")

            logger.info(f"Successfully generated {len(charts)} charts")

        except Exception as e:
            logger.error(f"Error generating charts: {str(e)}", exc_info=True)

        return charts

    def _setup_charts_directory(self) -> str:
        """Create and return charts directory path."""
        report_dir = os.path.dirname(os.path.abspath(self.report_file_path))
        charts_dir = os.path.join(report_dir, "resilience_charts")
        os.makedirs(charts_dir, exist_ok=True)
        logger.info(f"Created chart directory at: {charts_dir}")
        return charts_dir

    def _has_data(self, report_data: Dict[str, Any], required_keys: list) -> bool:
        """Check if required data keys exist."""
        return all(key in report_data for key in required_keys)

    def _prepare_test_type_data(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare data for test type comparison radar chart.

        Parameters:
        -----------
        report_data : Dict[str, Any]
            Transformed report data

        Returns:
        --------
        Dict[str, Any] : Data in ChartRegistry format
        """
        test_types = []
        scores = []

        # Extract test types and their scores
        test_results = report_data.get('test_results', {})

        for test_type, result in test_results.items():
            if isinstance(result, dict) and 'score' in result:
                test_types.append(test_type)
                scores.append(result['score'])

        # If no test results, use alternative data structure
        if not test_types:
            # Try to extract from performance_metrics
            perf_metrics = report_data.get('performance_metrics', {})
            if perf_metrics:
                for metric_name, value in perf_metrics.items():
                    if not metric_name.startswith('_'):  # Skip internal keys
                        test_types.append(metric_name.replace('_', ' ').title())
                        scores.append(float(value) if value is not None else 0.0)

        # Ensure we have at least some default data
        if not test_types:
            test_types = ['Resilience Score', 'Performance Gap']
            scores = [
                report_data.get('resilience_score', 0.0),
                report_data.get('avg_performance_gap', 0.0)
            ]

        return {
            'test_types': test_types,
            'scores': scores
        }

    def _prepare_scenario_degradation_data(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for scenario degradation chart."""
        scenarios = []
        psi_values = []
        performance = []

        # Extract scenario data
        scenario_data = report_data.get('scenarios', {})

        for scenario_name, scenario_info in scenario_data.items():
            if isinstance(scenario_info, dict):
                scenarios.append(scenario_name)
                psi_values.append(scenario_info.get('psi', 0.0))
                performance.append(scenario_info.get('performance', scenario_info.get('score', 0.0)))

        # If no scenarios, create default with base performance
        if not scenarios:
            scenarios = ['Base']
            psi_values = [0.0]
            performance = [report_data.get('base_performance', 1.0)]

        return {
            'scenarios': scenarios,
            'psi_values': psi_values,
            'performance': performance
        }

    def _prepare_feature_shift_data(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for feature distribution shift chart."""
        features = []
        robustness_scores = []

        feature_distances = report_data.get('feature_distances', {})

        # Sort by distance (highest shift first)
        sorted_features = sorted(
            feature_distances.items(),
            key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0,
            reverse=True
        )

        # Take top 10 features with highest shift
        for feature, distance in sorted_features[:10]:
            features.append(feature)
            # Convert distance to "stability score" (1 - normalized_distance)
            # Assuming distance is already normalized or we use inverse
            stability = max(0, 1 - min(distance, 1)) if isinstance(distance, (int, float)) else 0.5
            robustness_scores.append(stability)

        return {
            'features': features,
            'robustness_scores': robustness_scores
        }

    def _prepare_model_comparison_data(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for model comparison chart."""
        models = [report_data.get('model_name', 'Primary Model')]
        metrics = {
            'resilience': [report_data.get('resilience_score', 0.0)],
            'performance': [report_data.get('base_performance', 0.0)]
        }

        # Add alternative models
        for model_name, model_data in report_data.get('alternative_models', {}).items():
            models.append(model_name)
            metrics['resilience'].append(model_data.get('resilience_score', 0.0))
            metrics['performance'].append(model_data.get('performance', 0.0))

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
            'test_type': 'resilience',
            'report_type': 'static'
        }

        # Add assets
        context.update(self.base_renderer._get_assets('resilience'))

        return context

    def _render_html(self, context: Dict[str, Any]) -> str:
        """Render HTML from template and context."""
        # Find template
        template_paths = self.template_manager.get_template_paths('resilience', 'static')
        template_path = self.template_manager.find_template(template_paths)

        if not template_path:
            raise FileNotFoundError(f"No static template found for resilience report")

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

        logger.info(f"Static resilience report written to: {file_path}")
        return file_path
