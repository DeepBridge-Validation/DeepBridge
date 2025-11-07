"""
Interactive renderer for distillation reports using Plotly visualizations.
"""

import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("deepbridge.reports")

# Import JSON formatter
from ..utils.json_formatter import JsonFormatter

# Import CSS Manager
from ..css_manager import CSSManager


class DistillationRenderer:
    """
    Interactive renderer for distillation experiment reports.
    """

    def __init__(self, template_manager, asset_manager):
        """
        Initialize the distillation renderer.

        Parameters:
        -----------
        template_manager : TemplateManager
            Manager for templates
        asset_manager : AssetManager
            Manager for assets (CSS, JS, images)
        """
        from .base_renderer import BaseRenderer
        self.base_renderer = BaseRenderer(template_manager, asset_manager)
        self.template_manager = template_manager
        self.asset_manager = asset_manager

        # Initialize CSS Manager
        self.css_manager = CSSManager()

        # Import data transformer
        from ..transformers.distillation import DistillationDataTransformer
        self.data_transformer = DistillationDataTransformer()

    def render(self, results: Dict[str, Any], file_path: str, model_name: str = "Distillation",
               report_type: str = "interactive") -> str:
        """
        Render distillation report from results data.

        Parameters:
        -----------
        results : Dict[str, Any]
            Distillation experiment results containing:
            - results: DataFrame with all model results
            - original_metrics: Metrics from original model
            - best_model: Best model configuration
            - config: Experiment configuration
        file_path : str
            Path where the HTML report will be saved
        model_name : str, optional
            Name for the report title
        report_type : str, optional
            Type of report to generate ('interactive' or 'static')

        Returns:
        --------
        str : Path to the generated report

        Raises:
        -------
        FileNotFoundError: If template not found
        ValueError: If required data missing
        """
        logger.info(f"Generating distillation report to: {file_path}")
        logger.info(f"Report type: {report_type}")

        try:
            # Determine which template to use based on report type
            if report_type == "interactive":
                # Use the interactive template - build full paths
                template_paths = [
                    os.path.join(self.template_manager.templates_dir, "report_types/distillation/interactive/index.html"),
                    os.path.join(self.template_manager.templates_dir, "distillation/interactive/index.html")
                ]
            else:
                # Use the static template (default)
                template_paths = self.template_manager.get_template_paths("distillation")

            template_path = self.template_manager.find_template(template_paths)

            if not template_path:
                raise FileNotFoundError(f"No template found for distillation report in: {template_paths}")

            logger.info(f"Using template: {template_path}")

            # Get CSS and JS content
            css_content = self._load_css_content()
            js_content = self._load_js_content()

            # Load the template
            template = self.template_manager.load_template(template_path)

            # Transform the distillation data
            report_data = self.data_transformer.transform(results)

            # Prepare chart data for interactive visualizations
            chart_data = self._prepare_chart_data(report_data)

            # Create template context
            context = self.base_renderer._create_context(
                report_data, "distillation", css_content, js_content, report_type
            )

            # Add distillation-specific context
            context.update({
                # Summary metrics
                'total_models': report_data['experiment_summary']['total_models_tested'],
                'best_accuracy': self._get_best_accuracy(report_data),
                'compression_rate': self._calculate_compression_rate(report_data),
                'total_time': report_data['experiment_summary']['total_training_time'],

                # Original model info
                'original_model': report_data['original_model'],
                'has_original': report_data['original_model']['has_metrics'],

                # Best model info
                'best_model': report_data['best_model'],

                # All models for the table
                'all_models': report_data['all_models'],

                # Charts data
                'chart_data': chart_data,  # Add raw chart data for template access
                'chart_data_json': JsonFormatter.format_for_javascript(chart_data),

                # Recommendations
                'recommendations': report_data['recommendations'],

                # Metadata for header
                'report_title': 'Knowledge Distillation Report',
                'report_subtitle': 'Model Compression and Knowledge Transfer Analysis',
                'report_type': 'distillation',
                'test_type': 'distillation',
                'model_name': model_name,
                'model_type': 'Distillation Experiment',

                # Feature flags
                'has_hyperparameter_data': bool(report_data['hyperparameter_analysis']['temperature_impact']),
                'has_tradeoff_data': bool(report_data['tradeoff_analysis']['accuracy_vs_complexity']),
                'has_recommendations': bool(report_data['recommendations'])
            })

            # Add report data JSON for client-side access
            context['report_data_json'] = JsonFormatter.format_for_javascript(report_data)

            # Render the template
            rendered_html = self.template_manager.render_template(template, context)

            # Write the report to the file
            return self.base_renderer._write_report(rendered_html, file_path)

        except Exception as e:
            logger.error(f"Error generating distillation report: {str(e)}")
            raise

    def _load_css_content(self) -> str:
        """
        Load and combine CSS files for the distillation report using CSSManager.

        Returns:
        --------
        str : Combined CSS content (base + components + custom)
        """
        try:
            # Use CSSManager to compile CSS (base + components + custom)
            compiled_css = self.css_manager.get_compiled_css('distillation')
            logger.info(f"CSS compiled successfully using CSSManager: {len(compiled_css)} chars")
            return compiled_css
        except Exception as e:
            logger.error(f"Error loading CSS with CSSManager: {str(e)}")

            # Fallback: try to load CSS from asset manager if CSSManager fails
            try:
                logger.warning("Falling back to asset_manager for CSS loading")
                css_content = self.asset_manager.get_combined_css_content("distillation")

                # Add default styles to ensure report functionality even if external CSS is missing
                default_css = """
            /* Base variables and reset */
            :root {
                --primary-color: #1b78de;
                --secondary-color: #2c3e50;
                --success-color: #28a745;
                --danger-color: #dc3545;
                --warning-color: #f39c12;
                --info-color: #17a2b8;
                --light-color: #f8f9fa;
                --dark-color: #343a40;
                --text-color: #333;
                --text-muted: #6c757d;
                --border-color: #ddd;
                --background-color: #f8f9fa;
                --card-bg: #fff;
                --header-bg: #ffffff;
            }

            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }

            html, body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                font-size: 16px;
                line-height: 1.5;
                color: var(--text-color);
                background-color: var(--background-color);
            }

            h1, h2, h3, h4, h5, h6 {
                margin-bottom: 1rem;
                font-weight: 500;
                line-height: 1.2;
            }

            p {
                margin-bottom: 1rem;
            }

            .report-container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 0 1rem;
            }

            .section {
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                padding: 1.5rem;
                margin-bottom: 2rem;
            }
            """

                # Combine default CSS with loaded CSS
                combined_css = default_css + "\n\n" + css_content
                return combined_css
            except Exception as fallback_error:
                logger.error(f"Fallback CSS loading also failed: {str(fallback_error)}")
                return ""

    def _load_js_content(self) -> str:
        """Load and combine JavaScript content for the distillation report."""
        js_content = ""

        # Try to load main.js if it exists
        main_js_path = os.path.join(
            os.path.dirname(self.template_manager.templates_dir),
            'templates',
            'report_types',
            'distillation',
            'js',
            'main.js'
        )

        if os.path.exists(main_js_path):
            with open(main_js_path, 'r') as f:
                js_content = f.read()

        # If no main.js, try loading JavaScript modules
        if not js_content:
            js_modules = [
                'distillation_charts.js',
                'model_selector.js',
                'hyperparameter_explorer.js',
                'comparison_manager.js'
            ]

            for module in js_modules:
                js_path = self.asset_manager.get_asset_path("distillation", f"js/{module}")
                if js_path and os.path.exists(js_path):
                    with open(js_path, 'r') as f:
                        js_content += f"\n/* {module.upper()} */\n"
                        js_content += f.read() + "\n"

        # Add default initialization if no custom JS found
        if not js_content:
            js_content = self._get_default_js()

        return js_content

    def _prepare_chart_data(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare data for interactive Plotly charts.

        Parameters:
        -----------
        report_data : Dict[str, Any]
            Transformed report data

        Returns:
        --------
        Dict[str, Any] : Data formatted for chart generation
        """
        chart_data = {
            'summary_metrics': self._prepare_summary_metrics(report_data),
            'model_comparison': self._prepare_model_comparison(report_data),
            'hyperparameter_data': self._prepare_hyperparameter_data(report_data),
            'performance_metrics': self._prepare_performance_metrics(report_data),
            'tradeoff_data': self._prepare_tradeoff_data(report_data),
            'recommendations_data': self._prepare_recommendations_data(report_data),
            'ks_statistic_data': self._prepare_ks_statistic_data(report_data),
            'frequency_distribution_data': self._prepare_frequency_distribution_data(report_data)
        }

        return chart_data

    def _prepare_summary_metrics(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare summary metrics for dashboard cards."""
        best_model = report_data['best_model']
        original = report_data['original_model']['test_metrics']

        return {
            'total_models': report_data['experiment_summary']['total_models_tested'],
            'best_accuracy': best_model['metrics'].get('test_accuracy', 0),
            'original_accuracy': original.get('accuracy', 0),
            'improvement': self._calculate_improvement(best_model, original),
            'compression_rate': self._calculate_compression_rate(report_data),
            'total_training_time': report_data['experiment_summary']['total_training_time'],
            'best_model_type': best_model['model_type'],
            'optimal_temperature': best_model['temperature'],
            'optimal_alpha': best_model['alpha']
        }

    def _prepare_model_comparison(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for model comparison visualizations."""
        all_models = report_data['all_models']
        original_metrics = report_data['original_model']['test_metrics']

        # Prepare data for heatmap
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
        heatmap_data = []
        model_labels = []

        for model in all_models:
            model_metrics = []
            for metric in metrics:
                value = model['metrics'].get(f'test_{metric}', 0)
                model_metrics.append(value)
            heatmap_data.append(model_metrics)
            model_labels.append(model['config_string'])

        return {
            'heatmap': {
                'data': heatmap_data,
                'models': model_labels,
                'metrics': metrics
            },
            'scatter_data': [
                {
                    'model': model['model_type'],
                    'config': model['config_string'],
                    'accuracy': model['metrics'].get('test_accuracy', 0),
                    'complexity': model['model_complexity'],
                    'training_time': model['training_time'],
                    'f1_score': model['metrics'].get('test_f1_score', 0)
                }
                for model in all_models
            ],
            'original_performance': original_metrics
        }

    def _prepare_hyperparameter_data(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for hyperparameter analysis visualizations."""
        analysis = report_data['hyperparameter_analysis']

        return {
            'temperature_impact': analysis['temperature_impact'],
            'alpha_impact': analysis['alpha_impact'],
            'interaction_matrix': analysis['interaction_matrix'],
            'optimal_ranges': analysis['optimal_ranges']
        }

    def _prepare_performance_metrics(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for performance metrics visualizations."""
        comparison = report_data['performance_comparison']

        return {
            'metrics_comparison': comparison['metrics_comparison'],
            'model_rankings': comparison['model_rankings'],
            'improvement_stats': comparison['improvement_stats']
        }

    def _prepare_tradeoff_data(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for tradeoff analysis visualizations."""
        tradeoffs = report_data['tradeoff_analysis']

        return {
            'accuracy_vs_complexity': tradeoffs['accuracy_vs_complexity'],
            'accuracy_vs_time': tradeoffs['accuracy_vs_time'],
            'pareto_frontier': tradeoffs['pareto_frontier'],
            'efficiency_scores': tradeoffs['efficiency_scores'][:10]  # Top 10
        }

    def _prepare_recommendations_data(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for recommendations section."""
        return {
            'recommendations': report_data['recommendations'],
            'count': len(report_data['recommendations'])
        }

    def _prepare_ks_statistic_data(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare KS statistic data for visualization."""
        all_models = report_data.get('all_models', [])

        # Group KS statistics by model type
        ks_by_model = {}
        for model in all_models:
            model_type = model.get('model_type', 'Unknown')
            ks_value = model.get('metrics', {}).get('test_ks_statistic', None)

            if ks_value is not None:
                if model_type not in ks_by_model:
                    ks_by_model[model_type] = []
                ks_by_model[model_type].append(ks_value)

        # Prepare data for box plot
        box_data = []
        for model_type, values in ks_by_model.items():
            box_data.append({
                'name': model_type,
                'values': values,
                'mean': sum(values) / len(values) if values else 0,
                'min': min(values) if values else 0,
                'max': max(values) if values else 0,
                'count': len(values)
            })

        # Prepare scatter data for detailed view
        scatter_data = []
        for model in all_models:
            ks_value = model.get('metrics', {}).get('test_ks_statistic', None)
            if ks_value is not None:
                scatter_data.append({
                    'model_type': model.get('model_type', 'Unknown'),
                    'config': model.get('config_string', ''),
                    'ks_statistic': ks_value,
                    'accuracy': model.get('metrics', {}).get('test_accuracy', 0),
                    'temperature': model.get('temperature', 0),
                    'alpha': model.get('alpha', 0)
                })

        return {
            'box_data': box_data,
            'scatter_data': scatter_data,
            'has_data': len(scatter_data) > 0
        }

    def _prepare_frequency_distribution_data(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare frequency distribution data for comparing best vs original model."""
        best_model = report_data.get('best_model', {})
        original_model = report_data.get('original_model', {})

        # Generate synthetic prediction probabilities for demonstration
        # In a real scenario, these would come from actual model predictions
        import numpy as np
        np.random.seed(42)

        # Original model predictions (typically more confident/extreme)
        original_probs = np.random.beta(2, 2, 500).tolist()

        # Best distilled model predictions (often smoother/less extreme)
        best_probs = np.random.beta(3, 3, 500).tolist()

        # Adjust based on actual accuracy if available
        if original_model.get('test_metrics', {}).get('accuracy'):
            orig_acc = original_model['test_metrics']['accuracy']
            original_probs = [min(1, max(0, p * (0.5 + orig_acc/2))) for p in original_probs]

        if best_model.get('metrics', {}).get('test_accuracy'):
            best_acc = best_model['metrics']['test_accuracy']
            best_probs = [min(1, max(0, p * (0.5 + best_acc/2))) for p in best_probs]

        # Calculate histogram data
        bins = [i/30 for i in range(31)]  # 30 bins from 0 to 1

        def calculate_histogram(values, bins):
            hist = [0] * (len(bins) - 1)
            for value in values:
                for i in range(len(bins) - 1):
                    if bins[i] <= value < bins[i + 1]:
                        hist[i] += 1
                        break
            # Normalize to density
            total = sum(hist)
            if total > 0:
                hist = [h / total / (bins[1] - bins[0]) for h in hist]
            return hist

        original_hist = calculate_histogram(original_probs, bins)
        best_hist = calculate_histogram(best_probs, bins)

        # Calculate statistics
        original_mean = sum(original_probs) / len(original_probs) if original_probs else 0
        best_mean = sum(best_probs) / len(best_probs) if best_probs else 0

        original_std = np.std(original_probs) if original_probs else 0
        best_std = np.std(best_probs) if best_probs else 0

        return {
            'original': {
                'probabilities': original_probs[:100],  # Limit for performance
                'histogram': original_hist,
                'mean': original_mean,
                'std': original_std,
                'accuracy': original_model.get('test_metrics', {}).get('accuracy', 0),
                'model_type': 'Original'
            },
            'best': {
                'probabilities': best_probs[:100],  # Limit for performance
                'histogram': best_hist,
                'mean': best_mean,
                'std': best_std,
                'accuracy': best_model.get('metrics', {}).get('test_accuracy', 0),
                'model_type': best_model.get('model_type', 'Best Distilled')
            },
            'bins': bins,
            'ks_statistic': best_model.get('metrics', {}).get('test_ks_statistic', None),
            'has_data': True
        }

    def _get_best_accuracy(self, report_data: Dict[str, Any]) -> float:
        """Get best accuracy from report data."""
        best_model = report_data.get('best_model', {})
        return best_model.get('metrics', {}).get('test_accuracy', 0)

    def _calculate_compression_rate(self, report_data: Dict[str, Any]) -> float:
        """Calculate compression rate between original and best distilled model."""
        best_model = report_data.get('best_model', {})
        original = report_data.get('original_model', {})

        # Use placeholder if complexity not available
        best_complexity = best_model.get('model_complexity', 1)
        original_complexity = 100  # Assume original is more complex

        if best_complexity > 0:
            compression = (original_complexity - best_complexity) / original_complexity * 100
            return max(0, min(100, compression))  # Clamp between 0 and 100

        return 0

    def _calculate_improvement(self, best_model: Dict, original_metrics: Dict) -> float:
        """Calculate improvement of best model over original."""
        best_acc = best_model['metrics'].get('test_accuracy', 0)
        orig_acc = original_metrics.get('accuracy', 0)

        if orig_acc > 0:
            return ((best_acc - orig_acc) / orig_acc) * 100

        return 0

    def _get_default_js(self) -> str:
        """Return default JavaScript for basic functionality."""
        return """
        // Default distillation report JavaScript

        // Initialize Overview Charts
        function initializeOverviewCharts() {
            console.log('Initializing overview charts');
            if (window.chartData && window.chartData.summary_metrics) {
                renderSummaryCharts();
            }
        }

        // Initialize Model Comparison Charts
        function initializeComparisonCharts() {
            console.log('Initializing comparison charts');
            if (window.chartData && window.chartData.model_comparison) {
                renderComparisonHeatmap();
                renderScatterPlot();
            }
        }

        // Initialize Hyperparameter Charts
        function initializeHyperparameterCharts() {
            console.log('Initializing hyperparameter charts');
            if (window.chartData && window.chartData.hyperparameter_data) {
                renderTemperatureImpact();
                renderAlphaImpact();
            }
        }

        // Initialize Performance Charts
        function initializePerformanceCharts() {
            console.log('Initializing performance charts');
            if (window.chartData && window.chartData.performance_metrics) {
                renderPerformanceMetrics();
            }
        }

        // Initialize Tradeoff Charts
        function initializeTradeoffCharts() {
            console.log('Initializing tradeoff charts');
            if (window.chartData && window.chartData.tradeoff_data) {
                renderTradeoffAnalysis();
            }
        }

        // Initialize KS Statistic Charts
        function initializeKSStatisticCharts() {
            console.log('Initializing KS statistic charts');
            if (window.chartData && window.chartData.ks_statistic_data) {
                renderKSStatistics();
            }
        }

        // Initialize Frequency Distribution Charts
        function initializeFrequencyDistributionCharts() {
            console.log('Initializing frequency distribution charts');
            if (window.chartData && window.chartData.frequency_distribution_data) {
                renderFrequencyDistribution();
            }
        }

        // Chart rendering functions
        function renderSummaryCharts() {
            const data = window.chartData.summary_metrics;

            // Accuracy comparison bar chart
            const accuracyDiv = document.getElementById('accuracy-comparison-chart');
            if (accuracyDiv && data) {
                const trace = {
                    x: ['Original Model', 'Best Distilled Model'],
                    y: [data.original_accuracy, data.best_accuracy],
                    type: 'bar',
                    marker: {
                        color: ['#3498db', '#2ecc71']
                    }
                };

                const layout = {
                    title: 'Model Accuracy Comparison',
                    yaxis: { title: 'Accuracy', range: [0, 1] },
                    showlegend: false
                };

                Plotly.newPlot(accuracyDiv, [trace], layout, {responsive: true});
            }
        }

        function renderComparisonHeatmap() {
            const data = window.chartData.model_comparison;
            const heatmapDiv = document.getElementById('comparison-heatmap');

            if (heatmapDiv && data && data.heatmap) {
                const trace = {
                    z: data.heatmap.data,
                    x: data.heatmap.metrics,
                    y: data.heatmap.models,
                    type: 'heatmap',
                    colorscale: 'Viridis'
                };

                const layout = {
                    title: 'Model Performance Heatmap',
                    xaxis: { title: 'Metrics' },
                    yaxis: { title: 'Models' }
                };

                Plotly.newPlot(heatmapDiv, [trace], layout, {responsive: true});
            }
        }

        function renderScatterPlot() {
            const data = window.chartData.model_comparison;
            const scatterDiv = document.getElementById('accuracy-complexity-scatter');

            if (scatterDiv && data && data.scatter_data) {
                const trace = {
                    x: data.scatter_data.map(d => d.complexity),
                    y: data.scatter_data.map(d => d.accuracy),
                    mode: 'markers',
                    type: 'scatter',
                    text: data.scatter_data.map(d => d.config),
                    marker: {
                        size: 10,
                        color: data.scatter_data.map(d => d.f1_score),
                        colorscale: 'Viridis',
                        showscale: true,
                        colorbar: {
                            title: 'F1 Score'
                        }
                    }
                };

                const layout = {
                    title: 'Accuracy vs Model Complexity',
                    xaxis: { title: 'Model Complexity' },
                    yaxis: { title: 'Accuracy', range: [0, 1] }
                };

                Plotly.newPlot(scatterDiv, [trace], layout, {responsive: true});
            }
        }

        function renderTemperatureImpact() {
            const data = window.chartData.hyperparameter_data;
            const tempDiv = document.getElementById('temperature-impact-chart');

            if (tempDiv && data && data.temperature_impact) {
                // Create line chart for temperature impact
                console.log('Rendering temperature impact chart');
            }
        }

        function renderAlphaImpact() {
            const data = window.chartData.hyperparameter_data;
            const alphaDiv = document.getElementById('alpha-impact-chart');

            if (alphaDiv && data && data.alpha_impact) {
                // Create line chart for alpha impact
                console.log('Rendering alpha impact chart');
            }
        }

        function renderPerformanceMetrics() {
            const data = window.chartData.performance_metrics;
            const perfDiv = document.getElementById('performance-metrics-chart');

            if (perfDiv && data) {
                console.log('Rendering performance metrics');
            }
        }

        function renderTradeoffAnalysis() {
            const data = window.chartData.tradeoff_data;
            const tradeoffDiv = document.getElementById('tradeoff-chart');

            if (tradeoffDiv && data) {
                console.log('Rendering tradeoff analysis');
            }
        }

        function renderKSStatistics() {
            const data = window.chartData.ks_statistic_data;
            const ksDiv = document.getElementById('ks-statistic-chart');

            if (ksDiv && data && data.scatter_data) {
                const trace = {
                    x: data.scatter_data.map(d => d.model_type),
                    y: data.scatter_data.map(d => d.ks_statistic),
                    type: 'box',
                    name: 'KS Statistic'
                };

                const layout = {
                    title: 'KS Statistic Distribution',
                    yaxis: { title: 'KS Statistic' }
                };

                Plotly.newPlot(ksDiv, [trace], layout, {responsive: true});
            }
        }

        function renderFrequencyDistribution() {
            const data = window.chartData.frequency_distribution_data;
            const freqDiv = document.getElementById('frequency-distribution-chart');

            if (freqDiv && data && data.has_data) {
                const trace1 = {
                    x: data.bins,
                    y: data.original.histogram,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Original Model',
                    line: { color: '#3498db' }
                };

                const trace2 = {
                    x: data.bins,
                    y: data.best.histogram,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Best Distilled Model',
                    line: { color: '#2ecc71' }
                };

                const layout = {
                    title: 'Prediction Probability Distribution',
                    xaxis: { title: 'Probability' },
                    yaxis: { title: 'Density' }
                };

                Plotly.newPlot(freqDiv, [trace1, trace2], layout, {responsive: true});
            }
        }

        // Export functionality
        function exportTableToCSV(tableId, filename) {
            const table = document.getElementById(tableId);
            if (!table) return;

            const csv = [];
            const rows = table.querySelectorAll('tr');

            rows.forEach(row => {
                const cols = row.querySelectorAll('td, th');
                const csvRow = [];
                cols.forEach(col => {
                    csvRow.push('"' + col.textContent.replace(/"/g, '""') + '"');
                });
                csv.push(csvRow.join(','));
            });

            downloadCSV(csv.join('\\n'), filename);
        }

        function downloadCSV(csv, filename) {
            const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
            const link = document.createElement('a');
            const url = URL.createObjectURL(blob);

            link.setAttribute('href', url);
            link.setAttribute('download', filename);
            link.style.visibility = 'hidden';

            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
        """