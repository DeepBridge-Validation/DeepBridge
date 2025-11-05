"""
Static robustness report renderer that uses Seaborn for visualizations.
"""

import os
import logging
from typing import Dict, List, Any, Optional

# Configure logger
logger = logging.getLogger("deepbridge.reports")

class StaticRobustnessRenderer:
    """
    Renderer for static robustness test reports using Seaborn charts.
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
        from ...transformers.initial_results import InitialResultsTransformer
        self.data_transformer = RobustnessDataTransformer()
        self.initial_results_transformer = InitialResultsTransformer()
        
        # Import Seaborn chart utilities
        from ...utils.seaborn_utils import SeabornChartGenerator
        self.chart_generator = SeabornChartGenerator()
    
    def render(self, results: Dict[str, Any], file_path: str, model_name: str = "Model", report_type: str = "static", save_chart: bool = False) -> str:
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
            Whether to save charts as separate PNG files (default: False)
            
        Returns:
        --------
        str : Path to the generated report
        
        Raises:
        -------
        FileNotFoundError: If template or assets not found
        ValueError: If required data missing
        """
        logger.info(f"Generating static robustness report to: {file_path}")
        
        try:
            # Find template
            template_paths = self.template_manager.get_template_paths("robustness", "static")
            template_path = self.template_manager.find_template(template_paths)
            
            if not template_path:
                raise FileNotFoundError(f"No static template found for robustness report in: {template_paths}")
            
            logger.info(f"Using static template: {template_path}")
            
            # Get CSS content using CSSManager (via base_renderer)
            css_content = self.base_renderer._load_static_css_content('robustness')
            
            # Load the template
            template = self.template_manager.load_template(template_path)
            
            # Transform the robustness data
            # First use the standard transformer
            report_data = self.data_transformer.transform(results, model_name)

            # Then apply additional transformations for static reports
            try:
                from ...transformers.static import StaticRobustnessTransformer
                static_transformer = StaticRobustnessTransformer()
                report_data = static_transformer.transform(results, model_name)
                logger.info("Applied static transformations to report data")
            except ImportError:
                logger.warning("Static transformer not available, using standard transformations")
            
            # Transform initial results data if available
            initial_results = {}
            if 'initial_results' in results:
                logger.info("Found initial_results in results, transforming...")
                initial_results = self.initial_results_transformer.transform(results.get('initial_results', {}))
                logger.info(f"Initial results transformed for {len(initial_results.get('models', {}))} models")
                report_data['initial_results'] = initial_results
            
            # Create the context for the template
            context = self.base_renderer._create_static_context(report_data, "robustness", css_content)

            # Add logo and favicon to context
            try:
                logo_base64 = self.asset_manager.get_logo_base64()
                favicon_base64 = self.asset_manager.get_favicon_base64()
                context['logo'] = logo_base64
                context['favicon'] = favicon_base64
            except Exception as e:
                logger.warning(f"Error loading logo/favicon: {str(e)}")
                context['logo'] = ""
                context['favicon'] = ""

            # Generate charts for the static report
            try:
                charts = self._generate_charts(report_data)
                context['charts'] = charts
            except Exception as chart_error:
                logger.error(f"Error generating charts: {str(chart_error)}")
                context['charts'] = {}  # Empty charts to prevent template errors
            
            # Add robustness-specific context with default values
            robustness_score = report_data.get('robustness_score', 0)

            # Extrair a lista completa de features do objeto de dados
            all_features = []

            # Tenta obter features diretamente do objeto
            if 'features' in report_data and isinstance(report_data['features'], list):
                all_features = report_data['features']
            # Se não encontrar, tenta buscar em feature_importance
            elif 'feature_importance' in report_data and isinstance(report_data['feature_importance'], dict):
                all_features = list(report_data['feature_importance'].keys())
            # Se não encontrar, tenta buscar no modelo primário
            elif 'primary_model' in report_data:
                if 'features' in report_data['primary_model'] and isinstance(report_data['primary_model']['features'], list):
                    all_features = report_data['primary_model']['features']
                elif 'feature_importance' in report_data['primary_model'] and isinstance(report_data['primary_model']['feature_importance'], dict):
                    all_features = list(report_data['primary_model']['feature_importance'].keys())

            logger.info(f"Extracted {len(all_features)} features for display")

            # Extrair métricas para a tabela de comparação
            metrics = report_data.get('metrics', {})
            metrics_details = report_data.get('metrics_details', {})

            # Verificar se temos métricas em outro caminho no dados
            if not metrics and 'primary_model' in report_data:
                if 'metrics' in report_data['primary_model']:
                    metrics = report_data['primary_model']['metrics']
                    logger.info(f"Using metrics from primary_model with keys: {list(metrics.keys())}")

            # Log de métricas
            logger.info(f"Available metrics: {list(metrics.keys()) if isinstance(metrics, dict) else 'None'}")
            logger.info(f"Available metrics_details: {list(metrics_details.keys()) if isinstance(metrics_details, dict) else 'None'}")

            # Garantir que métricas de modelos alternativos também estejam disponíveis
            if 'alternative_models' in report_data:
                for model_name, model_data in report_data['alternative_models'].items():
                    if 'metrics' in model_data:
                        logger.info(f"Model {model_name} has metrics: {list(model_data['metrics'].keys()) if isinstance(model_data['metrics'], dict) else 'None'}")

            context.update({
                # Core metrics with defaults - ensure numeric values
                'robustness_score': float(robustness_score) if robustness_score is not None else 0.0,
                'resilience_score': float(robustness_score) if robustness_score is not None else 0.0,  # Backward compatibility
                'raw_impact': float(report_data.get('raw_impact', 0)) if report_data.get('raw_impact') is not None else 0.0,
                'quantile_impact': float(report_data.get('quantile_impact', 0)) if report_data.get('quantile_impact') is not None else 0.0,
                'base_score': float(report_data.get('base_score', 0)) if report_data.get('base_score') is not None else 0.0,

                # Feature importance data
                'feature_importance': report_data.get('feature_importance', {}),
                'model_feature_importance': report_data.get('model_feature_importance', {}),
                'has_feature_importance': bool(report_data.get('feature_importance', {})),
                'has_model_feature_importance': bool(report_data.get('model_feature_importance', {})),

                # Test metadata
                'iterations': report_data.get('n_iterations', 1),
                'test_type': 'robustness',

                # Additional context
                'features': all_features,
                'metrics': metrics,
                'metrics_details': metrics_details,
                'feature_subset': report_data.get('feature_subset', [])
            })
            
            # Render the template with robust error handling
            try:
                rendered_html = self.template_manager.render_template(template, context)
            except Exception as template_error:
                logger.error(f"Error rendering template: {str(template_error)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                # Generate a simplified fallback report
                rendered_html = self._generate_fallback_report(report_data, model_name)

            # Write the report to the file
            return self.base_renderer._write_report(rendered_html, file_path)
            
        except Exception as e:
            logger.error(f"Error generating static robustness report: {str(e)}")
            try:
                # Last resort fallback
                fallback_html = self._generate_fallback_report(results, model_name)
                return self.base_renderer._write_report(fallback_html, file_path)
            except Exception as fallback_error:
                logger.error(f"Fallback report generation failed: {str(fallback_error)}")
                raise ValueError(f"Failed to generate static robustness report: {str(e)}")
    
    def _generate_fallback_report(self, data: Dict[str, Any], model_name: str) -> str:
        """
        Generate a simple fallback report when template rendering fails.

        Parameters:
        -----------
        data : Dict[str, Any]
            Report data (may be raw or transformed)
        model_name : str
            Name of the model

        Returns:
        --------
        str : Simple HTML report content
        """
        # Extract essential data with safe defaults
        robustness_score = 0.0
        base_score = 0.0
        raw_impact = 0.0

        # Try to extract from transformed data
        if isinstance(data, dict):
            robustness_score = data.get('robustness_score', 0.0)
            if not isinstance(robustness_score, (int, float)):
                robustness_score = 0.0

            base_score = data.get('base_score', 0.0)
            if not isinstance(base_score, (int, float)):
                base_score = 0.0

            raw_impact = data.get('raw_impact', 0.0)
            if not isinstance(raw_impact, (int, float)):
                raw_impact = 0.0

            # Try to extract from raw data structure
            if 'primary_model' in data:
                primary_model = data['primary_model']
                if isinstance(primary_model, dict):
                    if 'base_score' in primary_model and isinstance(primary_model['base_score'], (int, float)):
                        base_score = primary_model['base_score']
                    if 'robustness_score' in primary_model and isinstance(primary_model['robustness_score'], (int, float)):
                        robustness_score = primary_model['robustness_score']

        # Generate simple HTML
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Robustness Report: {model_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 800px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1, h2 {{ color: #333; }}
        .metrics {{ display: flex; flex-wrap: wrap; margin: 20px 0; }}
        .metric-card {{ flex: 1 1 200px; margin: 10px; padding: 15px; background-color: #f9f9f9; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #1b78de; margin: 10px 0; }}
        .footer {{ margin-top: 30px; text-align: center; color: #666; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Robustness Analysis: {model_name}</h1>
        <p>This is a simplified robustness report. Template rendering failed for the full report.</p>

        <div class="metrics">
            <div class="metric-card">
                <h2>Robustness Score</h2>
                <div class="metric-value">{robustness_score:.4f}</div>
                <p>Higher is better</p>
            </div>

            <div class="metric-card">
                <h2>Base Score</h2>
                <div class="metric-value">{base_score:.4f}</div>
                <p>Without perturbation</p>
            </div>

            <div class="metric-card">
                <h2>Average Impact</h2>
                <div class="metric-value">{raw_impact:.4f}</div>
                <p>Lower is better</p>
            </div>
        </div>

        <div class="footer">
            <p>Generated by DeepBridge</p>
        </div>
    </div>
</body>
</html>"""

        return html

    def _generate_charts(self, report_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate all charts needed for the static report.
        
        Parameters:
        -----------
        report_data : Dict[str, Any]
            Transformed report data
            
        Returns:
        --------
        Dict[str, str] : Dictionary of chart names and their base64 encoded images
        """
        charts = {}
        
        try:
            # Generate robustness overview chart (performance by perturbation level)
            perturbation_levels = []
            perturbed_scores = []
            feature_subset_scores = []
            worst_scores = []  # For worst performance chart
            feature_subset_worst_scores = []  # For worst performance chart
            has_feature_subset = False

            if 'raw' in report_data and 'by_level' in report_data['raw']:
                raw_data = report_data['raw']['by_level']

                # Sort levels numerically
                levels = sorted([float(level) for level in raw_data.keys()])
                perturbation_levels = levels

                # Extract scores for each level
                max_feature_subset_impact = 0
                max_feature_subset_impact_level = None

                for level in levels:
                    level_str = str(level)
                    if level_str in raw_data and 'overall_result' in raw_data[level_str]:
                        result = raw_data[level_str]['overall_result']

                        # All features scores - mean scores for average performance
                        if 'all_features' in result:
                            perturbed_scores.append(result['all_features'].get('mean_score', None))
                            # Also extract worst score for each level
                            worst_scores.append(result['all_features'].get('min_score', result['all_features'].get('worst_score', None)))
                        else:
                            perturbed_scores.append(None)
                            worst_scores.append(None)

                        # Feature subset scores
                        if 'feature_subset' in result:
                            feature_subset_scores.append(result['feature_subset'].get('mean_score', None))
                            # Also extract worst scores for feature subset
                            feature_subset_worst_scores.append(result['feature_subset'].get('min_score', result['feature_subset'].get('worst_score', None)))
                            has_feature_subset = True

                            # Track max feature subset impact
                            feature_subset_impact = result['feature_subset'].get('impact', 0)
                            if feature_subset_impact > max_feature_subset_impact:
                                max_feature_subset_impact = feature_subset_impact
                                max_feature_subset_impact_level = level_str

                            logger.info(f"Level {level_str} feature subset impact: {feature_subset_impact}")
                        else:
                            feature_subset_scores.append(None)
                            feature_subset_worst_scores.append(None)

                # Log max feature subset impact
                if has_feature_subset:
                    logger.info(f"Max feature subset impact: {max_feature_subset_impact} at level {max_feature_subset_impact_level}")
                    # Add to report data for easier access in template
                    if 'feature_subset_max_impact' not in report_data:
                        report_data['feature_subset_max_impact'] = {
                            'value': max_feature_subset_impact,
                            'level': max_feature_subset_impact_level
                        }

            if perturbation_levels and perturbed_scores:
                # Only include feature subset if we have data
                subset_data = feature_subset_scores if has_feature_subset else None

                # Generate average performance chart
                charts['overview_chart'] = self.chart_generator.robustness_overview_chart(
                    perturbation_levels=perturbation_levels,
                    scores=perturbed_scores,
                    base_score=report_data.get('base_score', 0),
                    metric_name=report_data.get('metric', 'Score'),
                    feature_subset_scores=subset_data
                )
                logger.info("Generated robustness overview chart")

                # Generate worst performance chart if we have worst_scores
                if worst_scores and any(score is not None for score in worst_scores):
                    subset_worst_data = feature_subset_worst_scores if has_feature_subset else None
                    
                    charts['worst_performance_chart'] = self.chart_generator.worst_performance_chart(
                        perturbation_levels=perturbation_levels,
                        worst_scores=worst_scores,
                        base_score=report_data.get('base_score', 0),
                        metric_name=report_data.get('metric', 'Score'),
                        feature_subset_worst_scores=subset_worst_data
                    )
                    logger.info("Generated worst performance chart")
            
            # Generate model comparison chart if alternative models exist
            if 'alternative_models' in report_data and report_data['alternative_models']:
                # Prepare data for all models
                models_data = {}
                
                # Primary model
                primary_model_scores = perturbed_scores
                
                if primary_model_scores:
                    models_data[report_data.get('model_name', 'Primary Model')] = {
                        'scores': primary_model_scores,
                        'base_score': report_data.get('base_score', 0)
                    }
                
                # Alternative models
                for model_name, model_data in report_data['alternative_models'].items():
                    model_scores = []
                    
                    if 'raw' in model_data and 'by_level' in model_data['raw']:
                        alt_raw_data = model_data['raw']['by_level']
                        
                        # Extract scores for each level using the same perturbation levels
                        for level in perturbation_levels:
                            level_str = str(level)
                            if level_str in alt_raw_data and 'overall_result' in alt_raw_data[level_str]:
                                result = alt_raw_data[level_str]['overall_result']
                                
                                # All features scores
                                if 'all_features' in result:
                                    model_scores.append(result['all_features'].get('mean_score', None))
                                else:
                                    model_scores.append(None)
                            else:
                                model_scores.append(None)
                    
                    if model_scores:
                        models_data[model_name] = {
                            'scores': model_scores,
                            'base_score': model_data.get('base_score', 0)
                        }
                
                if len(models_data) > 1:
                    charts['comparison_chart'] = self.chart_generator.model_comparison_chart(
                        perturbation_levels=perturbation_levels,
                        models_data=models_data,
                        metric_name=report_data.get('metric', 'Score')
                    )
                    logger.info(f"Generated model comparison chart with {len(models_data)} models")
            
            # Generate feature importance chart
            feature_importance = report_data.get('feature_importance', {})
            if feature_importance:
                charts['feature_importance_chart'] = self.chart_generator.feature_importance_chart(
                    features=feature_importance,
                    title="Feature Importance by Robustness Impact"
                )
                logger.info(f"Generated feature importance chart with {len(feature_importance)} features")
            
            # Generate feature comparison chart if both types of importance exist
            model_feature_importance = report_data.get('model_feature_importance', {})
            if feature_importance and model_feature_importance:
                charts['feature_comparison_chart'] = self.chart_generator.feature_comparison_chart(
                    model_importance=model_feature_importance,
                    robustness_importance=feature_importance,
                    title="Model vs. Robustness Feature Importance"
                )
                logger.info("Generated feature comparison chart")
            
            # Generate boxplot chart for score distributions
            boxplot_models = []
            
            # Primary model
            primary_scores = []
            if 'iterations_by_level' in report_data:
                for level_scores in report_data['iterations_by_level'].values():
                    if isinstance(level_scores, list) and level_scores:
                        primary_scores.extend(level_scores)
            
            if primary_scores:
                boxplot_models.append({
                    'name': report_data.get('model_name', 'Primary Model'),
                    'scores': primary_scores,
                    'baseScore': report_data.get('base_score', 0)
                })
            
            # Alternative models
            if 'alternative_models' in report_data and 'alt_iterations_by_level' in report_data:
                for model_name, model_data in report_data['alternative_models'].items():
                    if model_name in report_data['alt_iterations_by_level']:
                        alt_scores = []
                        for level_scores in report_data['alt_iterations_by_level'][model_name].values():
                            if isinstance(level_scores, list) and level_scores:
                                alt_scores.extend(level_scores)
                        
                        if alt_scores:
                            boxplot_models.append({
                                'name': model_name,
                                'scores': alt_scores,
                                'baseScore': model_data.get('base_score', 0)
                            })
            
            # Also check for scores in initial_results
            if 'initial_results' in report_data and 'models' in report_data['initial_results']:
                for model_id, model_data in report_data['initial_results']['models'].items():
                    # Skip if already in boxplot
                    model_name = model_data.get('name', model_id)
                    if any(m['name'] == model_name for m in boxplot_models):
                        continue
                    
                    # Get scores if available
                    if ('evaluation_results' in model_data and 
                        'scores' in model_data['evaluation_results'] and 
                        model_data['evaluation_results']['scores']):
                        
                        boxplot_models.append({
                            'name': model_name,
                            'scores': model_data['evaluation_results']['scores'],
                            'baseScore': model_data.get('base_score', 0)
                        })
            
            if boxplot_models:
                charts['boxplot_chart'] = self.chart_generator.boxplot_chart(
                    models_data=boxplot_models,
                    metric_name=report_data.get('metric', 'Score')
                )
                logger.info(f"Generated boxplot chart with {len(boxplot_models)} models")
            
        except Exception as e:
            logger.error(f"Error generating charts: {str(e)}")
        
        return charts