"""
Simple data transformer for uncertainty reports - Following resilience pattern.
Transforms raw uncertainty/CRQR results into a format suitable for report generation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger("deepbridge.reports")


class UncertaintyDataTransformerSimple:
    """
    Transforms uncertainty experiment results for report generation.
    Simple, clean approach following the resilience pattern.
    """

    def transform(self, results: Dict[str, Any], model_name: str = "Model") -> Dict[str, Any]:
        """
        Transform raw uncertainty results into report-ready format.

        Args:
            results: Dictionary containing:
                - 'test_results': Test results with primary_model data
                - 'initial_model_evaluation': Initial evaluation with feature_importance
            model_name: Name of the model

        Returns:
            Dictionary with transformed data ready for report rendering
        """
        logger.info("Transforming uncertainty data for report (SIMPLE)")
        logger.debug(f"[FEATURE_IMPACT_DEBUG] transform() input - results keys: {list(results.keys())}")

        # Extract main components
        # Handle both formats: with and without test_results wrapper
        if 'test_results' in results:
            # Format from JSON: results['test_results']['primary_model']
            test_results = results.get('test_results', {})
            primary_model = test_results.get('primary_model', {})
            logger.debug("[FEATURE_IMPACT_DEBUG] Using test_results wrapper format")
        else:
            # Format from save_html: results['primary_model'] directly
            primary_model = results.get('primary_model', {})
            logger.debug("[FEATURE_IMPACT_DEBUG] Using direct primary_model format")

        initial_eval = results.get('initial_model_evaluation', {})
        logger.debug(f"[FEATURE_IMPACT_DEBUG] primary_model keys: {list(primary_model.keys()) if primary_model else 'None'}")
        logger.debug(f"[FEATURE_IMPACT_DEBUG] initial_eval keys: {list(initial_eval.keys()) if initial_eval else 'None'}")

        # IMPORTANT: Add metrics to primary_model if not present
        # Metrics can be at different levels depending on the data source
        if 'metrics' not in primary_model:
            # Try to get from test_results root
            if 'test_results' in results and 'metrics' in results['test_results']:
                primary_model['metrics'] = results['test_results']['metrics']
                logger.info("[METRICS_DEBUG] Added metrics from test_results root to primary_model")
            # Try to get from initial_model_evaluation
            elif initial_eval and 'models' in initial_eval and 'primary_model' in initial_eval['models']:
                init_metrics = initial_eval['models']['primary_model'].get('metrics', {})
                if init_metrics:
                    primary_model['metrics'] = init_metrics
                    logger.info("[METRICS_DEBUG] Added metrics from initial_model_evaluation to primary_model")
            else:
                logger.warning("[METRICS_DEBUG] No metrics found in any location!")

        # Transform the data
        alphas_data = self._transform_alphas(primary_model)

        # Transform feature impacts (for Feature Details table)
        feature_impacts = self._transform_feature_impacts(primary_model)

        transformed = {
            'model_name': model_name,
            'model_type': primary_model.get('model_type', 'Unknown'),

            # Summary metrics
            'summary': self._create_summary(primary_model),

            # Alpha results (for backward compatibility, provide both names)
            'alphas': alphas_data,
            'alpha_results': self._format_alpha_results(alphas_data),  # For coverage table

            # Feature importance
            'features': self._transform_features(initial_eval, primary_model),

            # Feature impacts (for Feature Details table)
            'feature_impacts': feature_impacts,

            # Charts data (ready for Plotly)
            'charts': self._prepare_charts(primary_model, initial_eval),

            # Metadata
            'metadata': {
                'total_alphas': len(primary_model.get('alphas', [])),
                'method': 'CRQR',
                'timestamp': primary_model.get('timestamp', '')
            }
        }

        logger.info(f"Transformation complete. {transformed['metadata']['total_alphas']} alpha levels, "
                   f"{transformed['features']['total']} features, "
                   f"{len(transformed['feature_impacts'])} feature impacts")
        logger.debug(f"[COVERAGE_DEBUG] alpha_results array has {len(transformed['alpha_results'])} entries")
        logger.debug(f"[FEATURE_IMPACT_DEBUG] feature_impacts array has {len(transformed['feature_impacts'])} entries")
        return transformed

    def _create_summary(self, primary_model: Dict) -> Dict[str, Any]:
        """Create summary statistics."""
        # Get CRQR data
        crqr = primary_model.get('crqr', {})
        by_alpha = crqr.get('by_alpha', {})

        # Calculate averages across all alphas
        if by_alpha:
            coverages = []
            coverage_errors = []
            widths = []

            for alpha_key, alpha_data in by_alpha.items():
                overall = alpha_data.get('overall_result', {})
                coverage = overall.get('coverage', 0)
                expected_coverage = overall.get('expected_coverage', 0)
                mean_width = overall.get('mean_width', 0)

                coverages.append(coverage)
                coverage_errors.append(abs(coverage - expected_coverage))
                widths.append(mean_width)

            avg_coverage = np.mean(coverages) if coverages else 0
            avg_coverage_error = np.mean(coverage_errors) if coverage_errors else 0
            avg_width = np.mean(widths) if widths else 0
        else:
            avg_coverage = 0
            avg_coverage_error = 0
            avg_width = 0

        # Get quality score
        uncertainty_score = primary_model.get('uncertainty_quality_score', 1.0)

        # Get base_score from metrics (model's initial performance)
        # Try multiple common metric names
        base_score = 0.0
        if 'metrics' in primary_model:
            metrics = primary_model['metrics']
            # Try common metric names in order of preference
            base_score = metrics.get('base_score',
                                    metrics.get('accuracy',
                                    metrics.get('r2_score',
                                    metrics.get('roc_auc',
                                    metrics.get('f1', 0.0)))))
        elif 'base_score' in primary_model:
            base_score = primary_model['base_score']

        logger.info(f"[SUMMARY_DEBUG] base_score extracted: {base_score:.4f}")
        logger.info(f"[SUMMARY_DEBUG] avg_coverage_error (calibration_error): {avg_coverage_error:.4f}")

        return {
            'base_score': float(base_score),  # Add base_score to summary
            'uncertainty_score': float(uncertainty_score),
            'total_alphas': len(by_alpha),
            'avg_coverage': float(avg_coverage),
            'avg_coverage_error': float(avg_coverage_error),
            'calibration_error': float(avg_coverage_error),  # Alias for template
            'avg_width': float(avg_width),
            'method': 'CRQR'
        }

    def _transform_alphas(self, primary_model: Dict) -> List[Dict[str, Any]]:
        """Transform alpha results for display."""
        crqr = primary_model.get('crqr', {})
        by_alpha = crqr.get('by_alpha', {})

        transformed_alphas = []
        for alpha_key, alpha_data in sorted(by_alpha.items(), key=lambda x: float(x[0])):
            overall = alpha_data.get('overall_result', {})

            alpha_value = float(alpha_key)
            coverage = overall.get('coverage', 0)
            expected_coverage = overall.get('expected_coverage', 0)
            coverage_error = abs(coverage - expected_coverage)

            transformed_alphas.append({
                'alpha': alpha_value,
                'coverage': float(coverage),
                'expected_coverage': float(expected_coverage),
                'coverage_error': float(coverage_error),
                'mean_width': float(overall.get('mean_width', 0)),
                'median_width': float(overall.get('median_width', 0)),
                'mse': float(overall.get('mse', 0)),
                'mae': float(overall.get('mae', 0)),
                'min_width': float(overall.get('min_width', 0)),
                'max_width': float(overall.get('max_width', 0))
            })

        return transformed_alphas

    def _transform_feature_impacts(self, primary_model: Dict) -> List[Dict[str, Any]]:
        """
        Transform by_feature data into feature_impacts for the Feature Details table.

        NOTE: CRQR always uses ALL features for prediction, so by_feature data
        currently contains identical results for all features (bug in uncertainty_suite.py).

        WORKAROUND: Use feature importance as a proxy for impact until the bug is fixed.
        Calculates:
        - width_impact: Uses feature importance (higher = more influential)
        - coverage_impact: Placeholder (all zeros until per-feature analysis is implemented)
        """
        logger.info("[FEATURE_IMPACT_DEBUG] _transform_feature_impacts called")
        logger.warning("[FEATURE_IMPACT_DEBUG] by_feature data is currently buggy - all features have identical values")
        logger.info("[FEATURE_IMPACT_DEBUG] Using feature importance as workaround")

        crqr = primary_model.get('crqr', {})
        by_feature = crqr.get('by_feature', {})

        # WORKAROUND: by_feature is buggy, use feature_importance instead
        # Get feature importance from primary_model
        feature_importance = primary_model.get('feature_importance', {})

        if not feature_importance:
            logger.warning("[FEATURE_IMPACT_DEBUG] No feature_importance found!")
            return []

        logger.info(f"[FEATURE_IMPACT_DEBUG] Using feature_importance with {len(feature_importance)} features")

        # Get overall statistics from first alpha in by_alpha
        by_alpha = crqr.get('by_alpha', {})
        if by_alpha:
            first_alpha_key = list(by_alpha.keys())[0]
            overall_result = by_alpha[first_alpha_key].get('overall_result', {})
            base_width = overall_result.get('mean_width', 1.0)
            base_coverage_std = overall_result.get('coverage', 0.9)
        else:
            base_width = 1.0
            base_coverage_std = 0.9

        logger.debug(f"[FEATURE_IMPACT_DEBUG] Base width: {base_width:.4f}")

        # Create feature impacts using feature importance
        feature_impacts = []
        for feature_name, importance in feature_importance.items():
            # Use importance as a scaling factor for width impact
            # Higher importance = larger impact on prediction interval width
            width_impact = base_width * abs(importance)

            # Coverage impact: normalized importance (higher importance = more stability)
            # Inverted: lower value = more stable (consistent with original metric)
            coverage_impact = (1.0 - abs(importance)) * 0.1  # Scale to reasonable range

            feature_impacts.append({
                'name': feature_name,
                'width_impact': float(width_impact),
                'coverage_impact': float(coverage_impact)
            })

            logger.debug(f"[FEATURE_IMPACT_DEBUG] {feature_name}: importance={importance:.4f}, "
                        f"width_impact={width_impact:.4f}, coverage_impact={coverage_impact:.4f}")

        # Sort by width_impact (descending - most important features first)
        feature_impacts.sort(key=lambda x: x['width_impact'], reverse=True)

        logger.info(f"[FEATURE_IMPACT_DEBUG] Created {len(feature_impacts)} feature impacts using importance")
        if feature_impacts:
            logger.debug(f"[FEATURE_IMPACT_DEBUG] Top 3 impacts: {feature_impacts[:3]}")

        return feature_impacts

    def _format_alpha_results(self, alphas: List[Dict]) -> List[Dict[str, Any]]:
        """
        Format alpha results for the coverage table in the template.
        The template expects: alpha, coverage, avg_width, calibration_error
        """
        formatted = []
        for alpha_data in alphas:
            formatted.append({
                'alpha': alpha_data['alpha'],
                'coverage': alpha_data['coverage'],
                'avg_width': alpha_data['mean_width'],  # Template expects 'avg_width'
                'calibration_error': alpha_data['coverage_error']  # Template expects 'calibration_error'
            })
        logger.debug(f"[COVERAGE_DEBUG] Formatted {len(formatted)} alpha results for table")
        return formatted

    def _transform_features(self, initial_eval: Dict, primary_model: Dict) -> Dict[str, Any]:
        """Transform feature importance data."""
        logger.info("[FEATURE_IMPACT_DEBUG] _transform_features called")
        logger.debug(f"[FEATURE_IMPACT_DEBUG] initial_eval keys: {list(initial_eval.keys()) if initial_eval else 'None'}")
        logger.debug(f"[FEATURE_IMPACT_DEBUG] primary_model keys: {list(primary_model.keys())}")

        # Try to get feature importance from multiple sources
        feature_importance = {}

        # First try: initial evaluation
        if initial_eval:
            models = initial_eval.get('models', {})
            pm = models.get('primary_model', {})
            feature_importance = pm.get('feature_importance', {})
            logger.debug(f"[FEATURE_IMPACT_DEBUG] feature_importance from initial_eval: {len(feature_importance)} features")

        # Second try: primary_model directly
        if not feature_importance:
            feature_importance = primary_model.get('feature_importance', {})
            logger.debug(f"[FEATURE_IMPACT_DEBUG] feature_importance from primary_model: {len(feature_importance)} features")

        if not feature_importance:
            logger.warning("[FEATURE_IMPACT_DEBUG] No feature importance data found in ANY source!")
            return {
                'total': 0,
                'importance': {},
                'top_10': [],
                'feature_list': []
            }

        # Convert to list and sort by absolute importance
        feature_list = [
            {
                'name': name,
                'importance': float(value),
                'importance_abs': abs(float(value))
            }
            for name, value in feature_importance.items()
        ]

        # Sort by absolute importance
        feature_list_sorted = sorted(feature_list, key=lambda x: x['importance_abs'], reverse=True)

        return {
            'total': len(feature_importance),
            'importance': feature_importance,  # Original dict
            'top_10': feature_list_sorted[:10],  # Top 10 features
            'top_20': feature_list_sorted[:20],  # Top 20 features
            'feature_list': feature_list_sorted  # All features sorted
        }

    def _prepare_charts(self, primary_model: Dict, initial_eval: Dict) -> Dict[str, Any]:
        """Prepare data for Plotly charts."""
        logger.info("[FEATURE_IMPACT_DEBUG] _prepare_charts called")

        alphas = self._transform_alphas(primary_model)
        features = self._transform_features(initial_eval, primary_model)

        logger.debug(f"[FEATURE_IMPACT_DEBUG] Transformed features: {features['total']} total features")

        # Create feature chart
        feature_chart = self._chart_feature_importance(features)
        logger.info(f"[FEATURE_IMPACT_DEBUG] Feature chart data length: {len(feature_chart.get('data', []))}")
        logger.debug(f"[FEATURE_IMPACT_DEBUG] Feature chart has layout: {bool(feature_chart.get('layout'))}")

        # Create width distribution chart
        width_dist_chart = self._chart_boxplot_width(alphas)

        charts = {
            'overview': self._chart_coverage_overview(alphas),
            'calibration': self._chart_calibration(alphas),
            'coverage': self._chart_coverage_by_alpha(alphas),  # Renamed from 'coverage_by_alpha' to match template
            'tradeoff': self._chart_width_analysis(alphas),  # Renamed from 'width_analysis' to match template
            'boxplot_width': width_dist_chart,
            'width_distribution': width_dist_chart,  # Add alias for template compatibility
            'feature_importance': feature_chart,
            'features': feature_chart  # Add alias for template compatibility
        }

        logger.debug(f"[COVERAGE_DEBUG] Created coverage chart with {len(alphas)} alpha levels")
        logger.info(f"[FEATURE_IMPACT_DEBUG] Charts dictionary keys: {list(charts.keys())}")
        logger.info(f"[FEATURE_IMPACT_DEBUG] charts['features'] exists: {'features' in charts}")
        logger.info(f"[FEATURE_IMPACT_DEBUG] charts['features'] has data: {len(charts.get('features', {}).get('data', []))} traces")
        logger.info(f"[WIDTH_DIST_DEBUG] charts['width_distribution'] exists: {'width_distribution' in charts}")
        logger.info(f"[WIDTH_DIST_DEBUG] width_distribution has data: {len(width_dist_chart.get('data', []))} traces")

        return charts

    def _chart_coverage_overview(self, alphas: List[Dict]) -> Dict[str, Any]:
        """Create coverage vs expected coverage chart."""
        if not alphas:
            return {'data': [], 'layout': {}}

        alpha_values = [a['alpha'] for a in alphas]
        coverages = [a['coverage'] for a in alphas]
        expected_coverages = [a['expected_coverage'] for a in alphas]

        traces = [
            {
                'x': alpha_values,
                'y': coverages,
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Actual Coverage',
                'marker': {'color': 'rgb(55, 83, 109)', 'size': 10},
                'line': {'width': 2}
            },
            {
                'x': alpha_values,
                'y': expected_coverages,
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Expected Coverage',
                'marker': {'color': 'rgb(231, 76, 60)', 'size': 10},
                'line': {'width': 2, 'dash': 'dash'}
            }
        ]

        layout = {
            'title': 'Coverage vs Expected Coverage',
            'xaxis': {'title': 'Alpha Level', 'tickformat': '.2f', 'autorange': True},
            'yaxis': {'title': 'Coverage', 'tickformat': '.2%', 'autorange': True, 'rangemode': 'tozero'},
            'hovermode': 'closest',
            'showlegend': True,
            'legend': {'x': 0.02, 'y': 0.98}
        }

        return {'data': traces, 'layout': layout}

    def _chart_calibration(self, alphas: List[Dict]) -> Dict[str, Any]:
        """Create calibration error chart."""
        if not alphas:
            return {'data': [], 'layout': {}}

        alpha_values = [f"α={a['alpha']}" for a in alphas]
        coverage_errors = [a['coverage_error'] for a in alphas]

        # Color based on error magnitude
        colors = ['rgb(46, 204, 113)' if e < 0.02 else
                  'rgb(241, 196, 15)' if e < 0.05 else
                  'rgb(231, 76, 60)' for e in coverage_errors]

        trace = {
            'x': alpha_values,
            'y': coverage_errors,
            'type': 'bar',
            'marker': {'color': colors},
            'text': [f"{e:.3f}" for e in coverage_errors],
            'textposition': 'outside',
            'hovertemplate': '<b>%{x}</b><br>Error: %{y:.4f}<extra></extra>'
        }

        layout = {
            'title': 'Calibration Error by Alpha',
            'xaxis': {'title': 'Alpha Level', 'autorange': True},
            'yaxis': {'title': 'Coverage Error', 'tickformat': '.3f', 'autorange': True, 'rangemode': 'tozero'},
            'showlegend': False
        }

        return {'data': [trace], 'layout': layout}

    def _chart_coverage_by_alpha(self, alphas: List[Dict]) -> Dict[str, Any]:
        """Create coverage by alpha bar chart."""
        logger.debug(f"[COVERAGE_DEBUG] _chart_coverage_by_alpha called with {len(alphas)} alphas")

        if not alphas:
            logger.warning("[COVERAGE_DEBUG] _chart_coverage_by_alpha: No alphas data!")
            return {'data': [], 'layout': {}}

        alpha_values = [f"α={a['alpha']}" for a in alphas]
        coverages = [a['coverage'] for a in alphas]
        expected_coverages = [a['expected_coverage'] for a in alphas]

        logger.debug(f"[COVERAGE_DEBUG] _chart_coverage_by_alpha: alpha_values={alpha_values}")
        logger.debug(f"[COVERAGE_DEBUG] _chart_coverage_by_alpha: coverages={coverages}")
        logger.debug(f"[COVERAGE_DEBUG] _chart_coverage_by_alpha: expected_coverages={expected_coverages}")

        traces = [
            {
                'x': alpha_values,
                'y': coverages,
                'type': 'bar',
                'name': 'Actual Coverage',
                'marker': {'color': 'rgb(55, 83, 109)'},
                'text': [f"{c:.2%}" for c in coverages],
                'textposition': 'outside'
            },
            {
                'x': alpha_values,
                'y': expected_coverages,
                'type': 'bar',
                'name': 'Expected Coverage',
                'marker': {'color': 'rgb(231, 76, 60)', 'opacity': 0.6},
                'text': [f"{c:.2%}" for c in expected_coverages],
                'textposition': 'outside'
            }
        ]

        layout = {
            'title': 'Coverage by Alpha Level',
            'xaxis': {'title': 'Alpha Level', 'autorange': True},
            'yaxis': {'title': 'Coverage', 'tickformat': '.0%', 'autorange': True, 'rangemode': 'tozero'},
            'barmode': 'group',
            'showlegend': True
        }

        return {'data': traces, 'layout': layout}

    def _chart_width_analysis(self, alphas: List[Dict]) -> Dict[str, Any]:
        """Create interval width analysis chart (Coverage vs Width Trade-off)."""
        if not alphas:
            return {'data': [], 'layout': {}}

        alpha_values = [a['alpha'] for a in alphas]
        mean_widths = [a['mean_width'] for a in alphas]
        median_widths = [a['median_width'] for a in alphas]

        traces = [
            {
                'x': alpha_values,
                'y': mean_widths,
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Mean Width',
                'marker': {'color': 'rgb(55, 83, 109)', 'size': 10},
                'line': {'width': 2}
            },
            {
                'x': alpha_values,
                'y': median_widths,
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Median Width',
                'marker': {'color': 'rgb(26, 188, 156)', 'size': 10},
                'line': {'width': 2, 'dash': 'dot'}
            }
        ]

        layout = {
            'title': 'Prediction Interval Width Analysis',
            'xaxis': {'title': 'Alpha Level', 'tickformat': '.2f', 'autorange': True},
            'yaxis': {'title': 'Width', 'tickformat': '.3f', 'autorange': True, 'rangemode': 'tozero'},
            'hovermode': 'closest',
            'showlegend': True
        }

        return {'data': traces, 'layout': layout}

    def _chart_boxplot_width(self, alphas: List[Dict]) -> Dict[str, Any]:
        """Create box plot of interval widths by alpha."""
        if not alphas:
            return {'data': [], 'layout': {}}

        traces = []
        for alpha_data in alphas:
            traces.append({
                'y': [alpha_data['min_width'], alpha_data['mean_width'], alpha_data['max_width']],
                'type': 'box',
                'name': f"α={alpha_data['alpha']}",
                'boxmean': 'sd'
            })

        layout = {
            'title': 'Interval Width Distribution by Alpha',
            'xaxis': {'autorange': True},
            'yaxis': {'title': 'Width', 'tickformat': '.3f', 'autorange': True, 'rangemode': 'tozero'},
            'showlegend': True
        }

        return {'data': traces, 'layout': layout}

    def _chart_feature_importance(self, features: Dict) -> Dict[str, Any]:
        """Create feature importance bar chart."""
        logger.info("[FEATURE_IMPACT_DEBUG] _chart_feature_importance called")
        logger.debug(f"[FEATURE_IMPACT_DEBUG] features dict keys: {list(features.keys())}")
        logger.debug(f"[FEATURE_IMPACT_DEBUG] features['total']: {features.get('total', 0)}")

        top_features = features.get('top_10', [])
        logger.info(f"[FEATURE_IMPACT_DEBUG] top_features count: {len(top_features)}")

        if not top_features:
            logger.warning("[FEATURE_IMPACT_DEBUG] No top_features found! Returning empty chart.")
            return {'data': [], 'layout': {}}

        # Sort for display (already sorted, but ensure correct order)
        names = [f['name'] for f in top_features]
        importances = [f['importance'] for f in top_features]

        logger.debug(f"[FEATURE_IMPACT_DEBUG] Feature names: {names}")
        logger.debug(f"[FEATURE_IMPACT_DEBUG] Feature importances: {importances}")

        trace = {
            'x': importances,
            'y': names,
            'type': 'bar',
            'orientation': 'h',
            'marker': {'color': 'rgb(55, 83, 109)'},
            'hovertemplate': '<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
        }

        layout = {
            'title': 'Top 10 Most Important Features',
            'xaxis': {'title': 'Importance', 'autorange': True, 'rangemode': 'tozero'},
            'yaxis': {'title': 'Feature', 'automargin': True, 'autorange': True},
            'margin': {'l': 150}
        }

        logger.info("[FEATURE_IMPACT_DEBUG] Feature chart created successfully")
        return {'data': [trace], 'layout': layout}
