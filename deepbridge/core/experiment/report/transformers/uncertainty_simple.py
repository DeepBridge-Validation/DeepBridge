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

        # Extract main components
        # Handle both formats: with and without test_results wrapper
        if 'test_results' in results:
            # Format from JSON: results['test_results']['primary_model']
            test_results = results.get('test_results', {})
            primary_model = test_results.get('primary_model', {})
        else:
            # Format from save_html: results['primary_model'] directly
            primary_model = results.get('primary_model', {})

        initial_eval = results.get('initial_model_evaluation', {})

        # Transform the data
        transformed = {
            'model_name': model_name,
            'model_type': primary_model.get('model_type', 'Unknown'),

            # Summary metrics
            'summary': self._create_summary(primary_model),

            # Alpha results
            'alphas': self._transform_alphas(primary_model),

            # Feature importance
            'features': self._transform_features(initial_eval, primary_model),

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
                   f"{transformed['features']['total']} features")
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

        return {
            'uncertainty_score': float(uncertainty_score),
            'total_alphas': len(by_alpha),
            'avg_coverage': float(avg_coverage),
            'avg_coverage_error': float(avg_coverage_error),
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

    def _transform_features(self, initial_eval: Dict, primary_model: Dict) -> Dict[str, Any]:
        """Transform feature importance data."""
        # Try to get feature importance from multiple sources
        feature_importance = {}

        # First try: initial evaluation
        if initial_eval:
            models = initial_eval.get('models', {})
            pm = models.get('primary_model', {})
            feature_importance = pm.get('feature_importance', {})

        # Second try: primary_model directly
        if not feature_importance:
            feature_importance = primary_model.get('feature_importance', {})

        if not feature_importance:
            logger.warning("No feature importance data found")
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
        alphas = self._transform_alphas(primary_model)
        features = self._transform_features(initial_eval, primary_model)

        charts = {
            'overview': self._chart_coverage_overview(alphas),
            'calibration': self._chart_calibration(alphas),
            'coverage_by_alpha': self._chart_coverage_by_alpha(alphas),
            'width_analysis': self._chart_width_analysis(alphas),
            'boxplot_width': self._chart_boxplot_width(alphas),
            'feature_importance': self._chart_feature_importance(features)
        }

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
            'xaxis': {'title': 'Alpha Level', 'tickformat': '.2f'},
            'yaxis': {'title': 'Coverage', 'range': [0, 1.1], 'tickformat': '.2%'},
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
            'xaxis': {'title': 'Alpha Level'},
            'yaxis': {'title': 'Coverage Error', 'tickformat': '.3f'},
            'showlegend': False
        }

        return {'data': [trace], 'layout': layout}

    def _chart_coverage_by_alpha(self, alphas: List[Dict]) -> Dict[str, Any]:
        """Create coverage by alpha bar chart."""
        if not alphas:
            return {'data': [], 'layout': {}}

        alpha_values = [f"α={a['alpha']}" for a in alphas]
        coverages = [a['coverage'] for a in alphas]
        expected_coverages = [a['expected_coverage'] for a in alphas]

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
            'xaxis': {'title': 'Alpha Level'},
            'yaxis': {'title': 'Coverage', 'range': [0, 1.1], 'tickformat': '.0%'},
            'barmode': 'group',
            'showlegend': True
        }

        return {'data': traces, 'layout': layout}

    def _chart_width_analysis(self, alphas: List[Dict]) -> Dict[str, Any]:
        """Create interval width analysis chart."""
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
            'xaxis': {'title': 'Alpha Level', 'tickformat': '.2f'},
            'yaxis': {'title': 'Width', 'tickformat': '.3f'},
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
            'yaxis': {'title': 'Width', 'tickformat': '.3f'},
            'showlegend': True
        }

        return {'data': traces, 'layout': layout}

    def _chart_feature_importance(self, features: Dict) -> Dict[str, Any]:
        """Create feature importance bar chart."""
        top_features = features.get('top_10', [])

        if not top_features:
            return {'data': [], 'layout': {}}

        # Sort for display (already sorted, but ensure correct order)
        names = [f['name'] for f in top_features]
        importances = [f['importance'] for f in top_features]

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
            'xaxis': {'title': 'Importance'},
            'yaxis': {'title': 'Feature', 'automargin': True},
            'margin': {'l': 150}
        }

        return {'data': [trace], 'layout': layout}
