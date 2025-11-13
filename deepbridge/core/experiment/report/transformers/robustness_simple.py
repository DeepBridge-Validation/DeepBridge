"""
Simple data transformer for robustness reports - Following resilience/uncertainty pattern.
Transforms raw robustness results into a format suitable for simple report generation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
import plotly.graph_objects as go
import plotly.io as pio

logger = logging.getLogger("deepbridge.reports")


class RobustnessDataTransformerSimple:
    """
    Transforms robustness experiment results for simple report generation.
    Follows the resilience/uncertainty pattern.
    """

    def transform(self, results: Dict[str, Any], model_name: str = "Model") -> Dict[str, Any]:
        """
        Transform raw robustness results into report-ready format.

        Args:
            results: Dictionary containing:
                - 'test_results': Test results with primary_model data
                - 'initial_model_evaluation': Initial evaluation
            model_name: Name of the model

        Returns:
            Dictionary with transformed data ready for report rendering
        """
        logger.info("Transforming robustness data for report (SIMPLE)")

        # Extract main components
        if 'test_results' in results:
            test_results = results.get('test_results', {})
            primary_model = test_results.get('primary_model', {})
        else:
            primary_model = results.get('primary_model', {})

        initial_eval = results.get('initial_model_evaluation', {})

        # Transform the data
        transformed = {
            'model_name': model_name,
            'model_type': primary_model.get('model_type', 'Unknown'),

            # Summary metrics
            'summary': self._create_summary(primary_model),

            # Perturbation levels
            'levels': self._transform_levels(primary_model),

            # Feature importance
            'features': self._transform_features(initial_eval, primary_model),

            # Charts data (Plotly JSON)
            'charts': self._prepare_charts(primary_model, initial_eval),

            # Metadata
            'metadata': {
                'total_levels': len(self._get_levels(primary_model)),
                'total_features': self._count_features(initial_eval, primary_model),
                'n_iterations': primary_model.get('n_iterations', 10),
                'metric': primary_model.get('metric', 'AUC')
            }
        }

        logger.info(f"Transformation complete. {transformed['metadata']['total_levels']} levels, "
                   f"{transformed['metadata']['total_features']} features")
        return transformed

    def _create_summary(self, primary_model: Dict) -> Dict[str, Any]:
        """Create summary statistics."""
        base_score = primary_model.get('base_score', 0.0)
        robustness_score = primary_model.get('robustness_score', 0.0)
        avg_raw_impact = primary_model.get('avg_raw_impact', 0.0)
        avg_quantile_impact = primary_model.get('avg_quantile_impact', 0.0)

        return {
            'base_score': float(base_score),
            'robustness_score': float(robustness_score),
            'avg_raw_impact': float(avg_raw_impact),
            'avg_quantile_impact': float(avg_quantile_impact),
            'avg_overall_impact': float((avg_raw_impact + avg_quantile_impact) / 2) if avg_raw_impact or avg_quantile_impact else 0.0,
            'metric': primary_model.get('metric', 'AUC')
        }

    def _transform_levels(self, primary_model: Dict) -> List[Dict]:
        """Transform perturbation levels data."""
        levels_data = []

        # Get raw perturbation data
        raw_data = primary_model.get('raw', {}).get('by_level', {})

        for level_str, level_data in sorted(raw_data.items(), key=lambda x: float(x[0])):
            level_float = float(level_str)
            overall_result = level_data.get('overall_result', {}).get('all_features', {})

            levels_data.append({
                'level': level_float,
                'level_display': f"{level_float:.1f}",
                'mean_score': float(overall_result.get('mean_score', 0.0)),
                'std_score': float(overall_result.get('std_score', 0.0)),
                'impact': float(overall_result.get('impact', 0.0)),
                'worst_score': float(overall_result.get('worst_score', 0.0))
            })

        return levels_data

    def _transform_features(self, initial_eval: Dict, primary_model: Dict) -> List[Dict]:
        """Transform feature importance data."""
        features_data = []

        # Get feature importance from initial evaluation
        # Try multiple possible locations for feature importance
        feature_importance = {}

        # Method 1: From initial_eval.models.primary_model.feature_importance
        models_data = initial_eval.get('models', {})
        primary_model_data = models_data.get('primary_model', {})
        feature_importance = primary_model_data.get('feature_importance', {})

        # Method 2: If not found, try initial_eval directly (it might BE the models dict)
        if not feature_importance and 'primary_model' in initial_eval:
            feature_importance = initial_eval.get('primary_model', {}).get('feature_importance', {})

        # Method 3: Try from initial_model_evaluation if passed differently
        if not feature_importance and 'initial_model_evaluation' in initial_eval:
            ime = initial_eval['initial_model_evaluation']
            if 'models' in ime and 'primary_model' in ime['models']:
                feature_importance = ime['models']['primary_model'].get('feature_importance', {})

        # Method 4: FALLBACK - Use model_feature_importance directly from primary_model if available
        # This happens when using run_test() instead of run_tests()
        if not feature_importance and 'model_feature_importance' in primary_model:
            feature_importance = primary_model.get('model_feature_importance', {})
            logger.info(f"Using model_feature_importance from primary_model: {len(feature_importance)} features")

        # Get robustness-specific feature importance (from perturbation impact)
        robustness_importance = primary_model.get('feature_importance', {})

        # If we still don't have feature importance, log a warning
        if not feature_importance:
            logger.warning("No feature importance data found. Features section will be empty.")
            logger.debug(f"initial_eval keys: {list(initial_eval.keys())}")
            logger.debug(f"primary_model keys: {list(primary_model.keys())}")
        else:
            logger.info(f"Found {len(feature_importance)} features for report")

        for feature_name, importance in sorted(feature_importance.items(),
                                               key=lambda x: x[1],
                                               reverse=True):
            features_data.append({
                'name': feature_name,
                'importance': float(importance),
                'robustness_impact': float(robustness_importance.get(feature_name, 0.0))
            })

        return features_data

    def _prepare_charts(self, primary_model: Dict, initial_eval: Dict) -> Dict[str, str]:
        """Prepare Plotly charts as JSON strings."""
        charts = {}

        # Chart 1: Perturbation Impact Overview
        charts['overview'] = self._create_overview_chart(primary_model)

        # Chart 2: Score by Level
        charts['by_level'] = self._create_by_level_chart(primary_model)

        # Chart 3: Feature Importance
        charts['feature_importance'] = self._create_feature_importance_chart(initial_eval, primary_model)

        # Chart 4: Score Distribution (boxplot style)
        charts['score_distribution'] = self._create_score_distribution_chart(primary_model)

        return charts

    def _create_overview_chart(self, primary_model: Dict) -> str:
        """Create overview chart showing base vs perturbed scores."""
        levels_data = self._transform_levels(primary_model)

        if not levels_data:
            return self._create_empty_chart("No data available")

        levels = [d['level'] for d in levels_data]
        mean_scores = [d['mean_score'] for d in levels_data]
        worst_scores = [d['worst_score'] for d in levels_data]
        base_score = primary_model.get('base_score', 1.0)

        fig = go.Figure()

        # Base score line
        fig.add_trace(go.Scatter(
            x=levels,
            y=[base_score] * len(levels),
            mode='lines',
            name='Base Score',
            line=dict(color='green', width=2, dash='dash')
        ))

        # Mean perturbed scores
        fig.add_trace(go.Scatter(
            x=levels,
            y=mean_scores,
            mode='lines+markers',
            name='Mean Score',
            line=dict(color='blue', width=3),
            marker=dict(size=8)
        ))

        # Worst scores
        fig.add_trace(go.Scatter(
            x=levels,
            y=worst_scores,
            mode='lines+markers',
            name='Worst Score',
            line=dict(color='red', width=2),
            marker=dict(size=6)
        ))

        fig.update_layout(
            title='Robustness Overview: Performance by Perturbation Level',
            xaxis_title='Perturbation Level',
            yaxis_title='Score',
            hovermode='x unified',
            template='plotly_white',
            height=400
        )

        return pio.to_json(fig)

    def _create_by_level_chart(self, primary_model: Dict) -> str:
        """Create chart showing impact by perturbation level."""
        levels_data = self._transform_levels(primary_model)

        if not levels_data:
            return self._create_empty_chart("No data available")

        levels = [d['level_display'] for d in levels_data]
        impacts = [d['impact'] for d in levels_data]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=levels,
            y=impacts,
            marker=dict(
                color=impacts,
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title="Impact")
            ),
            text=[f"{i:.4f}" for i in impacts],
            textposition='outside'
        ))

        fig.update_layout(
            title='Impact by Perturbation Level',
            xaxis_title='Perturbation Level',
            yaxis_title='Performance Impact',
            template='plotly_white',
            height=400
        )

        return pio.to_json(fig)

    def _create_feature_importance_chart(self, initial_eval: Dict, primary_model: Dict) -> str:
        """Create feature importance chart."""
        features_data = self._transform_features(initial_eval, primary_model)

        if not features_data:
            return self._create_empty_chart("No feature data available")

        # Take top 10 features
        top_features = features_data[:10]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            y=[f['name'] for f in top_features],
            x=[f['importance'] for f in top_features],
            orientation='h',
            marker=dict(color='#1b78de'),
            text=[f"{f['importance']:.4f}" for f in top_features],
            textposition='outside'
        ))

        fig.update_layout(
            title='Top 10 Feature Importance',
            xaxis_title='Importance',
            yaxis_title='Feature',
            template='plotly_white',
            height=400,
            yaxis=dict(autorange='reversed')
        )

        return pio.to_json(fig)

    def _create_score_distribution_chart(self, primary_model: Dict) -> str:
        """Create score distribution chart."""
        raw_data = primary_model.get('raw', {}).get('by_level', {})

        if not raw_data:
            return self._create_empty_chart("No distribution data available")

        fig = go.Figure()

        for level_str in sorted(raw_data.keys(), key=float):
            level_data = raw_data[level_str]
            runs = level_data.get('runs', {}).get('all_features', [])

            if runs:
                run_data = runs[0]
                iterations = run_data.get('iterations', {})
                scores = iterations.get('scores', [])

                if scores:
                    fig.add_trace(go.Box(
                        y=scores,
                        name=f"Level {level_str}",
                        boxmean='sd'
                    ))

        fig.update_layout(
            title='Score Distribution by Perturbation Level',
            xaxis_title='Perturbation Level',
            yaxis_title='Score',
            template='plotly_white',
            height=400
        )

        return pio.to_json(fig)

    def _create_empty_chart(self, message: str) -> str:
        """Create an empty chart with a message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            template='plotly_white',
            height=400,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return pio.to_json(fig)

    def _get_levels(self, primary_model: Dict) -> List[float]:
        """Get list of perturbation levels."""
        raw_data = primary_model.get('raw', {}).get('by_level', {})
        return sorted([float(level) for level in raw_data.keys()])

    def _get_features(self, initial_eval: Dict) -> List[str]:
        """Get list of feature names."""
        # Try multiple possible locations for feature importance
        feature_importance = {}

        # Method 1: From initial_eval.models.primary_model.feature_importance
        models_data = initial_eval.get('models', {})
        primary_model_data = models_data.get('primary_model', {})
        feature_importance = primary_model_data.get('feature_importance', {})

        # Method 2: If not found, try initial_eval directly
        if not feature_importance and 'primary_model' in initial_eval:
            feature_importance = initial_eval.get('primary_model', {}).get('feature_importance', {})

        # Method 3: Try from initial_model_evaluation
        if not feature_importance and 'initial_model_evaluation' in initial_eval:
            ime = initial_eval['initial_model_evaluation']
            if 'models' in ime and 'primary_model' in ime['models']:
                feature_importance = ime['models']['primary_model'].get('feature_importance', {})

        # Note: We don't check primary_model here as it's not passed to this method
        # The caller should ensure initial_eval has the feature importance data

        return list(feature_importance.keys())

    def _count_scenarios(self, primary_model: Dict) -> int:
        """Count total number of scenarios."""
        return len(self._get_levels(primary_model))

    def _count_features(self, initial_eval: Dict, primary_model: Dict = None) -> int:
        """Count total number of features."""
        features = self._get_features(initial_eval)

        # Fallback: if no features from initial_eval, try primary_model
        if not features and primary_model and 'model_feature_importance' in primary_model:
            features = list(primary_model['model_feature_importance'].keys())

        return len(features)
