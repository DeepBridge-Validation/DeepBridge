"""
Simple data transformer for fairness reports.
Transforms raw fairness results into a format suitable for report generation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

logger = logging.getLogger("deepbridge.reports")


class FairnessDataTransformerSimple:
    """
    Transforms fairness experiment results for report generation.
    """

    def transform(self, results: Dict[str, Any], model_name: str = "Model") -> Dict[str, Any]:
        """
        Transform raw fairness results into report-ready format.

        Args:
            results: Dictionary containing fairness analysis results from FairnessSuite
            model_name: Name of the model

        Returns:
            Dictionary with transformed data ready for report rendering
        """
        logger.info("Transforming fairness data for report")

        # Extract main components
        protected_attrs = results.get('protected_attributes', [])
        pretrain_metrics = results.get('pretrain_metrics', {})
        posttrain_metrics = results.get('posttrain_metrics', {})
        confusion_matrix = results.get('confusion_matrix', {})
        threshold_analysis = results.get('threshold_analysis', None)
        warnings = results.get('warnings', [])
        critical_issues = results.get('critical_issues', [])
        overall_score = results.get('overall_fairness_score', 0.0)

        # Transform the data
        transformed = {
            'model_name': model_name,
            'model_type': 'Classification Model',

            # Summary metrics
            'summary': self._create_summary(results),

            # Protected attributes data
            'protected_attributes': self._transform_protected_attributes(
                protected_attrs, pretrain_metrics, posttrain_metrics
            ),

            # Issues and warnings
            'issues': self._transform_issues(warnings, critical_issues),

            # Charts data (Plotly JSON)
            'charts': self._prepare_charts(results),

            # Metadata
            'metadata': {
                'total_attributes': len(protected_attrs),
                'total_pretrain_metrics': sum(len(m) for m in pretrain_metrics.values()),
                'total_posttrain_metrics': sum(len(m) for m in posttrain_metrics.values()),
                'has_threshold_analysis': threshold_analysis is not None,
                'has_confusion_matrix': bool(confusion_matrix)
            }
        }

        logger.info(f"Transformation complete. {len(protected_attrs)} protected attributes analyzed")
        return transformed

    def _create_summary(self, results: Dict) -> Dict[str, Any]:
        """Create summary statistics."""
        return {
            'overall_fairness_score': float(results.get('overall_fairness_score', 0.0)),
            'total_warnings': len(results.get('warnings', [])),
            'total_critical': len(results.get('critical_issues', [])),
            'total_attributes': len(results.get('protected_attributes', [])),
            'config': results.get('config', 'custom'),
            'assessment': self._get_assessment(results.get('overall_fairness_score', 0.0))
        }

    def _get_assessment(self, score: float) -> str:
        """Get textual assessment based on overall score."""
        if score >= 0.9:
            return "EXCELLENT - Very high fairness"
        elif score >= 0.8:
            return "GOOD - Adequate fairness for production"
        elif score >= 0.6:
            return "MODERATE - Requires improvements before production"
        else:
            return "CRITICAL - Not recommended for production"

    def _transform_protected_attributes(
        self,
        attributes: List[str],
        pretrain: Dict[str, Dict],
        posttrain: Dict[str, Dict]
    ) -> List[Dict]:
        """Transform protected attributes data."""
        transformed_attrs = []

        for attr in attributes:
            attr_data = {
                'name': attr,
                'pretrain_metrics': [],
                'posttrain_metrics': []
            }

            # Transform pre-training metrics
            if attr in pretrain:
                for metric_name, metric_result in pretrain[attr].items():
                    if isinstance(metric_result, dict):
                        attr_data['pretrain_metrics'].append({
                            'name': metric_name.replace('_', ' ').title(),
                            'value': metric_result.get('value', 0.0),
                            'interpretation': metric_result.get('interpretation', ''),
                            'status': self._get_status_from_interpretation(
                                metric_result.get('interpretation', '')
                            )
                        })

            # Transform post-training metrics
            if attr in posttrain:
                for metric_name, metric_result in posttrain[attr].items():
                    if isinstance(metric_result, dict):
                        attr_data['posttrain_metrics'].append({
                            'name': metric_name.replace('_', ' ').title(),
                            'value': abs(metric_result.get('value', 0.0)),  # Use abs for display
                            'interpretation': metric_result.get('interpretation', ''),
                            'status': self._get_status_from_interpretation(
                                metric_result.get('interpretation', '')
                            )
                        })

            transformed_attrs.append(attr_data)

        return transformed_attrs

    def _get_status_from_interpretation(self, interpretation: str) -> str:
        """Extract status from interpretation string."""
        if '✗' in interpretation or 'CRÍTICO' in interpretation or 'Vermelho' in interpretation:
            return 'critical'
        elif '⚠' in interpretation or 'MODERADO' in interpretation or 'Amarelo' in interpretation:
            return 'warning'
        else:
            return 'ok'

    def _transform_issues(self, warnings: List[str], critical: List[str]) -> Dict[str, List]:
        """Transform warnings and critical issues."""
        return {
            'warnings': warnings,
            'critical': critical,
            'total': len(warnings) + len(critical)
        }

    def _prepare_charts(self, results: Dict) -> Dict[str, str]:
        """
        Prepare Plotly charts as JSON strings.

        Returns dict with chart names as keys and Plotly JSON as values.
        """
        charts = {}

        # 1. Metrics comparison chart
        charts['metrics_comparison'] = self._create_metrics_comparison_chart(
            results.get('posttrain_metrics', {}),
            results.get('protected_attributes', [])
        )

        # 2. Fairness radar chart
        charts['fairness_radar'] = self._create_fairness_radar_chart(
            results.get('posttrain_metrics', {})
        )

        # 3. Confusion matrices
        if results.get('confusion_matrix'):
            charts['confusion_matrices'] = self._create_confusion_matrices_chart(
                results.get('confusion_matrix', {}),
                results.get('protected_attributes', [])
            )

        # 4. Threshold analysis
        if results.get('threshold_analysis'):
            charts['threshold_analysis'] = self._create_threshold_chart(
                results.get('threshold_analysis', {})
            )

        return charts

    def _create_metrics_comparison_chart(
        self,
        posttrain_metrics: Dict[str, Dict],
        protected_attrs: List[str]
    ) -> str:
        """Create metrics comparison bar chart."""
        if not posttrain_metrics:
            return '{}'

        # Prepare data
        data = []
        for attr in protected_attrs:
            if attr in posttrain_metrics:
                for metric_name, metric_result in posttrain_metrics[attr].items():
                    if isinstance(metric_result, dict) and 'value' in metric_result:
                        data.append({
                            'attribute': attr,
                            'metric': metric_name.replace('_', ' ').title(),
                            'value': abs(metric_result['value']),
                            'status': self._get_status_from_interpretation(
                                metric_result.get('interpretation', '')
                            )
                        })

        if not data:
            return '{}'

        df = pd.DataFrame(data)

        # Create color map
        color_map = {'ok': '#2ecc71', 'warning': '#f39c12', 'critical': '#e74c3c'}

        # Create figure
        fig = px.bar(
            df,
            x='value',
            y='metric',
            color='status',
            facet_col='attribute',
            color_discrete_map=color_map,
            labels={'value': 'Metric Value (Absolute)', 'metric': 'Fairness Metric'},
            title='Fairness Metrics Comparison by Protected Attribute',
            orientation='h'
        )

        fig.update_layout(
            height=500,
            showlegend=True,
            legend_title_text='Status',
            font=dict(size=11)
        )

        # Add reference line at 0.1
        fig.add_hline(y=0.1, line_dash="dash", line_color="gray", opacity=0.5)

        return pio.to_json(fig)

    def _create_fairness_radar_chart(self, posttrain_metrics: Dict[str, Dict]) -> str:
        """Create radar chart for fairness metrics."""
        if not posttrain_metrics:
            return '{}'

        # Select key metrics for radar
        key_metrics = [
            'statistical_parity',
            'disparate_impact',
            'equal_opportunity',
            'equalized_odds',
            'precision_difference'
        ]

        fig = go.Figure()

        for attr, metrics in posttrain_metrics.items():
            values = []
            labels = []

            for metric in key_metrics:
                if metric in metrics and isinstance(metrics[metric], dict):
                    value = metrics[metric].get('value', 0)
                    # Normalize for radar (closer to 1 = better fairness)
                    if metric == 'disparate_impact':
                        normalized = min(abs(value), 1.0)
                    else:
                        normalized = max(0, 1 - abs(value))

                    values.append(normalized)
                    labels.append(metric.replace('_', ' ').title())

            if values:
                # Close the polygon
                values.append(values[0])
                labels.append(labels[0])

                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=labels,
                    fill='toself',
                    name=attr.title()
                ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title='Fairness Radar Chart (1.0 = Perfect Fairness)',
            height=500
        )

        return pio.to_json(fig)

    def _create_confusion_matrices_chart(
        self,
        confusion_matrices: Dict[str, Dict],
        protected_attrs: List[str]
    ) -> str:
        """Create heatmap visualization of confusion matrices."""
        if not confusion_matrices:
            return '{}'

        # Count total number of groups (each group gets its own subplot)
        total_groups = 0
        subplot_titles = []

        for attr in protected_attrs:
            if attr in confusion_matrices:
                groups = list(confusion_matrices[attr].keys())
                total_groups += len(groups)
                subplot_titles.extend([f"{attr}: {g}" for g in groups])

        if total_groups == 0:
            return '{}'

        # Create subplots based on total groups (3 columns)
        cols = min(total_groups, 3)
        rows = (total_groups + cols - 1) // cols  # Ceiling division

        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=subplot_titles,
            specs=[[{'type': 'heatmap'}] * cols for _ in range(rows)]
        )

        row, col = 1, 1
        for attr in protected_attrs:
            if attr in confusion_matrices:
                for group, cm_data in confusion_matrices[attr].items():
                    # Create confusion matrix
                    matrix = [
                        [cm_data.get('TN', 0), cm_data.get('FP', 0)],
                        [cm_data.get('FN', 0), cm_data.get('TP', 0)]
                    ]

                    fig.add_trace(
                        go.Heatmap(
                            z=matrix,
                            x=['Pred Neg', 'Pred Pos'],
                            y=['Act Neg', 'Act Pos'],
                            colorscale='Blues',
                            showscale=False,
                            text=matrix,
                            texttemplate='%{text}',
                            textfont={"size": 12}
                        ),
                        row=row,
                        col=col
                    )

                    col += 1
                    if col > cols:
                        col = 1
                        row += 1

        fig.update_layout(
            height=250 * rows,
            title='Confusion Matrices by Group',
            showlegend=False
        )

        return pio.to_json(fig)

    def _create_threshold_chart(self, threshold_analysis: Dict) -> str:
        """Create threshold impact chart."""
        if not threshold_analysis or 'threshold_curve' not in threshold_analysis:
            return '{}'

        curve_data = threshold_analysis['threshold_curve']
        if not curve_data:
            return '{}'

        df = pd.DataFrame(curve_data)

        fig = go.Figure()

        # Plot each metric
        if 'disparate_impact_ratio' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['threshold'],
                y=df['disparate_impact_ratio'],
                mode='lines',
                name='Disparate Impact',
                line=dict(color='blue', width=2)
            ))

        if 'statistical_parity' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['threshold'],
                y=df['statistical_parity'],
                mode='lines',
                name='Statistical Parity',
                line=dict(color='green', width=2)
            ))

        if 'f1_score' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['threshold'],
                y=df['f1_score'],
                mode='lines',
                name='F1 Score',
                line=dict(color='purple', width=2)
            ))

        # Mark optimal threshold
        optimal_threshold = threshold_analysis.get('optimal_threshold', 0.5)
        fig.add_vline(
            x=optimal_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Optimal: {optimal_threshold:.3f}"
        )

        # Add EEOC threshold
        fig.add_hline(
            y=0.8,
            line_dash="dot",
            line_color="orange",
            annotation_text="EEOC 80%"
        )

        fig.update_layout(
            title='Threshold Impact on Fairness Metrics',
            xaxis_title='Classification Threshold',
            yaxis_title='Metric Value',
            height=500,
            showlegend=True,
            hovermode='x unified'
        )

        return pio.to_json(fig)
