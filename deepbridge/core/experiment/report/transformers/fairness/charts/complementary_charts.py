"""
Complementary fairness charts.

Contains visualizations for complementary fairness metrics.
"""

from typing import Dict, Any, List
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from .base_chart import BaseChart
from ..utils import (
    POSTTRAIN_COMPLEMENTARY_METRICS,
    METRIC_SHORT_LABELS,
    format_attribute_name
)


class PrecisionAccuracyComparisonChart(BaseChart):
    """
    Grouped bar chart comparing precision and accuracy by group.

    Shows performance metrics (precision, accuracy) for each group within
    each protected attribute.
    """

    def create(self, data: Dict[str, Any]) -> str:
        """
        Create precision and accuracy comparison chart.

        Args:
            data: Dictionary with 'confusion_matrix' and 'protected_attrs'

        Returns:
            Plotly JSON string
        """
        confusion_matrix = data.get('confusion_matrix', {})
        protected_attrs = data.get('protected_attrs', [])

        if not confusion_matrix:
            return '{}'

        # Collect data from confusion matrices
        chart_data = []
        for attr in protected_attrs:
            if attr not in confusion_matrix:
                continue

            for group, cm_data in confusion_matrix[attr].items():
                tp = cm_data.get('TP', 0)
                tn = cm_data.get('TN', 0)
                fp = cm_data.get('FP', 0)
                fn = cm_data.get('FN', 0)

                # Calculate metrics
                total = tp + tn + fp + fn
                accuracy = (tp + tn) / total if total > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0

                chart_data.append({
                    'attribute': format_attribute_name(attr),
                    'group': str(group),
                    'metric': 'Accuracy',
                    'value': accuracy
                })
                chart_data.append({
                    'attribute': format_attribute_name(attr),
                    'group': str(group),
                    'metric': 'Precision',
                    'value': precision
                })

        if not chart_data:
            return '{}'

        df = pd.DataFrame(chart_data)

        # Create grouped bar chart
        fig = px.bar(
            df,
            x='group',
            y='value',
            color='metric',
            facet_col='attribute',
            barmode='group',
            labels={'value': 'Score', 'group': 'Group'},
            title='Precision & Accuracy Comparison by Group',
            color_discrete_map={'Accuracy': self.COLOR_INFO, 'Precision': '#9b59b6'}
        )

        self._apply_common_layout(
            fig,
            height=400,
            showlegend=True,
            legend_title_text='Metric',
            yaxis=dict(
                autorange=True,
                gridcolor='rgba(255, 255, 255, 0.1)',
                tickformat='.0%',
                range=[0, 1]
            )
        )

        fig.update_xaxes(gridcolor='rgba(255, 255, 255, 0.1)')

        return self._to_json(fig)


class TreatmentEqualityScatterChart(BaseChart):
    """
    Scatter plot showing treatment equality (FN vs FP rates).

    Shows if errors are balanced between groups - ideal is on diagonal.
    """

    def create(self, data: Dict[str, Any]) -> str:
        """
        Create treatment equality scatter chart.

        Args:
            data: Dictionary with 'confusion_matrix' and 'protected_attrs'

        Returns:
            Plotly JSON string
        """
        confusion_matrix = data.get('confusion_matrix', {})
        protected_attrs = data.get('protected_attrs', [])

        if not confusion_matrix:
            return '{}'

        # Collect data from confusion matrices
        chart_data = []
        for attr in protected_attrs:
            if attr not in confusion_matrix:
                continue

            for group, cm_data in confusion_matrix[attr].items():
                tp = cm_data.get('TP', 0)
                tn = cm_data.get('TN', 0)
                fp = cm_data.get('FP', 0)
                fn = cm_data.get('FN', 0)

                # Calculate error rates
                total_positives = tp + fn
                total_negatives = tn + fp
                fn_rate = fn / total_positives if total_positives > 0 else 0
                fp_rate = fp / total_negatives if total_negatives > 0 else 0

                # Calculate sample size for bubble size
                sample_size = tp + tn + fp + fn

                chart_data.append({
                    'attribute': format_attribute_name(attr),
                    'group': str(group),
                    'fn_rate': fn_rate,
                    'fp_rate': fp_rate,
                    'sample_size': sample_size
                })

        if not chart_data:
            return '{}'

        df = pd.DataFrame(chart_data)

        # Create scatter plot
        fig = go.Figure()

        # Get unique attributes for coloring
        unique_attrs = df['attribute'].unique()
        colors = ['#636efa', '#EF553B', '#00cc96', '#ab63fa', '#FFA15A', '#19d3f3']

        for i, attr in enumerate(unique_attrs):
            attr_data = df[df['attribute'] == attr]

            # Convert to Python lists to avoid binary serialization
            x_values = attr_data['fp_rate'].tolist()
            y_values = attr_data['fn_rate'].tolist()
            text_labels = [str(row['group']) for _, row in attr_data.iterrows()]
            marker_sizes = [max(8, min(30, s / 200)) for s in attr_data['sample_size'].tolist()]

            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode='markers+text',
                    name=attr,
                    text=text_labels,
                    textposition='top center',
                    textfont=dict(color='#2c3e50', size=10),
                    marker=dict(
                        size=marker_sizes,
                        color=colors[i % len(colors)],
                        opacity=0.7,
                        line=dict(color='#2c3e50', width=1)
                    ),
                    hovertemplate='<b>%{text}</b><br>' +
                                'FP Rate: %{x}<br>' +
                                'FN Rate: %{y}<br>' +
                                '<extra></extra>'
                )
            )

        # Add diagonal reference line (perfect balance)
        max_rate = max(df['fp_rate'].max(), df['fn_rate'].max()) if len(df) > 0 else 1
        fig.add_trace(
            go.Scatter(
                x=[0, max_rate],
                y=[0, max_rate],
                mode='lines',
                line=dict(color='white', dash='dash', width=2),
                name='Perfect Balance (FN=FP)',
                showlegend=True,
                hoverinfo='skip'
            )
        )

        self._apply_common_layout(
            fig,
            title='Treatment Equality - Error Balance Analysis',
            height=500,
            showlegend=True,
            xaxis=dict(
                autorange=True,
                title='False Positive Rate',
                gridcolor='rgba(255, 255, 255, 0.1)',
                tickformat='.0%',
                range=[0, max_rate * 1.1] if max_rate > 0 else [0, 1]
            ),
            yaxis=dict(
                autorange=True,
                title='False Negative Rate',
                gridcolor='rgba(255, 255, 255, 0.1)',
                tickformat='.0%',
                range=[0, max_rate * 1.1] if max_rate > 0 else [0, 1]
            )
        )

        return self._to_json(fig)


class ComplementaryMetricsRadarChart(BaseChart):
    """
    Radar chart for complementary metrics.

    Shows profile of 6 complementary fairness metrics for each attribute.
    """

    def create(self, data: Dict[str, Any]) -> str:
        """
        Create complementary metrics radar chart.

        Args:
            data: Dictionary with 'posttrain_metrics' and 'protected_attrs'

        Returns:
            Plotly JSON string
        """
        posttrain_metrics = data.get('posttrain_metrics', {})
        protected_attrs = data.get('protected_attrs', [])

        if not posttrain_metrics:
            return '{}'

        fig = go.Figure()

        for attr in protected_attrs:
            if attr not in posttrain_metrics:
                continue

            values = []
            labels = []

            for metric in POSTTRAIN_COMPLEMENTARY_METRICS:
                if metric not in posttrain_metrics[attr]:
                    continue

                metric_data = posttrain_metrics[attr][metric]
                if not isinstance(metric_data, dict):
                    continue

                # Extract value
                if 'value' in metric_data:
                    value = abs(metric_data['value'])
                elif 'disparity' in metric_data:
                    value = abs(metric_data['disparity'])
                elif 'ratio' in metric_data:
                    value = abs(1 - metric_data['ratio'])
                else:
                    continue

                # Normalize to 0-1 scale (closer to 0 = better fairness)
                normalized = min(value, 1.0)

                values.append(1 - normalized)  # Invert so 1 = good fairness
                labels.append(METRIC_SHORT_LABELS.get(metric, metric))

            if values:
                # Close the polygon
                values.append(values[0])
                labels.append(labels[0])

                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=labels,
                    fill='toself',
                    name=format_attribute_name(attr)
                ))

        self._apply_common_layout(
            fig,
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickformat='.0%'
                )
            ),
            showlegend=True,
            title='Complementary Metrics Radar (1.0 = Perfect Fairness)',
            height=500
        )

        return self._to_json(fig)
