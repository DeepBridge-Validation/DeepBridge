"""
Post-training fairness charts.

Contains specialized visualizations for post-training fairness metrics.
"""

from typing import Dict, Any, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from .base_chart import BaseChart
from ..utils import (
    get_status_from_interpretation,
    POSTTRAIN_MAIN_METRICS,
    METRIC_SHORT_LABELS,
    format_attribute_name
)


class DisparateImpactGaugeChart(BaseChart):
    """
    Gauge chart for Disparate Impact metric (EEOC 80% Rule).

    Shows compliance with EEOC 80% rule - most critical legal metric.
    """

    def create(self, data: Dict[str, Any]) -> str:
        """
        Create disparate impact gauge chart.

        Args:
            data: Dictionary with 'posttrain_metrics' and 'protected_attrs'

        Returns:
            Plotly JSON string
        """
        posttrain_metrics = data.get('posttrain_metrics', {})
        protected_attrs = data.get('protected_attrs', [])

        if not posttrain_metrics:
            return '{}'

        # Extract disparate impact data for each attribute
        gauge_data = []
        for attr in protected_attrs:
            if attr in posttrain_metrics:
                di_metric = posttrain_metrics[attr].get('disparate_impact', {})
                if isinstance(di_metric, dict) and 'ratio' in di_metric:
                    ratio = di_metric.get('ratio', 0.0)
                    passes = di_metric.get('passes_80_rule', False)

                    gauge_data.append({
                        'attribute': attr,
                        'ratio': ratio,
                        'passes': passes
                    })

        if not gauge_data:
            return '{}'

        # Create subplot for each attribute (side by side gauges)
        n_attrs = len(gauge_data)
        fig = make_subplots(
            rows=1,
            cols=n_attrs,
            subplot_titles=[f"{format_attribute_name(d['attribute'])}" for d in gauge_data],
            horizontal_spacing=0.1,
            specs=[[{'type': 'indicator'} for _ in range(n_attrs)]]
        )

        for i, item in enumerate(gauge_data, start=1):
            ratio = item['ratio']

            # Color based on compliance
            if ratio >= 0.8:
                color = self.COLOR_SUCCESS
            elif ratio >= 0.7:
                color = self.COLOR_WARNING
            else:
                color = self.COLOR_CRITICAL

            # Create gauge
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=ratio,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': f"Ratio: {ratio:.3f}"},
                    delta={'reference': 0.8, 'increasing': {'color': self.COLOR_SUCCESS},
                           'decreasing': {'color': self.COLOR_CRITICAL}},
                    gauge={
                        'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "white"},
                        'bar': {'color': color},
                        'bgcolor': "rgba(255, 255, 255, 0.1)",
                        'borderwidth': 2,
                        'bordercolor': "white",
                        'steps': [
                            {'range': [0, 0.7], 'color': 'rgba(231, 76, 60, 0.3)'},
                            {'range': [0.7, 0.8], 'color': 'rgba(243, 156, 18, 0.3)'},
                            {'range': [0.8, 1], 'color': 'rgba(46, 204, 113, 0.3)'}
                        ],
                        'threshold': {
                            'line': {'color': "white", 'width': 4},
                            'thickness': 0.75,
                            'value': 0.8
                        }
                    }
                ),
                row=1,
                col=i
            )

        self._apply_common_layout(
            fig,
            title={
                'text': 'Disparate Impact - EEOC 80% Rule Compliance ⚖️',
                'font': {'size': 20, 'color': '#2c3e50'}
            },
            height=400,
            showlegend=False
        )

        return self._to_json(fig)


class DisparityComparisonChart(BaseChart):
    """
    Diverging bar chart for Statistical Parity disparity values.

    Shows how far each attribute deviates from perfect fairness (0.0).
    """

    def create(self, data: Dict[str, Any]) -> str:
        """
        Create disparity comparison chart.

        Args:
            data: Dictionary with 'posttrain_metrics' and 'protected_attrs'

        Returns:
            Plotly JSON string
        """
        posttrain_metrics = data.get('posttrain_metrics', {})
        protected_attrs = data.get('protected_attrs', [])

        if not posttrain_metrics:
            return '{}'

        # Extract statistical parity disparity for each attribute
        chart_data = []
        for attr in protected_attrs:
            if attr in posttrain_metrics:
                sp_metric = posttrain_metrics[attr].get('statistical_parity', {})
                if isinstance(sp_metric, dict) and 'disparity' in sp_metric:
                    disparity = sp_metric.get('disparity', 0.0)
                    interpretation = sp_metric.get('interpretation', '')
                    status = get_status_from_interpretation(interpretation)

                    chart_data.append({
                        'attribute': format_attribute_name(attr),
                        'disparity': disparity,
                        'status': status
                    })

        if not chart_data:
            return '{}'

        df = pd.DataFrame(chart_data)
        colors = [self._get_color_for_status(status) for status in df['status']]

        # Create diverging bar chart
        fig = go.Figure()

        fig.add_trace(go.Bar(
            y=df['attribute'],
            x=df['disparity'],
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='white', width=1)
            ),
            text=[self._format_decimal(d) for d in df['disparity']],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Disparity: %{x}<extra></extra>'
        ))

        # Add reference lines
        fig.add_vline(x=0, line_dash="solid", line_color="white", line_width=2)
        fig.add_vline(x=0.1, line_dash="dash", line_color=self.COLOR_WARNING, line_width=1,
                     annotation_text="Warning Threshold", annotation_position="top")
        fig.add_vline(x=-0.1, line_dash="dash", line_color=self.COLOR_WARNING, line_width=1)

        self._apply_common_layout(
            fig,
            title='Statistical Parity - Disparity Analysis',
            xaxis_title='Disparity (0.0 = Perfect Fairness)',
            yaxis_title='Protected Attribute',
            height=max(300, 100 * len(chart_data)),
            showlegend=False,
            xaxis=dict(
                autorange=True,
                gridcolor='rgba(255, 255, 255, 0.1)',
                zerolinecolor='white',
                zerolinewidth=2
            ),
            yaxis=dict(autorange=True, gridcolor='rgba(255, 255, 255, 0.1)')
        )

        return self._to_json(fig)


class ComplianceStatusMatrixChart(BaseChart):
    """
    Heatmap matrix showing status of all post-training metrics.

    Executive dashboard view with color-coded compliance status.
    """

    def create(self, data: Dict[str, Any]) -> str:
        """
        Create compliance status matrix chart.

        Args:
            data: Dictionary with 'posttrain_metrics' and 'protected_attrs'

        Returns:
            Plotly JSON string
        """
        posttrain_metrics = data.get('posttrain_metrics', {})
        protected_attrs = data.get('protected_attrs', [])

        if not posttrain_metrics:
            return '{}'

        # Build matrix data
        matrix = []
        text_matrix = []
        attr_labels = []

        for attr in protected_attrs:
            row = []
            text_row = []
            attr_labels.append(format_attribute_name(attr))

            for metric in POSTTRAIN_MAIN_METRICS:
                if attr in posttrain_metrics and metric in posttrain_metrics[attr]:
                    metric_data = posttrain_metrics[attr][metric]
                    if isinstance(metric_data, dict):
                        interpretation = metric_data.get('interpretation', '')
                        status = get_status_from_interpretation(interpretation)

                        # Map status to numeric value for color scale
                        status_value = {
                            'success': 1.0,
                            'ok': 1.0,
                            'warning': 0.5,
                            'critical': 0.0
                        }.get(status, 0.5)

                        # Get symbol
                        symbol = {
                            'success': '✓',
                            'ok': '✓',
                            'warning': '⚠',
                            'critical': '✗'
                        }.get(status, '?')

                        row.append(status_value)
                        text_row.append(symbol)
                    else:
                        row.append(0.5)
                        text_row.append('?')
                else:
                    row.append(None)
                    text_row.append('N/A')

            matrix.append(row)
            text_matrix.append(text_row)

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=[METRIC_SHORT_LABELS.get(m, m) for m in POSTTRAIN_MAIN_METRICS],
            y=attr_labels,
            text=text_matrix,
            texttemplate='<b>%{text}</b>',
            textfont={'size': 20, 'color': '#ffffff'},
            colorscale=[
                [0.0, self.COLOR_CRITICAL],
                [0.5, self.COLOR_WARNING],
                [1.0, self.COLOR_SUCCESS]
            ],
            showscale=False,
            hovertemplate='<b>%{y}</b><br>%{x}<br>Status: %{text}<extra></extra>'
        ))

        self._apply_common_layout(
            fig,
            title='Compliance Status Matrix - Main Post-Training Metrics',
            height=max(300, 80 * len(protected_attrs)),
            xaxis=dict(
                autorange=True,
                tickfont={'size': 11},
                side='top'
            ),
            yaxis=dict(
                autorange=True,
                tickfont={'size': 12}
            )
        )

        return self._to_json(fig)
