"""
Pre-training fairness charts.

Contains specialized visualizations for pre-training fairness metrics.
"""

from typing import Dict, Any, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .base_chart import BaseChart
from ..utils import (
    get_status_from_interpretation,
    PRETRAIN_METRICS,
    METRIC_LABELS,
    format_attribute_name
)


class PretrainMetricsOverviewChart(BaseChart):
    """
    Grouped bar chart for all pre-training metrics.

    Shows all 4 pre-training metrics (BCL, BCO, KL, JS) for each attribute.
    """

    def create(self, data: Dict[str, Any]) -> str:
        """
        Create pre-training metrics overview chart.

        Args:
            data: Dictionary with 'pretrain_metrics' and 'protected_attrs'

        Returns:
            Plotly JSON string
        """
        pretrain_metrics = data.get('pretrain_metrics', {})
        protected_attrs = data.get('protected_attrs', [])

        if not pretrain_metrics:
            return '{}'

        # Create figure
        fig = go.Figure()

        # Collect data per metric
        for metric_name in PRETRAIN_METRICS:
            x_data = []
            y_data = []
            colors = []
            text_data = []

            for attr in protected_attrs:
                if attr not in pretrain_metrics:
                    continue

                if metric_name not in pretrain_metrics[attr]:
                    continue

                metric = pretrain_metrics[attr][metric_name]
                if not isinstance(metric, dict):
                    continue

                value = abs(metric.get('value', 0.0))
                interpretation = metric.get('interpretation', '')
                status = get_status_from_interpretation(interpretation)

                x_data.append(format_attribute_name(attr))
                y_data.append(value)
                colors.append(self._get_color_for_status(status))
                text_data.append(self._format_decimal(value))

            if x_data:
                fig.add_trace(go.Bar(
                    name=METRIC_LABELS[metric_name],
                    x=x_data,
                    y=y_data,
                    text=text_data,
                    textposition='outside',
                    marker=dict(color=colors),
                    hovertemplate='<b>%{x}</b><br>' + METRIC_LABELS[metric_name] + '<br>Value: %{y}<extra></extra>'
                ))

        self._apply_common_layout(
            fig,
            title='Pre-Training Metrics Overview (All 4 Metrics)',
            xaxis_title='Protected Attribute',
            yaxis_title='Metric Value (Absolute)',
            barmode='group',
            height=500,
            showlegend=True,
            legend_title_text='Metric',
            xaxis=dict(autorange=True, gridcolor='rgba(255, 255, 255, 0.1)'),
            yaxis=dict(autorange=True, gridcolor='rgba(255, 255, 255, 0.1)')
        )

        return self._to_json(fig)


class GroupSizesChart(BaseChart):
    """
    Bar chart showing group size distribution from dataset_info.

    Shows sample count and percentage for each group within each attribute.
    """

    def create(self, data: Dict[str, Any]) -> str:
        """
        Create group sizes chart.

        Args:
            data: Dictionary with 'protected_attrs_distribution' and 'protected_attrs'

        Returns:
            Plotly JSON string
        """
        protected_attrs_distribution = data.get('protected_attrs_distribution', {})
        protected_attrs = data.get('protected_attrs', [])

        if not protected_attrs_distribution:
            return '{}'

        # Create subplots - one per attribute
        n_attrs = len(protected_attrs)
        fig = make_subplots(
            rows=1,
            cols=n_attrs,
            subplot_titles=[format_attribute_name(attr) for attr in protected_attrs],
            specs=[[{'type': 'bar'}] * n_attrs],
            horizontal_spacing=0.15
        )

        for i, attr in enumerate(protected_attrs, start=1):
            if attr not in protected_attrs_distribution:
                continue

            attr_data = protected_attrs_distribution[attr]
            distribution = attr_data.get('distribution', {})

            # Extract data
            group_names = list(distribution.keys())
            counts = [distribution[label]['count'] for label in group_names]
            percentages = [distribution[label]['percentage'] for label in group_names]

            if not group_names:
                continue

            # Color by percentage (gradient)
            max_pct = max(percentages) if percentages else 1.0
            colors = []
            for p in percentages:
                r = int(66 + (135 * p / max_pct))
                g = int(135 + (69 * p / max_pct))
                b = int(245 - (132 * p / max_pct))
                colors.append(f'rgba({r}, {g}, {b}, 0.8)')

            # Add bar chart
            fig.add_trace(
                go.Bar(
                    x=group_names,
                    y=counts,
                    marker=dict(
                        color=colors,
                        line=dict(color='white', width=1)
                    ),
                    text=[self._format_percentage(p) for p in percentages],
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>Count: %{y}<br>%{text}<extra></extra>',
                    showlegend=False,
                    cliponaxis=False
                ),
                row=1,
                col=i
            )

            # Update axes
            fig.update_xaxes(
                title_text="Group",
                gridcolor='rgba(255, 255, 255, 0.1)',
                row=1,
                col=i
            )
            fig.update_yaxes(
                title_text="Sample Count" if i == 1 else "",
                gridcolor='rgba(255, 255, 255, 0.1)',
                range=[0, max(counts) * 1.2] if counts else [0, 100],
                row=1,
                col=i
            )

        self._apply_common_layout(
            fig,
            title='Group Size Distribution - Sample Balance',
            height=450,
            showlegend=False,
            margin=dict(t=100, b=80)
        )

        return self._to_json(fig)


class ConceptBalanceChart(BaseChart):
    """
    Simple bar chart showing concept balance comparison.

    Uses group_a and group_b data from pretrain metrics.
    """

    def create(self, data: Dict[str, Any]) -> str:
        """
        Create concept balance chart.

        Args:
            data: Dictionary with 'pretrain_metrics' and 'protected_attrs'

        Returns:
            Plotly JSON string
        """
        pretrain_metrics = data.get('pretrain_metrics', {})
        protected_attrs = data.get('protected_attrs', [])

        if not pretrain_metrics:
            return '{}'

        # Collect data from concept_balance metrics
        chart_data = []
        for attr in protected_attrs:
            if attr not in pretrain_metrics or 'concept_balance' not in pretrain_metrics[attr]:
                continue

            metric = pretrain_metrics[attr]['concept_balance']
            if not isinstance(metric, dict):
                continue

            # Extract group data
            group_a = metric.get('group_a', 'Group A')
            group_b = metric.get('group_b', 'Group B')
            group_a_rate = metric.get('group_a_positive_rate', 0.0)
            group_b_rate = metric.get('group_b_positive_rate', 0.0)

            chart_data.append({
                'attribute': format_attribute_name(attr),
                'group': str(group_a).strip(),
                'rate': group_a_rate
            })
            chart_data.append({
                'attribute': format_attribute_name(attr),
                'group': str(group_b).strip(),
                'rate': group_b_rate
            })

        if not chart_data:
            return '{}'

        # Create grouped bar chart
        fig = go.Figure()

        # Group data by group name
        groups = {}
        for item in chart_data:
            group = item['group']
            if group not in groups:
                groups[group] = {'attributes': [], 'rates': []}
            groups[group]['attributes'].append(item['attribute'])
            groups[group]['rates'].append(item['rate'])

        # Add bars for each group
        for group, group_data in groups.items():
            text_values = [f'{r:.2%}' for r in group_data['rates']]

            fig.add_trace(go.Bar(
                name=group,
                x=group_data['attributes'],
                y=group_data['rates'],
                text=text_values,
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>%{fullData.name}<br>Rate: %{y}<extra></extra>'
            ))

        self._apply_common_layout(
            fig,
            title='Concept Balance - Positive Class Rate Comparison',
            xaxis_title='Protected Attribute',
            yaxis_title='Positive Class Rate',
            barmode='group',
            height=450,
            showlegend=True,
            legend_title_text='Group',
            xaxis=dict(autorange=True, gridcolor='rgba(255, 255, 255, 0.1)'),
            yaxis=dict(
                autorange=True,
                gridcolor='rgba(255, 255, 255, 0.1)',
                tickformat='.1%'
            )
        )

        return self._to_json(fig)
