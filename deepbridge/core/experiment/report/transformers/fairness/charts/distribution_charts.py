"""
Distribution charts for fairness analysis.

Contains visualizations for dataset distributions and demographics.
"""

from typing import Dict, Any, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .base_chart import BaseChart
from ..utils import format_attribute_name


class ProtectedAttributesDistributionChart(BaseChart):
    """
    Bar charts for protected attributes distribution.

    Shows count and percentage for each group within each protected attribute.
    """

    def create(self, data: Dict[str, Any]) -> str:
        """
        Create protected attributes distribution chart.

        Args:
            data: Dictionary with 'protected_attrs_distribution' and 'protected_attrs'

        Returns:
            Plotly JSON string
        """
        protected_attrs_distribution = data.get('protected_attrs_distribution', {})
        protected_attrs = data.get('protected_attrs', [])

        if not protected_attrs_distribution:
            return '{}'

        # Count attributes
        n_attrs = len(protected_attrs)
        if n_attrs == 0:
            return '{}'

        # Create subplots
        cols = min(n_attrs, 2)
        rows = (n_attrs + cols - 1) // cols

        subplot_titles = [format_attribute_name(attr) for attr in protected_attrs]

        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=subplot_titles,
            specs=[[{'type': 'bar'}] * cols for _ in range(rows)]
        )

        row, col = 1, 1
        for attr in protected_attrs:
            if attr not in protected_attrs_distribution:
                continue

            attr_data = protected_attrs_distribution[attr]
            distribution = attr_data.get('distribution', {})

            # Extract data
            labels = list(distribution.keys())
            counts = [distribution[label]['count'] for label in labels]
            percentages = [distribution[label]['percentage'] for label in labels]

            # Create hover text
            hover_text = [
                f"{label}<br>Count: {count:,}<br>Percentage: {pct:.1f}%"
                for label, count, pct in zip(labels, counts, percentages)
            ]

            # Add bar chart
            fig.add_trace(
                go.Bar(
                    x=labels,
                    y=counts,
                    text=[self._format_percentage(pct) for pct in percentages],
                    textposition='outside',
                    textangle=0,
                    hovertext=hover_text,
                    hoverinfo='text',
                    marker=dict(
                        color=percentages,
                        colorscale='Viridis',
                        showscale=False
                    ),
                    showlegend=False,
                    cliponaxis=False
                ),
                row=row,
                col=col
            )

            # Update axes with extended range to prevent text clipping
            fig.update_xaxes(title_text="Group", row=row, col=col)
            max_count = max(counts) if counts else 1
            fig.update_yaxes(
                title_text="Count",
                range=[0, max_count * 1.15],  # Add 15% extra space for text
                row=row,
                col=col
            )

            col += 1
            if col > cols:
                col = 1
                row += 1

        self._apply_common_layout(
            fig,
            title='Protected Attributes Distribution',
            height=350 * rows,
            showlegend=False,
            margin=dict(
                l=60,
                r=60,
                t=100,
                b=80
            ),
            uniformtext=dict(
                mode='hide',
                minsize=8
            )
        )

        return self._to_json(fig)


class TargetDistributionChart(BaseChart):
    """
    Pie chart for target variable distribution.

    Shows the distribution of classes in the target variable.
    """

    def create(self, data: Dict[str, Any]) -> str:
        """
        Create target distribution chart.

        Args:
            data: Dictionary with 'target_distribution'

        Returns:
            Plotly JSON string
        """
        target_distribution = data.get('target_distribution', {})

        if not target_distribution:
            return '{}'

        # Extract data
        labels = list(target_distribution.keys())
        counts = [target_distribution[label]['count'] for label in labels]

        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=[f"Class {label}" for label in labels],
            values=counts,
            textposition='inside',
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
            marker=dict(
                colors=[self.COLOR_INFO, self.COLOR_CRITICAL, self.COLOR_SUCCESS, self.COLOR_WARNING],
                line=dict(color='white', width=2)
            )
        )])

        self._apply_common_layout(
            fig,
            title='Target Variable Distribution',
            height=400,
            showlegend=True
        )

        return self._to_json(fig)
