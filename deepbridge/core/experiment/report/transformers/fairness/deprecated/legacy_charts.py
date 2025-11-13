"""
Legacy fairness charts.

Contains older chart implementations for backward compatibility.
"""

from typing import Dict, Any, List
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

from ..charts.base_chart import BaseChart
from ..utils import get_status_from_interpretation, format_attribute_name


class MetricsComparisonChart(BaseChart):
    """
    Metrics comparison bar chart.

    Legacy chart showing all post-training metrics side by side.
    """

    def create(self, data: Dict[str, Any]) -> str:
        """
        Create metrics comparison chart.

        Args:
            data: Dictionary with 'posttrain_metrics' and 'protected_attrs'

        Returns:
            Plotly JSON string
        """
        posttrain_metrics = data.get('posttrain_metrics', {})
        protected_attrs = data.get('protected_attrs', [])

        if not posttrain_metrics:
            return '{}'

        # Only show MAIN post-training metrics (not complementary)
        main_metrics = [
            'statistical_parity',
            'equal_opportunity',
            'equalized_odds',
            'disparate_impact',
            'false_negative_rate_difference'
        ]

        # Prepare data
        chart_data = []
        for attr in protected_attrs:
            if attr in posttrain_metrics:
                for metric_name, metric_result in posttrain_metrics[attr].items():
                    # Only include main metrics
                    if metric_name not in main_metrics:
                        continue

                    if isinstance(metric_result, dict):
                        # Get the appropriate value based on metric type
                        if metric_name == 'disparate_impact':
                            # Disparate impact uses 'ratio', convert to disparity-like scale
                            ratio = metric_result.get('ratio', 1.0)
                            # Convert ratio to disparity: how far from 1.0 (perfect fairness)
                            value = abs(1.0 - ratio)
                        elif 'disparity' in metric_result:
                            # Most metrics use 'disparity'
                            value = abs(metric_result.get('disparity', 0))
                        elif 'value' in metric_result:
                            # Fallback to 'value'
                            value = abs(metric_result.get('value', 0))
                        else:
                            continue

                        # Ensure value is float and normalized
                        value = float(value)

                        chart_data.append({
                            'attribute': attr,
                            'metric': metric_name.replace('_', ' ').title(),
                            'value': value,
                            'status': get_status_from_interpretation(
                                metric_result.get('interpretation', '')
                            )
                        })

        if not chart_data:
            return '{}'

        df = pd.DataFrame(chart_data)

        # Create figure
        fig = px.bar(
            df,
            x='value',
            y='metric',
            color='status',
            facet_col='attribute',
            color_discrete_map=self.COLOR_MAP_STATUS,
            labels={'value': 'Disparity', 'metric': 'Fairness Metric'},
            title='Fairness Metrics Comparison by Protected Attribute',
            orientation='h'
        )

        # Update hover template to show values with 4 decimal places
        fig.update_traces(
            hovertemplate='<b>%{y}</b><br>Disparity: %{x:.4f}<extra></extra>'
        )

        # Update all xaxis titles to "Disparity" and constrain range
        # Set a maximum range to prevent auto-scaling beyond 1.0
        max_value = df['value'].max()
        # Use the larger of: actual max value + 10% padding, or 0.3 (minimum for readability)
        xaxis_max = max(max_value * 1.1, 0.3)

        fig.update_xaxes(
            title_text="Disparity",
            range=[0, xaxis_max]  # Constrain to reasonable disparity scale
        )

        # Add reference line at 0.1 (recommended threshold) for all subplots
        fig.add_vline(x=0.1, line_dash="dash", line_color="gray", opacity=0.5)

        # Apply common layout last (so it doesn't override our customizations)
        self._apply_common_layout(
            fig,
            height=500,
            showlegend=True,
            legend_title_text='Status'
        )

        return self._to_json(fig)


class FairnessRadarChart(BaseChart):
    """
    Radar chart for fairness metrics.

    Legacy overview chart showing key metrics in radar format.
    """

    def create(self, data: Dict[str, Any]) -> str:
        """
        Create fairness radar chart.

        Args:
            data: Dictionary with 'posttrain_metrics'

        Returns:
            Plotly JSON string
        """
        posttrain_metrics = data.get('posttrain_metrics', {})

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
                    # Get the appropriate value based on metric type
                    if metric == 'disparate_impact':
                        value = metrics[metric].get('ratio', 0)
                    else:
                        value = metrics[metric].get('value', 0)

                    # Normalize for radar (closer to 1 = better fairness)
                    if metric == 'disparate_impact':
                        # Disparate impact: ratio closer to 1.0 = better fairness
                        normalized = min(abs(value), 1.0)
                    else:
                        # Other metrics: smaller absolute value = better fairness
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
                    name=format_attribute_name(attr)
                ))

        self._apply_common_layout(
            fig,
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

        return self._to_json(fig)


class ConfusionMatricesChart(BaseChart):
    """
    Heatmap visualization of confusion matrices.

    Shows confusion matrices for each group within each protected attribute.
    """

    def create(self, data: Dict[str, Any]) -> str:
        """
        Create confusion matrices chart.

        Args:
            data: Dictionary with 'confusion_matrix' and 'protected_attrs'

        Returns:
            Plotly JSON string
        """
        confusion_matrices = data.get('confusion_matrix', {})
        protected_attrs = data.get('protected_attrs', [])

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

        self._apply_common_layout(
            fig,
            height=250 * rows,
            title='Confusion Matrices by Group',
            showlegend=False
        )

        return self._to_json(fig)


class ThresholdAnalysisChart(BaseChart):
    """
    Threshold impact chart.

    Shows how different thresholds affect fairness and performance metrics.
    """

    def create(self, data: Dict[str, Any]) -> str:
        """
        Create threshold analysis chart.

        Args:
            data: Dictionary with 'threshold_analysis'

        Returns:
            Plotly JSON string
        """
        threshold_analysis = data.get('threshold_analysis', {})

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
                name='Disparate Impact Ratio',
                line=dict(color=self.COLOR_INFO, width=3)
            ))

        if 'f1_score' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['threshold'],
                y=df['f1_score'],
                mode='lines',
                name='F1 Score (Performance)',
                line=dict(color='#9B59B6', width=3)
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

        self._apply_common_layout(
            fig,
            title='Threshold Impact: Fairness vs Performance Trade-off',
            xaxis_title='Classification Threshold',
            yaxis_title='Metric Value',
            height=500,
            showlegend=True,
            hovermode='x unified',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            )
        )

        return self._to_json(fig)
