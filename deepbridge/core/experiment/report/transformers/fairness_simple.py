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
        age_grouping_applied = results.get('age_grouping_applied', {})

        # Extract dataset information
        dataset_info = results.get('dataset_info', {})
        config = results.get('config', {})

        # Transform the data
        transformed = {
            'model_name': model_name,
            'model_type': 'Classification Model',

            # Summary metrics
            'summary': self._create_summary(results),

            # Protected attributes data (NEW FORMAT: split into 3 categories)
            'protected_attributes': self._transform_protected_attributes(
                protected_attrs, pretrain_metrics, posttrain_metrics
            ),

            # Issues and warnings
            'issues': self._transform_issues(warnings, critical_issues),

            # Dataset information (NEW)
            'dataset_info': self._transform_dataset_info(dataset_info),

            # Test configuration (NEW)
            'test_config': self._transform_test_config(config, age_grouping_applied),

            # Charts data (Plotly JSON)
            'charts': self._prepare_charts(results),

            # Metadata (includes age grouping info)
            'metadata': {
                'total_attributes': len(protected_attrs),
                'total_pretrain_metrics': sum(len(m) for m in pretrain_metrics.values()),
                'total_posttrain_metrics': sum(len(m) for m in posttrain_metrics.values()),
                'has_threshold_analysis': threshold_analysis is not None,
                'has_confusion_matrix': bool(confusion_matrix),
                'age_grouping_applied': age_grouping_applied,
                'age_grouping_enabled': len(age_grouping_applied) > 0
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
        """
        Transform protected attributes data.

        NEW FORMAT: Splits post-training metrics into:
        - posttrain_main: 5 critical compliance metrics
        - posttrain_complementary: 6 additional metrics
        """
        # Define metric categories
        POSTTRAIN_MAIN = [
            'statistical_parity',
            'equal_opportunity',
            'equalized_odds',
            'disparate_impact',
            'false_negative_rate_difference'
        ]

        POSTTRAIN_COMPLEMENTARY = [
            'conditional_acceptance',
            'conditional_rejection',
            'precision_difference',
            'accuracy_difference',
            'treatment_equality',
            'entropy_index'
        ]

        transformed_attrs = []

        for attr in attributes:
            attr_data = {
                'name': attr,
                'pretrain_metrics': [],
                'posttrain_main': [],
                'posttrain_complementary': []
            }

            # Transform pre-training metrics (unchanged)
            if attr in pretrain:
                for metric_name, metric_result in pretrain[attr].items():
                    if isinstance(metric_result, dict):
                        metric_data = {
                            'name': metric_name.replace('_', ' ').title(),
                            'value': metric_result.get('value', 0.0),
                            'interpretation': metric_result.get('interpretation', ''),
                            'status': self._get_status_from_interpretation(
                                metric_result.get('interpretation', '')
                            )
                        }

                        # Include groups if available
                        if 'all_groups' in metric_result:
                            metric_data['groups'] = metric_result['all_groups']

                        attr_data['pretrain_metrics'].append(metric_data)

            # Transform post-training metrics (SPLIT into main and complementary)
            if attr in posttrain:
                for metric_name, metric_result in posttrain[attr].items():
                    if isinstance(metric_result, dict):
                        # Base metric data
                        metric_data = {
                            'name': metric_name.replace('_', ' ').title(),
                            'interpretation': metric_result.get('interpretation', ''),
                            'status': self._get_status_from_interpretation(
                                metric_result.get('interpretation', '')
                            )
                        }

                        # Add disparity and ratio if available (FASE 1)
                        if 'disparity' in metric_result:
                            metric_data['disparity'] = metric_result.get('disparity')
                        if 'ratio' in metric_result:
                            metric_data['ratio'] = metric_result.get('ratio')
                        if 'value' in metric_result:
                            metric_data['value'] = metric_result.get('value')

                        # Add FASE 1 metadata (testable_groups, excluded_groups)
                        if 'testable_groups' in metric_result:
                            metric_data['testable_groups'] = metric_result['testable_groups']
                        if 'excluded_groups' in metric_result:
                            metric_data['excluded_groups'] = metric_result['excluded_groups']
                        if 'min_representation_pct' in metric_result:
                            metric_data['min_representation_pct'] = metric_result['min_representation_pct']

                        # Categorize into main or complementary
                        if metric_name in POSTTRAIN_MAIN:
                            attr_data['posttrain_main'].append(metric_data)
                        elif metric_name in POSTTRAIN_COMPLEMENTARY:
                            attr_data['posttrain_complementary'].append(metric_data)
                        else:
                            # Unknown metric, add to complementary by default
                            attr_data['posttrain_complementary'].append(metric_data)

            transformed_attrs.append(attr_data)

        return transformed_attrs

    def _get_status_from_interpretation(self, interpretation: str) -> str:
        """Extract status from interpretation string."""
        interp_upper = interpretation.upper()

        if ('✗' in interpretation or 'CRÍTICO' in interp_upper or 'CRITICAL' in interp_upper or
            'VERMELHO' in interp_upper or 'RED' in interp_upper or 'HIGH LEGAL RISK' in interp_upper):
            return 'critical'
        elif ('⚠' in interpretation or 'MODERADO' in interp_upper or 'MODERATE' in interp_upper or
              'AMARELO' in interp_upper or 'YELLOW' in interp_upper or 'WARNING' in interp_upper):
            return 'warning'
        elif ('✓' in interpretation or 'EXCELENTE' in interp_upper or 'EXCELLENT' in interp_upper or
              'BOM' in interp_upper or 'GOOD' in interp_upper or 'VERDE' in interp_upper or 'GREEN' in interp_upper):
            return 'success'
        else:
            return 'ok'

    def _transform_issues(self, warnings: List[str], critical: List[str]) -> Dict[str, List]:
        """Transform warnings and critical issues."""
        return {
            'warnings': warnings,
            'critical': critical,
            'total': len(warnings) + len(critical)
        }

    def _transform_dataset_info(self, dataset_info: Dict) -> Dict[str, Any]:
        """
        Transform dataset information for report display.

        Args:
            dataset_info: Dictionary containing dataset information

        Returns:
            Transformed dataset information with formatted distributions
        """
        if not dataset_info:
            return {
                'total_samples': 0,
                'target_distribution': {},
                'protected_attributes_distribution': {}
            }

        return {
            'total_samples': dataset_info.get('total_samples', 0),
            'target_distribution': dataset_info.get('target_distribution', {}),
            'protected_attributes_distribution': dataset_info.get('protected_attributes_distribution', {})
        }

    def _transform_test_config(self, config: Dict, age_grouping_applied: Dict) -> Dict[str, Any]:
        """
        Transform test configuration for report display.

        Args:
            config: Test configuration dictionary
            age_grouping_applied: Age grouping information

        Returns:
            Formatted test configuration information
        """
        if not config:
            return {}

        transformed_config = {
            'config_name': config.get('name', 'custom'),
            'metrics_tested': config.get('metrics_tested', []),
            'pretrain_enabled': config.get('include_pretrain', False),
            'confusion_matrix_enabled': config.get('include_confusion_matrix', False),
            'threshold_analysis_enabled': config.get('include_threshold_analysis', False),
            'age_grouping_enabled': config.get('age_grouping', False),
            'age_grouping_strategy': config.get('age_grouping_strategy'),
            'age_grouping_details': []
        }

        # Add age grouping details if applied
        if age_grouping_applied:
            for attr, info in age_grouping_applied.items():
                transformed_config['age_grouping_details'].append({
                    'attribute': attr,
                    'strategy': info.get('strategy'),
                    'original_range': info.get('original_range'),
                    'groups': info.get('groups', [])
                })

        return transformed_config

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

        # 5. Distribution charts (NEW)
        dataset_info = results.get('dataset_info', {})
        if dataset_info and dataset_info.get('protected_attributes_distribution'):
            charts['protected_attributes_distribution'] = self._create_distribution_charts(
                dataset_info.get('protected_attributes_distribution', {}),
                results.get('protected_attributes', [])
            )

        # 6. Target distribution chart (NEW)
        if dataset_info and dataset_info.get('target_distribution'):
            charts['target_distribution'] = self._create_target_distribution_chart(
                dataset_info.get('target_distribution', {})
            )

        # 7. Post-Training Charts - FASE 1 (NEW)
        posttrain_metrics = results.get('posttrain_metrics', {})
        if posttrain_metrics:
            # Disparate Impact Gauge (CRITICAL LEGAL METRIC)
            charts['posttrain_disparate_impact_gauge'] = self._create_posttrain_disparate_impact_gauge(
                posttrain_metrics,
                results.get('protected_attributes', [])
            )

            # Statistical Parity Disparity Comparison
            charts['posttrain_disparity_comparison'] = self._create_posttrain_disparity_comparison(
                posttrain_metrics,
                results.get('protected_attributes', [])
            )

            # Compliance Status Matrix
            charts['posttrain_status_matrix'] = self._create_posttrain_status_matrix(
                posttrain_metrics,
                results.get('protected_attributes', [])
            )

        # 8. Pre-Training Charts - FASE 2 (NEW)
        pretrain_metrics = results.get('pretrain_metrics', {})
        if pretrain_metrics:
            # Metrics Overview - All 4 pre-training metrics
            charts['pretrain_metrics_overview'] = self._create_pretrain_metrics_overview(
                pretrain_metrics,
                results.get('protected_attributes', [])
            )

        # Group Sizes - Use dataset_info instead of pretrain_metrics
        if dataset_info and dataset_info.get('protected_attributes_distribution'):
            charts['pretrain_group_sizes'] = self._create_pretrain_group_sizes_from_dataset(
                dataset_info.get('protected_attributes_distribution', {}),
                results.get('protected_attributes', [])
            )

        # Concept Balance - Use pretrain_metrics with simplified logic
        if pretrain_metrics:
            charts['pretrain_concept_balance'] = self._create_pretrain_concept_balance_simple(
                pretrain_metrics,
                results.get('protected_attributes', [])
            )

        # 9. Complementary Charts - FASE 3 (NEW)
        confusion_matrix = results.get('confusion_matrix', {})
        if confusion_matrix and posttrain_metrics:
            # Precision & Accuracy Comparison
            charts['complementary_precision_accuracy'] = self._create_complementary_precision_accuracy(
                posttrain_metrics,
                confusion_matrix,
                results.get('protected_attributes', [])
            )

            # Treatment Equality Scatter
            charts['complementary_treatment_equality'] = self._create_complementary_treatment_equality(
                confusion_matrix,
                results.get('protected_attributes', [])
            )

        # Complementary Radar Chart
        if posttrain_metrics:
            charts['complementary_radar'] = self._create_complementary_radar(
                posttrain_metrics,
                results.get('protected_attributes', [])
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

    def _create_distribution_charts(
        self,
        protected_attrs_distribution: Dict[str, Dict],
        protected_attrs: List[str]
    ) -> str:
        """
        Create bar charts for protected attributes distribution.

        Args:
            protected_attrs_distribution: Distribution data for each protected attribute
            protected_attrs: List of protected attribute names

        Returns:
            Plotly JSON string
        """
        if not protected_attrs_distribution:
            return '{}'

        # Count attributes
        n_attrs = len(protected_attrs)
        if n_attrs == 0:
            return '{}'

        # Create subplots
        cols = min(n_attrs, 2)
        rows = (n_attrs + cols - 1) // cols

        subplot_titles = [attr.replace('_', ' ').title() for attr in protected_attrs]

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
                    text=[f"{pct:.1f}%" for pct in percentages],
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

        fig.update_layout(
            height=350 * rows,
            title='Protected Attributes Distribution',
            showlegend=False,
            margin=dict(
                l=60,   # Left margin
                r=60,   # Right margin
                t=100,  # Top margin (for title and percentages)
                b=80    # Bottom margin (for x-axis labels)
            ),
            uniformtext=dict(
                mode='hide',
                minsize=8
            )
        )

        return pio.to_json(fig)

    def _create_target_distribution_chart(self, target_distribution: Dict) -> str:
        """
        Create pie chart for target variable distribution.

        Args:
            target_distribution: Distribution data for target variable

        Returns:
            Plotly JSON string
        """
        if not target_distribution:
            return '{}'

        # Extract data
        labels = list(target_distribution.keys())
        counts = [target_distribution[label]['count'] for label in labels]
        percentages = [target_distribution[label]['percentage'] for label in labels]

        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=[f"Class {label}" for label in labels],
            values=counts,
            textposition='inside',
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent}<extra></extra>',
            marker=dict(
                colors=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'],
                line=dict(color='white', width=2)
            )
        )])

        fig.update_layout(
            title='Target Variable Distribution',
            height=400,
            showlegend=True
        )

        return pio.to_json(fig)

    def _create_posttrain_disparate_impact_gauge(
        self,
        posttrain_metrics: Dict[str, Dict],
        protected_attrs: List[str]
    ) -> str:
        """
        Create gauge chart for Disparate Impact metric (EEOC 80% Rule).

        Shows compliance with EEOC 80% rule - most critical legal metric.

        Args:
            posttrain_metrics: Post-training metrics dictionary
            protected_attrs: List of protected attribute names

        Returns:
            Plotly JSON string
        """
        if not posttrain_metrics:
            return '{}'

        # Extract disparate impact data for each attribute
        data = []
        for attr in protected_attrs:
            if attr in posttrain_metrics:
                di_metric = posttrain_metrics[attr].get('disparate_impact', {})
                if isinstance(di_metric, dict) and 'ratio' in di_metric:
                    ratio = di_metric.get('ratio', 0.0)
                    passes = di_metric.get('passes_80_rule', False)

                    data.append({
                        'attribute': attr,
                        'ratio': ratio,
                        'passes': passes
                    })

        if not data:
            return '{}'

        # Create subplot for each attribute (side by side gauges)
        n_attrs = len(data)
        fig = make_subplots(
            rows=1,
            cols=n_attrs,
            subplot_titles=[f"{d['attribute'].replace('_', ' ').title()}" for d in data],
            horizontal_spacing=0.1,
            specs=[[{'type': 'indicator'} for _ in range(n_attrs)]]
        )

        for i, item in enumerate(data, start=1):
            ratio = item['ratio']
            passes = item['passes']

            # Color based on compliance
            if ratio >= 0.8:
                color = '#2ecc71'  # Green - compliant
            elif ratio >= 0.7:
                color = '#f39c12'  # Yellow - warning
            else:
                color = '#e74c3c'  # Red - critical

            # Create gauge
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=ratio,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': f"Ratio: {ratio:.3f}"},
                    delta={'reference': 0.8, 'increasing': {'color': '#2ecc71'}, 'decreasing': {'color': '#e74c3c'}},
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

        fig.update_layout(
            title={
                'text': 'Disparate Impact - EEOC 80% Rule Compliance ⚖️',
                'font': {'size': 20, 'color': 'white'}
            },
            height=400,
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'}
        )

        return pio.to_json(fig)

    def _create_posttrain_disparity_comparison(
        self,
        posttrain_metrics: Dict[str, Dict],
        protected_attrs: List[str]
    ) -> str:
        """
        Create diverging bar chart for Statistical Parity disparity values.

        Shows how far each attribute deviates from perfect fairness (0.0).

        Args:
            posttrain_metrics: Post-training metrics dictionary
            protected_attrs: List of protected attribute names

        Returns:
            Plotly JSON string
        """
        if not posttrain_metrics:
            return '{}'

        # Extract statistical parity disparity for each attribute
        data = []
        for attr in protected_attrs:
            if attr in posttrain_metrics:
                sp_metric = posttrain_metrics[attr].get('statistical_parity', {})
                if isinstance(sp_metric, dict) and 'disparity' in sp_metric:
                    disparity = sp_metric.get('disparity', 0.0)
                    interpretation = sp_metric.get('interpretation', '')
                    status = self._get_status_from_interpretation(interpretation)

                    data.append({
                        'attribute': attr.replace('_', ' ').title(),
                        'disparity': disparity,
                        'status': status
                    })

        if not data:
            return '{}'

        df = pd.DataFrame(data)

        # Color mapping
        color_map = {
            'success': '#2ecc71',
            'ok': '#2ecc71',
            'warning': '#f39c12',
            'critical': '#e74c3c'
        }

        colors = [color_map.get(status, '#95a5a6') for status in df['status']]

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
            text=[f"{d:.4f}" for d in df['disparity']],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Disparity: %{x:.4f}<extra></extra>'
        ))

        # Add reference lines
        fig.add_vline(x=0, line_dash="solid", line_color="white", line_width=2)
        fig.add_vline(x=0.1, line_dash="dash", line_color="#f39c12", line_width=1,
                     annotation_text="Warning Threshold", annotation_position="top")
        fig.add_vline(x=-0.1, line_dash="dash", line_color="#f39c12", line_width=1)

        fig.update_layout(
            title='Statistical Parity - Disparity Analysis',
            xaxis_title='Disparity (0.0 = Perfect Fairness)',
            yaxis_title='Protected Attribute',
            height=max(300, 100 * len(data)),
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'},
            xaxis=dict(
                gridcolor='rgba(255, 255, 255, 0.1)',
                zerolinecolor='white',
                zerolinewidth=2
            ),
            yaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)')
        )

        return pio.to_json(fig)

    def _create_posttrain_status_matrix(
        self,
        posttrain_metrics: Dict[str, Dict],
        protected_attrs: List[str]
    ) -> str:
        """
        Create heatmap matrix showing status of all post-training metrics.

        Executive dashboard view with color-coded compliance status.

        Args:
            posttrain_metrics: Post-training metrics dictionary
            protected_attrs: List of protected attribute names

        Returns:
            Plotly JSON string
        """
        if not posttrain_metrics:
            return '{}'

        # Define the 5 main post-training metrics
        main_metrics = [
            'statistical_parity',
            'equal_opportunity',
            'equalized_odds',
            'disparate_impact',
            'false_negative_rate_difference'
        ]

        # Short labels for display
        metric_labels = {
            'statistical_parity': 'Statistical<br>Parity',
            'equal_opportunity': 'Equal<br>Opportunity',
            'equalized_odds': 'Equalized<br>Odds',
            'disparate_impact': 'Disparate<br>Impact ⚖️',
            'false_negative_rate_difference': 'FNR<br>Difference'
        }

        # Build matrix data
        matrix = []
        text_matrix = []
        attr_labels = []

        for attr in protected_attrs:
            row = []
            text_row = []
            attr_labels.append(attr.replace('_', ' ').title())

            for metric in main_metrics:
                if attr in posttrain_metrics and metric in posttrain_metrics[attr]:
                    metric_data = posttrain_metrics[attr][metric]
                    if isinstance(metric_data, dict):
                        interpretation = metric_data.get('interpretation', '')
                        status = self._get_status_from_interpretation(interpretation)

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
            x=[metric_labels[m] for m in main_metrics],
            y=attr_labels,
            text=text_matrix,
            texttemplate='<b>%{text}</b>',
            textfont={'size': 20, 'color': 'white'},
            colorscale=[
                [0.0, '#e74c3c'],    # Critical - Red
                [0.5, '#f39c12'],    # Warning - Yellow
                [1.0, '#2ecc71']     # Success - Green
            ],
            showscale=False,
            hovertemplate='<b>%{y}</b><br>%{x}<br>Status: %{text}<extra></extra>'
        ))

        fig.update_layout(
            title='Compliance Status Matrix - Main Post-Training Metrics',
            height=max(300, 80 * len(protected_attrs)),
            xaxis=dict(
                tickfont={'size': 11},
                side='top'
            ),
            yaxis=dict(
                tickfont={'size': 12}
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'}
        )

        return pio.to_json(fig)

    def _create_pretrain_concept_balance_simple(
        self,
        pretrain_metrics: Dict[str, Dict],
        protected_attrs: List[str]
    ) -> str:
        """
        Create simple bar chart showing concept balance comparison.

        Uses group_a and group_b data from pretrain metrics.

        Args:
            pretrain_metrics: Pre-training metrics dictionary
            protected_attrs: List of protected attribute names

        Returns:
            Plotly JSON string
        """
        if not pretrain_metrics:
            return '{}'

        # Collect data from concept_balance metrics
        data = []
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

            data.append({
                'attribute': attr.replace('_', ' ').title(),
                'group': str(group_a).strip(),
                'rate': group_a_rate
            })
            data.append({
                'attribute': attr.replace('_', ' ').title(),
                'group': str(group_b).strip(),
                'rate': group_b_rate
            })

        if not data:
            return '{}'

        df = pd.DataFrame(data)

        # Create grouped bar chart with explicit text formatting
        fig = go.Figure()

        # Get unique groups and attributes
        groups = df['group'].unique()
        attributes = df['attribute'].unique()

        # Add bars for each group
        for group in groups:
            group_data = df[df['group'] == group]
            # Convert to Python lists to avoid binary serialization
            x_values = group_data['attribute'].tolist()
            y_values = group_data['rate'].tolist()
            text_values = [f'{r:.2%}' for r in y_values]

            fig.add_trace(go.Bar(
                name=group,
                x=x_values,
                y=y_values,
                text=text_values,
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>%{fullData.name}<br>Rate: %{y:.2%}<extra></extra>'
            ))

        fig.update_layout(
            title='Concept Balance - Positive Class Rate Comparison',
            xaxis_title='Protected Attribute',
            yaxis_title='Positive Class Rate',
            barmode='group',
            height=450,
            showlegend=True,
            legend_title_text='Group',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'},
            xaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)'),
            yaxis=dict(
                gridcolor='rgba(255, 255, 255, 0.1)',
                tickformat='.1%'
            )
        )

        return pio.to_json(fig)

    def _create_pretrain_metrics_overview(
        self,
        pretrain_metrics: Dict[str, Dict],
        protected_attrs: List[str]
    ) -> str:
        """
        Create grouped bar chart for all pre-training metrics.

        Shows all 4 pre-training metrics (BCL, BCO, KL, JS) for each attribute.

        Args:
            pretrain_metrics: Pre-training metrics dictionary
            protected_attrs: List of protected attribute names

        Returns:
            Plotly JSON string
        """
        if not pretrain_metrics:
            return '{}'

        # Define pre-training metrics to show
        pretrain_metric_names = [
            'class_balance',
            'concept_balance',
            'kl_divergence',
            'js_divergence'
        ]

        metric_labels = {
            'class_balance': 'Class Balance (BCL)',
            'concept_balance': 'Concept Balance (BCO)',
            'kl_divergence': 'KL Divergence',
            'js_divergence': 'JS Divergence'
        }

        # Color mapping
        color_map = {
            'success': '#2ecc71',
            'ok': '#2ecc71',
            'warning': '#f39c12',
            'critical': '#e74c3c'
        }

        # Create figure
        fig = go.Figure()

        # Collect data per metric
        for metric_name in pretrain_metric_names:
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
                status = self._get_status_from_interpretation(interpretation)

                x_data.append(attr.replace('_', ' ').title())
                y_data.append(value)
                colors.append(color_map.get(status, '#95a5a6'))
                text_data.append(f'{value:.4f}')

            if x_data:
                fig.add_trace(go.Bar(
                    name=metric_labels[metric_name],
                    x=x_data,
                    y=y_data,
                    text=text_data,
                    textposition='outside',
                    marker=dict(color=colors),
                    hovertemplate='<b>%{x}</b><br>' + metric_labels[metric_name] + '<br>Value: %{y:.4f}<extra></extra>'
                ))

        fig.update_layout(
            title='Pre-Training Metrics Overview (All 4 Metrics)',
            xaxis_title='Protected Attribute',
            yaxis_title='Metric Value (Absolute)',
            barmode='group',
            height=500,
            showlegend=True,
            legend_title_text='Metric',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'},
            xaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)'),
            yaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)')
        )

        return pio.to_json(fig)

    def _create_pretrain_group_sizes_from_dataset(
        self,
        protected_attrs_distribution: Dict[str, Dict],
        protected_attrs: List[str]
    ) -> str:
        """
        Create bar chart showing group size distribution from dataset_info.

        Shows sample count and percentage for each group within each attribute.

        Args:
            protected_attrs_distribution: Distribution data from dataset_info
            protected_attrs: List of protected attribute names

        Returns:
            Plotly JSON string
        """
        if not protected_attrs_distribution:
            return '{}'

        # Create subplots - one per attribute
        n_attrs = len(protected_attrs)
        fig = make_subplots(
            rows=1,
            cols=n_attrs,
            subplot_titles=[attr.replace('_', ' ').title() for attr in protected_attrs],
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
                    text=[f"{p:.1f}%" for p in percentages],
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>Count: %{y:,}<br>%{text}<extra></extra>',
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

        fig.update_layout(
            title='Group Size Distribution - Sample Balance',
            height=450,
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'},
            margin=dict(t=100, b=80)
        )

        return pio.to_json(fig)

    def _create_complementary_precision_accuracy(
        self,
        posttrain_metrics: Dict[str, Dict],
        confusion_matrix: Dict[str, Dict],
        protected_attrs: List[str]
    ) -> str:
        """
        Create grouped bar chart comparing precision and accuracy by group.

        Shows performance metrics (precision, accuracy) for each group within
        each protected attribute.

        Args:
            posttrain_metrics: Post-training metrics dictionary
            confusion_matrix: Confusion matrix data by attribute and group
            protected_attrs: List of protected attribute names

        Returns:
            Plotly JSON string
        """
        if not confusion_matrix:
            return '{}'

        # Collect data from confusion matrices
        data = []
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

                data.append({
                    'attribute': attr.replace('_', ' ').title(),
                    'group': str(group),
                    'metric': 'Accuracy',
                    'value': accuracy
                })
                data.append({
                    'attribute': attr.replace('_', ' ').title(),
                    'group': str(group),
                    'metric': 'Precision',
                    'value': precision
                })

        if not data:
            return '{}'

        df = pd.DataFrame(data)

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
            color_discrete_map={'Accuracy': '#3498db', 'Precision': '#9b59b6'}
        )

        fig.update_layout(
            height=400,
            showlegend=True,
            legend_title_text='Metric',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'},
            yaxis=dict(
                gridcolor='rgba(255, 255, 255, 0.1)',
                tickformat='.0%',
                range=[0, 1]
            )
        )

        fig.update_xaxes(gridcolor='rgba(255, 255, 255, 0.1)')

        return pio.to_json(fig)

    def _create_complementary_treatment_equality(
        self,
        confusion_matrix: Dict[str, Dict],
        protected_attrs: List[str]
    ) -> str:
        """
        Create scatter plot showing treatment equality (FN vs FP rates).

        Shows if errors are balanced between groups - ideal is on diagonal.

        Args:
            confusion_matrix: Confusion matrix data by attribute and group
            protected_attrs: List of protected attribute names

        Returns:
            Plotly JSON string
        """
        if not confusion_matrix:
            return '{}'

        # Collect data from confusion matrices
        data = []
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

                data.append({
                    'attribute': attr.replace('_', ' ').title(),
                    'group': str(group),
                    'fn_rate': fn_rate,
                    'fp_rate': fp_rate,
                    'sample_size': sample_size
                })

        if not data:
            return '{}'

        df = pd.DataFrame(data)

        # Create scatter plot with go.Scatter for better control
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
            custom_data = [[row['group'], row['sample_size']] for _, row in attr_data.iterrows()]

            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode='markers+text',
                    name=attr,
                    text=text_labels,
                    textposition='top center',
                    textfont=dict(color='white', size=10),
                    marker=dict(
                        size=marker_sizes,
                        color=colors[i % len(colors)],
                        opacity=0.7,
                        line=dict(color='white', width=1)
                    ),
                    hovertemplate='<b>%{text}</b><br>' +
                                'FP Rate: %{x:.1%}<br>' +
                                'FN Rate: %{y:.1%}<br>' +
                                '<extra></extra>',
                    customdata=custom_data
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

        fig.update_layout(
            title='Treatment Equality - Error Balance Analysis',
            height=500,
            showlegend=True,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'},
            xaxis=dict(
                title='False Positive Rate',
                gridcolor='rgba(255, 255, 255, 0.1)',
                tickformat='.0%',
                range=[0, max_rate * 1.1] if max_rate > 0 else [0, 1]
            ),
            yaxis=dict(
                title='False Negative Rate',
                gridcolor='rgba(255, 255, 255, 0.1)',
                tickformat='.0%',
                range=[0, max_rate * 1.1] if max_rate > 0 else [0, 1]
            )
        )

        return pio.to_json(fig)

    def _create_complementary_radar(
        self,
        posttrain_metrics: Dict[str, Dict],
        protected_attrs: List[str]
    ) -> str:
        """
        Create radar chart for complementary metrics.

        Shows profile of 6 complementary fairness metrics for each attribute.

        Args:
            posttrain_metrics: Post-training metrics dictionary
            protected_attrs: List of protected attribute names

        Returns:
            Plotly JSON string
        """
        if not posttrain_metrics:
            return '{}'

        # Define complementary metrics
        complementary_metrics = [
            'conditional_acceptance',
            'conditional_rejection',
            'precision_difference',
            'accuracy_difference',
            'treatment_equality',
            'entropy_index'
        ]

        metric_labels = {
            'conditional_acceptance': 'Conditional<br>Acceptance',
            'conditional_rejection': 'Conditional<br>Rejection',
            'precision_difference': 'Precision<br>Difference',
            'accuracy_difference': 'Accuracy<br>Difference',
            'treatment_equality': 'Treatment<br>Equality',
            'entropy_index': 'Entropy<br>Index'
        }

        fig = go.Figure()

        for attr in protected_attrs:
            if attr not in posttrain_metrics:
                continue

            values = []
            labels = []

            for metric in complementary_metrics:
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
                labels.append(metric_labels.get(metric, metric))

            if values:
                # Close the polygon
                values.append(values[0])
                labels.append(labels[0])

                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=labels,
                    fill='toself',
                    name=attr.replace('_', ' ').title()
                ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickformat='.0%'
                )
            ),
            showlegend=True,
            title='Complementary Metrics Radar (1.0 = Perfect Fairness)',
            height=500,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'}
        )

        return pio.to_json(fig)
