"""
Fairness Visualizations for DeepBridge.

This module provides visualization tools for fairness analysis,
generating publication-ready charts for fairness metrics, threshold analysis,
confusion matrices, and group comparisons.

Main class:
    FairnessVisualizer: Static methods for generating fairness visualizations

Available visualizations:
    1. Distribution by Group - Target distribution across protected groups
    2. Metrics Comparison - Bar chart comparing all fairness metrics
    3. Threshold Impact - Line charts showing fairness vs threshold
    4. Confusion Matrices - Side-by-side confusion matrices by group
    5. Fairness Radar - Radar chart with all fairness dimensions
    6. Group Comparison - Detailed comparison of model performance by group
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
import warnings

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("matplotlib not available. Visualizations will not work.")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    warnings.warn("seaborn not available. Some visualizations may be limited.")


class FairnessVisualizer:
    """
    Static methods for generating fairness visualizations.

    All methods return the path to the saved figure (or display it if output_path is None).
    Supports PNG, SVG, and PDF output formats.

    Example:
        >>> from deepbridge.validation.fairness.visualizations import FairnessVisualizer
        >>>
        >>> # After running fairness tests
        >>> FairnessVisualizer.plot_metrics_comparison(
        ...     results['posttrain_metrics'],
        ...     protected_attrs=['gender', 'race'],
        ...     output_path='fairness_metrics.png'
        ... )
    """

    # Color schemes
    COLORS = {
        'green': '#2ecc71',
        'yellow': '#f39c12',
        'red': '#e74c3c',
        'blue': '#3498db',
        'purple': '#9b59b6',
        'gray': '#95a5a6'
    }

    METRIC_LABELS = {
        'statistical_parity': 'Statistical Parity',
        'equal_opportunity': 'Equal Opportunity',
        'equalized_odds': 'Equalized Odds',
        'disparate_impact': 'Disparate Impact',
        'false_negative_rate_difference': 'FNR Difference',
        'conditional_acceptance': 'Conditional Accept',
        'conditional_rejection': 'Conditional Reject',
        'precision_difference': 'Precision Diff',
        'accuracy_difference': 'Accuracy Diff',
        'treatment_equality': 'Treatment Equality',
        'entropy_index': 'Entropy Index',
        'class_balance': 'Class Balance',
        'concept_balance': 'Concept Balance',
        'kl_divergence': 'KL Divergence',
        'js_divergence': 'JS Divergence'
    }

    @staticmethod
    def _check_dependencies():
        """Check if required visualization libraries are available"""
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError(
                "matplotlib is required for visualizations. "
                "Install it with: pip install matplotlib"
            )

    @staticmethod
    def _get_color_for_interpretation(interpretation: str) -> str:
        """Get color based on interpretation text"""
        if '✓ Verde' in interpretation or 'EXCELENTE' in interpretation or 'BOM' in interpretation:
            return FairnessVisualizer.COLORS['green']
        elif '⚠ Amarelo' in interpretation or 'MODERADO' in interpretation:
            return FairnessVisualizer.COLORS['yellow']
        elif '✗ Vermelho' in interpretation or 'CRÍTICO' in interpretation:
            return FairnessVisualizer.COLORS['red']
        else:
            return FairnessVisualizer.COLORS['gray']

    @staticmethod
    def _save_or_show(fig, output_path: Optional[str] = None, dpi: int = 300):
        """Save figure to file or display it"""
        if output_path:
            fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            return output_path
        else:
            plt.tight_layout()
            plt.show()
            return None

    @staticmethod
    def plot_distribution_by_group(
        df: pd.DataFrame,
        target_col: str,
        sensitive_feature: str,
        output_path: Optional[str] = None,
        title: Optional[str] = None,
        figsize: tuple = (12, 6)
    ) -> Optional[str]:
        """
        Plot target distribution by protected group.

        Shows both overall distribution and positive rate by group to visualize
        potential bias in the data.

        Parameters:
        -----------
        df : DataFrame
            Dataset with target and sensitive features
        target_col : str
            Name of target column
        sensitive_feature : str
            Name of protected attribute column
        output_path : str, optional
            Path to save figure. If None, displays instead.
        title : str, optional
            Custom title for the plot
        figsize : tuple, default=(12, 6)
            Figure size (width, height)

        Returns:
        --------
        str or None : Path to saved file, or None if displayed

        Example:
        --------
        >>> FairnessVisualizer.plot_distribution_by_group(
        ...     df, 'loan_approved', 'gender',
        ...     output_path='distribution.png'
        ... )
        """
        FairnessVisualizer._check_dependencies()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot 1: Count by group
        df_counts = df.groupby([sensitive_feature, target_col]).size().unstack(fill_value=0)
        df_counts.plot(kind='bar', stacked=False, ax=ax1, color=['#e74c3c', '#2ecc71'])
        ax1.set_title(f'Distribution of {target_col} by {sensitive_feature}', fontsize=14, fontweight='bold')
        ax1.set_xlabel(sensitive_feature, fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.legend(['Negative (0)', 'Positive (1)'], loc='upper right')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)

        # Plot 2: Positive rate by group
        positive_rates = df.groupby(sensitive_feature)[target_col].mean()
        bars = ax2.bar(positive_rates.index, positive_rates.values,
                       color=FairnessVisualizer.COLORS['blue'], alpha=0.7, edgecolor='black')

        # Add overall mean line
        overall_mean = df[target_col].mean()
        ax2.axhline(y=overall_mean, color=FairnessVisualizer.COLORS['red'],
                   linestyle='--', linewidth=2, label=f'Overall Mean ({overall_mean:.2%})')

        ax2.set_title(f'Positive Rate by {sensitive_feature}', fontsize=14, fontweight='bold')
        ax2.set_xlabel(sensitive_feature, fontsize=12)
        ax2.set_ylabel('Positive Rate', fontsize=12)
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2%}', ha='center', va='bottom', fontweight='bold')

        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)

        return FairnessVisualizer._save_or_show(fig, output_path)

    @staticmethod
    def plot_metrics_comparison(
        metrics_results: Dict[str, Dict[str, Any]],
        protected_attrs: List[str],
        output_path: Optional[str] = None,
        title: Optional[str] = None,
        figsize: tuple = (14, 8)
    ) -> Optional[str]:
        """
        Plot comparison of all fairness metrics across protected attributes.

        Creates horizontal bar chart with color-coded bars based on interpretation.

        Parameters:
        -----------
        metrics_results : Dict
            Post-training metrics results (from fairness_suite results['posttrain_metrics'])
        protected_attrs : List[str]
            List of protected attributes tested
        output_path : str, optional
            Path to save figure
        title : str, optional
            Custom title
        figsize : tuple, default=(14, 8)
            Figure size

        Returns:
        --------
        str or None : Path to saved file

        Example:
        --------
        >>> FairnessVisualizer.plot_metrics_comparison(
        ...     results['posttrain_metrics'],
        ...     ['gender', 'race'],
        ...     output_path='metrics.png'
        ... )
        """
        FairnessVisualizer._check_dependencies()

        # Collect all metrics
        data = []
        for attr in protected_attrs:
            if attr not in metrics_results:
                continue

            attr_metrics = metrics_results[attr]
            for metric_name, metric_result in attr_metrics.items():
                # Extract key value based on metric type
                if 'ratio' in metric_result:
                    value = metric_result['ratio']
                elif 'disparity' in metric_result:
                    value = metric_result.get('combined_disparity', metric_result['disparity'])
                elif 'value' in metric_result:
                    value = abs(metric_result['value'])  # Use absolute for differences
                else:
                    value = 0.0

                interpretation = metric_result.get('interpretation', '')
                color = FairnessVisualizer._get_color_for_interpretation(interpretation)

                data.append({
                    'attribute': attr,
                    'metric': FairnessVisualizer.METRIC_LABELS.get(metric_name, metric_name),
                    'value': value,
                    'color': color,
                    'interpretation': interpretation
                })

        df_metrics = pd.DataFrame(data)

        if df_metrics.empty:
            warnings.warn("No metrics data to plot")
            return None

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        # Group by attribute
        n_attrs = len(protected_attrs)
        n_metrics = len(df_metrics) // n_attrs if n_attrs > 0 else 0

        y_pos = np.arange(len(df_metrics))
        bars = ax.barh(y_pos, df_metrics['value'], color=df_metrics['color'], alpha=0.8, edgecolor='black')

        # Add labels
        labels = [f"{row['attribute']}: {row['metric']}" for _, row in df_metrics.iterrows()]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel('Metric Value (closer to 0 or 1 is better)', fontsize=12, fontweight='bold')
        ax.set_title(title or 'Fairness Metrics Comparison', fontsize=16, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # Add reference lines
        ax.axvline(x=0.8, color='orange', linestyle='--', linewidth=1.5, alpha=0.5, label='EEOC 80% Rule')
        ax.axvline(x=1.0, color='green', linestyle='--', linewidth=1.5, alpha=0.5, label='Perfect Parity')

        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, df_metrics['value'])):
            ax.text(value, bar.get_y() + bar.get_height()/2,
                   f' {value:.3f}', va='center', fontsize=9, fontweight='bold')

        # Legend for colors
        legend_elements = [
            mpatches.Patch(color=FairnessVisualizer.COLORS['green'], label='✓ Verde (OK)'),
            mpatches.Patch(color=FairnessVisualizer.COLORS['yellow'], label='⚠ Amarelo (Atenção)'),
            mpatches.Patch(color=FairnessVisualizer.COLORS['red'], label='✗ Vermelho (Crítico)')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

        return FairnessVisualizer._save_or_show(fig, output_path)

    @staticmethod
    def plot_threshold_impact(
        threshold_results: Dict[str, Any],
        metrics: List[str] = ['disparate_impact_ratio', 'f1_score'],
        output_path: Optional[str] = None,
        title: Optional[str] = None,
        figsize: tuple = (14, 6)
    ) -> Optional[str]:
        """
        Plot impact of classification threshold on fairness and performance.

        Shows how fairness metrics and F1 score vary across different thresholds,
        highlighting the optimal threshold.

        Parameters:
        -----------
        threshold_results : Dict
            Results from run_threshold_analysis()
        metrics : List[str], default=['disparate_impact_ratio', 'f1_score']
            Metrics to plot
        output_path : str, optional
            Path to save figure
        title : str, optional
            Custom title
        figsize : tuple, default=(14, 6)
            Figure size

        Returns:
        --------
        str or None : Path to saved file

        Example:
        --------
        >>> FairnessVisualizer.plot_threshold_impact(
        ...     results['threshold_analysis'],
        ...     output_path='threshold.png'
        ... )
        """
        FairnessVisualizer._check_dependencies()

        df = pd.DataFrame(threshold_results['threshold_curve'])
        optimal_threshold = threshold_results['optimal_threshold']

        fig, ax = plt.subplots(figsize=figsize)

        # Plot each metric
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        for i, metric in enumerate(metrics):
            if metric in df.columns:
                ax.plot(df['threshold'], df[metric], label=metric.replace('_', ' ').title(),
                       linewidth=2.5, color=colors[i % len(colors)])

        # Add EEOC 80% rule line
        ax.axhline(y=0.8, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='EEOC 80% Rule')

        # Add optimal threshold line
        ax.axvline(x=optimal_threshold, color='green', linestyle='--', linewidth=2.5,
                  alpha=0.8, label=f'Optimal Threshold ({optimal_threshold:.3f})')

        # Add default threshold line
        ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=2, alpha=0.6, label='Default (0.5)')

        ax.set_xlabel('Classification Threshold', fontsize=13, fontweight='bold')
        ax.set_ylabel('Metric Value', fontsize=13, fontweight='bold')
        ax.set_title(title or 'Threshold Impact on Fairness and Performance',
                    fontsize=16, fontweight='bold')
        ax.legend(loc='best', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)

        # Add annotations
        optimal_di = df[df['threshold'] == optimal_threshold]['disparate_impact_ratio'].values[0]
        ax.annotate(f'DI: {optimal_di:.3f}',
                   xy=(optimal_threshold, optimal_di),
                   xytext=(optimal_threshold + 0.1, optimal_di + 0.05),
                   arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                   fontsize=10, fontweight='bold')

        return FairnessVisualizer._save_or_show(fig, output_path)

    @staticmethod
    def plot_confusion_matrices(
        cm_by_group: Dict[str, Dict[str, int]],
        attribute_name: str,
        output_path: Optional[str] = None,
        title: Optional[str] = None,
        figsize: tuple = (12, 5)
    ) -> Optional[str]:
        """
        Plot confusion matrices side-by-side for each group.

        Visualizes TP, FP, TN, FN for each protected group to identify
        disparities in error types.

        Parameters:
        -----------
        cm_by_group : Dict
            Confusion matrix per group from results['confusion_matrix'][attr]
        attribute_name : str
            Name of protected attribute
        output_path : str, optional
            Path to save figure
        title : str, optional
            Custom title
        figsize : tuple, default=(12, 5)
            Figure size

        Returns:
        --------
        str or None : Path to saved file

        Example:
        --------
        >>> FairnessVisualizer.plot_confusion_matrices(
        ...     results['confusion_matrix']['gender'],
        ...     'gender',
        ...     output_path='confusion_matrix.png'
        ... )
        """
        FairnessVisualizer._check_dependencies()

        n_groups = len(cm_by_group)
        fig, axes = plt.subplots(1, n_groups, figsize=figsize)

        if n_groups == 1:
            axes = [axes]

        for idx, (group, cm) in enumerate(cm_by_group.items()):
            ax = axes[idx]

            # Create confusion matrix array
            cm_array = np.array([[cm['TN'], cm['FP']],
                                [cm['FN'], cm['TP']]])

            # Plot heatmap
            if SEABORN_AVAILABLE:
                sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues',
                           cbar=True, ax=ax, square=True, linewidths=2, linecolor='black',
                           annot_kws={'size': 14, 'weight': 'bold'})
            else:
                im = ax.imshow(cm_array, cmap='Blues', aspect='auto')
                ax.figure.colorbar(im, ax=ax)
                for i in range(2):
                    for j in range(2):
                        text = ax.text(j, i, cm_array[i, j],
                                     ha="center", va="center", color="black",
                                     fontsize=14, fontweight='bold')

            ax.set_xlabel('Predicted', fontsize=11, fontweight='bold')
            ax.set_ylabel('Actual', fontsize=11, fontweight='bold')
            ax.set_title(f'{group}\n(n={cm["total"]})', fontsize=12, fontweight='bold')
            ax.set_xticklabels(['Negative (0)', 'Positive (1)'])
            ax.set_yticklabels(['Negative (0)', 'Positive (1)'])

        fig.suptitle(title or f'Confusion Matrices by {attribute_name}',
                    fontsize=16, fontweight='bold', y=1.05)

        return FairnessVisualizer._save_or_show(fig, output_path)

    @staticmethod
    def plot_fairness_radar(
        metrics_summary: Dict[str, Dict],
        output_path: Optional[str] = None,
        title: Optional[str] = None,
        figsize: tuple = (10, 10),
        selected_metrics: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Plot radar chart showing fairness across multiple dimensions.

        Creates a spider/radar chart with all fairness metrics normalized to 0-1 scale.

        Parameters:
        -----------
        metrics_summary : Dict
            Dictionary with attribute names as keys and metric dicts as values
            Expected structure: {'gender': {'metric1': {'value': 0.5, ...}, ...}, ...}
        output_path : str, optional
            Path to save figure
        title : str, optional
            Custom title
        figsize : tuple, default=(10, 10)
            Figure size
        selected_metrics : list, optional
            List of metric names to include. If None, uses key fairness metrics

        Returns:
        --------
        str or None : Path to saved file

        Example:
        --------
        >>> metrics = {
        ...     'gender': {
        ...         'statistical_parity': {'value': 0.85},
        ...         'disparate_impact': {'value': 0.92}
        ...     }
        ... }
        >>> FairnessVisualizer.plot_fairness_radar(
        ...     metrics, output_path='radar.png'
        ... )
        """
        FairnessVisualizer._check_dependencies()

        # Default key fairness metrics to display
        if selected_metrics is None:
            selected_metrics = [
                'statistical_parity',
                'disparate_impact',
                'equal_opportunity',
                'equalized_odds',
                'precision_difference'
            ]

        # Extract metric values for each attribute
        # Structure: {metric_name: {attr_name: value}}
        metric_data = {}
        attributes = list(metrics_summary.keys())

        for metric_name in selected_metrics:
            metric_data[metric_name] = {}
            for attr_name, attr_metrics in metrics_summary.items():
                if metric_name in attr_metrics:
                    # Extract the numeric value from the metric dict
                    metric_value = attr_metrics[metric_name]
                    if isinstance(metric_value, dict) and 'value' in metric_value:
                        value = abs(metric_value['value'])  # Use absolute value for radar
                        # Normalize to 0-1 scale (1 = perfect fairness)
                        # For most metrics, closer to 0 is better, but for disparate_impact, closer to 1 is better
                        if metric_name == 'disparate_impact':
                            normalized_value = min(value, 1.0)  # Cap at 1.0
                        else:
                            # For difference metrics, invert (1 - value) so higher is better
                            normalized_value = max(0, 1 - value)
                    else:
                        normalized_value = 0.5  # Default if can't extract

                    metric_data[metric_name][attr_name] = normalized_value

        # Prepare data for plotting
        categories = [m.replace('_', ' ').title() for m in selected_metrics]
        N = len(categories)

        # Compute angles for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle

        # Initialize figure
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='polar')

        # Plot each attribute as a separate line
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        for i, attr_name in enumerate(attributes):
            # Get values for this attribute across all metrics
            values = []
            for metric_name in selected_metrics:
                values.append(metric_data[metric_name].get(attr_name, 0.5))

            values += values[:1]  # Complete the circle

            # Plot
            color = colors[i % len(colors)]
            ax.plot(angles, values, 'o-', linewidth=2.5, color=color,
                   label=attr_name.title(), markersize=6)
            ax.fill(angles, values, alpha=0.15, color=color)

        # Fix axis to go in the right order
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)

        # Set y-axis limits
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)

        # Add grid
        ax.grid(True, linewidth=1, alpha=0.3)

        # Add reference circle at 0.8 (good fairness threshold)
        ax.plot(angles, [0.8] * len(angles), '--', linewidth=1.5,
               color='gray', alpha=0.5, label='Good Threshold')

        ax.set_title(title or 'Fairness Radar Chart\n(1.0 = Perfect Fairness)',
                    fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)

        return FairnessVisualizer._save_or_show(fig, output_path)

    @staticmethod
    def plot_group_comparison(
        metrics_results: Dict[str, Dict[str, Any]],
        attribute_name: str,
        metrics_to_plot: Optional[List[str]] = None,
        output_path: Optional[str] = None,
        title: Optional[str] = None,
        figsize: tuple = (14, 8)
    ) -> Optional[str]:
        """
        Plot detailed comparison of metrics between groups for a single protected attribute.

        Creates grouped bar chart showing metric values (absolute values) for comparison.

        Parameters:
        -----------
        metrics_results : Dict
            Metrics results for all attributes from results['posttrain_metrics']
        attribute_name : str
            Name of protected attribute to plot
        metrics_to_plot : List[str], optional
            Specific metrics to plot. If None, uses key fairness metrics.
        output_path : str, optional
            Path to save figure
        title : str, optional
            Custom title
        figsize : tuple, default=(14, 8)
            Figure size

        Returns:
        --------
        str or None : Path to saved file

        Example:
        --------
        >>> FairnessVisualizer.plot_group_comparison(
        ...     results['posttrain_metrics'],
        ...     'gender',
        ...     output_path='group_comparison.png'
        ... )
        """
        FairnessVisualizer._check_dependencies()

        # Get metrics for the specified attribute
        if attribute_name not in metrics_results:
            warnings.warn(f"Attribute '{attribute_name}' not found in metrics_results")
            return None

        attr_metrics = metrics_results[attribute_name]

        # Default metrics to plot
        if metrics_to_plot is None:
            metrics_to_plot = [
                'statistical_parity',
                'equal_opportunity',
                'disparate_impact',
                'precision_difference',
                'accuracy_difference'
            ]

        # Extract metric values
        data = []
        for metric_name in metrics_to_plot:
            if metric_name in attr_metrics:
                metric_result = attr_metrics[metric_name]
                if isinstance(metric_result, dict) and 'value' in metric_result:
                    value = abs(metric_result['value'])  # Use absolute value
                    interpretation = metric_result.get('interpretation', '')

                    # Determine color based on interpretation
                    if '✗' in interpretation or 'CRÍTICO' in interpretation or 'Vermelho' in interpretation:
                        color = FairnessVisualizer.COLORS['red']
                        status = 'Critical'
                    elif '⚠' in interpretation or 'MODERADO' in interpretation or 'Amarelo' in interpretation:
                        color = FairnessVisualizer.COLORS['yellow']
                        status = 'Warning'
                    else:
                        color = FairnessVisualizer.COLORS['green']
                        status = 'OK'

                    data.append({
                        'metric': FairnessVisualizer.METRIC_LABELS.get(metric_name, metric_name.replace('_', ' ').title()),
                        'value': value,
                        'color': color,
                        'status': status
                    })

        if not data:
            warnings.warn(f"No metrics data available for '{attribute_name}'")
            return None

        df = pd.DataFrame(data)

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        # Create horizontal bar chart with color coding
        y_pos = np.arange(len(df))
        colors = df['color'].values

        bars = ax.barh(y_pos, df['value'], color=colors, alpha=0.7, edgecolor='black', linewidth=1)

        # Customize plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df['metric'], fontsize=11)
        ax.set_xlabel('Metric Value (Absolute)', fontsize=13, fontweight='bold')
        ax.set_title(title or f'Fairness Metrics for {attribute_name.title()}',
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlim(0, max(1.0, df['value'].max() * 1.1))
        ax.grid(axis='x', alpha=0.3, linestyle='--')

        # Add value labels on bars
        for i, (bar, row) in enumerate(zip(bars, df.itertuples())):
            width = bar.get_width()
            label = f'{width:.3f}'
            ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                   label, ha='left', va='center', fontsize=10, fontweight='bold')

        # Add reference line for fairness thresholds
        if any(m in metrics_to_plot for m in ['disparate_impact']):
            ax.axvline(x=0.8, color='orange', linestyle='--', linewidth=2,
                      alpha=0.7, label='EEOC 80% Rule')
        ax.axvline(x=0.1, color='gray', linestyle=':', linewidth=1.5,
                  alpha=0.5, label='Threshold 0.1')

        # Create legend for status colors
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=FairnessVisualizer.COLORS['green'], label='OK', alpha=0.7),
            Patch(facecolor=FairnessVisualizer.COLORS['yellow'], label='Warning', alpha=0.7),
            Patch(facecolor=FairnessVisualizer.COLORS['red'], label='Critical', alpha=0.7)
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=11, framealpha=0.9)

        plt.tight_layout()

        return FairnessVisualizer._save_or_show(fig, output_path)
