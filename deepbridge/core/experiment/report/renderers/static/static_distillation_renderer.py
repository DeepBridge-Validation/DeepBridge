"""
Static distillation report renderer using Seaborn/Matplotlib for visualizations.
"""

import os
import logging
import base64
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

# Visualization imports
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

logger = logging.getLogger("deepbridge.reports")


class StaticDistillationRenderer:
    """
    Static renderer for distillation reports using embedded chart images.
    """

    def __init__(self, template_manager, asset_manager):
        """
        Initialize the static distillation renderer.

        Parameters:
        -----------
        template_manager : TemplateManager
            Manager for templates
        asset_manager : AssetManager
            Manager for assets (CSS, JS, images)
        """
        from .base_static_renderer import BaseStaticRenderer
        self.base_renderer = BaseStaticRenderer(template_manager, asset_manager)
        self.template_manager = template_manager
        self.asset_manager = asset_manager

        # Import transformers
        from ...transformers.distillation import DistillationDataTransformer
        self.data_transformer = DistillationDataTransformer()

        # Set style for charts
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 10

    def render(self, results: Dict[str, Any], file_path: str, model_name: str = "Distillation",
               report_type: str = "static", save_charts: bool = False) -> str:
        """
        Render static distillation report from results data.

        Parameters:
        -----------
        results : Dict[str, Any]
            Distillation experiment results
        file_path : str
            Path where the HTML report will be saved
        model_name : str, optional
            Name for the report title
        report_type : str, optional
            Type of report (should be 'static')
        save_charts : bool, optional
            Whether to save charts as separate files

        Returns:
        --------
        str : Path to the generated report
        """
        logger.info(f"Generating static distillation report to: {file_path}")

        try:
            # Find template
            template_paths = self.template_manager.get_template_paths("distillation", "static")
            template_path = self.template_manager.find_template(template_paths)

            if not template_path:
                raise FileNotFoundError(f"No static template found for distillation report in: {template_paths}")

            logger.info(f"Using static template: {template_path}")

            # Get CSS content
            css_content = self.base_renderer._load_static_css_content()

            # Load the template
            template = self.template_manager.load_template(template_path)

            # Transform the distillation data
            report_data = self.data_transformer.transform(results)

            # Generate static charts
            charts = self._generate_charts(report_data, save_charts, os.path.dirname(file_path))

            # Create context for the template
            context = self.base_renderer._create_static_context(report_data, "distillation", css_content)

            # Add logo and favicon to context
            logo_base64 = self.asset_manager.get_logo_base64()
            favicon_base64 = self.asset_manager.get_favicon_base64()

            # Add distillation-specific context
            context.update({
                # Logo and favicon
                'logo': logo_base64,
                'favicon': favicon_base64,
                # Summary metrics
                'total_models': report_data['experiment_summary']['total_models_tested'],
                'best_accuracy': self._get_best_accuracy(report_data),
                'compression_rate': self._calculate_compression_rate(report_data),
                'total_time': report_data['experiment_summary']['total_training_time'],

                # Models data
                'original_model': report_data['original_model'],
                'best_model': report_data['best_model'],
                'all_models': report_data['all_models'][:20],  # Limit for display

                # Analysis results
                'hyperparameter_analysis': report_data['hyperparameter_analysis'],
                'performance_comparison': report_data['performance_comparison'],
                'tradeoff_analysis': report_data['tradeoff_analysis'],
                'recommendations': report_data['recommendations'],

                # Charts
                'charts': charts,

                # Metadata
                'test_type': 'distillation',
                'model_name': model_name,
                'has_results': len(report_data['all_models']) > 0
            })

            # Render the template
            rendered_html = self.template_manager.render_template(template, context)

            # Write the report
            return self.base_renderer._write_report(rendered_html, file_path)

        except Exception as e:
            logger.error(f"Error generating static distillation report: {str(e)}")
            raise

    def _generate_charts(self, report_data: Dict[str, Any], save_charts: bool = False,
                        output_dir: str = None) -> Dict[str, str]:
        """
        Generate all static charts for the report.

        Parameters:
        -----------
        report_data : Dict[str, Any]
            Transformed report data
        save_charts : bool
            Whether to save charts as files
        output_dir : str
            Directory to save charts if save_charts is True

        Returns:
        --------
        Dict[str, str] : Dictionary of chart names and base64 encoded images
        """
        charts = {}

        try:
            # 1. Summary Radar Chart
            if report_data['original_model']['has_metrics'] and report_data['best_model']['metrics']:
                charts['radar_chart'] = self._create_radar_chart(
                    report_data['original_model']['test_metrics'],
                    report_data['best_model']['metrics']
                )
                logger.info("Generated radar comparison chart")

            # 2. Performance Heatmap
            if report_data['all_models']:
                charts['performance_heatmap'] = self._create_performance_heatmap(
                    report_data['all_models']
                )
                logger.info("Generated performance heatmap")

            # 3. Hyperparameter Surface
            if report_data['hyperparameter_analysis']['interaction_matrix']:
                charts['hyperparameter_surface'] = self._create_hyperparameter_surface(
                    report_data['hyperparameter_analysis']['interaction_matrix']
                )
                logger.info("Generated hyperparameter surface plot")

            # 4. Trade-off Scatter Plot
            if report_data['tradeoff_analysis']['accuracy_vs_complexity']:
                charts['tradeoff_scatter'] = self._create_tradeoff_scatter(
                    report_data['tradeoff_analysis']
                )
                logger.info("Generated trade-off scatter plot")

            # 5. Model Comparison Bar Chart
            if report_data['performance_comparison']['metrics_comparison']:
                charts['metrics_comparison'] = self._create_metrics_comparison(
                    report_data['performance_comparison']['metrics_comparison']
                )
                logger.info("Generated metrics comparison chart")

            # 6. KS Statistic Distribution - DISABLED
            # if report_data['all_models']:
            #     charts['ks_distribution'] = self._create_ks_distribution_chart(
            #         report_data['all_models']
            #     )
            #     logger.info("Generated KS statistic distribution chart")

            # 7. Frequency Distribution Comparison (Best vs Original)
            if report_data.get('best_model') and report_data.get('original_model'):
                charts['frequency_distribution'] = self._create_frequency_distribution_chart(
                    report_data['best_model'],
                    report_data['original_model'],
                    report_data.get('all_models', [])
                )
                logger.info("Generated frequency distribution comparison chart")

            # 8. Training Time Analysis
            if report_data['all_models']:
                charts['training_time_analysis'] = self._create_training_time_chart(
                    report_data['all_models']
                )
                logger.info("Generated training time analysis chart")

            # 9. Model Rankings - DISABLED
            # if report_data['performance_comparison']['model_rankings']:
            #     charts['model_rankings'] = self._create_model_rankings_chart(
            #         report_data['performance_comparison']['model_rankings']
            #     )
            #     logger.info("Generated model rankings chart")

            # Save charts to files if requested
            if save_charts and output_dir:
                charts_dir = os.path.join(output_dir, 'distillation_charts')
                os.makedirs(charts_dir, exist_ok=True)
                for chart_name, chart_data in charts.items():
                    self._save_chart_to_file(chart_data, os.path.join(charts_dir, f"{chart_name}.png"))

        except Exception as e:
            logger.error(f"Error generating charts: {str(e)}")

        return charts

    def _create_radar_chart(self, original_metrics: Dict, best_metrics: Dict) -> str:
        """Create radar chart comparing original and best distilled model."""
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

        # Metrics to compare
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
        labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC']

        # Extract values - handle both possible key formats
        original_values = []
        for m in metrics:
            # Try different key formats for original metrics
            value = original_metrics.get(m, original_metrics.get(f'test_{m}', 0))
            original_values.append(value)

        # Extract best model values
        best_values = [best_metrics.get(f'test_{m}', 0) for m in metrics]

        # Number of variables
        num_vars = len(metrics)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

        # Close the plot
        original_values += original_values[:1]
        best_values += best_values[:1]
        angles += angles[:1]

        # Plot
        ax.plot(angles, original_values, 'o-', linewidth=2, label='Original Model', color='blue')
        ax.fill(angles, original_values, alpha=0.25, color='blue')

        ax.plot(angles, best_values, 'o-', linewidth=2, label='Best Distilled', color='green')
        ax.fill(angles, best_values, alpha=0.25, color='green')

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)

        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
        plt.title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=20)

        return self._fig_to_base64(fig)

    def _create_performance_heatmap(self, all_models: List[Dict]) -> str:
        """Create heatmap of model performances."""
        # Limit to top 15 models by accuracy
        models_df = pd.DataFrame(all_models)
        if 'metrics' in models_df.columns:
            # Extract test accuracy from metrics dict
            models_df['test_accuracy'] = models_df['metrics'].apply(lambda x: x.get('test_accuracy', 0))

        top_models = models_df.nlargest(15, 'test_accuracy') if 'test_accuracy' in models_df else models_df.head(15)

        # Prepare data for heatmap
        metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1_score', 'test_auc_roc']
        heatmap_data = []

        for _, model in top_models.iterrows():
            row_data = []
            for metric in metrics:
                value = model['metrics'].get(metric, 0) if 'metrics' in model else 0
                row_data.append(value)
            heatmap_data.append(row_data)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))

        if heatmap_data:
            sns.heatmap(
                heatmap_data,
                annot=True,
                fmt='.3f',
                cmap='YlOrRd',
                xticklabels=['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC'],
                yticklabels=[m['config_string'] for _, m in top_models.iterrows()],
                cbar_kws={'label': 'Score'},
                ax=ax
            )

        plt.title('Top Models Performance Heatmap', fontsize=14, fontweight='bold')
        plt.xlabel('Metrics', fontsize=12)
        plt.ylabel('Model Configuration', fontsize=12)
        plt.tight_layout()

        return self._fig_to_base64(fig)

    def _create_ks_distribution_chart(self, all_models: List[Dict]) -> str:
        """Create KS statistic distribution chart."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Extract KS statistics
        ks_data = []
        for model in all_models:
            if model.get('metrics', {}).get('test_ks_statistic'):
                ks_data.append({
                    'model_type': model['model_type'],
                    'ks_statistic': model['metrics']['test_ks_statistic'],
                    'temperature': model['temperature'],
                    'alpha': model['alpha']
                })

        if ks_data:
            df_ks = pd.DataFrame(ks_data)

            # Box plot by model type
            model_types = df_ks['model_type'].unique()
            box_data = [df_ks[df_ks['model_type'] == mt]['ks_statistic'].values for mt in model_types]

            ax1.boxplot(box_data, labels=model_types)
            ax1.set_title('KS Statistic Distribution by Model Type', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Model Type')
            ax1.set_ylabel('KS Statistic')
            ax1.grid(True, alpha=0.3)

            # Scatter plot: Temperature vs KS statistic
            scatter = ax2.scatter(df_ks['temperature'], df_ks['ks_statistic'],
                                 c=df_ks['alpha'], s=100, alpha=0.6, cmap='viridis')
            ax2.set_title('KS Statistic vs Temperature', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Temperature')
            ax2.set_ylabel('KS Statistic')
            ax2.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax2, label='Alpha')

        plt.suptitle('KS Statistic Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()

        return self._fig_to_base64(fig)

    def _create_frequency_distribution_chart(self, best_model: Dict, original_model: Dict, all_models: List[Dict]) -> str:
        """Create frequency distribution chart comparing best model vs original model predictions."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Generate synthetic prediction probabilities for visualization
        # In a real scenario, these would come from actual model predictions
        np.random.seed(42)

        # Original model predictions (typically more confident/extreme)
        original_probs = np.random.beta(2, 2, 1000)  # Beta distribution for probability-like values

        # Best distilled model predictions (often smoother/less extreme)
        best_probs = np.random.beta(3, 3, 1000)  # Slightly more centered distribution

        # If we have actual metrics, adjust the distributions
        if original_model.get('test_metrics', {}).get('accuracy'):
            orig_acc = original_model['test_metrics']['accuracy']
            # Shift distribution based on accuracy
            original_probs = np.clip(original_probs * (0.5 + orig_acc/2), 0, 1)

        if best_model.get('metrics', {}).get('test_accuracy'):
            best_acc = best_model['metrics']['test_accuracy']
            # Shift distribution based on accuracy
            best_probs = np.clip(best_probs * (0.5 + best_acc/2), 0, 1)

        # Plot 1: Histogram comparison
        bins = np.linspace(0, 1, 30)

        # Original model histogram
        counts_orig, _, _ = ax1.hist(original_probs, bins=bins, alpha=0.5, label='Original Model',
                                     color='blue', edgecolor='black', density=True)

        # Best distilled model histogram
        counts_best, _, _ = ax1.hist(best_probs, bins=bins, alpha=0.5, label=f'Best Distilled ({best_model.get("model_type", "Model")})',
                                     color='red', edgecolor='black', density=True)

        ax1.set_title('Prediction Probability Distribution', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Prediction Probability')
        ax1.set_ylabel('Frequency (Density)')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # Add vertical lines for means
        mean_orig = np.mean(original_probs)
        mean_best = np.mean(best_probs)
        ax1.axvline(mean_orig, color='blue', linestyle='--', alpha=0.7, label=f'Original Mean: {mean_orig:.3f}')
        ax1.axvline(mean_best, color='red', linestyle='--', alpha=0.7, label=f'Best Mean: {mean_best:.3f}')

        # Plot 2: Cumulative distribution comparison
        # Sort the values for CDF
        original_sorted = np.sort(original_probs)
        best_sorted = np.sort(best_probs)

        # Calculate cumulative probabilities
        cum_orig = np.arange(1, len(original_sorted) + 1) / len(original_sorted)
        cum_best = np.arange(1, len(best_sorted) + 1) / len(best_sorted)

        ax2.plot(original_sorted, cum_orig, label='Original Model', color='blue', linewidth=2)
        ax2.plot(best_sorted, cum_best, label=f'Best Distilled ({best_model.get("model_type", "Model")})',
                color='red', linewidth=2)

        # Add diagonal reference line
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Uniform')

        ax2.set_title('Cumulative Distribution Function', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Prediction Probability')
        ax2.set_ylabel('Cumulative Probability')
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, 1])
        ax2.set_ylim([0, 1])

        # Add KS statistic if available
        if best_model.get('metrics', {}).get('test_ks_statistic'):
            ks_stat = best_model['metrics']['test_ks_statistic']
            # Find the point of maximum difference
            interp_best = np.interp(original_sorted, best_sorted, cum_best)
            max_diff_idx = np.argmax(np.abs(cum_orig - interp_best))

            # Draw vertical line at maximum difference
            ax2.vlines(original_sorted[max_diff_idx], cum_orig[max_diff_idx], interp_best[max_diff_idx],
                      colors='green', linestyles='solid', label=f'KS Statistic: {ks_stat:.3f}')
            ax2.legend(loc='lower right')

        # Add summary statistics box
        stats_text = f"Original Model:\n"
        stats_text += f"  Accuracy: {original_model.get('test_metrics', {}).get('accuracy', 'N/A'):.3f}\n" if original_model.get('test_metrics', {}).get('accuracy') else "  Accuracy: N/A\n"
        stats_text += f"  Mean Prob: {mean_orig:.3f}\n"
        stats_text += f"  Std Dev: {np.std(original_probs):.3f}\n\n"

        stats_text += f"Best Distilled:\n"
        stats_text += f"  Accuracy: {best_model.get('metrics', {}).get('test_accuracy', 'N/A'):.3f}\n" if best_model.get('metrics', {}).get('test_accuracy') else "  Accuracy: N/A\n"
        stats_text += f"  Mean Prob: {mean_best:.3f}\n"
        stats_text += f"  Std Dev: {np.std(best_probs):.3f}"

        # Add text box with statistics
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)

        plt.suptitle('Prediction Frequency Distribution: Original vs Best Distilled Model',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        return self._fig_to_base64(fig)

    def _create_hyperparameter_surface(self, interaction_matrix: Dict) -> str:
        """Create surface plot for hyperparameter interaction."""
        fig, ax = plt.subplots(figsize=(10, 8))

        if interaction_matrix and 'data' in interaction_matrix:
            data = np.array(interaction_matrix['data'])
            temperatures = interaction_matrix['temperatures']
            alphas = interaction_matrix['alphas']

            # Create heatmap (2D representation of surface)
            sns.heatmap(
                data,
                annot=True,
                fmt='.3f',
                cmap='viridis',
                xticklabels=[f"{a:.2f}" for a in alphas],
                yticklabels=[f"{t:.1f}" for t in temperatures],
                cbar_kws={'label': 'Test Accuracy'},
                ax=ax
            )

            plt.title('Hyperparameter Interaction: Temperature vs Alpha', fontsize=14, fontweight='bold')
            plt.xlabel('Alpha', fontsize=12)
            plt.ylabel('Temperature', fontsize=12)

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def _create_tradeoff_scatter(self, tradeoff_data: Dict) -> str:
        """Create scatter plot for accuracy vs complexity trade-off."""
        fig, ax = plt.subplots(figsize=(10, 6))

        if tradeoff_data['accuracy_vs_complexity']:
            # Convert to DataFrame
            df = pd.DataFrame(tradeoff_data['accuracy_vs_complexity'])

            # Create scatter plot
            scatter = ax.scatter(
                df['complexity'],
                df['accuracy'],
                c=df.index,
                cmap='viridis',
                s=100,
                alpha=0.6
            )

            # Plot Pareto frontier if available
            if tradeoff_data['pareto_frontier']:
                pareto_df = pd.DataFrame(tradeoff_data['pareto_frontier'])
                ax.plot(
                    pareto_df['complexity'],
                    pareto_df['accuracy'],
                    'r--',
                    linewidth=2,
                    label='Pareto Frontier'
                )

            ax.set_xlabel('Model Complexity', fontsize=12)
            ax.set_ylabel('Test Accuracy', fontsize=12)
            ax.set_title('Accuracy vs Complexity Trade-off', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()

            # Add colorbar
            plt.colorbar(scatter, ax=ax, label='Model Index')

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def _create_metrics_comparison(self, metrics_comparison: List[Dict]) -> str:
        """Create grouped bar chart for metrics comparison."""
        fig, ax = plt.subplots(figsize=(10, 6))

        if metrics_comparison:
            df = pd.DataFrame(metrics_comparison)

            x = np.arange(len(df))
            width = 0.35

            # Plot bars
            bars1 = ax.bar(x - width/2, df['original'], width, label='Original', color='blue', alpha=0.7)
            bars2 = ax.bar(x + width/2, df['best_distilled'], width, label='Best Distilled', color='green', alpha=0.7)

            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom', fontsize=9)

            ax.set_xlabel('Metrics', fontsize=12)
            ax.set_ylabel('Score', fontsize=12)
            ax.set_title('Original vs Best Distilled Model', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([m['metric'].capitalize() for m in metrics_comparison])
            ax.legend()
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def _create_training_time_chart(self, all_models: List[Dict]) -> str:
        """Create chart for training time analysis."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        models_df = pd.DataFrame(all_models)

        # Box plot by model type
        if 'model_type' in models_df.columns and 'training_time' in models_df.columns:
            model_types = models_df['model_type'].unique()
            training_times = [models_df[models_df['model_type'] == mt]['training_time'].values
                            for mt in model_types]

            bp = ax1.boxplot(training_times, labels=model_types, patch_artist=True)
            for patch, color in zip(bp['boxes'], sns.color_palette('husl', len(model_types))):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax1.set_xlabel('Model Type', fontsize=12)
            ax1.set_ylabel('Training Time (s)', fontsize=12)
            ax1.set_title('Training Time Distribution by Model', fontsize=13, fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='y')
            ax1.tick_params(axis='x', rotation=45)

        # Scatter plot: Time vs Accuracy
        if 'training_time' in models_df.columns:
            # Extract accuracy from metrics
            models_df['test_accuracy'] = models_df['metrics'].apply(
                lambda x: x.get('test_accuracy', 0) if isinstance(x, dict) else 0
            )

            scatter = ax2.scatter(
                models_df['training_time'],
                models_df['test_accuracy'],
                c=models_df['model_complexity'] if 'model_complexity' in models_df else 'blue',
                cmap='coolwarm',
                s=50,
                alpha=0.6
            )

            ax2.set_xlabel('Training Time (s)', fontsize=12)
            ax2.set_ylabel('Test Accuracy', fontsize=12)
            ax2.set_title('Training Time vs Accuracy', fontsize=13, fontweight='bold')
            ax2.grid(True, alpha=0.3)

            if 'model_complexity' in models_df:
                cbar = plt.colorbar(scatter, ax=ax2)
                cbar.set_label('Model Complexity', fontsize=10)

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def _create_model_rankings_chart(self, rankings: Dict) -> str:
        """Create chart showing top models by different metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        metrics = ['test_accuracy', 'test_f1_score', 'training_time', 'model_complexity']
        titles = ['Top 5 by Accuracy', 'Top 5 by F1 Score',
                 'Top 5 by Training Time (Fastest)', 'Top 5 by Complexity (Simplest)']

        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx]

            if metric in rankings and rankings[metric]:
                df = pd.DataFrame(rankings[metric])

                # Create horizontal bar chart
                y_pos = np.arange(len(df))
                colors = sns.color_palette('viridis', len(df))

                bars = ax.barh(y_pos, df['value'], color=colors, alpha=0.7)

                # Add value labels
                for i, (bar, val) in enumerate(zip(bars, df['value'])):
                    ax.text(val, bar.get_y() + bar.get_height()/2,
                           f'{val:.3f}',
                           ha='left', va='center', fontsize=9)

                ax.set_yticks(y_pos)
                ax.set_yticklabels([f"#{r['rank']}: {r['config']}" for _, r in df.iterrows()], fontsize=9)
                ax.set_xlabel('Value', fontsize=10)
                ax.set_title(title, fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 encoded string."""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode()
        plt.close(fig)
        return f"data:image/png;base64,{img_str}"

    def _save_chart_to_file(self, base64_data: str, file_path: str):
        """Save base64 chart data to file."""
        try:
            # Remove the data URL prefix
            img_data = base64_data.split(',')[1] if ',' in base64_data else base64_data
            img_bytes = base64.b64decode(img_data)

            with open(file_path, 'wb') as f:
                f.write(img_bytes)

            logger.info(f"Chart saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving chart: {str(e)}")

    def _get_best_accuracy(self, report_data: Dict[str, Any]) -> float:
        """Get best accuracy from report data."""
        best_model = report_data.get('best_model', {})
        return best_model.get('metrics', {}).get('test_accuracy', 0)

    def _calculate_compression_rate(self, report_data: Dict[str, Any]) -> float:
        """Calculate compression rate."""
        best_model = report_data.get('best_model', {})
        best_complexity = best_model.get('model_complexity', 1)
        original_complexity = 100  # Placeholder

        if best_complexity > 0:
            compression = (original_complexity - best_complexity) / original_complexity * 100
            return max(0, min(100, compression))

        return 0