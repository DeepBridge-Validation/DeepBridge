"""
Boxplot chart for uncertainty interval widths comparison.

This module generates boxplot visualizations comparing interval widths
across different model configurations and confidence levels.
"""

import base64
import io
import logging
import numpy as np
from typing import Dict, Any, List, Optional

logger = logging.getLogger("deepbridge.reports")

class IntervalBoxplotChart:
    """Generate boxplot comparison of interval widths across models."""

    @staticmethod
    def generate(data: Dict[str, Any], **kwargs) -> Optional[str]:
        """
        Generate boxplot chart comparing interval widths.

        Parameters:
        -----------
        data : Dict[str, Any]
            Uncertainty test results containing widths data

        Returns:
        --------
        str : Base64-encoded image or None if generation fails
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import pandas as pd

            # Set style
            sns.set_style("whitegrid")

            # Extract widths data from results
            if 'results' not in data:
                logger.error("No results found in data")
                return None

            results = data['results']

            # Prepare data for plotting
            all_widths = []
            model_names = []
            coverage_values = []
            mean_widths = []

            # Handle both single and multiple model results
            if isinstance(results, list):
                # Multiple configurations (different alpha values)
                for i, result in enumerate(results):
                    if 'widths' in result:
                        widths = result['widths']
                        alpha = result.get('alpha', 0.1)
                        coverage = result.get('coverage', 0)
                        mean_width = result.get('mean_width', np.mean(widths))

                        model_names.append(f"α={alpha}")
                        all_widths.append(widths)
                        coverage_values.append(coverage)
                        mean_widths.append(mean_width)
                    else:
                        logger.warning(f"No widths in result {i}")
            else:
                # Single configuration
                if 'widths' in results:
                    widths = results['widths']
                    alpha = results.get('alpha', 0.1)
                    coverage = results.get('coverage', 0)
                    mean_width = results.get('mean_width', np.mean(widths))

                    all_widths.append(widths)
                    model_names.append(f"α={alpha}")
                    coverage_values.append(coverage)
                    mean_widths.append(mean_width)

            if not all_widths:
                logger.error("No width data found")
                return None

            # Create figure with subplots
            fig, axes = plt.subplots(2, 1, figsize=(14, 10))

            # ===== Top subplot: Traditional Boxplot =====
            ax1 = axes[0]
            positions = range(len(model_names))

            # Create boxplot
            bp = ax1.boxplot(all_widths,
                            positions=positions,
                            labels=model_names,
                            patch_artist=True,
                            showmeans=True,
                            meanprops=dict(marker='o', markerfacecolor='red',
                                         markersize=8, markeredgecolor='darkred'),
                            medianprops=dict(color='black', linewidth=2),
                            boxprops=dict(linewidth=1.5),
                            whiskerprops=dict(linewidth=1.5),
                            capprops=dict(linewidth=1.5))

            # Color the boxes with gradient
            colors = sns.color_palette("Set2", len(model_names))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            # Add title and labels
            ax1.set_title('Confidence Interval Width Distribution',
                         fontsize=14, fontweight='bold')
            ax1.set_xlabel('Configuration (Level α)', fontsize=12)
            ax1.set_ylabel('Interval Width', fontsize=12)
            ax1.grid(True, alpha=0.3, linestyle='--')

            # Add statistics annotations
            for i, (pos, name) in enumerate(zip(positions, model_names)):
                if i < len(coverage_values) and i < len(mean_widths):
                    stats_text = f'Cov: {coverage_values[i]:.2f}\nμ: {mean_widths[i]:.3f}'
                    ax1.text(pos, ax1.get_ylim()[1] * 0.95, stats_text,
                            ha='center', va='top', fontsize=8,
                            bbox=dict(boxstyle='round,pad=0.3',
                                    facecolor='white', alpha=0.7))

            # ===== Bottom subplot: Combined Violin + Box + Strip =====
            ax2 = axes[1]

            # Create DataFrame for seaborn
            df_list = []
            for name, widths in zip(model_names, all_widths):
                df_temp = pd.DataFrame({
                    'Model': [name] * len(widths),
                    'Width': widths
                })
                df_list.append(df_temp)

            if df_list:
                df_combined = pd.concat(df_list, ignore_index=True)
            else:
                logger.error("Could not create combined dataframe")
                return None

            # Violin plot (background)
            sns.violinplot(data=df_combined, x='Model', y='Width', ax=ax2,
                          palette='Set2', inner=None, alpha=0.4, cut=0)

            # Box plot overlay
            sns.boxplot(data=df_combined, x='Model', y='Width', ax=ax2,
                       palette='Set2', width=0.3, showcaps=True,
                       boxprops=dict(alpha=0.8),
                       showmeans=True,
                       meanprops=dict(marker='o', markerfacecolor='red',
                                    markersize=6, markeredgecolor='darkred'),
                       medianprops=dict(color='black', linewidth=2),
                       flierprops=dict(marker='o', markerfacecolor='gray',
                                      markersize=3, alpha=0.5))

            # Sample points for strip plot (limit to avoid overcrowding)
            n_samples_per_model = min(100, len(df_combined) // len(model_names))
            if n_samples_per_model > 10:
                df_sample = df_combined.groupby('Model').sample(
                    n=n_samples_per_model,
                    replace=False,
                    random_state=42
                )
                sns.stripplot(data=df_sample, x='Model', y='Width', ax=ax2,
                             color='navy', size=2, alpha=0.3, jitter=True)

            # Customize appearance
            ax2.set_title('Detailed Distribution (Violin + Box + Points)',
                         fontsize=14, fontweight='bold')
            ax2.set_xlabel('Configuration (Level α)', fontsize=12)
            ax2.set_ylabel('Interval Width', fontsize=12)
            ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

            # Add comprehensive statistics table
            stats_data = []
            for name, widths in zip(model_names, all_widths):
                stats_data.append([
                    name,
                    f"{np.mean(widths):.3f}",
                    f"{np.median(widths):.3f}",
                    f"{np.std(widths):.3f}",
                    f"{np.min(widths):.3f}",
                    f"{np.max(widths):.3f}"
                ])

            # Create statistics table
            table_ax = fig.add_axes([0.85, 0.55, 0.12, 0.35])
            table_ax.axis('off')

            table_data = [['Config', 'Mean', 'Median', 'SD', 'Min', 'Max']] + stats_data
            table = table_ax.table(cellText=table_data,
                                  cellLoc='center',
                                  loc='center',
                                  colWidths=[0.15, 0.15, 0.15, 0.15, 0.15, 0.15])

            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1.2, 1.5)

            # Color header row
            for i in range(6):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')

            # Color data rows alternately
            for i in range(1, len(stats_data) + 1):
                color = '#f0f0f0' if i % 2 == 0 else 'white'
                for j in range(6):
                    table[(i, j)].set_facecolor(color)

            plt.tight_layout()

            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            plt.close('all')  # Clean up

            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.read()).decode('utf-8')

            return f"data:image/png;base64,{img_base64}"

        except Exception as e:
            logger.error(f"Error generating interval boxplot: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    @staticmethod
    def get_chart_info():
        """
        Get information about this chart type.

        Returns:
        --------
        dict : Chart metadata
        """
        return {
            'name': 'Interval Width Boxplot',
            'description': 'Boxplot comparison of prediction interval widths across different confidence levels',
            'type': 'boxplot',
            'requires': ['widths', 'alpha', 'coverage']
        }