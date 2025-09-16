"""
PSI (Population Stability Index) analysis chart for uncertainty.

This module generates bar charts showing PSI values for features,
indicating distribution shifts between reliable and unreliable predictions.
"""

import base64
import io
import logging
import numpy as np
from typing import Dict, Any, Optional, List

logger = logging.getLogger("deepbridge.reports")

class PSIAnalysisChart:
    """Generate PSI analysis bar chart for feature distribution shifts."""

    @staticmethod
    def generate(data: Dict[str, Any], **kwargs) -> Optional[str]:
        """
        Generate PSI analysis chart showing distribution shifts.

        Parameters:
        -----------
        data : Dict[str, Any]
            Uncertainty test results with reliability analysis

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

            # Extract PSI data from results
            psi_values = PSIAnalysisChart._extract_psi_values(data)

            if not psi_values:
                logger.warning(f"No PSI values found in data. Data keys: {list(data.keys())}")
                if 'reliability_analysis' in data:
                    logger.warning(f"reliability_analysis keys: {list(data.get('reliability_analysis', {}).keys())}")
                return None

            # Create DataFrame for plotting
            psi_df = pd.DataFrame({
                'Feature': list(psi_values.keys()),
                'PSI': list(psi_values.values())
            })

            # Sort by PSI value (descending)
            psi_df = psi_df.sort_values('PSI', ascending=False)

            # Take top features if too many
            max_features = kwargs.get('max_features', 20)
            if len(psi_df) > max_features:
                psi_df = psi_df.head(max_features)

            # Create figure
            fig, ax = plt.subplots(figsize=(12, 7))

            # Create bar plot
            bars = ax.bar(range(len(psi_df)), psi_df['PSI'].values)

            # Color bars based on PSI thresholds
            colors = []
            labels_added = {'insignificant': False, 'small': False, 'significant': False}

            for i, (bar, psi) in enumerate(zip(bars, psi_df['PSI'].values)):
                if psi < 0.1:
                    bar.set_color('#2ecc71')  # Green - insignificant change
                    bar.set_alpha(0.8)
                    if not labels_added['insignificant']:
                        bar.set_label('Insignificant Change (<0.1)')
                        labels_added['insignificant'] = True
                    colors.append('green')
                elif psi < 0.25:
                    bar.set_color('#f39c12')  # Orange - small change
                    bar.set_alpha(0.8)
                    if not labels_added['small']:
                        bar.set_label('Small Change (0.1-0.25)')
                        labels_added['small'] = True
                    colors.append('orange')
                else:
                    bar.set_color('#e74c3c')  # Red - significant change
                    bar.set_alpha(0.8)
                    if not labels_added['significant']:
                        bar.set_label('Significant Change (≥0.25)')
                        labels_added['significant'] = True
                    colors.append('red')

            # Customize plot
            ax.set_xticks(range(len(psi_df)))
            ax.set_xticklabels(psi_df['Feature'].values, rotation=45, ha='right', fontsize=10)
            ax.set_xlabel('Features', fontsize=12, fontweight='bold')
            ax.set_ylabel('PSI (Population Stability Index)', fontsize=12, fontweight='bold')
            ax.set_title('Distribution Shift Analysis: Unreliable vs Reliable Data',
                        fontsize=14, fontweight='bold', pad=20)

            # Add threshold lines
            ax.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, linewidth=1.5)
            ax.axhline(y=0.25, color='orange', linestyle='--', alpha=0.5, linewidth=1.5)

            # Add threshold annotations
            y_max = ax.get_ylim()[1]
            ax.text(len(psi_df) - 0.5, 0.1, 'Threshold: 0.1',
                   color='green', fontsize=9, ha='right', va='bottom')
            ax.text(len(psi_df) - 0.5, 0.25, 'Threshold: 0.25',
                   color='orange', fontsize=9, ha='right', va='bottom')

            # Add value labels on bars
            for bar, value in zip(bars, psi_df['PSI'].values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}',
                       ha='center', va='bottom' if height >= 0 else 'top',
                       fontsize=8, fontweight='bold')

            # Add legend
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, loc='upper right', framealpha=0.9,
                         edgecolor='gray', title='PSI Classification')

            # Grid customization
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            ax.set_axisbelow(True)

            # Add interpretation text box
            interpretation = PSIAnalysisChart._interpret_psi(psi_df)
            props = dict(boxstyle='round,pad=0.5', facecolor='lightgray',
                        alpha=0.9, edgecolor='gray', linewidth=1)
            ax.text(0.02, 0.98, interpretation, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=props, fontweight='normal')

            # Add statistics summary
            stats_text = PSIAnalysisChart._generate_stats_summary(psi_df)
            props_stats = dict(boxstyle='round,pad=0.5', facecolor='white',
                             alpha=0.9, edgecolor='gray', linewidth=1)
            ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='bottom',
                   horizontalalignment='right', bbox=props_stats)

            # Adjust layout
            plt.tight_layout()

            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            plt.close('all')

            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.read()).decode('utf-8')

            return f"data:image/png;base64,{img_base64}"

        except Exception as e:
            logger.error(f"Error generating PSI analysis chart: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    @staticmethod
    def _extract_psi_values(data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract PSI values from the uncertainty results data.

        Parameters:
        -----------
        data : Dict[str, Any]
            The uncertainty test results

        Returns:
        --------
        Dict[str, float] : Dictionary of feature names to PSI values
        """
        psi_values = {}

        # Handle different data structures
        if 'results' in data:
            results = data['results']

            # Handle list of results (multiple configurations)
            if isinstance(results, list):
                # Look for the first result with reliability analysis
                for result in results:
                    if 'reliability_analysis' in result:
                        reliability = result['reliability_analysis']
                        if 'psi_values' in reliability:
                            psi_values = reliability['psi_values']
                            break
            # Handle single result
            elif isinstance(results, dict):
                if 'reliability_analysis' in results:
                    reliability = results['reliability_analysis']
                    if 'psi_values' in reliability:
                        psi_values = reliability['psi_values']

        # Also check directly in data (even if results exist)
        if not psi_values and 'reliability_analysis' in data:
            logger.info("Found reliability_analysis in data (top level)")
            reliability = data['reliability_analysis']
            if 'psi_values' in reliability:
                logger.info(f"Found psi_values in reliability_analysis with {len(reliability['psi_values'])} items")
                psi_values = reliability['psi_values']

        # Alternative: check for psi_values directly
        if not psi_values and 'psi_values' in data:
            psi_values = data['psi_values']

        # Ensure all values are floats
        if psi_values:
            psi_values = {str(k): float(v) for k, v in psi_values.items()}

        logger.info(f"_extract_psi_values returning {len(psi_values)} values")
        return psi_values

    @staticmethod
    def _interpret_psi(psi_df) -> str:
        """
        Generate interpretation text for PSI values.

        Parameters:
        -----------
        psi_df : pd.DataFrame
            DataFrame with Feature and PSI columns

        Returns:
        --------
        str : Interpretation text
        """
        n_total = len(psi_df)
        n_significant = len(psi_df[psi_df['PSI'] >= 0.25])
        n_small = len(psi_df[(psi_df['PSI'] >= 0.1) & (psi_df['PSI'] < 0.25)])
        n_insignificant = len(psi_df[psi_df['PSI'] < 0.1])

        # Build interpretation text
        text_lines = ["PSI Analysis Summary:"]

        # Overall summary
        text_lines.append(f"Total features: {n_total}")

        # Breakdown by category
        if n_significant > 0:
            text_lines.append(f"• {n_significant} significant changes")
        if n_small > 0:
            text_lines.append(f"• {n_small} small changes")
        if n_insignificant > 0:
            text_lines.append(f"• {n_insignificant} insignificant changes")

        # Top feature if significant
        if n_significant > 0:
            top_feature = psi_df.iloc[0]['Feature']
            top_psi = psi_df.iloc[0]['PSI']
            text_lines.append(f"\nLargest shift: {top_feature} (PSI={top_psi:.3f})")

        return '\n'.join(text_lines)

    @staticmethod
    def _generate_stats_summary(psi_df) -> str:
        """
        Generate statistical summary of PSI values.

        Parameters:
        -----------
        psi_df : pd.DataFrame
            DataFrame with PSI values

        Returns:
        --------
        str : Statistics summary text
        """
        psi_values = psi_df['PSI'].values

        stats_lines = ["PSI Statistics:"]
        stats_lines.append(f"Mean: {np.mean(psi_values):.3f}")
        stats_lines.append(f"Median: {np.median(psi_values):.3f}")
        stats_lines.append(f"Max: {np.max(psi_values):.3f}")
        stats_lines.append(f"Min: {np.min(psi_values):.3f}")

        return '\n'.join(stats_lines)

    @staticmethod
    def get_chart_info() -> Dict[str, Any]:
        """
        Get information about this chart type.

        Returns:
        --------
        dict : Chart metadata
        """
        return {
            'name': 'PSI Analysis Chart',
            'description': 'Bar chart showing Population Stability Index for feature distribution shifts',
            'type': 'bar',
            'requires': ['reliability_analysis', 'psi_values']
        }