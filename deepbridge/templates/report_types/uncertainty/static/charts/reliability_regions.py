"""
Module for generating reliability regions charts showing confidence levels across feature values.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional
from .base_chart import BaseChartGenerator

# Configure logger
logger = logging.getLogger("deepbridge.reports")

class ReliabilityRegionsChart(BaseChartGenerator):
    """
    Generate charts showing reliability/confidence regions for specific features.
    This visualizes where the model is more or less confident based on feature values.
    """

    def generate(self,
                reliability_data: Dict[str, Dict[str, Any]],
                title: str = "Feature Reliability Regions",
                show_top_n: int = 3) -> str:
        """
        Generate reliability regions chart for top features.

        Parameters:
        -----------
        reliability_data : Dict[str, Dict[str, Any]]
            Dictionary with feature names as keys and reliability analysis results:
            {
                "feature_name": {
                    "bins": list of bin definitions,
                    "confidence_scores": list of confidence scores per bin,
                    "sample_counts": list of sample counts per bin,
                    "low_confidence_regions": list of low confidence regions,
                    "high_confidence_regions": list of high confidence regions,
                    "avg_confidence": average confidence,
                    "feature_name": name of the feature
                },
                ...
            }
        title : str
            Chart title
        show_top_n : int
            Number of top features to show

        Returns:
        --------
        str : Base64 encoded image or empty string if data invalid
        """
        self._validate_chart_generator()

        if not reliability_data:
            logger.warning("No reliability data provided for chart")
            return ""

        try:
            # Select top features to display
            features_to_plot = list(reliability_data.keys())[:show_top_n]
            n_features = len(features_to_plot)

            if n_features == 0:
                logger.warning("No features to plot in reliability regions")
                return ""

            # Create subplots for each feature
            fig, axes = self.plt.subplots(n_features, 1, figsize=(12, 4 * n_features))
            if n_features == 1:
                axes = [axes]  # Make it iterable

            # Color scheme for confidence levels
            colors = {
                'low': '#ff4444',     # Red for low confidence
                'medium': '#ffaa00',  # Orange for medium confidence
                'high': '#44ff44'     # Green for high confidence
            }

            # Plot each feature
            for idx, feature_name in enumerate(features_to_plot):
                ax = axes[idx]
                data = reliability_data[feature_name]

                if 'bins' not in data or 'confidence_scores' not in data:
                    logger.warning(f"Missing required data for feature {feature_name}")
                    continue

                bins = data['bins']
                confidence_scores = data['confidence_scores']
                sample_counts = data.get('sample_counts', [1] * len(bins))

                # Prepare data for plotting
                bin_centers = [b['center'] for b in bins]
                bin_widths = [(b['end'] - b['start']) for b in bins]

                # Create bar colors based on confidence levels
                bar_colors = []
                for score in confidence_scores:
                    if score < 0.4:
                        bar_colors.append(colors['low'])
                    elif score < 0.7:
                        bar_colors.append(colors['medium'])
                    else:
                        bar_colors.append(colors['high'])

                # Create main bar plot
                bars = ax.bar(bin_centers, confidence_scores, width=bin_widths,
                             color=bar_colors, alpha=0.7, edgecolor='black', linewidth=1)

                # Add sample count as secondary information
                ax2 = ax.twinx()
                ax2.plot(bin_centers, sample_counts, 'k--', alpha=0.5, label='Sample Count')
                ax2.set_ylabel('Sample Count', fontsize=10, color='gray')
                ax2.tick_params(axis='y', labelcolor='gray')

                # Highlight low and high confidence regions
                if 'low_confidence_regions' in data:
                    for region in data['low_confidence_regions']:
                        ax.axvspan(region['range'][0], region['range'][1],
                                  alpha=0.2, color='red', zorder=0)

                if 'high_confidence_regions' in data:
                    for region in data['high_confidence_regions']:
                        ax.axvspan(region['range'][0], region['range'][1],
                                  alpha=0.2, color='green', zorder=0)

                # Add confidence thresholds as horizontal lines
                ax.axhline(y=0.4, color='red', linestyle='--', alpha=0.5, label='Low Conf. Threshold')
                ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='High Conf. Threshold')

                # Customize the plot
                ax.set_xlabel(f'{feature_name} Value', fontsize=11)
                ax.set_ylabel('Confidence Score', fontsize=11)
                ax.set_title(f'Reliability Regions for {feature_name}', fontsize=12, fontweight='bold')
                ax.set_ylim([0, 1])
                ax.grid(True, alpha=0.3, axis='y')

                # Add legend
                handles1, labels1 = ax.get_legend_handles_labels()
                handles2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(handles1 + handles2, labels1 + labels2, loc='upper right', fontsize=9)

                # Add summary statistics as text
                if 'avg_confidence' in data:
                    stats_text = (f"Avg Confidence: {data['avg_confidence']:.3f}\n"
                                 f"Min: {data.get('min_confidence', 0):.3f}\n"
                                 f"Max: {data.get('max_confidence', 0):.3f}")
                    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                           fontsize=9, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # Add main title
            fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
            fig.tight_layout()

            # Save to base64
            return self._save_figure_to_base64(fig)

        except Exception as e:
            logger.error(f"Error generating reliability regions chart: {str(e)}")
            return ""

    def generate_single_feature(self,
                               feature_data: Dict[str, Any],
                               feature_name: str,
                               title: Optional[str] = None) -> str:
        """
        Generate reliability regions chart for a single feature.

        Parameters:
        -----------
        feature_data : Dict[str, Any]
            Reliability analysis results for a single feature
        feature_name : str
            Name of the feature
        title : str, optional
            Chart title (defaults to "Reliability Regions for {feature_name}")

        Returns:
        --------
        str : Base64 encoded image
        """
        if title is None:
            title = f"Reliability Regions for {feature_name}"

        # Wrap single feature data in expected format
        reliability_data = {feature_name: feature_data}

        return self.generate(reliability_data, title=title, show_top_n=1)

    def generate_comparison(self,
                           reliability_data: Dict[str, Dict[str, Any]],
                           title: str = "Feature Confidence Comparison") -> str:
        """
        Generate a comparison chart showing confidence levels across multiple features.

        Parameters:
        -----------
        reliability_data : Dict[str, Dict[str, Any]]
            Reliability analysis results for multiple features
        title : str
            Chart title

        Returns:
        --------
        str : Base64 encoded image
        """
        self._validate_chart_generator()

        if not reliability_data:
            return ""

        try:
            fig, ax = self.plt.subplots(figsize=(10, 6))

            features = []
            avg_confidences = []
            min_confidences = []
            max_confidences = []

            # Extract summary statistics for each feature
            for feature_name, data in reliability_data.items():
                if 'avg_confidence' in data:
                    features.append(feature_name)
                    avg_confidences.append(data['avg_confidence'])
                    min_confidences.append(data.get('min_confidence', 0))
                    max_confidences.append(data.get('max_confidence', 1))

            if not features:
                logger.warning("No features with confidence data for comparison")
                return ""

            # Create grouped bar chart
            x = np.arange(len(features))
            width = 0.25

            bars1 = ax.bar(x - width, min_confidences, width, label='Min Confidence',
                          color='#ff4444', alpha=0.7)
            bars2 = ax.bar(x, avg_confidences, width, label='Avg Confidence',
                          color='#4444ff', alpha=0.7)
            bars3 = ax.bar(x + width, max_confidences, width, label='Max Confidence',
                          color='#44ff44', alpha=0.7)

            # Add value labels on bars
            for bars in [bars1, bars2, bars3]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}', ha='center', va='bottom', fontsize=8)

            # Customize the plot
            ax.set_xlabel('Feature', fontsize=12)
            ax.set_ylabel('Confidence Score', fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(features, rotation=45, ha='right')
            ax.set_ylim([0, 1])
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3, axis='y')

            # Add reference lines
            ax.axhline(y=0.4, color='red', linestyle='--', alpha=0.3)
            ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.3)

            fig.tight_layout()
            return self._save_figure_to_base64(fig)

        except Exception as e:
            logger.error(f"Error generating confidence comparison chart: {str(e)}")
            return ""