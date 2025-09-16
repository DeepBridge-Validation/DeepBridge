"""
Utility functions for creating Seaborn-based charts for static reports.
"""

import base64
import io
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import math

# Configure logger
logger = logging.getLogger("deepbridge.reports")

class SeabornChartGenerator:
    """
    Helper class for generating Seaborn charts for static reports.
    """
    
    def __init__(self):
        """
        Initialize the chart generator.
        """
        # Try to import required libraries
        try:
            import seaborn as sns
            import matplotlib.pyplot as plt
            import pandas as pd
            import numpy as np
            self.sns = sns
            self.plt = plt
            self.pd = pd
            self.np = np
            self.has_visualization_libs = True
            
            # Set default style
            sns.set_theme(style="whitegrid")
            # Use a color palette that works well for most charts
            sns.set_palette("deep")
            # Improve font scaling for better readability
            plt.rcParams.update({
                'font.size': 12,
                'axes.labelsize': 14,
                'axes.titlesize': 16,
                'xtick.labelsize': 12,
                'ytick.labelsize': 12,
                'legend.fontsize': 12,
                'figure.titlesize': 18
            })
            
        except ImportError as e:
            logger.error(f"Required libraries for visualization not available: {str(e)}")
            self.has_visualization_libs = False
    
    def generate_encoded_chart(self, func, *args, **kwargs) -> str:
        """
        Generate a chart using a function and return base64 encoded image.
        
        Parameters:
        -----------
        func : callable
            Function to call to generate the chart
        *args, **kwargs
            Arguments to pass to the function
            
        Returns:
        --------
        str : Base64 encoded image data
        """
        if not self.has_visualization_libs:
            logger.error("Required libraries for visualization not available")
            return ""
        
        try:
            # Create a figure
            figsize = kwargs.pop('figsize', (10, 6))
            dpi = kwargs.pop('dpi', 100)
            fig, ax = self.plt.subplots(figsize=figsize, dpi=dpi)
            
            # Call the function with the figure's axis and other arguments
            func(ax, *args, **kwargs)
            
            # Save the figure to a bytes buffer
            buf = io.BytesIO()

            # Use bbox_inches='tight' instead of tight_layout to avoid warnings
            # This adjusts the figure size automatically based on content
            fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0.5)
            buf.seek(0)
            
            # Encode the image to base64
            img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            # Close the figure to avoid memory leaks
            self.plt.close(fig)
            
            return f"data:image/png;base64,{img_base64}"
        except Exception as e:
            logger.error(f"Error generating chart: {str(e)}")
            return ""
    
    def robustness_overview_chart(self, perturbation_levels: List[float], scores: List[float],
                                  base_score: float, title: str = "Model Performance by Perturbation Level",
                                  metric_name: str = "Score", feature_subset_scores: List[float] = None) -> str:
        """
        Generate a robustness overview chart showing model performance at different perturbation levels.

        Parameters:
        -----------
        perturbation_levels : List[float]
            List of perturbation intensity levels
        scores : List[float]
            List of model scores at each perturbation level
        base_score : float
            Base score of the model (without perturbation)
        title : str, optional
            Chart title
        metric_name : str, optional
            Name of the metric being displayed
        feature_subset_scores : List[float], optional
            List of scores for feature subset at each perturbation level

        Returns:
        --------
        str : Base64 encoded image data
        """
        def _generate_chart(ax, perturbation_levels, scores, base_score, title, metric_name, feature_subset_scores=None):
            # Create DataFrame for Seaborn
            data = {
                'Perturbation Level': perturbation_levels,
                'All Features': scores
            }

            # Add feature subset scores if available
            has_feature_subset = feature_subset_scores is not None and len(feature_subset_scores) == len(perturbation_levels)
            if has_feature_subset:
                data['Feature Subset'] = feature_subset_scores

            df = self.pd.DataFrame(data)

            # Convert to long format for seaborn
            if has_feature_subset:
                df_melted = self.pd.melt(df, id_vars=['Perturbation Level'],
                                        value_vars=['All Features', 'Feature Subset'],
                                        var_name='Feature Set', value_name=metric_name)
                # Plot the line chart with both lines
                self.sns.lineplot(x='Perturbation Level', y=metric_name, hue='Feature Set',
                                style='Feature Set', markers=True, data=df_melted, ax=ax)
            else:
                # Plot just the single line for all features
                self.sns.lineplot(x='Perturbation Level', y='All Features', data=df,
                                marker='o', linewidth=2, markersize=8, ax=ax,
                                label=f'All Features')

            # Add a horizontal line for the base score
            ax.axhline(y=base_score, color='r', linestyle='--', alpha=0.7,
                       label=f'Base {metric_name}: {base_score:.4f}')

            # Set labels and title
            ax.set_xlabel('Perturbation Intensity')
            ax.set_ylabel(metric_name)
            ax.set_title(title)

            # Add legend
            ax.legend(loc='best')

            # Calculate min and max for y-axis limits
            all_values = [base_score]
            all_values.extend(scores)
            if has_feature_subset:
                all_values.extend(feature_subset_scores)

            y_min = min(all_values) * 0.95 if min(all_values) > 0 else min(all_values) * 1.05
            y_max = max(all_values) * 1.05
            ax.set_ylim(y_min, y_max)

            # Add grid for better readability
            ax.grid(True, linestyle='--', alpha=0.7)

            # Add some padding to x-axis
            ax.set_xlim(min(perturbation_levels) - 0.05, max(perturbation_levels) + 0.05)

        return self.generate_encoded_chart(_generate_chart, perturbation_levels, scores,
                                          base_score, title, metric_name, feature_subset_scores)
                                          
    def worst_performance_chart(self, perturbation_levels: List[float], worst_scores: List[float],
                               base_score: float, title: str = "Worst Performance by Perturbation Level",
                               metric_name: str = "Score", feature_subset_worst_scores: List[float] = None) -> str:
        """
        Generate a chart showing worst model performance at different perturbation levels.

        Parameters:
        -----------
        perturbation_levels : List[float]
            List of perturbation intensity levels
        worst_scores : List[float]
            List of worst model scores at each perturbation level
        base_score : float
            Base score of the model (without perturbation)
        title : str, optional
            Chart title
        metric_name : str, optional
            Name of the metric being displayed
        feature_subset_worst_scores : List[float], optional
            List of worst scores for feature subset at each perturbation level

        Returns:
        --------
        str : Base64 encoded image data
        """
        def _generate_chart(ax, perturbation_levels, worst_scores, base_score, title, metric_name, feature_subset_worst_scores=None):
            # Create DataFrame for Seaborn
            data = {
                'Perturbation Level': perturbation_levels,
                'All Features': worst_scores
            }

            # Add feature subset scores if available
            has_feature_subset = feature_subset_worst_scores is not None and len(feature_subset_worst_scores) == len(perturbation_levels)
            if has_feature_subset:
                data['Feature Subset'] = feature_subset_worst_scores

            df = self.pd.DataFrame(data)

            # Convert to long format for seaborn
            if has_feature_subset:
                df_melted = self.pd.melt(df, id_vars=['Perturbation Level'],
                                        value_vars=['All Features', 'Feature Subset'],
                                        var_name='Feature Set', value_name=metric_name)
                # Plot the line chart with both lines
                self.sns.lineplot(x='Perturbation Level', y=metric_name, hue='Feature Set',
                                style='Feature Set', markers=True, data=df_melted, ax=ax,
                                dashes=False, alpha=0.8)
            else:
                # Plot just the single line for all features with dashed line
                self.sns.lineplot(x='Perturbation Level', y='All Features', data=df,
                                marker='x', linewidth=2, markersize=8, ax=ax, 
                                label=f'Worst Performance', color='#d32f2f')

            # Add a horizontal line for the base score
            ax.axhline(y=base_score, color='r', linestyle='--', alpha=0.7,
                       label=f'Base {metric_name}: {base_score:.4f}')

            # Set labels and title
            ax.set_xlabel('Perturbation Intensity')
            ax.set_ylabel(metric_name)
            ax.set_title(title)

            # Add legend
            ax.legend(loc='best')

            # Calculate min and max for y-axis limits
            all_values = [base_score]
            all_values.extend(worst_scores)
            if has_feature_subset:
                all_values.extend(feature_subset_worst_scores)

            y_min = min(all_values) * 0.95 if min(all_values) > 0 else min(all_values) * 1.05
            y_max = max(all_values) * 1.05
            ax.set_ylim(y_min, y_max)

            # Add grid for better readability
            ax.grid(True, linestyle='--', alpha=0.7)

            # Add some padding to x-axis
            ax.set_xlim(min(perturbation_levels) - 0.05, max(perturbation_levels) + 0.05)

        return self.generate_encoded_chart(_generate_chart, perturbation_levels, worst_scores,
                                          base_score, title, metric_name, feature_subset_worst_scores)
    
    def model_comparison_chart(self, perturbation_levels: List[float], models_data: Dict[str, Dict],
                              title: str = "Model Performance Comparison", 
                              metric_name: str = "Score") -> str:
        """
        Generate a model comparison chart showing multiple models' performance.
        
        Parameters:
        -----------
        perturbation_levels : List[float]
            List of perturbation intensity levels
        models_data : Dict[str, Dict]
            Dictionary of model data with keys as model names and values as dictionaries 
            containing 'scores' and 'base_score'
        title : str, optional
            Chart title
        metric_name : str, optional
            Name of the metric being displayed
            
        Returns:
        --------
        str : Base64 encoded image data
        """
        def _generate_chart(ax, perturbation_levels, models_data, title, metric_name):
            # Prepare data for plotting
            df_list = []
            
            for model_name, model_info in models_data.items():
                scores = model_info.get('scores', [])
                
                # Skip models with no scores
                if not scores or len(scores) != len(perturbation_levels):
                    continue
                
                # Create data for this model
                model_df = self.pd.DataFrame({
                    'Perturbation Level': perturbation_levels,
                    metric_name: scores,
                    'Model': model_name
                })
                df_list.append(model_df)
            
            # Combine all dataframes
            if not df_list:
                logger.error("No valid model data for comparison chart")
                ax.text(0.5, 0.5, "No model data available", ha='center', va='center', transform=ax.transAxes)
                return
                
            df = self.pd.concat(df_list, ignore_index=True)
            
            # Plot the comparison chart
            self.sns.lineplot(x='Perturbation Level', y=metric_name, hue='Model', 
                              style='Model', markers=True, dashes=False,
                              data=df, ax=ax)
            
            # Add horizontal lines for base scores
            for model_name, model_info in models_data.items():
                base_score = model_info.get('base_score', None)
                if base_score is not None:
                    ax.axhline(y=base_score, linestyle='--', alpha=0.5, 
                              label=f'{model_name} Base: {base_score:.4f}')
            
            # Set labels and title
            ax.set_xlabel('Perturbation Intensity')
            ax.set_ylabel(metric_name)
            ax.set_title(title)
            
            # Improve the legend
            ax.legend(loc='best', frameon=True, framealpha=0.9)
            
            # Add grid for better readability
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add some padding to x-axis
            ax.set_xlim(min(perturbation_levels) - 0.05, max(perturbation_levels) + 0.05)
        
        return self.generate_encoded_chart(_generate_chart, perturbation_levels, models_data, 
                                          title, metric_name, figsize=(12, 8))
    
    def feature_importance_chart(self, features: Dict[str, float], title: str = "Feature Importance",
                                max_features: int = 15, color: str = "viridis") -> str:
        """
        Generate a feature importance chart.

        Parameters:
        -----------
        features : Dict[str, float]
            Dictionary of feature names and their importance scores
        title : str, optional
            Chart title
        max_features : int, optional
            Maximum number of features to display
        color : str, optional
            Color palette for the chart

        Returns:
        --------
        str : Base64 encoded image data
        """
        # Check if we have valid data first
        if not features:
            logger.warning("Empty features dictionary provided for feature importance chart")
            return ""

        # Check that we have numeric values and convert to float
        clean_features = {}
        for feature, value in features.items():
            try:
                clean_features[feature] = float(value)
            except (ValueError, TypeError):
                logger.warning(f"Non-numeric importance for feature {feature}: {value}")
                continue

        if not clean_features:
            logger.warning("No valid numeric values in features dictionary")
            return ""

        def _generate_chart(ax, features, title, max_features, color):
            # Convert features dict to DataFrame
            df = self.pd.DataFrame({
                'Feature': list(features.keys()),
                'Importance': list(features.values())
            })
            
            # Sort by importance
            df = df.sort_values('Importance', ascending=False)
            
            # Limit to max_features
            if len(df) > max_features:
                df = df.head(max_features)
                title += f" (Top {max_features})"
            
            # Create horizontal bar chart
            self.sns.barplot(x='Importance', y='Feature', hue='Feature', data=df,
                             palette=color, legend=False, ax=ax)
            
            # Set labels and title
            ax.set_xlabel('Importance Score')
            ax.set_ylabel('Feature')
            ax.set_title(title)
            
            # Add value labels to the bars
            for i, importance in enumerate(df['Importance']):
                ax.text(importance + 0.01, i, f"{importance:.4f}",
                       va='center', fontsize=10)
            
            # Adjust x-axis to ensure all labels are visible
            x_max = df['Importance'].max() * 1.15
            ax.set_xlim(0, x_max)
        
        return self.generate_encoded_chart(_generate_chart, clean_features, title,
                                         max_features, color, figsize=(10, max(6, min(15, len(features) * 0.4))))
    
    def feature_comparison_chart(self, model_importance: Dict[str, float], 
                                robustness_importance: Dict[str, float],
                                title: str = "Feature Importance Comparison",
                                max_features: int = 15) -> str:
        """
        Generate a chart comparing model-defined feature importance with robustness-based importance.
        
        Parameters:
        -----------
        model_importance : Dict[str, float]
            Dictionary of feature names and their model-defined importance scores
        robustness_importance : Dict[str, float]
            Dictionary of feature names and their robustness-based importance scores
        title : str, optional
            Chart title
        max_features : int, optional
            Maximum number of features to display
            
        Returns:
        --------
        str : Base64 encoded image data
        """
        # Check if we have valid data first
        if not model_importance or not robustness_importance:
            logger.warning("Empty feature importance dictionaries provided for comparison chart")
            return ""
        
        # Validate and clean the importance dictionaries
        clean_model_importance = {}
        clean_robustness_importance = {}
        
        for feature, value in model_importance.items():
            try:
                clean_model_importance[feature] = float(value)
            except (ValueError, TypeError):
                logger.warning(f"Non-numeric model importance for feature {feature}: {value}")
                continue
                
        for feature, value in robustness_importance.items():
            try:
                clean_robustness_importance[feature] = float(value)
            except (ValueError, TypeError):
                logger.warning(f"Non-numeric robustness importance for feature {feature}: {value}")
                continue
                
        # Check if we have any common features
        common_features = set(clean_model_importance.keys()) & set(clean_robustness_importance.keys())
        if not common_features:
            logger.warning("No common features found between model and robustness importance")
            # Create merged set from both dictionaries
            all_features = set(clean_model_importance.keys()) | set(clean_robustness_importance.keys())
            if not all_features:
                return ""
                
        def _generate_chart(ax, model_importance, robustness_importance, title, max_features):
            # Get common features
            common_features = set(model_importance.keys()) & set(robustness_importance.keys())
            
            # If no common features, use union
            if not common_features:
                common_features = set(model_importance.keys()) | set(robustness_importance.keys())
            
            # Create DataFrame
            data = []
            for feature in common_features:
                model_imp = model_importance.get(feature, 0)
                robust_imp = robustness_importance.get(feature, 0)
                data.append({
                    'Feature': feature,
                    'Model Importance': model_imp,
                    'Robustness Impact': robust_imp,
                    'Difference': abs(model_imp - robust_imp),
                    'Max Importance': max(model_imp, robust_imp)
                })
            
            df = self.pd.DataFrame(data)
            
            # Sort by max importance
            df = df.sort_values('Max Importance', ascending=False)
            
            # Limit to max_features
            if len(df) > max_features:
                df = df.head(max_features)
                title += f" (Top {max_features})"
            
            # Melt DataFrame for Seaborn
            df_melted = self.pd.melt(df, id_vars=['Feature'], 
                                     value_vars=['Model Importance', 'Robustness Impact'],
                                     var_name='Importance Type', value_name='Importance Score')
            
            # Create grouped bar chart
            self.sns.barplot(x='Feature', y='Importance Score', hue='Importance Type', 
                            data=df_melted, ax=ax)
            
            # Set labels and title
            ax.set_xlabel('Feature')
            ax.set_ylabel('Importance Score')
            ax.set_title(title)
            
            # Rotate x-axis labels for better readability
            labels = ax.get_xticklabels()
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(labels, rotation=45, ha='right')
            
            # Improve the legend
            ax.legend(loc='best', frameon=True)
            
            # Add grid for better readability
            ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        return self.generate_encoded_chart(_generate_chart, clean_model_importance, clean_robustness_importance,
                                         title, max_features, figsize=(12, 8))

    def individual_feature_impact_chart(self, feature_impacts: Dict[str, float],
                                       title: str = "Individual Feature Robustness Impact",
                                       max_features: int = 20) -> str:
        """
        Generate a horizontal bar chart showing the impact of perturbing each feature individually.

        Parameters:
        -----------
        feature_impacts : Dict[str, float]
            Dictionary of feature names and their impact scores (negative = performance drop)
        title : str, optional
            Chart title
        max_features : int, optional
            Maximum number of features to display

        Returns:
        --------
        str : Base64 encoded image data
        """
        if not feature_impacts:
            logger.warning("Empty feature impacts dictionary provided")
            return ""

        # Sort features by absolute impact and take top N
        sorted_features = sorted(feature_impacts.items(),
                               key=lambda x: abs(x[1]),
                               reverse=True)[:max_features]

        if not sorted_features:
            logger.warning("No valid features found for impact chart")
            return ""

        def _generate_chart(ax, sorted_features, title):
            features = [f[0] for f in sorted_features]
            impacts = [f[1] for f in sorted_features]

            # Create color map: red for negative impact, green for positive
            colors = ['#d62728' if impact < 0 else '#2ca02c' for impact in impacts]

            # Create horizontal bar chart
            bars = ax.barh(range(len(features)), impacts, color=colors, alpha=0.7)

            # Customize the chart
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features, fontsize=10)
            ax.set_xlabel('Impact on Performance', fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

            # Add a vertical line at x=0
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

            # Add value labels on the bars
            for i, (bar, impact) in enumerate(zip(bars, impacts)):
                # Position text based on bar direction
                if impact >= 0:
                    ha = 'left'
                    offset = 0.002
                else:
                    ha = 'right'
                    offset = -0.002

                ax.text(impact + offset, bar.get_y() + bar.get_height()/2,
                       f'{impact:.3f}',
                       ha=ha, va='center', fontsize=9)

            # Add grid for better readability
            ax.grid(True, axis='x', linestyle='--', alpha=0.3)
            ax.set_xlim([min(impacts) * 1.1 if min(impacts) < 0 else -0.05,
                        max(impacts) * 1.1 if max(impacts) > 0 else 0.05])

            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#d62728', alpha=0.7, label='Negative Impact'),
                Patch(facecolor='#2ca02c', alpha=0.7, label='Positive Impact')
            ]
            ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

            # Use the figure's tight_layout instead of plt
            ax.figure.tight_layout()

        # Calculate appropriate figure height based on number of features
        fig_height = max(6, min(12, len(sorted_features) * 0.4))

        return self.generate_encoded_chart(_generate_chart, sorted_features, title,
                                          figsize=(10, fig_height))

    def method_comparison_chart(self, perturbation_levels: List[float],
                               raw_scores: List[float], quantile_scores: List[float],
                               base_score: float, metric_name: str = "Score",
                               raw_worst_scores: List[float] = None,
                               quantile_worst_scores: List[float] = None) -> str:
        """
        Generate a chart comparing performance between raw and quantile perturbation methods.

        Parameters:
        -----------
        perturbation_levels : List[float]
            List of perturbation levels used
        raw_scores : List[float]
            Average scores for raw perturbation method
        quantile_scores : List[float]
            Average scores for quantile perturbation method
        base_score : float
            Baseline score without perturbation
        metric_name : str, optional
            Name of the performance metric
        raw_worst_scores : List[float], optional
            Worst case scores for raw method
        quantile_worst_scores : List[float], optional
            Worst case scores for quantile method

        Returns:
        --------
        str : Base64 encoded image data
        """
        if not perturbation_levels or not raw_scores or not quantile_scores:
            logger.warning("Empty data provided for method comparison chart")
            return ""

        if len(perturbation_levels) != len(raw_scores) or len(perturbation_levels) != len(quantile_scores):
            logger.warning("Mismatched data lengths for method comparison chart")
            return ""

        def _generate_chart(ax, perturbation_levels, raw_scores, quantile_scores, base_score,
                           metric_name, raw_worst_scores, quantile_worst_scores):

            # Plot baseline
            ax.axhline(y=base_score, color='black', linestyle='--', linewidth=2,
                      label=f'Baseline {metric_name}', alpha=0.8)

            # Plot average performance lines
            ax.plot(perturbation_levels, raw_scores, 'o-', color='#1f77b4', linewidth=2.5,
                   markersize=8, label='Raw Perturbation (avg)', alpha=0.8)
            ax.plot(perturbation_levels, quantile_scores, 's-', color='#ff7f0e', linewidth=2.5,
                   markersize=8, label='Quantile Perturbation (avg)', alpha=0.8)

            # Plot worst case lines if available
            if raw_worst_scores and len(raw_worst_scores) == len(perturbation_levels):
                ax.plot(perturbation_levels, raw_worst_scores, 'o--', color='#1f77b4',
                       linewidth=1.5, markersize=6, alpha=0.6, label='Raw Perturbation (worst)')

            if quantile_worst_scores and len(quantile_worst_scores) == len(perturbation_levels):
                ax.plot(perturbation_levels, quantile_worst_scores, 's--', color='#ff7f0e',
                       linewidth=1.5, markersize=6, alpha=0.6, label='Quantile Perturbation (worst)')

            # Fill area between methods to highlight differences
            ax.fill_between(perturbation_levels, raw_scores, quantile_scores,
                           alpha=0.2, color='gray', label='Performance Difference')

            # Customize the chart
            ax.set_xlabel('Perturbation Level', fontsize=12)
            ax.set_ylabel(f'{metric_name}', fontsize=12)
            ax.set_title('Perturbation Method Comparison', fontsize=14, fontweight='bold', pad=20)

            # Add grid
            ax.grid(True, linestyle='--', alpha=0.3)

            # Set x-axis to start from 0
            ax.set_xlim(left=0)

            # Calculate which method is better at each level
            differences = [q - r for q, r in zip(quantile_scores, raw_scores)]
            avg_diff = sum(differences) / len(differences)

            # Add annotation about which method performs better
            if abs(avg_diff) > 0.001:  # Only if there's a meaningful difference
                better_method = "Quantile" if avg_diff > 0 else "Raw"
                worse_method = "Raw" if avg_diff > 0 else "Quantile"
                ax.text(0.02, 0.98, f'{better_method} method generally outperforms {worse_method}',
                       transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', horizontalalignment='left',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            # Position legend
            ax.legend(loc='best', fontsize=10, framealpha=0.9)

            # Use the figure's tight_layout
            ax.figure.tight_layout()

        return self.generate_encoded_chart(_generate_chart, perturbation_levels, raw_scores, quantile_scores,
                                          base_score, metric_name, raw_worst_scores, quantile_worst_scores,
                                          figsize=(12, 6))

    def selected_features_comparison_chart(self, perturbation_levels: List[float],
                                          all_features_scores: List[float],
                                          selected_features_scores: List[float],
                                          selected_features: List[str],
                                          base_score: float,
                                          metric_name: str = "Score",
                                          all_features_worst: List[float] = None,
                                          selected_features_worst: List[float] = None) -> str:
        """
        Generate a chart comparing the impact of perturbing all features vs selected features.

        Parameters:
        -----------
        perturbation_levels : List[float]
            List of perturbation levels used
        all_features_scores : List[float]
            Average scores when perturbing all features
        selected_features_scores : List[float]
            Average scores when perturbing only selected features
        selected_features : List[str]
            List of feature names that were selected for perturbation
        base_score : float
            Baseline score without perturbation
        metric_name : str, optional
            Name of the performance metric
        all_features_worst : List[float], optional
            Worst case scores when perturbing all features
        selected_features_worst : List[float], optional
            Worst case scores when perturbing selected features

        Returns:
        --------
        str : Base64 encoded image data
        """
        if not perturbation_levels or not all_features_scores or not selected_features_scores:
            logger.warning("Empty data provided for selected features comparison chart")
            return ""

        if len(perturbation_levels) != len(all_features_scores) or len(perturbation_levels) != len(selected_features_scores):
            logger.warning("Mismatched data lengths for selected features comparison chart")
            return ""

        def _generate_chart(ax, perturbation_levels, all_features_scores, selected_features_scores,
                           selected_features, base_score, metric_name, all_features_worst, selected_features_worst):

            # Plot baseline
            ax.axhline(y=base_score, color='black', linestyle='--', linewidth=2,
                      label=f'Baseline {metric_name}', alpha=0.8)

            # Plot main performance lines
            ax.plot(perturbation_levels, all_features_scores, 'o-', color='#d62728', linewidth=2.5,
                   markersize=8, label=f'All Features (avg)', alpha=0.8)
            ax.plot(perturbation_levels, selected_features_scores, 's-', color='#2ca02c', linewidth=2.5,
                   markersize=8, label=f'Selected Features (avg)', alpha=0.8)

            # Plot worst case lines if available
            if all_features_worst and len(all_features_worst) == len(perturbation_levels):
                ax.plot(perturbation_levels, all_features_worst, 'o--', color='#d62728',
                       linewidth=1.5, markersize=6, alpha=0.6, label='All Features (worst)')

            if selected_features_worst and len(selected_features_worst) == len(perturbation_levels):
                ax.plot(perturbation_levels, selected_features_worst, 's--', color='#2ca02c',
                       linewidth=1.5, markersize=6, alpha=0.6, label='Selected Features (worst)')

            # Fill area between curves to highlight differences
            ax.fill_between(perturbation_levels, all_features_scores, selected_features_scores,
                           alpha=0.2, color='orange', label='Performance Difference')

            # Customize the chart
            ax.set_xlabel('Perturbation Level', fontsize=12)
            ax.set_ylabel(f'{metric_name}', fontsize=12)
            ax.set_title('All Features vs Selected Features Comparison', fontsize=14, fontweight='bold', pad=20)

            # Add grid
            ax.grid(True, linestyle='--', alpha=0.3)

            # Set x-axis to start from 0
            ax.set_xlim(left=0)

            # Calculate performance differences
            differences = [sel - all_feat for sel, all_feat in zip(selected_features_scores, all_features_scores)]
            avg_diff = sum(differences) / len(differences) if differences else 0

            # Create info box with selected features and performance summary
            info_text = []
            info_text.append(f"Selected Features ({len(selected_features)}):")

            # Show up to 6 features in the annotation, with "..." if more
            display_features = selected_features[:6]
            if len(selected_features) > 6:
                display_features.append("...")

            for i, feat in enumerate(display_features):
                if i < 3:  # First 3 on first line
                    info_text.append(f"  {feat}")
                else:  # Rest on second line
                    if i == 3:
                        info_text.append(f"  {feat}")
                    else:
                        info_text[-1] += f", {feat}"

            # Add performance summary
            if abs(avg_diff) > 0.001:
                better_approach = "Selected features" if avg_diff > 0 else "All features"
                info_text.append(f"\n{better_approach} approach")
                info_text.append(f"performs better on average")
            else:
                info_text.append(f"\nSimilar performance between approaches")

            # Position info box
            ax.text(0.02, 0.98, '\n'.join(info_text),
                   transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))

            # Position legend at bottom right
            ax.legend(loc='lower right', fontsize=10, framealpha=0.9)

            # Use the figure's tight_layout
            ax.figure.tight_layout()

        return self.generate_encoded_chart(_generate_chart, perturbation_levels, all_features_scores,
                                          selected_features_scores, selected_features, base_score, metric_name,
                                          all_features_worst, selected_features_worst, figsize=(12, 6))

    def detailed_boxplot_chart(self, perturbation_data: Dict[float, List[float]],
                              base_score: float,
                              metric_name: str = "Score",
                              title: str = "Performance Distribution by Perturbation Level",
                              show_coverage: bool = True,
                              coverage_threshold: float = 0.95) -> str:
        """
        Generate a detailed boxplot with comprehensive annotations and statistics.

        Parameters:
        -----------
        perturbation_data : Dict[float, List[float]]
            Dictionary mapping perturbation levels to lists of scores
        base_score : float
            Baseline score without perturbation
        metric_name : str, optional
            Name of the performance metric
        title : str, optional
            Chart title
        show_coverage : bool, optional
            Whether to show coverage percentages
        coverage_threshold : float, optional
            Threshold for coverage calculation (default 95%)

        Returns:
        --------
        str : Base64 encoded image data
        """
        if not perturbation_data:
            logger.warning("Empty perturbation data provided for detailed boxplot")
            return ""

        # Sort levels and prepare data
        sorted_levels = sorted(perturbation_data.keys())

        # Prepare data for plotting
        plot_data = []
        level_labels = []
        statistics = {}

        for level in sorted_levels:
            scores = perturbation_data[level]
            if not scores:
                continue

            level_str = f"{level:.1f}"
            level_labels.append(level_str)
            plot_data.extend([(level_str, score) for score in scores])

            # Calculate detailed statistics
            scores_array = self.np.array(scores)
            statistics[level_str] = {
                'mean': self.np.mean(scores_array),
                'median': self.np.median(scores_array),
                'std': self.np.std(scores_array),
                'min': self.np.min(scores_array),
                'max': self.np.max(scores_array),
                'q25': self.np.percentile(scores_array, 25),
                'q75': self.np.percentile(scores_array, 75),
                'count': len(scores_array),
                'coverage': self.np.sum(scores_array >= base_score * coverage_threshold) / len(scores_array) * 100
            }

        if not plot_data:
            logger.warning("No valid data for detailed boxplot")
            return ""

        def _generate_chart(ax, plot_data, level_labels, statistics, base_score, metric_name, title,
                           show_coverage, coverage_threshold):

            # Convert to DataFrame for seaborn
            df = self.pd.DataFrame(plot_data, columns=['Perturbation_Level', 'Score'])

            # Create detailed boxplot with gradient colors
            colors = self.plt.cm.RdYlBu_r(self.np.linspace(0.2, 0.8, len(level_labels)))

            # Create boxplot
            box_plot = self.sns.boxplot(
                data=df,
                x='Perturbation_Level',
                y='Score',
                ax=ax,
                palette=colors,
                linewidth=1.5
            )

            # Customize boxplot appearance
            for patch, color in zip(box_plot.artists, colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            # Add baseline reference line
            ax.axhline(y=base_score, color='red', linestyle='--', linewidth=2,
                      alpha=0.8, label=f'Baseline {metric_name}')

            # Add coverage threshold line if different from baseline
            coverage_line = base_score * coverage_threshold
            if abs(coverage_line - base_score) > 0.001:
                ax.axhline(y=coverage_line, color='orange', linestyle=':', linewidth=1.5,
                          alpha=0.6, label=f'{coverage_threshold*100:.0f}% Coverage Threshold')

            # Add detailed annotations for each level
            for i, level_str in enumerate(level_labels):
                stats = statistics[level_str]

                # Position for annotations (alternating above/below)
                y_pos = ax.get_ylim()[1] if i % 2 == 0 else ax.get_ylim()[0]
                y_offset = -0.02 if i % 2 == 0 else 0.08
                va = 'top' if i % 2 == 0 else 'bottom'

                # Create annotation text
                annotation_lines = []
                annotation_lines.append(f"μ={stats['mean']:.3f}")
                annotation_lines.append(f"σ={stats['std']:.3f}")
                annotation_lines.append(f"n={stats['count']}")

                if show_coverage:
                    annotation_lines.append(f"cov={stats['coverage']:.1f}%")

                annotation_text = '\n'.join(annotation_lines)

                # Add annotation box
                ax.annotate(annotation_text,
                           xy=(i, y_pos),
                           xytext=(i, y_pos + y_offset),
                           ha='center', va=va,
                           fontsize=8,
                           bbox=dict(boxstyle='round,pad=0.3',
                                   facecolor='white',
                                   edgecolor='gray',
                                   alpha=0.8),
                           arrowprops=dict(arrowstyle='->',
                                         color='gray',
                                         alpha=0.5))

            # Customize chart appearance
            ax.set_xlabel('Perturbation Level', fontsize=12, fontweight='bold')
            ax.set_ylabel(f'{metric_name}', fontsize=12, fontweight='bold')
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

            # Add grid
            ax.grid(True, axis='y', linestyle='--', alpha=0.3)

            # Improve x-axis labels
            ax.set_xticklabels([f"Level {label}" for label in level_labels], fontsize=10)

            # Add legend
            ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

            # Add summary statistics box
            total_scores = len(plot_data)
            overall_mean = self.np.mean([score for _, score in plot_data])
            overall_std = self.np.std([score for _, score in plot_data])

            summary_text = f"Overall Statistics:\n"
            summary_text += f"Total Samples: {total_scores}\n"
            summary_text += f"Mean {metric_name}: {overall_mean:.3f}\n"
            summary_text += f"Std {metric_name}: {overall_std:.3f}\n"
            summary_text += f"Baseline: {base_score:.3f}"

            ax.text(0.02, 0.02, summary_text,
                   transform=ax.transAxes, fontsize=9,
                   verticalalignment='bottom', horizontalalignment='left',
                   bbox=dict(boxstyle='round,pad=0.5',
                           facecolor='lightcyan',
                           alpha=0.8))

            # Use the figure's tight_layout
            ax.figure.tight_layout()

        return self.generate_encoded_chart(_generate_chart, plot_data, level_labels, statistics,
                                          base_score, metric_name, title, show_coverage, coverage_threshold,
                                          figsize=(14, 8))

    def boxplot_chart(self, models_data: List[Dict], title: str = "Performance Distribution",
                     metric_name: str = "Score") -> str:
        """
        Generate an enhanced distribution chart showing the model scores with violin plots, boxplots, and individual points.

        Parameters:
        -----------
        models_data : List[Dict]
            List of model data dictionaries with 'name', 'scores', and 'baseScore'
        title : str, optional
            Chart title
        metric_name : str, optional
            Name of the metric being displayed

        Returns:
        --------
        str : Base64 encoded image data
        """
        def _generate_chart(ax, models_data, title, metric_name):
            # Create DataFrame
            df_list = []
            model_names = []

            for model in models_data:
                name = model.get('name', 'Unknown')
                scores = model.get('scores', [])

                if not scores:
                    continue

                model_names.append(name)

                # Create data for this model
                model_df = self.pd.DataFrame({
                    'Model': [name] * len(scores),
                    metric_name: scores
                })
                df_list.append(model_df)

            # Combine all dataframes
            if not df_list:
                logger.error("No valid model data for distribution chart")
                ax.text(0.5, 0.5, "No model scores available", ha='center', va='center', transform=ax.transAxes)
                return

            df = self.pd.concat(df_list, ignore_index=True)

            # Create a color palette for the plots
            palette = self.sns.color_palette("Set2", n_colors=len(model_names))

            # 1. Create violin plot as the base layer (semi-transparent)
            self.sns.violinplot(
                x='Model',
                y=metric_name,
                data=df,
                ax=ax,
                palette=palette,
                inner=None,  # No inner points/box
                alpha=0.4,   # Semi-transparent
                saturation=0.7
            )

            # 2. Add boxplot on top of violin plots
            self.sns.boxplot(
                x='Model',
                y=metric_name,
                data=df,
                ax=ax,
                width=0.3,  # Narrower width to fit inside violin
                palette=palette,
                saturation=0.9,
                showfliers=False  # Hide fliers since we'll show all points with stripplot
            )

            # 3. Add stripplot for individual points
            self.sns.stripplot(
                x='Model',
                y=metric_name,
                data=df,
                ax=ax,
                size=2.5,
                alpha=0.3,
                jitter=True,
                color='#444444',
                edgecolor='none'
            )

            # Add markers for base scores
            for i, model in enumerate(models_data):
                base_score = model.get('baseScore')
                if base_score is not None:
                    ax.scatter(i, base_score, marker='D', s=100, color='red', edgecolor='black', linewidth=0.5,
                              label='Base Score' if i == 0 else "", zorder=10)

            # Set labels and title
            ax.set_xlabel('Model', fontsize=12)
            ax.set_ylabel(metric_name, fontsize=12)
            ax.set_title(title, fontsize=14)

            # Adjust x-axis labels
            if len(models_data) > 3:
                labels = ax.get_xticklabels()
                ax.set_xticks(ax.get_xticks())
                ax.set_xticklabels(labels, rotation=45, ha='right')

            # Add legend for base score
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles=[handles[0]], labels=['Base Score'], loc='best', framealpha=0.9)

            # Add grid for better readability (behind the plots)
            ax.grid(True, axis='y', linestyle='--', alpha=0.5, zorder=-1)

            # Add statistical annotations if possible
            try:
                # Calculate and add some statistical info for each model
                for i, model in enumerate(models_data):
                    scores = model.get('scores', [])
                    if scores:
                        # Calculate statistics
                        mean = self.np.mean(scores)
                        std = self.np.std(scores)
                        q1 = self.np.percentile(scores, 25)
                        q3 = self.np.percentile(scores, 75)

                        # Get the y-axis limits
                        y_min, y_max = ax.get_ylim()
                        range_height = y_max - y_min

                        # Add mean annotation with a line
                        ax.annotate(f'μ={mean:.3f}',
                                  xy=(i, mean),
                                  xytext=(i+0.25, mean),
                                  arrowprops=dict(arrowstyle='-', color='black', alpha=0.7),
                                  color='black', fontsize=9, fontweight='bold', alpha=0.9)
            except Exception as e:
                logger.warning(f"Could not add statistical annotations: {e}")
                pass

        # Adjust figure size based on number of models - taller for more models
        height = max(8, min(12, 6 + len([m for m in models_data if m.get('scores')]) * 0.8))
        return self.generate_encoded_chart(_generate_chart, models_data, title,
                                         metric_name, figsize=(12, height))

    def uncertainty_violin_chart(self, models_data: List[Dict], title: str = "Uncertainty Model Comparison",
                     metric_name: str = "Interval Width") -> str:
        """DEBUG: This method was improved to ensure all models are displayed in the chart"""
        """
        Generate an enhanced distribution chart showing all models on the x-axis with violin plots,
        boxplots and individual points. The primary model is used as a reference.

        Parameters:
        -----------
        models_data : List[Dict]
            List of model data dictionaries with 'name', 'scores', and 'baseScore'
        title : str, optional
            Chart title
        metric_name : str, optional
            Name of the metric being displayed

        Returns:
        --------
        str : Base64 encoded image data
        """
        def _generate_chart(ax, models_data, title, metric_name):
            # Create DataFrame
            df_list = []
            model_names = []
            base_model = None

            # Find base model (primary model)
            if models_data and len(models_data) > 0:
                base_model = models_data[0]  # Assume first model is the primary model

            # Log how many models we have for debugging
            logger.info(f"Processing {len(models_data)} models for uncertainty_violin_chart")

            for model in models_data:
                name = model.get('name', 'Unknown')
                scores = model.get('scores', [])

                # Log model details for debugging
                logger.info(f"Model: {name}, Has scores: {bool(scores)}, Scores count: {len(scores)}")

                if not scores:
                    logger.warning(f"Skipping model {name} due to empty scores")
                    continue

                model_names.append(name)

                # Create data for this model
                model_df = self.pd.DataFrame({
                    'Model': [name] * len(scores),
                    metric_name: scores
                })
                df_list.append(model_df)

            # Combine all dataframes
            if not df_list:
                logger.error("No valid model data for distribution chart")
                ax.text(0.5, 0.5, "No model scores available", ha='center', va='center', transform=ax.transAxes)
                return

            df = self.pd.concat(df_list, ignore_index=True)

            # Create a color palette for the plots - ensure we have enough colors
            # Use a palette that provides more visual distinction between models
            palette = self.sns.color_palette("husl", n_colors=max(8, len(model_names)))

            # Log unique models in DataFrame for debugging
            unique_models_in_df = df['Model'].unique()
            logger.info(f"Unique models in dataframe: {list(unique_models_in_df)}")
            logger.info(f"Models we're displaying: {model_names}")

            if len(unique_models_in_df) < len(model_names):
                logger.warning(f"Some models are missing from the DataFrame. Expected {len(model_names)}, got {len(unique_models_in_df)}")

            # Log the number of models being displayed
            logger.info(f"Displaying {len(model_names)} models with {len(palette)} colors")

            # Highlight the primary model with a different color
            if base_model and base_model.get('name') in model_names:
                primary_idx = model_names.index(base_model.get('name'))
                palette[primary_idx] = "#1b78de"  # Use primary color for base model

            # Ensure we're displaying the right models by checking unique values
            unique_models = df['Model'].unique()
            logger.info(f"Unique models in DataFrame: {unique_models}")

            if len(unique_models) < len(model_names):
                logger.warning(f"Some models are missing from the plot. Expected {len(model_names)}, got {len(unique_models)}")

            # 1. Create violin plot as the base layer
            self.sns.violinplot(
                x='Model',
                y=metric_name,
                data=df,
                ax=ax,
                palette=palette,
                inner=None,  # No inner points/box
                alpha=0.6,   # Semi-transparent
                saturation=0.9,
                order=model_names  # Explicitly set the order of models to ensure all are displayed
            )

            # 2. Add boxplot on top of violin plots
            self.sns.boxplot(
                x='Model',
                y=metric_name,
                data=df,
                ax=ax,
                width=0.3,  # Narrower width to fit inside violin
                palette=palette,
                saturation=1.0,
                showfliers=False,  # Hide fliers since we'll show all points with stripplot
                order=model_names  # Explicitly set the order of models to ensure all are displayed
            )

            # 3. Add stripplot for individual points
            self.sns.stripplot(
                x='Model',
                y=metric_name,
                data=df,
                ax=ax,
                size=2.5,
                alpha=0.4,
                jitter=True,
                color='#444444',
                edgecolor='none',
                order=model_names  # Explicitly set the order of models to ensure all are displayed
            )

            # Add markers for base scores - making sure to map to correct model positions
            for model in models_data:
                name = model.get('name', 'Unknown')
                base_score = model.get('baseScore')

                # Skip models not in the display list
                if name not in model_names:
                    continue

                if base_score is not None:
                    # Find the correct position based on model name in model_names
                    model_position = model_names.index(name)

                    marker_size = 120 if model == base_model else 100
                    marker_color = "red" if model == base_model else "orange"
                    ax.scatter(model_position, base_score, marker='D', s=marker_size, color=marker_color,
                              edgecolor='black', linewidth=0.5,
                              label=f"{name} Base Score" if model == base_model else "", zorder=10)

            # Set labels and title
            ax.set_xlabel('Models Analyzed', fontsize=14)
            ax.set_ylabel(metric_name, fontsize=14)
            ax.set_title(title, fontsize=16)

            # Adjust x-axis labels
            if len(models_data) > 3:
                labels = ax.get_xticklabels()
                ax.set_xticks(ax.get_xticks())
                ax.set_xticklabels(labels, rotation=45, ha='right')

            # Add legend for base score
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles=[handles[0]], labels=['Base Score'], loc='best', framealpha=0.9)

            # Add grid for better readability (behind the plots)
            ax.grid(True, axis='y', linestyle='--', alpha=0.5, zorder=-1)

            # Add statistical annotations
            try:
                # Calculate and add some statistical info for each model
                for model in models_data:
                    name = model.get('name', 'Unknown')
                    scores = model.get('scores', [])

                    # Skip models not in the display list
                    if name not in model_names:
                        continue

                    if scores:
                        # Find the correct position based on model name in model_names
                        model_position = model_names.index(name)

                        # Calculate statistics
                        mean = self.np.mean(scores)
                        median = self.np.median(scores)

                        # Annotate mean with larger, more visible text
                        is_base = model == base_model
                        fontweight = 'bold' if is_base else 'normal'
                        fontsize = 10 if is_base else 9

                        # Add mean annotation with a line
                        ax.annotate(f'μ={mean:.3f}',
                                  xy=(model_position, mean),
                                  xytext=(model_position+0.25, mean),
                                  arrowprops=dict(arrowstyle='-', color='black', alpha=0.7),
                                  color='black', fontsize=fontsize, fontweight=fontweight, alpha=0.9)

                        # Add median annotation below
                        ax.annotate(f'med={median:.3f}',
                                  xy=(model_position, median),
                                  xytext=(model_position-0.25, median),
                                  arrowprops=dict(arrowstyle='-', color='black', alpha=0.7),
                                  color='black', fontsize=fontsize-1, fontweight=fontweight, alpha=0.8)
            except Exception as e:
                logger.warning(f"Could not add statistical annotations: {e}")
                pass

        # Adjust figure size based on number of models - wider for more models
        width = max(12, min(16, 8 + len([m for m in models_data if m.get('scores')]) * 1.2))
        return self.generate_encoded_chart(_generate_chart, models_data, title,
                                         metric_name, figsize=(width, 8))
    
    def bar_chart(self, data: Dict[str, List],
                   title: str = "Bar Chart") -> str:
        """
        Generate a simple bar chart.

        Parameters:
        -----------
        data : Dict[str, List]
            Dictionary with 'x' and 'y' lists and optional labels
        title : str, optional
            Chart title

        Returns:
        --------
        str : Base64 encoded image data
        """
        def _generate_chart(ax, data, title):
            # Create DataFrame
            x = data.get('x', [])
            y = data.get('y', [])

            if not x or not y or len(x) != len(y):
                logger.error("Invalid data for bar chart")
                ax.text(0.5, 0.5, "No valid data available", ha='center', va='center', transform=ax.transAxes)
                return

            df = self.pd.DataFrame({'x': x, 'y': y})

            # Create barplot
            self.sns.barplot(x='x', y='y', data=df, ax=ax, palette='deep')

            # Add value labels on top of bars
            for i, v in enumerate(y):
                ax.text(i, v + 0.01, f"{v:.4f}", ha='center', fontsize=10, fontweight='bold')

            # Set labels and title
            ax.set_xlabel(data.get('x_label', ''), fontsize=12)
            ax.set_ylabel(data.get('y_label', ''), fontsize=12)
            ax.set_title(title, fontsize=14)

            # Rotate x-axis labels for better readability
            if len(x) > 3:
                plt_labels = ax.get_xticklabels()
                ax.set_xticklabels(plt_labels, rotation=45, ha='right')

            # Add grid for better readability
            ax.grid(True, axis='y', linestyle='--', alpha=0.7)

        return self.generate_encoded_chart(_generate_chart, data, title, figsize=(10, 6))

    def feature_psi_chart(self, psi_data: Dict[str, List],
                           title: str = "Feature Distribution Shift (PSI)") -> str:
        """
        Generate a bar chart showing the Population Stability Index (PSI)
        for different features, indicating distribution shift.

        Parameters:
        -----------
        psi_data : Dict[str, List]
            Dictionary with 'Feature' and 'PSI' lists
        title : str, optional
            Chart title

        Returns:
        --------
        str : Base64 encoded image data
        """
        def _generate_chart(ax, psi_data, title):
            # Create DataFrame
            features = psi_data['Feature']
            psi_values = psi_data['PSI']

            if not features or not psi_values or len(features) == 0:
                logger.error("Empty PSI data for feature PSI chart")
                ax.text(0.5, 0.5, "No PSI data available", ha='center', va='center', transform=ax.transAxes)
                return

            df = self.pd.DataFrame({'Feature': features, 'PSI': psi_values})

            # Sort by PSI value (descending)
            df = df.sort_values('PSI', ascending=False)

            # Create barplot with a blue color palette
            self.sns.barplot(x='Feature', y='PSI', data=df, ax=ax, palette='Blues_d')

            # Add value labels on top of bars
            for i, v in enumerate(df['PSI']):
                ax.text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=9, fontweight='bold')

            # Set labels and title
            ax.set_xlabel('Features', fontsize=12)
            ax.set_ylabel('PSI (Higher indicates more shift)', fontsize=12)
            ax.set_title(title, fontsize=14)

            # Rotate x-axis labels for better readability
            plt_labels = ax.get_xticklabels()
            ax.set_xticklabels(plt_labels, rotation=45, ha='right')

            # Add grid for better readability
            ax.grid(True, axis='y', linestyle='--', alpha=0.7)

        return self.generate_encoded_chart(_generate_chart, psi_data, title, figsize=(12, 7))

    def distribution_grid_chart(self, models_data: Dict[str, Dict[float, List[float]]],
                               title: str = "Performance Distribution Grid",
                               metric_name: str = "Score",
                               show_stats: bool = True,
                               baseline_scores: Dict[str, float] = None) -> str:
        """
        Generate a comprehensive distribution grid chart showing performance distributions
        across multiple models and perturbation levels in a matrix layout.

        Parameters:
        -----------
        models_data : Dict[str, Dict[float, List[float]]]
            Dictionary with model names as keys, each containing perturbation levels
            and their corresponding score lists
        title : str, optional
            Chart title
        metric_name : str, optional
            Name of the metric being displayed
        show_stats : bool, optional
            Whether to show statistical annotations
        baseline_scores : Dict[str, float], optional
            Baseline scores for each model

        Returns:
        --------
        str : Base64 encoded image data
        """
        if not models_data:
            logger.error("No models data provided for distribution grid chart")
            return ""

        # Extract all unique perturbation levels across all models
        all_levels = set()
        for model_data in models_data.values():
            all_levels.update(model_data.keys())
        sorted_levels = sorted(list(all_levels))

        model_names = list(models_data.keys())
        n_models = len(model_names)
        n_levels = len(sorted_levels)

        if n_models == 0 or n_levels == 0:
            logger.warning("Insufficient data for grid visualization")
            return ""

        # Calculate appropriate figure size
        fig_width = max(4 * n_levels, 12)
        fig_height = max(3 * n_models, 8)

        # Create a new figure with subplots
        fig, axes = self.plt.subplots(n_models, n_levels,
                                     figsize=(fig_width, fig_height),
                                     sharex=False, sharey=True)

        # Handle single row/column cases
        if n_models == 1 and n_levels == 1:
            axes = [[axes]]
        elif n_models == 1:
            axes = [axes]
        elif n_levels == 1:
            axes = [[ax] for ax in axes]

        # Color palette for models
        colors = self.plt.cm.Set3(self.np.linspace(0, 1, n_models))

        # Create plots for each model-level combination
        for i, model_name in enumerate(model_names):
            model_data = models_data[model_name]

            for j, level in enumerate(sorted_levels):
                ax_current = axes[i][j]

                if level in model_data and len(model_data[level]) > 0:
                    scores = model_data[level]

                    # Create violin plot
                    parts = ax_current.violinplot([scores], positions=[0],
                                                widths=0.6, showmeans=True,
                                                showmedians=True)

                    # Set violin color
                    for pc in parts['bodies']:
                        pc.set_facecolor(colors[i])
                        pc.set_alpha(0.7)

                    # Overlay boxplot
                    bp = ax_current.boxplot([scores], positions=[0], widths=0.3,
                                          patch_artist=True, showfliers=False)
                    bp['boxes'][0].set_facecolor(colors[i])
                    bp['boxes'][0].set_alpha(0.9)

                    # Add statistical annotations if enabled
                    if show_stats and len(scores) > 1:
                        mean_score = self.np.mean(scores)
                        std_score = self.np.std(scores)
                        n_samples = len(scores)

                        stats_text = f"μ={mean_score:.3f}\nσ={std_score:.3f}\nn={n_samples}"
                        ax_current.text(0.02, 0.98, stats_text,
                                      transform=ax_current.transAxes,
                                      verticalalignment='top',
                                      fontsize=8,
                                      bbox=dict(boxstyle="round,pad=0.3",
                                               facecolor='white', alpha=0.8))

                    # Add baseline reference if available
                    if baseline_scores and model_name in baseline_scores:
                        baseline = baseline_scores[model_name]
                        ax_current.axhline(y=baseline, color='red',
                                         linestyle='--', alpha=0.7, linewidth=1)

                    ax_current.set_xlim(-0.5, 0.5)
                    ax_current.set_xticks([])

                else:
                    # No data for this combination
                    ax_current.text(0.5, 0.5, "No data",
                                  ha='center', va='center',
                                  transform=ax_current.transAxes,
                                  fontsize=10, alpha=0.5)
                    ax_current.set_xticks([])
                    ax_current.set_yticks([])

                # Set subplot titles
                if i == 0:  # Top row
                    ax_current.set_title(f"Level {level}", fontsize=10, fontweight='bold')

                if j == 0:  # Left column
                    ax_current.set_ylabel(f"{model_name}\n{metric_name}",
                                        fontsize=10, fontweight='bold')

        # Add main title
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

        # Add legend for baseline if used
        if baseline_scores:
            legend_elements = [self.plt.Line2D([0], [0], color='red',
                                             linestyle='--', label='Baseline')]
            fig.legend(handles=legend_elements, loc='upper right',
                      bbox_to_anchor=(0.98, 0.95))

        # Adjust layout
        self.plt.tight_layout()
        self.plt.subplots_adjust(top=0.93)

        # Convert to base64
        import io
        import base64

        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        self.plt.close(fig)

        return f"data:image/png;base64,{image_base64}"

    def coverage_analysis_chart(self, alpha_values: List[float], coverage_values: List[float],
                               title: str = "Coverage Analysis") -> str:
        """
        Generate a chart showing the relationship between nominal confidence level (1-alpha)
        and actual coverage.

        Parameters:
        -----------
        alpha_values : List[float]
            List of alpha values (significance levels)
        coverage_values : List[float]
            List of actual coverage values corresponding to each alpha
        title : str, optional
            Chart title

        Returns:
        --------
        str : Base64 encoded image data
        """
        def _generate_chart(ax, alpha_values, coverage_values, title):
            if not alpha_values or not coverage_values or len(alpha_values) != len(coverage_values):
                logger.error("Invalid data for coverage analysis chart")
                ax.text(0.5, 0.5, "No valid coverage data available", ha='center', va='center', transform=ax.transAxes)
                return

            # Create nominal coverage values (1-alpha)
            nominal_coverage = [1-alpha for alpha in alpha_values]

            # Create DataFrame
            df = self.pd.DataFrame({
                'Nominal Coverage': nominal_coverage,
                'Actual Coverage': coverage_values
            })

            # Plot the calibration curve - actual vs nominal coverage
            self.sns.lineplot(x='Nominal Coverage', y='Actual Coverage', data=df,
                           marker='o', linewidth=2, markersize=8, color='blue', ax=ax)

            # Add ideal calibration line (diagonal)
            ax.plot([0, 1], [0, 1], linestyle='--', color='red', alpha=0.7, label='Ideal Calibration')

            # Set labels and title
            ax.set_xlabel('Nominal Coverage (1-α)', fontsize=12)
            ax.set_ylabel('Actual Coverage', fontsize=12)
            ax.set_title(title, fontsize=14)

            # Set equal aspect ratio
            ax.set_aspect('equal')

            # Set axis limits
            min_val = min(min(nominal_coverage), min(coverage_values), 0)
            max_val = max(max(nominal_coverage), max(coverage_values), 1)
            padding = 0.05
            ax.set_xlim(min_val - padding, max_val + padding)
            ax.set_ylim(min_val - padding, max_val + padding)

            # Add legend
            ax.legend()

            # Add grid for better readability
            ax.grid(True, linestyle='--', alpha=0.7)

        return self.generate_encoded_chart(_generate_chart, alpha_values, coverage_values, title, figsize=(9, 9))

    def model_metrics_heatmap(self, results_df: Dict[str, List],
                          title: str = "Model Metrics Comparison") -> str:
        """
        Create a heatmap for comparing metrics across different models.

        Parameters:
        -----------
        results_df : Dict[str, List]
            Dictionary with model metrics data
        title : str, optional
            Chart title

        Returns:
        --------
        str : Base64 encoded image data
        """
        def _generate_chart(ax, results_df, title):
            # Prepare data for the heatmap
            if 'model' not in results_df or not results_df['model']:
                logger.error("No model data for metrics heatmap")
                ax.text(0.5, 0.5, "No model data available", ha='center', va='center', transform=ax.transAxes)
                return

            metrics = [key for key in results_df.keys() if key != 'model']
            models = results_df['model']

            if not metrics:
                logger.error("No metrics found for heatmap")
                ax.text(0.5, 0.5, "No metrics available", ha='center', va='center', transform=ax.transAxes)
                return

            # Create a matrix for the heatmap
            heatmap_data = []
            for metric in metrics:
                # Skip if metric not in results
                if metric not in results_df or len(results_df[metric]) != len(models):
                    continue
                heatmap_data.append(results_df[metric])

            # Convert to numpy array
            heatmap_array = self.np.array(heatmap_data)

            # Normalize data for the heatmap (by metric)
            normalized_data = self.np.zeros_like(heatmap_array)
            for i, row in enumerate(heatmap_array):
                # Some metrics are better when lower (like mse, mae, mean_width)
                if metrics[i] in ["mse", "mae", "mean_width"]:
                    # For metrics where lower is better, invert the scaling
                    if self.np.max(row) > self.np.min(row):
                        normalized_data[i] = 1 - (row - self.np.min(row)) / (self.np.max(row) - self.np.min(row))
                    else:
                        normalized_data[i] = 0.5  # All values are the same
                else:
                    # For metrics where higher is better (like coverage)
                    if self.np.max(row) > self.np.min(row):
                        normalized_data[i] = (row - self.np.min(row)) / (self.np.max(row) - self.np.min(row))
                    else:
                        normalized_data[i] = 0.5  # All values are the same

            # Create the heatmap
            self.sns.heatmap(
                normalized_data,
                annot=heatmap_array.round(4),
                fmt=".4f",
                cmap="YlGnBu",
                linewidths=0.5,
                cbar_kws={"label": "Normalized Performance"},
                ax=ax
            )

            # Set labels
            ax.set_yticks(self.np.arange(len(metrics)) + 0.5)
            ax.set_yticklabels(metrics, rotation=0)

            ax.set_xticks(self.np.arange(len(models)) + 0.5)
            ax.set_xticklabels(models, rotation=45, ha="right")

            # Set title
            ax.set_title(title, fontsize=14)

        return self.generate_encoded_chart(_generate_chart, results_df, title, figsize=(12, 8))

    def metrics_radar_chart(self, models_metrics: Dict[str, Dict],
                          title: str = "Model Metrics Comparison") -> str:
        """
        Generate a radar chart for comparing multiple models across metrics.
        
        Parameters:
        -----------
        models_metrics : Dict[str, Dict]
            Dictionary of model data with keys as model names and values as dictionaries 
            containing 'metrics' with metric names and values
        title : str, optional
            Chart title
            
        Returns:
        --------
        str : Base64 encoded image data
        """
        def _generate_chart(ax, models_metrics, title):
            try:
                # Get all unique metrics
                all_metrics = set()
                for model_data in models_metrics.values():
                    if 'metrics' in model_data:
                        all_metrics.update(model_data['metrics'].keys())
                
                # Skip if no metrics
                if not all_metrics:
                    logger.error("No metrics found for radar chart")
                    ax.text(0.5, 0.5, "No metrics available", ha='center', va='center', transform=ax.transAxes)
                    return
                
                # Convert to list and sort
                metrics = sorted(list(all_metrics))
                
                # Number of metrics
                N = len(metrics)
                
                # Create angle values
                angles = [n / float(N) * 2 * math.pi for n in range(N)]
                angles += angles[:1]  # Close the loop
                
                # Set up the axis as a polar plot
                ax = self.plt.subplot(111, polar=True)
                
                # Set first axis at the top
                ax.set_theta_offset(math.pi / 2)
                ax.set_theta_direction(-1)
                
                # Draw axis lines for each angle
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(metrics)
                
                # Draw y-axis grid lines
                ax.set_rlabel_position(0)
                ax.set_ylim(0, 1)
                ax.set_yticks([0.25, 0.5, 0.75, 1])
                ax.set_yticklabels(["0.25", "0.5", "0.75", "1"], color="grey", size=8)
                
                # Add plot title
                ax.set_title(title, size=16, y=1.1)
                
                # Plot data for each model
                for i, (model_name, model_data) in enumerate(models_metrics.items()):
                    if 'metrics' not in model_data:
                        continue
                    
                    # Get values for each metric, defaulting to 0
                    values = [model_data['metrics'].get(metric, 0) for metric in metrics]
                    
                    # Ensure all values are in [0, 1] range
                    values = [min(max(v, 0), 1) for v in values]
                    
                    # Close the loop
                    values += values[:1]
                    
                    # Plot values
                    ax.plot(angles, values, linewidth=2, linestyle='solid', label=model_name)
                    ax.fill(angles, values, alpha=0.1)
                
                # Add legend
                ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
                
            except Exception as e:
                logger.error(f"Error generating radar chart: {str(e)}")
                ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', transform=ax.transAxes)
        
        return self.generate_encoded_chart(_generate_chart, models_metrics, title, figsize=(10, 10))
    
    def heatmap_chart(self, matrix: List[List[float]], 
                     x_labels: List[str] = None, y_labels: List[str] = None,
                     title: str = "Correlation Heatmap", 
                     cmap: str = "viridis") -> str:
        """
        Generate a heatmap chart.
        
        Parameters:
        -----------
        matrix : List[List[float]]
            2D matrix of values to display
        x_labels : List[str], optional
            Labels for x-axis
        y_labels : List[str], optional
            Labels for y-axis
        title : str, optional
            Chart title
        cmap : str, optional
            Colormap name
            
        Returns:
        --------
        str : Base64 encoded image data
        """
        def _generate_chart(ax, matrix, x_labels, y_labels, title, cmap):
            # Convert to numpy array if not already
            matrix_array = self.np.array(matrix)
            
            # Create heatmap
            heatmap = self.sns.heatmap(
                matrix_array,
                annot=True,
                fmt=".2f",
                cmap=cmap,
                linewidths=.5,
                ax=ax,
                xticklabels=x_labels,
                yticklabels=y_labels,
                cbar_kws={'label': 'Value'}
            )
            
            # Set title
            ax.set_title(title)
            
            # Rotate x-axis labels if there are many
            if x_labels and len(x_labels) > 5:
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Determine figure size based on matrix dimensions
        rows = len(matrix)
        cols = len(matrix[0]) if matrix else 0
        figsize = (max(6, cols * 0.8), max(6, rows * 0.8))
        
        return self.generate_encoded_chart(_generate_chart, matrix, x_labels, y_labels, 
                                         title, cmap, figsize=figsize)