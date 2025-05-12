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