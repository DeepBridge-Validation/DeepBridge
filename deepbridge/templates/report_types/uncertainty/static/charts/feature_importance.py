"""
Module for generating feature importance heatmap charts for uncertainty.
"""

import logging
from typing import Dict, List, Any, Optional
import numpy as np

from .base_chart import BaseChartGenerator

# Configure logger
logger = logging.getLogger("deepbridge.reports")

class FeatureImportanceChart(BaseChartGenerator):
    """
    Generate charts showing feature importance for uncertainty.
    """
    
    def generate(self,
                feature_importance_data: Dict[str, Dict[str, Dict[str, float]]],
                title: str = "Feature Importance for Uncertainty") -> str:
        """
        Generate a heatmap chart showing feature importance for uncertainty across different models.

        Parameters:
        -----------
        feature_importance_data : Dict[str, Dict[str, Dict[str, float]]]
            Dictionary with different data structures supported:
            1. Multiple models with feature importance:
               {
                   "model_name": {
                       "feature_importance": {
                           "feature_name": importance_value,
                           ...
                       }
                   },
                   ...
               }
            2. Direct feature importance for a single model:
               {
                   "feature_name": importance_value,
                   ...
               }
            3. Multiple models with different feature types:
               {
                   "model_name": {
                       "feature_type": {
                           "feature_name": importance_value,
                           ...
                       }
                   },
                   ...
               }
        title : str, optional
            Chart title

        Returns:
        --------
        str : Base64 encoded image or empty string if data invalid
        """
        self._validate_chart_generator()

        # Validate input data
        if not self._validate_data(feature_importance_data):
            logger.warning("Invalid feature importance data for chart")
            return ""

        # Determine data structure type
        data_type = self._detect_data_structure(feature_importance_data)
        
        if data_type == "invalid":
            logger.warning("Unsupported feature importance data structure")
            return ""

        # If using existing chart generator
        if self.chart_generator and hasattr(self.chart_generator, 'heatmap_chart'):
            try:
                # Prepare data for chart generator
                # This depends on the expected format for the chart generator
                # You'll need to adjust this based on your chart generator's API
                return ""
            except Exception as e:
                logger.error(f"Error using chart generator for feature importance: {str(e)}")

        # Fallback - implement direct charting
        try:
            # Process data based on detected structure
            processed_data = self._process_feature_importance_data(feature_importance_data, data_type)

            # Check if processed_data is valid
            if not processed_data:
                logger.warning("Failed to process feature importance data - returned None")
                return ""

            # Check individual components
            has_matrix = 'matrix' in processed_data and processed_data['matrix'] is not None
            has_x_labels = 'x_labels' in processed_data and processed_data['x_labels']
            has_y_labels = 'y_labels' in processed_data and processed_data['y_labels']

            # For numpy arrays, check size properly
            if has_matrix and hasattr(processed_data['matrix'], 'size'):
                has_matrix = processed_data['matrix'].size > 0

            if not (has_matrix and has_x_labels and has_y_labels):
                logger.warning("Failed to process feature importance data")
                return ""
            
            # Create figure with appropriate size
            num_features = len(processed_data['y_labels'])
            num_models = len(processed_data['x_labels'])
            
            # Adjust figure size based on number of features and models
            fig_height = max(6, min(20, 4 + 0.3 * num_features))
            fig_width = max(8, min(16, 6 + 0.5 * num_models))
            
            fig, ax = self.plt.subplots(figsize=(fig_width, fig_height))
            
            # Generate heatmap
            heatmap = self.sns.heatmap(
                processed_data['matrix'],
                annot=True,
                cmap="YlOrRd",
                cbar_kws={'label': 'Importance Score'},
                fmt=".3f",
                linewidths=.5,
                ax=ax
            )
            
            # Set labels and title
            ax.set_title(title)
            ax.set_xlabel(processed_data.get('x_label', 'Model'))
            ax.set_ylabel(processed_data.get('y_label', 'Feature'))
            
            # Set tick labels
            ax.set_xticklabels(processed_data['x_labels'], rotation=45, ha='right')
            ax.set_yticklabels(processed_data['y_labels'])
            
            # Adjust layout
            fig.tight_layout()
            
            # Save the figure to base64
            return self._save_figure_to_base64(fig)
        except Exception as e:
            logger.error(f"Error generating feature importance chart: {str(e)}")
            return ""
            
    def _detect_data_structure(self, data):
        """Detect the structure of the feature importance data."""
        # Case 1: Single model direct feature importance
        # Check if all values are numeric (including numpy types)
        try:
            if all(isinstance(v, (int, float)) or (hasattr(v, 'item') and callable(v.item)) for v in data.values()):
                return "single_model_direct"
        except:
            pass
            
        # Case 2: Multiple models with feature importance
        if all(isinstance(v, dict) for v in data.values()):
            # Check if models have feature_importance key
            sample_model = list(data.values())[0]
            if "feature_importance" in sample_model and isinstance(sample_model["feature_importance"], dict):
                return "multi_model_with_fi_key"
            
            # Check if first level keys are model names and second level keys are feature names
            if all(isinstance(v, dict) for model in data.values() for v in model.values()):
                for model_name, model_data in data.items():
                    for feature_type, features in model_data.items():
                        if isinstance(features, dict) and all(isinstance(v, (int, float)) for v in features.values()):
                            return "multi_model_with_feature_types"
        
        return "invalid"
        
    def _process_feature_importance_data(self, data, data_type):
        """Process the feature importance data based on detected structure."""
        if data_type == "single_model_direct":
            # Convert numpy types to float if necessary
            converted_data = {}
            for k, v in data.items():
                if hasattr(v, 'item'):
                    converted_data[k] = float(v.item())
                else:
                    converted_data[k] = float(v)

            # Sort features by importance
            sorted_features = sorted(converted_data.items(), key=lambda x: x[1], reverse=True)
            features = [f[0] for f in sorted_features]
            values = [f[1] for f in sorted_features]
            
            # Top N features only
            top_n = min(15, len(features))
            features = features[:top_n]
            values = values[:top_n]
            
            # Create a matrix with a single column
            matrix = np.array(values).reshape(-1, 1)
            
            return {
                'matrix': matrix,
                'x_labels': ['Importance'],
                'y_labels': features,
                'x_label': 'Score',
                'y_label': 'Feature'
            }
            
        elif data_type == "multi_model_with_fi_key":
            # Extract feature importance for each model
            models_fi = {}
            for model_name, model_data in data.items():
                if "feature_importance" in model_data and isinstance(model_data["feature_importance"], dict):
                    models_fi[model_name] = model_data["feature_importance"]
            
            # Get unique features across all models
            all_features = set()
            for model_fi in models_fi.values():
                all_features.update(model_fi.keys())
            
            # Sort features by average importance
            feature_avg_importance = {}
            for feature in all_features:
                values = [model_fi.get(feature, 0) for model_fi in models_fi.values()]
                feature_avg_importance[feature] = sum(values) / len(values)
            
            sorted_features = sorted(feature_avg_importance.items(), key=lambda x: x[1], reverse=True)
            features = [f[0] for f in sorted_features]
            
            # Top N features only
            top_n = min(15, len(features))
            features = features[:top_n]
            
            # Create matrix
            model_names = list(models_fi.keys())
            matrix = np.zeros((len(features), len(model_names)))
            
            for i, feature in enumerate(features):
                for j, model_name in enumerate(model_names):
                    matrix[i, j] = models_fi[model_name].get(feature, 0)
            
            return {
                'matrix': matrix,
                'x_labels': model_names,
                'y_labels': features,
                'x_label': 'Model',
                'y_label': 'Feature'
            }
            
        elif data_type == "multi_model_with_feature_types":
            # This is more complex as it involves multiple models and feature types
            # For simplicity, we'll focus on one feature type for now
            feature_type = None
            for model_name, model_data in data.items():
                for ft in model_data.keys():
                    if isinstance(model_data[ft], dict) and all(isinstance(v, (int, float)) for v in model_data[ft].values()):
                        feature_type = ft
                        break
                if feature_type:
                    break
            
            if not feature_type:
                return None
                
            # Extract feature importance for the selected feature type
            models_fi = {}
            for model_name, model_data in data.items():
                if feature_type in model_data and isinstance(model_data[feature_type], dict):
                    models_fi[model_name] = model_data[feature_type]
            
            # Get unique features across all models
            all_features = set()
            for model_fi in models_fi.values():
                all_features.update(model_fi.keys())
            
            # Sort features by average importance
            feature_avg_importance = {}
            for feature in all_features:
                values = [model_fi.get(feature, 0) for model_fi in models_fi.values()]
                feature_avg_importance[feature] = sum(values) / len(values)
            
            sorted_features = sorted(feature_avg_importance.items(), key=lambda x: x[1], reverse=True)
            features = [f[0] for f in sorted_features]
            
            # Top N features only
            top_n = min(15, len(features))
            features = features[:top_n]
            
            # Create matrix
            model_names = list(models_fi.keys())
            matrix = np.zeros((len(features), len(model_names)))
            
            for i, feature in enumerate(features):
                for j, model_name in enumerate(model_names):
                    matrix[i, j] = models_fi[model_name].get(feature, 0)
            
            return {
                'matrix': matrix,
                'x_labels': model_names,
                'y_labels': features,
                'x_label': 'Model',
                'y_label': f'{feature_type} Feature'
            }
            
        return None