"""
Module for evaluating model robustness against perturbations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
import time

from deepbridge.validation.wrappers.robustness.data_perturber import DataPerturber

class RobustnessEvaluator:
    """
    Evaluates model robustness against data perturbations.
    """
    
    def __init__(self, 
                 dataset, 
                 metric: str = 'AUC', 
                 verbose: bool = False,
                 random_state: Optional[int] = None,
                 n_iterations: int = 1):
        """
        Initialize the robustness evaluator.
        
        Parameters:
        -----------
        dataset : DBDataset
            Dataset object containing training/test data and model
        metric : str
            Performance metric to use for evaluation ('AUC', 'accuracy', 'mse', etc.)
        verbose : bool
            Whether to print progress information
        random_state : int or None
            Random seed for reproducibility
        n_iterations : int
            Number of iterations to perform for each perturbation level to get statistical robustness
        """
        self.dataset = dataset
        self.metric = metric
        self.verbose = verbose
        self.n_iterations = n_iterations
        
        # Create data perturber
        self.data_perturber = DataPerturber()
        if random_state is not None:
            self.data_perturber.set_random_state(random_state)
        
        # Determine problem type based on dataset or model
        self._problem_type = self._determine_problem_type()
        
        if self.verbose:
            print(f"Problem type detected: {self._problem_type}")
            print(f"Using metric: {self.metric}")
            print(f"Performing {self.n_iterations} iterations per perturbation level")
    
    def _determine_problem_type(self) -> str:
        """Determine if the problem is classification or regression"""
        # Try to get problem type from dataset
        if hasattr(self.dataset, 'problem_type'):
            return self.dataset.problem_type
        
        # Try to infer from the model
        if hasattr(self.dataset, 'model'):
            model = self.dataset.model
            if hasattr(model, 'predict_proba'):
                return 'classification'
            else:
                return 'regression'
        
        # Default to classification
        return 'classification'
    
    def get_model_feature_importance(self) -> Dict[str, float]:
        """
        Extract native feature importance from the model if available.
        
        Returns:
        --------
        Dict[str, float] : Dictionary mapping feature names to importance scores,
                         or empty dict if not available
        """
        if not hasattr(self.dataset, 'model') or self.dataset.model is None:
            return {}
            
        model = self.dataset.model
        feature_names = self.dataset.features
        feature_importance = {}
        
        # Try different attributes/methods to get feature importance
        if hasattr(model, 'feature_importances_'):
            # For tree-based models (Random Forest, XGBoost, etc.)
            importances = model.feature_importances_
            if len(importances) == len(feature_names):
                feature_importance = dict(zip(feature_names, importances))
        elif hasattr(model, 'coef_'):
            # For linear models
            if hasattr(model.coef_, 'shape') and len(model.coef_.shape) > 1:
                # For multi-class models, take the average of coefficients
                importances = np.abs(model.coef_).mean(axis=0)
            else:
                importances = np.abs(model.coef_)
                
            if len(importances) == len(feature_names):
                feature_importance = dict(zip(feature_names, importances))
        elif hasattr(model, 'feature_importance'):
            # For models with a feature_importance method
            try:
                importances = model.feature_importance()
                if len(importances) == len(feature_names):
                    feature_importance = dict(zip(feature_names, importances))
            except:
                pass
                
        # Normalize importances to sum to 1
        if feature_importance:
            total = sum(feature_importance.values())
            if total > 0:
                feature_importance = {k: v / total for k, v in feature_importance.items()}
                
        return feature_importance
    
    def calculate_base_score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Calculate baseline score on unperturbed data.
        
        Parameters:
        -----------
        X : DataFrame
            Feature data
        y : Series
            Target variable
            
        Returns:
        --------
        float : Baseline performance score
        """
        if not hasattr(self.dataset, 'model') or self.dataset.model is None:
            raise ValueError("Dataset has no model for evaluation")
            
        model = self.dataset.model
        
        if self._problem_type == 'classification':
            # For classification, use predict_proba if available
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X)

                # Use appropriate metric
                if self.metric.upper() in ['AUC', 'ROC_AUC']:
                    from sklearn.metrics import roc_auc_score

                    # Check if multiclass
                    n_classes = y_pred_proba.shape[1] if len(y_pred_proba.shape) > 1 else 1

                    if n_classes > 2:
                        # Multiclass: use ovr (one-vs-rest) strategy
                        score = roc_auc_score(y, y_pred_proba, multi_class='ovr', average='macro')
                    elif n_classes == 2:
                        # Binary: use second column (positive class probability)
                        score = roc_auc_score(y, y_pred_proba[:, 1])
                    else:
                        # Single class (unusual case)
                        score = 0.0
                else:
                    # For other metrics, convert probabilities to class labels
                    if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 2:
                        # Multiclass: use argmax
                        y_pred = np.argmax(y_pred_proba, axis=1)
                    elif len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] == 2:
                        # Binary: threshold at 0.5
                        y_pred = (y_pred_proba[:, 1] > 0.5).astype(int)
                    else:
                        # Single column
                        y_pred = (y_pred_proba > 0.5).astype(int)
                    score = self._get_metric_score(y, y_pred)
            else:
                # Fall back to predict for models without predict_proba
                y_pred = model.predict(X)
                score = self._get_metric_score(y, y_pred)
        else:
            # For regression, use predict
            y_pred = model.predict(X)
            score = self._get_metric_score(y, y_pred)
            
        return score
    
    def _get_metric_score(self, y_true: pd.Series, y_pred: Union[pd.Series, np.ndarray]) -> float:
        """
        Calculate score for the selected metric.
        
        Parameters:
        -----------
        y_true : Series
            True target values
        y_pred : Series or ndarray
            Predicted values
            
        Returns:
        --------
        float : Score for the selected metric
        """
        if self._problem_type == 'classification':
            metric_map = {
                'ACCURACY': 'accuracy_score',
                'F1': 'f1_score',
                'PRECISION': 'precision_score',
                'RECALL': 'recall_score'
            }
            
            if self.metric.upper() in metric_map:
                from sklearn import metrics
                metric_func = getattr(metrics, metric_map[self.metric.upper()])
                return metric_func(y_true, y_pred)
            else:
                # Default to accuracy
                from sklearn.metrics import accuracy_score
                return accuracy_score(y_true, y_pred)
        else:
            # Regression metrics
            metric_map = {
                'MSE': 'mean_squared_error',
                'MAE': 'mean_absolute_error',
                'R2': 'r2_score',
                'RMSE': 'root_mean_squared_error'
            }
            
            if self.metric.upper() in metric_map:
                from sklearn import metrics
                if self.metric.upper() == 'RMSE':
                    from sklearn.metrics import mean_squared_error
                    return np.sqrt(mean_squared_error(y_true, y_pred))
                else:
                    metric_func = getattr(metrics, metric_map[self.metric.upper()])
                    return metric_func(y_true, y_pred)
            else:
                # Default to MSE for regression
                from sklearn.metrics import mean_squared_error
                return mean_squared_error(y_true, y_pred)
    
    def evaluate_perturbation(self, 
                              X: pd.DataFrame, 
                              y: pd.Series, 
                              perturb_method: str, 
                              level: float, 
                              feature_subset: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate model performance on perturbed data.
        
        Parameters:
        -----------
        X : DataFrame
            Feature data
        y : Series
            Target variable
        perturb_method : str
            Method to use ('raw' or 'quantile')
        level : float
            Level of perturbation to apply
        feature_subset : List[str] or None
            Specific features to perturb (None for all)
            
        Returns:
        --------
        Dict[str, Any] : Results of the evaluation
        """
        # Create a copy of the original X for predictions
        X_full = X.copy()
        
        # Calculate baseline score first using full feature set
        base_score = self.calculate_base_score(X_full, y)
        
        # Run multiple iterations for statistical robustness
        perturbed_scores = []
        seed_offset = 0
        
        for iteration in range(self.n_iterations):
            # Use a different random seed for each iteration
            if iteration > 0:
                seed_offset = iteration * 1000
                if hasattr(self.data_perturber.rng, 'seed'):
                    original_seed = self.data_perturber.rng.get_state()[1][0]
                    self.data_perturber.set_random_state(original_seed + seed_offset)
            
            # Perturb features based on feature_subset
            X_perturbed = X_full.copy()
            
            # Apply perturbation only to the specified features (or all if none specified)
            perturb_features = feature_subset
            X_perturbed = self.data_perturber.perturb_data(
                X_full, 
                perturb_method, 
                level, 
                perturb_features
            )
            
            # Calculate score on perturbed data
            perturbed_score = self.calculate_score_on_perturbed_data(X_perturbed, y)
            perturbed_scores.append(perturbed_score)
        
        # Calculate statistics across iterations
        mean_perturbed_score = np.mean(perturbed_scores)
        std_perturbed_score = np.std(perturbed_scores)
        
        # Determine if higher scores are better based on metric
        higher_is_better = not (self._problem_type == 'regression' and self.metric.upper() in ['MSE', 'MAE', 'RMSE'])
        
        # Calculate impact - for regression metrics like MSE, lower is better
        if not higher_is_better:
            # For these metrics, higher values mean worse performance
            mean_impact = (mean_perturbed_score - base_score) / max(base_score, 1e-10)
        else:
            # For classification metrics, higher values mean better performance
            mean_impact = (base_score - mean_perturbed_score) / max(base_score, 1e-10)
            
        # Calculate worst_score based on n_iterations and alpha (default 0.1 or 10%)
        alpha = 0.1  # Proportion of "worst" cases to consider
        all_scores_np = np.array(perturbed_scores)
        worst_count = max(1, int(len(all_scores_np) * alpha))
        
        if higher_is_better:
            # For metrics where higher is better, worst scores are the lowest ones
            worst_indices = np.argsort(all_scores_np)[:worst_count]
        else:
            # For metrics where lower is better, worst scores are the highest ones
            worst_indices = np.argsort(all_scores_np)[-worst_count:]
            
        worst_score = np.mean(all_scores_np[worst_indices])
        
        # Return comprehensive results with iteration statistics
        return {
            'base_score': base_score,
            'perturbed_score': mean_perturbed_score,
            'std_perturbed_score': std_perturbed_score,
            'impact': mean_impact,
            'worst_score': worst_score,
            'iterations': {
                'n_iterations': self.n_iterations,
                'scores': perturbed_scores
            },
            'perturbation': {
                'method': perturb_method,
                'level': level,
                'features': feature_subset
            }
        }
    
    def calculate_score_on_perturbed_data(self, X_perturbed: pd.DataFrame, y: pd.Series) -> float:
        """
        Calculate score on perturbed data.
        
        Parameters:
        -----------
        X_perturbed : DataFrame
            Perturbed feature data
        y : Series
            Target variable
            
        Returns:
        --------
        float : Performance score on perturbed data
        """
        if not hasattr(self.dataset, 'model') or self.dataset.model is None:
            raise ValueError("Dataset has no model for evaluation")
            
        model = self.dataset.model
        
        if self._problem_type == 'classification':
            # For classification, use predict_proba if available
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_perturbed)

                # Use appropriate metric
                if self.metric.upper() in ['AUC', 'ROC_AUC']:
                    from sklearn.metrics import roc_auc_score

                    # Check if multiclass
                    n_classes = y_pred_proba.shape[1] if len(y_pred_proba.shape) > 1 else 1

                    if n_classes > 2:
                        # Multiclass: use ovr (one-vs-rest) strategy
                        score = roc_auc_score(y, y_pred_proba, multi_class='ovr', average='macro')
                    elif n_classes == 2:
                        # Binary: use second column (positive class probability)
                        score = roc_auc_score(y, y_pred_proba[:, 1])
                    else:
                        # Single class (unusual case)
                        score = 0.0
                else:
                    # For other metrics, convert probabilities to class labels
                    if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 2:
                        # Multiclass: use argmax
                        y_pred = np.argmax(y_pred_proba, axis=1)
                    elif len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] == 2:
                        # Binary: threshold at 0.5
                        y_pred = (y_pred_proba[:, 1] > 0.5).astype(int)
                    else:
                        # Single column
                        y_pred = (y_pred_proba > 0.5).astype(int)
                    score = self._get_metric_score(y, y_pred)
            else:
                # Fall back to predict for models without predict_proba
                y_pred = model.predict(X_perturbed)
                score = self._get_metric_score(y, y_pred)
        else:
            # For regression, use predict
            y_pred = model.predict(X_perturbed)
            score = self._get_metric_score(y, y_pred)
            
        return score
    
    def evaluate_feature_importance(self, 
                                   X: pd.DataFrame, 
                                   y: pd.Series, 
                                   perturb_method: str, 
                                   level: float, 
                                   feature_subset: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Evaluate feature importance for robustness by perturbing each feature individually.
        
        Parameters:
        -----------
        X : DataFrame
            Feature data
        y : Series
            Target variable
        perturb_method : str
            Method to use ('raw' or 'quantile')
        level : float
            Level of perturbation to apply
        feature_subset : List[str] or None
            Specific features to evaluate (None for all features)
            
        Returns:
        --------
        Dict[str, float] : Mapping of feature names to importance scores
        """
        # Create a copy of the original X for predictions
        X_full = X.copy()
        
        # Calculate baseline score using full feature set
        base_score = self.calculate_base_score(X_full, y)
        
        # Get features to test
        features_to_test = feature_subset or X.columns.tolist()
        
        # Ensure all features in feature_subset are in X
        if feature_subset:
            valid_features = [f for f in feature_subset if f in X.columns]
            if len(valid_features) < len(feature_subset) and self.verbose:
                missing = set(feature_subset) - set(valid_features)
                print(f"Warning: Some requested features not found in dataset: {missing}")
            features_to_test = valid_features
        
        # Dictionary to store importance scores and detailed results
        importance_scores = {}
        feature_results = {}
        
        if self.verbose:
            print(f"Evaluating feature importance with {perturb_method} perturbation at level {level}")
            print(f"Using {self.n_iterations} iterations per feature for statistical robustness")
            if feature_subset:
                print(f"Analyzing subset of {len(features_to_test)} features")
        
        # Determine if higher scores are better based on metric
        higher_is_better = not (self._problem_type == 'regression' and self.metric.upper() in ['MSE', 'MAE', 'RMSE'])
                
        # Perturb each feature and evaluate with multiple iterations
        for i, feature in enumerate(features_to_test):
            if self.verbose and (i+1) % 10 == 0:
                print(f"  - Processed {i+1}/{len(features_to_test)} features")
            
            # Run multiple iterations for each feature
            feature_impacts = []
            feature_scores = []
            
            for iteration in range(self.n_iterations):
                # Use a different random seed for each iteration
                if iteration > 0:
                    seed_offset = iteration * 1000
                    if hasattr(self.data_perturber.rng, 'seed'):
                        original_seed = self.data_perturber.rng.get_state()[1][0]
                        self.data_perturber.set_random_state(original_seed + seed_offset)
                
                # Create perturbed dataset with only this feature perturbed
                # Always starting from the full feature set to avoid compounding effects
                X_perturbed = self.data_perturber.perturb_data(X_full.copy(), perturb_method, level, [feature])
                
                # Calculate score on perturbed data
                perturbed_score = self.calculate_score_on_perturbed_data(X_perturbed, y)
                feature_scores.append(perturbed_score)
                
                # Calculate impact - for regression metrics like MSE, lower is better
                if not higher_is_better:
                    # For these metrics, higher values mean worse performance
                    impact = (perturbed_score - base_score) / max(base_score, 1e-10)
                else:
                    # For classification metrics, higher values mean better performance
                    impact = (base_score - perturbed_score) / max(base_score, 1e-10)
                
                feature_impacts.append(impact)
            
            # Calculate mean and std of impacts across iterations
            mean_impact = np.mean(feature_impacts)
            std_impact = np.std(feature_impacts)
            
            # Calculate worst_score based on n_iterations and alpha (default 0.1 or 10%)
            alpha = 0.1  # Proportion of "worst" cases to consider
            all_scores_np = np.array(feature_scores)
            worst_count = max(1, int(len(all_scores_np) * alpha))
            
            if higher_is_better:
                # For metrics where higher is better, worst scores are the lowest ones
                worst_indices = np.argsort(all_scores_np)[:worst_count]
            else:
                # For metrics where lower is better, worst scores are the highest ones
                worst_indices = np.argsort(all_scores_np)[-worst_count:]
                
            worst_score = np.mean(all_scores_np[worst_indices])
            
            # Store importance score (mean impact)
            importance_scores[feature] = mean_impact
            
            # Store detailed results for this feature
            feature_results[feature] = {
                'mean_impact': mean_impact,
                'std_impact': std_impact,
                'mean_score': np.mean(feature_scores),
                'std_score': np.std(feature_scores),
                'worst_score': worst_score,
                'iterations': {
                    'n_iterations': self.n_iterations,
                    'impacts': feature_impacts,
                    'scores': feature_scores
                }
            }
        
        # Add detailed results to importance_scores dictionary
        # importance_scores['_detailed_results'] = feature_results
        
        return importance_scores