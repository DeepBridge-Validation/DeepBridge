import numpy as np
import pandas as pd
from typing import Dict, List, Union, Callable, Optional, Tuple, Any


from deepbridge.validation.perturbation import RawPerturbation, QuantilePerturbation, CategoricalPerturbation
from deepbridge.validation.robustness_metrics import get_metric_function

class RobustnessTest:
    """
    Evaluates the robustness of machine learning models against data perturbations.
    
    This class implements methods to test how a model behaves when its input data
    is slightly perturbed, following predictability, interpretability, and model validation
    best practices.
    """
    
    def __init__(self):
        """Initialize the RobustnessTest class."""
        # Instantiate perturbation methods
        self.perturbation_methods = {
            'raw': RawPerturbation(),
            'quantile': QuantilePerturbation(),
            'categorical': CategoricalPerturbation()
        }
    
    def _is_classification(self, y: np.ndarray) -> bool:
        """
        Determine if the problem is classification based on the target values.
        
        Args:
            y: Array with target values
            
        Returns:
            bool: True if classification, False if regression
        """
        # Check if target contains only discrete values (classification)
        unique_vals = np.unique(y)
        return len(unique_vals) <= 10 and all(isinstance(val, (int, bool)) or (hasattr(val, 'is_integer') and val.is_integer()) for val in unique_vals)
    
    def _ensure_dataframe(self, X: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
        """
        Ensure X is a pandas DataFrame.
        
        Args:
            X: Input data as array or DataFrame
            
        Returns:
            pd.DataFrame: Input data as DataFrame
        """
        if isinstance(X, np.ndarray):
            return pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        return X
    
    def _get_perturb_method(self, method_name: str) -> Any:
        """
        Get the perturbation method based on name.
        
        Args:
            method_name: Name of the perturbation method
            
        Returns:
            Perturbation method object
        """
        if method_name not in self.perturbation_methods:
            raise ValueError(f"Perturbation method '{method_name}' not supported. Available methods: {list(self.perturbation_methods.keys())}")
        return self.perturbation_methods[method_name]
    
    def evaluate_robustness(
        self,
        models: Dict[str, object],
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        perturb_method: str = 'raw',
        perturb_sizes: List[float] = [0.1, 0.3, 0.5, 0.7, 1.0],
        perturb_features: Optional[List[str]] = None,
        cat_features: Optional[List[str]] = None,
        metric: Optional[str] = None,
        n_iterations: int = 10,
        alpha: float = 0.1,
        random_state: Optional[int] = None
    ) -> Dict:
        """
        Evaluate the robustness of multiple models under perturbations.
        
        Args:
            models: Dictionary of models to evaluate
            X: Input features
            y: Target values
            perturb_method: Perturbation method ('raw', 'quantile', or 'categorical')
            perturb_sizes: List of perturbation sizes to test
            perturb_features: List of features to perturb (None = all)
            cat_features: List of categorical features for specific perturbation
            metric: Evaluation metric name
            n_iterations: Number of iterations per perturbation size
            alpha: Proportion of worst samples to consider
            random_state: Random seed for reproducibility
            
        Returns:
            Dict: Robustness evaluation results
        """
        # Set random seed if provided
        if random_state is not None:
            np.random.seed(random_state)
        
        # Verify if it's classification or regression
        is_classification = self._is_classification(y)
        
        # Set default metric if not specified
        if metric is None:
            metric = 'AUC' if is_classification else 'MSE'
        
        # Get metric function
        metric_func = get_metric_function(metric, is_classification)
        
        # Ensure X is a DataFrame
        X = self._ensure_dataframe(X)
        
        # If perturb_features not specified, use all features
        if perturb_features is None:
            perturb_features = X.columns.tolist()
        
        # If cat_features not specified, assume empty
        if cat_features is None:
            cat_features = []
        
        # Get the perturbation method
        perturbation = self._get_perturb_method(perturb_method)
        
        # Store results for each model
        results = {model_name: {
            'perturb_sizes': perturb_sizes,
            'mean_scores': [],
            'worst_scores': [],
            'all_scores': []
        } for model_name in models.keys()}
        
        # For each perturbation size
        for size in perturb_sizes:
            # For each model
            for model_name, model in models.items():
                size_scores = []
                
                # Run n_iterations perturbations
                for _ in range(n_iterations):
                    # Create copy of data
                    X_copy = X.copy()
                    
                    # Separate numeric and categorical features to perturb
                    num_features = [f for f in perturb_features if f not in cat_features]
                    cat_features_to_perturb = [f for f in perturb_features if f in cat_features]
                    
                    # Perturb numeric features
                    if num_features:
                        X_numeric = X_copy[num_features].values
                        
                        if perturb_method == 'raw':
                            X_perturbed_numeric = perturbation.perturb(X_numeric, size)
                        elif perturb_method == 'quantile':
                            X_perturbed_numeric = perturbation.perturb(X_numeric, size)
                        else:
                            raise ValueError(f"Perturbation method '{perturb_method}' not supported for numeric features")
                            
                        X_copy[num_features] = X_perturbed_numeric
                    
                    # Perturb categorical features
                    if cat_features_to_perturb:
                        X_copy = self.perturbation_methods['categorical'].perturb(X_copy, cat_features_to_perturb, size)
                    
                    # Make prediction with perturbed data
                    if is_classification and metric == 'AUC':
                        y_pred = model.predict_proba(X_copy)[:, 1]
                    else:
                        y_pred = model.predict(X_copy)
                    
                    # Calculate metric for this perturbation
                    if metric == 'F1' and is_classification:
                        score = metric_func(y, y_pred, average='weighted')
                    else:
                        score = metric_func(y, y_pred)
                    
                    size_scores.append(score)
                
                # Organize results
                all_scores = np.array(size_scores)
                mean_score = np.mean(all_scores)
                
                # Calculate mean of worst samples (according to alpha)
                worst_count = max(1, int(len(all_scores) * alpha))
                if metric in ['MSE', 'MAE']:  # Metrics where lower is better
                    worst_indices = np.argsort(all_scores)[-worst_count:]
                else:  # Metrics where higher is better
                    worst_indices = np.argsort(all_scores)[:worst_count]
                
                worst_score = np.mean(all_scores[worst_indices])
                
                # Store results
                results[model_name]['all_scores'].append(all_scores)
                results[model_name]['mean_scores'].append(mean_score)
                results[model_name]['worst_scores'].append(worst_score)
        
        return results
    
    def analyze_feature_importance(
        self,
        model: Any,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        perturb_method: str = 'raw',
        perturb_size: float = 0.5,
        metric: Optional[str] = None,
        n_iterations: int = 10,
        random_state: Optional[int] = None
    ) -> Dict:
        """
        Analyze the impact of perturbing each feature individually on model robustness.
        
        Args:
            model: Model to evaluate
            X: Input features
            y: Target values
            perturb_method: Perturbation method ('raw', 'quantile', or 'categorical')
            perturb_size: Perturbation size
            metric: Evaluation metric name
            n_iterations: Number of iterations for each test
            random_state: Random seed for reproducibility
            
        Returns:
            Dict: Feature importance analysis results
        """
        # Set random seed if provided
        if random_state is not None:
            np.random.seed(random_state)
        
        # Ensure X is a DataFrame
        X = self._ensure_dataframe(X)
        
        # Verify if it's classification or regression
        is_classification = self._is_classification(y)
        
        # Set default metric if not specified
        if metric is None:
            metric = 'AUC' if is_classification else 'MSE'
        
        # Get metric function
        metric_func = get_metric_function(metric, is_classification)
        
        # Calculate base performance without perturbation
        if is_classification and metric == 'AUC':
            y_pred_base = model.predict_proba(X)[:, 1]
        else:
            y_pred_base = model.predict(X)
        
        if metric == 'F1' and is_classification:
            base_score = metric_func(y, y_pred_base, average='weighted')
        else:
            base_score = metric_func(y, y_pred_base)
        
        # Store the impact of perturbing each feature
        feature_impacts = []
        feature_names = []
        
        # For each feature, evaluate the impact of perturbing only it
        for feature in X.columns:
            feature_names.append(feature)
            
            # Evaluate robustness perturbing only this feature
            feature_results = self.evaluate_robustness(
                models={'model': model},
                X=X,
                y=y,
                perturb_method=perturb_method,
                perturb_sizes=[perturb_size],
                perturb_features=[feature],
                metric=metric,
                n_iterations=n_iterations,
                random_state=random_state
            )
            
            # Calculate difference between base performance and performance with perturbation
            if metric in ['MSE', 'MAE']:  # Metrics where lower is better
                impact = feature_results['model']['mean_scores'][0] - base_score
            else:  # Metrics where higher is better
                impact = base_score - feature_results['model']['mean_scores'][0]
            
            feature_impacts.append(impact)
        
        # Normalize impacts for easier comparison
        max_impact = max(abs(np.array(feature_impacts)))
        if max_impact > 0:
            normalized_impacts = np.array(feature_impacts) / max_impact
        else:
            normalized_impacts = np.array(feature_impacts)
        
        # Sort features by impact
        sorted_indices = np.argsort(np.abs(normalized_impacts))[::-1]
        sorted_features = [feature_names[i] for i in sorted_indices]
        sorted_impacts = [normalized_impacts[i] for i in sorted_indices]
        
        return {
            'base_score': base_score,
            'feature_names': feature_names,
            'feature_impacts': feature_impacts,
            'normalized_impacts': normalized_impacts.tolist(),
            'sorted_features': sorted_features,
            'sorted_impacts': sorted_impacts,
            'metric': metric
        }
    
    def analyze_feature_importance_multiple(
        self,
        models: Dict[str, Any],
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        perturb_method: str = 'raw',
        perturb_size: float = 0.5,
        metric: Optional[str] = None,
        n_iterations: int = 10,
        random_state: Optional[int] = None
    ) -> Dict[str, Dict]:
        """
        Analyze the impact of perturbing each feature individually on robustness for multiple models.
        
        Args:
            models: Dictionary of models to evaluate
            X: Input features
            y: Target values
            perturb_method: Perturbation method ('raw', 'quantile', or 'categorical')
            perturb_size: Perturbation size
            metric: Evaluation metric name
            n_iterations: Number of iterations for each test
            random_state: Random seed for reproducibility
            
        Returns:
            Dict: Feature importance analysis results for each model
        """
        # Results dictionary to store analysis for each model
        all_results = {}
        
        # Set random seed if provided
        if random_state is not None:
            np.random.seed(random_state)
        
        # Ensure X is a DataFrame
        X = self._ensure_dataframe(X)
        
        # Process each model
        for model_name, model in models.items():
            print(f"Analyzing feature importance for {model_name}...")
            
            # Use the existing method for a single model
            model_results = self.analyze_feature_importance(
                model=model,
                X=X,
                y=y,
                perturb_method=perturb_method,
                perturb_size=perturb_size,
                metric=metric,
                n_iterations=n_iterations,
                random_state=random_state
            )
            
            # Store results for this model
            all_results[model_name] = model_results
        
        return all_results
    
    def compare_perturbation_methods(
        self,
        model: Any,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        perturb_methods: List[str] = ['raw', 'quantile'],
        perturb_sizes: List[float] = [0.1, 0.3, 0.5, 0.7, 1.0],
        metric: Optional[str] = None,
        n_iterations: int = 10,
        random_state: Optional[int] = None
    ) -> Dict:
        """
        Compare different perturbation methods for the same model.
        
        Args:
            model: Model to evaluate
            X: Input features
            y: Target values
            perturb_methods: List of perturbation methods to compare
            perturb_sizes: List of perturbation sizes to test
            metric: Evaluation metric name
            n_iterations: Number of iterations per perturbation size
            random_state: Random seed for reproducibility
            
        Returns:
            Dict: Perturbation method comparison results
        """
        # Set random seed if provided
        if random_state is not None:
            np.random.seed(random_state)
            
        # Validate perturbation methods
        for method in perturb_methods:
            if method not in self.perturbation_methods:
                raise ValueError(f"Perturbation method '{method}' not supported. Available methods: {list(self.perturbation_methods.keys())}")
        
        # Evaluate robustness for each method
        all_results = {}
        for method in perturb_methods:
            method_results = self.evaluate_robustness(
                {'model': model},
                X, y,
                perturb_method=method,
                perturb_sizes=perturb_sizes,
                metric=metric,
                n_iterations=n_iterations,
                random_state=random_state
            )
            all_results[f"Method: {method}"] = method_results['model']
        
        return all_results