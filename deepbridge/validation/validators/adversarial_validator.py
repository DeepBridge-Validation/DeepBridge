
# adversarial_validator.py - placeholder file
"""
Adversarial validator for machine learning models.

This module provides a validator for testing model robustness
against adversarial examples.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
import matplotlib.pyplot as plt
import os
import warnings

from ..core.base_validator import BaseValidator
from ..utils.model_inspection import is_classifier, get_model_type, get_model_library
from ..utils.validation_utils import save_validation_results


class AdversarialValidator(BaseValidator):
    """
    Validator for testing model robustness against adversarial examples.
    
    This class provides methods for evaluating model performance
    against various types of adversarial attacks.
    """
    
    def __init__(
        self,
        model: Any,
        attack_types: Optional[List[str]] = None,
        epsilons: Optional[List[float]] = None,
        metrics: Optional[Dict[str, Callable]] = None,
        random_state: Optional[int] = None,
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize the adversarial validator.
        
        Parameters:
        -----------
        model : Any
            Machine learning model to validate
        attack_types : list of str or None
            Types of adversarial attacks to test
        epsilons : list of float or None
            Attack strength parameters to test
        metrics : dict or None
            Custom evaluation metrics
        random_state : int or None
            Random seed for reproducibility
        verbose : bool
            Whether to print progress information
        **kwargs : dict
            Additional parameters
        """
        super().__init__(model=model, **kwargs)
        
        # Default attack types if not provided
        self.attack_types = attack_types or ['fgsm', 'pgd', 'boundary']
        
        # Default epsilon values if not provided
        self.epsilons = epsilons or [0.01, 0.05, 0.1, 0.2]
        
        self.random_state = random_state
        self.verbose = verbose
        
        # Check if model is a classifier
        if not is_classifier(model):
            raise ValueError("Adversarial validation is only applicable to classification models")
            
        # Determine model type
        self.model_type = kwargs.get('model_type', get_model_type(model))
        
        # Determine model framework
        self.model_library = get_model_library(model)
        
        # Set up metrics based on model type
        if metrics is None:
            from ..utils.validation_utils import get_default_metrics
            self.metrics = get_default_metrics('classification')
        else:
            self.metrics = metrics
            
        # Initialize results
        self.results = {}
    
    def validate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Validate model robustness against adversarial examples.
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Features to validate with
        y : array-like or Series
            Target values
        **kwargs : dict
            Additional parameters
            
        Returns:
        --------
        dict : Validation results
        """
        # Override parameters if provided
        attack_types = kwargs.get('attack_types', self.attack_types)
        epsilons = kwargs.get('epsilons', self.epsilons)
        
        # Start with clean results
        results = {
            'model_type': self.model_type,
            'model_library': self.model_library,
            'attack_types': attack_types,
            'epsilons': epsilons,
            'baseline': {},
            'attacks': {},
            'adversarial_robustness_score': 0.0
        }
        
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X
            
        if isinstance(y, pd.Series):
            y_values = y.values
        else:
            y_values = y
            
        # Get baseline performance
        baseline_performance = self._evaluate_baseline(X_values, y_values)
        results['baseline'] = baseline_performance
        
        # Check if necessary libraries are available
        required_libraries = self._check_required_libraries()
        
        if not required_libraries['any_available']:
            warnings.warn("No required libraries available for adversarial attacks. "
                        "Install TensorFlow, PyTorch, or ART for full functionality.")
            
            # Skip adversarial validation
            results['error'] = "Required libraries not available"
            self.results = results
            return results
            
        # Filter attack types based on available libraries
        available_attacks = []
        for attack in attack_types:
            if attack == 'fgsm' and (
                required_libraries['tensorflow'] or
                required_libraries['pytorch'] or
                required_libraries['art']):
                available_attacks.append(attack)
            elif attack == 'pgd' and (
                required_libraries['tensorflow'] or
                required_libraries['pytorch'] or
                required_libraries['art']):
                available_attacks.append(attack)
            elif attack == 'boundary' and required_libraries['art']:
                available_attacks.append(attack)
                
        if not available_attacks:
            warnings.warn("No attacks are available with the current libraries.")
            
            # Skip adversarial validation
            results['error'] = "No attacks available with installed libraries"
            self.results = results
            return results
            
        # Run adversarial attacks
        all_attack_results = {}
        for attack in available_attacks:
            if self.verbose:
                print(f"Testing adversarial attack: {attack}")
                
            attack_results = self._run_attack(
                X_values, y_values, attack, epsilons
            )
            
            all_attack_results[attack] = attack_results
            
        results['attacks'] = all_attack_results
        
        # Calculate adversarial robustness score
        robustness_score = self._calculate_robustness_score(all_attack_results)
        results['adversarial_robustness_score'] = robustness_score
        
        # Store results
        self.results = results
        
        return results
    
    def _check_required_libraries(self) -> Dict[str, bool]:
        """
        Check if required libraries are available.
        
        Returns:
        --------
        dict : Availability of libraries
        """
        # Check for TensorFlow
        try:
            import tensorflow as tf
            tensorflow_available = True
        except ImportError:
            tensorflow_available = False
            
        # Check for PyTorch
        try:
            import torch
            pytorch_available = True
        except ImportError:
            pytorch_available = False
            
        # Check for ART (Adversarial Robustness Toolbox)
        try:
            import art
            art_available = True
        except ImportError:
            art_available = False
            
        return {
            'tensorflow': tensorflow_available,
            'pytorch': pytorch_available,
            'art': art_available,
            'any_available': tensorflow_available or pytorch_available or art_available
        }
    
    def _evaluate_baseline(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate baseline model performance.
        
        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            Target values
            
        Returns:
        --------
        dict : Baseline performance metrics
        """
        try:
            # Make predictions
            y_pred = self.model.predict(X)
            
            # For classifiers with predict_proba
            has_proba = hasattr(self.model, 'predict_proba')
            if has_proba:
                y_proba = self.model.predict_proba(X)
            else:
                y_proba = None
                
            # Calculate metrics
            metrics_results = {}
            for metric_name, metric_func in self.metrics.items():
                try:
                    if metric_name == 'roc_auc' and y_proba is not None:
                        # ROC AUC requires probabilities
                        if y_proba.shape[1] == 2:
                            # Binary classification
                            score = metric_func(y, y_proba[:, 1])
                        else:
                            # Multiclass classification
                            score = metric_func(y, y_proba)
                    else:
                        # Standard metric
                        score = metric_func(y, y_pred)
                        
                    metrics_results[metric_name] = float(score)
                except Exception as e:
                    if self.verbose:
                        print(f"Error calculating metric {metric_name}: {str(e)}")
                    metrics_results[metric_name] = float('nan')
                    
            return metrics_results
            
        except Exception as e:
            if self.verbose:
                print(f"Error evaluating baseline: {str(e)}")
            return {metric: float('nan') for metric in self.metrics}
    
    def _run_attack(
        self,
        X: np.ndarray,
        y: np.ndarray,
        attack_type: str,
        epsilons: List[float]
    ) -> Dict[str, Any]:
        """
        Run a specific adversarial attack.
        
        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            Target values
        attack_type : str
            Type of attack to run
        epsilons : list of float
            Attack strength parameters
            
        Returns:
        --------
        dict : Attack results
        """
        # Initialize results
        results = {
            'epsilon_results': {},
            'success_rate': {},
            'average_perturbation': {},
            'attack_library': 'none'
        }
        
        # Determine which library to use
        # Try to use native implementations first, then fall back to ART
        if self.model_library == 'tensorflow' and self._check_required_libraries()['tensorflow']:
            library = 'tensorflow'
        elif self.model_library == 'pytorch' and self._check_required_libraries()['pytorch']:
            library = 'pytorch'
        elif self._check_required_libraries()['art']:
            library = 'art'
        else:
            # No suitable library found
            results['error'] = "No suitable library found for attack"
            return results
            
        # Store attack library
        results['attack_library'] = library
        
        # Run attack with each epsilon
        for epsilon in epsilons:
            # Generate adversarial examples
            try:
                X_adv, attack_success, perturbation_size = self._generate_adversarial_examples(
                    X, y, attack_type, epsilon, library
                )
                
                # Evaluate model on adversarial examples
                adv_performance = self._evaluate_baseline(X_adv, y)
                
                # Calculate relative change from baseline
                relative_change = {}
                for metric, value in adv_performance.items():
                    if metric in self.results.get('baseline', {}):
                        baseline_value = self.results['baseline'][metric]
                        if baseline_value != 0 and not np.isnan(baseline_value):
                            relative_change[metric] = (value - baseline_value) / abs(baseline_value)
                        else:
                            relative_change[metric] = 0.0
                    else:
                        relative_change[metric] = 0.0
                        
                # Store results for this epsilon
                results['epsilon_results'][str(epsilon)] = {
                    'performance': adv_performance,
                    'relative_change': relative_change,
                    'success_rate': float(attack_success),
                    'perturbation_size': float(perturbation_size)
                }
                
                # Store success rate and perturbation size
                results['success_rate'][str(epsilon)] = float(attack_success)
                results['average_perturbation'][str(epsilon)] = float(perturbation_size)
                
            except Exception as e:
                if self.verbose:
                    print(f"Error running {attack_type} attack with epsilon={epsilon}: {str(e)}")
                    
                # Store error
                results['epsilon_results'][str(epsilon)] = {
                    'error': str(e)
                }
        
        return results
    
    def _generate_adversarial_examples(
        self,
        X: np.ndarray,
        y: np.ndarray,
        attack_type: str,
        epsilon: float,
        library: str
    ) -> Tuple[np.ndarray, float, float]:
        """
        Generate adversarial examples.
        
        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            Target values
        attack_type : str
            Type of attack to run
        epsilon : float
            Attack strength parameter
        library : str
            Library to use for attack
            
        Returns:
        --------
        tuple : (adversarial_examples, success_rate, perturbation_size)
        """
        # Implementation depends on the library and attack type
        if library == 'tensorflow':
            return self._generate_adversarial_tf(X, y, attack_type, epsilon)
        elif library == 'pytorch':
            return self._generate_adversarial_torch(X, y, attack_type, epsilon)
        elif library == 'art':
            return self._generate_adversarial_art(X, y, attack_type, epsilon)
        else:
            raise ValueError(f"Unsupported library: {library}")
    
    def _generate_adversarial_tf(
        self,
        X: np.ndarray,
        y: np.ndarray,
        attack_type: str,
        epsilon: float
    ) -> Tuple[np.ndarray, float, float]:
        """
        Generate adversarial examples using TensorFlow.
        
        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            Target values
        attack_type : str
            Type of attack to run
        epsilon : float
            Attack strength parameter
            
        Returns:
        --------
        tuple : (adversarial_examples, success_rate, perturbation_size)
        """
        # Import TensorFlow
        import tensorflow as tf
        
        # Get model predictions
        y_pred = self.model.predict(X)
        
        # Convert to one-hot encoding if needed
        if len(y_pred.shape) == 1 or y_pred.shape[1] == 1:
            from tensorflow.keras.utils import to_categorical
            y_pred_one_hot = to_categorical(y_pred)
        else:
            y_pred_one_hot = y_pred
            
        # Convert inputs to TensorFlow tensors
        x_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
        y_tensor = tf.convert_to_tensor(y, dtype=tf.int32)
        
        # Implement attacks
        if attack_type == 'fgsm':
            # Fast Gradient Sign Method
            with tf.GradientTape() as tape:
                tape.watch(x_tensor)
                
                # Get model predictions
                predictions = self.model(x_tensor)
                
                # Calculate loss
                loss = tf.keras.losses.sparse_categorical_crossentropy(y_tensor, predictions)
                
            # Get gradients of loss w.r.t input
            gradients = tape.gradient(loss, x_tensor)
            
            # Create adversarial examples
            x_adv = x_tensor + epsilon * tf.sign(gradients)
            
            # Clip to valid range if specified
            if hasattr(self, 'clip_min') and hasattr(self, 'clip_max'):
                x_adv = tf.clip_by_value(x_adv, self.clip_min, self.clip_max)
                
        elif attack_type == 'pgd':
            # Projected Gradient Descent
            x_adv = x_tensor
            
            # PGD parameters
            alpha = epsilon / 10  # Step size
            num_steps = 10
            
            for i in range(num_steps):
                with tf.GradientTape() as tape:
                    tape.watch(x_adv)
                    
                    # Get model predictions
                    predictions = self.model(x_adv)
                    
                    # Calculate loss
                    loss = tf.keras.losses.sparse_categorical_crossentropy(y_tensor, predictions)
                    
                # Get gradients of loss w.r.t input
                gradients = tape.gradient(loss, x_adv)
                
                # Update adversarial examples
                x_adv = x_adv + alpha * tf.sign(gradients)
                
                # Project back to epsilon ball
                delta = x_adv - x_tensor
                delta = tf.clip_by_value(delta, -epsilon, epsilon)
                x_adv = x_tensor + delta
                
                # Clip to valid range if specified
                if hasattr(self, 'clip_min') and hasattr(self, 'clip_max'):
                    x_adv = tf.clip_by_value(x_adv, self.clip_min, self.clip_max)
        else:
            raise ValueError(f"Unsupported attack type for TensorFlow: {attack_type}")
            
        # Convert back to numpy
        X_adv = x_adv.numpy()
        
        # Calculate success rate and perturbation size
        y_adv_pred = self.model.predict(X_adv)
        success_rate = np.mean(np.argmax(y_adv_pred, axis=1) != y if len(y_adv_pred.shape) > 1 else y_adv_pred != y)
        perturbation_size = np.mean(np.linalg.norm(X_adv - X, axis=1))
        
        return X_adv, success_rate, perturbation_size
    
    def _generate_adversarial_torch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        attack_type: str,
        epsilon: float
    ) -> Tuple[np.ndarray, float, float]:
        """
        Generate adversarial examples using PyTorch.
        
        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            Target values
        attack_type : str
            Type of attack to run
        epsilon : float
            Attack strength parameter
            
        Returns:
        --------
        tuple : (adversarial_examples, success_rate, perturbation_size)
        """
        # Import PyTorch
        import torch
        
        # Convert inputs to PyTorch tensors
        device = next(self.model.parameters()).device
        x_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(y, dtype=torch.long, device=device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Implement attacks
        if attack_type == 'fgsm':
            # Fast Gradient Sign Method
            x_adv = x_tensor.clone().detach().requires_grad_(True)
            
            # Get model predictions
            outputs = self.model(x_adv)
            
            # Calculate loss
            loss = torch.nn.functional.cross_entropy(outputs, y_tensor)
            
            # Calculate gradients
            loss.backward()
            gradients = x_adv.grad.data
            
            # Create adversarial examples
            x_adv = x_adv + epsilon * torch.sign(gradients)
            
            # Clip to valid range if specified
            if hasattr(self, 'clip_min') and hasattr(self, 'clip_max'):
                x_adv = torch.clamp(x_adv, self.clip_min, self.clip_max)
                
        elif attack_type == 'pgd':
            # Projected Gradient Descent
            x_adv = x_tensor.clone().detach()
            
            # PGD parameters
            alpha = epsilon / 10  # Step size
            num_steps = 10
            
            for i in range(num_steps):
                x_adv = x_adv.clone().detach().requires_grad_(True)
                
                # Get model predictions
                outputs = self.model(x_adv)
                
                # Calculate loss
                loss = torch.nn.functional.cross_entropy(outputs, y_tensor)
                
                # Calculate gradients
                loss.backward()
                gradients = x_adv.grad.data
                
                # Update adversarial examples
                x_adv = x_adv + alpha * torch.sign(gradients)
                
                # Project back to epsilon ball
                delta = x_adv - x_tensor
                delta = torch.clamp(delta, -epsilon, epsilon)
                x_adv = x_tensor + delta
                
                # Clip to valid range if specified
                if hasattr(self, 'clip_min') and hasattr(self, 'clip_max'):
                    x_adv = torch.clamp(x_adv, self.clip_min, self.clip_max)
        else:
            raise ValueError(f"Unsupported attack type for PyTorch: {attack_type}")
            
        # Convert back to numpy
        X_adv = x_adv.detach().cpu().numpy()
        
        # Calculate success rate and perturbation size
        y_adv_pred = self.model(torch.tensor(X_adv, dtype=torch.float32, device=device))
        y_adv_pred = y_adv_pred.detach().cpu().numpy()
        
        success_rate = np.mean(np.argmax(y_adv_pred, axis=1) != y if len(y_adv_pred.shape) > 1 else y_adv_pred != y)
        perturbation_size = np.mean(np.linalg.norm(X_adv - X, axis=1))
        
        return X_adv, success_rate, perturbation_size
    
    def _generate_adversarial_art(
        self,
        X: np.ndarray,
        y: np.ndarray,
        attack_type: str,
        epsilon: float
    ) -> Tuple[np.ndarray, float, float]:
        """
        Generate adversarial examples using ART.
        
        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            Target values
        attack_type : str
            Type of attack to run
        epsilon : float
            Attack strength parameter
            
        Returns:
        --------
        tuple : (adversarial_examples, success_rate, perturbation_size)
        """
        try:
            from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, BoundaryAttack
            from art.estimators.classification import SklearnClassifier, TensorFlowV2Classifier, PyTorchClassifier
        except ImportError:
            raise ImportError("Adversarial Robustness Toolbox (ART) is required for this attack.")
            
        # Wrap model in ART classifier
        if self.model_library == 'tensorflow':
            import tensorflow as tf
            
            # Define prediction function
            def predict_fn(x):
                return self.model(x).numpy()
                
            # Create classifier
            classifier = TensorFlowV2Classifier(
                model=self.model,
                loss_object=tf.keras.losses.SparseCategoricalCrossentropy(),
                input_shape=X.shape[1:],
                nb_classes=len(np.unique(y)),
                clip_values=(0, 1) if hasattr(self, 'clip_min') and hasattr(self, 'clip_max') else None
            )
            
        elif self.model_library == 'pytorch':
            import torch
            
            # Define prediction function
            def predict_fn(x):
                self.model.eval()
                with torch.no_grad():
                    return self.model(torch.tensor(x, dtype=torch.float32)).numpy()
                    
            # Create classifier
            classifier = PyTorchClassifier(
                model=self.model,
                loss=torch.nn.CrossEntropyLoss(),
                input_shape=X.shape[1:],
                nb_classes=len(np.unique(y)),
                clip_values=(0, 1) if hasattr(self, 'clip_min') and hasattr(self, 'clip_max') else None
            )
            
        else:
            # Try to use sklearn wrapper
            try:
                classifier = SklearnClassifier(self.model)
            except:
                raise ValueError(f"Unsupported model type for ART: {self.model_library}")
                
        # Create attack
        if attack_type == 'fgsm':
            attack = FastGradientMethod(
                estimator=classifier,
                eps=epsilon,
                targeted=False
            )
        elif attack_type == 'pgd':
            attack = ProjectedGradientDescent(
                estimator=classifier,
                eps=epsilon,
                eps_step=epsilon / 10,
                max_iter=10,
                targeted=False
            )
        elif attack_type == 'boundary':
            attack = BoundaryAttack(
                estimator=classifier,
                targeted=False
            )
        else:
            raise ValueError(f"Unsupported attack type for ART: {attack_type}")
            
        # Generate adversarial examples
        X_adv = attack.generate(X)
        
        # Calculate success rate and perturbation size
        y_adv_pred = classifier.predict(X_adv)
        success_rate = np.mean(np.argmax(y_adv_pred, axis=1) != y if len(y_adv_pred.shape) > 1 else y_adv_pred != y)
        perturbation_size = np.mean(np.linalg.norm(X_adv - X, axis=1))
        
        return X_adv, success_rate, perturbation_size
    
    def _calculate_robustness_score(
        self,
        attack_results: Dict[str, Dict[str, Any]]
    ) -> float:
        """
        Calculate adversarial robustness score.
        
        Parameters:
        -----------
        attack_results : dict
            Results from adversarial attacks
            
        Returns:
        --------
        float : Robustness score [0, 1]
        """
        # Collect success rates across all attacks and epsilons
        success_rates = []
        
        for attack, results in attack_results.items():
            if 'success_rate' in results:
                for epsilon, rate in results['success_rate'].items():
                    success_rates.append(rate)
                    
        # Compute robustness score (1 - average success rate)
        if not success_rates:
            return 0.5  # Default score if no data
            
        avg_success_rate = float(np.mean(success_rates))
        
        # Robust models have low attack success rates
        return max(0, min(1, 1 - avg_success_rate))
    
    def plot_attack_success(
        self,
        attack_type: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot attack success rate by epsilon.
        
        Parameters:
        -----------
        attack_type : str or None
            Type of attack to plot (None for all)
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.Figure : Attack success plot
        """
        if not self.results or 'attacks' not in self.results:
            raise ValueError("No attack data available. Run validate() first.")
            
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get attacks to plot
        if attack_type is not None:
            if attack_type not in self.results['attacks']:
                raise ValueError(f"Attack type '{attack_type}' not found in results")
                
            attacks_to_plot = {attack_type: self.results['attacks'][attack_type]}
        else:
            attacks_to_plot = self.results['attacks']
            
        # Plot success rate for each attack
        for attack, results in attacks_to_plot.items():
            if 'success_rate' not in results:
                continue
                
            # Get epsilon values and success rates
            epsilons = sorted([float(eps) for eps in results['success_rate'].keys()])
            rates = [results['success_rate'][str(eps)] for eps in epsilons]
            
            # Plot line
            ax.plot(epsilons, rates, 'o-', label=f'{attack.upper()} Attack')
            
        # Set labels and title
        ax.set_xlabel('Epsilon (Attack Strength)')
        ax.set_ylabel('Success Rate')
        ax.set_title('Adversarial Attack Success Rate')
        
        # Add reference line at 0
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Add legend
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_perturbation_size(
        self,
        attack_type: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot perturbation size by epsilon.
        
        Parameters:
        -----------
        attack_type : str or None
            Type of attack to plot (None for all)
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.Figure : Perturbation size plot
        """
        if not self.results or 'attacks' not in self.results:
            raise ValueError("No attack data available. Run validate() first.")
            
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get attacks to plot
        if attack_type is not None:
            if attack_type not in self.results['attacks']:
                raise ValueError(f"Attack type '{attack_type}' not found in results")
                
            attacks_to_plot = {attack_type: self.results['attacks'][attack_type]}
        else:
            attacks_to_plot = self.results['attacks']
            
        # Plot perturbation size for each attack
        for attack, results in attacks_to_plot.items():
            if 'average_perturbation' not in results:
                continue
                
            # Get epsilon values and perturbation sizes
            epsilons = sorted([float(eps) for eps in results['average_perturbation'].keys()])
            sizes = [results['average_perturbation'][str(eps)] for eps in epsilons]
            
            # Plot line
            ax.plot(epsilons, sizes, 'o-', label=f'{attack.upper()} Attack')
            
        # Set labels and title
        ax.set_xlabel('Epsilon (Attack Strength)')
        ax.set_ylabel('Average Perturbation Size')
        ax.set_title('Adversarial Perturbation Size')
        
        # Add legend
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def save_results(
        self,
        output_dir: str,
        prefix: str = 'adversarial',
        include_plots: bool = True
    ) -> Dict[str, str]:
        """
        Save validation results and plots.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save results in
        prefix : str
            Prefix for output files
        include_plots : bool
            Whether to save plots
            
        Returns:
        --------
        dict : Paths to saved files
        """
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Paths for saved files
        saved_files = {}
        
        # Save results as JSON
        results_path = os.path.join(output_dir, f"{prefix}_results.json")
        save_validation_results(self.results, results_path, 'json')
        saved_files['results_json'] = results_path
        
        # Save results as CSV
        csv_path = os.path.join(output_dir, f"{prefix}_results.csv")
        save_validation_results(self.results, csv_path, 'dataframe')
        saved_files['results_csv'] = csv_path
        
        # Save plots if requested
        if include_plots and self.results and 'attacks' in self.results:
            # Save attack success plot
            try:
                fig = self.plot_attack_success()
                fig_path = os.path.join(output_dir, f"{prefix}_attack_success.png")
                fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                saved_files['attack_success_plot'] = fig_path
                plt.close(fig)
            except Exception as e:
                if self.verbose:
                    print(f"Error saving attack success plot: {str(e)}")
                    
            # Save perturbation size plot
            try:
                fig = self.plot_perturbation_size()
                fig_path = os.path.join(output_dir, f"{prefix}_perturbation_size.png")
                fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                saved_files['perturbation_size_plot'] = fig_path
                plt.close(fig)
            except Exception as e:
                if self.verbose:
                    print(f"Error saving perturbation size plot: {str(e)}")
                    
        return saved_files