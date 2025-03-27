"""
Adversarial robustness tests for model validation.

This module provides a wrapper for testing model robustness
against adversarial attacks using DBDataset objects.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
import matplotlib.pyplot as plt
import warnings

from .base_wrapper import BaseWrapper
from deepbridge.validation.frameworks.robustness.adversarial.fgsm_attack import generate_fgsm_attack, FGSMAttacker
from deepbridge.validation.frameworks.robustness.adversarial.pgd_attack import generate_pgd_attack, PGDAttacker
from deepbridge.validation.frameworks.robustness.adversarial.blackbox_attack import generate_blackbox_attack, BlackboxAttacker


class AdversarialRobustnessTests(BaseWrapper):
    """
    Tests for model robustness against adversarial attacks.
    
    This class provides methods for testing how well a model maintains
    its performance when faced with adversarial examples designed to
    fool the model.
    """
    
    def __init__(self, dataset, verbose: bool = False):
        """
        Initialize the adversarial robustness tests.
        
        Parameters:
        -----------
        dataset : DBDataset
            Dataset object containing training/test data and model
        verbose : bool
            Whether to print progress information
        """
        super().__init__(dataset, verbose)
        
        # Check if model is compatible with adversarial attacks
        self._check_compatibility()
        
        # Store baseline performance for reference
        self.baseline_performance = self._evaluate_baseline_performance()
        
        if self.verbose:
            print(f"Baseline performance: {self.baseline_performance}")
    
    def _check_compatibility(self):
        """
        Check if model is compatible with adversarial attacks.
        
        For some attacks like FGSM and PGD, the model needs to be
        a differential model (e.g., using PyTorch or TensorFlow).
        """
        self.supports_gradients = False
        
        try:
            import tensorflow as tf
            if isinstance(self.model, tf.keras.Model):
                self.supports_gradients = True
                self.framework = 'tensorflow'
        except ImportError:
            pass
        
        try:
            import torch
            if isinstance(self.model, torch.nn.Module):
                self.supports_gradients = True
                self.framework = 'pytorch'
        except ImportError:
            pass
        
        if not self.supports_gradients and self.verbose:
            print("Model doesn't support gradients. Only blackbox attacks will be available.")
    
    def _evaluate_baseline_performance(self) -> Dict[str, float]:
        """
        Evaluate baseline model performance.
        
        Returns:
        --------
        dict : Baseline performance metrics
        """
        X = self._get_feature_data()
        y = self._get_target_data()
        
        if self.is_classification():
            from sklearn.metrics import accuracy_score, f1_score
            
            # Make predictions
            y_pred = self.model.predict(X)
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'f1': f1_score(y, y_pred, average='weighted')
            }
            
            # Add AUC if model supports probability predictions
            if hasattr(self.model, 'predict_proba'):
                from sklearn.metrics import roc_auc_score
                try:
                    y_proba = self.model.predict_proba(X)
                    
                    # Handle binary and multiclass cases
                    if y_proba.shape[1] == 2:  # Binary
                        metrics['roc_auc'] = roc_auc_score(y, y_proba[:, 1])
                    else:  # Multiclass
                        metrics['roc_auc'] = roc_auc_score(y, y_proba, multi_class='ovr')
                except Exception as e:
                    if self.verbose:
                        print(f"Could not calculate ROC AUC: {str(e)}")
        else:
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            
            # Make predictions
            y_pred = self.model.predict(X)
            
            # Calculate metrics
            metrics = {
                'mse': mean_squared_error(y, y_pred),
                'mae': mean_absolute_error(y, y_pred)
            }
        
        return metrics
    
    def _evaluate_adversarial_performance(self, X_adv: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance on adversarial examples.
        
        Parameters:
        -----------
        X_adv : array-like
            Adversarial examples
            
        Returns:
        --------
        dict : Performance metrics on adversarial examples
        """
        y = self._get_target_data()
        
        if self.is_classification():
            from sklearn.metrics import accuracy_score, f1_score
            
            # Make predictions
            y_pred = self.model.predict(X_adv)
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'f1': f1_score(y, y_pred, average='weighted')
            }
            
            # Add AUC if model supports probability predictions
            if hasattr(self.model, 'predict_proba'):
                from sklearn.metrics import roc_auc_score
                try:
                    y_proba = self.model.predict_proba(X_adv)
                    
                    # Handle binary and multiclass cases
                    if y_proba.shape[1] == 2:  # Binary
                        metrics['roc_auc'] = roc_auc_score(y, y_proba[:, 1])
                    else:  # Multiclass
                        metrics['roc_auc'] = roc_auc_score(y, y_proba, multi_class='ovr')
                except Exception as e:
                    if self.verbose:
                        print(f"Could not calculate ROC AUC: {str(e)}")
        else:
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            
            # Make predictions
            y_pred = self.model.predict(X_adv)
            
            # Calculate metrics
            metrics = {
                'mse': mean_squared_error(y, y_pred),
                'mae': mean_absolute_error(y, y_pred)
            }
        
        return metrics
    
    def test_fgsm(self, epsilon: float = 0.1, norm: str = 'inf', targeted: bool = False) -> Dict[str, Any]:
        """
        Test robustness against Fast Gradient Sign Method (FGSM) attack.
        
        Parameters:
        -----------
        epsilon : float
            Attack strength
        norm : str
            Norm to use for the attack ('inf', 'l1', 'l2')
        targeted : bool
            Whether the attack is targeted
            
        Returns:
        --------
        dict : FGSM test results
        """
        if not self.supports_gradients:
            raise ValueError("FGSM attack requires a model that supports gradients (PyTorch or TensorFlow)")
        
        if self.verbose:
            print(f"Testing FGSM attack (epsilon={epsilon}, norm={norm}, targeted={targeted})")
        
        X_test = self._get_feature_data()
        y_test = self._get_target_data()
        
        # Generate adversarial examples
        try:
            X_adv = generate_fgsm_attack(
                model=self.model,
                X=X_test,
                y=y_test,
                framework=self.framework,
                epsilon=epsilon,
                norm=norm,
                targeted=targeted
            )
            
            # Calculate perturbation magnitude
            if isinstance(X_test, pd.DataFrame):
                X_test_arr = X_test.values
            else:
                X_test_arr = X_test
                
            if norm == 'inf':
                perturbation_magnitude = np.max(np.abs(X_adv - X_test_arr))
            elif norm == 'l2':
                perturbation_magnitude = np.mean(np.sqrt(np.sum((X_adv - X_test_arr)**2, axis=1)))
            elif norm == 'l1':
                perturbation_magnitude = np.mean(np.sum(np.abs(X_adv - X_test_arr), axis=1))
            else:
                perturbation_magnitude = 0.0
            
            # Evaluate performance on adversarial examples
            adv_performance = self._evaluate_adversarial_performance(X_adv)
            
            # Calculate relative change in performance
            rel_change = {}
            for metric, value in adv_performance.items():
                if metric in self.baseline_performance:
                    baseline = self.baseline_performance[metric]
                    if baseline != 0:
                        rel_change[metric] = (value - baseline) / abs(baseline)
                    else:
                        rel_change[metric] = 0.0
            
            results = {
                'attack_type': 'fgsm',
                'epsilon': epsilon,
                'norm': norm,
                'targeted': targeted,
                'perturbation_magnitude': float(perturbation_magnitude),
                'performance': adv_performance,
                'baseline_performance': self.baseline_performance,
                'relative_change': rel_change,
                'success_rate': 1.0 - adv_performance.get('accuracy', 0.0) if self.is_classification() else 0.0
            }
            
            return results
            
        except Exception as e:
            if self.verbose:
                print(f"FGSM attack failed: {str(e)}")
                
            # Return failure results
            return {
                'attack_type': 'fgsm',
                'epsilon': epsilon,
                'norm': norm,
                'targeted': targeted,
                'error': str(e),
                'success': False
            }
    
    def test_pgd(self, epsilon: float = 0.1, alpha: float = 0.01, iterations: int = 10, norm: str = 'inf', targeted: bool = False) -> Dict[str, Any]:
        """
        Test robustness against Projected Gradient Descent (PGD) attack.
        
        Parameters:
        -----------
        epsilon : float
            Maximum perturbation
        alpha : float
            Step size
        iterations : int
            Number of iterations
        norm : str
            Norm to use for the attack ('inf', 'l1', 'l2')
        targeted : bool
            Whether the attack is targeted
            
        Returns:
        --------
        dict : PGD test results
        """
        if not self.supports_gradients:
            raise ValueError("PGD attack requires a model that supports gradients (PyTorch or TensorFlow)")
        
        if self.verbose:
            print(f"Testing PGD attack (epsilon={epsilon}, alpha={alpha}, iterations={iterations}, norm={norm}, targeted={targeted})")
        
        X_test = self._get_feature_data()
        y_test = self._get_target_data()
        
        # Generate adversarial examples
        try:
            X_adv = generate_pgd_attack(
                model=self.model,
                X=X_test,
                y=y_test,
                framework=self.framework,
                epsilon=epsilon,
                alpha=alpha,
                iterations=iterations,
                norm=norm,
                targeted=targeted
            )
            
            # Calculate perturbation magnitude
            if isinstance(X_test, pd.DataFrame):
                X_test_arr = X_test.values
            else:
                X_test_arr = X_test
                
            if norm == 'inf':
                perturbation_magnitude = np.max(np.abs(X_adv - X_test_arr))
            elif norm == 'l2':
                perturbation_magnitude = np.mean(np.sqrt(np.sum((X_adv - X_test_arr)**2, axis=1)))
            elif norm == 'l1':
                perturbation_magnitude = np.mean(np.sum(np.abs(X_adv - X_test_arr), axis=1))
            else:
                perturbation_magnitude = 0.0
            
            # Evaluate performance on adversarial examples
            adv_performance = self._evaluate_adversarial_performance(X_adv)
            
            # Calculate relative change in performance
            rel_change = {}
            for metric, value in adv_performance.items():
                if metric in self.baseline_performance:
                    baseline = self.baseline_performance[metric]
                    if baseline != 0:
                        rel_change[metric] = (value - baseline) / abs(baseline)
                    else:
                        rel_change[metric] = 0.0
            
            results = {
                'attack_type': 'pgd',
                'epsilon': epsilon,
                'alpha': alpha,
                'iterations': iterations,
                'norm': norm,
                'targeted': targeted,
                'perturbation_magnitude': float(perturbation_magnitude),
                'performance': adv_performance,
                'baseline_performance': self.baseline_performance,
                'relative_change': rel_change,
                'success_rate': 1.0 - adv_performance.get('accuracy', 0.0) if self.is_classification() else 0.0
            }
            
            return results
            
        except Exception as e:
            if self.verbose:
                print(f"PGD attack failed: {str(e)}")
                
            # Return failure results
            return {
                'attack_type': 'pgd',
                'epsilon': epsilon,
                'alpha': alpha,
                'iterations': iterations,
                'norm': norm,
                'targeted': targeted,
                'error': str(e),
                'success': False
            }
    
    def test_blackbox_random(self, epsilon: float = 0.1, n_trials: int = 100) -> Dict[str, Any]:
        """
        Test robustness against random black-box attack.
        
        Parameters:
        -----------
        epsilon : float
            Maximum perturbation
        n_trials : int
            Number of trials for random search
            
        Returns:
        --------
        dict : Black-box attack test results
        """
        if self.verbose:
            print(f"Testing black-box random attack (epsilon={epsilon}, n_trials={n_trials})")
        
        X_test = self._get_feature_data()
        y_test = self._get_target_data()
        
        # Generate adversarial examples
        try:
            X_adv = generate_blackbox_attack(
                model=self.model,
                X=X_test,
                y=y_test,
                attack_type='random',
                epsilon=epsilon,
                n_trials=n_trials
            )
            
            # Calculate perturbation magnitude
            if isinstance(X_test, pd.DataFrame):
                X_test_arr = X_test.values
            else:
                X_test_arr = X_test
                
            perturbation_magnitude = np.mean(np.sqrt(np.sum((X_adv - X_test_arr)**2, axis=1)))
            
            # Evaluate performance on adversarial examples
            adv_performance = self._evaluate_adversarial_performance(X_adv)
            
            # Calculate relative change in performance
            rel_change = {}
            for metric, value in adv_performance.items():
                if metric in self.baseline_performance:
                    baseline = self.baseline_performance[metric]
                    if baseline != 0:
                        rel_change[metric] = (value - baseline) / abs(baseline)
                    else:
                        rel_change[metric] = 0.0
            
            results = {
                'attack_type': 'blackbox_random',
                'epsilon': epsilon,
                'n_trials': n_trials,
                'perturbation_magnitude': float(perturbation_magnitude),
                'performance': adv_performance,
                'baseline_performance': self.baseline_performance,
                'relative_change': rel_change,
                'success_rate': 1.0 - adv_performance.get('accuracy', 0.0) if self.is_classification() else 0.0
            }
            
            return results
            
        except Exception as e:
            if self.verbose:
                print(f"Black-box random attack failed: {str(e)}")
                
            # Return failure results
            return {
                'attack_type': 'blackbox_random',
                'epsilon': epsilon,
                'n_trials': n_trials,
                'error': str(e),
                'success': False
            }
    
    def test_blackbox_boundary(self, epsilon: float = 0.1, n_trials: int = 100) -> Dict[str, Any]:
        """
        Test robustness against boundary black-box attack.
        
        Parameters:
        -----------
        epsilon : float
            Maximum perturbation
        n_trials : int
            Number of trials for boundary search
            
        Returns:
        --------
        dict : Black-box attack test results
        """
        if self.verbose:
            print(f"Testing black-box boundary attack (epsilon={epsilon}, n_trials={n_trials})")
            print("Note: Boundary attack can be computationally expensive.")
        
        X_test = self._get_feature_data()
        y_test = self._get_target_data()
        
        # Generate adversarial examples
        try:
            X_adv = generate_blackbox_attack(
                model=self.model,
                X=X_test,
                y=y_test,
                attack_type='boundary',
                epsilon=epsilon,
                n_trials=n_trials
            )
            
            # Calculate perturbation magnitude
            if isinstance(X_test, pd.DataFrame):
                X_test_arr = X_test.values
            else:
                X_test_arr = X_test
                
            perturbation_magnitude = np.mean(np.sqrt(np.sum((X_adv - X_test_arr)**2, axis=1)))
            
            # Evaluate performance on adversarial examples
            adv_performance = self._evaluate_adversarial_performance(X_adv)
            
            # Calculate relative change in performance
            rel_change = {}
            for metric, value in adv_performance.items():
                if metric in self.baseline_performance:
                    baseline = self.baseline_performance[metric]
                    if baseline != 0:
                        rel_change[metric] = (value - baseline) / abs(baseline)
                    else:
                        rel_change[metric] = 0.0
            
            results = {
                'attack_type': 'blackbox_boundary',
                'epsilon': epsilon,
                'n_trials': n_trials,
                'perturbation_magnitude': float(perturbation_magnitude),
                'performance': adv_performance,
                'baseline_performance': self.baseline_performance,
                'relative_change': rel_change,
                'success_rate': 1.0 - adv_performance.get('accuracy', 0.0) if self.is_classification() else 0.0
            }
            
            return results
            
        except Exception as e:
            if self.verbose:
                print(f"Black-box boundary attack failed: {str(e)}")
                
            # Return failure results
            return {
                'attack_type': 'blackbox_boundary',
                'epsilon': epsilon,
                'n_trials': n_trials,
                'error': str(e),
                'success': False
            }
    
    def compare_attacks(self, epsilons: List[float] = [0.01, 0.05, 0.1, 0.2], figsize: Tuple[int, int] = (12, 8)) -> Dict[str, Any]:
        """
        Compare different adversarial attacks at various strengths.
        
        Parameters:
        -----------
        epsilons : list of float
            Different attack strengths to test
        figsize : tuple
            Figure size for plots
            
        Returns:
        --------
        dict : Comparison results and plots
        """
        if self.verbose:
            print(f"Comparing different adversarial attacks at epsilon values: {epsilons}")
        
        # Initialize results storage
        attack_results = {
            'epsilons': epsilons,
            'attacks': {},
            'success_rates': {},
            'performance_drop': {}
        }
        
        # Get primary metric based on problem type
        if self.is_classification():
            primary_metric = 'accuracy'
        else:
            primary_metric = 'mse'
        
        # Test different attacks
        attack_types = []
        
        # Always test black-box random
        if self.verbose:
            print("Testing black-box random attack...")
            
        blackbox_random_results = []
        for eps in epsilons:
            result = self.test_blackbox_random(epsilon=eps, n_trials=100)
            blackbox_random_results.append(result)
            
        attack_results['attacks']['blackbox_random'] = blackbox_random_results
        attack_types.append('blackbox_random')
        
        # Test FGSM if gradients are supported
        if self.supports_gradients:
            if self.verbose:
                print("Testing FGSM attack...")
                
            fgsm_results = []
            for eps in epsilons:
                result = self.test_fgsm(epsilon=eps)
                fgsm_results.append(result)
                
            attack_results['attacks']['fgsm'] = fgsm_results
            attack_types.append('fgsm')
            
            if self.verbose:
                print("Testing PGD attack...")
                
            pgd_results = []
            for eps in epsilons:
                result = self.test_pgd(epsilon=eps, alpha=eps/10, iterations=10)
                pgd_results.append(result)
                
            attack_results['attacks']['pgd'] = pgd_results
            attack_types.append('pgd')
        
        # Extract success rates and performance drop for comparison
        for attack_type in attack_types:
            success_rates = []
            performance_values = []
            
            for result in attack_results['attacks'][attack_type]:
                # Get success rate
                success_rate = result.get('success_rate', 0.0)
                success_rates.append(success_rate)
                
                # Get performance on primary metric
                if 'performance' in result and primary_metric in result['performance']:
                    perf_value = result['performance'][primary_metric]
                    
                    # For metrics where lower is better, negate the change
                    if primary_metric in ['mse', 'rmse', 'mae']:
                        baseline = self.baseline_performance[primary_metric]
                        perf_drop = (perf_value - baseline) / baseline if baseline != 0 else 0
                    else:
                        baseline = self.baseline_performance[primary_metric]
                        perf_drop = (baseline - perf_value) / baseline if baseline != 0 else 0
                        
                    performance_values.append(perf_drop)
                else:
                    performance_values.append(0.0)
                    
            attack_results['success_rates'][attack_type] = success_rates
            attack_results['performance_drop'][attack_type] = performance_values
        
        # Create success rate plot
        fig_success, ax_success = plt.subplots(figsize=figsize)
        
        for attack_type in attack_types:
            ax_success.plot(epsilons, attack_results['success_rates'][attack_type], 'o-', label=attack_type)
            
        ax_success.set_xlabel('Epsilon (attack strength)')
        ax_success.set_ylabel('Attack Success Rate')
        ax_success.set_title('Adversarial Attack Success Rate vs. Strength')
        ax_success.legend()
        ax_success.grid(True, alpha=0.3)
        
        # Create performance drop plot
        fig_perf, ax_perf = plt.subplots(figsize=figsize)
        
        for attack_type in attack_types:
            ax_perf.plot(epsilons, attack_results['performance_drop'][attack_type], 'o-', label=attack_type)
            
        ax_perf.set_xlabel('Epsilon (attack strength)')
        ax_perf.set_ylabel(f'Relative Performance Drop ({primary_metric})')
        ax_perf.set_title('Model Performance Drop vs. Attack Strength')
        ax_perf.legend()
        ax_perf.grid(True, alpha=0.3)
        
        # Add plots to results
        attack_results['plots'] = {
            'success_rate': fig_success,
            'performance_drop': fig_perf
        }
        
        return attack_results
    
    def plot_attack_success_rates(self, results: Optional[Dict[str, Any]] = None, epsilons: Optional[List[float]] = None, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot attack success rates for different attack methods.
        
        Parameters:
        -----------
        results : dict or None
            Results from compare_attacks() or None to run comparison
        epsilons : list of float or None
            Different attack strengths to test if results is None
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.Figure : Success rates plot
        """
        # Run comparison if results not provided
        if results is None:
            if epsilons is None:
                epsilons = [0.01, 0.05, 0.1, 0.2]
                
            results = self.compare_attacks(epsilons=epsilons)
        
        # Return success rate plot if already created
        if 'plots' in results and 'success_rate' in results['plots']:
            return results['plots']['success_rate']
        
        # Create plot from results data
        fig, ax = plt.subplots(figsize=figsize)
        
        epsilons = results['epsilons']
        
        for attack_type, success_rates in results['success_rates'].items():
            ax.plot(epsilons, success_rates, 'o-', label=attack_type)
            
        ax.set_xlabel('Epsilon (attack strength)')
        ax.set_ylabel('Attack Success Rate')
        ax.set_title('Adversarial Attack Success Rate vs. Strength')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_performance_drop(self, results: Optional[Dict[str, Any]] = None, epsilons: Optional[List[float]] = None, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot performance drop for different attack methods.
        
        Parameters:
        -----------
        results : dict or None
            Results from compare_attacks() or None to run comparison
        epsilons : list of float or None
            Different attack strengths to test if results is None
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.Figure : Performance drop plot
        """
        # Run comparison if results not provided
        if results is None:
            if epsilons is None:
                epsilons = [0.01, 0.05, 0.1, 0.2]
                
            results = self.compare_attacks(epsilons=epsilons)
        
        # Return performance drop plot if already created
        if 'plots' in results and 'performance_drop' in results['plots']:
            return results['plots']['performance_drop']
        
        # Get primary metric based on problem type
        if self.is_classification():
            primary_metric = 'accuracy'
        else:
            primary_metric = 'mse'
        
        # Create plot from results data
        fig, ax = plt.subplots(figsize=figsize)
        
        epsilons = results['epsilons']
        
        for attack_type, perf_drops in results['performance_drop'].items():
            ax.plot(epsilons, perf_drops, 'o-', label=attack_type)
            
        ax.set_xlabel('Epsilon (attack strength)')
        ax.set_ylabel(f'Relative Performance Drop ({primary_metric})')
        ax.set_title('Model Performance Drop vs. Attack Strength')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def compute_robustness_score(self, epsilon: float = 0.1) -> float:
        """
        Compute an overall adversarial robustness score.
        
        Parameters:
        -----------
        epsilon : float
            Attack strength to use for score computation
            
        Returns:
        --------
        float : Robustness score (0-1, higher is more robust)
        """
        if self.verbose:
            print(f"Computing adversarial robustness score at epsilon={epsilon}")
        
        # Run black-box attack (works for all models)
        result = self.test_blackbox_random(epsilon=epsilon)
        
        # Get primary metric
        if self.is_classification():
            primary_metric = 'accuracy'
        else:
            primary_metric = 'mse'
        
        # Calculate robustness score
        if 'performance' in result and primary_metric in result['performance']:
            adv_perf = result['performance'][primary_metric]
            baseline_perf = self.baseline_performance[primary_metric]
            
            if primary_metric in ['mse', 'rmse', 'mae']:
                # For metrics where lower is better, robustness is inversely proportional to increase
                if baseline_perf != 0:
                    ratio = baseline_perf / adv_perf  # Ratio < 1 means worse performance
                    score = max(0, min(1, ratio))  # Clamp to [0, 1]
                else:
                    score = 0.0
            else:
                # For metrics where higher is better, robustness is proportional to retention
                if baseline_perf != 0:
                    ratio = adv_perf / baseline_perf  # Ratio < 1 means worse performance
                    score = max(0, min(1, ratio))  # Clamp to [0, 1]
                else:
                    score = 0.0
                    
            return float(score)
        else:
            # Default score if attack failed
            return 0.0