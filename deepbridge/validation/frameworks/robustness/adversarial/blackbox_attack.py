"""
Black-box attack implementation.

This module provides implementations of black-box adversarial attacks
for evaluating model robustness when gradients are not available.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
import warnings
from sklearn.preprocessing import StandardScaler


class BlackboxAttacker:
    """
    Generate adversarial examples using black-box techniques.
    
    This class provides methods for creating adversarial examples without
    requiring access to model gradients.
    """
    
    def __init__(
        self,
        model: Any,
        attack_type: str = 'random',
        epsilon: float = 0.1,
        n_trials: int = 100,
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None,
        verbose: bool = False
    ):
        """
        Initialize the black-box attacker.
        
        Parameters:
        -----------
        model : Any
            Model to attack (must have a predict or predict_proba method)
        attack_type : str
            Type of black-box attack:
            - 'random': Random perturbation
            - 'boundary': Boundary attack
        epsilon : float
            Maximum perturbation
        n_trials : int
            Number of trials for random search
        clip_min : float or None
            Minimum value to clip adversarial examples
        clip_max : float or None
            Maximum value to clip adversarial examples
        verbose : bool
            Whether to print progress information
        """
        self.model = model
        self.attack_type = attack_type
        self.epsilon = epsilon
        self.n_trials = n_trials
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.verbose = verbose
        
        # Check if model has required methods
        if not hasattr(model, 'predict') and not hasattr(model, 'predict_proba'):
            raise ValueError("Model must have predict or predict_proba method")
            
        # Determine prediction method
        if hasattr(model, 'predict_proba'):
            self.predict_fn = model.predict_proba
        else:
            self.predict_fn = model.predict
    
    def generate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series]
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Generate adversarial examples.
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Original inputs
        y : array-like or Series
            Target labels
            
        Returns:
        --------
        array-like or DataFrame : Adversarial examples
        """
        # Convert to appropriate format
        is_dataframe = isinstance(X, pd.DataFrame)
        if is_dataframe:
            columns = X.columns
            index = X.index
            X_values = X.values
        else:
            X_values = X
            
        # Convert y to appropriate format
        if isinstance(y, pd.Series):
            y_values = y.values
        else:
            y_values = y
            
        # Generate adversarial examples
        if self.attack_type == 'random':
            X_adv = self._random_attack(X_values, y_values)
        elif self.attack_type == 'boundary':
            X_adv = self._boundary_attack(X_values, y_values)
        else:
            raise ValueError(f"Unknown attack type: {self.attack_type}")
            
        # Convert back to original format
        if is_dataframe:
            return pd.DataFrame(X_adv, columns=columns, index=index)
        else:
            return X_adv
    
    def _random_attack(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Generate adversarial examples using random perturbation.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Original inputs
        y : numpy.ndarray
            Target labels
            
        Returns:
        --------
        numpy.ndarray : Adversarial examples
        """
        # Initialize adversarial examples with originals
        X_adv = X.copy()
        
        # Get original predictions
        original_preds = np.argmax(self.predict_fn(X), axis=1) if hasattr(
            self.predict_fn(X), 'shape') and self.predict_fn(X).ndim > 1 else self.predict_fn(X)
        
        # For each sample
        for i in range(len(X)):
            if self.verbose and (i + 1) % (len(X) // 10 or 1) == 0:
                print(f"Processing sample {i + 1}/{len(X)}")
                
            # Skip if already misclassified
            if original_preds[i] != y[i]:
                continue
                
            # Try random perturbations
            best_dist = float('inf')
            best_adv = None
            
            for _ in range(self.n_trials):
                # Generate random perturbation
                noise = np.random.normal(0, self.epsilon, X[i].shape)
                
                # Apply perturbation
                perturbed = X[i] + noise
                
                # Clip values if specified
                if self.clip_min is not None or self.clip_max is not None:
                    perturbed = np.clip(perturbed, self.clip_min, self.clip_max)
                    
                # Reshape to match model input
                perturbed_input = perturbed.reshape(1, -1)
                
                # Get prediction
                pred = np.argmax(self.predict_fn(perturbed_input), axis=1)[0] if hasattr(
                    self.predict_fn(perturbed_input), 'shape') and self.predict_fn(perturbed_input).ndim > 1 else self.predict_fn(perturbed_input)[0]
                
                # Check if adversarial
                if pred != y[i]:
                    # Calculate distance
                    dist = np.linalg.norm(perturbed - X[i])
                    
                    # Update best adversarial example
                    if dist < best_dist:
                        best_dist = dist
                        best_adv = perturbed
                        
            # Update adversarial example
            if best_adv is not None:
                X_adv[i] = best_adv
                
        return X_adv
    
    def _boundary_attack(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Generate adversarial examples using boundary attack.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Original inputs
        y : numpy.ndarray
            Target labels
            
        Returns:
        --------
        numpy.ndarray : Adversarial examples
        """
        warnings.warn("Boundary attack is computationally expensive and may take a long time.")
        
        # Initialize adversarial examples with originals
        X_adv = X.copy()
        
        # Get original predictions
        original_preds = np.argmax(self.predict_fn(X), axis=1) if hasattr(
            self.predict_fn(X), 'shape') and self.predict_fn(X).ndim > 1 else self.predict_fn(X)
        
        # For each sample
        for i in range(len(X)):
            if self.verbose and (i + 1) % (len(X) // 10 or 1) == 0:
                print(f"Processing sample {i + 1}/{len(X)}")
                
            # Skip if already misclassified
            if original_preds[i] != y[i]:
                continue
                
            # Find an adversarial starting point
            adv_found = False
            for _ in range(self.n_trials):
                # Generate random sample (far from original)
                if self.clip_min is not None and self.clip_max is not None:
                    random_sample = np.random.uniform(self.clip_min, self.clip_max, X[i].shape)
                else:
                    # Use mean and std of the dataset
                    scaler = StandardScaler()
                    scaler.fit(X)
                    random_sample = np.random.normal(scaler.mean_, scaler.scale_, X[i].shape)
                    
                # Reshape to match model input
                sample_input = random_sample.reshape(1, -1)
                
                # Get prediction
                pred = np.argmax(self.predict_fn(sample_input), axis=1)[0] if hasattr(
                    self.predict_fn(sample_input), 'shape') and self.predict_fn(sample_input).ndim > 1 else self.predict_fn(sample_input)[0]
                
                # Check if adversarial
                if pred != y[i]:
                    adv_found = True
                    current_adv = random_sample
                    break
                    
            if not adv_found:
                continue
                
            # Boundary attack iterations
            n_steps = 10  # Number of steps for boundary attack
            step_size = 0.2  # Initial step size
            
            for step in range(n_steps):
                if self.verbose and (step + 1) % (n_steps // 5 or 1) == 0:
                    print(f"  Boundary step {step + 1}/{n_steps}")
                    
                # Calculate direction towards original
                direction = X[i] - current_adv
                direction_norm = np.linalg.norm(direction)
                
                if direction_norm < 1e-8:  # Avoid division by zero
                    break
                    
                # Normalize direction
                direction = direction / direction_norm
                
                # New candidate closer to original
                candidate = current_adv + step_size * direction
                
                # Clip values if specified
                if self.clip_min is not None or self.clip_max is not None:
                    candidate = np.clip(candidate, self.clip_min, self.clip_max)
                    
                # Reshape to match model input
                candidate_input = candidate.reshape(1, -1)
                
                # Get prediction
                pred = np.argmax(self.predict_fn(candidate_input), axis=1)[0] if hasattr(
                    self.predict_fn(candidate_input), 'shape') and self.predict_fn(candidate_input).ndim > 1 else self.predict_fn(candidate_input)[0]
                
                # Check if still adversarial
                if pred != y[i]:
                    current_adv = candidate
                    step_size *= 1.1  # Increase step size
                else:
                    step_size *= 0.5  # Decrease step size
                    
            # Update adversarial example
            X_adv[i] = current_adv
            
        return X_adv


def generate_blackbox_attack(
    model: Any,
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    attack_type: str = 'random',
    epsilon: float = 0.1,
    n_trials: int = 100,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
    verbose: bool = False
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Generate adversarial examples using black-box techniques.
    
    Parameters:
    -----------
    model : Any
        Model to attack
    X : array-like or DataFrame
        Original inputs
    y : array-like or Series
        Target labels
    attack_type : str
        Type of black-box attack
    epsilon : float
        Maximum perturbation
    n_trials : int
        Number of trials for random search
    clip_min : float or None
        Minimum value to clip adversarial examples
    clip_max : float or None
        Maximum value to clip adversarial examples
    verbose : bool
        Whether to print progress information
        
    Returns:
    --------
    array-like or DataFrame : Adversarial examples
    """
    attacker = BlackboxAttacker(
        model=model,
        attack_type=attack_type,
        epsilon=epsilon,
        n_trials=n_trials,
        clip_min=clip_min,
        clip_max=clip_max,
        verbose=verbose
    )
    
    return attacker.generate(X, y)