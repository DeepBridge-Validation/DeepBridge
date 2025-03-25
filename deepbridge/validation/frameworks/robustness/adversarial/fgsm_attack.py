"""
Fast Gradient Sign Method (FGSM) attack implementation.

This module provides implementations of FGSM adversarial attacks
for evaluating model robustness.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
import warnings

try:
    # Try to import TensorFlow
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    
try:
    # Try to import PyTorch
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class FGSMAttacker:
    """
    Generate adversarial examples using the Fast Gradient Sign Method.
    
    This class provides methods for creating adversarial examples to test
    model robustness against gradient-based attacks.
    """
    
    def __init__(
        self,
        model: Any,
        framework: str = 'auto',
        loss_fn: Optional[Any] = None,
        epsilon: float = 0.1,
        norm: str = 'inf',
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None,
        targeted: bool = False,
        verbose: bool = False
    ):
        """
        Initialize the FGSM attacker.
        
        Parameters:
        -----------
        model : Any
            Model to attack
        framework : str
            Framework used by the model:
            - 'tensorflow': TensorFlow model
            - 'pytorch': PyTorch model
            - 'auto': Attempt to detect
        loss_fn : callable or None
            Loss function to use (None uses default)
        epsilon : float
            Attack strength
        norm : str
            Norm to use for the attack ('inf', 'l1', 'l2')
        clip_min : float or None
            Minimum value to clip adversarial examples
        clip_max : float or None
            Maximum value to clip adversarial examples
        targeted : bool
            Whether the attack is targeted
        verbose : bool
            Whether to print progress information
        """
        self.model = model
        self.epsilon = epsilon
        self.norm = norm
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted
        self.verbose = verbose
        
        # Determine framework
        if framework == 'auto':
            if TF_AVAILABLE and isinstance(model, tf.keras.Model):
                self.framework = 'tensorflow'
            elif TORCH_AVAILABLE and isinstance(model, torch.nn.Module):
                self.framework = 'pytorch'
            else:
                raise ValueError("Could not automatically determine framework. "
                               "Please specify 'tensorflow' or 'pytorch'.")
        else:
            self.framework = framework
            
        # Check framework availability
        if self.framework == 'tensorflow' and not TF_AVAILABLE:
            raise ImportError("TensorFlow not available")
        if self.framework == 'pytorch' and not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
            
        # Set up loss function
        if loss_fn is None:
            if self.framework == 'tensorflow':
                self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            else:  # pytorch
                self.loss_fn = torch.nn.CrossEntropyLoss()
        else:
            self.loss_fn = loss_fn
    
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
        if self.framework == 'tensorflow':
            X_adv = self._generate_tf(X_values, y_values)
        else:  # pytorch
            X_adv = self._generate_torch(X_values, y_values)
            
        # Convert back to original format
        if is_dataframe:
            return pd.DataFrame(X_adv, columns=columns, index=index)
        else:
            return X_adv
    
    def _generate_tf(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Generate adversarial examples using TensorFlow.
        
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
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available")
            
        # Convert to tensors
        X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
        y_tensor = tf.convert_to_tensor(y, dtype=tf.int32)
        
        # Create gradient tape
        with tf.GradientTape() as tape:
            tape.watch(X_tensor)
            
            # Forward pass
            predictions = self.model(X_tensor)
            
            # Calculate loss
            loss = self.loss_fn(y_tensor, predictions)
            
            if self.targeted:
                # For targeted attacks, we want to minimize the loss
                loss = -loss
                
        # Calculate gradients
        gradients = tape.gradient(loss, X_tensor)
        
        # Apply FGSM
        if self.norm == 'inf':
            # L-infinity norm: Take sign of gradients
            perturbation = self.epsilon * tf.sign(gradients)
        elif self.norm == 'l2':
            # L2 norm: Normalize gradients
            gradients_norm = tf.norm(gradients, ord=2, axis=tuple(range(1, len(gradients.shape))))
            gradients_norm = tf.maximum(gradients_norm, 1e-12)  # Avoid division by zero
            gradients_normalized = gradients / tf.reshape(gradients_norm, [-1] + [1] * (len(gradients.shape) - 1))
            perturbation = self.epsilon * gradients_normalized
        elif self.norm == 'l1':
            # L1 norm: Similar to L2 but with L1 normalization
            gradients_norm = tf.norm(gradients, ord=1, axis=tuple(range(1, len(gradients.shape))))
            gradients_norm = tf.maximum(gradients_norm, 1e-12)  # Avoid division by zero
            gradients_normalized = gradients / tf.reshape(gradients_norm, [-1] + [1] * (len(gradients.shape) - 1))
            perturbation = self.epsilon * gradients_normalized
        else:
            raise ValueError(f"Unsupported norm: {self.norm}")
            
        # Apply perturbation
        X_adv = X_tensor + perturbation
        
        # Clip values if specified
        if self.clip_min is not None or self.clip_max is not None:
            X_adv = tf.clip_by_value(X_adv, self.clip_min, self.clip_max)
            
        return X_adv.numpy()
    
    def _generate_torch(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Generate adversarial examples using PyTorch.
        
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
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
            
        # Convert to tensors
        device = next(self.model.parameters()).device
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device, requires_grad=True)
        y_tensor = torch.tensor(y, dtype=torch.long, device=device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Forward pass
        predictions = self.model(X_tensor)
        
        # Calculate loss
        loss = self.loss_fn(predictions, y_tensor)
        
        if self.targeted:
            # For targeted attacks, we want to minimize the loss
            loss = -loss
            
        # Calculate gradients
        loss.backward()
        gradients = X_tensor.grad.data
        
        # Apply FGSM
        if self.norm == 'inf':
            # L-infinity norm: Take sign of gradients
            perturbation = self.epsilon * torch.sign(gradients)
        elif self.norm == 'l2':
            # L2 norm: Normalize gradients
            gradients_norm = torch.norm(gradients.view(gradients.shape[0], -1), p=2, dim=1)
            gradients_norm = torch.clamp(gradients_norm, min=1e-12)  # Avoid division by zero
            gradients_normalized = gradients / gradients_norm.view(-1, *([1] * (len(gradients.shape) - 1)))
            perturbation = self.epsilon * gradients_normalized
        elif self.norm == 'l1':
            # L1 norm: Similar to L2 but with L1 normalization
            gradients_norm = torch.norm(gradients.view(gradients.shape[0], -1), p=1, dim=1)
            gradients_norm = torch.clamp(gradients_norm, min=1e-12)  # Avoid division by zero
            gradients_normalized = gradients / gradients_norm.view(-1, *([1] * (len(gradients.shape) - 1)))
            perturbation = self.epsilon * gradients_normalized
        else:
            raise ValueError(f"Unsupported norm: {self.norm}")
            
        # Apply perturbation
        X_adv = X_tensor + perturbation
        
        # Clip values if specified
        if self.clip_min is not None or self.clip_max is not None:
            X_adv = torch.clamp(X_adv, self.clip_min, self.clip_max)
            
        # Convert back to numpy
        return X_adv.detach().cpu().numpy()


def generate_fgsm_attack(
    model: Any,
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    framework: str = 'auto',
    epsilon: float = 0.1,
    norm: str = 'inf',
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
    targeted: bool = False
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Generate adversarial examples using the Fast Gradient Sign Method.
    
    Parameters:
    -----------
    model : Any
        Model to attack
    X : array-like or DataFrame
        Original inputs
    y : array-like or Series
        Target labels
    framework : str
        Framework used by the model
    epsilon : float
        Attack strength
    norm : str
        Norm to use for the attack ('inf', 'l1', 'l2')
    clip_min : float or None
        Minimum value to clip adversarial examples
    clip_max : float or None
        Maximum value to clip adversarial examples
    targeted : bool
        Whether the attack is targeted
        
    Returns:
    --------
    array-like or DataFrame : Adversarial examples
    """
    attacker = FGSMAttacker(
        model=model,
        framework=framework,
        epsilon=epsilon,
        norm=norm,
        clip_min=clip_min,
        clip_max=clip_max,
        targeted=targeted
    )
    
    return attacker.generate(X, y)