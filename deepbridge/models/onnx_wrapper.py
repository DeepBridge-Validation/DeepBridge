"""
ONNX Model Wrapper for DeepBridge

This module provides a scikit-learn compatible wrapper for ONNX models,
allowing them to be used seamlessly with DeepBridge experiments.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, List, Dict, Any
import logging

# Configure logger
logger = logging.getLogger("deepbridge.models.onnx")


class ONNXModelWrapper:
    """
    Wrapper class for ONNX models that provides scikit-learn compatible interface.

    This wrapper allows ONNX models to be used as the primary model in DeepBridge
    experiments, supporting both classification and regression tasks.
    """

    def __init__(
        self,
        onnx_path: str = None,
        onnx_session = None,
        task_type: str = "classification",
        input_name: str = None,
        output_name: str = None,
        feature_names: List[str] = None,
        class_names: List[str] = None,
        metadata: Dict[str, Any] = None
    ):
        """
        Initialize the ONNX model wrapper.

        Parameters
        ----------
        onnx_path : str, optional
            Path to the ONNX model file
        onnx_session : onnxruntime.InferenceSession, optional
            Pre-loaded ONNX inference session
        task_type : str, default="classification"
            Type of task: "classification" or "regression"
        input_name : str, optional
            Name of the input node. If None, will use the first input
        output_name : str, optional
            Name of the output node. If None, will use the first output
        feature_names : List[str], optional
            Names of the features expected by the model
        class_names : List[str], optional
            Names of the classes for classification tasks
        metadata : Dict[str, Any], optional
            Additional metadata about the model
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required for ONNX model support. "
                "Install it with: pip install onnxruntime"
            )

        self.ort = ort
        self.task_type = task_type.lower()
        self.feature_names = feature_names
        self.class_names = class_names
        self.metadata = metadata or {}

        # Load or set the ONNX session
        if onnx_session is not None:
            self.session = onnx_session
            logger.info("Using provided ONNX session")
        elif onnx_path is not None:
            self.session = ort.InferenceSession(onnx_path)
            logger.info(f"Loaded ONNX model from: {onnx_path}")
        else:
            raise ValueError("Either onnx_path or onnx_session must be provided")

        # Get input and output names
        self.input_name = input_name or self.session.get_inputs()[0].name
        self.output_name = output_name or self.session.get_outputs()[0].name

        # Get input shape information
        input_info = self.session.get_inputs()[0]
        self.input_shape = input_info.shape
        self.n_features = input_info.shape[1] if len(input_info.shape) > 1 else None

        # Store model information
        self.model_info = {
            'input_names': [inp.name for inp in self.session.get_inputs()],
            'output_names': [out.name for out in self.session.get_outputs()],
            'input_shapes': [inp.shape for inp in self.session.get_inputs()],
            'output_shapes': [out.shape for out in self.session.get_outputs()]
        }

        logger.info(f"ONNX Model initialized - Task: {self.task_type}")
        logger.info(f"Input: {self.input_name}, Output: {self.output_name}")

        # For classification, determine number of classes
        if self.task_type == "classification":
            output_shape = self.session.get_outputs()[0].shape
            if len(output_shape) > 1 and output_shape[1] is not None:
                self.n_classes_ = output_shape[1]
            else:
                self.n_classes_ = 2  # Default to binary classification

            if self.class_names is None:
                self.class_names = [f"class_{i}" for i in range(self.n_classes_)]

            logger.info(f"Classification model with {self.n_classes_} classes")

    def _prepare_input(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Prepare input data for ONNX inference.

        Parameters
        ----------
        X : array-like
            Input features

        Returns
        -------
        np.ndarray
            Prepared input array
        """
        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        elif not isinstance(X, np.ndarray):
            X = np.array(X)

        # Ensure float32 type (most ONNX models expect float32)
        X = X.astype(np.float32)

        # Ensure 2D shape
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        return X

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions using the ONNX model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features

        Returns
        -------
        np.ndarray
            Predictions (class labels for classification, values for regression)
        """
        X = self._prepare_input(X)

        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: X})
        predictions = outputs[0]

        if self.task_type == "classification":
            # For classification, return class labels
            if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                # Multi-class or binary with probability output
                predictions = np.argmax(predictions, axis=1)
            else:
                # Binary classification with single output
                predictions = (predictions > 0.5).astype(int).ravel()
        else:
            # For regression, return values directly
            predictions = predictions.ravel()

        return predictions

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict class probabilities for classification tasks.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
            Class probabilities
        """
        if self.task_type != "classification":
            raise ValueError("predict_proba is only available for classification tasks")

        X = self._prepare_input(X)

        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: X})
        probabilities = outputs[0]

        # Handle different output formats
        if len(probabilities.shape) == 1 or probabilities.shape[1] == 1:
            # Binary classification with single probability output
            probabilities = probabilities.ravel()
            # Convert to two-class format
            probabilities = np.column_stack([1 - probabilities, probabilities])
        elif probabilities.shape[1] == 2:
            # Already in correct format for binary classification
            pass
        else:
            # Multi-class classification
            # Apply softmax if not already applied
            if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-3):
                # Apply softmax
                exp_scores = np.exp(probabilities - np.max(probabilities, axis=1, keepdims=True))
                probabilities = exp_scores / exp_scores.sum(axis=1, keepdims=True)

        return probabilities

    def score(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> float:
        """
        Return the mean accuracy for classification or R² score for regression.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features
        y : array-like of shape (n_samples,)
            True labels or values

        Returns
        -------
        float
            Score value
        """
        predictions = self.predict(X)

        if isinstance(y, pd.Series):
            y = y.values

        if self.task_type == "classification":
            # Return accuracy
            return np.mean(predictions == y)
        else:
            # Return R² score for regression
            from sklearn.metrics import r2_score
            return r2_score(y, predictions)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get parameters for this estimator (scikit-learn compatibility).

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator

        Returns
        -------
        Dict[str, Any]
            Parameter names mapped to their values
        """
        return {
            'task_type': self.task_type,
            'input_name': self.input_name,
            'output_name': self.output_name,
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'metadata': self.metadata
        }

    def set_params(self, **params) -> 'ONNXModelWrapper':
        """
        Set the parameters of this estimator (scikit-learn compatibility).

        Parameters
        ----------
        **params : dict
            Estimator parameters

        Returns
        -------
        self
            Estimator instance
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    @property
    def feature_importances_(self) -> np.ndarray:
        """
        Feature importances (not available for ONNX models).

        Returns None or raises AttributeError.
        """
        # ONNX models typically don't have feature importances
        # You could implement SHAP or LIME here if needed
        raise AttributeError("ONNX models do not provide feature importances directly")

    def __repr__(self) -> str:
        """String representation of the wrapper."""
        return (f"ONNXModelWrapper(task_type='{self.task_type}', "
                f"input={self.input_name}, output={self.output_name}, "
                f"n_features={self.n_features})")

    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"ONNX {self.task_type.title()} Model"


def load_onnx_model(
    onnx_path: str,
    task_type: str = "classification",
    **kwargs
) -> ONNXModelWrapper:
    """
    Convenience function to load an ONNX model.

    Parameters
    ----------
    onnx_path : str
        Path to the ONNX model file
    task_type : str, default="classification"
        Type of task: "classification" or "regression"
    **kwargs : dict
        Additional arguments passed to ONNXModelWrapper

    Returns
    -------
    ONNXModelWrapper
        Wrapped ONNX model ready for use with DeepBridge

    Examples
    --------
    >>> from deepbridge.models import load_onnx_model
    >>> model = load_onnx_model("model.onnx", task_type="classification")
    >>> predictions = model.predict(X_test)
    """
    return ONNXModelWrapper(onnx_path=onnx_path, task_type=task_type, **kwargs)