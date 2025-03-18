import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from typing import Dict, Any, Union, Optional, List

# Imports absolutos
from deepbridge.utils.model_registry import ModelRegistry, ModelType, ModelMode
from deepbridge.metrics.classification import Classification

class SurrogateModel:
    """
    SurrogateModel provides a simple and direct approach to model distillation
    by fitting regression models to the output probabilities of a complex model.
    
    Unlike the more sophisticated KnowledgeDistillation class, this implementation
    takes a direct regression-based approach to mimicking the teacher model's outputs.
    """
    
    def __init__(
        self, 
        model_type: ModelType = ModelType.GBM, 
        model_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the surrogate model with the chosen model type.

        Args:
            model_type: Type of model to use from ModelType enum
            model_params: Optional custom parameters for the model
        """
        self.model_type = model_type
        self.model_params = model_params or {}
        self.is_fitted = False
        self.metrics_calculator = Classification()
        
        # Create the model using ModelRegistry with REGRESSION mode
        self.model = ModelRegistry.get_model(
            model_type, 
            self.model_params, 
            mode=ModelMode.REGRESSION
        )

    def fit(
        self, 
        X: Union[np.ndarray, pd.DataFrame], 
        probas: Union[np.ndarray, pd.DataFrame, pd.Series], 
        test_size: float = 0.2,
        random_state: Optional[int] = 42,
        verbose: bool = True
    ) -> 'SurrogateModel':
        """
        Train the surrogate model using original features and teacher model probabilities.

        Args:
            X: Original features
            probas: Probabilities predicted by the teacher model
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            verbose: Whether to print training metrics

        Returns:
            self: The trained surrogate model
        """
        # Split data into train and test sets
        X_train, X_test, probas_train, probas_test = train_test_split(
            X, probas, test_size=test_size, random_state=random_state
        )

        # Process probabilities based on format
        probas_train = self._process_probabilities(probas_train)
        probas_test = self._process_probabilities(probas_test)

        # Convert probabilities to binary labels for evaluation
        probas_train_binary = (probas_train > 0.5).astype(int)
        probas_test_binary = (probas_test > 0.5).astype(int)

        # Apply logit transformation to probabilities (with small epsilon to avoid log(0) or log(1))
        y_train = logit(np.clip(probas_train, 0.0001, 0.9999))
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_fitted = True

        # Evaluate the model if verbose
        if verbose:
            # Make predictions
            train_logits = self.model.predict(X_train)
            test_logits = self.model.predict(X_test)
            
            # Convert logits back to probabilities
            train_probas = expit(train_logits)
            test_probas = expit(test_logits)

            # Use the Classification metrics calculator
            train_metrics = self.metrics_calculator.calculate_metrics(
                y_true=probas_train_binary,
                y_pred=(train_probas > 0.5).astype(int),
                y_prob=train_probas
            )
            
            test_metrics = self.metrics_calculator.calculate_metrics(
                y_true=probas_test_binary,
                y_pred=(test_probas > 0.5).astype(int),
                y_prob=test_probas
            )

            print("Surrogate Model Training Results:")
            print(f"Train metrics: Accuracy={train_metrics.get('accuracy', 'N/A'):.4f}, "
                  f"AUC-ROC={train_metrics.get('auc_roc', 'N/A'):.4f}")
            print(f"Test metrics: Accuracy={test_metrics.get('accuracy', 'N/A'):.4f}, "
                  f"AUC-ROC={test_metrics.get('auc_roc', 'N/A'):.4f}")
            
            # Store metrics
            self.train_metrics = train_metrics
            self.test_metrics = test_metrics

        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make probability predictions with the surrogate model.

        Args:
            X: Input features

        Returns:
            Probability predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        # Make predictions (logits)
        logits = self.model.predict(X)
        
        # Convert logits to probabilities
        probabilities = expit(logits)
        
        return probabilities
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make probability predictions with the surrogate model.
        Returns probabilities for both classes.

        Args:
            X: Input features

        Returns:
            Array with shape (n_samples, 2) containing probabilities for both classes
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        # Get positive class probabilities
        pos_probas = self.predict(X)
        
        # Create array with both class probabilities
        probas = np.column_stack([1 - pos_probas, pos_probas])
        
        return probas
        
    def _process_probabilities(self, probas: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """
        Process input probabilities to extract positive class probabilities.

        Args:
            probas: Input probabilities in various formats

        Returns:
            Numpy array with positive class probabilities
        """
        # Handle DataFrame
        if isinstance(probas, pd.DataFrame):
            # If we have a DataFrame with multiple columns, extract the positive class column
            if probas.shape[1] >= 2:
                # Try to find the right column
                if 'prob_class_1' in probas.columns:
                    return probas['prob_class_1'].values
                elif 'prob_1' in probas.columns:
                    return probas['prob_1'].values
                else:
                    # Default to the second column (typically positive class)
                    return probas.iloc[:, 1].values
            else:
                # Single column DataFrame
                return probas.iloc[:, 0].values
                
        # Handle Series
        elif isinstance(probas, pd.Series):
            return probas.values
            
        # Handle numpy array
        elif isinstance(probas, np.ndarray):
            # If 2D array with multiple columns
            if len(probas.shape) > 1 and probas.shape[1] >= 2:
                return probas[:, 1]  # Return positive class (second column)
            else:
                # 1D array or single column
                return probas.flatten()
                
        else:
            raise ValueError(f"Unsupported probability format: {type(probas)}")
    
    def evaluate(
        self, 
        X: Union[np.ndarray, pd.DataFrame], 
        y_true: Union[np.ndarray, pd.Series], 
        teacher_prob: Optional[Union[np.ndarray, pd.DataFrame, pd.Series]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the surrogate model using the Classification metrics calculator.

        Args:
            X: Input features
            y_true: True labels
            teacher_prob: Optional teacher model probabilities for comparison

        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
            
        # Get surrogate model predictions
        surrogate_probas = self.predict(X)
        surrogate_preds = (surrogate_probas > 0.5).astype(int)
        
        # Calculate metrics using Classification metrics calculator
        metrics = self.metrics_calculator.calculate_metrics(
            y_true=y_true,
            y_pred=surrogate_preds,
            y_prob=surrogate_probas,
            teacher_prob=teacher_prob
        )
        
        return metrics

    @classmethod
    def from_probabilities(
        cls,
        probabilities: Union[np.ndarray, pd.DataFrame],
        student_model_type: ModelType = ModelType.GBM,
        student_params: Dict[str, Any] = None,
        random_state: int = 42,
        validation_split: float = 0.2,
        n_trials: int = 10
    ) -> 'SurrogateModel':
        """
        Create a SurrogateModel instance from pre-calculated probabilities.
        
        Args:
            probabilities: Array or DataFrame containing class probabilities
            student_model_type: Type of student model to use
            student_params: Custom parameters for student model (if None, will use defaults)
            random_state: Random seed for reproducibility
            validation_split: Parameter added for compatibility with KnowledgeDistillation API
            n_trials: Parameter added for compatibility with KnowledgeDistillation API
            
        Returns:
            SurrogateModel instance
        """
        # These parameters are ignored but included for API compatibility with KnowledgeDistillation
        model_params = student_params or {}
        if random_state is not None:
            model_params['random_state'] = random_state

        return cls(
            model_type=student_model_type,
            model_params=model_params
        )