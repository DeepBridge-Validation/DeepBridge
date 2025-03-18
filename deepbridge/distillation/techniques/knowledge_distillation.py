import numpy as np
import pandas as pd
import optuna
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from typing import Dict, Any, Optional, Union, Callable, List, Tuple
import warnings

# Imports absolutos
from deepbridge.utils.model_registry import ModelRegistry, ModelType
from deepbridge.metrics.classification import Classification

class KnowledgeDistillation(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        teacher_model: Optional[BaseEstimator] = None,
        teacher_probabilities: Optional[np.ndarray] = None,
        student_model_type: ModelType = ModelType.LOGISTIC_REGRESSION,
        student_params: Dict[str, Any] = None,
        temperature: float = 1.0,
        alpha: float = 0.5,
        n_trials: int = 50,
        validation_split: float = 0.2,
        random_state: int = 42
    ):
        """
        Initialize the Knowledge Distillation model.
        
        Args:
            teacher_model: Pre-trained teacher model (optional if teacher_probabilities is provided)
            teacher_probabilities: Pre-calculated teacher probabilities (optional if teacher_model is provided)
            student_model_type: Type of student model to use
            student_params: Custom parameters for student model (if None, will be optimized)
            temperature: Temperature parameter for softening probability distributions
            alpha: Weight between teacher's loss and true label loss
            n_trials: Number of Optuna trials for hyperparameter optimization
            validation_split: Fraction of data to use for validation during optimization
            random_state: Random seed for reproducibility
        """
        if teacher_model is None and teacher_probabilities is None:
            raise ValueError("Either teacher_model or teacher_probabilities must be provided")
            
        self.teacher_model = teacher_model
        self.teacher_probabilities = teacher_probabilities
        self.student_model_type = student_model_type
        self.student_params = student_params
        self.temperature = temperature
        self.alpha = alpha
        self.n_trials = n_trials
        self.validation_split = validation_split
        self.random_state = random_state
        self.metrics_calculator = Classification()
        self.student_model = None
        self.best_params = None
        
    def _get_teacher_soft_labels(self, X: np.ndarray) -> np.ndarray:
        """
        Generate soft labels from either the teacher model or pre-calculated probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Soft labels (probabilities)
        """
        print(f"\n=== DEBUG: _get_teacher_soft_labels ===")
        print(f"X shape: {X.shape}")
        print(f"teacher_model: {self.teacher_model is not None}")
        print(f"teacher_probabilities: {self.teacher_probabilities is not None}")
        
        if self.teacher_probabilities is not None:
            # Use pre-calculated probabilities
            print(f"Using pre-calculated probabilities")
            if isinstance(self.teacher_probabilities, pd.DataFrame):
                print(f"teacher_probabilities is DataFrame with shape {self.teacher_probabilities.shape}")
                print(f"teacher_probabilities columns: {self.teacher_probabilities.columns.tolist()}")
                
                # Check if we have column names like 'prob_class_0', 'prob_class_1'
                if all(col in self.teacher_probabilities.columns for col in ['prob_class_0', 'prob_class_1']):
                    print(f"Found prob_class_0 and prob_class_1 columns")
                    probabilities = self.teacher_probabilities[['prob_class_0', 'prob_class_1']].values
                else:
                    probabilities = self.teacher_probabilities.values
            else:
                print(f"teacher_probabilities is {type(self.teacher_probabilities)} with shape {self.teacher_probabilities.shape}")
                probabilities = self.teacher_probabilities
                
            if len(probabilities) != len(X):
                raise ValueError(
                    f"Number of teacher probabilities ({len(probabilities)}) "
                    f"doesn't match number of samples ({len(X)})"
                )
                
            print(f"First 3 probabilities: {probabilities[:3]}")
            print(f"Probabilities shape: {probabilities.shape}")
            
            # Handle single column probabilities (convert to two columns)
            if len(probabilities.shape) == 1 or probabilities.shape[1] == 1:
                print(f"Single column probabilities detected, converting to two columns")
                if len(probabilities.shape) == 1:
                    pos_proba = probabilities
                else:
                    pos_proba = probabilities[:, 0]
                    
                # Ensure probabilities are between 0 and 1
                pos_proba = np.clip(pos_proba, 0.0, 1.0)
                probabilities = np.column_stack([1 - pos_proba, pos_proba])
                print(f"After conversion: probabilities shape {probabilities.shape}")
                print(f"First 3 rows after conversion: {probabilities[:3]}")
            elif probabilities.shape[1] != 2:
                print(f"WARNING: Expected 2 columns for binary classification, got {probabilities.shape[1]}")
                if probabilities.shape[1] > 2:
                    print(f"Using first two columns of probability array")
                    probabilities = probabilities[:, :2]
                    print(f"After selection: probabilities shape {probabilities.shape}")
                    print(f"First 3 rows after selection: {probabilities[:3]}")
            
            # Verify probabilities sum to 1
            row_sums = probabilities.sum(axis=1)
            if not np.allclose(row_sums, 1.0, rtol=1e-3):
                print(f"WARNING: Probabilities don't sum to 1. Min sum: {np.min(row_sums)}, Max sum: {np.max(row_sums)}")
                # Normalize
                probabilities = probabilities / row_sums[:, np.newaxis]
                print(f"Normalized probabilities. New sums: {probabilities.sum(axis=1)[:5]}")
                
            # Apply temperature scaling to probabilities
            print(f"Applying temperature={self.temperature} scaling")
            # Convert to logits
            epsilon = 1e-7
            probabilities = np.clip(probabilities, epsilon, 1 - epsilon)
            logits = np.log(probabilities)
            
            # Apply temperature scaling
            scaled_logits = logits / self.temperature
            
            # Convert back to probabilities using softmax
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            soft_labels = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            print(f"Soft labels shape: {soft_labels.shape}")
            print(f"First 3 soft labels: {soft_labels[:3]}")
            print(f"=== END DEBUG ===\n")
            
            return soft_labels
            
        # Use teacher model
        print(f"Using teacher model predictions")
        try:
            # Try to get logits using decision_function
            if hasattr(self.teacher_model, 'decision_function'):
                print(f"Using teacher model's decision_function")
                teacher_logits = self.teacher_model.decision_function(X)
                print(f"teacher_logits shape: {teacher_logits.shape}")
                
                if len(teacher_logits.shape) == 1:
                    # Convert to 2D array if necessary
                    print(f"Converting 1D logits to 2D, first 3 values: {teacher_logits[:3]}")
                    teacher_logits = np.column_stack([-teacher_logits, teacher_logits])
                    print(f"After conversion: {teacher_logits[:3]}")
            else:
                print(f"Teacher model has no decision_function, using predict_proba")
                # Fallback to predict_proba
                teacher_probs = self.teacher_model.predict_proba(X)
                print(f"teacher_probs shape: {teacher_probs.shape}, first 3 rows: {teacher_probs[:3]}")
                
                # Convert to logits
                epsilon = 1e-7
                teacher_probs = np.clip(teacher_probs, epsilon, 1-epsilon)
                teacher_logits = np.log(teacher_probs)
                print(f"Converted to logits, first 3 rows: {teacher_logits[:3]}")
        except (AttributeError, NotImplementedError) as e:
            print(f"Error getting teacher predictions: {str(e)}")
            print(f"Falling back to predict_proba")
            # Fallback to predict_proba
            teacher_probs = self.teacher_model.predict_proba(X)
            print(f"teacher_probs shape: {teacher_probs.shape}, first 3 rows: {teacher_probs[:3]}")
            
            # Convert to logits
            epsilon = 1e-7
            teacher_probs = np.clip(teacher_probs, epsilon, 1-epsilon)
            teacher_logits = np.log(teacher_probs)
        
        # Apply temperature scaling    
        print(f"Applying temperature={self.temperature} scaling to logits")
        scaled_logits = teacher_logits / self.temperature
        
        # Apply softmax
        soft_labels = softmax(scaled_logits, axis=1)
        print(f"Final soft labels shape: {soft_labels.shape}")
        print(f"First 3 soft labels: {soft_labels[:3]}")
        print(f"=== END DEBUG ===\n")
        
        return soft_labels
        
    def _get_param_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Obter o espaço de parâmetros do ModelRegistry.
        
        Args:
            trial: Optuna trial
            
        Returns:
            Dictionary of hyperparameters to try
        """
        # Utilizamos o método centralizado no ModelRegistry
        return ModelRegistry.get_param_space(self.student_model_type, trial)
        
    def _kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Calculate the Kullback-Leibler divergence between two probability distributions.
        
        Args:
            p: Target probability distribution
            q: Predicted probability distribution
            
        Returns:
            KL divergence value
        """
        # Add small value to avoid log(0)
        epsilon = 1e-10
        q = np.clip(q, epsilon, 1-epsilon)
        return np.sum(p * np.log(p / q))
        
    def _combined_loss(self, y_true: np.ndarray, soft_labels: np.ndarray, student_probs: np.ndarray) -> float:
        """
        Calculate the combined loss using both hard and soft labels.
        
        Args:
            y_true: One-hot encoded true labels
            soft_labels: Soft labels from teacher model
            student_probs: Probabilities from student model
            
        Returns:
            Combined loss value
        """
        # KL divergence for soft labels (distillation loss)
        distillation_loss = self._kl_divergence(soft_labels, student_probs)
        
        # Cross-entropy loss for hard labels
        epsilon = 1e-10
        student_probs = np.clip(student_probs, epsilon, 1-epsilon)
        hard_loss = -np.mean(np.sum(y_true * np.log(student_probs), axis=1))
        
        # Combined loss with alpha weighting
        return self.alpha * distillation_loss + (1 - self.alpha) * hard_loss
        
    def _objective(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray,
                soft_labels: np.ndarray) -> float:
        """
        Objective function for Optuna.
        
        Args:
            trial: Optuna trial
            X: Training features
            y: True labels
            soft_labels: Soft labels from teacher model
            
        Returns:
            Loss value to minimize
        """
        # Get hyperparameters for this trial
        trial_params = self._get_param_space(trial)
        
        
        # Split data for validation
        X_train, X_val, y_train, y_val, soft_train, soft_val = train_test_split(
            X, y, soft_labels, test_size=self.validation_split, random_state=self.random_state
        )
        
        # Create and train student model
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            student = ModelRegistry.get_model(self.student_model_type, trial_params)
            student.fit(X_train, y_train)
        
        # Get probabilities from student model
        student_probs = student.predict_proba(X_val)
        
        # Convert y_val to one-hot encoding for loss calculation
        n_classes = student_probs.shape[1]
        y_val_onehot = np.zeros((len(y_val), n_classes))
        y_val_onehot[np.arange(len(y_val)), y_val] = 1
        
        # Calculate combined loss
        return self._combined_loss(y_val_onehot, soft_val, student_probs)

    @classmethod
    def from_probabilities(
        cls,
        probabilities: Union[np.ndarray, pd.DataFrame],
        student_model_type: ModelType = ModelType.LOGISTIC_REGRESSION,
        student_params: Dict[str, Any] = None,
        temperature: float = 1.0,
        alpha: float = 0.5,
        n_trials: int = 50,
        validation_split: float = 0.2,
        random_state: int = 42
    ) -> 'KnowledgeDistillation':
        """
        Create a KnowledgeDistillation instance from pre-calculated probabilities.
        
        Args:
            probabilities: Array or DataFrame with shape (n_samples, 2) containing class probabilities
            student_model_type: Type of student model to use
            student_params: Custom parameters for student model (if None, will be optimized)
            temperature: Temperature parameter
            alpha: Weight parameter
            n_trials: Number of Optuna trials for hyperparameter optimization
            validation_split: Fraction of data to use for validation during optimization
            random_state: Random seed for reproducibility
            
        Returns:
            KnowledgeDistillation instance
        """
        print(f"Creating KnowledgeDistillation from probabilities")
        print(f"Probabilities type: {type(probabilities)}")
        
        if isinstance(probabilities, pd.DataFrame):
            # Special handling for named probability columns
            if all(col in probabilities.columns for col in ['prob_class_0', 'prob_class_1']):
                print(f"Found prob_class_0 and prob_class_1 columns in DataFrame")
                probs_array = probabilities[['prob_class_0', 'prob_class_1']].values
            else:
                print(f"Using all columns from DataFrame, shape: {probabilities.shape}")
                probs_array = probabilities.values
        else:
            probs_array = probabilities
            
        print(f"Processed probabilities shape: {probs_array.shape}")
        print(f"First 3 probabilities: {probs_array[:3]}")
        
        if probs_array.shape[1] != 2:
            print(f"WARNING: Expected probabilities with shape (n_samples, 2), got {probs_array.shape}")
            if len(probs_array.shape) == 1 or probs_array.shape[1] == 1:
                # Convert single-column probabilities to two columns
                if len(probs_array.shape) == 1:
                    pos_class = probs_array
                else:
                    pos_class = probs_array[:, 0]
                    
                # Ensure in range [0, 1]
                pos_class = np.clip(pos_class, 0.0, 1.0)
                probs_array = np.column_stack([1 - pos_class, pos_class])
                print(f"Converted to two-column format: {probs_array.shape}")
            elif probs_array.shape[1] > 2:
                # Use first two columns
                probs_array = probs_array[:, :2]
                print(f"Using first two columns: {probs_array.shape}")
                
        # Ensure probabilities sum to 1
        row_sums = probs_array.sum(axis=1)
        if not np.allclose(row_sums, 1.0, rtol=1e-3):
            print(f"Normalizing probabilities: min_sum={np.min(row_sums)}, max_sum={np.max(row_sums)}")
            probs_array = probs_array / row_sums[:, np.newaxis]
            
        if not np.allclose(probs_array.sum(axis=1), 1.0, rtol=1e-5):
            raise ValueError("Probabilities must sum to 1 for each sample")
            
        return cls(
            teacher_probabilities=probs_array,
            student_model_type=student_model_type,
            student_params=student_params,
            temperature=temperature,
            alpha=alpha,
            n_trials=n_trials,
            validation_split=validation_split,
            random_state=random_state
        )

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> 'KnowledgeDistillation':
        """
        Train the student model using Knowledge Distillation with hyperparameter optimization.
        
        Args:
            X: Training features
            y: True labels
            verbose: Whether to print optimization progress and results
            
        Returns:
            self: The trained model
        """
        # Generate soft labels
        soft_labels = self._get_teacher_soft_labels(X)
        
        if self.student_params is None:
            # Filter warnings during Optuna optimization
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                
                # Suppress Optuna logs
                import logging
                optuna_logger = logging.getLogger("optuna")
                optuna_logger_level = optuna_logger.getEffectiveLevel()
                optuna_logger.setLevel(logging.WARNING if verbose else logging.ERROR)
                
                # Optimize hyperparameters using Optuna
                study = optuna.create_study(direction="minimize")
                objective = lambda trial: self._objective(trial, X, y, soft_labels)
                study.optimize(objective, n_trials=self.n_trials)
                
                # Restore Optuna logger level
                optuna_logger.setLevel(optuna_logger_level)
                
                # Get the best hyperparameters
                self.best_params = study.best_params
                if verbose:
                    print(f"Best hyperparameters found: {self.best_params}")
                
                # Create student model with best parameters
                self.student_model = ModelRegistry.get_model(
                    model_type=self.student_model_type,
                    custom_params=self.best_params
                )
        else:
            # Use provided hyperparameters
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                self.student_model = ModelRegistry.get_model(
                    model_type=self.student_model_type,
                    custom_params=self.student_params
                )
            self.best_params = self.student_params
        
        # Train the student model, suppressing warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            self.student_model.fit(X, y)
        
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability predictions from the student model.
        
        Args:
            X: Input features
            
        Returns:
            Probability predictions
        """
        if self.student_model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        return self.student_model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Get class predictions from the student model.
        
        Args:
            X: Input features
            
        Returns:
            Class predictions
        """
        if self.student_model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        return self.student_model.predict(X)

    def evaluate(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        return_predictions: bool = False
    ) -> dict:
        """
        Evaluate the student model performance using multiple metrics.
        
        Args:
            X: Input features
            y_true: True labels
            return_predictions: Whether to include predictions in the output
            
        Returns:
            Dictionary containing evaluation metrics and optionally predictions
        """
        print("\n=== EVALUATING DISTILLATION MODEL ===")
        if self.student_model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
            
        # Get predictions
        y_pred = self.predict(X)
        student_probs = self.predict_proba(X)
        print(f"Student probabilities shape: {student_probs.shape}")
        print(f"First 3 student probabilities: {student_probs[:3]}")
        
        # Extract probability of positive class
        y_prob = student_probs[:, 1] if student_probs.shape[1] > 1 else student_probs
        print(f"y_prob shape: {y_prob.shape}")
        print(f"First 5 y_prob values: {y_prob[:5]}")
        
        # Get teacher soft labels for comparison
        teacher_soft_labels = self._get_teacher_soft_labels(X)
        print(f"Teacher soft labels shape: {teacher_soft_labels.shape}")
        print(f"First 3 teacher soft labels: {teacher_soft_labels[:3]}")
        
        # Extract teacher probability for positive class (consistently)
        teacher_prob = teacher_soft_labels[:, 1] if teacher_soft_labels.shape[1] > 1 else teacher_soft_labels
        print(f"teacher_prob shape: {teacher_prob.shape}")
        print(f"First 5 teacher_prob values: {teacher_prob[:5]}")
        
        # Verify both probability arrays
        print(f"y_prob stats: min={np.min(y_prob)}, max={np.max(y_prob)}, mean={np.mean(y_prob)}")
        print(f"teacher_prob stats: min={np.min(teacher_prob)}, max={np.max(teacher_prob)}, mean={np.mean(teacher_prob)}")
        
        # Ensure no NaN values
        if np.isnan(y_prob).any() or np.isnan(teacher_prob).any():
            print("WARNING: NaN values detected in probability arrays!")
        
        # Calculate metrics using Classification class
        print(f"Calculating metrics with Classification.calculate_metrics")
        metrics = self.metrics_calculator.calculate_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            teacher_prob=teacher_prob
        )
        
        print(f"Resulting metrics: {metrics}")
        
        # Manually calculate KS and R² if they're None in metrics
        if metrics.get('ks_statistic') is None:
            print("KS statistic is None, calculating manually...")
            try:
                from scipy import stats
                ks_stat, p_value = stats.ks_2samp(teacher_prob, y_prob)
                metrics['ks_statistic'] = float(ks_stat)
                metrics['ks_pvalue'] = float(p_value)
                print(f"Manual KS calculation: statistic={ks_stat}, p-value={p_value}")
            except Exception as e:
                print(f"Manual KS calculation failed: {str(e)}")
        
        if metrics.get('r2_score') is None:
            print("R² score is None, calculating manually...")
            try:
                from sklearn.metrics import r2_score
                # Sort for distribution comparison
                teacher_sorted = np.sort(teacher_prob)
                student_sorted = np.sort(y_prob)
                min_len = min(len(teacher_sorted), len(student_sorted))
                r2 = r2_score(teacher_sorted[:min_len], student_sorted[:min_len])
                metrics['r2_score'] = float(r2)
                print(f"Manual R² calculation: {r2}")
            except Exception as e:
                print(f"Manual R² calculation failed: {str(e)}")
        
        # Add hyperparameter info
        metrics['best_params'] = self.best_params
        
        if return_predictions:
            # Create DataFrame with predictions
            predictions_df = pd.DataFrame({
                'y_true': y_true,
                'y_pred': y_pred,
                'y_prob': y_prob,
                'teacher_prob': teacher_prob
            })
            return {'metrics': metrics, 'predictions': predictions_df}
        
        print("=== EVALUATION COMPLETE ===\n")
        return metrics

    def evaluate_from_dataframe(
        self,
        data: pd.DataFrame,
        features_columns: list,
        target_column: str,
        return_predictions: bool = False
    ) -> dict:
        """
        Evaluate model using a DataFrame as input.
        
        Args:
            data: Input DataFrame
            features_columns: List of feature column names
            target_column: Name of the target column
            return_predictions: Whether to include predictions in the output
            
        Returns:
            Dictionary containing evaluation metrics and optionally predictions
        """
        X = data[features_columns].values
        y_true = data[target_column].values
        
        return self.evaluate(X, y_true, return_predictions=return_predictions)