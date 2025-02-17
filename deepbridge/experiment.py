import typing as t
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Imports absolutos
from deepbridge.metrics.classification import Classification
from deepbridge.distillation.classification import KnowledgeDistillation, ModelType

class Experiment:
    """
    Experiment class to handle different types of modeling tasks and their configurations.
    """
    
    VALID_TYPES = ["binary_classification", "regression", "forecasting"]
    
    def __init__(
        self,
        dataset: 'DBDataset',
        experiment_type: str,
        test_size: float = 0.2,
        random_state: int = 42,
        config: t.Optional[dict] = None
    ):
        """
        Initialize the experiment with configuration and data.
        """
        if experiment_type not in self.VALID_TYPES:
            raise ValueError(f"experiment_type must be one of {self.VALID_TYPES}")
            
        self.experiment_type = experiment_type
        self.dataset = dataset
        self.test_size = test_size
        self.random_state = random_state
        self.config = config or {}
        
        # Initialize metrics calculator based on experiment type
        if experiment_type == "binary_classification":
            self.metrics_calculator = Classification()
            
        # Initialize results storage
        self.results = {
            'train': {},
            'test': {}
        }
        
        # Initialize distillation model
        self.distillation_model = None
        
        # Perform train-test split
        self._prepare_data()

    def fit(
        self,
        student_model_type: ModelType = ModelType.LOGISTIC_REGRESSION,
        student_params: t.Optional[dict] = None,
        temperature: float = 1.0,
        alpha: float = 0.5,
        use_probabilities: bool = True
    ) -> 'Experiment':
        """
        Train a Knowledge Distillation model using either teacher probabilities or teacher model.
        
        Args:
            student_model_type: Type of student model to use
            student_params: Custom parameters for student model
            temperature: Temperature parameter for softening probability distributions
            alpha: Weight between teacher's loss and true label loss
            use_probabilities: Whether to use pre-calculated probabilities (True) or teacher model (False)
            
        Returns:
            self: The experiment instance with trained distillation model
        """
        if self.experiment_type != "binary_classification":
            raise ValueError("Knowledge Distillation is only supported for binary classification")
            
        if use_probabilities:
            if self.prob_train is None:
                raise ValueError("No teacher probabilities available. Set use_probabilities=False to use teacher model")
                
            # Create distillation model from probabilities
            self.distillation_model = KnowledgeDistillation.from_probabilities(
                probabilities=self.prob_train,
                student_model_type=student_model_type,
                student_params=student_params,
                temperature=temperature,
                alpha=alpha
            )
        else:
            if self.dataset.model is None:
                raise ValueError("No teacher model available. Set use_probabilities=True to use pre-calculated probabilities")
                
            # Create distillation model from teacher model
            self.distillation_model = KnowledgeDistillation(
                teacher_model=self.dataset.model,
                student_model_type=student_model_type,
                student_params=student_params,
                temperature=temperature,
                alpha=alpha
            )
        
        # Train the model
        self.distillation_model.fit(self.X_train, self.y_train)
        
        # Evaluate on train set
        train_metrics = self.distillation_model.evaluate(
            self.X_train,
            self.y_train,
            return_predictions=True
        )
        self.results['train'] = train_metrics['metrics']
        
        # Evaluate on test set
        test_metrics = self.distillation_model.evaluate(
            self.X_test,
            self.y_test,
            return_predictions=True
        )
        self.results['test'] = test_metrics['metrics']
        
        return self
    
    def get_student_predictions(self, dataset: str = 'test') -> pd.DataFrame:
        """
        Get predictions from the trained student model.
        
        Args:
            dataset: Which dataset to get predictions for ('train' or 'test')
            
        Returns:
            DataFrame with predictions and probabilities
        """
        if self.distillation_model is None:
            raise ValueError("No trained distillation model available. Call fit() first")
            
        X = self.X_train if dataset == 'train' else self.X_test
        y_true = self.y_train if dataset == 'train' else self.y_test
        
        # Get predictions
        y_pred = self.distillation_model.predict(X)
        y_prob = self.distillation_model.predict_proba(X)
        
        # Create DataFrame
        predictions = pd.DataFrame({
            'y_true': y_true,
            'y_pred': y_pred,
            'prob_0': y_prob[:, 0],
            'prob_1': y_prob[:, 1]
        })
        
        return predictions
        
    def calculate_student_metrics(self, dataset: str = 'test') -> dict:
        """
        Calculate metrics for the distilled (student) model.
        
        Args:
            dataset: Which dataset to calculate metrics for ('train' or 'test')
            
        Returns:
            dict: Dictionary containing evaluation metrics for the student model
        """
        if self.distillation_model is None:
            raise ValueError("No trained distillation model available. Call fit() first")
            
        # Get predictions from student model
        predictions_df = self.get_student_predictions(dataset)
        
        # Calculate metrics
        metrics = self.calculate_metrics(
            y_true=predictions_df['y_true'],
            y_pred=predictions_df['y_pred'],
            y_prob=predictions_df['prob_1']  # Use probability of positive class
        )
        
        # Store metrics in results
        self.results[dataset] = metrics
        
        return metrics
        
    def compare_teacher_student_metrics(self) -> pd.DataFrame:
        """
        Compare metrics between teacher and student models for both train and test sets.
        
        Returns:
            pd.DataFrame: DataFrame containing metrics comparison
        """
        if self.distillation_model is None:
            raise ValueError("No trained distillation model available. Call fit() first")
            
        results = []
        
        # Calculate metrics for both datasets
        for dataset in ['train', 'test']:
            # Get teacher metrics
            teacher_metrics = None
            if dataset == 'train' and self.prob_train is not None:
                teacher_metrics = self.calculate_metrics(
                    y_true=self.y_train,
                    y_pred=self._get_binary_predictions(self.prob_train),
                    y_prob=self.prob_train['prob_1'] if 'prob_1' in self.prob_train.columns else self.prob_train.iloc[:, -1]
                )
            elif dataset == 'test' and self.prob_test is not None:
                teacher_metrics = self.calculate_metrics(
                    y_true=self.y_test,
                    y_pred=self._get_binary_predictions(self.prob_test),
                    y_prob=self.prob_test['prob_1'] if 'prob_1' in self.prob_test.columns else self.prob_test.iloc[:, -1]
                )
                
            # Get student metrics
            student_metrics = self.calculate_student_metrics(dataset)
            
            # Add to results
            for metric_name in student_metrics.keys():
                result = {
                    'dataset': dataset,
                    'metric': metric_name,
                    'student_value': student_metrics[metric_name]
                }
                if teacher_metrics:
                    result['teacher_value'] = teacher_metrics[metric_name]
                    result['difference'] = student_metrics[metric_name] - teacher_metrics[metric_name]
                results.append(result)
        
        # Create DataFrame
        comparison_df = pd.DataFrame(results)
        
        # Reorder columns
        if 'teacher_value' in comparison_df.columns:
            column_order = ['dataset', 'metric', 'teacher_value', 'student_value', 'difference']
            comparison_df = comparison_df[column_order]
            
        return comparison_df

    def _prepare_data(self) -> None:
        """
        Prepare the data by performing train-test split on features and target.
        """
        X = self.dataset.X
        y = self.dataset.target
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        # If we have original probabilities, split them too
        if self.dataset.original_prob is not None:
            prob_train_idx = self.X_train.index
            prob_test_idx = self.X_test.index
            
            self.prob_train = self.dataset.original_prob.loc[prob_train_idx]
            self.prob_test = self.dataset.original_prob.loc[prob_test_idx]
        else:
            self.prob_train = None
            self.prob_test = None
            
    def calculate_metrics(self, 
                         y_true: t.Union[np.ndarray, pd.Series],
                         y_pred: t.Union[np.ndarray, pd.Series],
                         y_prob: t.Optional[t.Union[np.ndarray, pd.Series]] = None) -> dict:
        """
        Calculate metrics based on experiment type.
        """
        if self.experiment_type == "binary_classification":
            return self.metrics_calculator.calculate_metrics(y_true, y_pred, y_prob)
        else:
            raise NotImplementedError(f"Metrics calculation not implemented for {self.experiment_type}")
            
    def evaluate_predictions(self, 
                           predictions: pd.DataFrame,
                           dataset: str = 'train',
                           pred_column: t.Optional[str] = None,
                           prob_column: t.Optional[str] = None,
                           threshold: float = 0.5) -> dict:
        """
        Evaluate predictions for the specified dataset.
        """
        if dataset not in ['train', 'test']:
            raise ValueError("dataset must be either 'train' or 'test'")
            
        y_true = self.y_train if dataset == 'train' else self.y_test
        
        # If pred_column is provided, use it directly
        if pred_column is not None:
            y_pred = predictions[pred_column]
        # Otherwise, convert probabilities to binary predictions
        else:
            y_pred = self._get_binary_predictions(predictions, threshold)
        
        # Get probabilities if prob_column is provided
        y_prob = predictions[prob_column] if prob_column else None
        
        metrics = self.calculate_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob
        )
        
        self.results[dataset] = metrics
        return metrics
    
    def get_dataset_split(self, dataset: str = 'train') -> tuple:
        """
        Get the features and target for specified dataset split.
        """
        if dataset == 'train':
            return self.X_train, self.y_train, self.prob_train
        elif dataset == 'test':
            return self.X_test, self.y_test, self.prob_test
        else:
            raise ValueError("dataset must be either 'train' or 'test'")
    
    def _get_binary_predictions(self, probabilities: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
        """
        Convert probability predictions to binary predictions using a threshold.
        """
        # If we have multiple columns, assume the last one is for class 1
        prob_values = probabilities.iloc[:, -1] if len(probabilities.columns) > 1 else probabilities.iloc[:, 0]
        return (prob_values >= threshold).astype(int)

    @property
    def metrics(self) -> dict:
        """
        Get all metrics for both train and test datasets.
        """
        # Calculate metrics if they haven't been calculated yet
        if not self.results['train'] and self.prob_train is not None:
            binary_preds = self._get_binary_predictions(self.prob_train)
            prob_values = self.prob_train.iloc[:, -1] if len(self.prob_train.columns) > 1 else self.prob_train.iloc[:, 0]
            
            metrics = self.calculate_metrics(
                y_true=self.y_train,
                y_pred=binary_preds,
                y_prob=prob_values
            )
            self.results['train'] = metrics
            
        if not self.results['test'] and self.prob_test is not None:
            binary_preds = self._get_binary_predictions(self.prob_test)
            prob_values = self.prob_test.iloc[:, -1] if len(self.prob_test.columns) > 1 else self.prob_test.iloc[:, 0]
            
            metrics = self.calculate_metrics(
                y_true=self.y_test,
                y_pred=binary_preds,
                y_prob=prob_values
            )
            self.results['test'] = metrics
            
        return {
            'train': self.results['train'],
            'test': self.results['test']
        }