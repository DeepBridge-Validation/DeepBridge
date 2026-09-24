import typing as t

import numpy as np
import pandas as pd


class ModelEvaluation:
    """
    Handles model evaluation, metric calculation, and model comparison.
    """

    def __init__(self, experiment_type, metrics_calculator):
        self.experiment_type = experiment_type
        self.metrics_calculator = metrics_calculator

    def calculate_metrics(
        self,
        y_true: t.Union[np.ndarray, pd.Series],
        y_pred: t.Union[np.ndarray, pd.Series],
        y_prob: t.Optional[t.Union[np.ndarray, pd.Series]] = None,
    ) -> dict:
        """
        Calculate metrics based on experiment type.
        """
        if self.experiment_type == 'binary_classification':
            return self.metrics_calculator.calculate_metrics(
                y_true, y_pred, y_prob
            )
        else:
            raise NotImplementedError(
                f'Metrics calculation not implemented for {self.experiment_type}'
            )

    def get_predictions(self, model, X, y_true):
        """Get predictions from a model"""
        # Get probabilities
        probs = model.predict(X)

        # Convert to binary predictions
        y_pred = (probs > 0.5).astype(int)

        # Get probability distributions
        y_prob = model.predict_proba(X)

        # Create DataFrame
        predictions = pd.DataFrame(
            {
                'y_true': y_true,
                'y_pred': y_pred,
                'prob_0': y_prob[:, 0],
                'prob_1': y_prob[:, 1],
            }
        )

        return predictions

    def evaluate_model(self, model, model_name, model_type, X, y):
        """Evaluate a single model"""
        try:
            # Check if the model is a regressor used for classification
            is_regressor = 'regressor' in model.__class__.__name__.lower()

            if (
                is_regressor
                and self.experiment_type == 'binary_classification'
            ):
                # For regressors in classification problems:
                # 1. Get continuous predictions (logits)
                logits = model.predict(X)

                # 2. Convert to probabilities using the sigmoid function
                from scipy.special import expit

                y_prob = expit(logits)

                # 3. Convert to binary predictions using threshold
                y_pred = (y_prob > 0.5).astype(int)

            else:
                # For regular models
                y_pred = model.predict(X)

                # Get probabilities if available
                y_prob = None
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(X)
                    if probs.shape[1] > 1:  # Binary or multiclass
                        y_prob = probs[:, 1]  # Probability of positive class

            # Calculate metrics
            metrics = self.calculate_metrics(
                y_true=y,
                y_pred=y_pred,
                y_prob=y_prob if y_prob is not None else None,
            )

            # Add model info
            metrics['model_name'] = model_name
            metrics['model_type'] = model_type

            return metrics
        except Exception as e:
            print(f'Failed to evaluate model {model_name}: {str(e)}')
            return None

    def compare_all_models(
        self,
        dataset,
        original_model,
        alternative_models,
        X,
        y,
    ):
        """Compare all models on the specified dataset"""
        results = []

        # Add original model if available
        if original_model is not None:
            original_name = original_model.__class__.__name__
            metrics = self.evaluate_model(
                original_model, original_name, 'original', X, y
            )
            if metrics:
                results.append(metrics)

        # Add alternative models
        for name, model in alternative_models.items():
            metrics = self.evaluate_model(model, name, 'alternative', X, y)
            if metrics:
                results.append(metrics)

        # Convert results to DataFrame
        comparison_df = pd.DataFrame(results)

        # Reorder columns to put model info first
        if not comparison_df.empty:
            cols = ['model_name', 'model_type'] + [
                col
                for col in comparison_df.columns
                if col not in ['model_name', 'model_type']
            ]
            comparison_df = comparison_df[cols]

        return comparison_df
