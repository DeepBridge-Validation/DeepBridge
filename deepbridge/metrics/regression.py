import typing as t

import numpy as np
import pandas as pd
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


class Regression:
    """
    Calculates evaluation metrics for regression models.
    """

    @staticmethod
    def calculate_metrics(
        y_true: t.Union[np.ndarray, pd.Series],
        y_pred: t.Union[np.ndarray, pd.Series],
        reference_pred: t.Optional[t.Union[np.ndarray, pd.Series]] = None,
    ) -> dict:
        """
        Calculate multiple evaluation metrics for regression.

        Args:
            y_true: Ground truth (correct) target values
            y_pred: Predicted values
            reference_pred: Predictions of a reference model to compare
                against (optional). When given, agreement metrics between
                ``reference_pred`` and ``y_pred`` are added to the result.

        Returns:
            dict: Dictionary containing calculated metrics
        """
        metrics = {}

        # Basic metrics
        metrics['mse'] = float(mean_squared_error(y_true, y_pred))
        metrics['rmse'] = float(np.sqrt(metrics['mse']))
        metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
        metrics['r2'] = float(r2_score(y_true, y_pred))
        metrics['explained_variance'] = float(
            explained_variance_score(y_true, y_pred)
        )

        # Agreement with a reference model, when one is supplied
        if reference_pred is not None:
            try:
                # Ensure we're working with numpy arrays
                if isinstance(reference_pred, pd.Series):
                    reference_pred = reference_pred.values
                if isinstance(y_pred, pd.Series):
                    y_pred = y_pred.values

                # R² between the reference predictions and this model's
                metrics['reference_r2'] = float(
                    r2_score(reference_pred, y_pred)
                )

                # MSE between the reference predictions and this model's
                metrics['reference_mse'] = float(
                    mean_squared_error(reference_pred, y_pred)
                )

                # Correlation between the reference predictions and this model's
                metrics['reference_corr'] = float(
                    np.corrcoef(reference_pred, y_pred)[0, 1]
                )

            except Exception as e:
                print(f'Error calculating comparison metrics: {str(e)}')
                metrics['reference_r2'] = None
                metrics['reference_mse'] = None
                metrics['reference_corr'] = None

        return metrics

    @staticmethod
    def calculate_metrics_from_predictions(
        data: pd.DataFrame,
        target_column: str,
        pred_column: str,
        reference_pred_column: t.Optional[str] = None,
    ) -> dict:
        """
        Calculates metrics using DataFrame columns.

        Args:
            data: DataFrame containing the predictions
            target_column: Name of the column with ground truth values
            pred_column: Name of the column with predictions
            reference_pred_column: Name of the column holding a reference
                model's predictions to compare against (optional)

        Returns:
            dict: Dictionary containing the calculated metrics
        """
        y_true = data[target_column]
        y_pred = data[pred_column]
        reference_pred = (
            data[reference_pred_column] if reference_pred_column else None
        )

        return Regression.calculate_metrics(y_true, y_pred, reference_pred)
