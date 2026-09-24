"""
Comprehensive tests for Regression metrics calculator.

This test suite validates:
1. calculate_metrics - basic regression metrics calculation
2. calculate_metrics - with a reference model's predictions for comparison
3. calculate_metrics_from_predictions - DataFrame-based calculation
4. Error handling for reference comparison metrics
5. Edge cases

Coverage Target: ~100%
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

from deepbridge.metrics.regression import Regression


# Test error handling
def test_error_handling():
    """Test error handling in reference model comparison"""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
    reference_pred = "invalid"  # Will cause error

    with patch('builtins.print') as mock_print:
        metrics = Regression.calculate_metrics(y_true, y_pred, reference_pred)

        assert metrics['reference_r2'] is None
        assert metrics['reference_mse'] is None
        assert metrics['reference_corr'] is None
        assert mock_print.called


def test_calculate_from_dataframe_with_empty_reference():
    """Test with empty reference column name"""
    df = pd.DataFrame({
        'true': [1.0, 2.0, 3.0],
        'pred': [1.1, 2.1, 2.9]
    })

    metrics = Regression.calculate_metrics_from_predictions(
        df,
        target_column='true',
        pred_column='pred',
        reference_pred_column=''
    )

    assert 'mse' in metrics



def test_calculate_metrics_with_pandas_series_reference():
    """Test calculate_metrics with pandas Series for reference predictions."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
    reference_pred = pd.Series([1.05, 2.05, 3.05, 4.05, 5.05])  # Pandas Series

    metrics = Regression.calculate_metrics(y_true, y_pred, reference_pred)

    assert "reference_r2" in metrics
    assert "reference_mse" in metrics
    assert "reference_corr" in metrics
    assert metrics["reference_r2"] is not None


def test_calculate_metrics_with_pandas_series_pred_and_reference():
    """Test with both y_pred and reference_pred as pandas Series."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = pd.Series([1.1, 2.1, 2.9, 4.2, 4.8])  # Pandas Series
    reference_pred = pd.Series([1.05, 2.05, 3.05, 4.05, 5.05])  # Pandas Series

    metrics = Regression.calculate_metrics(y_true, y_pred, reference_pred)

    assert "reference_r2" in metrics
    assert "reference_mse" in metrics
    assert "reference_corr" in metrics
    assert isinstance(metrics["reference_r2"], float)
    assert isinstance(metrics["reference_mse"], float)
    assert isinstance(metrics["reference_corr"], float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
