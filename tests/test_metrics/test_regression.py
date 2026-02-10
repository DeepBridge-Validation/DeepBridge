"""
Comprehensive tests for Regression metrics calculator.

This test suite validates:
1. calculate_metrics - basic regression metrics calculation
2. calculate_metrics - with teacher predictions for distillation  
3. calculate_metrics_from_predictions - DataFrame-based calculation
4. Error handling for teacher comparison metrics
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
    """Test error handling in teacher comparison"""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
    teacher_pred = "invalid"  # Will cause error

    with patch('builtins.print') as mock_print:
        metrics = Regression.calculate_metrics(y_true, y_pred, teacher_pred)
        
        assert metrics['teacher_student_r2'] is None
        assert metrics['teacher_student_mse'] is None
        assert metrics['teacher_student_corr'] is None
        assert mock_print.called


def test_calculate_from_dataframe_with_empty_teacher():
    """Test with empty teacher column name"""
    df = pd.DataFrame({
        'true': [1.0, 2.0, 3.0],
        'pred': [1.1, 2.1, 2.9]
    })

    metrics = Regression.calculate_metrics_from_predictions(
        df,
        target_column='true',
        pred_column='pred',
        teacher_pred_column=''
    )

    assert 'mse' in metrics



def test_calculate_metrics_with_pandas_series_teacher():
    """Test calculate_metrics with pandas Series for teacher predictions."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
    teacher_pred = pd.Series([1.05, 2.05, 3.05, 4.05, 5.05])  # Pandas Series

    metrics = Regression.calculate_metrics(y_true, y_pred, teacher_pred)

    assert "teacher_student_r2" in metrics
    assert "teacher_student_mse" in metrics
    assert "teacher_student_corr" in metrics
    assert metrics["teacher_student_r2"] is not None


def test_calculate_metrics_with_pandas_series_pred_and_teacher():
    """Test with both y_pred and teacher_pred as pandas Series."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = pd.Series([1.1, 2.1, 2.9, 4.2, 4.8])  # Pandas Series
    teacher_pred = pd.Series([1.05, 2.05, 3.05, 4.05, 5.05])  # Pandas Series

    metrics = Regression.calculate_metrics(y_true, y_pred, teacher_pred)

    assert "teacher_student_r2" in metrics
    assert "teacher_student_mse" in metrics
    assert "teacher_student_corr" in metrics
    assert isinstance(metrics["teacher_student_r2"], float)
    assert isinstance(metrics["teacher_student_mse"], float)
    assert isinstance(metrics["teacher_student_corr"], float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

