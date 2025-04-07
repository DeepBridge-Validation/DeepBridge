import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from deepbridge.core.experiment.experiment import Experiment
from deepbridge.core.db_data import DBDataset
from sklearn.linear_model import LogisticRegression

class TestExperiment(unittest.TestCase):
    """Test the Experiment class functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple dataset
        X = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 100),
            'feature_2': np.random.normal(0, 1, 100),
            'feature_3': np.random.normal(0, 1, 100),
            'feature_4': np.random.normal(0, 1, 100),
            'feature_5': np.random.normal(0, 1, 100),
        })
        y = pd.Series((X['feature_1'] + X['feature_2'] > 0).astype(int))
        
        # Train a simple model
        model = LogisticRegression()
        model.fit(X, y)
        
        # Create a DBDataset
        self.dataset = DBDataset(
            feature_data=X,
            target_data=y,
            model=model,
            model_type="binary_classification"
        )
    
    @patch('deepbridge.core.experiment.test_runner.TestRunner.run_tests')
    def test_experiment_with_suite_parameter(self, mock_run_tests):
        """Test that providing the suite parameter automatically runs tests."""
        # Setup mock
        mock_run_tests.return_value = {'test_results': 'mocked_results'}
        
        # Create experiment with suite parameter
        experiment = Experiment(
            dataset=self.dataset,
            experiment_type="binary_classification",
            tests=['robustness'],
            suite='quick'
        )
        
        # Verify that run_tests was called with 'quick'
        mock_run_tests.assert_called_once_with('quick')
        
        # Verify that test_results contains the mocked results
        self.assertEqual(experiment.test_results, {'test_results': 'mocked_results'})
    
    @patch('deepbridge.core.experiment.test_runner.TestRunner')
    def test_experiment_with_feature_select_parameter(self, MockTestRunner):
        """Test that feature_select parameter is passed to TestRunner."""
        # Create a mock TestRunner instance
        mock_test_runner_instance = MagicMock()
        MockTestRunner.return_value = mock_test_runner_instance
        
        # Selected features
        features_select = ['feature_1', 'feature_3', 'feature_5']
        
        # Create experiment with features_select parameter
        experiment = Experiment(
            dataset=self.dataset,
            experiment_type="binary_classification",
            tests=['robustness'],
            features_select=features_select
        )
        
        # Verify that TestRunner was initialized with the correct features_select
        MockTestRunner.assert_called_once()
        _, kwargs = MockTestRunner.call_args
        self.assertEqual(kwargs['features_select'], features_select)
    
    @patch('deepbridge.core.experiment.test_runner.TestRunner.run_tests')
    def test_experiment_with_both_parameters(self, mock_run_tests):
        """Test that both features_select and suite parameters work together."""
        # Setup mock
        mock_run_tests.return_value = {'test_results': 'mocked_results'}
        
        # Selected features
        features_select = ['feature_2', 'feature_4']
        
        # Create experiment with both parameters
        experiment = Experiment(
            dataset=self.dataset,
            experiment_type="binary_classification",
            tests=['robustness'],
            features_select=features_select,
            suite='medium'
        )
        
        # Verify that run_tests was called with 'medium'
        mock_run_tests.assert_called_once_with('medium')
        
        # Verify that test_results contains the mocked results
        self.assertEqual(experiment.test_results, {'test_results': 'mocked_results'})
        
        # Verify that features_select is stored correctly
        self.assertEqual(experiment.features_select, features_select)

if __name__ == '__main__':
    unittest.main()