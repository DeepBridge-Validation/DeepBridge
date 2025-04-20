import unittest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

from DeepBridge.deepbridge.validation import RobustnessTest, RobustnessScore
from deepbridge.visualization import RobustnessViz
from deepbridge.utils.robustness_report_generator import RobustnessReportGenerator
from deepbridge.utils import generate_robustness_report

class TestRobustnessFeatures(unittest.TestCase):
    """Test suite for the robustness testing features."""

    @classmethod
    def setUpClass(cls):
        """Set up test data and models once for all tests."""
        # Generate synthetic data
        X, y = make_classification(
            n_samples=100,  # Small dataset for quick tests
            n_features=5,
            n_informative=3,
            n_redundant=1,
            random_state=42
        )
        
        # Add column names
        cls.X = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(X.shape[1])])
        cls.y = y
        
        # Train models
        cls.rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
        cls.lr_model = LogisticRegression(random_state=42)
        
        cls.rf_model.fit(cls.X, cls.y)
        cls.lr_model.fit(cls.X, cls.y)
        
        # Create a dictionary of models
        cls.models = {
            'Random Forest': cls.rf_model,
            'Logistic Regression': cls.lr_model
        }

    def test_robustness_test_initialization(self):
        """Test that RobustnessTest can be initialized."""
        robustness_test = RobustnessTest()
        self.assertIsNotNone(robustness_test)
        self.assertIn('raw', robustness_test.perturbation_methods)
        self.assertIn('quantile', robustness_test.perturbation_methods)
        self.assertIn('categorical', robustness_test.perturbation_methods)

    def test_evaluate_robustness(self):
        """Test the evaluate_robustness method."""
        robustness_test = RobustnessTest()
        
        results = robustness_test.evaluate_robustness(
            models=self.models,
            X=self.X,
            y=self.y,
            perturb_method='raw',
            perturb_sizes=[0.1, 0.5],
            metric='AUC',
            n_iterations=3,  # Small number for quick tests
            random_state=42
        )
        
        # Check structure of results
        self.assertIn('Random Forest', results)
        self.assertIn('Logistic Regression', results)
        
        for model_name, model_results in results.items():
            self.assertIn('perturb_sizes', model_results)
            self.assertIn('mean_scores', model_results)
            self.assertIn('worst_scores', model_results)
            self.assertIn('all_scores', model_results)
            
            self.assertEqual(len(model_results['perturb_sizes']), 2)
            self.assertEqual(len(model_results['mean_scores']), 2)
            self.assertEqual(len(model_results['worst_scores']), 2)
            self.assertEqual(len(model_results['all_scores']), 2)

    def test_feature_importance_analysis(self):
        """Test the feature importance analysis."""
        robustness_test = RobustnessTest()
        
        feature_importance = robustness_test.analyze_feature_importance(
            model=self.rf_model,
            X=self.X,
            y=self.y,
            perturb_method='raw',
            perturb_size=0.5,
            metric='AUC',
            n_iterations=3,
            random_state=42
        )
        
        # Check structure of results
        self.assertIn('base_score', feature_importance)
        self.assertIn('feature_names', feature_importance)
        self.assertIn('feature_impacts', feature_importance)
        self.assertIn('normalized_impacts', feature_importance)
        self.assertIn('sorted_features', feature_importance)
        self.assertIn('sorted_impacts', feature_importance)
        
        # Check that feature_names match the input data
        self.assertEqual(set(feature_importance['feature_names']), set(self.X.columns))
        
        # Check lengths
        self.assertEqual(len(feature_importance['feature_names']), self.X.shape[1])
        self.assertEqual(len(feature_importance['feature_impacts']), self.X.shape[1])
        self.assertEqual(len(feature_importance['normalized_impacts']), self.X.shape[1])
        self.assertEqual(len(feature_importance['sorted_features']), self.X.shape[1])
        self.assertEqual(len(feature_importance['sorted_impacts']), self.X.shape[1])

    def test_robustness_score(self):
        """Test the robustness score calculation."""
        robustness_test = RobustnessTest()
        
        results = robustness_test.evaluate_robustness(
            models=self.models,
            X=self.X,
            y=self.y,
            perturb_method='raw',
            perturb_sizes=[0.1, 0.5, 1.0],
            metric='AUC',
            n_iterations=3,
            random_state=42
        )
        
        robustness_indices = RobustnessScore.calculate_robustness_index(
            results=results,
            metric='AUC'
        )
        
        # Check that indices are calculated for all models
        self.assertEqual(set(robustness_indices.keys()), set(self.models.keys()))
        
        # Check that indices are between 0 and 1
        for index in robustness_indices.values():
            self.assertGreaterEqual(index, 0.0)
            self.assertLessEqual(index, 1.0)

    def test_visualization_functions(self):
        """Test that visualization functions work."""
        robustness_test = RobustnessTest()
        
        results = robustness_test.evaluate_robustness(
            models=self.models,
            X=self.X,
            y=self.y,
            perturb_method='raw',
            perturb_sizes=[0.1, 0.5],
            metric='AUC',
            n_iterations=3,
            random_state=42
        )
        
        # Test models comparison plot
        fig1 = RobustnessViz.plot_models_comparison(
            results=results,
            metric_name='AUC Score'
        )
        self.assertIsNotNone(fig1)
        
        # Test boxplot performance
        fig2 = RobustnessViz.plot_boxplot_performance(
            results=results,
            model_name='Random Forest',
            metric_name='AUC Score'
        )
        self.assertIsNotNone(fig2)
        
        # Clean up
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_report_generator(self):
        """Test the report generator."""
        robustness_test = RobustnessTest()
        
        results = robustness_test.evaluate_robustness(
            models=self.models,
            X=self.X,
            y=self.y,
            perturb_method='raw',
            perturb_sizes=[0.1, 0.5],
            metric='AUC',
            n_iterations=3,
            random_state=42
        )
        
        robustness_indices = RobustnessScore.calculate_robustness_index(
            results=results,
            metric='AUC'
        )
        
        # Test report generation
        report_generator = RobustnessReportGenerator()
        
        html_report = report_generator.generate_report(
            robustness_results=results,
            robustness_indices=robustness_indices
        )
        
        self.assertIsInstance(html_report, str)
        self.assertGreater(len(html_report), 0)
        self.assertIn('<!DOCTYPE html>', html_report)
        self.assertIn('Random Forest', html_report)
        self.assertIn('Logistic Regression', html_report)

    def test_integrated_report_function(self):
        """Test the integrated report generation function."""
        html_report, analysis_results = generate_robustness_report(
            models=self.models,
            X=self.X,
            y=self.y,
            perturb_method='raw',
            perturb_sizes=[0.1, 0.5],
            metric='AUC',
            n_iterations=3,
            analyze_features=True,
            compare_methods=False,  # Skip for speed
            random_state=42
        )
        
        self.assertIsInstance(html_report, str)
        self.assertGreater(len(html_report), 0)
        
        self.assertIn('robustness_results', analysis_results)
        self.assertIn('robustness_indices', analysis_results)
        self.assertIn('feature_importance_results', analysis_results)

if __name__ == '__main__':
    unittest.main()