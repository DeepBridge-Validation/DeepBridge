"""
Example script demonstrating how to use the enhanced robustness tests.

This example creates synthetic data, trains multiple models, and runs robustness 
tests on all models with enhanced visualization capabilities.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Import DeepBridge components
from deepbridge.core.db_data import DBDataset
from deepbridge.core.experiment import Experiment
from deepbridge.utils.model_registry import ModelType
from deepbridge.utils.robustness import (
    run_robustness_tests, 
    plot_robustness_results, 
    compare_models_robustness,
    robustness_report_to_html
)

def main():
    print("Generating synthetic dataset...")
    # Create synthetic classification dataset
    X, y = make_classification(
        n_samples=1000, 
        n_features=10, 
        n_informative=5,
        n_redundant=2,
        random_state=42
    )
    
    # Convert to dataframe
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="target")
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.3, random_state=42
    )
    
    # Create complete dataframes with target
    train_df = X_train.copy()
    train_df["target"] = y_train
    test_df = X_test.copy()
    test_df["target"] = y_test
    
    print("Training Random Forest as primary model...")
    # Train primary model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print(f"Primary model accuracy on test set: {model.score(X_test, y_test):.4f}")
    
    # Create DBDataset with the primary model
    rf_dataset = DBDataset(
        train_data=train_df,
        test_data=test_df,
        target_column="target",
        model=model
    )
    
    # Create alternative models
    print("Training alternative models...")
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)
    print(f"GBM accuracy on test set: {gb_model.score(X_test, y_test):.4f}")
    
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train, y_train)
    print(f"LogisticRegression accuracy on test set: {lr_model.score(X_test, y_test):.4f}")
    
    # Create datasets for alternative models
    gb_dataset = DBDataset(
        train_data=train_df,
        test_data=test_df,
        target_column="target",
        model=gb_model
    )
    
    lr_dataset = DBDataset(
        train_data=train_df,
        test_data=test_df,
        target_column="target",
        model=lr_model
    )
    
    # -------- Method 1: Using the Experiment class --------
    print("\nMethod 1: Using Experiment class")
    experiment = Experiment(
        dataset=rf_dataset,
        experiment_type="binary_classification", 
        tests=['robustness']
    )
    
    # Add alternative models to the experiment
    experiment.alternative_models = {
        "GBM": gb_model,
        "LOGISTIC_REGRESSION": lr_model
    }
    
    # Run robustness tests with the 'quick' configuration
    print("Running robustness tests through Experiment...")
    exp_results = experiment.run_tests(config_name="quick")
    
    # Print summary
    print("\nExperiment Robustness Results Summary:")
    primary_score = exp_results['robustness']['primary_model'].get('robustness_score', 0)
    print(f"Primary model (RandomForest) robustness score: {primary_score:.3f}")
    
    for model_name, model_results in exp_results['robustness']['alternative_models'].items():
        model_score = model_results.get('robustness_score', 0)
        print(f"{model_name} robustness score: {model_score:.3f}")
    
    # -------- Method 2: Using direct robustness testing --------
    print("\nMethod 2: Using direct robustness testing with enhanced features")
    
    # Run tests on all models
    print("Running tests on all models...")
    rf_results = run_robustness_tests(rf_dataset, config_name="full", metric="AUC")
    gb_results = run_robustness_tests(gb_dataset, config_name="full", metric="AUC")
    lr_results = run_robustness_tests(lr_dataset, config_name="full", metric="AUC")
    
    # Store results in a dictionary for comparison
    all_results = {
        "RandomForest": rf_results,
        "GradientBoosting": gb_results,
        "LogisticRegression": lr_results
    }
    
    # Print summary
    print("\nDirect Testing Robustness Results Summary:")
    for model_name, results in all_results.items():
        print(f"{model_name} robustness score: {results.get('robustness_score', 0):.3f}")
    
    # -------- Generate Plots --------
    print("\nGenerating robustness plots...")
    
    # 1. Plot Random Forest robustness
    print("Creating Random Forest robustness plot...")
    rf_plot = plot_robustness_results(rf_results, plot_type="robustness")
    rf_plot.write_html("rf_robustness.html")
    
    # 2. Plot feature importance 
    print("Creating feature importance plot...")
    importance_plot = plot_robustness_results(rf_results, plot_type="feature_importance")
    importance_plot.write_html("feature_importance.html")
    
    # 3. Plot perturbation methods comparison
    print("Creating methods comparison plot...")
    methods_plot = plot_robustness_results(rf_results, plot_type="methods_comparison")
    methods_plot.write_html("methods_comparison.html")
    
    # 4. Plot model comparison
    print("Creating models comparison plot...")
    models_plot = compare_models_robustness(all_results)
    models_plot.write_html("models_comparison.html")
    
    # Generate HTML reports
    print("\nGenerating HTML reports...")
    for model_name, results in all_results.items():
        html_report = robustness_report_to_html(results)
        with open(f"{model_name}_robustness_report.html", "w") as f:
            f.write(html_report)
        print(f"Report saved to {model_name}_robustness_report.html")
    
    print("\nRun completed successfully!")

if __name__ == "__main__":
    main()