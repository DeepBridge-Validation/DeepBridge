"""
Example script demonstrating how to run a robustness experiment and generate a report.
This example shows how to use feature_subset to perturb only specific features
and generate a comprehensive HTML report with visualizations.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Import DeepBridge components
from deepbridge.utils.dataset_factory import DBDatasetFactory
from deepbridge.core.experiment.experiment import Experiment
from deepbridge.utils.robustness import (
    run_robustness_tests, 
    plot_robustness_results, 
    compare_models_robustness
)

def main():
    print("Generating synthetic dataset...")
    # Create synthetic classification dataset
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_informative=10,
        n_redundant=5, 
        n_classes=2, 
        random_state=42
    )
    
    # Convert to dataframe
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42
    )
    
    # Define a subset of features to perturb
    # We'll select a mix of informative and non-informative features
    feature_subset = ['feature_3', 'feature_7', 'feature_12', 'feature_18']
    
    print("Training models...")
    # Train primary model - Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    print(f"RandomForest accuracy on test set: {rf_model.score(X_test, y_test):.4f}")
    
    # Train alternative models
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)
    print(f"GradientBoosting accuracy on test set: {gb_model.score(X_test, y_test):.4f}")
    
    lr_model = LogisticRegression(max_iter=200, random_state=42)
    lr_model.fit(X_train, y_train)
    print(f"LogisticRegression accuracy on test set: {lr_model.score(X_test, y_test):.4f}")
    
    # Create a DBDataset for the primary model
    print("Creating datasets...")
    primary_dataset = DBDatasetFactory.create_from_model(
        model=rf_model,
        train_data=X_train,
        train_target=y_train,
        test_data=X_test,
        test_target=y_test,
        problem_type='binary_classification'
    )
    
    # -----------------------------------------------------
    # Method 1: Using Experiment with feature subset
    # -----------------------------------------------------
    print("\nMethod 1: Using Experiment with feature subset")
    
    # Create experiment with feature_subset
    experiment = Experiment(
        dataset=primary_dataset,
        experiment_type="binary_classification", 
        tests=['robustness'],
        feature_subset=feature_subset,
        # Automatically runs the tests
        config_name="quick"
    )
    
    # Add alternative models to the experiment
    experiment.alternative_models = {
        "GBM": gb_model,
        "LOGISTIC_REGRESSION": lr_model
    }
    
    # Get the test results
    results = experiment.test_results
    
    # Print some information from the results
    print("\nRobustness Test Results:")
    
    # Get primary model results
    primary_results = results.get('robustness', {}).get('primary_model', {})
    print(f"Primary model base score: {primary_results.get('base_score', 0):.3f}")
    print(f"Primary model average raw impact: {primary_results.get('avg_raw_impact', 0):.3f}")
    print(f"Primary model average quantile impact: {primary_results.get('avg_quantile_impact', 0):.3f}")
    robustness_score = 1 - primary_results.get('avg_overall_impact', 0)
    print(f"Primary model robustness score: {robustness_score:.3f}")
    
    # Print feature subset information
    if feature_subset:
        print(f"\nFeatures perturbed: {', '.join(feature_subset)}")
    
    # Print feature importance
    print("\nFeature Importance:")
    feature_importance = primary_results.get('feature_importance', {})
    sorted_features = sorted(
        [(f, i) for f, i in feature_importance.items() if f != '_detailed_results'],
        key=lambda x: abs(x[1]),
        reverse=True
    )
    
    for feature, importance in sorted_features[:5]:
        print(f"  {feature}: {importance:.3f}")
    
    # Generate and save reports for all models
    print("\nGenerating robustness reports...")
    
    # Generate primary model report
    primary_report_path = experiment.save_report("robustness", "robustness_report_main_model.html")
    print(f"Primary model report saved to: {primary_report_path}")
    
    # Generate reports for alternative models
    for model_name, model in experiment.alternative_models.items():
        # Create a temporary dataset with this model
        alt_dataset = DBDatasetFactory.create_from_model(
            model=model,
            train_data=X_train,
            train_target=y_train,
            test_data=X_test,
            test_target=y_test,
            problem_type='binary_classification'
        )
        
        # Create a temporary experiment
        alt_experiment = Experiment(
            dataset=alt_dataset,
            experiment_type="binary_classification", 
            tests=['robustness'],
            feature_subset=feature_subset,
            config_name="quick"  # Automatically runs the tests
        )
        
        # Generate and save report
        alt_report_path = alt_experiment.save_report("robustness", f"robustness_report_{model_name}.html")
        print(f"{model_name} report saved to: {alt_report_path}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()