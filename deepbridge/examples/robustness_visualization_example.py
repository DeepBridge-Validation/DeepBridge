"""
Example script demonstrating how to use the simplified RobustnessSuite.

This example creates a synthetic dataset, trains a Random Forest classifier,
and then tests the model's robustness against Gaussian noise and Quantile perturbation.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Import DeepBridge components
from deepbridge.core.db_data import DBDataset
from deepbridge.utils.robustness import run_robustness_tests

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
    
    print("Training Random Forest classifier...")
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print(f"Model accuracy on test set: {model.score(X_test, y_test):.4f}")
    
    # Create DBDataset
    dataset = DBDataset(
        train_data=train_df,
        test_data=test_df,
        target_column="target",
        model=model
    )
    
    # Run robustness tests
    print("\nRunning quick robustness tests...")
    quick_results = run_robustness_tests(dataset, config_name="quick", verbose=True)
    
    print("\nRunning full robustness tests...")
    full_results = run_robustness_tests(dataset, config_name="full", verbose=True)
    
    # Save a report
    from deepbridge.validation.wrappers.robustness_suite import RobustnessSuite
    suite = RobustnessSuite(dataset, verbose=True)
    suite.config("full").run()
    suite.save_report("robustness_report.txt")
    print("\nSaved report to robustness_report.txt")

if __name__ == "__main__":
    main()