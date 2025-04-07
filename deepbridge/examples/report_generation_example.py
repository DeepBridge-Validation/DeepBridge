"""
Example of generating HTML reports from experiment test results.
This demonstrates how to use the report_from_results function to convert test results into HTML reports.
"""

import os
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

# Import the necessary modules
from deepbridge.core.experiment.experiment import Experiment
from deepbridge.core.db_data import DBDataset
from deepbridge.utils.html_report_generator import generate_report_from_results

def main():
    """Demonstrate how to generate reports from experiment results."""
    print("DeepBridge - Report Generation Example")
    print("=====================================")
    
    # Step 1: Create a synthetic dataset
    print("\nCreating synthetic dataset...")
    X, y = make_classification(
        n_samples=1000, 
        n_features=10, 
        n_informative=5,
        n_redundant=2,
        random_state=42
    )
    
    # Convert to pandas DataFrame and Series
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="target")
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.3, random_state=42
    )
    
    # Step 2: Train a model
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print(f"Model accuracy on test set: {model.score(X_test, y_test):.4f}")
    
    # Step 3: Create a DBDataset
    print("\nCreating DBDataset...")
    # Create dataframes with target column
    train_df = X_train.copy()
    train_df["target"] = y_train
    test_df = X_test.copy()
    test_df["target"] = y_test
    
    dataset = DBDataset(
        train_data=train_df,
        test_data=test_df,
        target_column="target",
        model=model
    )
    
    # Step 4: Create an Experiment object
    print("\nCreating Experiment object...")
    try:
        experiment = Experiment(
            dataset=dataset,
            experiment_type="binary_classification",
            tests=["robustness", "uncertainty"]
        )
        print("✅ Experiment created successfully!")
        
        # Step 5: Run tests
        print("\nRunning tests with 'quick' configuration...")
        results = experiment.run_tests("quick")
        print("✅ Tests executed successfully!")
        
        # Step 6: Generate report using automatic method (from experiment)
        print("\nGenerating report using experiment's save_report method...")
        try:
            # Try to use the built-in save_report method if available
            report_path = results.save_report("./experiment_auto_report.html")
            print(f"✅ Report saved to: {report_path}")
        except (AttributeError, Exception) as e:
            print(f"⚠️  Could not use built-in save_report method: {str(e)}")
            print("   This is normal if you're using the dictionary-based experiment API.")
        
        # Step 7: Generate report using standalone function
        print("\nGenerating HTML report from results dictionary...")
        output_dir = os.path.dirname(os.path.abspath(__file__))
        report_path = generate_report_from_results(
            results,
            os.path.join(output_dir, "experiment_report.html"),
            "Model Validation Report"
        )
        print(f"✅ Report saved to: {report_path}")
        
    except Exception as e:
        print(f"❌ Error executing experiment: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\nExample completed.")


if __name__ == "__main__":
    main()