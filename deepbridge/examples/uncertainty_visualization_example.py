"""
Example script showing how to generate and view uncertainty visualizations.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import os
import shutil

# Import DeepBridge components
from deepbridge.core.db_data import DBDataset
from deepbridge.validation.wrappers.uncertainty_suite import UncertaintySuite

# Function to create a simple dataset
def create_dataset():
    # Generate a regression dataset
    X, y = make_regression(
        n_samples=1000, 
        n_features=10, 
        n_informative=5, 
        noise=0.1, 
        random_state=42
    )
    
    # Convert to DataFrame for better feature naming
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y_series = pd.Series(y, name='target')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.3, random_state=42
    )
    
    # Train a model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Create a DBDataset instance
    dataset = DBDataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model=model,
        problem_type='regression'
    )
    
    return dataset

def main():
    # Create the dataset
    print("Creating dataset...")
    dataset = create_dataset()
    
    # Initialize the uncertainty suite
    print("Initializing uncertainty suite...")
    uncertainty_suite = UncertaintySuite(dataset, verbose=True)
    
    # Configure and run the tests
    print("Running uncertainty tests...")
    uncertainty_suite.config('quick')  # Run a quick test with alpha=0.1 and alpha=0.2
    results = uncertainty_suite.run()
    
    # Save the report
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                             'examples', 'Synthetic')
    report_path = os.path.join(output_dir, 'uncertainty_report.html')
    
    print(f"Saving uncertainty report to {report_path}...")
    uncertainty_suite.save_report(report_path)
    
    # Also copy the standalone visualization example
    vis_src = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                          'examples', 'Synthetic', 'uncertainty_visualizations.html')
    
    if os.path.exists(vis_src):
        print("Standalone visualization example already exists.")
    else:
        print("Creating standalone visualization example...")
        # Create a simple example if it doesn't exist
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Uncertainty Visualizations</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
</head>
<body>
    <h1>Uncertainty Visualizations</h1>
    <p>Open the uncertainty_report.html file to see the complete visualizations.</p>
</body>
</html>
        """
        with open(vis_src, 'w') as f:
            f.write(html_content)
    
    print(f"Done! Reports saved to {output_dir}")
    print("To view the report, open the HTML file in a web browser.")

if __name__ == "__main__":
    main()