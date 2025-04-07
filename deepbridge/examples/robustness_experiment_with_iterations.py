"""
Example script demonstrating robustness testing with multiple iterations per perturbation level.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from deepbridge.core.db_data import DBDataset
from deepbridge.validation.wrappers.robustness_suite import RobustnessSuite

def create_sample_dataset():
    """Create a sample dataset for demonstration purposes."""
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Generate correlated features
    X = np.random.randn(n_samples, n_features)
    # Add some correlation between features
    X[:, 1] = X[:, 0] * 0.5 + X[:, 1] * 0.5
    X[:, 2] = X[:, 0] * 0.3 + X[:, 1] * 0.3 + X[:, 2] * 0.4
    
    # Create target variable with some relationship to features
    y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.5) > 0.5
    y = y.astype(int)
    
    # Convert to pandas DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='target')
    
    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_series, test_size=0.2, random_state=42)
    
    # Create and train a model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Create DBDataset
    dataset = DBDataset(
        feature_data=X_df,
        target_data=y_series,
        model=model,
        model_type="binary_classification"
    )
    
    return dataset

def main():
    """Run the robustness experiment with varying iterations."""
    print("DeepBridge - Robustness Testing with Multiple Iterations")
    print("======================================================")
    
    # Create dataset
    print("\nCreating sample dataset...")
    dataset = create_sample_dataset()
    
    # Test with different numbers of iterations
    iteration_levels = [1, 3, 10]
    results = {}
    
    for n_iterations in iteration_levels:
        print(f"\nRunning robustness test with {n_iterations} iterations per perturbation level...")
        
        # Create robustness suite with specified iterations
        suite = RobustnessSuite(
            dataset=dataset,
            verbose=True,
            metric='AUC',
            n_iterations=n_iterations
        )
        
        # Run quick test configuration
        results[n_iterations] = suite.config('quick').run()
        
        # Print summary of results
        print(f"\nResults with {n_iterations} iterations:")
        print(f"Base score: {results[n_iterations]['base_score']:.4f}")
        print(f"Average impact: {results[n_iterations]['avg_overall_impact']:.4f}")
        
        # Print standard deviation information if available
        if n_iterations > 1:
            # Extract standard deviations for raw perturbation
            raw_stds = []
            for level, level_data in results[n_iterations]['raw']['by_level'].items():
                if 'overall_result' in level_data and 'std_score' in level_data['overall_result']:
                    std = level_data['overall_result']['std_score']
                    raw_stds.append(std)
                    print(f"  Raw level {level} std: {std:.4f}")
            
            # Extract standard deviations for quantile perturbation
            quantile_stds = []
            for level, level_data in results[n_iterations]['quantile']['by_level'].items():
                if 'overall_result' in level_data and 'std_score' in level_data['overall_result']:
                    std = level_data['overall_result']['std_score']
                    quantile_stds.append(std)
                    print(f"  Quantile level {level} std: {std:.4f}")
            
            # Average standard deviations
            if raw_stds:
                print(f"Average raw std: {np.mean(raw_stds):.4f}")
            if quantile_stds:
                print(f"Average quantile std: {np.mean(quantile_stds):.4f}")
    
    # Compare statistical robustness between different iteration counts
    print("\nComparison of statistical robustness with different iteration counts:")
    print("Higher iterations provide more reliable estimates of robustness")
    print("-" * 60)
    print(f"{'Iterations':<10} {'Avg Raw Impact':<20} {'Avg Quantile Impact':<20}")
    print("-" * 60)
    
    for n_iterations in iteration_levels:
        raw_impact = results[n_iterations]['avg_raw_impact']
        quantile_impact = results[n_iterations]['avg_quantile_impact']
        print(f"{n_iterations:<10} {raw_impact:<20.4f} {quantile_impact:<20.4f}")
    
    print("\nExample completed.")

if __name__ == "__main__":
    main()