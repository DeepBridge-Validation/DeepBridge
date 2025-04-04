"""
Example script demonstrating how to use the standard implementations of BaseReporter,
BaseProcessor, and BaseGenerator.

This example creates a synthetic dataset, processes it, and generates reports.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Import DeepBridge components
from deepbridge.core import StandardProcessor
from deepbridge.synthetic import StandardGenerator
from deepbridge.reporting import StandardReporter

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
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    
    print(f"Dataset shape: {df.shape}")
    print(f"Sample data:\n{df.head()}")
    
    # Split into train/test
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    
    # ---- Standard Processor Example ----
    print("\n=== Using StandardProcessor ===")
    processor = StandardProcessor(
        verbose=True, 
        scaler_type='standard', 
        handle_missing=True,
        handle_outliers=True,
        categorical_encoding='onehot'
    )
    
    # Process data with automatic column type inference
    processed_data = processor.process(
        train_df,
        fit=True,
        target_column='target'
    )
    
    print(f"Processed data shape: {processed_data.shape}")
    print(f"Numerical columns: {processor.numerical_columns}")
    print(f"Categorical columns: {processor.categorical_columns}")
    
    # Process test data using the fitted processor
    processed_test = processor.process(test_df, target_column='target')
    
    # ---- Standard Generator Example ----
    print("\n=== Using StandardGenerator ===")
    generator = StandardGenerator(
        random_state=42,
        verbose=True,
        method='gmm',  # Gaussian Mixture Model
        n_components=3,
        preserve_correlations=True,
        outlier_rate=0.05
    )
    
    # Fit generator to the training data
    generator.fit(
        train_df,
        target_column='target'
    )
    
    # Generate synthetic data
    synthetic_data = generator.generate(500, noise_level=0.1)
    
    print(f"Synthetic data shape: {synthetic_data.shape}")
    print(f"Synthetic data sample:\n{synthetic_data.head()}")
    
    # Compare distributions
    print("\nComparison of mean values:")
    real_means = train_df.mean(numeric_only=True)
    synthetic_means = synthetic_data.mean(numeric_only=True)
    
    comparison = pd.DataFrame({
        'Real': real_means,
        'Synthetic': synthetic_means,
        'Difference': abs(real_means - synthetic_means)
    })
    print(comparison)
    
    # ---- Standard Reporter Example ----
    print("\n=== Using StandardReporter ===")
    
    # Create reporter
    reporter = StandardReporter()
    
    # Prepare data for report
    report_data = {
        'dataset_info': {
            'original_shape': train_df.shape,
            'processed_shape': processed_data.shape,
            'synthetic_shape': synthetic_data.shape,
            'numerical_features': processor.numerical_columns,
            'categorical_features': processor.categorical_columns
        },
        'comparison': comparison.to_dict(),
        'samples': {
            'original': train_df.head(5).to_dict(),
            'processed': processed_data.head(5).to_dict(),
            'synthetic': synthetic_data.head(5).to_dict()
        },
        'generation_method': generator.method,
        'processing_config': {
            'scaler_type': processor.scaler_type,
            'handle_outliers': processor.handle_outliers,
            'categorical_encoding': processor.categorical_encoding
        }
    }
    
    # Generate basic report (without template)
    report_content = reporter._generate_data_blocks(report_data)
    
    # Save report to file
    report_path = 'standard_components_report.html'
    with open(report_path, 'w') as f:
        f.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>DeepBridge Standard Components Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .section {{ margin-bottom: 20px; padding: 15px; border: 1px solid #ddd; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                h1, h2 {{ color: #2c3e50; }}
            </style>
        </head>
        <body>
            <h1>DeepBridge Standard Components Report</h1>
            {report_content}
        </body>
        </html>
        """)
    
    print(f"Report saved to {report_path}")
    
if __name__ == "__main__":
    main()