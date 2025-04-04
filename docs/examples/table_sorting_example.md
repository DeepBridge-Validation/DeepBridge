# Table Sorting Example

This page demonstrates the table sorting functionality in DeepBridge documentation.

## Model Comparison

Click on any column header to sort the table.

| Model | Accuracy | F1 Score | Training Time (s) | Model Size (MB) |
| ----- | -------- | -------- | ----------------- | --------------- |
| LSTM | 89.3% | 0.87 | 325.4 | 24.5 |
| GRU | 88.7% | 0.85 | 217.8 | 18.2 |
| Transformer | 92.1% | 0.91 | 542.3 | 47.3 |
| Decision Tree | 82.4% | 0.80 | 8.5 | 2.1 |
| Random Forest | 86.9% | 0.85 | 32.7 | 15.8 |
| Logistic Regression | 77.2% | 0.75 | 4.2 | 0.8 |
| XGBoost | 90.3% | 0.89 | 45.1 | 12.4 |
| LightGBM | 89.8% | 0.88 | 28.6 | 9.3 |

## Version Comparison

This table demonstrates semantic versioning sort capability.

| Version | Release Date | Features | Breaking Changes |
| ------- | ------------ | -------- | --------------- |
| 1.0.0 | 2023-01-15 | Base implementation | None |
| 1.1.0 | 2023-02-28 | Added synthetic data generation | None |
| 1.1.1 | 2023-03-10 | Bug fixes | None |
| 1.2.0 | 2023-04-22 | Added robustness testing | None |
| 2.0.0 | 2023-06-30 | New architecture | API changes |
| 2.0.1 | 2023-07-15 | Bug fixes | None |
| 2.1.0 | 2023-08-20 | Added distillation module | None |
| 2.2.0-alpha | 2023-09-05 | Experimental features | Potential instability |
| 2.2.0-beta | 2023-09-25 | Beta features | Potential instability |
| 2.2.0 | 2023-10-15 | New features | None |

## API Components

The table below lists the main API components of DeepBridge.

| Component | Category | Description | Since Version |
| --------- | -------- | ----------- | ------------ |
| Experiment | Core | Main entry point for experiments | 1.0.0 |
| DBDataset | Core | Data handling and augmentation | 1.0.0 |
| BaseProcessor | Core | Abstract base for all processors | 2.0.0 |
| StandardProcessor | Core | Concrete implementation with standard behavior | 2.0.0 |
| ModelManager | Managers | Handles model creation and registry | 2.0.0 |
| RobustnessManager | Managers | Conducts robustness testing | 1.2.0 |
| UncertaintyManager | Managers | Evaluates model uncertainty | 2.1.0 |
| HyperparameterManager | Managers | Manages hyperparameter optimization | 1.1.0 |
| TestRunner | Testing | Coordinates test execution | 2.0.0 |
| VisualizationManager | Visualization | Creates visualizations | 2.0.0 |
| AutoDistiller | Distillation | Automates model distillation | 2.1.0 |
| StandardSynthesizer | Synthetic | Generates synthetic data | 1.1.0 |

## File Sizes

This table demonstrates sorting file sizes correctly.

| Dataset | Original Size | Compressed Size | Format |
| ------- | ------------- | --------------- | ------ |
| MNIST | 11.8 MB | 4.2 MB | GZ |
| ImageNet | 155.8 GB | 68.4 GB | TAR |
| CIFAR-10 | 170.5 MB | 132.4 MB | ZIP |
| Adult Census | 3.8 MB | 1.2 MB | GZ |
| Movie Reviews | 22.5 MB | 8.3 MB | GZ |
| Wikipedia Text | 14.5 GB | 5.2 GB | BZ2 |
| Twitter Sentiment | 850.3 MB | 245.7 MB | ZIP |
| Medical Images | 1.8 TB | 780.5 GB | TAR |