"""
Advanced Robustness Testing Examples - DeepBridge

This script demonstrates the advanced robustness analysis capabilities:
1. WeakSpot Detection: Identify regions where model performance degrades
2. Sliced Overfitting Analysis: Detect localized train-test gaps
3. Combined Analysis: Using all robustness tools together

Author: DeepBridge Team
Date: October 2025
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression, make_classification
from sklearn.metrics import r2_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

from deepbridge.core.db_data import DBDataset
from deepbridge.validation.wrappers.robustness_suite import RobustnessSuite

# Set random seed for reproducibility
np.random.seed(42)


def create_synthetic_dataset_with_weakspots():
    """
    Create a synthetic regression dataset with intentional weak regions.

    The model will perform poorly for:
    - High values of feature_1 (> 80th percentile)
    - Low values of feature_2 (< 20th percentile)
    """
    print("Creating synthetic dataset with weak spots...")

    # Generate base data
    n_samples = 1000
    X, y = make_regression(
        n_samples=n_samples,
        n_features=10,
        n_informative=5,
        noise=10,
        random_state=42
    )

    # Convert to DataFrame
    feature_names = [f'feature_{i}' for i in range(10)]
    X_df = pd.DataFrame(X, columns=feature_names)

    # Add intentional weak spots
    # Region 1: High feature_1 values get extra noise
    high_feature1_mask = X_df['feature_1'] > X_df['feature_1'].quantile(0.8)
    y[high_feature1_mask] += np.random.normal(0, 30, size=np.sum(high_feature1_mask))

    # Region 2: Low feature_2 values get systematic bias
    low_feature2_mask = X_df['feature_2'] < X_df['feature_2'].quantile(0.2)
    y[low_feature2_mask] += 40

    print(f"Dataset shape: {X_df.shape}")
    print(f"High feature_1 samples (weak spot): {np.sum(high_feature1_mask)}")
    print(f"Low feature_2 samples (weak spot): {np.sum(low_feature2_mask)}")

    return X_df, y


def create_overfitting_dataset():
    """
    Create a dataset where the model will overfit in specific regions.

    Training data has different distributions in certain feature ranges
    compared to test data.
    """
    print("Creating dataset with localized overfitting patterns...")

    n_samples = 1500
    X, y = make_classification(
        n_samples=n_samples,
        n_features=8,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        class_sep=0.8,
        flip_y=0.1,
        random_state=42
    )

    feature_names = [f'feature_{i}' for i in range(8)]
    X_df = pd.DataFrame(X, columns=feature_names)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.3, random_state=42, stratify=y
    )

    # Add noise to test set in specific regions to simulate distribution shift
    # This will cause localized overfitting
    high_feat0_mask = X_test['feature_0'] > X_test['feature_0'].quantile(0.75)

    # Randomly flip labels in this region for test set
    # Convert y_test to Series to preserve indices
    y_test_series = pd.Series(y_test, index=X_test.index)
    flip_indices = X_test[high_feat0_mask].sample(frac=0.3, random_state=42).index
    y_test_modified = y_test_series.copy()
    y_test_modified[flip_indices] = 1 - y_test_modified[flip_indices]
    y_test_modified = y_test_modified.values  # Convert back to array

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Modified {len(flip_indices)} test labels to simulate distribution shift")

    return X_train, X_test, y_train, y_test_modified


# =============================================================================
# EXAMPLE 1: WeakSpot Detection for Regression
# =============================================================================
def example_1_weakspot_detection():
    """Demonstrate weakspot detection on regression problem."""

    print("\n" + "="*70)
    print("EXAMPLE 1: WeakSpot Detection - Regression")
    print("="*70)

    # Create dataset with weak spots
    X, y = create_synthetic_dataset_with_weakspots()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train model
    print("\nTraining RandomForest model...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Global performance
    train_score = r2_score(y_train, model.predict(X_train))
    test_score = r2_score(y_test, model.predict(X_test))
    print(f"\nGlobal R2 - Train: {train_score:.3f}, Test: {test_score:.3f}")
    print("Global metrics look acceptable, but let's check for weakspots...")

    # Create DBDataset
    # Combine features and target into DataFrames
    train_df = X_train.copy()
    train_df['target'] = y_train
    test_df = X_test.copy()
    test_df['target'] = y_test

    dataset = DBDataset(
        train_data=train_df,
        test_data=test_df,
        target_column='target',
        model=model
    )

    # Initialize RobustnessSuite
    suite = RobustnessSuite(
        dataset=dataset,
        verbose=True,
        metric='mse'
    )

    # Run weakspot detection
    print("\n" + "-"*70)
    print("Running WeakSpot Detection...")
    print("-"*70)

    weakspot_results = suite.run_weakspot_detection(
        X=X_test,
        y=y_test,
        slice_features=['feature_1', 'feature_2', 'feature_3'],
        slice_method='quantile',
        n_slices=10,
        severity_threshold=0.15,
        metric='mae'
    )

    # Analyze top weakspots
    print("\n" + "-"*70)
    print("TOP 3 WEAKSPOTS")
    print("-"*70)

    for i, ws in enumerate(weakspot_results['weakspots'][:3], 1):
        print(f"\n{i}. Feature: {ws['feature']}")
        print(f"   Range: {ws['range_str']}")
        print(f"   Samples: {ws['n_samples']}")
        print(f"   Mean Residual: {ws['mean_residual']:.2f} (global: {ws['global_mean_residual']:.2f})")
        print(f"   Severity: {ws['severity']:.1%} worse than global average")

        if ws['severity'] > 0.5:
            print(f"   ⚠️  CRITICAL: Consider retraining with more data in this region")


# =============================================================================
# EXAMPLE 2: Sliced Overfitting Analysis for Classification
# =============================================================================
def example_2_overfitting_analysis():
    """Demonstrate overfitting analysis on classification problem."""

    print("\n" + "="*70)
    print("EXAMPLE 2: Sliced Overfitting Analysis - Classification")
    print("="*70)

    # Create dataset with localized overfitting
    X_train, X_test, y_train, y_test = create_overfitting_dataset()

    # Train model (intentionally overfit)
    print("\nTraining RandomForest classifier (with some overfitting)...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,  # Deep trees = more prone to overfitting
        min_samples_leaf=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Global performance
    train_score = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    test_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    global_gap = train_score - test_score

    print(f"\nGlobal ROC AUC:")
    print(f"  Train: {train_score:.3f}")
    print(f"  Test:  {test_score:.3f}")
    print(f"  Gap:   {global_gap:.3f} ({global_gap/train_score:.1%})")

    if global_gap < 0.1:
        print("Global gap looks acceptable, but are there localized issues?")

    # Create DBDataset
    # Combine features and target into DataFrames
    train_df = X_train.copy()
    train_df['target'] = y_train
    test_df = X_test.copy()
    test_df['target'] = y_test

    dataset = DBDataset(
        train_data=train_df,
        test_data=test_df,
        target_column='target',
        model=model
    )

    # Initialize RobustnessSuite
    suite = RobustnessSuite(
        dataset=dataset,
        verbose=True
    )

    # Run overfitting analysis
    print("\n" + "-"*70)
    print("Running Sliced Overfitting Analysis...")
    print("-"*70)

    overfit_results = suite.run_overfitting_analysis(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        slice_features=['feature_0', 'feature_1', 'feature_2'],
        n_slices=10,
        slice_method='quantile',
        gap_threshold=0.1,
        metric_func=lambda y_true, y_pred: roc_auc_score(y_true, y_pred)
    )

    # Analyze results
    print("\n" + "-"*70)
    print("ANALYSIS SUMMARY")
    print("-"*70)

    if 'features' in overfit_results:
        # Multiple features analyzed
        print(f"\nWorst Feature: {overfit_results['worst_feature']}")
        print(f"Max Gap: {overfit_results['summary']['global_max_gap']:.3f}")
        print(f"Features with Overfitting: {overfit_results['summary']['features_with_overfitting']}")

        # Show details for worst feature
        worst_feat = overfit_results['worst_feature']
        worst_results = overfit_results['features'][worst_feat]

        print(f"\n{worst_feat} - Top Overfit Slices:")
        for i, s in enumerate(worst_results['overfit_slices'][:3], 1):
            print(f"\n  {i}. Range: {s['range_str']}")
            print(f"     Train Metric: {s['train_metric']:.3f}")
            print(f"     Test Metric:  {s['test_metric']:.3f}")
            print(f"     Gap: {s['gap']:.3f} ({s['gap_percentage']:.1f}%)")
    else:
        # Single feature analyzed
        print(f"\nFeature: {overfit_results['feature']}")
        print(f"Max Gap: {overfit_results['max_gap']:.3f}")
        print(f"Overfit Slices: {overfit_results['summary']['overfit_slices_count']}")


# =============================================================================
# EXAMPLE 3: Combined Analysis - Full Robustness Suite
# =============================================================================
def example_3_combined_analysis():
    """
    Demonstrate using all robustness analysis tools together:
    - Standard robustness tests
    - WeakSpot detection
    - Overfitting analysis
    """

    print("\n" + "="*70)
    print("EXAMPLE 3: Combined Robustness Analysis")
    print("="*70)

    # Create dataset
    X, y = create_synthetic_dataset_with_weakspots()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train model
    print("\nTraining model...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Create DBDataset
    # Combine features and target into DataFrames
    train_df = X_train.copy()
    train_df['target'] = y_train
    test_df = X_test.copy()
    test_df['target'] = y_test

    dataset = DBDataset(
        train_data=train_df,
        test_data=test_df,
        target_column='target',
        model=model
    )

    # Initialize RobustnessSuite
    suite = RobustnessSuite(
        dataset=dataset,
        verbose=True,
        metric='mse'
    )

    # 1. Run standard robustness tests
    print("\n" + "="*70)
    print("STEP 1: Standard Robustness Tests")
    print("="*70)

    robustness_results = suite.config('quick').run()

    print(f"\nBase Score (MSE): {robustness_results['base_score']:.2f}")
    print(f"Average Impact: {robustness_results['avg_overall_impact']:.3f}")

    # 2. Run weakspot detection
    print("\n" + "="*70)
    print("STEP 2: WeakSpot Detection")
    print("="*70)

    weakspot_results = suite.run_weakspot_detection(
        X=X_test,
        y=y_test,
        slice_features=['feature_1', 'feature_2'],
        n_slices=8,
        severity_threshold=0.15
    )

    # 3. Run overfitting analysis
    print("\n" + "="*70)
    print("STEP 3: Overfitting Analysis")
    print("="*70)

    overfit_results = suite.run_overfitting_analysis(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        slice_features=['feature_1', 'feature_2'],
        n_slices=8,
        gap_threshold=0.1
    )

    # Combined summary
    print("\n" + "="*70)
    print("COMBINED ROBUSTNESS ASSESSMENT")
    print("="*70)

    print(f"\n1. Standard Robustness:")
    print(f"   Average Impact: {robustness_results['avg_overall_impact']:.3f}")

    if robustness_results['avg_overall_impact'] < 0.1:
        print("   ✓ Model is robust to perturbations")
    else:
        print("   ⚠️  Model shows sensitivity to perturbations")

    print(f"\n2. WeakSpots:")
    print(f"   Total Found: {weakspot_results['summary']['total_weakspots']}")
    print(f"   Critical (>50% degradation): {weakspot_results['summary']['critical_weakspots']}")

    if weakspot_results['summary']['total_weakspots'] == 0:
        print("   ✓ No significant weakspots detected")
    else:
        print("   ⚠️  Weakspots require attention")

    print(f"\n3. Localized Overfitting:")
    if 'features' in overfit_results:
        total_overfit = overfit_results['summary']['features_with_overfitting']
        print(f"   Features with Overfitting: {total_overfit}")
        print(f"   Max Gap: {overfit_results['summary']['global_max_gap']:.3f}")
    else:
        print(f"   Overfit Slices: {overfit_results['summary']['overfit_slices_count']}")
        print(f"   Max Gap: {overfit_results['max_gap']:.3f}")

    if overfit_results.get('summary', {}).get('features_with_overfitting', 0) == 0:
        print("   ✓ No significant localized overfitting")
    else:
        print("   ⚠️  Localized overfitting detected")

    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)

    # Generate overall recommendation
    issues = []

    if robustness_results['avg_overall_impact'] > 0.1:
        issues.append("high perturbation sensitivity")

    if weakspot_results['summary']['total_weakspots'] > 0:
        issues.append(f"{weakspot_results['summary']['total_weakspots']} weakspots")

    if overfit_results.get('summary', {}).get('features_with_overfitting', 0) > 0:
        issues.append("localized overfitting")

    if not issues:
        print("\n✓ Model passes all robustness checks!")
        print("  Safe to deploy with standard monitoring.")
    else:
        print(f"\n⚠️  Model has robustness issues: {', '.join(issues)}")
        print("\nRecommended Actions:")

        if "high perturbation sensitivity" in ', '.join(issues):
            print("  1. Add regularization or use ensemble methods")

        if "weakspots" in ', '.join(issues):
            print("  2. Collect more data in weak regions")
            print("  3. Consider feature engineering for weak spots")

        if "localized overfitting" in ', '.join(issues):
            print("  4. Reduce model complexity (max_depth, min_samples_leaf)")
            print("  5. Add more training data in overfit regions")


# =============================================================================
# EXAMPLE 4: Direct Usage of WeakspotDetector and OverfitAnalyzer
# =============================================================================
def example_4_direct_usage():
    """Demonstrate direct usage without RobustnessSuite."""

    print("\n" + "="*70)
    print("EXAMPLE 4: Direct Usage of Detectors")
    print("="*70)

    from deepbridge.validation.robustness import WeakspotDetector, OverfitAnalyzer

    # Create simple dataset
    X, y = make_regression(n_samples=500, n_features=5, noise=10, random_state=42)
    X_df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(5)])

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.3, random_state=42
    )

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    # Direct WeakspotDetector usage
    print("\n" + "-"*70)
    print("Using WeakspotDetector directly:")
    print("-"*70)

    detector = WeakspotDetector(
        slice_method='quantile',
        n_slices=8,
        severity_threshold=0.15
    )

    y_pred = model.predict(X_test)

    ws_results = detector.detect_weak_regions(
        X=X_test,
        y_true=y_test,
        y_pred=y_pred,
        slice_features=['feat_0', 'feat_1'],
        metric='mae'
    )

    print(f"Found {ws_results['summary']['total_weakspots']} weakspots")

    # Direct OverfitAnalyzer usage
    print("\n" + "-"*70)
    print("Using OverfitAnalyzer directly:")
    print("-"*70)

    analyzer = OverfitAnalyzer(
        n_slices=8,
        slice_method='quantile',
        gap_threshold=0.1
    )

    of_results = analyzer.compute_gap_by_slice(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        model=model,
        slice_feature='feat_0',
        metric_func=r2_score
    )

    print(f"Max gap: {of_results['max_gap']:.3f}")
    print(f"Overfit slices: {of_results['summary']['overfit_slices_count']}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("ADVANCED ROBUSTNESS TESTING - DeepBridge")
    print("="*70)
    print("\nThis script demonstrates advanced robustness analysis:")
    print("  1. WeakSpot Detection")
    print("  2. Sliced Overfitting Analysis")
    print("  3. Combined Analysis")
    print("  4. Direct API Usage")

    # Run examples
    try:
        example_1_weakspot_detection()
    except Exception as e:
        print(f"\n❌ Example 1 failed: {str(e)}")

    try:
        example_2_overfitting_analysis()
    except Exception as e:
        print(f"\n❌ Example 2 failed: {str(e)}")

    try:
        example_3_combined_analysis()
    except Exception as e:
        print(f"\n❌ Example 3 failed: {str(e)}")

    try:
        example_4_direct_usage()
    except Exception as e:
        print(f"\n❌ Example 4 failed: {str(e)}")

    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETED")
    print("="*70)
    print("\nFor more information, see:")
    print("  - deepbridge/validation/robustness/weakspot_detector.py")
    print("  - deepbridge/validation/robustness/overfit_analyzer.py")
    print("  - deepbridge/validation/wrappers/robustness_suite.py")
