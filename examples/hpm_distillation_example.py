"""
Example: Using HPM-KD for Knowledge Distillation

This example demonstrates how to use the new HPM-KD (Hierarchical Progressive
Multi-Teacher Knowledge Distillation) method for efficient model distillation.

HPM-KD features:
- 87.5% reduction in training time (640 → 80 models)
- Intelligent configuration selection
- Progressive distillation chain
- Multi-teacher ensemble with attention
- Adaptive temperature scheduling
- Parallel processing and caching
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Import DeepBridge components
from deepbridge.core.db_data import DBDataset
from deepbridge.distillation import AutoDistiller


def create_sample_dataset(n_samples=5000, n_features=20):
    """
    Create a sample dataset for demonstration.

    Args:
        n_samples: Number of samples
        n_features: Number of features

    Returns:
        DBDataset instance
    """
    # Generate synthetic data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=15,
        n_redundant=5,
        n_clusters_per_class=2,
        random_state=42
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train a teacher model
    teacher = RandomForestClassifier(n_estimators=100, random_state=42)
    teacher.fit(X_train, y_train)

    # Get teacher probabilities
    probs_train = teacher.predict_proba(X_train)
    probs_test = teacher.predict_proba(X_test)

    # Combine data back
    X_full = np.vstack([X_train, X_test])
    y_full = np.hstack([y_train, y_test])
    probs_full = np.vstack([probs_train, probs_test])

    # Create DBDataset
    dataset = DBDataset(
        X=pd.DataFrame(X_full),
        y=pd.Series(y_full),
        probabilities=pd.DataFrame(probs_full, columns=['prob_0', 'prob_1'])
    )

    print(f"Created dataset with {n_samples} samples and {n_features} features")
    print(f"Teacher model accuracy: {teacher.score(X_test, y_test):.3f}")

    return dataset


def example_legacy_distillation():
    """
    Example using traditional (legacy) distillation approach.
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Legacy Distillation (Traditional Grid Search)")
    print("="*60)

    # Create dataset
    dataset = create_sample_dataset(n_samples=1000)

    # Initialize distiller with legacy method
    distiller = AutoDistiller(
        dataset=dataset,
        method='legacy',  # Force legacy method
        n_trials=10,
        verbose=True
    )

    # Run distillation
    print("\nRunning legacy distillation...")
    import time
    start_time = time.time()

    results = distiller.run()

    elapsed = time.time() - start_time
    print(f"\nLegacy distillation completed in {elapsed:.2f} seconds")
    print(f"Tested {len(results)} configurations")

    # Get best model
    best_model = distiller.best_model()
    print(f"Best model selected: {best_model}")

    return elapsed


def example_hpm_distillation():
    """
    Example using HPM-KD distillation approach.
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: HPM-KD Distillation (Optimized)")
    print("="*60)

    # Create dataset
    dataset = create_sample_dataset(n_samples=1000)

    # Initialize distiller with HPM method
    distiller = AutoDistiller(
        dataset=dataset,
        method='hpm',  # Use HPM-KD
        n_trials=10,  # Will be reduced automatically
        verbose=True
    )

    # Run distillation
    print("\nRunning HPM distillation...")
    print("Features enabled:")
    print("- Adaptive configuration selection (64 → 16 configs)")
    print("- Progressive distillation chain")
    print("- Multi-teacher ensemble")
    print("- Parallel processing")
    print("- Intelligent caching")

    import time
    start_time = time.time()

    results = distiller.run()

    elapsed = time.time() - start_time
    print(f"\nHPM distillation completed in {elapsed:.2f} seconds")

    # Get best model
    best_model = distiller.best_model()
    print(f"Best model selected: {best_model}")

    return elapsed


def example_auto_selection():
    """
    Example using automatic method selection.
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Automatic Method Selection")
    print("="*60)

    # Small dataset - should select legacy
    print("\nSmall dataset (500 samples):")
    small_dataset = create_sample_dataset(n_samples=500)

    distiller_small = AutoDistiller(
        dataset=small_dataset,
        method='auto',  # Automatic selection
        verbose=True
    )
    print(f"Auto-selected method: {distiller_small.method}")

    # Large dataset - should select HPM
    print("\nLarge dataset (10000 samples):")
    large_dataset = create_sample_dataset(n_samples=10000)

    distiller_large = AutoDistiller(
        dataset=large_dataset,
        method='auto',  # Automatic selection
        verbose=True
    )
    print(f"Auto-selected method: {distiller_large.method}")


def example_hybrid_comparison():
    """
    Example using hybrid mode to compare both methods.
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Hybrid Mode (Compare Both Methods)")
    print("="*60)

    # Create dataset
    dataset = create_sample_dataset(n_samples=2000)

    # Initialize distiller with hybrid method
    distiller = AutoDistiller(
        dataset=dataset,
        method='hybrid',  # Run both methods
        n_trials=5,  # Reduced for demo
        verbose=True
    )

    print("\nRunning hybrid distillation (both methods)...")
    results = distiller.run()

    print("\nComparison complete!")
    print("Results will show performance from both methods")


def example_custom_hpm_config():
    """
    Example with custom HPM configuration.
    """
    print("\n" + "="*60)
    print("EXAMPLE 5: Custom HPM Configuration")
    print("="*60)

    # Create dataset
    dataset = create_sample_dataset(n_samples=3000)

    # Import HPM components for custom configuration
    from deepbridge.distillation.techniques.hpm import HPMDistiller, HPMConfig

    # Create custom HPM configuration
    hpm_config = HPMConfig(
        max_configs=8,  # Even fewer configurations
        n_trials=3,  # Minimal trials with warm start
        use_progressive=True,  # Enable progressive chain
        use_multi_teacher=True,  # Enable multi-teacher
        use_adaptive_temperature=True,  # Adaptive temperature
        use_parallel=True,  # Parallel processing
        parallel_workers=4,  # Use 4 workers
        cache_memory_gb=1.0,  # Limit cache to 1GB
        verbose=True
    )

    # Use HPM distiller directly
    hpm_distiller = HPMDistiller(config=hpm_config)

    print("\nCustom HPM configuration:")
    print(f"- Max configurations: {hpm_config.max_configs}")
    print(f"- Parallel workers: {hpm_config.parallel_workers}")
    print(f"- Cache memory: {hpm_config.cache_memory_gb} GB")

    # Split data for training
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.X.values,
        dataset.y.values,
        test_size=0.2,
        random_state=42
    )

    # Train with custom configuration
    print("\nTraining with custom HPM configuration...")
    hpm_distiller.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        teacher_probs=dataset.probabilities.values[:len(X_train)]
    )

    # Get statistics
    stats = hpm_distiller.get_stats()
    print(f"\nTotal training time: {stats['total_time']:.2f} seconds")
    print(f"Best model performance: {stats['best_metrics']}")


def main():
    """
    Main function to run all examples.
    """
    print("HPM-KD Knowledge Distillation Examples")
    print("======================================")

    # Run examples
    # Note: Comment out examples you don't want to run

    # 1. Legacy method
    legacy_time = example_legacy_distillation()

    # 2. HPM method
    hpm_time = example_hpm_distillation()

    # 3. Auto selection
    example_auto_selection()

    # 4. Hybrid comparison
    # example_hybrid_comparison()  # Commented out as it takes longer

    # 5. Custom HPM config
    example_custom_hpm_config()

    # Performance comparison
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    print(f"Legacy method time: {legacy_time:.2f} seconds")
    print(f"HPM method time: {hpm_time:.2f} seconds")
    print(f"Speedup: {legacy_time/hpm_time:.2f}x")
    print(f"Time saved: {legacy_time - hpm_time:.2f} seconds ({(1 - hpm_time/legacy_time)*100:.1f}%)")


if __name__ == "__main__":
    main()