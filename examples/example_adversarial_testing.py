"""
Example usage of the adversarial robustness testing framework.

This script demonstrates how to test a simple machine learning model against
various adversarial attacks.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Import the adversarial testing framework
from deepbridge.validation.frameworks.robustness.adversarial_robustness import (
    FGSM, PGD, BlackBoxAttack
)
from deepbridge.validation.frameworks.robustness.adversarial_robustness_tester import (
    AdversarialRobustnessTester, test_adversarial_robustness
)


def main():
    """Run an example adversarial robustness test."""
    print("Generating synthetic dataset...")
    # Generate synthetic dataset
    X, y = make_classification(
        n_samples=1000, n_features=10, n_informative=5, 
        n_redundant=2, random_state=42
    )
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Create and train a model
    print("Training a RandomForest classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate clean accuracy
    y_pred = model.predict(X_test)
    clean_acc = accuracy_score(y_test, y_pred)
    print(f"Clean accuracy: {clean_acc:.4f}")
    
    # Test adversarial robustness
    print("\nTesting adversarial robustness...")
    print("This may take a few minutes...\n")
    
    # Method 1: Using the simplified function
    print("Method 1: Using the simplified test_adversarial_robustness function")
    results, fig = test_adversarial_robustness(
        model, X_test, y_test,
        attack_types=['fgsm', 'pgd'],  # Exclude 'blackbox' for faster execution
        epsilons=[0.1, 0.2],
        metrics=['accuracy']
    )
    
    # Show comparison figure
    plt.figure(fig.number)
    plt.show()
    
    # Print results summary
    print("\nResults summary:")
    for attack_name, attack_results in results.items():
        print(f"{attack_name}:")
        print(f"  Clean accuracy: {attack_results.get('clean_accuracy', 'N/A'):.4f}")
        print(f"  Adversarial accuracy: {attack_results.get('adversarial_accuracy', 'N/A'):.4f}")
        print(f"  Accuracy drop: {attack_results.get('accuracy_drop', 'N/A'):.4f}")
        print(f"  Attack success rate: {attack_results.get('success_rate', 'N/A'):.2f}%")
    
    # Method 2: Using the AdversarialRobustnessTester class for more control
    print("\nMethod 2: Using the AdversarialRobustnessTester class")
    tester = AdversarialRobustnessTester(model)
    
    # Run a specific attack
    print("Running FGSM attack with epsilon=0.1...")
    fgsm_results = tester.run_attack(
        attack_type='fgsm',
        X=X_test,
        y=y_test,
        epsilon=0.1
    )
    
    # Print FGSM results
    print("FGSM attack results:")
    print(f"  Clean accuracy: {fgsm_results.get('clean_accuracy', 'N/A'):.4f}")
    print(f"  Adversarial accuracy: {fgsm_results.get('adversarial_accuracy', 'N/A'):.4f}")
    print(f"  Attack success rate: {fgsm_results.get('success_rate', 'N/A'):.2f}%")
    
    # Visualize successful adversarial examples
    success_fig = tester.visualize_attack_success(n_examples=3)
    plt.figure(success_fig.number)
    plt.show()
    
    # Generate a report
    print("\nGenerating markdown report...")
    report = tester.generate_robustness_report(output_format='markdown')
    print("\nReport excerpt:")
    print(report[:500] + "...")  # Print first 500 chars


if __name__ == "__main__":
    main()