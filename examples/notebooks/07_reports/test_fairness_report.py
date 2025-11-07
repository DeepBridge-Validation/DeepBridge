#!/usr/bin/env python3
"""
Script to generate fairness reports - Test Script
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path

# sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# DeepBridge
from deepbridge import DBDataset, Experiment
from deepbridge.core.experiment.report import ReportManager

# Settings
warnings.filterwarnings('ignore')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Create output directory
output_dir = Path('outputs/fairness_reports')
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("🔬 Fairness Report Generation Test")
print("=" * 80)

# Load data
print("\n📊 Loading dataset...")
cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['target'] = cancer.target

# Create synthetic sensitive attribute
np.random.seed(RANDOM_STATE)
df['sensitive_attr'] = np.random.choice(['Group A', 'Group B'], size=len(df))

print(f"Dataset shape: {df.shape}")
print(f"Target distribution:\n{df['target'].value_counts()}")

# Train model
print("\n🤖 Training model...")
X = df.drop(['target', 'sensitive_attr'], axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
model.fit(X_train, y_train)

print(f"✅ Model trained")
print(f"   Train accuracy: {model.score(X_train, y_train):.4f}")
print(f"   Test accuracy: {model.score(X_test, y_test):.4f}")

# Create experiment
print("\n🔬 Creating experiment...")
dataset = DBDataset(
    data=df,
    target_column='target',
    model=model,
    test_size=0.2,
    random_state=RANDOM_STATE
)

exp = Experiment(
    dataset=dataset,
    experiment_type='binary_classification',
    test_size=0.2,
    random_state=RANDOM_STATE
)

print("✅ Experiment created")

# Run fairness test
print("\n🔍 Running fairness test...")
fairness_result = exp.run_test(
    'fairness',
    sensitive_features=['sensitive_attr'],
    config_name='medium'  # Valid options: 'quick', 'medium', 'full'
)

print("✅ Fairness test complete!")
print(f"Result type: {type(fairness_result)}")
print(f"Result keys: {list(fairness_result.keys()) if isinstance(fairness_result, dict) else 'N/A'}")
print(f"Result empty? {not fairness_result if isinstance(fairness_result, dict) else 'N/A'}")

# Generate reports using ReportManager
print("\n📄 Generating HTML reports...")

# IMPORTANT: Store result properly in experiment._test_results
# Initialize _test_results if it doesn't exist
if not hasattr(exp, '_test_results'):
    exp._test_results = {}

# Store fairness results with proper key
exp._test_results['fairness'] = fairness_result

# Debug: verify it was stored
print(f"✓ Stored in _test_results: {'fairness' in exp._test_results}")
print(f"✓ Keys in _test_results: {list(exp._test_results.keys())}")

# Generate Static Report
static_html_path = output_dir / 'fairness_report_static.html'
print(f"\n📊 Generating STATIC report...")
try:
    # Try using the FairnessResult.save_html() method directly
    if hasattr(fairness_result, 'save_html'):
        fairness_result.save_html(
            file_path=str(static_html_path),
            model_name='Breast Cancer Model',
            report_type='static'
        )
        print(f"   ✅ Static report: {static_html_path}")
        print(f"   💾 Size: {static_html_path.stat().st_size / 1024:.1f} KB")
    else:
        print(f"   ⚠️  FairnessResult doesn't have save_html method")
except Exception as e:
    print(f"   ⚠️  Static generation error: {e}")
    import traceback
    traceback.print_exc()

# Generate Interactive Report
interactive_html_path = output_dir / 'fairness_report_interactive.html'
print(f"\n🎯 Generating INTERACTIVE report...")
try:
    # Try using the FairnessResult.save_html() method directly
    if hasattr(fairness_result, 'save_html'):
        fairness_result.save_html(
            file_path=str(interactive_html_path),
            model_name='Breast Cancer Model',
            report_type='interactive'
        )
        print(f"   ✅ Interactive report: {interactive_html_path}")
        print(f"   💾 Size: {interactive_html_path.stat().st_size / 1024:.1f} KB")
    else:
        print(f"   ⚠️  FairnessResult doesn't have save_html method")
except Exception as e:
    print(f"   ⚠️  Interactive generation error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("✅ Report generation complete!")
print("=" * 80)
print(f"\n📁 Reports saved in: {output_dir}")
print(f"\nGenerated files:")
for file in sorted(output_dir.glob("*")):
    size_kb = file.stat().st_size / 1024
    print(f"   • {file.name} ({size_kb:.1f} KB)")
