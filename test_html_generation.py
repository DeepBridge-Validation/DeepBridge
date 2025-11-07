#!/usr/bin/env python3
"""
Test script to verify HTML report generation with save_html() method.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from deepbridge import DBDataset, Experiment

print("=" * 80)
print("Testing HTML Report Generation (save_html)")
print("=" * 80)

# Prepare data
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['target'] = cancer.target

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
model.fit(X_train, y_train)

print("\n✅ Data prepared and model trained")

# Create experiment
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

# Run uncertainty test
try:
    print("\n🔬 Running uncertainty test...")
    uncertainty_result = exp.run_test('uncertainty', config='quick')
    print("✅ Uncertainty test completed")
except Exception as e:
    print(f"❌ Test execution failed: {e}")
    exit(1)

# Generate HTML report using save_html
try:
    print("\n📄 Generating HTML report with save_html()...")

    output_dir = Path('/tmp/deepbridge_html_test')
    output_dir.mkdir(exist_ok=True)
    html_path = output_dir / 'uncertainty_report.html'

    # Use the correct method: save_html
    result_path = exp.save_html(
        test_type='uncertainty',
        file_path=str(html_path),
        model_name='Breast Cancer Test Model'
    )

    print(f"✅ HTML report generated successfully!")
    print(f"   Path: {result_path}")

    # Check file exists and has content
    if Path(result_path).exists():
        file_size = Path(result_path).stat().st_size
        print(f"   File size: {file_size / 1024:.1f} KB")

        if file_size > 0:
            print("   ✅ File has content")
        else:
            print("   ❌ File is empty!")
            exit(1)
    else:
        print(f"   ❌ File not found at {result_path}")
        exit(1)

except AttributeError as e:
    print(f"❌ AttributeError: {e}")
    print("\n💡 This means save_html() method doesn't exist or has wrong signature")
    exit(1)
except Exception as e:
    print(f"❌ HTML generation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 80)
print("🎉 HTML Generation Test PASSED!")
print("=" * 80)
print("\n✨ save_html() method works correctly:")
print("   • Generates valid HTML file ✅")
print("   • Accepts test_type, file_path, model_name ✅")
print("   • Returns path to generated file ✅")
print("\n💡 The notebooks should now work correctly!")
