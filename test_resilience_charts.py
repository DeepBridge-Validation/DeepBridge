#!/usr/bin/env python
"""Test script to verify which resilience charts are generated"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Add DeepBridge to path
sys.path.insert(0, '/home/guhaase/projetos/DeepBridge')

from deepbridge.core.db_data import DBDataset
from deepbridge.core.experiment import Experiment

print("🔄 Generating synthetic data...")
# Generate synthetic data
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=5,
    n_redundant=10,
    n_classes=2,
    flip_y=0.2,
    class_sep=0.5,
    random_state=42
)

X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(20)])
y = pd.Series(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

print("🤖 Training model...")
# Train model
model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Prepare dataframes
train_df = X_train.copy()
train_df['target'] = y_train
test_df = X_test.copy()
test_df['target'] = y_test

print("📊 Creating dataset and experiment...")
# Create dataset
dataset = DBDataset(
    train_data=train_df,
    test_data=test_df,
    target_column='target',
    model=model
)

# Create and run experiment
experiment = Experiment(
    dataset=dataset,
    experiment_type="binary_classification",
    tests=["resilience"],
    feature_subset=['feature_6', 'feature_11', 'feature_17']
)

print("🚀 Running resilience tests...")
results = experiment.run_tests("full")

# Create output directory
output_dir = "/home/guhaase/projetos/DeepBridge/simular_lib/test_resilience_output"
os.makedirs(output_dir, exist_ok=True)

print(f"💾 Saving resilience report...")
output_file = os.path.join(output_dir, "test_resilience_report.html")
results.save_html("resilience", output_file, save_chart=True)

print(f"✅ Report saved to: {output_file}")

# Check which charts were generated
print("\n📈 Checking generated charts in HTML...")
with open(output_file, 'r') as f:
    html_content = f.read()

# List of expected charts based on our analysis
expected_charts = [
    ("Feature Distribution Shift", "feature_distribution_shift"),
    ("Critical Feature Distributions", "critical_feature_distributions"),
    ("Performance Gap", "performance_gap"),
    ("Performance Gap by Alpha", "performance_gap_by_alpha"),
    ("Model Resilience Scores", "model_resilience_scores"),
    ("Residual Distribution", "residual_distribution"),
    ("Feature-Residual Correlation", "feature_residual_correlation"),
    ("Distance Metrics Comparison", "distance_metrics_comparison"),
    ("Feature Distance Heatmap", "feature_distance_heatmap"),
    ("Model Comparison", "model_comparison"),
    ("Model Comparison Scatter", "model_comparison_scatter")
]

print("\n📊 Chart Generation Status:")
print("-" * 50)

generated_count = 0
missing_charts = []

for chart_name, chart_id in expected_charts:
    # Check if chart reference or title exists in HTML
    if chart_id in html_content or chart_name in html_content:
        print(f"✅ {chart_name}")
        generated_count += 1
    else:
        print(f"❌ {chart_name}")
        missing_charts.append(chart_name)

print("-" * 50)
print(f"\nTotal: {generated_count}/{len(expected_charts)} charts generated")

if missing_charts:
    print(f"\n⚠️ Missing charts: {', '.join(missing_charts)}")
else:
    print("\n✅ All expected charts were generated!")

# Additional check for base64 images
import re
images = re.findall(r'data:image/png;base64,', html_content)
print(f"\n🖼️ Total base64 images in report: {len(images)}")

print("\n✅ Test completed!")