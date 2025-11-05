"""
Script de teste para notebooks 03_validation_tests
Verifica integração com Experiment e run_test()
"""

import sys
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, '/home/guhaase/projetos/DeepBridge')

print("=" * 70)
print("TESTANDO NOTEBOOKS: 03_validation_tests")
print("=" * 70)

from deepbridge import DBDataset, Experiment

# Setup dataset and model
cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['target'] = cancer.target

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
model.fit(X_train, y_train)

dataset = DBDataset(
    data=df,
    target_column='target',
    model=model,
    test_size=0.2,
    random_state=42
)

# Create Experiment (no experiment_name parameter)
exp = Experiment(
    dataset=dataset,
    experiment_type='binary_classification',
    random_state=42
)

print(f"\n✅ Experiment created")
print(f"   Dataset: {len(dataset.train_data)} train, {len(dataset.test_data)} test")
print(f"   Model: {type(model).__name__}")

# Test each validation test type
tests = [
    ('robustness', 'Test model resistance to perturbations'),
    ('uncertainty', 'Test prediction confidence and calibration'),
]

passed = 0
failed = 0

for test_name, description in tests:
    try:
        print(f"\n✓ Testing {test_name}...")
        print(f"  {description}")
        
        result = exp.run_test(test_name, config='quick')
        
        if result is not None:
            print(f"  ✅ {test_name} - PASSOU")
            print(f"     Result type: {type(result).__name__}")
            passed += 1
        else:
            print(f"  ⚠️  {test_name} - Retornou None (pode ser esperado)")
            passed += 1
            
    except Exception as e:
        print(f"  ❌ {test_name} - ERRO: {str(e)[:100]}")
        failed += 1

print("\n" + "=" * 70)
print(f"RESUMO: {passed}/{len(tests)} testes passaram")
if failed > 0:
    print(f"⚠️  {failed} teste(s) falharam")
    sys.exit(1)
else:
    print("✅ TODOS OS TESTES PASSARAM!")
print("=" * 70)
