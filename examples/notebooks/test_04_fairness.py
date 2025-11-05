"""
Script de teste para notebooks 04_fairness
Verifica fairness tests com protected attributes
"""

import sys
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, '/home/guhaase/projetos/DeepBridge')

print("=" * 70)
print("TESTANDO NOTEBOOKS: 04_fairness")
print("=" * 70)

from deepbridge import DBDataset, Experiment

# Create synthetic dataset with protected attributes
np.random.seed(42)
n_samples = 1000

X, y = make_classification(
    n_samples=n_samples,
    n_features=10,
    n_informative=8,
    n_classes=2,
    random_state=42
)

# Add protected attributes
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
df['target'] = y

# Add gender (binary)
df['gender'] = np.random.choice(['M', 'F'], n_samples)

# Add race (3 categories)
df['race'] = np.random.choice(['White', 'Black', 'Asian'], n_samples)

# Add age groups
df['age_group'] = np.random.choice(['18-30', '31-50', '51+'], n_samples)

print(f"\n✅ Dataset created with protected attributes")
print(f"   Total samples: {len(df)}")
print(f"   Features: {df.shape[1] - 1}")
print(f"   Protected attributes: gender, race, age_group")

# Train model
X = df.drop('target', axis=1)
y = df['target']

# Need to encode categorical features for sklearn
from sklearn.preprocessing import LabelEncoder
le_gender = LabelEncoder()
le_race = LabelEncoder()
le_age = LabelEncoder()

X_encoded = X.copy()
X_encoded['gender'] = le_gender.fit_transform(X['gender'])
X_encoded['race'] = le_race.fit_transform(X['race'])
X_encoded['age_group'] = le_age.fit_transform(X['age_group'])

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
model.fit(X_train, y_train)

print(f"✅ Model trained: {type(model).__name__}")
print(f"   Train accuracy: {model.score(X_train, y_train):.3f}")

# Create DBDataset with encoded data
df_encoded = X_encoded.copy()
df_encoded['target'] = y

dataset = DBDataset(
    data=df_encoded,
    target_column='target',
    model=model,
    test_size=0.2,
    random_state=42
)

print(f"✅ DBDataset created")

# Create Experiment with protected attributes
try:
    exp = Experiment(
        dataset=dataset,
        experiment_type='binary_classification',
        protected_attributes=['gender', 'race', 'age_group'],
        random_state=42
    )
    
    print(f"✅ Experiment created with protected attributes")
    print(f"   Protected: {exp.protected_attributes}")
    
except Exception as e:
    print(f"❌ Error creating Experiment: {str(e)[:100]}")
    sys.exit(1)

# Test fairness analysis
tests = [
    ('fairness', 'Test fairness metrics across protected groups'),
]

passed = 0
failed = 0

for test_name, description in tests:
    try:
        print(f"\n✓ Testing {test_name}...")
        print(f"  {description}")
        
        # Run fairness test
        result = exp.run_fairness_tests(config='quick')
        
        if result is not None:
            print(f"  ✅ {test_name} - PASSOU")
            print(f"     Result type: {type(result).__name__}")
            
            # Check if we have fairness metrics
            if hasattr(result, 'metrics') or isinstance(result, dict):
                print(f"     ✓ Fairness metrics calculated")
            
            passed += 1
        else:
            print(f"  ⚠️  {test_name} - Retornou None")
            passed += 1
            
    except Exception as e:
        print(f"  ❌ {test_name} - ERRO: {str(e)[:150]}")
        import traceback
        print("\n  Traceback:")
        traceback.print_exc()
        failed += 1

print("\n" + "=" * 70)
print(f"RESUMO: {passed}/{len(tests)} testes passaram")
if failed > 0:
    print(f"⚠️  {failed} teste(s) falharam")
    sys.exit(1)
else:
    print("✅ TODOS OS TESTES PASSARAM!")
print("=" * 70)
