"""
Script de teste rápido para notebooks 02_dbdataset
"""

import sys
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, '/home/guhaase/projetos/DeepBridge')

print("=" * 70)
print("TESTANDO NOTEBOOKS: 02_dbdataset")
print("=" * 70)

# Lista de testes a executar
tests = []

# Test 1: Simple Loading
def test_simple_loading():
    """01_simple_loading.ipynb - Create DBDataset from DataFrame"""
    from deepbridge import DBDataset
    
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    
    dataset = DBDataset(
        data=df,
        target_column='target',
        test_size=0.2,
        random_state=42
    )
    
    assert len(dataset.train_data) + len(dataset.test_data) == len(df)
    assert dataset.target_name == 'target'
    assert len(dataset.features) > 0
    return True

tests.append(("01_simple_loading", test_simple_loading))

# Test 2: Pre-separated Data
def test_pre_separated():
    """02_pre_separated_data.ipynb - Use pre-split train/test"""
    from deepbridge import DBDataset
    
    cancer = load_breast_cancer()
    X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    y = pd.Series(cancer.target, name='target')
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    dataset = DBDataset(
        train_data=train_df,
        test_data=test_df,
        target_column='target'
    )
    
    assert len(dataset.train_data) == len(train_df)
    assert len(dataset.test_data) == len(test_df)
    return True

tests.append(("02_pre_separated", test_pre_separated))

# Test 3: Model Integration
def test_model_integration():
    """03_model_integration.ipynb - DBDataset with model"""
    from deepbridge import DBDataset
    
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    dataset = DBDataset(
        data=df,
        target_column='target',
        model=model,
        test_size=0.2,
        random_state=42
    )
    
    assert dataset.model is not None
    assert hasattr(dataset.model, 'predict')
    return True

tests.append(("03_model_integration", test_model_integration))

# Test 4: Saved Models
def test_saved_models():
    """04_saved_models.ipynb - Load model from file"""
    from deepbridge import DBDataset
    import joblib
    import tempfile
    import os
    
    # Create and save a model
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Save model to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
        joblib.dump(model, f.name)
        model_path = f.name
    
    try:
        # Create dataset with model_path
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['target'] = iris.target
        
        dataset = DBDataset(
            data=df,
            target_column='target',
            model_path=model_path,
            test_size=0.2,
            random_state=42
        )
        
        assert dataset.model is not None
        return True
    finally:
        # Cleanup
        if os.path.exists(model_path):
            os.remove(model_path)

tests.append(("04_saved_models", test_saved_models))

# Run all tests
passed = 0
failed = 0

for name, test_func in tests:
    try:
        print(f"\n✓ Testando {name}...")
        result = test_func()
        if result:
            print(f"  ✅ {name} - PASSOU")
            passed += 1
        else:
            print(f"  ❌ {name} - FALHOU")
            failed += 1
    except Exception as e:
        print(f"  ❌ {name} - ERRO: {str(e)[:80]}")
        failed += 1

print("\n" + "=" * 70)
print(f"RESUMO: {passed}/{len(tests)} testes passaram")
if failed > 0:
    print(f"⚠️  {failed} teste(s) falharam")
    sys.exit(1)
else:
    print("✅ TODOS OS TESTES PASSARAM!")
print("=" * 70)
