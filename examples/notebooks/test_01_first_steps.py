"""
Script de teste para o notebook 01_first_steps.ipynb
Verifica se o código executa sem erros
"""

import sys
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

# Adicionar o path do DeepBridge
sys.path.insert(0, '/home/guhaase/projetos/DeepBridge')

print("=" * 70)
print("TESTANDO NOTEBOOK: 01_first_steps.ipynb")
print("=" * 70)

try:
    # Test imports
    print("\n✓ Testando imports...")
    from deepbridge import DBDataset
    print("  ✅ DBDataset importado com sucesso")

    # Test data loading
    print("\n✓ Testando carregamento de dados...")
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    print(f"  ✅ Iris dataset carregado: {df.shape}")

    # Test DBDataset creation
    print("\n✓ Testando criação do DBDataset...")
    dataset = DBDataset(
        data=df,
        target_column='target',
        test_size=0.2,
        random_state=42
    )
    print(f"  ✅ DBDataset criado com sucesso")
    print(f"     - Train samples: {len(dataset.train_data)}")
    print(f"     - Test samples: {len(dataset.test_data)}")

    # Test properties access
    print("\n✓ Testando propriedades do DBDataset...")

    # Features
    features = dataset.features
    print(f"  ✅ Features acessadas: {len(features)} features")

    # Categorical features
    cat_features = dataset.categorical_features
    print(f"  ✅ Categorical features: {len(cat_features)}")

    # Numerical features
    num_features = dataset.numerical_features
    print(f"  ✅ Numerical features: {len(num_features)}")

    # Target name
    target_name = dataset.target_name
    print(f"  ✅ Target name: {target_name}")

    # Test get_feature_data and get_target_data
    print("\n✓ Testando métodos de acesso aos dados...")
    X_train = dataset.get_feature_data('train')
    y_train = dataset.get_target_data('train')
    X_test = dataset.get_feature_data('test')
    y_test = dataset.get_target_data('test')

    print(f"  ✅ X_train shape: {X_train.shape}")
    print(f"  ✅ y_train shape: {y_train.shape}")
    print(f"  ✅ X_test shape: {X_test.shape}")
    print(f"  ✅ y_test shape: {y_test.shape}")

    # Verify split is correct
    print("\n✓ Verificando integridade do split...")
    total_samples = len(X_train) + len(X_test)
    expected_samples = len(df)

    if total_samples == expected_samples:
        print(f"  ✅ Split correto: {total_samples} samples total")
    else:
        print(f"  ❌ ERRO: Total de samples não bate ({total_samples} vs {expected_samples})")

    # Check class distribution
    print("\n✓ Verificando distribuição de classes...")
    print(f"  Train: {y_train.value_counts().to_dict()}")
    print(f"  Test:  {y_test.value_counts().to_dict()}")

    # Check if any class is missing in test
    train_classes = set(y_train.unique())
    test_classes = set(y_test.unique())

    if train_classes == test_classes:
        print(f"  ✅ Todas as classes presentes em train e test")
    else:
        missing_in_test = train_classes - test_classes
        if missing_in_test:
            print(f"  ⚠️  WARNING: Classes ausentes no test: {missing_in_test}")

    print("\n" + "=" * 70)
    print("✅ TODOS OS TESTES PASSARAM!")
    print("=" * 70)

except Exception as e:
    print(f"\n❌ ERRO ENCONTRADO:")
    print(f"   {type(e).__name__}: {str(e)}")
    import traceback
    print("\nTraceback completo:")
    traceback.print_exc()
    sys.exit(1)
