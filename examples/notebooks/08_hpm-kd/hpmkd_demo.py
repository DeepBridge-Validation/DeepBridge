#!/usr/bin/env python3
"""
HPM-KD Demo Script
Demonstração do framework HPM-KD conforme Listing 1 do paper
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Configuração
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print('=' * 60)
print('HPM-KD DEMONSTRATION')
print('=' * 60)

# 1. Carregar dados
print('\n1. Carregando dataset Digits...')
digits = load_digits()
X, y = digits.data, digits.target
print(f'   Shape: {X.shape}, Classes: {len(np.unique(y))}')

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)
print(f'   Train: {X_train.shape[0]}, Test: {X_test.shape[0]}')

# Data loaders
train_loader = (X_train, y_train)
test_loader = (X_test, y_test)

# 2. Treinar Professor
print('\n2. Treinando modelo PROFESSOR (Random Forest)...')
teacher = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=2,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
teacher.fit(X_train, y_train)
teacher_test_acc = accuracy_score(y_test, teacher.predict(X_test))
teacher_params = teacher.n_estimators * sum(
    [t.tree_.node_count for t in teacher.estimators_]
)
print(f'   Test Accuracy: {teacher_test_acc:.4f}')
print(f'   Model Size: {teacher_params:,} nodes')

# 3. Treinar Baseline
print('\n3. Treinando BASELINE (Decision Tree - Direct Training)...')
baseline_student = DecisionTreeClassifier(
    max_depth=10, min_samples_split=5, random_state=RANDOM_STATE
)
baseline_student.fit(X_train, y_train)
baseline_test_acc = accuracy_score(y_test, baseline_student.predict(X_test))
baseline_params = baseline_student.tree_.node_count
print(f'   Test Accuracy: {baseline_test_acc:.4f}')
print(f'   Model Size: {baseline_params:,} nodes')

# 4. HPM-KD (Listing 1 do paper)
print('\n4. Executando HPM-KD (Listing 1 do paper)...')
print('   ' + '-' * 56)

try:
    # ==================== LISTING 1 DO PAPER ====================
    from deepbridge.distillation import HPMKD

    # Configuração automática via meta-learning
    hpmkd = HPMKD(
        teacher_model=teacher,
        student_model=DecisionTreeClassifier(
            max_depth=10, random_state=RANDOM_STATE
        ),
        train_loader=train_loader,
        test_loader=test_loader,
        auto_config=True,  # Sem ajuste manual!
    )

    # Destilação progressiva multi-professor
    hpmkd.distill(epochs=150)

    # Avaliar estudante comprimido
    student_acc = hpmkd.evaluate()
    print(f'   Compressão: {hpmkd.compression_ratio:.1f}x')
    print(f'   Retenção: {hpmkd.retention_pct:.1f}%')
    # ==================== FIM DO LISTING 1 ======================

    # 5. Resultados
    print('\n' + '=' * 60)
    print('RESULTADOS FINAIS')
    print('=' * 60)
    print(f'\nAcurácia no TEST:')
    print(f'  Professor (RF 200):      {teacher_test_acc:.4f}')
    print(f'  Baseline (Direct):       {baseline_test_acc:.4f}')
    print(f'  HPM-KD (Distilled):      {student_acc:.4f}')

    print(f'\nMÉTRICAS DE COMPRESSÃO:')
    print(f'  Compression Ratio:       {hpmkd.compression_ratio:.1f}x')
    print(f'  Retenção:                {hpmkd.retention_pct:.1f}%')
    print(
        f'  Ganho sobre Baseline:    +{(student_acc - baseline_test_acc)*100:.2f} pp'
    )

    student_model = hpmkd.student
    if hasattr(student_model, 'tree_'):
        student_params = student_model.tree_.node_count
        print(f'\nTAMANHO DOS MODELOS:')
        print(f'  Professor:               {teacher_params:,} nodes')
        print(f'  Estudante:               {student_params:,} nodes')
        print(
            f'  Redução:                 {(1 - student_params/teacher_params)*100:.1f}%'
        )

    print('\n' + '=' * 60)
    print('✅ HPM-KD executado com sucesso!')
    print('   API do Listing 1 do paper funcionando perfeitamente!')
    print('=' * 60)

except ImportError as e:
    print(f'   ❌ ERRO: {e}')
    print('   Certifique-se de que DeepBridge está instalado:')
    print('   pip install -e /path/to/DeepBridge')
except Exception as e:
    print(f'   ❌ ERRO durante execução: {e}')
    import traceback

    traceback.print_exc()

print('\n✅ Demo concluída!')
