"""
Script de teste para validar a Fase 2 do FairnessSuite.

Este script testa:
- Config 'quick': 2 métricas pós-treino
- Config 'medium': 5 pós-treino + 4 pré-treino + confusion matrix
- Config 'full': 11 pós-treino + 4 pré-treino + confusion matrix + threshold analysis
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from deepbridge.core.db_data import DBDataset
from deepbridge.validation.wrappers.fairness_suite import FairnessSuite

print("=" * 80)
print("TESTE DA FASE 2 - FAIRNESS SUITE EXPANDIDO")
print("=" * 80)

# ============================================================================
# 1. CRIAR DADOS SINTÉTICOS COM VIÉS
# ============================================================================

print("\n📊 1. Gerando dados sintéticos com viés...")

np.random.seed(42)
n_samples = 1000

# Criar dados com features numéricas
X_numeric = np.random.randn(n_samples, 5)

# Criar atributos protegidos
gender = np.random.choice(['M', 'F'], n_samples, p=[0.7, 0.3])
race = np.random.choice(['White', 'Black', 'Hispanic'], n_samples, p=[0.6, 0.25, 0.15])

# Criar target com viés (mais positivos para grupo privilegiado)
y = np.zeros(n_samples)
for i in range(n_samples):
    base_prob = 0.3

    # Adicionar viés por gênero
    if gender[i] == 'M':
        base_prob += 0.15  # Viés favorecendo homens

    # Adicionar viés por raça
    if race[i] == 'White':
        base_prob += 0.10  # Viés favorecendo brancos

    y[i] = 1 if np.random.rand() < base_prob else 0

# Criar DataFrame completo
df = pd.DataFrame(X_numeric, columns=[f'feature_{i}' for i in range(5)])
df['gender'] = gender
df['race'] = race
df['target'] = y

print(f"  Total de amostras: {len(df)}")
print(f"  Distribuição de gênero: M={np.sum(gender=='M')}, F={np.sum(gender=='F')}")
print(f"  Distribuição de raça: White={np.sum(race=='White')}, Black={np.sum(race=='Black')}, Hispanic={np.sum(race=='Hispanic')}")
print(f"  Taxa positiva geral: {np.mean(y):.3f}")
print(f"  Taxa positiva (M): {np.mean(y[gender=='M']):.3f}")
print(f"  Taxa positiva (F): {np.mean(y[gender=='F']):.3f}")

# ============================================================================
# 2. TREINAR MODELO
# ============================================================================

print("\n🤖 2. Treinando modelo RandomForest...")

# Separar features (sem atributos protegidos) e target
X_train = df.drop(['gender', 'race', 'target'], axis=1)
y_train = df['target']

model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
model.fit(X_train, y_train)

print(f"  Acurácia no treino: {model.score(X_train, y_train):.3f}")

# Gerar predições para todo o dataset
y_pred = model.predict(X_train)
y_pred_proba = model.predict_proba(X_train)

print(f"  Predições geradas: {len(y_pred)}")

# ============================================================================
# 3. CRIAR DBDATASET
# ============================================================================

print("\n📦 3. Criando DBDataset...")

# Criar DataFrame completo incluindo atributos protegidos e predições
df_with_preds = df.copy()
df_with_preds['prediction'] = y_pred
df_with_preds['proba_class_0'] = y_pred_proba[:, 0]
df_with_preds['proba_class_1'] = y_pred_proba[:, 1]

dataset = DBDataset(
    data=df_with_preds,
    target_column='target',
    model=model,
    train_predictions=pd.DataFrame({
        'prediction': y_pred,
        'proba_class_0': y_pred_proba[:, 0],
        'proba_class_1': y_pred_proba[:, 1]
    })
)

print(f"  Dataset shape: {df_with_preds.shape}")
print(f"  Protected attributes: gender, race")

# ============================================================================
# 4. TESTE CONFIG 'QUICK'
# ============================================================================

print("\n" + "=" * 80)
print("📋 TESTE 1: CONFIG 'QUICK' (2 métricas pós-treino)")
print("=" * 80)

fairness_quick = FairnessSuite(
    dataset=dataset,
    protected_attributes=['gender', 'race'],
    verbose=True
)

results_quick = fairness_quick.config('quick').run()

print("\n✅ Resultados 'quick':")
print(f"  - Overall Score: {results_quick['overall_fairness_score']:.3f}")
print(f"  - Pré-treino calculado: {bool(results_quick['pretrain_metrics'])}")
print(f"  - Pós-treino calculado: {bool(results_quick['posttrain_metrics'])}")
print(f"  - Confusion matrix: {bool(results_quick['confusion_matrix'])}")
print(f"  - Threshold analysis: {results_quick['threshold_analysis'] is not None}")
print(f"  - Warnings: {len(results_quick['warnings'])}")
print(f"  - Critical: {len(results_quick['critical_issues'])}")

# Validações
assert not results_quick['pretrain_metrics'], "Quick não deve ter pré-treino"
assert results_quick['posttrain_metrics'], "Quick deve ter pós-treino"
assert not results_quick['confusion_matrix'], "Quick não deve ter confusion matrix"
assert results_quick['threshold_analysis'] is None, "Quick não deve ter threshold analysis"
print("✓ Todas as validações do 'quick' passaram!")

# ============================================================================
# 5. TESTE CONFIG 'MEDIUM'
# ============================================================================

print("\n" + "=" * 80)
print("📋 TESTE 2: CONFIG 'MEDIUM' (5 pós + 4 pré + confusion matrix)")
print("=" * 80)

fairness_medium = FairnessSuite(
    dataset=dataset,
    protected_attributes=['gender', 'race'],
    verbose=True
)

results_medium = fairness_medium.config('medium').run()

print("\n✅ Resultados 'medium':")
print(f"  - Overall Score: {results_medium['overall_fairness_score']:.3f}")
print(f"  - Pré-treino calculado: {bool(results_medium['pretrain_metrics'])}")
print(f"  - Pós-treino calculado: {bool(results_medium['posttrain_metrics'])}")
print(f"  - Confusion matrix: {bool(results_medium['confusion_matrix'])}")
print(f"  - Threshold analysis: {results_medium['threshold_analysis'] is not None}")
print(f"  - Warnings: {len(results_medium['warnings'])}")
print(f"  - Critical: {len(results_medium['critical_issues'])}")

# Validações
assert results_medium['pretrain_metrics'], "Medium deve ter pré-treino"
assert results_medium['posttrain_metrics'], "Medium deve ter pós-treino"
assert results_medium['confusion_matrix'], "Medium deve ter confusion matrix"
assert results_medium['threshold_analysis'] is None, "Medium não deve ter threshold analysis"

# Validar que tem 4 métricas pré-treino por atributo
for attr in ['gender', 'race']:
    assert attr in results_medium['pretrain_metrics']
    assert len(results_medium['pretrain_metrics'][attr]) == 4, \
        f"Deve ter 4 métricas pré-treino para {attr}"

# Validar confusion matrix
for attr in ['gender', 'race']:
    assert attr in results_medium['confusion_matrix']
    cm = results_medium['confusion_matrix'][attr]
    for group in cm.keys():
        assert all(k in cm[group] for k in ['TP', 'FP', 'TN', 'FN', 'total'])

print("✓ Todas as validações do 'medium' passaram!")

# ============================================================================
# 6. TESTE CONFIG 'FULL'
# ============================================================================

print("\n" + "=" * 80)
print("📋 TESTE 3: CONFIG 'FULL' (11 pós + 4 pré + CM + threshold)")
print("=" * 80)

fairness_full = FairnessSuite(
    dataset=dataset,
    protected_attributes=['gender', 'race'],
    verbose=True
)

results_full = fairness_full.config('full').run()

print("\n✅ Resultados 'full':")
print(f"  - Overall Score: {results_full['overall_fairness_score']:.3f}")
print(f"  - Pré-treino calculado: {bool(results_full['pretrain_metrics'])}")
print(f"  - Pós-treino calculado: {bool(results_full['posttrain_metrics'])}")
print(f"  - Confusion matrix: {bool(results_full['confusion_matrix'])}")
print(f"  - Threshold analysis: {results_full['threshold_analysis'] is not None}")
print(f"  - Warnings: {len(results_full['warnings'])}")
print(f"  - Critical: {len(results_full['critical_issues'])}")

# Validações
assert results_full['pretrain_metrics'], "Full deve ter pré-treino"
assert results_full['posttrain_metrics'], "Full deve ter pós-treino"
assert results_full['confusion_matrix'], "Full deve ter confusion matrix"
assert results_full['threshold_analysis'] is not None, "Full deve ter threshold analysis"

# Validar threshold analysis
ta = results_full['threshold_analysis']
assert 'optimal_threshold' in ta
assert 'optimal_metrics' in ta
assert 'threshold_curve' in ta
assert 'recommendations' in ta
assert 0 < ta['optimal_threshold'] < 1, "Threshold deve estar entre 0 e 1"
assert len(ta['threshold_curve']) > 0, "Deve ter curva de threshold"

print(f"\n📈 Threshold Analysis:")
print(f"  - Optimal threshold: {ta['optimal_threshold']:.3f}")
print(f"  - Disparate Impact @ optimal: {ta['optimal_metrics']['disparate_impact_ratio']:.3f}")
print(f"  - F1 Score @ optimal: {ta['optimal_metrics']['f1_score']:.3f}")
print(f"  - Recommendations: {len(ta['recommendations'])}")
for rec in ta['recommendations']:
    print(f"    • {rec}")

print("\n✓ Todas as validações do 'full' passaram!")

# ============================================================================
# 7. VALIDAÇÃO DAS NOVAS MÉTRICAS
# ============================================================================

print("\n" + "=" * 80)
print("📊 VALIDAÇÃO DAS NOVAS MÉTRICAS")
print("=" * 80)

# Validar que todas as 11 métricas pós-treino foram calculadas
expected_posttrain = [
    'statistical_parity', 'equal_opportunity', 'equalized_odds',
    'disparate_impact', 'false_negative_rate_difference',
    'conditional_acceptance', 'conditional_rejection',
    'precision_difference', 'accuracy_difference',
    'treatment_equality', 'entropy_index'
]

for attr in ['gender', 'race']:
    metrics_calculated = set(results_full['posttrain_metrics'][attr].keys())
    expected_set = set(expected_posttrain)

    print(f"\n{attr}:")
    print(f"  Esperadas: {len(expected_set)}")
    print(f"  Calculadas: {len(metrics_calculated)}")

    missing = expected_set - metrics_calculated
    if missing:
        print(f"  ❌ Faltando: {missing}")
    else:
        print(f"  ✅ Todas as métricas pós-treino presentes")

    assert metrics_calculated == expected_set, f"Métricas faltando para {attr}: {missing}"

# Validar métricas pré-treino
expected_pretrain = ['class_balance', 'concept_balance', 'kl_divergence', 'js_divergence']

for attr in ['gender', 'race']:
    pretrain_calculated = set(results_full['pretrain_metrics'][attr].keys())
    expected_set = set(expected_pretrain)

    print(f"\nPré-treino {attr}:")
    print(f"  Esperadas: {len(expected_set)}")
    print(f"  Calculadas: {len(pretrain_calculated)}")

    assert pretrain_calculated == expected_set, f"Métricas pré-treino faltando para {attr}"
    print(f"  ✅ Todas as métricas pré-treino presentes")

# ============================================================================
# 8. RESUMO FINAL
# ============================================================================

print("\n" + "=" * 80)
print("🎉 RESUMO FINAL - FASE 2")
print("=" * 80)

print("\n✅ SUCESSOS:")
print("  ✓ Config 'quick' funcionando (2 métricas)")
print("  ✓ Config 'medium' funcionando (5 pós + 4 pré + CM)")
print("  ✓ Config 'full' funcionando (11 pós + 4 pré + CM + TA)")
print("  ✓ Todas as 15 métricas disponíveis")
print("  ✓ Métricas pré-treino implementadas")
print("  ✓ Confusion matrix por grupo")
print("  ✓ Threshold analysis funcional")
print("  ✓ Sistema de warnings/critical expandido")
print("  ✓ Overall fairness score v2")

print("\n📊 ESTATÍSTICAS:")
print(f"  - Total de configs testados: 3")
print(f"  - Métricas pré-treino: {len(expected_pretrain)}")
print(f"  - Métricas pós-treino: {len(expected_posttrain)}")
print(f"  - Total de métricas: {len(expected_pretrain) + len(expected_posttrain)}")
print(f"  - Atributos protegidos testados: 2")
print(f"  - Threshold points analisados: {len(ta['threshold_curve'])}")

print("\n" + "=" * 80)
print("✅ FASE 2 - TESTE COMPLETO PASSOU COM SUCESSO!")
print("=" * 80)
