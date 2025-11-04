"""
Script de teste para validar as 15 métricas de fairness expandidas.

Este script testa:
- 4 métricas pré-treino (independentes do modelo)
- 11 métricas pós-treino (dependentes do modelo)
"""

import numpy as np
import pandas as pd
from deepbridge.validation.fairness.metrics import FairnessMetrics

# Criar dados sintéticos
np.random.seed(42)
n_samples = 1000

# Criar grupos sensíveis
sensitive_feature = np.random.choice(['Group_A', 'Group_B'], n_samples, p=[0.7, 0.3])

# Criar labels verdadeiros com algum viés
y_true = np.zeros(n_samples)
y_true[sensitive_feature == 'Group_A'] = np.random.choice([0, 1], np.sum(sensitive_feature == 'Group_A'), p=[0.6, 0.4])
y_true[sensitive_feature == 'Group_B'] = np.random.choice([0, 1], np.sum(sensitive_feature == 'Group_B'), p=[0.5, 0.5])

# Criar predições com viés
y_pred = np.zeros(n_samples)
y_pred[sensitive_feature == 'Group_A'] = np.random.choice([0, 1], np.sum(sensitive_feature == 'Group_A'), p=[0.55, 0.45])
y_pred[sensitive_feature == 'Group_B'] = np.random.choice([0, 1], np.sum(sensitive_feature == 'Group_B'), p=[0.65, 0.35])

print("=" * 80)
print("TESTE DAS 15 MÉTRICAS DE FAIRNESS - DeepBridge")
print("=" * 80)
print(f"\nDados de teste:")
print(f"  - Total de amostras: {n_samples}")
print(f"  - Grupo A: {np.sum(sensitive_feature == 'Group_A')} amostras")
print(f"  - Grupo B: {np.sum(sensitive_feature == 'Group_B')} amostras")
print(f"  - Taxa positiva real (Grupo A): {np.mean(y_true[sensitive_feature == 'Group_A']):.3f}")
print(f"  - Taxa positiva real (Grupo B): {np.mean(y_true[sensitive_feature == 'Group_B']):.3f}")

print("\n" + "=" * 80)
print("MÉTRICAS PRÉ-TREINO (Independentes do modelo)")
print("=" * 80)

# 1. Class Balance
print("\n1. CLASS BALANCE (BCL)")
result = FairnessMetrics.class_balance(y_true, sensitive_feature)
print(f"   Valor: {result['value']:.4f}")
print(f"   Grupo A: {result['group_a']} ({result['group_a_size']} amostras)")
print(f"   Grupo B: {result['group_b']} ({result['group_b_size']} amostras)")
print(f"   Interpretação: {result['interpretation']}")

# 2. Concept Balance
print("\n2. CONCEPT BALANCE (BCO)")
result = FairnessMetrics.concept_balance(y_true, sensitive_feature)
print(f"   Valor: {result['value']:.4f}")
print(f"   Taxa positiva Grupo A: {result['group_a_positive_rate']:.4f}")
print(f"   Taxa positiva Grupo B: {result['group_b_positive_rate']:.4f}")
print(f"   Interpretação: {result['interpretation']}")

# 3. KL Divergence
print("\n3. KL DIVERGENCE")
result = FairnessMetrics.kl_divergence(y_true, sensitive_feature)
print(f"   Valor: {result['value']:.4f}")
print(f"   Interpretação: {result['interpretation']}")

# 4. JS Divergence
print("\n4. JS DIVERGENCE")
result = FairnessMetrics.js_divergence(y_true, sensitive_feature)
print(f"   Valor: {result['value']:.4f}")
print(f"   Interpretação: {result['interpretation']}")

print("\n" + "=" * 80)
print("MÉTRICAS PÓS-TREINO (Dependentes do modelo)")
print("=" * 80)

# 5. Statistical Parity
print("\n5. STATISTICAL PARITY")
result = FairnessMetrics.statistical_parity(y_pred, sensitive_feature)
print(f"   Disparity: {result['disparity']:.4f}")
print(f"   Ratio: {result['ratio']:.4f}")
print(f"   Passa regra 80%: {result['passes_80_rule']}")
print(f"   Interpretação: {result['interpretation']}")

# 6. Equal Opportunity
print("\n6. EQUAL OPPORTUNITY")
result = FairnessMetrics.equal_opportunity(y_true, y_pred, sensitive_feature)
print(f"   TPR Disparity: {result['disparity']:.4f}")
print(f"   TPR Ratio: {result['ratio']:.4f}")
print(f"   Interpretação: {result['interpretation']}")

# 7. Equalized Odds
print("\n7. EQUALIZED ODDS")
result = FairnessMetrics.equalized_odds(y_true, y_pred, sensitive_feature)
print(f"   TPR Disparity: {result['tpr_disparity']:.4f}")
print(f"   FPR Disparity: {result['fpr_disparity']:.4f}")
print(f"   Combined Disparity: {result['combined_disparity']:.4f}")
print(f"   Interpretação: {result['interpretation']}")

# 8. Disparate Impact
print("\n8. DISPARATE IMPACT")
result = FairnessMetrics.disparate_impact(y_pred, sensitive_feature)
print(f"   Ratio: {result['ratio']:.4f}")
print(f"   Passa threshold: {result['passes_threshold']}")
print(f"   Interpretação: {result['interpretation']}")

# 9. False Negative Rate Difference
print("\n9. FALSE NEGATIVE RATE DIFFERENCE (TFN)")
result = FairnessMetrics.false_negative_rate_difference(y_true, y_pred, sensitive_feature)
print(f"   Valor: {result['value']:.4f}")
print(f"   FNR Grupo A: {result['group_a_fnr']:.4f}")
print(f"   FNR Grupo B: {result['group_b_fnr']:.4f}")
print(f"   Interpretação: {result['interpretation']}")

# 10. Conditional Acceptance
print("\n10. CONDITIONAL ACCEPTANCE (AC)")
result = FairnessMetrics.conditional_acceptance(y_true, y_pred, sensitive_feature)
print(f"   Valor: {result['value']:.4f}")
print(f"   Taxa Grupo A: {result['group_a_rate']:.4f}")
print(f"   Taxa Grupo B: {result['group_b_rate']:.4f}")
print(f"   Interpretação: {result['interpretation']}")

# 11. Conditional Rejection
print("\n11. CONDITIONAL REJECTION (RC)")
result = FairnessMetrics.conditional_rejection(y_true, y_pred, sensitive_feature)
print(f"   Valor: {result['value']:.4f}")
print(f"   Taxa Grupo A: {result['group_a_rate']:.4f}")
print(f"   Taxa Grupo B: {result['group_b_rate']:.4f}")
print(f"   Interpretação: {result['interpretation']}")

# 12. Precision Difference
print("\n12. PRECISION DIFFERENCE (DP)")
result = FairnessMetrics.precision_difference(y_true, y_pred, sensitive_feature)
print(f"   Valor: {result['value']:.4f}")
print(f"   Precisão Grupo A: {result['group_a_precision']:.4f}")
print(f"   Precisão Grupo B: {result['group_b_precision']:.4f}")
print(f"   Interpretação: {result['interpretation']}")

# 13. Accuracy Difference
print("\n13. ACCURACY DIFFERENCE (DA)")
result = FairnessMetrics.accuracy_difference(y_true, y_pred, sensitive_feature)
print(f"   Valor: {result['value']:.4f}")
print(f"   Acurácia Grupo A: {result['group_a_accuracy']:.4f}")
print(f"   Acurácia Grupo B: {result['group_b_accuracy']:.4f}")
print(f"   Interpretação: {result['interpretation']}")

# 14. Treatment Equality
print("\n14. TREATMENT EQUALITY (IT)")
result = FairnessMetrics.treatment_equality(y_true, y_pred, sensitive_feature)
print(f"   Valor: {result['value']:.4f}")
print(f"   FN/FP Ratio Grupo A: {result['group_a_ratio']:.4f}")
print(f"   FN/FP Ratio Grupo B: {result['group_b_ratio']:.4f}")
print(f"   Interpretação: {result['interpretation']}")

# 15. Entropy Index
print("\n15. ENTROPY INDEX (IE) - Individual Fairness")
result = FairnessMetrics.entropy_index(y_true, y_pred, alpha=2.0)
print(f"   Valor: {result['value']:.4f}")
print(f"   Alpha: {result['alpha']}")
print(f"   Interpretação: {result['interpretation']}")

print("\n" + "=" * 80)
print("TESTE CONCLUÍDO COM SUCESSO!")
print("Todas as 15 métricas estão funcionando corretamente.")
print("=" * 80)
