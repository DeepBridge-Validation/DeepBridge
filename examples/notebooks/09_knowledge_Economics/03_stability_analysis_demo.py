#!/usr/bin/env python3
"""
Coefficient Stability Analysis via Bootstrap
=============================================

Demonstra análise de estabilidade de coeficientes via bootstrap resampling,
conforme descrito no paper Section 3.4 e 4.3.

Componentes:
- Bootstrap resampling (1000 amostras)
- Coeficiente de variação (CV)
- Intervalos de confiança (95%)
- Sign stability

Critérios de aceitação (Paper):
- CV < 0.15
- Sign stability > 95%
- CI não cruza zero (se efeito teoricamente não-nulo)
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*70)
print("COEFFICIENT STABILITY ANALYSIS VIA BOOTSTRAP")
print("Knowledge Distillation for Economics - Section 3.4 & 4.3")
print("="*70)


# ============================================================================
# 1. GERAÇÃO DE DATASET
# ============================================================================

print("\n1. Gerando dataset sintético de crédito...")

def generate_credit_dataset(n_samples=5000):
    """Gera dataset sintético simplificado."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_classes=2,
        class_sep=0.8,
        flip_y=0.1,
        random_state=RANDOM_STATE
    )

    feature_names = [
        'income', 'dti_ratio', 'interest_rate', 'age', 'employment_length',
        'credit_score', 'loan_amount', 'num_accounts', 'delinq_2yrs', 'revolving_util'
    ]

    df = pd.DataFrame(X, columns=feature_names)

    # Ajustar features
    df['income'] = np.abs(df['income']) * 30000 + 20000
    df['dti_ratio'] = (df['dti_ratio'] - df['dti_ratio'].min()) / \
                      (df['dti_ratio'].max() - df['dti_ratio'].min()) * 50
    df['interest_rate'] = np.abs(df['interest_rate']) * 15 + 5

    return df, y

X, y = generate_credit_dataset(n_samples=5000)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)

print(f"   Dataset: {X.shape}")
print(f"   Train: {len(X_train)}, Test: {len(X_test)}")


# ============================================================================
# 2. TREINAR TEACHER
# ============================================================================

print("\n2. Treinando modelo TEACHER (Gradient Boosting)...")

teacher = GradientBoostingClassifier(
    n_estimators=50,
    max_depth=4,
    random_state=RANDOM_STATE
)

teacher.fit(X_train, y_train)
teacher_auc = roc_auc_score(y_test, teacher.predict_proba(X_test)[:, 1])

print(f"   Teacher AUC: {teacher_auc:.4f}")


# ============================================================================
# 3. TREINAR STUDENT INTERPRETÁVEL
# ============================================================================

print("\n3. Treinando STUDENT (Logistic Regression)...")

student = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
student.fit(X_train, y_train)

student_auc = roc_auc_score(y_test, student.predict_proba(X_test)[:, 1])

print(f"   Student AUC: {student_auc:.4f}")

# Coeficientes originais
original_coefs = student.coef_[0]
print(f"\n   Coeficientes originais:")
for i, (feat, coef) in enumerate(zip(X.columns, original_coefs)):
    print(f"      {feat:20} → {coef:+.4f}")


# ============================================================================
# 4. BOOTSTRAP ANALYSIS (Core Contribution)
# ============================================================================

print("\n4. Executando análise de estabilidade via BOOTSTRAP...")
print("   " + "-"*66)

N_BOOTSTRAP = 1000
print(f"   Número de amostras bootstrap: {N_BOOTSTRAP}")
print("   (Pode levar 30-60 segundos...)")

# Armazenar coeficientes de cada bootstrap
bootstrap_coefs = []

for b in range(N_BOOTSTRAP):
    # Bootstrap sample
    indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
    X_boot = X_train.iloc[indices]
    y_boot = y_train[indices]  # y_train is numpy array, not pandas Series

    # Treinar student no bootstrap sample
    student_boot = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    student_boot.fit(X_boot, y_boot)

    # Extrair coeficientes
    bootstrap_coefs.append(student_boot.coef_[0])

    # Progress
    if (b + 1) % 200 == 0:
        print(f"      Progress: {b+1}/{N_BOOTSTRAP}")

bootstrap_coefs = np.array(bootstrap_coefs)  # Shape: (N_BOOTSTRAP, n_features)

print("   ✅ Bootstrap concluído!")


# ============================================================================
# 5. CALCULAR MÉTRICAS DE ESTABILIDADE
# ============================================================================

print("\n5. Calculando métricas de estabilidade...")

# Estatísticas bootstrap
coef_mean = np.mean(bootstrap_coefs, axis=0)
coef_std = np.std(bootstrap_coefs, axis=0)
coef_median = np.median(bootstrap_coefs, axis=0)

# Coeficiente de variação (CV) - Métrica chave do paper
# CV = std / mean (para coeficientes, usamos abs(mean))
coef_cv = coef_std / (np.abs(coef_mean) + 1e-10)

# Intervalos de confiança (95%)
ci_lower = np.percentile(bootstrap_coefs, 2.5, axis=0)
ci_upper = np.percentile(bootstrap_coefs, 97.5, axis=0)

# Sign stability (porcentagem de amostras com sinal consistente)
signs = np.sign(bootstrap_coefs)
mode_sign = stats.mode(signs, axis=0, keepdims=False)[0]
sign_stability = np.mean(signs == mode_sign, axis=0)

print("\n   Coeficientes e estatísticas de estabilidade:")
print("   " + "-"*66)

results = []

for i, feat in enumerate(X.columns):
    result = {
        'Feature': feat,
        'Mean': coef_mean[i],
        'Std': coef_std[i],
        'CV': coef_cv[i],
        'CI_Lower': ci_lower[i],
        'CI_Upper': ci_upper[i],
        'Sign_Stability': sign_stability[i] * 100
    }
    results.append(result)

df_results = pd.DataFrame(results)

# Mostrar features principais (top 5 por importância)
top_features = df_results.nlargest(5, 'Mean' if df_results['Mean'].abs().sum() > 0 else 'CV')

print(f"\n   {'Feature':<20} {'Mean':>8} {'Std':>7} {'CV':>7} {'Sign%':>7} Status")
print("   " + "-"*66)

for _, row in df_results.iterrows():
    feat = row['Feature']
    mean = row['Mean']
    std = row['Std']
    cv = row['CV']
    sign_pct = row['Sign_Stability']

    # Critérios de aceitação do paper
    cv_ok = cv < 0.15
    sign_ok = sign_pct >= 95.0

    status = "✅" if (cv_ok and sign_ok) else "⚠️"

    print(f"   {feat:<20} {mean:+8.4f} {std:7.4f} {cv:7.3f} {sign_pct:6.1f}%  {status}")

print("   " + "-"*66)


# ============================================================================
# 6. VERIFICAR CRITÉRIOS DE ACEITAÇÃO (Paper Section 3.4.3)
# ============================================================================

print("\n6. Verificação de critérios de aceitação (Paper Section 3.4.3)...")
print("   " + "-"*66)

# Critérios do paper:
# 1. CV < 0.15
# 2. Sign stability >= 95%
# 3. CI não cruza zero (para features teoricamente importantes)

n_features = len(X.columns)
n_cv_pass = np.sum(coef_cv < 0.15)
n_sign_pass = np.sum(sign_stability >= 0.95)

print(f"   Critério 1: CV < 0.15")
print(f"      Aprovados: {n_cv_pass}/{n_features} ({n_cv_pass/n_features*100:.1f}%)")
print(f"      Média CV:  {coef_cv.mean():.3f}")

print(f"\n   Critério 2: Sign Stability >= 95%")
print(f"      Aprovados: {n_sign_pass}/{n_features} ({n_sign_pass/n_features*100:.1f}%)")
print(f"      Média:     {sign_stability.mean()*100:.1f}%")

print(f"\n   Critério 3: CI não cruza zero (features principais)")
ci_crosses_zero = (ci_lower * ci_upper) < 0
n_ci_pass = np.sum(~ci_crosses_zero)
print(f"      Não cruzam zero: {n_ci_pass}/{n_features} ({n_ci_pass/n_features*100:.1f}%)")

print("   " + "-"*66)

# Overall stability score
overall_cv_ok = coef_cv.mean() < 0.15
overall_sign_ok = sign_stability.mean() >= 0.95

if overall_cv_ok and overall_sign_ok:
    print(f"   ✅ MODELO ESTÁVEL para inferência econômica")
else:
    print(f"   ⚠️  MODELO PODE NÃO SER ESTÁVEL para inferência rigorosa")


# ============================================================================
# 7. VISUALIZAÇÃO DE DISTRIBUIÇÕES (Top 3 Features)
# ============================================================================

print("\n7. Distribuições bootstrap (top 3 features por |mean|)...")

top3_indices = np.argsort(np.abs(coef_mean))[-3:][::-1]

for idx in top3_indices:
    feat = X.columns[idx]
    coefs = bootstrap_coefs[:, idx]

    print(f"\n   {feat}:")
    print(f"      Mean:  {coef_mean[idx]:+.4f}")
    print(f"      Std:   {coef_std[idx]:.4f}")
    print(f"      CV:    {coef_cv[idx]:.3f}")
    print(f"      95% CI: [{ci_lower[idx]:+.4f}, {ci_upper[idx]:+.4f}]")
    print(f"      Sign:  {sign_stability[idx]*100:.1f}% consistent")


# ============================================================================
# 8. COMPARAÇÃO COM PAPER
# ============================================================================

print("\n" + "="*70)
print("RESULTADOS FINAIS - STABILITY ANALYSIS")
print("="*70)

print(f"\n📊 MÉTRICAS DE ESTABILIDADE:")
print(f"   Média CV:                {coef_cv.mean():.3f}")
print(f"   Média Sign Stability:    {sign_stability.mean()*100:.1f}%")
print(f"   Features com CV < 0.15:  {n_cv_pass}/{n_features} ({n_cv_pass/n_features*100:.1f}%)")

print(f"\n📖 VALORES ESPERADOS DO PAPER (Section 5.2.3):")
print(f"   Média CV:                < 0.15")
print(f"   Sign Stability:          > 95%")
print(f"   Features estáveis:       100% (features principais)")

print(f"\n💡 INTERPRETAÇÃO:")
if coef_cv.mean() < 0.15:
    print(f"   ✅ Coeficientes ESTÁVEIS - confiáveis para policy analysis")
else:
    print(f"   ⚠️  Coeficientes podem ser INSTÁVEIS - cautela em interpretação")

if sign_stability.mean() >= 0.95:
    print(f"   ✅ Sinais CONSISTENTES - relações econômicas preservadas")
else:
    print(f"   ⚠️  Sinais INCONSISTENTES - relações podem variar")

print(f"\n📝 USO EM ECONOMIA:")
print(f"   - CV < 0.15: Coeficientes estáveis para inferência estatística")
print(f"   - Sign > 95%: Relações econômicas confiáveis")
print(f"   - CI não cruza zero: Efeito significativo e estável")
print(f"   - Permite: Testes de hipótese, análise de políticas, interpretação causal")

print("\n" + "="*70)
print("✅ Demo concluída!")
print("\n💡 PRÓXIMOS PASSOS:")
print("   Execute 04_structural_breaks_demo.py para detecção de quebras estruturais")
print("="*70)
