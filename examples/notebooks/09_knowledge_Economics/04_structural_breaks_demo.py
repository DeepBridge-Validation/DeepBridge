#!/usr/bin/env python3
"""
Structural Break Detection via Rolling Window Analysis
=======================================================

Demonstra detecção de quebras estruturais em relações econômicas usando
rolling window analysis e testes de Wald, conforme Paper Section 3.5 e 5.2.3.

Exemplo: Detecção de mudança em coeficientes de risco de crédito pré/pós-crise 2008.

Componentes:
- Rolling window analysis
- Teste de Wald para quebras estruturais
- Identificação de features que mudaram
- Interpretação econômica
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*70)
print("STRUCTURAL BREAK DETECTION VIA ROLLING WINDOWS")
print("Knowledge Distillation for Economics - Section 3.5 & 5.2.3")
print("="*70)


# ============================================================================
# 1. GERAÇÃO DE DATASET TEMPORAL COM QUEBRA ESTRUTURAL
# ============================================================================

print("\n1. Gerando dataset temporal de crédito (2005-2015)...")
print("   Simulando quebra estrutural em 2008 (crise financeira)")

def generate_temporal_credit_dataset(
    n_samples=8000,
    start_date='2005-01-01',
    end_date='2015-12-31',
    crisis_date='2008-09-15'  # Lehman Brothers collapse
):
    """
    Gera dataset temporal com quebra estrutural em 2008.

    Mudança pós-2008:
    - DTI Ratio: Coeficiente aumenta (+52%)
    - Interest Rate: Efeito mais forte
    - Income: Efeito se mantém
    """
    # Datas
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    crisis = pd.to_datetime(crisis_date)

    dates = pd.date_range(start, end, periods=n_samples)

    # Features
    income = np.random.lognormal(10.5, 0.8, n_samples)  # ~20k-200k
    dti_ratio = np.random.uniform(5, 50, n_samples)  # 5-50%
    interest_rate = np.random.uniform(5, 25, n_samples)  # 5-25%
    age = np.random.uniform(18, 70, n_samples)
    credit_score = np.random.uniform(300, 850, n_samples)

    # Criar DataFrame
    df = pd.DataFrame({
        'date': dates,
        'income': income,
        'dti_ratio': dti_ratio,
        'interest_rate': interest_rate,
        'age': age,
        'credit_score': credit_score
    })

    # Gerar target com quebra estrutural
    is_post_crisis = df['date'] >= crisis

    # Coeficientes pré-crise
    beta_income_pre = -0.00002
    beta_dti_pre = 0.015  # Baseline
    beta_rate_pre = 0.04

    # Coeficientes pós-crise (MUDANÇA ESTRUTURAL)
    beta_income_post = -0.00002  # Mantém
    beta_dti_post = 0.0228  # +52% (conforme paper)
    beta_rate_post = 0.06  # +50%

    # Calcular logit
    logit = np.zeros(n_samples)

    for i in range(n_samples):
        if is_post_crisis.iloc[i]:
            # Pós-crise: maior sensibilidade a risco
            logit[i] = (
                beta_income_post * df['income'].iloc[i] +
                beta_dti_post * df['dti_ratio'].iloc[i] +
                beta_rate_post * df['interest_rate'].iloc[i] +
                np.random.normal(0, 1)
            )
        else:
            # Pré-crise
            logit[i] = (
                beta_income_pre * df['income'].iloc[i] +
                beta_dti_pre * df['dti_ratio'].iloc[i] +
                beta_rate_pre * df['interest_rate'].iloc[i] +
                np.random.normal(0, 1)
            )

    # Probabilidade de default
    y_proba = 1 / (1 + np.exp(-logit))
    y = (y_proba > 0.5).astype(int)

    return df, y, crisis

X, y, crisis_date = generate_temporal_credit_dataset(n_samples=8000)

print(f"   Dataset shape: {X.shape}")
print(f"   Período: {X['date'].min().date()} a {X['date'].max().date()}")
print(f"   Quebra estrutural em: {crisis_date.date()}")
print(f"   Amostras pré-crise:  {(X['date'] < crisis_date).sum()}")
print(f"   Amostras pós-crise:  {(X['date'] >= crisis_date).sum()}")
print(f"   Default rate: {y.mean():.2%}")


# ============================================================================
# 2. TREINAR TEACHER GLOBAL
# ============================================================================

print("\n2. Treinando modelo TEACHER global (todo período)...")

# Features para modelagem (excluir date)
feature_cols = ['income', 'dti_ratio', 'interest_rate', 'age', 'credit_score']
X_features = X[feature_cols]

teacher = GradientBoostingClassifier(
    n_estimators=50,
    max_depth=4,
    random_state=RANDOM_STATE
)

teacher.fit(X_features, y)

teacher_auc = roc_auc_score(y, teacher.predict_proba(X_features)[:, 1])
print(f"   Teacher AUC: {teacher_auc:.4f}")


# ============================================================================
# 3. ROLLING WINDOW ANALYSIS
# ============================================================================

print("\n3. Executando Rolling Window Analysis...")
print("   " + "-"*66)

WINDOW_SIZE = 1000  # Amostras por janela
STEP_SIZE = 200     # Passo entre janelas

# Ordenar por data
X_sorted = X.sort_values('date').reset_index(drop=True)
# y is numpy array, need to reorder based on original indices
sort_indices = X.sort_values('date').index
y_sorted = pd.Series(y[sort_indices]).reset_index(drop=True)

print(f"   Window size: {WINDOW_SIZE} amostras")
print(f"   Step size:   {STEP_SIZE} amostras")

# Armazenar resultados
windows = []
coefficients = []
window_dates = []
aucs = []

n_windows = (len(X_sorted) - WINDOW_SIZE) // STEP_SIZE + 1

print(f"   Número de janelas: {n_windows}")
print(f"   (Pode levar 20-30 segundos...)")

for i in range(0, len(X_sorted) - WINDOW_SIZE + 1, STEP_SIZE):
    start_idx = i
    end_idx = i + WINDOW_SIZE

    # Dados da janela
    X_window = X_sorted.iloc[start_idx:end_idx][feature_cols]
    y_window = y_sorted.iloc[start_idx:end_idx]

    # Data média da janela
    date_window = X_sorted.iloc[start_idx:end_idx]['date'].mean()

    # Treinar student na janela
    student_window = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    student_window.fit(X_window, y_window)

    # Extrair coeficientes
    coefs = student_window.coef_[0]

    # AUC na janela
    auc = roc_auc_score(y_window, student_window.predict_proba(X_window)[:, 1])

    # Armazenar
    windows.append((start_idx, end_idx))
    coefficients.append(coefs)
    window_dates.append(date_window)
    aucs.append(auc)

coefficients = np.array(coefficients)  # Shape: (n_windows, n_features)

print(f"   ✅ Rolling window concluído!")
print(f"   Total de janelas: {len(windows)}")


# ============================================================================
# 4. DETECÇÃO DE QUEBRAS ESTRUTURAIS (Teste de Wald)
# ============================================================================

print("\n4. Detectando quebras estruturais (Teste de Wald)...")
print("   " + "-"*66)

def wald_test_for_break(coef_t, coef_t1, n_features):
    """
    Teste de Wald simplificado para quebra estrutural.

    W = (β_{t+1} - β_t)^T Σ^{-1} (β_{t+1} - β_t)

    Simplificação: assumir Σ = I (identidade)
    """
    diff = coef_t1 - coef_t
    W = np.sum(diff ** 2)

    # Chi-squared test
    p_value = 1 - stats.chi2.cdf(W, df=n_features)

    return W, p_value

breaks = []

for t in range(len(coefficients) - 1):
    coef_t = coefficients[t]
    coef_t1 = coefficients[t + 1]

    W, p_value = wald_test_for_break(coef_t, coef_t1, len(feature_cols))

    # Detectar quebra (α = 0.05)
    if p_value < 0.05:
        # Identificar features que mais mudaram
        diff = coef_t1 - coef_t
        rel_change = np.abs(diff / (np.abs(coef_t) + 1e-10))

        # Top feature que mudou
        top_feature_idx = np.argmax(rel_change)
        top_feature = feature_cols[top_feature_idx]

        breaks.append({
            'window': t,
            'date': window_dates[t],
            'statistic': W,
            'p_value': p_value,
            'changed_feature': top_feature,
            'coef_before': coef_t[top_feature_idx],
            'coef_after': coef_t1[top_feature_idx],
            'change_pct': rel_change[top_feature_idx] * 100
        })

print(f"   Quebras detectadas: {len(breaks)}")

if breaks:
    print(f"\n   Principais quebras estruturais:")
    for i, brk in enumerate(breaks[:5], 1):  # Top 5
        print(f"\n   Quebra {i}:")
        print(f"      Data (aproximada):  {brk['date'].date()}")
        print(f"      Estatística Wald:   {brk['statistic']:.3f}")
        print(f"      P-value:            {brk['p_value']:.4f}")
        print(f"      Feature mudada:     {brk['changed_feature']}")
        print(f"      Coef. antes:        {brk['coef_before']:+.5f}")
        print(f"      Coef. depois:       {brk['coef_after']:+.5f}")
        print(f"      Mudança:            {brk['change_pct']:+.1f}%")


# ============================================================================
# 5. ANÁLISE PRÉ/PÓS-CRISE 2008 (Specific Analysis)
# ============================================================================

print("\n5. Análise específica: Pré/Pós-Crise 2008...")
print("   " + "-"*66)

# Separar dados
pre_crisis_mask = X['date'] < crisis_date
post_crisis_mask = X['date'] >= crisis_date

X_pre = X[pre_crisis_mask][feature_cols]
y_pre = y[pre_crisis_mask]

X_post = X[post_crisis_mask][feature_cols]
y_post = y[post_crisis_mask]

print(f"   Pré-2008:  {len(X_pre)} amostras")
print(f"   Pós-2008:  {len(X_post)} amostras")

# Treinar modelos separados
student_pre = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
student_pre.fit(X_pre, y_pre)

student_post = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
student_post.fit(X_post, y_post)

# Comparar coeficientes
coefs_pre = student_pre.coef_[0]
coefs_post = student_post.coef_[0]

print(f"\n   Comparação de coeficientes Pré/Pós-Crise:")
print(f"\n   {'Feature':<20} {'Pré-2008':>12} {'Pós-2008':>12} {'Mudança':>12}")
print("   " + "-"*66)

for i, feat in enumerate(feature_cols):
    pre = coefs_pre[i]
    post = coefs_post[i]
    change_pct = ((post - pre) / (abs(pre) + 1e-10)) * 100

    print(f"   {feat:<20} {pre:+12.5f} {post:+12.5f} {change_pct:+11.1f}%")

print("   " + "-"*66)

# Feature com maior mudança (DTI conforme paper)
max_change_idx = np.argmax(np.abs(coefs_post - coefs_pre) / (np.abs(coefs_pre) + 1e-10))
max_change_feature = feature_cols[max_change_idx]
change_pct_max = ((coefs_post[max_change_idx] - coefs_pre[max_change_idx]) /
                  (abs(coefs_pre[max_change_idx]) + 1e-10)) * 100

print(f"\n   Feature com MAIOR mudança: {max_change_feature}")
print(f"   Mudança: {change_pct_max:+.1f}%")


# ============================================================================
# 6. INTERPRETAÇÃO ECONÔMICA
# ============================================================================

print("\n6. Interpretação econômica da quebra estrutural...")
print("   " + "-"*66)

print(f"\n   💡 INTERPRETAÇÃO:")
print(f"   Crise de 2008 alterou fundamentalmente as relações de risco de crédito:")
print(f"")
print(f"   1. DTI Ratio (Debt-to-Income):")
print(f"      - Pré-2008:  Sensibilidade moderada")
print(f"      - Pós-2008:  Sensibilidade aumentou ~50%+")
print(f"      - Razão:     Mercados mais cautelosos com endividamento")
print(f"")
print(f"   2. Interest Rate:")
print(f"      - Mudança:   Efeito mais pronunciado pós-crise")
print(f"      - Razão:     Taxas refletem melhor o risco percebido")
print(f"")
print(f"   3. Income:")
print(f"      - Mudança:   Relativamente estável")
print(f"      - Razão:     Relação fundamental não alterada pela crise")

print("   " + "-"*66)


# ============================================================================
# 7. RESULTADOS FINAIS
# ============================================================================

print("\n" + "="*70)
print("RESULTADOS FINAIS - STRUCTURAL BREAK DETECTION")
print("="*70)

print(f"\n📊 DETECÇÃO DE QUEBRAS:")
print(f"   Quebras detectadas:         {len(breaks)}")
print(f"   Principal quebra em:        {breaks[0]['date'].date() if breaks else 'N/A'}")
print(f"   Feature mais afetada:       {max_change_feature}")
print(f"   Magnitude da mudança:       {change_pct_max:+.1f}%")

print(f"\n📖 VALORES ESPERADOS DO PAPER (Section 5.2.3):")
print(f"   Quebra detectada:           Q4 2008")
print(f"   Feature principal:          DTI Ratio")
print(f"   Mudança DTI:                +52%")
print(f"   Interpretação:              Crise aumentou sensibilidade a endividamento")

print(f"\n💡 USO EM ECONOMIA:")
print(f"   - Identifica mudanças em relações econômicas")
print(f"   - Permite análise pré/pós eventos (crises, políticas)")
print(f"   - Mantém interpretabilidade durante mudanças")
print(f"   - Auxilia em adaptive modeling e policy evaluation")

print("\n" + "="*70)
print("✅ Demo concluída!")
print("\n💡 PRÓXIMOS PASSOS:")
print("   Execute 05_complete_demo.py para demo completo integrando todos componentes")
print("="*70)
