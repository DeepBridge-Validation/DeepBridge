"""
Script de Verificação Prévia - Análise de Fairness
===================================================

Execute este script ANTES da análise de fairness para verificar
se seus dados estão no formato correto.
"""

import pandas as pd
from pathlib import Path

# ============================================================================
# CONFIGURAÇÕES
# ============================================================================

DATA_DIR = Path("/home/guhaase/projetos/DeepBridge/simular_lib/analise_v4")
TRAIN_PATH = DATA_DIR / "train_predictions.parquet"
TEST_PATH = DATA_DIR / "test_predictions.parquet"

# Colunas esperadas
EXPECTED_COLS = {
    'target': 'in_cmst_fun',
    'proba_0': 'pred_proba_class_0',
    'proba_1': 'pred_proba_class_1',
    'pred_class': 'pred_class',
    'protected': ['nm_tip_gnr', 'nm_tip_raca', 'vl_idd_aa']
}

# ============================================================================
# VERIFICAÇÃO
# ============================================================================

print("=" * 80)
print("VERIFICAÇÃO PRÉVIA - DADOS PARA ANÁLISE DE FAIRNESS")
print("=" * 80)

all_ok = True

# 1. Verificar se arquivos existem
print("\n1. Verificando existência dos arquivos...")

for name, path in [("Train", TRAIN_PATH), ("Test", TEST_PATH)]:
    if path.exists():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"   ✓ {name}: {path.name} ({size_mb:.1f} MB)")
    else:
        print(f"   ✗ {name}: NÃO ENCONTRADO - {path}")
        all_ok = False

if not all_ok:
    print("\n❌ ERRO: Arquivos não encontrados!")
    print(f"   Verifique se os arquivos estão em: {DATA_DIR}")
    exit(1)

# 2. Carregar e verificar estrutura
print("\n2. Carregando dados...")

try:
    df_train = pd.read_parquet(TRAIN_PATH)
    df_test = pd.read_parquet(TEST_PATH)
    print(f"   ✓ Train: {df_train.shape}")
    print(f"   ✓ Test: {df_test.shape}")
except Exception as e:
    print(f"   ✗ ERRO ao carregar: {e}")
    exit(1)

# 3. Verificar colunas
print("\n3. Verificando colunas necessárias...")

df = df_test  # Usar test para verificação

# Target
if EXPECTED_COLS['target'] in df.columns:
    print(f"   ✓ Target: {EXPECTED_COLS['target']}")
else:
    print(f"   ✗ Target: {EXPECTED_COLS['target']} NÃO ENCONTRADA")
    all_ok = False

# Probabilidades
for proba in ['proba_0', 'proba_1']:
    col = EXPECTED_COLS[proba]
    if col in df.columns:
        print(f"   ✓ Probabilidade: {col}")
    else:
        print(f"   ✗ Probabilidade: {col} NÃO ENCONTRADA")
        all_ok = False

# Classe predita
if EXPECTED_COLS['pred_class'] in df.columns:
    print(f"   ✓ Classe predita: {EXPECTED_COLS['pred_class']}")
else:
    print(f"   ✗ Classe predita: {EXPECTED_COLS['pred_class']} NÃO ENCONTRADA")
    all_ok = False

# Atributos protegidos
print("\n4. Verificando atributos protegidos...")
found_protected = []
missing_protected = []

for attr in EXPECTED_COLS['protected']:
    if attr in df.columns:
        print(f"   ✓ {attr}")
        found_protected.append(attr)
    else:
        print(f"   ⚠️  {attr} NÃO ENCONTRADA (opcional)")
        missing_protected.append(attr)

if len(found_protected) == 0:
    print(f"\n   ❌ ERRO: Nenhum atributo protegido encontrado!")
    print(f"   É necessário ter pelo menos 1 atributo protegido para análise de fairness")
    all_ok = False
elif len(found_protected) < len(EXPECTED_COLS['protected']):
    print(f"\n   ⚠️  AVISO: {len(missing_protected)} atributo(s) não encontrado(s)")
    print(f"   Análise será feita apenas com: {found_protected}")

# 5. Verificar valores
print("\n5. Verificando valores das colunas...")

target_col = EXPECTED_COLS['target']
pred_col = EXPECTED_COLS['pred_class']

# Target
if target_col in df.columns:
    unique_vals = df[target_col].unique()
    print(f"   Target ({target_col}):")
    print(f"      Valores únicos: {sorted(unique_vals)}")

    if set(unique_vals) == {0, 1}:
        print(f"      ✓ Classificação binária (0, 1)")
    else:
        print(f"      ⚠️  AVISO: Valores não são apenas 0 e 1")

    # Distribuição
    for val in sorted(unique_vals):
        count = (df[target_col] == val).sum()
        pct = count / len(df) * 100
        print(f"      Classe {val}: {count:,} ({pct:.1f}%)")

# Predições
if pred_col in df.columns:
    unique_vals = df[pred_col].unique()
    print(f"\n   Predições ({pred_col}):")
    print(f"      Valores únicos: {sorted(unique_vals)}")

    for val in sorted(unique_vals):
        count = (df[pred_col] == val).sum()
        pct = count / len(df) * 100
        print(f"      Classe {val}: {count:,} ({pct:.1f}%)")

# Probabilidades
proba_cols = [EXPECTED_COLS['proba_0'], EXPECTED_COLS['proba_1']]
if all(col in df.columns for col in proba_cols):
    print(f"\n   Probabilidades:")
    for col in proba_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        mean_val = df[col].mean()
        print(f"      {col}:")
        print(f"         Min: {min_val:.4f}, Max: {max_val:.4f}, Mean: {mean_val:.4f}")

        if min_val < 0 or max_val > 1:
            print(f"         ⚠️  AVISO: Valores fora do range [0, 1]")

    # Verificar soma = 1
    prob_sum = df[proba_cols].sum(axis=1)
    if (prob_sum.round(4) == 1.0).all():
        print(f"      ✓ Soma das probabilidades = 1.0")
    else:
        print(f"      ⚠️  AVISO: Soma das probabilidades != 1.0 para algumas linhas")
        print(f"         Min soma: {prob_sum.min():.4f}, Max soma: {prob_sum.max():.4f}")

# 6. Verificar valores ausentes
print("\n6. Verificando valores ausentes (NaN)...")

critical_cols = [target_col, pred_col] + proba_cols + found_protected
has_nan = False

for col in critical_cols:
    if col in df.columns:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            pct = nan_count / len(df) * 100
            print(f"   ⚠️  {col}: {nan_count:,} NaN ({pct:.1f}%)")
            has_nan = True

if not has_nan:
    print(f"   ✓ Nenhum valor ausente em colunas críticas")
else:
    print(f"\n   ⚠️  ATENÇÃO: Linhas com NaN serão removidas automaticamente na análise")

# 7. Análise dos atributos protegidos
print("\n7. Análise dos atributos protegidos...")

for attr in found_protected:
    print(f"\n   📊 {attr}:")

    # Valores únicos
    unique_vals = df[attr].value_counts()
    print(f"      Total de grupos: {len(unique_vals)}")

    # Top 5
    print(f"      Top 5 grupos:")
    for val, count in unique_vals.head(5).items():
        pct = count / len(df) * 100
        print(f"         {val}: {count:,} ({pct:.1f}%)")

    # Verificar grupos muito pequenos
    min_group_size = unique_vals.min()
    min_group_pct = min_group_size / len(df) * 100

    if min_group_pct < 1.0:
        print(f"      ⚠️  AVISO: Menor grupo tem apenas {min_group_size:,} samples ({min_group_pct:.2f}%)")
        print(f"         Grupos muito pequenos podem ter métricas instáveis")

    # Taxa de aprovação por grupo
    if target_col in df.columns:
        print(f"      Taxa de aprovação por grupo:")
        approval_rates = df.groupby(attr)[target_col].mean()

        for val, rate in approval_rates.head(10).items():
            print(f"         {val}: {rate:.1%}")

        # Verificar disparidade
        max_rate = approval_rates.max()
        min_rate = approval_rates.min()
        disparity = max_rate - min_rate

        if disparity > 0.20:
            print(f"      ⚠️  ATENÇÃO: Grande disparidade detectada!")
            print(f"         Diferença max-min: {disparity:.1%}")
            print(f"         Isso sugere possível viés no modelo")

# 8. Resumo final
print("\n" + "=" * 80)
print("RESUMO DA VERIFICAÇÃO")
print("=" * 80)

if all_ok and len(found_protected) > 0:
    print("\n✅ DADOS PRONTOS PARA ANÁLISE DE FAIRNESS!")
    print(f"\n   Próximos passos:")
    print(f"   1. Executar: python analyze_fairness_production.py")
    print(f"      (análise completa, ~10-15 min)")
    print(f"   2. OU: python analyze_fairness_quick.py")
    print(f"      (análise rápida, ~5-8 min)")
    print(f"\n   Atributos protegidos que serão analisados:")
    for attr in found_protected:
        print(f"      - {attr}")

else:
    print("\n❌ PROBLEMAS ENCONTRADOS!")
    print(f"\n   Revise os erros acima antes de executar a análise")

    if not all_ok:
        print(f"   - Colunas necessárias faltando")

    if len(found_protected) == 0:
        print(f"   - Nenhum atributo protegido encontrado")
        print(f"\n   Sugestão: Verificar nomes das colunas:")
        print(f"   python -c \"import pandas as pd; df = pd.read_parquet('{TEST_PATH}'); print(df.columns.tolist())\"")

# 9. Informações adicionais
print("\n" + "=" * 80)
print("INFORMAÇÕES ADICIONAIS")
print("=" * 80)

print(f"\n📁 Localização dos dados:")
print(f"   {DATA_DIR}")

print(f"\n📊 Estatísticas:")
print(f"   Train samples: {len(df_train):,}")
print(f"   Test samples: {len(df_test):,}")
print(f"   Total: {len(df_train) + len(df_test):,}")

if target_col in df.columns:
    accuracy = (df[target_col] == df[pred_col]).mean()
    print(f"\n🎯 Performance (Test):")
    print(f"   Acurácia: {accuracy:.3f}")

print(f"\n📚 Documentação:")
print(f"   - README: FAIRNESS_PRODUCTION_ANALYSIS_README.md")
print(f"   - Tutorial: docs/FAIRNESS_TUTORIAL.md")
print(f"   - FAQ: docs/FAIRNESS_FAQ.md")

print("\n" + "=" * 80)
