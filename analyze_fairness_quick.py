"""
Análise Rápida de Fairness - Versão Simplificada
==================================================

Script simplificado para análise rápida de fairness em dados de produção.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings
warnings.filterwarnings('ignore')

from deepbridge.core.db_data import DBDataset
from deepbridge.core.experiment.experiment import Experiment

# ============================================================================
# CONFIGURAÇÕES
# ============================================================================

# Paths
DATA_DIR = Path("/home/guhaase/projetos/DeepBridge/simular_lib/analise_v4")
OUTPUT_DIR = Path('./fairness_quick_analysis')
OUTPUT_DIR.mkdir(exist_ok=True)

# Colunas
TARGET = 'in_cmst_fun'
PROBA_COLS = ['pred_proba_class_0', 'pred_proba_class_1']
PRED_COL = 'pred_class'

# Atributos protegidos - AJUSTE CONFORME NECESSÁRIO
PROTECTED_ATTRS = ['nm_tip_gnr', 'nm_tip_raca']

# ============================================================================
# CLASSE MODELO WRAPPER
# ============================================================================

class PrecomputedModel(BaseEstimator, ClassifierMixin):
    """Wrapper para usar predições pré-computadas"""

    def __init__(self, df, proba_cols):
        self.df = df.copy()
        self.proba_cols = proba_cols
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        return self

    def predict(self, X):
        indices = X.index if isinstance(X, pd.DataFrame) else range(len(X))
        return self.df.loc[indices, 'pred_class'].values

    def predict_proba(self, X):
        indices = X.index if isinstance(X, pd.DataFrame) else range(len(X))
        return self.df.loc[indices, self.proba_cols].values

# ============================================================================
# EXECUTAR ANÁLISE
# ============================================================================

print("=" * 80)
print("ANÁLISE RÁPIDA DE FAIRNESS")
print("=" * 80)

# 1. Carregar dados
print("\n1. Carregando dados...")
df = pd.read_parquet(DATA_DIR / "test_predictions.parquet")
print(f"   ✓ {len(df):,} amostras carregadas")

# 2. Análise exploratória básica
print("\n2. Estatísticas básicas:")
print(f"   Target (in_cmst_fun):")
print(f"      Classe 0: {(df[TARGET]==0).sum():,} ({(df[TARGET]==0).mean():.1%})")
print(f"      Classe 1: {(df[TARGET]==1).sum():,} ({(df[TARGET]==1).mean():.1%})")

print(f"\n   Por atributo protegido:")
for attr in PROTECTED_ATTRS:
    print(f"\n   {attr}:")
    for val in df[attr].value_counts().head(5).index:
        rate = df[df[attr] == val][TARGET].mean()
        count = len(df[df[attr] == val])
        print(f"      {val}: {count:,} samples, taxa aprovação: {rate:.1%}")

# 3. Criar grupos etários (opcional)
if 'vl_idd_aa' in df.columns:
    print("\n3. Criando grupos etários...")
    df['age_group'] = pd.cut(
        df['vl_idd_aa'],
        bins=[0, 30, 40, 50, 60, 100],
        labels=['18-30', '31-40', '41-50', '51-60', '60+']
    )
    PROTECTED_ATTRS.append('age_group')
    print(f"   ✓ Grupos criados: {df['age_group'].value_counts().to_dict()}")

# 4. Criar modelo wrapper
print("\n4. Preparando análise...")
model = PrecomputedModel(df, PROBA_COLS)
dataset = DBDataset(data=df, target_column=TARGET, model=model)

# 5. Criar experiment
print("\n5. Criando experiment...")
experiment = Experiment(
    dataset=dataset,
    experiment_type="binary_classification",
    tests=["fairness"],
    protected_attributes=PROTECTED_ATTRS,
    test_size=0.2,
    random_state=42
)

# 6. Executar análise completa
print("\n6. Executando análise completa (5-10 min)...")
print("   Por favor, aguarde...")

result = experiment.run_fairness_tests(config='full')

# 7. Resultados
print("\n" + "=" * 80)
print("RESULTADOS")
print("=" * 80)

print(f"\n📊 Overall Fairness Score: {result.overall_fairness_score:.3f}")

if result.overall_fairness_score >= 0.90:
    print("   ✅ EXCELENTE - Fairness muito boa")
elif result.overall_fairness_score >= 0.80:
    print("   ✓ BOA - Fairness aceitável")
elif result.overall_fairness_score >= 0.70:
    print("   ⚠️  MODERADA - Necessita atenção")
else:
    print("   ❌ CRÍTICA - Viés significativo")

print(f"\n⚠️  Critical Issues: {len(result.critical_issues)}")
if result.critical_issues:
    print("\n   Principais issues:")
    for issue in result.critical_issues[:5]:
        print(f"   - {issue}")

print(f"\n⚠️  Warnings: {len(result.warnings)}")
if result.warnings:
    print("\n   Principais warnings:")
    for warning in result.warnings[:5]:
        print(f"   - {warning}")

# 8. Métricas detalhadas
print("\n" + "=" * 80)
print("MÉTRICAS DETALHADAS")
print("=" * 80)

results_dict = result._results

print("\n📊 DISPARATE IMPACT (EEOC 80% Rule):")
for attr in PROTECTED_ATTRS:
    if attr in results_dict['posttrain_metrics']:
        di = results_dict['posttrain_metrics'][attr].get('disparate_impact', {})
        value = di.get('value', 'N/A')
        if isinstance(value, (int, float)):
            status = "✓ OK" if value >= 0.80 else "✗ VIOLADO"
            print(f"   {attr}: {value:.3f} ({status})")

print("\n📊 STATISTICAL PARITY:")
for attr in PROTECTED_ATTRS:
    if attr in results_dict['posttrain_metrics']:
        sp = results_dict['posttrain_metrics'][attr].get('statistical_parity', {})
        value = sp.get('value', 'N/A')
        if isinstance(value, (int, float)):
            status = "✓ OK" if abs(value) < 0.10 else "⚠️  ATENÇÃO"
            print(f"   {attr}: {value:+.3f} ({status})")

print("\n📊 EQUAL OPPORTUNITY:")
for attr in PROTECTED_ATTRS:
    if attr in results_dict['posttrain_metrics']:
        eo = results_dict['posttrain_metrics'][attr].get('equal_opportunity', {})
        value = eo.get('value', 'N/A')
        if isinstance(value, (int, float)):
            status = "✓ OK" if abs(value) < 0.10 else "⚠️  ATENÇÃO"
            print(f"   {attr}: {value:+.3f} ({status})")

# 9. Gerar relatório
print("\n" + "=" * 80)
print("GERANDO RELATÓRIO")
print("=" * 80)

report_path = result.save_html(
    file_path=str(OUTPUT_DIR / 'fairness_report.html'),
    model_name='Modelo de Produção - Análise de Fairness',
    report_type="interactive"  # Mesmo padrão dos outros módulos
)

print(f"\n✓ Relatório gerado:")
print(f"  {report_path}")
print(f"  Tamanho: {Path(report_path).stat().st_size / 1024:.1f} KB")

print(f"\n💡 Abrir no navegador:")
print(f"  file://{Path(report_path).absolute()}")

# 10. Recomendações
print("\n" + "=" * 80)
print("RECOMENDAÇÕES")
print("=" * 80)

if result.overall_fairness_score >= 0.80:
    print("\n✅ MODELO APROVADO para produção (do ponto de vista de fairness)")
    print("\n   Próximos passos:")
    print("   1. Revisar relatório HTML detalhado")
    print("   2. Validar com stakeholders legais/éticos")
    print("   3. Implementar monitoramento contínuo")
    print("   4. Documentar resultados para auditoria")
else:
    print("\n⚠️  MODELO NECESSITA MELHORIAS antes de produção")
    print("\n   Ações recomendadas:")
    print("   1. Revisar critical issues no relatório HTML")
    print("   2. Considerar técnicas de mitigação:")
    print("      - Re-balanceamento de dados por grupo")
    print("      - Ajuste de threshold de decisão")
    print("      - Fairness constraints no treinamento")
    print("   3. Re-treinar modelo")
    print("   4. Re-executar análise")

# Threshold Analysis
if 'threshold_analysis' in results_dict:
    optimal_threshold = results_dict['threshold_analysis'].get('optimal_threshold')
    if optimal_threshold:
        print(f"\n💡 DICA: Threshold ótimo para fairness: {optimal_threshold:.3f}")
        print(f"   Considerar ajustar threshold do modelo atual")

print("\n" + "=" * 80)
print("ANÁLISE CONCLUÍDA")
print("=" * 80)
