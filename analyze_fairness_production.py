"""
Análise de Fairness em Dados de Produção
=========================================

Este script analisa fairness em modelos já treinados com predições existentes.

Dados:
- train_predictions.parquet
- test_predictions.parquet

Target: in_cmst_fun
Probabilidades: pred_proba_class_0, pred_proba_class_1
Classe predita: pred_class (com threshold customizado)

Atributos Protegidos:
- nm_tip_gnr (gênero)
- nm_tip_raca (raça)
- vl_idd_aa (idade)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings
warnings.filterwarnings('ignore')

# DeepBridge imports
from deepbridge.core.db_data import DBDataset
from deepbridge.core.experiment.experiment import Experiment
from deepbridge.validation.wrappers import FairnessSuite
from deepbridge.validation.fairness import FairnessVisualizer

# ============================================================================
# CLASSE WRAPPER PARA MODELO COM PREDIÇÕES PRÉ-COMPUTADAS
# ============================================================================

class PrecomputedPredictionsModel(BaseEstimator, ClassifierMixin):
    """
    Wrapper que simula um modelo sklearn mas usa predições já calculadas.

    Útil quando você tem um modelo já treinado e quer analisar fairness
    das predições existentes sem re-treinar.
    """

    def __init__(self, predictions_df, proba_cols=['pred_proba_class_0', 'pred_proba_class_1']):
        """
        Parameters:
            predictions_df: DataFrame com as predições
            proba_cols: Colunas com probabilidades por classe
        """
        self.predictions_df = predictions_df.copy()
        self.proba_cols = proba_cols
        self.classes_ = np.array([0, 1])  # Binary classification
        self.n_classes_ = 2

        # Criar índice para lookup rápido
        self.predictions_df['_index'] = range(len(self.predictions_df))

    def fit(self, X, y):
        """Não faz nada - modelo já está 'treinado'"""
        return self

    def predict(self, X):
        """
        Retorna predições usando os índices para fazer lookup.

        IMPORTANTE: Assume que X tem os mesmos índices do DataFrame original.
        """
        if isinstance(X, pd.DataFrame):
            indices = X.index
        else:
            # Se for numpy array, assumir ordem sequencial
            indices = range(len(X))

        # Fazer lookup das predições
        predictions = self.predictions_df.loc[indices, 'pred_class'].values

        return predictions

    def predict_proba(self, X):
        """
        Retorna probabilidades usando os índices para fazer lookup.
        """
        if isinstance(X, pd.DataFrame):
            indices = X.index
        else:
            indices = range(len(X))

        # Fazer lookup das probabilidades
        probas = self.predictions_df.loc[indices, self.proba_cols].values

        return probas

# ============================================================================
# CONFIGURAÇÕES
# ============================================================================

print("=" * 80)
print("ANÁLISE DE FAIRNESS - DADOS DE PRODUÇÃO")
print("=" * 80)

# Paths dos dados
DATA_DIR = Path("/home/guhaase/projetos/DeepBridge/simular_lib/analise_v4")
TRAIN_PATH = DATA_DIR / "train_predictions.parquet"
TEST_PATH = DATA_DIR / "test_predictions.parquet"

# Configurações
TARGET_COL = 'in_cmst_fun'
PROBA_COLS = ['pred_proba_class_0', 'pred_proba_class_1']
PRED_COL = 'pred_class'

# Atributos protegidos (ajuste conforme necessário)
PROTECTED_ATTRIBUTES = ['nm_tip_gnr', 'nm_tip_raca']  # Gênero e Raça
# Nota: vl_idd_aa (idade) pode ser adicionado após criar grupos etários

# Diretório de output
OUTPUT_DIR = Path('./fairness_production_analysis')
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"\n📁 Diretórios configurados:")
print(f"   Dados: {DATA_DIR}")
print(f"   Output: {OUTPUT_DIR.absolute()}")

# ============================================================================
# PASSO 1: CARREGAR DADOS
# ============================================================================

print("\n" + "=" * 80)
print("PASSO 1: Carregando Dados")
print("=" * 80)

print("\n1. Carregando train_predictions.parquet...")
df_train = pd.read_parquet(TRAIN_PATH)
print(f"   ✓ Train: {df_train.shape}")

print("\n2. Carregando test_predictions.parquet...")
df_test = pd.read_parquet(TEST_PATH)
print(f"   ✓ Test: {df_test.shape}")

# Verificar se colunas necessárias existem
required_cols = [TARGET_COL] + PROBA_COLS + [PRED_COL] + PROTECTED_ATTRIBUTES
missing_cols = [col for col in required_cols if col not in df_train.columns]

if missing_cols:
    print(f"\n   ⚠️  AVISO: Colunas faltando: {missing_cols}")
    print(f"   Colunas disponíveis: {list(df_train.columns)}")
    raise ValueError(f"Colunas necessárias não encontradas: {missing_cols}")

print(f"\n   ✓ Todas as colunas necessárias presentes")

# ============================================================================
# PASSO 2: ANÁLISE EXPLORATÓRIA INICIAL
# ============================================================================

print("\n" + "=" * 80)
print("PASSO 2: Análise Exploratória")
print("=" * 80)

# Usar dados de test para análise (mais representativo de produção)
df = df_test.copy()

print(f"\n📊 ESTATÍSTICAS GERAIS (Test Set):")
print(f"   Total de amostras: {len(df):,}")
print(f"   Target ({TARGET_COL}):")

# Distribuição do target
target_dist = df[TARGET_COL].value_counts()
print(f"      Classe 0: {target_dist.get(0, 0):,} ({target_dist.get(0, 0)/len(df):.1%})")
print(f"      Classe 1: {target_dist.get(1, 0):,} ({target_dist.get(1, 0)/len(df):.1%})")

# Distribuição das predições
print(f"\n   Predições ({PRED_COL}):")
pred_dist = df[PRED_COL].value_counts()
print(f"      Classe 0: {pred_dist.get(0, 0):,} ({pred_dist.get(0, 0)/len(df):.1%})")
print(f"      Classe 1: {pred_dist.get(1, 0):,} ({pred_dist.get(1, 0)/len(df):.1%})")

# Acurácia geral
accuracy = (df[TARGET_COL] == df[PRED_COL]).mean()
print(f"\n   Acurácia geral: {accuracy:.3f}")

# ============================================================================
# PASSO 3: ANÁLISE DE ATRIBUTOS PROTEGIDOS
# ============================================================================

print("\n" + "=" * 80)
print("PASSO 3: Análise de Atributos Protegidos")
print("=" * 80)

for attr in PROTECTED_ATTRIBUTES:
    print(f"\n📊 {attr.upper()}:")

    # Verificar valores únicos
    unique_vals = df[attr].value_counts()
    print(f"   Valores únicos: {len(unique_vals)}")

    # Top 5 valores
    print(f"   Top 5 grupos:")
    for val, count in unique_vals.head(5).items():
        pct = count / len(df) * 100
        print(f"      {val}: {count:,} ({pct:.1f}%)")

    # Taxa de aprovação por grupo (assumindo classe 1 = aprovado)
    print(f"\n   Taxa de aprovação (classe 1) por grupo:")
    approval_by_group = df.groupby(attr)[TARGET_COL].mean()

    for val, rate in approval_by_group.head(10).items():
        print(f"      {val}: {rate:.1%}")

    # Taxa de PREDIÇÃO positiva por grupo
    print(f"\n   Taxa de PREDIÇÃO positiva por grupo:")
    pred_by_group = df.groupby(attr)[PRED_COL].mean()

    for val, rate in pred_by_group.head(10).items():
        print(f"      {val}: {rate:.1%}")

# ============================================================================
# PASSO 4: CRIAR GRUPOS ETÁRIOS (OPCIONAL)
# ============================================================================

print("\n" + "=" * 80)
print("PASSO 4: Criando Grupos Etários")
print("=" * 80)

if 'vl_idd_aa' in df.columns:
    print("\n1. Analisando distribuição de idade...")

    # Estatísticas de idade
    print(f"   Idade mínima: {df['vl_idd_aa'].min():.0f} anos")
    print(f"   Idade máxima: {df['vl_idd_aa'].max():.0f} anos")
    print(f"   Idade média: {df['vl_idd_aa'].mean():.1f} anos")
    print(f"   Idade mediana: {df['vl_idd_aa'].median():.1f} anos")

    print("\n2. Criando grupos etários...")

    # Criar grupos etários
    df['age_group'] = pd.cut(
        df['vl_idd_aa'],
        bins=[0, 30, 40, 50, 60, 100],
        labels=['18-30', '31-40', '41-50', '51-60', '60+'],
        include_lowest=True
    )

    # Adicionar aos atributos protegidos
    PROTECTED_ATTRIBUTES.append('age_group')

    print(f"   ✓ Grupos criados:")
    for group, count in df['age_group'].value_counts().sort_index().items():
        pct = count / len(df) * 100
        print(f"      {group}: {count:,} ({pct:.1f}%)")

    print(f"\n   Taxa de aprovação por grupo etário:")
    approval_by_age = df.groupby('age_group')[TARGET_COL].mean()
    for group, rate in approval_by_age.items():
        print(f"      {group}: {rate:.1%}")
else:
    print("\n   ⚠️  Coluna 'vl_idd_aa' não encontrada - pulando grupos etários")

print(f"\n✓ Atributos protegidos finais: {PROTECTED_ATTRIBUTES}")

# ============================================================================
# PASSO 5: PREPARAR DADOS PARA ANÁLISE DE FAIRNESS
# ============================================================================

print("\n" + "=" * 80)
print("PASSO 5: Preparando Dados para Análise de Fairness")
print("=" * 80)

# Features são todas as colunas EXCETO target, predições e atributos protegidos
exclude_cols = [TARGET_COL] + PROBA_COLS + [PRED_COL] + PROTECTED_ATTRIBUTES + ['_index']
feature_cols = [col for col in df.columns if col not in exclude_cols]

print(f"\n1. Features selecionadas: {len(feature_cols)} colunas")
print(f"   (Primeiras 10: {feature_cols[:10]})")

# Criar DataFrame final
# IMPORTANTE: Manter o index original para o wrapper funcionar
df_analysis = df.copy()

# Garantir que não há NaN nas colunas críticas
print(f"\n2. Verificando valores ausentes...")
critical_cols = [TARGET_COL] + PROBA_COLS + [PRED_COL] + PROTECTED_ATTRIBUTES
for col in critical_cols:
    nan_count = df_analysis[col].isna().sum()
    if nan_count > 0:
        print(f"   ⚠️  {col}: {nan_count} NaN encontrados")
        # Remover linhas com NaN em colunas críticas
        df_analysis = df_analysis.dropna(subset=[col])

print(f"   ✓ Dataset final: {df_analysis.shape}")

# ============================================================================
# PASSO 6: CRIAR MODELO WRAPPER
# ============================================================================

print("\n" + "=" * 80)
print("PASSO 6: Criando Modelo Wrapper")
print("=" * 80)

print("\n1. Criando PrecomputedPredictionsModel...")

# Criar modelo wrapper
model = PrecomputedPredictionsModel(
    predictions_df=df_analysis,
    proba_cols=PROBA_COLS
)

print(f"   ✓ Modelo criado")
print(f"   Classes: {model.classes_}")

# Testar modelo
print("\n2. Testando modelo wrapper...")
X_sample = df_analysis[feature_cols].iloc[:5]
y_pred_test = model.predict(X_sample)
y_proba_test = model.predict_proba(X_sample)

print(f"   ✓ predict() funcionando: {y_pred_test}")
print(f"   ✓ predict_proba() funcionando: shape {y_proba_test.shape}")

# ============================================================================
# PASSO 7: CRIAR DBDATASET
# ============================================================================

print("\n" + "=" * 80)
print("PASSO 7: Criando DBDataset")
print("=" * 80)

print("\n1. Preparando dados...")

# Criar dataset completo (com atributos protegidos para análise)
dataset = DBDataset(
    data=df_analysis,
    target_column=TARGET_COL,
    model=model
)

print(f"   ✓ DBDataset criado: {df_analysis.shape}")

# ============================================================================
# PASSO 8: ANÁLISE DE FAIRNESS - QUICK
# ============================================================================

print("\n" + "=" * 80)
print("PASSO 8: Análise de Fairness - Quick (Preview)")
print("=" * 80)

print("\n1. Criando Experiment...")

experiment = Experiment(
    dataset=dataset,
    experiment_type="binary_classification",
    tests=["fairness"],
    protected_attributes=PROTECTED_ATTRIBUTES,
    test_size=0.2,  # Mesmo que já seja test set, precisamos de um split interno
    random_state=42
)

print(f"   ✓ Experiment criado")
print(f"   Protected attributes: {experiment.protected_attributes}")

print("\n2. Executando análise rápida (config='quick')...")
print("   (Tempo estimado: 10-30 segundos)")

quick_result = experiment.run_fairness_tests(config='quick')

print(f"\n3. Resultados Quick:")
print(f"   Overall Fairness Score: {quick_result.overall_fairness_score:.3f}")
print(f"   Critical Issues: {len(quick_result.critical_issues)}")
print(f"   Warnings: {len(quick_result.warnings)}")

if quick_result.critical_issues:
    print(f"\n   ⚠️  CRITICAL ISSUES (primeiros 3):")
    for issue in quick_result.critical_issues[:3]:
        print(f"      - {issue}")

if quick_result.warnings:
    print(f"\n   ⚠️  WARNINGS (primeiros 3):")
    for warning in quick_result.warnings[:3]:
        print(f"      - {warning}")

# ============================================================================
# PASSO 9: ANÁLISE DE FAIRNESS - FULL
# ============================================================================

print("\n" + "=" * 80)
print("PASSO 9: Análise de Fairness - Full (Completa)")
print("=" * 80)

print("\n1. Executando análise completa (config='full')...")
print("   ⚠️  Isso pode levar 5-10 minutos dependendo do tamanho dos dados...")
print("   (Inclui threshold analysis com 99 thresholds)")

full_result = experiment.run_fairness_tests(config='full')

print(f"\n2. Resultados Full:")
print(f"   Overall Fairness Score: {full_result.overall_fairness_score:.3f}")
print(f"   Critical Issues: {len(full_result.critical_issues)}")
print(f"   Warnings: {len(full_result.warnings)}")
print(f"   Protected Attributes: {full_result.protected_attributes}")

# Interpretação do score
score = full_result.overall_fairness_score
if score >= 0.90:
    interpretation = "✅ EXCELENTE - Modelo apresenta fairness muito boa"
    recommendation = "Considerar deploy em produção. Monitorar continuamente."
elif score >= 0.80:
    interpretation = "✓ BOA - Modelo apresenta fairness aceitável"
    recommendation = "Revisar warnings antes do deploy. Considerar melhorias."
elif score >= 0.70:
    interpretation = "⚠️  MODERADA - Modelo apresenta problemas de fairness"
    recommendation = "Recomenda-se retreinar com técnicas de mitigação de viés."
else:
    interpretation = "❌ CRÍTICA - Modelo apresenta viés significativo"
    recommendation = "NÃO recomendado para deploy. Investigar fontes de viés."

print(f"\n3. Interpretação:")
print(f"   {interpretation}")
print(f"\n   Recomendação:")
print(f"   {recommendation}")

# ============================================================================
# PASSO 10: GERAR RELATÓRIOS HTML
# ============================================================================

print("\n" + "=" * 80)
print("PASSO 10: Gerando Relatórios HTML")
print("=" * 80)

print("\n1. Relatório Quick...")
quick_report_path = full_result.save_html(
    file_path=str(OUTPUT_DIR / 'fairness_report_quick.html'),
    model_name='Modelo de Produção - Quick Analysis',
    report_type="interactive"  # Mesmo padrão dos outros módulos
)
print(f"   ✓ Salvo: {Path(quick_report_path).name}")

print("\n2. Relatório Full...")
full_report_path = full_result.save_html(
    file_path=str(OUTPUT_DIR / 'fairness_report_full.html'),
    model_name='Modelo de Produção - Full Analysis',
    report_type="interactive"  # Mesmo padrão dos outros módulos
)
print(f"   ✓ Salvo: {Path(full_report_path).name}")

# ============================================================================
# PASSO 11: GERAR VISUALIZAÇÕES ESTÁTICAS
# ============================================================================

print("\n" + "=" * 80)
print("PASSO 11: Gerando Visualizações Estáticas")
print("=" * 80)

# Visualizações por atributo protegido
for attr in PROTECTED_ATTRIBUTES:
    print(f"\n📊 Visualizações para '{attr}':")

    # 1. Distribuição
    try:
        viz_path = FairnessVisualizer.plot_distribution_by_group(
            df=df_analysis,
            target_col=TARGET_COL,
            sensitive_feature=attr,
            output_path=str(OUTPUT_DIR / f'distribution_{attr}.png')
        )
        print(f"   ✓ Distribuição salva: distribution_{attr}.png")
    except Exception as e:
        print(f"   ⚠️  Erro ao gerar distribuição: {e}")

# Comparação de métricas (todas os atributos)
print(f"\n📊 Visualização comparativa:")

try:
    results = full_result._results
    viz_path = FairnessVisualizer.plot_metrics_comparison(
        metrics_results=results['posttrain_metrics'],
        protected_attrs=PROTECTED_ATTRIBUTES,
        output_path=str(OUTPUT_DIR / 'metrics_comparison.png')
    )
    print(f"   ✓ Comparação salva: metrics_comparison.png")
except Exception as e:
    print(f"   ⚠️  Erro ao gerar comparação: {e}")

# Radar de fairness
try:
    viz_path = FairnessVisualizer.plot_fairness_radar(
        metrics_summary=results['posttrain_metrics'],
        output_path=str(OUTPUT_DIR / 'fairness_radar.png')
    )
    print(f"   ✓ Radar salvo: fairness_radar.png")
except Exception as e:
    print(f"   ⚠️  Erro ao gerar radar: {e}")

# ============================================================================
# PASSO 12: ANÁLISE DETALHADA POR MÉTRICA
# ============================================================================

print("\n" + "=" * 80)
print("PASSO 12: Análise Detalhada por Métrica")
print("=" * 80)

results = full_result._results

# Métricas chave para análise
key_metrics = [
    'statistical_parity',
    'disparate_impact',
    'equal_opportunity',
    'equalized_odds'
]

for metric_name in key_metrics:
    print(f"\n📊 {metric_name.upper().replace('_', ' ')}:")

    for attr in PROTECTED_ATTRIBUTES:
        if attr in results['posttrain_metrics']:
            attr_metrics = results['posttrain_metrics'][attr]

            if metric_name in attr_metrics:
                metric_data = attr_metrics[metric_name]
                value = metric_data.get('value', 'N/A')
                interpretation = metric_data.get('interpretation', '')

                # Formatação especial para disparate impact (EEOC)
                if metric_name == 'disparate_impact':
                    eeoc_status = "✓ EEOC OK" if isinstance(value, (int, float)) and value >= 0.80 else "✗ EEOC VIOLADO"
                    print(f"   {attr}: {value:.3f if isinstance(value, (int, float)) else value} - {interpretation} ({eeoc_status})")
                else:
                    print(f"   {attr}: {value:.3f if isinstance(value, (int, float)) else value} - {interpretation}")

# Threshold Analysis (se disponível)
if 'threshold_analysis' in results:
    print(f"\n📊 THRESHOLD ANALYSIS:")
    threshold_data = results['threshold_analysis']

    optimal = threshold_data.get('optimal_threshold', 'N/A')
    print(f"\n   Threshold atual do modelo: [customizado]")
    print(f"   Threshold ótimo (fairness): {optimal:.3f if isinstance(optimal, (int, float)) else optimal}")

    if isinstance(optimal, (int, float)):
        print(f"\n   💡 SUGESTÃO: Considerar ajustar threshold para {optimal:.3f}")
        print(f"      para melhorar fairness (analisar impacto em performance)")

# ============================================================================
# PASSO 13: RESUMO FINAL E RECOMENDAÇÕES
# ============================================================================

print("\n" + "=" * 80)
print("RESUMO FINAL E RECOMENDAÇÕES")
print("=" * 80)

print(f"\n📊 ESTATÍSTICAS:")
print(f"   Amostras analisadas: {len(df_analysis):,}")
print(f"   Atributos protegidos: {len(PROTECTED_ATTRIBUTES)}")
print(f"   Overall Fairness Score: {full_result.overall_fairness_score:.3f}")
print(f"   Critical Issues: {len(full_result.critical_issues)}")
print(f"   Warnings: {len(full_result.warnings)}")

print(f"\n📁 ARQUIVOS GERADOS:")
generated_files = sorted(OUTPUT_DIR.glob('*'))
for f in generated_files:
    if f.is_file():
        size_kb = f.stat().st_size / 1024
        icon = "🌐" if f.suffix == '.html' else "📊"
        print(f"   {icon} {f.name} ({size_kb:.1f} KB)")

print(f"\n💡 PRÓXIMOS PASSOS:")

if full_result.overall_fairness_score >= 0.80:
    print(f"   1. ✅ Modelo apresenta fairness aceitável")
    print(f"   2. Revisar relatório HTML completo")
    print(f"   3. Validar com stakeholders")
    print(f"   4. Implementar monitoramento contínuo")
else:
    print(f"   1. ⚠️  Modelo apresenta problemas de fairness")
    print(f"   2. Revisar critical issues e warnings")
    print(f"   3. Considerar técnicas de mitigação:")
    print(f"      - Re-balanceamento de dados")
    print(f"      - Ajuste de threshold")
    print(f"      - Fairness constraints")
    print(f"   4. Re-treinar e re-avaliar")

print(f"\n📂 ABRIR RELATÓRIOS:")
print(f"   file://{(OUTPUT_DIR / 'fairness_report_full.html').absolute()}")

print("\n" + "=" * 80)
print("✅ ANÁLISE DE FAIRNESS CONCLUÍDA COM SUCESSO!")
print("=" * 80)

print(f"\n📚 Para mais informações:")
print(f"   - Tutorial: docs/FAIRNESS_TUTORIAL.md")
print(f"   - Best Practices: docs/FAIRNESS_BEST_PRACTICES.md")
print(f"   - FAQ: docs/FAIRNESS_FAQ.md")
