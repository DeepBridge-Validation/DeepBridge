"""
Exemplo Completo: Análise de Fairness End-to-End
=================================================

Este exemplo demonstra o fluxo completo de análise de fairness no DeepBridge,
desde a preparação dos dados até a geração de relatórios HTML interativos.

Cenário: Modelo de Aprovação de Crédito
- Dataset sintético com viés demográfico
- Atributos sensíveis: gender, race, age_group
- Objetivo: Detectar e quantificar viés no modelo

Autor: DeepBridge Team
Data: 2025-11-03
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Imports DeepBridge
from deepbridge.core.db_data import DBDataset
from deepbridge.core.experiment.experiment import Experiment
from deepbridge.validation.wrappers import FairnessSuite
from deepbridge.validation.fairness import FairnessVisualizer

# ============================================================================
# PARTE 1: PREPARAÇÃO DOS DADOS
# ============================================================================

print("=" * 80)
print("EXEMPLO COMPLETO: ANÁLISE DE FAIRNESS NO DEEPBRIDGE")
print("=" * 80)

print("\n📊 PARTE 1: Preparação dos Dados")
print("-" * 80)

# Configurar seed para reprodutibilidade
np.random.seed(42)
n_samples = 2000

# Gerar features numéricas
print("\n1. Gerando features numéricas...")
income = np.random.lognormal(10.5, 0.5, n_samples)  # Renda anual
credit_score = np.random.normal(700, 100, n_samples)  # Score de crédito
debt_ratio = np.random.beta(2, 5, n_samples)  # Razão dívida/renda
employment_years = np.random.gamma(2, 3, n_samples)  # Anos de emprego
savings = np.random.exponential(10000, n_samples)  # Poupança

# Gerar atributos protegidos/sensíveis
print("2. Gerando atributos protegidos...")
gender = np.random.choice(['Male', 'Female'], n_samples, p=[0.55, 0.45])
race = np.random.choice(
    ['White', 'Black', 'Hispanic', 'Asian'],
    n_samples,
    p=[0.60, 0.20, 0.15, 0.05]
)
age = np.random.normal(40, 15, n_samples)
age_group = pd.cut(
    age,
    bins=[0, 25, 40, 60, 100],
    labels=['Young', 'Adult', 'Middle-Aged', 'Senior']
).astype(str)

# Criar target com VIÉS INTENCIONAL (para demonstração)
print("3. Gerando target com viés demográfico...")
print("   ⚠️  Viés intencional para demonstração:")
print("   - Homens: +12% probabilidade de aprovação")
print("   - Brancos: +10% probabilidade de aprovação")
print("   - Jovens: -8% probabilidade de aprovação")

y = np.zeros(n_samples)
for i in range(n_samples):
    # Probabilidade base (features financeiras)
    base_prob = (
        0.3 +
        (credit_score[i] - 600) / 200 * 0.3 +
        (1 - debt_ratio[i]) * 0.2 +
        min(employment_years[i] / 10, 1) * 0.1
    )

    # Adicionar VIÉS demográfico (NÃO deve ser feito na prática!)
    bias = 0
    if gender[i] == 'Male':
        bias += 0.12  # Viés de gênero
    if race[i] == 'White':
        bias += 0.10  # Viés racial
    if age_group[i] == 'Young':
        bias -= 0.08  # Viés etário

    final_prob = np.clip(base_prob + bias, 0, 1)
    y[i] = 1 if np.random.rand() < final_prob else 0

# Criar DataFrame
df = pd.DataFrame({
    'income': income,
    'credit_score': credit_score,
    'debt_ratio': debt_ratio,
    'employment_years': employment_years,
    'savings': savings,
    'gender': gender,
    'race': race,
    'age_group': age_group,
    'approved': y
})

print(f"\n✓ Dataset criado: {df.shape}")
print(f"  - Features numéricas: {['income', 'credit_score', 'debt_ratio', 'employment_years', 'savings']}")
print(f"  - Atributos protegidos: {['gender', 'race', 'age_group']}")
print(f"  - Taxa de aprovação: {y.mean():.1%}")
print(f"  - Taxa por gênero:")
for g in df['gender'].unique():
    rate = df[df['gender'] == g]['approved'].mean()
    print(f"    {g}: {rate:.1%}")

# ============================================================================
# PARTE 2: TREINAMENTO DO MODELO
# ============================================================================

print("\n" + "=" * 80)
print("🤖 PARTE 2: Treinamento do Modelo")
print("-" * 80)

# Separar features (SEM atributos protegidos) e target
feature_cols = ['income', 'credit_score', 'debt_ratio', 'employment_years', 'savings']
X = df[feature_cols]
y = df['approved']

print(f"\n1. Separando train/test (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   Train: {X_train.shape[0]} samples")
print(f"   Test: {X_test.shape[0]} samples")

# Treinar modelo
print("\n2. Treinando Random Forest...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train, y_train)

# Avaliar performance
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print(f"   ✓ Acurácia (train): {train_acc:.3f}")
print(f"   ✓ Acurácia (test): {test_acc:.3f}")

# ============================================================================
# PARTE 3: CRIAR DBDATASET
# ============================================================================

print("\n" + "=" * 80)
print("📦 PARTE 3: Criação do DBDataset")
print("-" * 80)

print("\n1. Criando DBDataset com dados completos...")
dataset = DBDataset(
    data=df,
    target_column='approved',
    model=model
)

print(f"   ✓ Dataset shape: {df.shape}")
print(f"   ✓ Model type: {type(model).__name__}")
print(f"   ✓ Target: approved")

# ============================================================================
# PARTE 4: ANÁLISE DE FAIRNESS COM EXPERIMENT (RECOMENDADO)
# ============================================================================

print("\n" + "=" * 80)
print("🔍 PARTE 4: Análise de Fairness via Experiment (Método Recomendado)")
print("-" * 80)

# Método 4.1: Com protected_attributes EXPLÍCITOS (RECOMENDADO PARA PRODUÇÃO)
print("\n4.1 - Com protected_attributes explícitos:")
print("      (Recomendado para produção)")

experiment = Experiment(
    dataset=dataset,
    experiment_type="binary_classification",
    tests=["fairness"],
    protected_attributes=['gender', 'race', 'age_group'],  # Explícito
    test_size=0.2,
    random_state=42
)

print(f"   ✓ Experiment criado")
print(f"   ✓ Protected attributes: {experiment.protected_attributes}")

# Executar testes de fairness (config 'full' = 15 métricas + threshold analysis)
print("\n   Executando testes (config='full')...")
fairness_result = experiment.run_fairness_tests(config='full')

print(f"   ✓ Testes concluídos")
print(f"\n   📊 RESULTADOS:")
print(f"      Overall Fairness Score: {fairness_result.overall_fairness_score:.3f}")
print(f"      Critical Issues: {len(fairness_result.critical_issues)}")
print(f"      Warnings: {len(fairness_result.warnings)}")
print(f"      Protected Attributes: {fairness_result.protected_attributes}")

# Exibir issues
if fairness_result.critical_issues:
    print(f"\n   ⚠️  CRITICAL ISSUES:")
    for issue in fairness_result.critical_issues[:3]:  # Primeiros 3
        print(f"      - {issue}")

if fairness_result.warnings:
    print(f"\n   ⚠️  WARNINGS:")
    for warning in fairness_result.warnings[:3]:  # Primeiros 3
        print(f"      - {warning}")

# Gerar relatório HTML
output_dir = Path('./fairness_example_output')
output_dir.mkdir(exist_ok=True)

print(f"\n   Gerando relatório HTML...")
report_path = fairness_result.save_html(
    file_path=str(output_dir / 'fairness_report_experiment.html'),
    model_name='Credit Approval Model v1.0',
    report_type='interactive'  # Formato interativo como outros módulos
)

print(f"   ✓ Relatório gerado: {report_path}")
print(f"   📁 Tamanho: {Path(report_path).stat().st_size / 1024:.1f} KB")

# Método 4.2: Com AUTO-DETECÇÃO (PARA EXPLORAÇÃO RÁPIDA)
print("\n4.2 - Com auto-detecção:")
print("      (Útil para exploração rápida)")

experiment_auto = Experiment(
    dataset=dataset,
    experiment_type="binary_classification",
    tests=["fairness"],  # SEM protected_attributes
    test_size=0.2,
    random_state=42
)

print(f"   ✓ Auto-detectado: {experiment_auto.protected_attributes}")
print(f"   ⚠️  Para produção, sempre especifique explicitamente!")

# ============================================================================
# PARTE 5: ANÁLISE DE FAIRNESS COM FAIRNESSSUITE (AVANÇADO)
# ============================================================================

print("\n" + "=" * 80)
print("🔬 PARTE 5: Análise Avançada com FairnessSuite")
print("-" * 80)

# Comparar diferentes configurações
print("\n5.1 - Comparando configurações (quick/medium/full):")

configs = {
    'quick': 'Rápido (2 métricas)',
    'medium': 'Médio (5 métricas + pré-treino)',
    'full': 'Completo (15 métricas + threshold)'
}

suite_results = {}

for config_name, description in configs.items():
    print(f"\n   Executando config '{config_name}': {description}...")

    fairness_suite = FairnessSuite(
        dataset=dataset,
        protected_attributes=['gender', 'race', 'age_group']
    )

    results = fairness_suite.config(config_name).run()
    suite_results[config_name] = results

    score = results.get('overall_fairness_score', 0)
    print(f"   ✓ Score: {score:.3f}")

    # Gerar relatório para cada config
    from deepbridge.core.experiment.report.report_manager import ReportManager
    report_manager = ReportManager()

    report_path = report_manager.generate_report(
        test_type='fairness',
        results=results,
        file_path=str(output_dir / f'fairness_report_{config_name}.html'),
        model_name=f'Credit Model ({config_name})'
    )

    print(f"   ✓ Relatório: {Path(report_path).name}")

# ============================================================================
# PARTE 6: VISUALIZAÇÕES ESTÁTICAS
# ============================================================================

print("\n" + "=" * 80)
print("📊 PARTE 6: Gerando Visualizações Estáticas")
print("-" * 80)

print("\n6.1 - Distribuição por grupo (gender):")
viz_path = FairnessVisualizer.plot_distribution_by_group(
    df=df,
    target_col='approved',
    sensitive_feature='gender',
    output_path=str(output_dir / 'distribution_gender.png')
)
print(f"   ✓ Salvo: {Path(viz_path).name}")

print("\n6.2 - Distribuição por grupo (race):")
viz_path = FairnessVisualizer.plot_distribution_by_group(
    df=df,
    target_col='approved',
    sensitive_feature='race',
    output_path=str(output_dir / 'distribution_race.png')
)
print(f"   ✓ Salvo: {Path(viz_path).name}")

print("\n6.3 - Comparação de métricas:")
viz_path = FairnessVisualizer.plot_metrics_comparison(
    metrics_results=suite_results['full']['posttrain_metrics'],
    protected_attrs=['gender', 'race', 'age_group'],
    output_path=str(output_dir / 'metrics_comparison.png')
)
print(f"   ✓ Salvo: {Path(viz_path).name}")

print("\n6.4 - Radar de fairness:")
viz_path = FairnessVisualizer.plot_fairness_radar(
    metrics_summary=suite_results['full']['posttrain_metrics'],
    output_path=str(output_dir / 'fairness_radar.png')
)
print(f"   ✓ Salvo: {Path(viz_path).name}")

# ============================================================================
# PARTE 7: RECOMENDAÇÕES BASEADAS NOS RESULTADOS
# ============================================================================

print("\n" + "=" * 80)
print("💡 PARTE 7: Recomendações e Próximos Passos")
print("-" * 80)

score = fairness_result.overall_fairness_score
critical_count = len(fairness_result.critical_issues)
warning_count = len(fairness_result.warnings)

print(f"\n📊 RESUMO DA ANÁLISE:")
print(f"   Overall Fairness Score: {score:.3f}")
print(f"   Critical Issues: {critical_count}")
print(f"   Warnings: {warning_count}")

print(f"\n💡 RECOMENDAÇÕES:")

if score >= 0.9:
    print("   ✅ EXCELENTE - Modelo apresenta fairness muito boa")
    print("      - Considerar deploy em produção")
    print("      - Monitorar métricas continuamente")
elif score >= 0.8:
    print("   ✓ BOA - Modelo apresenta fairness aceitável")
    print("      - Revisar warnings antes do deploy")
    print("      - Considerar melhorias nas áreas identificadas")
elif score >= 0.7:
    print("   ⚠️  MODERADA - Modelo apresenta problemas de fairness")
    print("      - Recomenda-se retreinar com técnicas de mitigação de viés")
    print("      - Considerar re-balanceamento de dados")
    print("      - Avaliar threshold de decisão")
else:
    print("   ❌ CRÍTICA - Modelo apresenta viés significativo")
    print("      - NÃO recomendado para deploy")
    print("      - Investigar fontes de viés nos dados")
    print("      - Aplicar técnicas de fairness-aware learning")

print(f"\n🔧 TÉCNICAS DE MITIGAÇÃO:")
print("   1. Pré-processamento:")
print("      - Re-balanceamento de classes por grupo")
print("      - Remoção de features correlacionadas com atributos protegidos")
print("   2. Durante o treinamento:")
print("      - Adversarial debiasing")
print("      - Fairness constraints")
print("   3. Pós-processamento:")
print("      - Ajuste de thresholds por grupo")
print("      - Calibração de probabilidades")

# ============================================================================
# RESUMO FINAL
# ============================================================================

print("\n" + "=" * 80)
print("📁 RESUMO FINAL - ARQUIVOS GERADOS")
print("=" * 80)

print(f"\n📂 Diretório: {output_dir.absolute()}")

generated_files = sorted(output_dir.glob('*'))
total_size = sum(f.stat().st_size for f in generated_files if f.is_file())

print(f"\n📊 ESTATÍSTICAS:")
print(f"   - Total de arquivos: {len(generated_files)}")
print(f"   - Tamanho total: {total_size / 1024:.1f} KB")

print(f"\n📁 ARQUIVOS:")
for f in generated_files:
    if f.is_file():
        size_kb = f.stat().st_size / 1024
        icon = "🌐" if f.suffix == '.html' else "📊"
        print(f"   {icon} {f.name} ({size_kb:.1f} KB)")

print(f"\n💡 PRÓXIMOS PASSOS:")
print(f"   1. Abrir relatórios HTML em um navegador:")
print(f"      file://{output_dir.absolute()}/fairness_report_experiment.html")
print(f"   2. Revisar visualizações estáticas")
print(f"   3. Analisar métricas detalhadas por atributo")
print(f"   4. Implementar mitigações se necessário")
print(f"   5. Re-executar análise após mudanças")

print("\n" + "=" * 80)
print("✅ EXEMPLO COMPLETO EXECUTADO COM SUCESSO!")
print("=" * 80)
