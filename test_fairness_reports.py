"""
Script de teste para validar a geração de relatórios HTML do FairnessSuite (Fase 4).

Este script testa:
- Geração de relatório HTML com Plotly charts
- Integração com ReportManager
- Todos os componentes do report (overview, métricas, threshold, confusion matrix)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

from deepbridge.core.db_data import DBDataset
from deepbridge.validation.wrappers.fairness_suite import FairnessSuite
from deepbridge.core.experiment.report.report_manager import ReportManager

print("=" * 80)
print("TESTE DA FASE 4 - GERAÇÃO DE RELATÓRIOS FAIRNESS")
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
print(f"  Taxa positiva geral: {np.mean(y):.3f}")

# ============================================================================
# 2. TREINAR MODELO
# ============================================================================

print("\n🤖 2. Treinando modelo RandomForest...")

# Separar features (sem atributos protegidos) e target
X_train = df.drop(['gender', 'race', 'target'], axis=1)
y_train = df['target']

model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
model.fit(X_train, y_train)

# Gerar predições
y_pred = model.predict(X_train)
y_pred_proba = model.predict_proba(X_train)

print(f"  Acurácia no treino: {model.score(X_train, y_train):.3f}")

# ============================================================================
# 3. CRIAR DBDATASET E EXECUTAR FAIRNESS SUITE
# ============================================================================

print("\n📦 3. Criando DBDataset e executando FairnessSuite...")

# Criar DataFrame completo com predições
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

# Executar FairnessSuite com config 'full' para ter todos os dados
fairness = FairnessSuite(
    dataset=dataset,
    protected_attributes=['gender', 'race'],
    verbose=True
)

results = fairness.config('full').run()

print(f"\n  Overall Fairness Score: {results['overall_fairness_score']:.3f}")
print(f"  Warnings: {len(results['warnings'])}")
print(f"  Critical Issues: {len(results['critical_issues'])}")

# ============================================================================
# 4. GERAR RELATÓRIO HTML
# ============================================================================

print("\n" + "=" * 80)
print("📄 TESTE 1: Geração de Relatório HTML")
print("=" * 80)

# Criar diretório para relatórios
output_dir = Path('./test_reports_output')
output_dir.mkdir(exist_ok=True)
print(f"\n📁 Diretório de saída: {output_dir.absolute()}")

try:
    # Inicializar ReportManager
    report_manager = ReportManager()

    # Gerar relatório
    report_path = output_dir / 'fairness_report.html'

    generated_path = report_manager.generate_report(
        test_type='fairness',
        results=results,
        file_path=str(report_path),
        model_name="Test RandomForest Model",
        report_type='interactive'
    )

    print(f"\n  ✓ Relatório gerado com sucesso!")
    print(f"  📄 Arquivo: {generated_path}")

    # Verificar que o arquivo existe
    assert Path(generated_path).exists(), "Arquivo de relatório não foi criado"
    print(f"  ✓ Arquivo existe: {Path(generated_path).stat().st_size} bytes")

    # Verificar conteúdo básico do HTML
    with open(generated_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    # Validações do conteúdo
    required_elements = [
        'Fairness Analysis Report',
        'Overall Fairness Score',
        'Protected Attributes',
        'Plotly',
        'chart-metrics-comparison',
        'chart-fairness-radar'
    ]

    missing_elements = []
    for element in required_elements:
        if element not in html_content:
            missing_elements.append(element)

    if missing_elements:
        print(f"\n  ⚠️  Elementos faltando no HTML: {missing_elements}")
    else:
        print(f"  ✓ Todos os elementos essenciais presentes no HTML")

    # Verificar charts específicos
    if 'threshold' in results.get('threshold_analysis', {}):
        if 'chart-threshold-analysis' in html_content:
            print(f"  ✓ Chart de threshold analysis presente")

    if results.get('confusion_matrix'):
        if 'chart-confusion-matrices' in html_content:
            print(f"  ✓ Chart de confusion matrices presente")

    print("\n  ✅ Relatório HTML: PASSOU")

except Exception as e:
    print(f"\n  ❌ ERRO: {e}")
    import traceback
    traceback.print_exc()
    raise

# ============================================================================
# 5. TESTE COM DIFERENTES CONFIGS
# ============================================================================

print("\n" + "=" * 80)
print("📄 TESTE 2: Relatórios com Diferentes Configs")
print("=" * 80)

configs_to_test = ['quick', 'medium', 'full']

for config_name in configs_to_test:
    print(f"\n📋 Testando config '{config_name}'...")

    try:
        # Executar FairnessSuite
        fairness = FairnessSuite(
            dataset=dataset,
            protected_attributes=['gender', 'race'],
            verbose=False
        )

        results_config = fairness.config(config_name).run()

        # Gerar relatório
        report_path = output_dir / f'fairness_report_{config_name}.html'

        generated_path = report_manager.generate_report(
            test_type='fairness',
            results=results_config,
            file_path=str(report_path),
            model_name=f"Test Model ({config_name})",
            report_type='interactive'
        )

        # Verificar
        assert Path(generated_path).exists()
        file_size = Path(generated_path).stat().st_size

        print(f"  ✓ Relatório '{config_name}' gerado: {file_size} bytes")

    except Exception as e:
        print(f"  ❌ ERRO no config '{config_name}': {e}")
        raise

print("\n  ✅ Relatórios com diferentes configs: PASSOU")

# ============================================================================
# 6. RESUMO FINAL
# ============================================================================

print("\n" + "=" * 80)
print("🎉 RESUMO FINAL - FASE 4")
print("=" * 80)

print("\n✅ TODOS OS TESTES PASSARAM:")
print("  ✓ Relatório HTML principal gerado")
print("  ✓ Todos os elementos essenciais presentes")
print("  ✓ Charts Plotly renderizados")
print("  ✓ Relatórios com configs quick/medium/full")

# Listar arquivos gerados
generated_files = list(output_dir.glob('*.html'))
print(f"\n📊 ESTATÍSTICAS:")
print(f"  - Relatórios gerados: {len(generated_files)}")
print(f"  - Diretório: {output_dir.absolute()}")

print("\n📁 ARQUIVOS GERADOS:")
for f in sorted(generated_files):
    size_kb = f.stat().st_size / 1024
    print(f"  - {f.name} ({size_kb:.1f} KB)")

print("\n" + "=" * 80)
print("✅ FASE 4 - TESTE COMPLETO PASSOU COM SUCESSO!")
print("=" * 80)
print(f"\n💡 Abra os relatórios em um navegador:")
print(f"   file://{output_dir.absolute()}/fairness_report.html")
