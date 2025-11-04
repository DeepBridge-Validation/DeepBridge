"""
Script de teste para validar as visualizações do FairnessSuite (Fase 3).

Este script testa todos os 6 métodos de visualização:
1. plot_distribution_by_group
2. plot_metrics_comparison
3. plot_threshold_impact
4. plot_confusion_matrices
5. plot_fairness_radar
6. plot_group_comparison
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

from deepbridge.core.db_data import DBDataset
from deepbridge.validation.wrappers.fairness_suite import FairnessSuite
from deepbridge.validation.fairness.visualizations import FairnessVisualizer

print("=" * 80)
print("TESTE DA FASE 3 - SISTEMA DE VISUALIZAÇÕES")
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

print(f"  Overall Fairness Score: {results['overall_fairness_score']:.3f}")
print(f"  Warnings: {len(results['warnings'])}")
print(f"  Critical Issues: {len(results['critical_issues'])}")

# Criar diretório para salvar visualizações
output_dir = Path('./test_visualizations_output')
output_dir.mkdir(exist_ok=True)
print(f"\n📁 Diretório de saída: {output_dir.absolute()}")

# ============================================================================
# 4. TESTE 1: plot_distribution_by_group
# ============================================================================

print("\n" + "=" * 80)
print("📊 TESTE 1: plot_distribution_by_group")
print("=" * 80)

try:
    # Testar para gênero
    output_path = output_dir / 'distribution_gender.png'
    result = FairnessVisualizer.plot_distribution_by_group(
        df=df_with_preds,
        target_col='target',
        sensitive_feature='gender',
        output_path=str(output_path)
    )

    assert output_path.exists(), "Arquivo não foi criado"
    print(f"  ✓ Distribuição por gênero salva: {output_path}")

    # Testar para raça
    output_path = output_dir / 'distribution_race.png'
    result = FairnessVisualizer.plot_distribution_by_group(
        df=df_with_preds,
        target_col='target',
        sensitive_feature='race',
        output_path=str(output_path)
    )

    assert output_path.exists(), "Arquivo não foi criado"
    print(f"  ✓ Distribuição por raça salva: {output_path}")

    print("  ✅ plot_distribution_by_group: PASSOU")

except Exception as e:
    print(f"  ❌ ERRO: {e}")
    raise

# ============================================================================
# 5. TESTE 2: plot_metrics_comparison
# ============================================================================

print("\n" + "=" * 80)
print("📊 TESTE 2: plot_metrics_comparison")
print("=" * 80)

try:
    output_path = output_dir / 'metrics_comparison.png'
    result = FairnessVisualizer.plot_metrics_comparison(
        metrics_results=results['posttrain_metrics'],
        protected_attrs=['gender', 'race'],
        output_path=str(output_path)
    )

    assert output_path.exists(), "Arquivo não foi criado"
    print(f"  ✓ Comparação de métricas salva: {output_path}")
    print("  ✅ plot_metrics_comparison: PASSOU")

except Exception as e:
    print(f"  ❌ ERRO: {e}")
    raise

# ============================================================================
# 6. TESTE 3: plot_threshold_impact
# ============================================================================

print("\n" + "=" * 80)
print("📊 TESTE 3: plot_threshold_impact")
print("=" * 80)

try:
    # Testar para gênero
    output_path = output_dir / 'threshold_impact_gender.png'
    result = FairnessVisualizer.plot_threshold_impact(
        threshold_results=results['threshold_analysis'],
        metrics=['disparate_impact_ratio', 'statistical_parity', 'f1_score'],
        output_path=str(output_path)
    )

    assert output_path.exists(), "Arquivo não foi criado"
    print(f"  ✓ Impacto de threshold salvo: {output_path}")
    print("  ✅ plot_threshold_impact: PASSOU")

except Exception as e:
    print(f"  ❌ ERRO: {e}")
    raise

# ============================================================================
# 7. TESTE 4: plot_confusion_matrices
# ============================================================================

print("\n" + "=" * 80)
print("📊 TESTE 4: plot_confusion_matrices")
print("=" * 80)

try:
    # Testar para gênero
    output_path = output_dir / 'confusion_matrices_gender.png'
    result = FairnessVisualizer.plot_confusion_matrices(
        cm_by_group=results['confusion_matrix']['gender'],
        attribute_name='gender',
        output_path=str(output_path)
    )

    assert output_path.exists(), "Arquivo não foi criado"
    print(f"  ✓ Matrizes de confusão (gender) salvas: {output_path}")

    # Testar para raça
    output_path = output_dir / 'confusion_matrices_race.png'
    result = FairnessVisualizer.plot_confusion_matrices(
        cm_by_group=results['confusion_matrix']['race'],
        attribute_name='race',
        output_path=str(output_path)
    )

    assert output_path.exists(), "Arquivo não foi criado"
    print(f"  ✓ Matrizes de confusão (race) salvas: {output_path}")
    print("  ✅ plot_confusion_matrices: PASSOU")

except Exception as e:
    print(f"  ❌ ERRO: {e}")
    raise

# ============================================================================
# 8. TESTE 5: plot_fairness_radar
# ============================================================================

print("\n" + "=" * 80)
print("📊 TESTE 5: plot_fairness_radar")
print("=" * 80)

try:
    # Preparar summary de métricas
    metrics_summary = {}
    for attr in ['gender', 'race']:
        metrics_summary[attr] = results['posttrain_metrics'][attr]

    output_path = output_dir / 'fairness_radar.png'
    result = FairnessVisualizer.plot_fairness_radar(
        metrics_summary=metrics_summary,
        output_path=str(output_path)
    )

    assert output_path.exists(), "Arquivo não foi criado"
    print(f"  ✓ Radar de fairness salvo: {output_path}")
    print("  ✅ plot_fairness_radar: PASSOU")

except Exception as e:
    print(f"  ❌ ERRO: {e}")
    raise

# ============================================================================
# 9. TESTE 6: plot_group_comparison
# ============================================================================

print("\n" + "=" * 80)
print("📊 TESTE 6: plot_group_comparison")
print("=" * 80)

try:
    # Testar para gênero
    output_path = output_dir / 'group_comparison_gender.png'
    result = FairnessVisualizer.plot_group_comparison(
        metrics_results=results['posttrain_metrics'],
        attribute_name='gender',
        output_path=str(output_path)
    )

    assert output_path.exists(), "Arquivo não foi criado"
    print(f"  ✓ Comparação de grupos (gender) salva: {output_path}")

    # Testar para raça
    output_path = output_dir / 'group_comparison_race.png'
    result = FairnessVisualizer.plot_group_comparison(
        metrics_results=results['posttrain_metrics'],
        attribute_name='race',
        output_path=str(output_path)
    )

    assert output_path.exists(), "Arquivo não foi criado"
    print(f"  ✓ Comparação de grupos (race) salva: {output_path}")
    print("  ✅ plot_group_comparison: PASSOU")

except Exception as e:
    print(f"  ❌ ERRO: {e}")
    raise

# ============================================================================
# 10. RESUMO FINAL
# ============================================================================

print("\n" + "=" * 80)
print("🎉 RESUMO FINAL - FASE 3")
print("=" * 80)

print("\n✅ TODOS OS TESTES PASSARAM:")
print("  ✓ plot_distribution_by_group (2 variantes)")
print("  ✓ plot_metrics_comparison")
print("  ✓ plot_threshold_impact")
print("  ✓ plot_confusion_matrices (2 variantes)")
print("  ✓ plot_fairness_radar")
print("  ✓ plot_group_comparison (2 variantes)")

# Contar arquivos gerados
generated_files = list(output_dir.glob('*.png'))
print(f"\n📊 ESTATÍSTICAS:")
print(f"  - Visualizações geradas: {len(generated_files)}")
print(f"  - Diretório: {output_dir.absolute()}")

print("\n📁 ARQUIVOS GERADOS:")
for f in sorted(generated_files):
    print(f"  - {f.name}")

print("\n" + "=" * 80)
print("✅ FASE 3 - TESTE COMPLETO PASSOU COM SUCESSO!")
print("=" * 80)
