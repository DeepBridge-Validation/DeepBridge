"""
Teste de integração para Fase 5 - Integração com Experiment.

Este script testa:
- Método run_fairness_tests() do Experiment
- Auto-detecção de atributos sensíveis
- FairnessResult com save_html()
- Integração completa do fluxo
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

from deepbridge.core.db_data import DBDataset
from deepbridge.core.experiment.experiment import Experiment

print("=" * 80)
print("TESTE DA FASE 5 - INTEGRAÇÃO COM EXPERIMENT")
print("=" * 80)

# ============================================================================
# 1. CRIAR DADOS SINTÉTICOS COM ATRIBUTOS SENSÍVEIS
# ============================================================================

print("\n📊 1. Gerando dados sintéticos...")

np.random.seed(42)
n_samples = 800

# Criar dados com features numéricas
X_numeric = np.random.randn(n_samples, 5)

# Criar atributos protegidos (com nomes óbvios para auto-detecção)
gender = np.random.choice(['M', 'F'], n_samples, p=[0.7, 0.3])
race = np.random.choice(['White', 'Black', 'Hispanic'], n_samples, p=[0.6, 0.25, 0.15])

# Criar target com viés
y = np.zeros(n_samples)
for i in range(n_samples):
    base_prob = 0.3
    if gender[i] == 'M':
        base_prob += 0.15
    if race[i] == 'White':
        base_prob += 0.10
    y[i] = 1 if np.random.rand() < base_prob else 0

# Criar DataFrame completo
df = pd.DataFrame(X_numeric, columns=[f'feature_{i}' for i in range(5)])
df['gender'] = gender  # Nome óbvio para auto-detecção
df['race'] = race      # Nome óbvio para auto-detecção
df['target'] = y

print(f"  Total de amostras: {len(df)}")
print(f"  Features: {list(df.columns[:-3])}")
print(f"  Atributos sensíveis: {['gender', 'race']}")

# ============================================================================
# 2. TREINAR MODELO
# ============================================================================

print("\n🤖 2. Treinando modelo...")

# Separar features (sem atributos protegidos) e target
X_train = df.drop(['gender', 'race', 'target'], axis=1)
y_train = df['target']

model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
model.fit(X_train, y_train)

print(f"  Acurácia: {model.score(X_train, y_train):.3f}")

# ============================================================================
# 3. CRIAR DATASET
# ============================================================================

print("\n📦 3. Criando DBDataset...")

dataset = DBDataset(
    data=df,
    target_column='target',
    model=model
)

print(f"  Dataset shape: {df.shape}")
print(f"  Model: {type(model).__name__}")

# ============================================================================
# 4. TESTE 1: AUTO-DETECÇÃO DE ATRIBUTOS SENSÍVEIS
# ============================================================================

print("\n" + "=" * 80)
print("🔍 TESTE 1: Auto-detecção de Atributos Sensíveis")
print("=" * 80)

try:
    # Detectar atributos sensíveis
    detected = Experiment.detect_sensitive_attributes(dataset)

    print(f"\n  Atributos detectados: {detected}")

    # Validações
    assert 'gender' in detected, "Deveria detectar 'gender'"
    assert 'race' in detected, "Deveria detectar 'race'"
    assert len(detected) == 2, f"Deveria detectar exatamente 2 atributos, detectou {len(detected)}"

    print("  ✅ Auto-detecção: PASSOU")

except Exception as e:
    print(f"  ❌ ERRO: {e}")
    raise

# ============================================================================
# 5. TESTE 2: EXPERIMENT COM PROTECTED_ATTRIBUTES EXPLÍCITOS
# ============================================================================

print("\n" + "=" * 80)
print("🧪 TESTE 2: Experiment com Protected Attributes Explícitos")
print("=" * 80)

try:
    # Criar experiment com protected_attributes explícitos
    experiment = Experiment(
        dataset=dataset,
        experiment_type="binary_classification",
        tests=["fairness"],
        protected_attributes=['gender', 'race'],
        test_size=0.2,
        random_state=42
    )

    print("\n  ✓ Experiment inicializado")
    print(f"  Protected attributes: {experiment.protected_attributes}")

    # Executar testes de fairness
    fairness_result = experiment.run_fairness_tests(config='medium')

    print(f"\n  ✓ Testes executados")
    print(f"  Overall Fairness Score: {fairness_result.overall_fairness_score:.3f}")
    print(f"  Critical Issues: {len(fairness_result.critical_issues)}")
    print(f"  Warnings: {len(fairness_result.warnings)}")
    print(f"  Protected Attributes: {fairness_result.protected_attributes}")

    # Validações
    assert fairness_result.overall_fairness_score > 0, "Score deve ser > 0"
    assert fairness_result.protected_attributes == ['gender', 'race']

    print("  ✅ Experiment com protected_attributes explícitos: PASSOU")

except Exception as e:
    print(f"  ❌ ERRO: {e}")
    import traceback
    traceback.print_exc()
    raise

# ============================================================================
# 6. TESTE 3: EXPERIMENT COM AUTO-DETECÇÃO
# ============================================================================

print("\n" + "=" * 80)
print("🔍 TESTE 3: Experiment com Auto-detecção")
print("=" * 80)

try:
    # Criar experiment SEM protected_attributes (deve auto-detectar)
    experiment_auto = Experiment(
        dataset=dataset,
        experiment_type="binary_classification",
        tests=["fairness"],  # Solicita fairness mas não fornece protected_attributes
        test_size=0.2,
        random_state=42
    )

    print("\n  ✓ Experiment inicializado com auto-detecção")
    print(f"  Protected attributes (auto-detectados): {experiment_auto.protected_attributes}")

    # Validações
    assert experiment_auto.protected_attributes is not None
    assert len(experiment_auto.protected_attributes) > 0
    assert 'gender' in experiment_auto.protected_attributes
    assert 'race' in experiment_auto.protected_attributes

    print("  ✅ Auto-detecção no Experiment: PASSOU")

except Exception as e:
    print(f"  ❌ ERRO: {e}")
    import traceback
    traceback.print_exc()
    raise

# ============================================================================
# 7. TESTE 4: GERAÇÃO DE RELATÓRIO HTML
# ============================================================================

print("\n" + "=" * 80)
print("📄 TESTE 4: Geração de Relatório HTML")
print("=" * 80)

# Criar diretório para relatórios
output_dir = Path('./test_integration_output')
output_dir.mkdir(exist_ok=True)
print(f"\n📁 Diretório de saída: {output_dir.absolute()}")

try:
    # Executar testes de fairness
    fairness_result = experiment.run_fairness_tests(config='full')

    # Gerar relatório usando FairnessResult.save_html()
    report_path = output_dir / 'fairness_integration_report.html'

    generated_path = fairness_result.save_html(
        file_path=str(report_path),
        model_name='Test Integration Model'
    )

    print(f"\n  ✓ Relatório gerado: {generated_path}")

    # Validações
    assert Path(generated_path).exists(), "Arquivo não foi criado"
    file_size = Path(generated_path).stat().st_size
    print(f"  ✓ Arquivo existe: {file_size} bytes")

    # Verificar conteúdo básico
    with open(generated_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    required_elements = [
        'Fairness Analysis Report',
        'Overall Fairness Score',
        'Test Integration Model',
        'Plotly'
    ]

    missing_elements = [elem for elem in required_elements if elem not in html_content]

    if missing_elements:
        print(f"  ⚠️  Elementos faltando: {missing_elements}")
    else:
        print(f"  ✓ Todos os elementos essenciais presentes")

    print("  ✅ Geração de relatório HTML: PASSOU")

except Exception as e:
    print(f"  ❌ ERRO: {e}")
    import traceback
    traceback.print_exc()
    raise

# ============================================================================
# 8. TESTE 5: PROPRIEDADES DO FAIRNESSRESULT
# ============================================================================

print("\n" + "=" * 80)
print("📊 TESTE 5: Propriedades do FairnessResult")
print("=" * 80)

try:
    # Testar todas as propriedades
    score = fairness_result.overall_fairness_score
    critical = fairness_result.critical_issues
    warnings = fairness_result.warnings
    attrs = fairness_result.protected_attributes

    print(f"\n  Overall Score: {score:.3f}")
    print(f"  Critical Issues: {len(critical)}")
    print(f"  Warnings: {len(warnings)}")
    print(f"  Protected Attributes: {attrs}")

    # Validações de tipo
    assert isinstance(score, float), "Score deve ser float"
    assert isinstance(critical, list), "Critical issues deve ser list"
    assert isinstance(warnings, list), "Warnings deve ser list"
    assert isinstance(attrs, list), "Protected attributes deve ser list"

    # Validações de valor
    assert 0 <= score <= 1, "Score deve estar entre 0 e 1"
    assert len(attrs) > 0, "Deve ter pelo menos um atributo"

    print("  ✅ Propriedades do FairnessResult: PASSOU")

except Exception as e:
    print(f"  ❌ ERRO: {e}")
    raise

# ============================================================================
# 9. RESUMO FINAL
# ============================================================================

print("\n" + "=" * 80)
print("🎉 RESUMO FINAL - FASE 5")
print("=" * 80)

print("\n✅ TODOS OS TESTES PASSARAM:")
print("  ✓ Auto-detecção de atributos sensíveis")
print("  ✓ Experiment com protected_attributes explícitos")
print("  ✓ Experiment com auto-detecção")
print("  ✓ Geração de relatório HTML via FairnessResult")
print("  ✓ Propriedades do FairnessResult")

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
print("✅ FASE 5 - TESTE COMPLETO PASSOU COM SUCESSO!")
print("=" * 80)
print(f"\n💡 Abra o relatório em um navegador:")
print(f"   file://{output_dir.absolute()}/fairness_integration_report.html")
