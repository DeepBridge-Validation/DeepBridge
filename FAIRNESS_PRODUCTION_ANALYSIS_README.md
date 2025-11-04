# Análise de Fairness - Dados de Produção

Este documento explica como executar a análise de fairness nos seus dados de produção.

## 📊 Sobre os Dados

**Arquivos**:
- `train_predictions.parquet` - Dados de treino com predições
- `test_predictions.parquet` - Dados de teste com predições

**Colunas**:
- **Target**: `in_cmst_fun` (0 ou 1)
- **Probabilidades**: `pred_proba_class_0`, `pred_proba_class_1`
- **Classe Predita**: `pred_class` (usando threshold customizado)
- **Atributos Protegidos**:
  - `nm_tip_gnr` (gênero)
  - `nm_tip_raca` (raça)
  - `vl_idd_aa` (idade em anos)

---

## 🚀 Opção 1: Análise Completa (Recomendado)

### Script: `analyze_fairness_production.py`

Análise detalhada com todas as métricas, visualizações e relatórios.

**Características**:
- ✅ 15 métricas de fairness
- ✅ Threshold analysis (99 pontos)
- ✅ Relatórios HTML interativos
- ✅ Visualizações estáticas (PNG)
- ✅ Análise exploratória completa
- ⏱️ Tempo: ~10-15 minutos

### Como Executar

```bash
# Executar análise completa
python analyze_fairness_production.py
```

### Output

```
fairness_production_analysis/
├── fairness_report_quick.html      # Relatório análise rápida
├── fairness_report_full.html       # Relatório análise completa
├── distribution_nm_tip_gnr.png     # Distribuição por gênero
├── distribution_nm_tip_raca.png    # Distribuição por raça
├── distribution_age_group.png      # Distribuição por idade
├── metrics_comparison.png          # Comparação de métricas
└── fairness_radar.png              # Radar chart de fairness
```

---

## ⚡ Opção 2: Análise Rápida

### Script: `analyze_fairness_quick.py`

Análise simplificada focada nos resultados principais.

**Características**:
- ✅ Métricas principais (Disparate Impact, Statistical Parity, Equal Opportunity)
- ✅ Relatório HTML
- ✅ Recomendações diretas
- ⏱️ Tempo: ~5-8 minutos

### Como Executar

```bash
# Executar análise rápida
python analyze_fairness_quick.py
```

### Output

```
fairness_quick_analysis/
└── fairness_report.html            # Relatório completo
```

---

## 📖 Interpretação dos Resultados

### Overall Fairness Score

| Score | Interpretação | Ação |
|-------|--------------|------|
| **0.90 - 1.00** | ✅ Excelente | Modelo aprovado para produção |
| **0.80 - 0.89** | ✓ Boa | Revisar warnings, considerar melhorias |
| **0.70 - 0.79** | ⚠️ Moderada | Melhorias recomendadas antes de deploy |
| **< 0.70** | ❌ Crítica | NÃO recomendado para produção |

### Métricas Principais

#### 1. Disparate Impact (EEOC 80% Rule)

**O que mede**: Razão entre taxa de aprovação do grupo desfavorecido vs. favorecido

**Interpretação**:
- ✅ **≥ 0.80**: EEOC compliant (OK)
- ⚠️ **0.70-0.79**: Zona de atenção
- ❌ **< 0.70**: Violação EEOC

**Exemplo**:
```
nm_tip_gnr: 0.75 (✗ VIOLADO)
  → Mulheres têm 75% da taxa de aprovação dos homens
  → Abaixo do limite EEOC de 80%
```

#### 2. Statistical Parity

**O que mede**: Diferença absoluta entre taxas de aprovação

**Interpretação**:
- ✅ **|valor| < 0.10**: OK (diferença < 10%)
- ⚠️ **0.10 ≤ |valor| < 0.20**: Atenção
- ❌ **|valor| ≥ 0.20**: Crítico

**Exemplo**:
```
nm_tip_raca: -0.18 (⚠️ ATENÇÃO)
  → Grupo desfavorecido tem 18% menos aprovações
```

#### 3. Equal Opportunity

**O que mede**: Diferença na taxa de verdadeiros positivos (entre quem DEVERIA ser aprovado)

**Interpretação**:
- ✅ **|valor| < 0.10**: OK
- ⚠️ **0.10 ≤ |valor| < 0.15**: Atenção
- ❌ **|valor| ≥ 0.15**: Crítico

**Exemplo**:
```
nm_tip_gnr: 0.12 (⚠️ ATENÇÃO)
  → Entre qualificados, homens têm 12% mais chance de aprovação
```

---

## 🔧 Customização

### Ajustar Atributos Protegidos

Edite as variáveis no início dos scripts:

```python
# Em analyze_fairness_production.py ou analyze_fairness_quick.py

# Exemplo 1: Apenas gênero
PROTECTED_ATTRIBUTES = ['nm_tip_gnr']

# Exemplo 2: Gênero e raça
PROTECTED_ATTRIBUTES = ['nm_tip_gnr', 'nm_tip_raca']

# Exemplo 3: Todos (incluindo idade)
PROTECTED_ATTRIBUTES = ['nm_tip_gnr', 'nm_tip_raca', 'age_group']
```

### Ajustar Grupos Etários

Modifique os bins no script:

```python
# Grupos etários customizados
df['age_group'] = pd.cut(
    df['vl_idd_aa'],
    bins=[0, 25, 35, 45, 55, 100],      # Customize aqui
    labels=['<25', '25-34', '35-44', '45-54', '55+'],  # E aqui
    include_lowest=True
)
```

### Mudar Configuração da Análise

```python
# Análise rápida (2 métricas, ~30 segundos)
result = experiment.run_fairness_tests(config='quick')

# Análise média (5 métricas + pré-treino, ~2 minutos)
result = experiment.run_fairness_tests(config='medium')

# Análise completa (15 métricas + threshold, ~10 minutos)
result = experiment.run_fairness_tests(config='full')
```

---

## 🐛 Troubleshooting

### Erro: "Colunas faltando"

**Problema**: Script não encontra as colunas esperadas

**Solução**:
1. Verificar nomes das colunas no seu DataFrame:
   ```python
   df = pd.read_parquet("test_predictions.parquet")
   print(df.columns.tolist())
   ```

2. Ajustar variáveis no script:
   ```python
   TARGET_COL = 'sua_coluna_target'
   PROBA_COLS = ['sua_prob_0', 'sua_prob_1']
   PRED_COL = 'sua_pred_class'
   ```

### Erro: "Feature names mismatch"

**Problema**: Modelo espera features diferentes

**Solução**: O wrapper `PrecomputedPredictionsModel` usa as predições já existentes, não re-prediz. Não deve dar este erro.

### Análise muito lenta

**Problema**: Análise demora muito (> 20 minutos)

**Soluções**:

1. **Usar amostragem**:
   ```python
   # Adicionar antes de criar dataset
   df_sample = df.sample(n=10000, random_state=42)
   dataset = DBDataset(data=df_sample, ...)
   ```

2. **Usar config mais leve**:
   ```python
   result = experiment.run_fairness_tests(config='medium')
   ```

3. **Analisar um atributo por vez**:
   ```python
   for attr in ['nm_tip_gnr', 'nm_tip_raca']:
       experiment = Experiment(..., protected_attributes=[attr])
       result = experiment.run_fairness_tests(config='full')
       result.save_html(f'report_{attr}.html')
   ```

### Valores ausentes (NaN)

**Problema**: Dados têm NaN em colunas críticas

**Solução**: O script já remove automaticamente linhas com NaN. Para ver o impacto:
```python
# Antes de criar dataset, verificar
print(f"NaN em target: {df[TARGET_COL].isna().sum()}")
print(f"NaN em gênero: {df['nm_tip_gnr'].isna().sum()}")

# Remover ou imputar conforme necessário
df = df.dropna(subset=['nm_tip_gnr', 'nm_tip_raca'])
```

---

## 📊 Exemplos de Uso

### Exemplo 1: Análise Básica

```bash
# Executar análise completa
python analyze_fairness_production.py

# Abrir relatório
# file://./fairness_production_analysis/fairness_report_full.html
```

### Exemplo 2: Análise por Atributo

Criar script customizado:

```python
# analyze_by_attribute.py
import pandas as pd
from analyze_fairness_production import *

for attr in ['nm_tip_gnr', 'nm_tip_raca', 'age_group']:
    print(f"\n{'='*80}")
    print(f"Analisando: {attr}")
    print(f"{'='*80}")

    experiment = Experiment(
        dataset=dataset,
        experiment_type="binary_classification",
        tests=["fairness"],
        protected_attributes=[attr],  # Um por vez
        test_size=0.2,
        random_state=42
    )

    result = experiment.run_fairness_tests(config='full')

    # Salvar relatório individual
    result.save_html(f'fairness_report_{attr}.html', model_name=f'Analysis - {attr}')

    print(f"Score: {result.overall_fairness_score:.3f}")
    print(f"Critical: {len(result.critical_issues)}")
```

### Exemplo 3: Comparar Train vs. Test

```python
# compare_train_test.py
import pandas as pd

for dataset_name, file_path in [('train', TRAIN_PATH), ('test', TEST_PATH)]:
    df = pd.read_parquet(file_path)

    # Criar modelo e dataset
    model = PrecomputedPredictionsModel(df, PROBA_COLS)
    dataset = DBDataset(data=df, target_column=TARGET_COL, model=model)

    # Análise
    experiment = Experiment(...)
    result = experiment.run_fairness_tests(config='full')

    # Salvar
    result.save_html(f'fairness_report_{dataset_name}.html')

    print(f"{dataset_name}: {result.overall_fairness_score:.3f}")
```

---

## 💡 Próximos Passos

### Se Score ≥ 0.80 (Aprovado)

1. ✅ Revisar relatório HTML detalhado
2. ✅ Validar com stakeholders (legal, ético, negócio)
3. ✅ Documentar resultados para auditoria
4. ✅ Implementar monitoramento contínuo
5. ✅ Deploy em produção

### Se Score < 0.80 (Melhorias Necessárias)

1. ⚠️ Revisar critical issues e warnings no relatório
2. ⚠️ Identificar fontes de viés nos dados
3. ⚠️ Aplicar técnicas de mitigação:

   **Opção A: Re-balanceamento de Dados**
   ```python
   # Re-balancear por grupo antes de treinar
   from sklearn.utils import resample
   # ... código de re-balanceamento
   ```

   **Opção B: Threshold Adjustment**
   ```python
   # Usar threshold ótimo da análise
   optimal_threshold = results['threshold_analysis']['optimal_threshold']
   # Retreinar modelo com threshold customizado
   ```

   **Opção C: Fairness Constraints**
   ```python
   # Usar Fairlearn ou AIF360
   from fairlearn.reductions import ExponentiatedGradient, DemographicParity
   mitigator = ExponentiatedGradient(estimator=model, constraints=DemographicParity())
   ```

4. ⚠️ Re-treinar modelo
5. ⚠️ Re-executar análise de fairness
6. ⚠️ Repetir até Score ≥ 0.80

---

## 📚 Recursos Adicionais

### Documentação DeepBridge

- [Tutorial Completo](docs/FAIRNESS_TUTORIAL.md)
- [Guia de Boas Práticas](docs/FAIRNESS_BEST_PRACTICES.md)
- [FAQ](docs/FAIRNESS_FAQ.md)

### Bibliotecas Complementares

- **AIF360** (IBM): https://github.com/Trusted-AI/AIF360
- **Fairlearn** (Microsoft): https://fairlearn.org/
- **What-If Tool** (Google): https://pair-code.github.io/what-if-tool/

### Regulamentações

- **EEOC 80% Rule** (EUA): https://www.eeoc.gov/
- **GDPR** (Europa): https://gdpr.eu/
- **LGPD** (Brasil): https://www.gov.br/cidadania/pt-br/acesso-a-informacao/lgpd

---

## ❓ Dúvidas?

Para questões sobre:
- **Uso dos scripts**: Consultar este README
- **Interpretação de métricas**: Consultar `docs/FAIRNESS_FAQ.md`
- **Boas práticas**: Consultar `docs/FAIRNESS_BEST_PRACTICES.md`
- **Issues técnicos**: Abrir issue no GitHub

---

**Versão**: 1.0
**Data**: 2025-11-03
**Autor**: DeepBridge Team
