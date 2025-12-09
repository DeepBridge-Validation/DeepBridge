# Knowledge Distillation para Economia

Este diretório contém exemplos práticos do paper "Destilação de Conhecimento para Economia: Negociando Complexidade por Interpretabilidade em Modelos Econométricos".

## Visão Geral

O framework de destilação econométrica permite destilar modelos complexos (XGBoost, Neural Networks) para modelos interpretáveis (GAM, Linear), preservando:

1. **Intuição econômica** - Coeficientes e efeitos marginais interpretáveis
2. **Restrições econômicas** - Monotonia, consistência de sinais
3. **Estabilidade de coeficientes** - Inferência estatística válida

## Exemplos Disponíveis

### 1. Credit Risk Demo (`01_credit_risk_demo.py`)
Demonstra destilação com restrições econômicas para previsão de risco de crédito:
- Restrições de sinal (income → default negativo)
- Restrições de monotonia (age, employment length)
- Comparação: Teacher (XGBoost) vs Student (GAM) vs Baseline

**Execute:**
```bash
python 01_credit_risk_demo.py
```

### 2. Labor Economics Demo (`02_labor_economics_demo.py`)
Análise de economia do trabalho com efeitos marginais:
- Predição de probabilidade de emprego
- Efeitos marginais de educação
- Restrições econômicas de mercado de trabalho

**Execute:**
```bash
python 02_labor_economics_demo.py
```

### 3. Stability Analysis Demo (`03_stability_analysis_demo.py`)
Análise de estabilidade de coeficientes via bootstrap:
- Bootstrap resampling (1000 amostras)
- Coeficiente de variação (CV)
- Intervalos de confiança
- Sign stability

**Execute:**
```bash
python 03_stability_analysis_demo.py
```

### 4. Structural Breaks Demo (`04_structural_breaks_demo.py`)
Detecção de quebras estruturais em séries temporais:
- Rolling window analysis
- Teste de Wald para quebras
- Interpretação econômica de mudanças
- Análise pré/pós-crise 2008

**Execute:**
```bash
python 04_structural_breaks_demo.py
```

### 5. Complete Economic Distillation Demo (`05_complete_demo.py`)
Demonstração completa do pipeline:
- Treinamento do teacher
- Definição de restrições econômicas
- Destilação com constraints
- Análise de estabilidade
- Detecção de quebras estruturais
- Relatório econômico

**Execute:**
```bash
python 05_complete_demo.py
```

## Conceitos Principais

### Economic Constraints

```python
from deepbridge.distillation.economics import EconomicConstraints

constraints = EconomicConstraints()

# Sign constraint: higher income → lower default risk
constraints.add_sign('income', sign=-1,
                    justification="Higher income reduces default risk")

# Monotonicity: age → default (increasing up to 65)
constraints.add_monotonicity('age', direction='increasing', bounds=(18, 65))

# Magnitude bound
constraints.add_magnitude('interest_rate', lower=0.5, upper=2.0)
```

### Economic Distillation

```python
from deepbridge.distillation.economics import EconomicDistiller

distiller = EconomicDistiller(
    teacher=teacher_model,
    constraints=constraints,
    student_type='gam',
    temperature=2.0,
    alpha=0.5,
    beta=0.3  # Constraint penalty weight
)

student = distiller.fit(X_train, y_train)
```

### Stability Analysis

```python
from deepbridge.distillation.economics import StabilityAnalyzer

analyzer = StabilityAnalyzer(n_bootstrap=1000)
results = analyzer.analyze(distiller, X_train, y_train)

print(f"Coefficient of Variation: {results['cv']}")
print(f"Sign Stability: {results['sign_stability']}")
```

### Structural Break Detection

```python
from deepbridge.distillation.economics import StructuralBreakDetector

detector = StructuralBreakDetector(window_size=500, step_size=100)
breaks = detector.detect(X, y, teacher_probs, time_var='date')

for break_info in breaks['breaks']:
    print(f"Break detected at window {break_info['window']}")
    print(f"Changed features: {break_info['changed_features']}")
```

## Métricas Econômicas

O framework calcula métricas especializadas:

1. **Constraint Compliance Rate**: Taxa de conformidade com restrições
2. **Marginal Effect Preservation**: Preservação de efeitos marginais vs teacher
3. **Economic Interpretability Score**: Score agregado (0-100%)
4. **Coefficient Stability (CV)**: Coeficiente de variação < 0.15
5. **Sign Stability**: Estabilidade de sinal > 95%

## Resultados Esperados

De acordo com o paper, os resultados típicos são:

- **Trade-off Acurácia-Interpretabilidade**: Perda de 2-5% em acurácia vs teacher
- **Estabilidade de Coeficientes**: CV < 0.15 para coeficientes principais
- **Conformidade Econômica**: 95%+ das restrições preservadas
- **Ganho sobre Baseline**: +8-12% em acurácia vs regressão linear direta

## Dependências

```bash
pip install deepbridge numpy pandas scikit-learn statsmodels scipy
```

## Referências

Paper: "Destilação de Conhecimento para Economia: Negociando Complexidade por Interpretabilidade em Modelos Econométricos"

Seções relevantes:
- Seção 3: Design do Framework
- Seção 4: Implementação
- Seção 5: Avaliação (Case Studies)

## Notas de Implementação

**IMPORTANTE**: Alguns componentes específicos do paper (como `EconomicConstraints`, `EconomicDistiller`, `StabilityAnalyzer`, `StructuralBreakDetector`) são demonstrados conceitualmente nestes exemplos. A implementação completa está descrita no paper na Seção 4.

Os exemplos demonstram:
1. Como os componentes **seriam** usados quando implementados
2. Como adaptar as técnicas de destilação existentes do DeepBridge
3. Como calcular as métricas econômicas manualmente
4. Como realizar análises de estabilidade e detecção de quebras

Para implementação em produção, consulte a Seção 4 do paper para detalhes completos.

## Estrutura dos Arquivos

```
09_knowledge_Economics/
├── README.md                           # Este arquivo
├── 01_credit_risk_demo.py              # Risco de crédito
├── 02_labor_economics_demo.py          # Economia do trabalho
├── 03_stability_analysis_demo.py       # Análise de estabilidade
├── 04_structural_breaks_demo.py        # Quebras estruturais
└── 05_complete_demo.py                 # Demo completo
```

## Questões e Suporte

Para questões sobre a implementação, consulte:
- Paper: `/home/guhaase/projetos/DeepBridge/papers/15_Knowledge_Distillation_Economics/POR/`
- DeepBridge Docs: `deepbridge/docs/`
