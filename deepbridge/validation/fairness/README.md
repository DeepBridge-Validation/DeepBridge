# Fairness Testing Module - DeepBridge

## 📋 Visão Geral

O módulo de Fairness Testing do DeepBridge fornece ferramentas abrangentes para avaliar se modelos de Machine Learning apresentam discriminação contra grupos protegidos (protected groups).

**Compliance**: Segue padrões regulatórios de Banking (EEOC, ECOA), Healthcare (HIPAA), e Fair Lending Act.

**Baseado em**: AI Fairness 360 (IBM), Fairlearn (Microsoft), Aequitas, e frameworks acadêmicos state-of-the-art.

---

## 🎯 Quando Usar

### Casos de Uso Críticos

1. **Banking & Finance**
   - Aprovação de crédito
   - Limite de cartão de crédito
   - Taxas de juros
   - Detecção de fraude

2. **Healthcare**
   - Diagnóstico médico
   - Priorização de tratamento
   - Alocação de recursos

3. **Insurance**
   - Cálculo de prêmios
   - Aprovação de cobertura
   - Avaliação de risco

4. **Employment**
   - Seleção de candidatos
   - Promoções
   - Avaliações de performance

5. **Lending**
   - Empréstimos pessoais
   - Hipotecas
   - Financiamento de veículos

### Quando é Obrigatório

- ✅ Aplicações em setores altamente regulados
- ✅ Modelos que impactam decisões sobre pessoas
- ✅ Sistemas com atributos protegidos (raça, gênero, idade, etc.)
- ✅ Contextos onde discriminação é ilegal ou antiética

---

## 🚀 Quick Start

### Instalação

Já incluído no DeepBridge! Nenhuma instalação adicional necessária.

### Exemplo Básico

```python
from deepbridge.core.db_data import DBDataset
from deepbridge.core.experiment import Experiment
import pandas as pd

# 1. Preparar dados com atributos protegidos
X = pd.DataFrame({
    'income': [...],
    'age': [...],
    'credit_score': [...],
    'gender': ['M', 'F', ...],  # Protected
    'race': ['White', 'Black', ...]  # Protected
})
y = [...]  # Target binary

# 2. Treinar modelo
model = LogisticRegression()
model.fit(X, y)

# 3. Criar dataset
dataset = DBDataset(features=X, target=y, model=model)

# 4. Executar testes de fairness
experiment = Experiment(
    dataset=dataset,
    experiment_type="binary_classification",
    tests=["fairness"],
    protected_attributes=['gender', 'race']  # ← Especificar atributos protegidos
)

fairness_results = experiment.run_fairness_tests(config='full')

# 5. Analisar resultados
print(f"Fairness Score: {fairness_results.overall_fairness_score:.3f}")
print(f"Critical Issues: {len(fairness_results.critical_issues)}")

for issue in fairness_results.critical_issues:
    print(f"  - {issue}")
```

---

## 📊 Métricas de Fairness Implementadas

### 1. Statistical Parity (Demographic Parity)

**Definição**: Taxa de predições positivas deve ser igual entre grupos.

**Fórmula**: `P(Ŷ=1 | A=a) = P(Ŷ=1 | A=b)` para todos os grupos a, b

**Quando usar**:
- Garantir representação igual em outcomes positivos
- Contextos onde "equal treatment" é esperado

**Compliance**: Regra dos 80% da EEOC

```python
result = fairness_results.results['metrics']['gender']['statistical_parity']
print(f"Ratio: {result['ratio']:.3f}")
print(f"Passes 80% rule: {result['passes_80_rule']}")  # Must be >= 0.8
```

**Interpretação**:
- `ratio >= 0.95`: Excelente
- `ratio >= 0.80`: Compliant com EEOC
- `ratio < 0.80`: ⚠️ Evidência de discriminação

---

### 2. Equal Opportunity

**Definição**: True Positive Rate (TPR/Recall) deve ser igual entre grupos.

**Fórmula**: `P(Ŷ=1 | Y=1, A=a) = P(Ŷ=1 | Y=1, A=b)`

**Quando usar**:
- Focar em "benefícios" (outcomes positivos)
- Garantir que grupos têm igual chance de serem identificados quando qualificados

**Exemplo**: Aprovação de crédito - qualificados devem ter mesma taxa de aprovação independente de raça.

```python
result = fairness_results.results['metrics']['race']['equal_opportunity']
print(f"TPR Disparity: {result['disparity']:.3f}")
print(f"Group TPRs: {result['group_tpr']}")
```

**Interpretação**:
- `disparity < 0.05`: Excelente
- `disparity < 0.10`: Aceitável
- `disparity >= 0.20`: ⚠️ Crítico

---

### 3. Equalized Odds

**Definição**: TPR E FPR devem ser iguais entre grupos.

**Fórmula**: `P(Ŷ=1 | Y=y, A=a) = P(Ŷ=1 | Y=y, A=b)` para y ∈ {0,1}

**Quando usar**:
- Garantir fairness em benefícios E harms
- Aplicações onde false positives são críticos (ex: justiça criminal)

**Mais rigoroso** que Equal Opportunity.

```python
result = fairness_results.results['metrics']['gender']['equalized_odds']
print(f"TPR Disparity: {result['tpr_disparity']:.3f}")
print(f"FPR Disparity: {result['fpr_disparity']:.3f}")
print(f"Combined: {result['combined_disparity']:.3f}")
```

---

### 4. Disparate Impact

**Definição**: Razão entre taxa de seleção do grupo menos/mais favorecido.

**Fórmula**: `DI = P(Ŷ=1 | A=unprivileged) / P(Ŷ=1 | A=privileged)`

**Legal Threshold**: Ratio < 0.8 = evidência de discriminação (EEOC)

**Quando usar**:
- Compliance legal mandatório
- Banking, lending, employment

```python
result = fairness_results.results['metrics']['race']['disparate_impact']
print(f"Disparate Impact Ratio: {result['ratio']:.3f}")
print(f"Passes Threshold (0.8): {result['passes_threshold']}")  # CRÍTICO!
```

**Legal Risk**:
- `ratio < 0.8`: 🚨 ALTO RISCO LEGAL - ação necessária
- `ratio >= 0.8`: Compliant com EEOC

---

## ⚙️ Configurações de Teste

### Quick (2 métricas)
```python
fairness_results = experiment.run_fairness_tests(config='quick')
```
- Statistical Parity
- Disparate Impact
- **Tempo**: ~5-10s
- **Uso**: Exploração rápida

### Medium (3 métricas)
```python
fairness_results = experiment.run_fairness_tests(config='medium')
```
- Statistical Parity
- Equal Opportunity
- Disparate Impact
- **Tempo**: ~10-20s
- **Uso**: Validação padrão

### Full (4 métricas - RECOMENDADO)
```python
fairness_results = experiment.run_fairness_tests(config='full')
```
- Statistical Parity
- Equal Opportunity
- Equalized Odds
- Disparate Impact
- **Tempo**: ~15-30s
- **Uso**: Auditoria completa, compliance

---

## 🔍 Análise de Resultados

### Overall Fairness Score

Score agregado (0-1, higher = more fair) combinando todas as métricas:

```python
score = fairness_results.overall_fairness_score

if score >= 0.95:
    print("EXCELENTE - Modelo altamente fair")
elif score >= 0.85:
    print("BOM - Adequado para produção")
elif score >= 0.70:
    print("MODERADO - Requer atenção")
else:
    print("CRÍTICO - Intervenção necessária")
```

**Pesos**:
- Disparate Impact: 30% (mais crítico legalmente)
- Statistical Parity: 25%
- Equal Opportunity: 25%
- Equalized Odds: 20%

### Critical Issues

Issues que violam thresholds legais ou éticos:

```python
for issue in fairness_results.critical_issues:
    print(f"🚨 {issue}")

# Exemplo de output:
# 🚨 race: Disparate Impact CRÍTICO (ratio=0.65 < 0.8) - RISCO LEGAL
```

### Warnings

Issues que não são críticos mas requerem atenção:

```python
for warning in fairness_results.warnings:
    print(f"⚠️  {warning}")

# Exemplo de output:
# ⚠️  gender: Falha na regra dos 80% (ratio=0.78)
```

### Detailed Per-Attribute Analysis

```python
# Analisar atributo específico
gender_metrics = fairness_results.results['metrics']['gender']

# Statistical Parity
sp = gender_metrics['statistical_parity']
print(f"Group rates: {sp['group_rates']}")
# Exemplo: {'M': 0.65, 'F': 0.48}

print(f"Interpretation: {sp['interpretation']}")
# "MODERADO: Alguma disparidade presente - requer investigação"
```

---

## 🛠️ API Reference

### FairnessMetrics

Classe com métodos estáticos para cálculo de métricas:

```python
from deepbridge.validation.fairness.metrics import FairnessMetrics

# Statistical Parity
result = FairnessMetrics.statistical_parity(y_pred, sensitive_feature)

# Equal Opportunity
result = FairnessMetrics.equal_opportunity(y_true, y_pred, sensitive_feature)

# Equalized Odds
result = FairnessMetrics.equalized_odds(y_true, y_pred, sensitive_feature)

# Disparate Impact
result = FairnessMetrics.disparate_impact(y_pred, sensitive_feature, threshold=0.8)
```

### FairnessSuite

Suite completa de testes:

```python
from deepbridge.validation.wrappers.fairness_suite import FairnessSuite

suite = FairnessSuite(
    dataset=dataset,
    protected_attributes=['gender', 'race'],
    verbose=True
)

results = suite.config('full').run()
```

### FairnessResult

Object com resultados estruturados:

```python
# Properties
fairness_results.overall_fairness_score  # float: 0-1
fairness_results.critical_issues  # list
fairness_results.warnings  # list
fairness_results.protected_attributes  # list

# Methods
fairness_results.to_dict()  # Convert to dict
fairness_results.results  # Access raw results dict
```

---

## 📚 Exemplos Avançados

### Exemplo 1: Comparar Fairness de Múltiplos Modelos

```python
models = {
    'LogisticRegression': LogisticRegression(),
    'RandomForest': RandomForestClassifier(),
    'XGBoost': XGBClassifier()
}

fairness_scores = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    dataset = DBDataset(features=X_test, target=y_test, model=model)

    exp = Experiment(
        dataset=dataset,
        experiment_type="binary_classification",
        tests=["fairness"],
        protected_attributes=['gender', 'race']
    )

    results = exp.run_fairness_tests(config='full')
    fairness_scores[name] = results.overall_fairness_score

# Identificar modelo mais fair
best_model = max(fairness_scores, key=fairness_scores.get)
print(f"Modelo mais fair: {best_model} (score={fairness_scores[best_model]:.3f})")
```

### Exemplo 2: Integrar com Robustness Testing

```python
experiment = Experiment(
    dataset=dataset,
    experiment_type="binary_classification",
    tests=["robustness", "fairness"],  # Múltiplos testes
    protected_attributes=['gender', 'race']
)

# Run both tests
robustness_results = experiment.run_tests(config='full')
fairness_results = experiment.run_fairness_tests(config='full')

# Modelo deve ser robusto E fair
print(f"Robustness Score: {robustness_results.robustness_score:.3f}")
print(f"Fairness Score: {fairness_results.overall_fairness_score:.3f}")
```

---

## ⚠️ Limitações e Considerações

### 1. Remover Protected Attributes NÃO é suficiente

```python
# ❌ INCORRETO: Apenas remover 'gender' e 'race' do treinamento
X_train = df[['income', 'credit_score']]  # Remove protected attrs
model.fit(X_train, y)

# ⚠️ Modelo ainda pode ter bias devido a PROXY FEATURES
# Ex: 'income' correlacionado com 'race' por desigualdade histórica
```

**Solução**: Usar técnicas de de-biasing (fairness-aware ML).

### 2. Trade-off entre Fairness e Accuracy

Aumentar fairness pode reduzir accuracy:

```python
# Pode precisar aceitar accuracy ligeiramente menor para ganhar fairness
# Ex: Accuracy 0.92 → 0.90, mas Fairness 0.70 → 0.90
```

### 3. Choice de Métrica Depende do Contexto

- **Banking**: Disparate Impact (mandatório por lei)
- **Healthcare**: Equal Opportunity (benefícios iguais)
- **Criminal Justice**: Equalized Odds (harms e benefits)

### 4. Interseccionalidade

Testar múltiplos atributos pode não capturar intersecção:

```python
# Pode passar em 'gender' E 'race' separadamente,
# mas falhar em 'Black women' especificamente
```

---

## 📖 Referências

### Frameworks e Libraries
- [AI Fairness 360 (IBM)](https://github.com/Trusted-AI/AIF360)
- [Fairlearn (Microsoft)](https://github.com/fairlearn/fairlearn)
- [Aequitas (U. Chicago)](https://github.com/dssg/aequitas)

### Regulações
- EEOC Uniform Guidelines (1978)
- Equal Credit Opportunity Act (ECOA)
- Fair Lending Act
- GDPR Article 22 (EU)

### Artigos Acadêmicos
- Mehrabi et al. (2021): "A Survey on Bias and Fairness in Machine Learning"
- Barocas & Selbst (2016): "Big Data's Disparate Impact"

---

## 🤝 Contribuindo

Encontrou um bug ou tem uma sugestão? Abra uma issue no GitHub!

---

**Versão**: 1.0
**Data**: Outubro 2025
**Autor**: DeepBridge Team
