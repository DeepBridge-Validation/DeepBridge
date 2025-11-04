# Guia de Boas Práticas: Análise de Fairness no DeepBridge

## 📋 Índice

1. [Princípios Fundamentais](#princípios-fundamentais)
2. [Antes de Começar](#antes-de-começar)
3. [Preparação de Dados](#preparação-de-dados)
4. [Seleção de Métricas](#seleção-de-métricas)
5. [Interpretação de Resultados](#interpretação-de-resultados)
6. [Mitigação de Viés](#mitigação-de-viés)
7. [Monitoramento Contínuo](#monitoramento-contínuo)
8. [Considerações Legais e Éticas](#considerações-legais-e-éticas)
9. [Checklist de Validação](#checklist-de-validação)

---

## Princípios Fundamentais

### 1. Fairness é Multidimensional

Não existe uma única definição de fairness. Diferentes contextos exigem diferentes trade-offs:

- **Statistical Parity**: Resultados iguais entre grupos
- **Equal Opportunity**: Igualdade na taxa de verdadeiros positivos
- **Equalized Odds**: Igualdade em TPR e FPR
- **Individual Fairness**: Indivíduos similares tratados similarmente

⚠️ **IMPORTANTE**: É matematicamente impossível satisfazer todas as definições simultaneamente (Impossibility Theorem).

### 2. Contexto Importa

O que é "justo" depende do domínio:

- **Crédito**: EEOC 80% rule, Equal Opportunity
- **Recrutamento**: Disparate Impact, Statistical Parity
- **Saúde**: Equal Opportunity (evitar falsos negativos)
- **Justiça Criminal**: Equalized Odds, False Positive Rate

### 3. Transparência e Documentação

Sempre documente:
- Por que certos atributos foram considerados sensíveis
- Quais métricas foram priorizadas e por quê
- Trade-offs feitos entre fairness e performance
- Limitações conhecidas

---

## Antes de Começar

### ✅ Checklist Pré-Análise

- [ ] **Definir stakeholders**: Quem será afetado pelo modelo?
- [ ] **Identificar grupos protegidos**: Quais atributos são legalmente/eticamente sensíveis?
- [ ] **Estabelecer métricas de sucesso**: O que significa "justo" neste contexto?
- [ ] **Revisar regulamentações**: GDPR, CCPA, LGPD, EEOC, etc.
- [ ] **Obter consentimento**: Dados sensíveis foram coletados eticamente?

### ❌ Armadilhas Comuns

1. **Remover atributos protegidos não elimina viés**
   - Features correlacionadas (proxies) mantêm o viés
   - Exemplo: CEP correlacionado com raça

2. **Alta acurácia ≠ Fairness**
   - Modelo pode ter 95% de acurácia mas viés severo em grupos minoritários

3. **Viés no treinamento = Viés no modelo**
   - Dados históricos frequentemente refletem discriminação passada

---

## Preparação de Dados

### 1. Identificação de Atributos Sensíveis

#### Atributos Explícitos

Use sempre que possível especificar explicitamente:

```python
# ✅ RECOMENDADO: Explícito
experiment = Experiment(
    dataset=dataset,
    experiment_type="binary_classification",
    tests=["fairness"],
    protected_attributes=['gender', 'race', 'age_group']  # Explícito
)
```

```python
# ⚠️  CUIDADO: Auto-detecção (apenas para exploração)
experiment = Experiment(
    dataset=dataset,
    experiment_type="binary_classification",
    tests=["fairness"]  # Auto-detecta
)
```

#### Atributos Comuns

| Categoria | Exemplos |
|-----------|----------|
| **Demográficos** | gender, sex, race, ethnicity, nationality |
| **Idade** | age, age_group, birth_year |
| **Socioeconômicos** | income_bracket, education_level, marital_status |
| **Localização** | zip_code, neighborhood, state |
| **Saúde** | disability, medical_condition |
| **Outros** | religion, sexual_orientation, veteran_status |

### 2. Detecção de Proxies

Verifique correlações entre features e atributos protegidos:

```python
import pandas as pd

# Calcular correlações
correlations = df[feature_cols].corrwith(df['protected_attribute'].astype('category').cat.codes)

# Identificar proxies (correlação > 0.5)
proxies = correlations[abs(correlations) > 0.5]
print(f"Possíveis proxies: {proxies.to_dict()}")
```

**Exemplos de Proxies**:
- CEP → raça/renda
- Nome → etnia/gênero
- Tipo de escola → renda/raça

### 3. Balanceamento de Dados

```python
# Verificar balanceamento por grupo
for attr in protected_attributes:
    distribution = df[attr].value_counts(normalize=True)
    print(f"\n{attr}:")
    print(distribution)

    # Verificar target por grupo
    for group in df[attr].unique():
        target_rate = df[df[attr] == group]['target'].mean()
        print(f"  {group}: {target_rate:.1%}")
```

**Limiares de Atenção**:
- Grupo < 5% da população: Risco de underfitting
- Diferença > 20% na taxa de target: Possível viés nos dados

---

## Seleção de Métricas

### Configurações Recomendadas por Cenário

#### 1. Exploração Inicial (config='quick')

**Quando usar**: Primeira análise, prototipagem rápida

**Métricas**:
- Statistical Parity
- Disparate Impact

```python
fairness_result = experiment.run_fairness_tests(config='quick')
```

**Tempo**: ~10-30 segundos

---

#### 2. Validação Intermediária (config='medium')

**Quando usar**: Após ajustes iniciais, antes de produção

**Métricas**:
- 5 métricas pós-treino (Statistical Parity, Disparate Impact, Equal Opportunity, Equalized Odds, Precision Difference)
- 4 métricas pré-treino (Class Balance, Concept Balance, KL Divergence, JS Divergence)
- Confusion Matrix por grupo

```python
fairness_result = experiment.run_fairness_tests(config='medium')
```

**Tempo**: ~1-3 minutos

---

#### 3. Análise Completa (config='full')

**Quando usar**: Auditoria final, produção, compliance

**Métricas**:
- 11 métricas pós-treino
- 4 métricas pré-treino
- Confusion Matrix por grupo
- Threshold Analysis (99 pontos)

```python
fairness_result = experiment.run_fairness_tests(config='full')
```

**Tempo**: ~5-10 minutos

---

### Métricas por Domínio

| Domínio | Métricas Primárias | Métricas Secundárias |
|---------|-------------------|---------------------|
| **Crédito/Financeiro** | Disparate Impact, Equal Opportunity | Statistical Parity, Conditional Acceptance |
| **Recrutamento** | Statistical Parity, Disparate Impact | Equal Opportunity, Conditional Acceptance |
| **Saúde** | Equal Opportunity, False Negative Rate | Equalized Odds, Precision Difference |
| **Justiça Criminal** | Equalized Odds, False Positive Rate | Statistical Parity, Treatment Equality |
| **Educação** | Equal Opportunity, Statistical Parity | Disparate Impact, Accuracy Difference |

---

## Interpretação de Resultados

### 1. Overall Fairness Score

```python
score = fairness_result.overall_fairness_score
```

**Interpretação**:
- **0.90 - 1.00**: ✅ Excelente - Deploy recomendado
- **0.80 - 0.89**: ✓ Boa - Revisar warnings
- **0.70 - 0.79**: ⚠️  Moderada - Melhorias recomendadas
- **< 0.70**: ❌ Crítica - NÃO deploy

### 2. Análise de Issues

```python
critical = fairness_result.critical_issues
warnings = fairness_result.warnings

print(f"Critical: {len(critical)}")
print(f"Warnings: {len(warnings)}")

# Revisar cada issue
for issue in critical:
    print(f"  - {issue}")
```

**Priorização**:
1. **Critical Issues**: Resolver antes de deploy
2. **Warnings**: Documentar e monitorar
3. **OK**: Verificar periodicamente

### 3. Métricas Individuais

#### Statistical Parity (Paridade Estatística)

```
Valor: -0.15
Interpretação: ⚠️  Amarelo (Warning)
```

**Significado**: Grupo desfavorecido tem 15 pontos percentuais a menos de outcomes positivos

**Ação**:
- Se |valor| < 0.10: OK
- Se 0.10 ≤ |valor| < 0.20: Investigar
- Se |valor| ≥ 0.20: Mitigação necessária

#### Disparate Impact (Impacto Desproporcional)

```
Valor: 0.72
Interpretação: ✗ Vermelho (Critical)
EEOC: 0.80 (não atende)
```

**Significado**: Taxa de aprovação do grupo desfavorecido é 72% da taxa do grupo favorecido

**Ação**:
- Se valor ≥ 0.80: OK (EEOC compliant)
- Se 0.70 ≤ valor < 0.80: Revisar
- Se valor < 0.70: Violação EEOC - mitigação urgente

#### Equal Opportunity (Oportunidade Igual)

```
Valor: 0.08
Interpretação: ✓ Verde (OK)
```

**Significado**: Diferença de 8% na taxa de verdadeiros positivos entre grupos

**Ação**:
- Se |valor| < 0.10: OK
- Se 0.10 ≤ |valor| < 0.15: Atenção
- Se |valor| ≥ 0.15: Mitigação necessária

### 4. Threshold Analysis

```python
# Se disponível (config='full')
if 'threshold_analysis' in fairness_result._results:
    optimal = fairness_result._results['threshold_analysis']['optimal_threshold']
    print(f"Threshold ótimo: {optimal:.3f}")
```

**Uso**:
- Ajustar threshold de decisão para melhorar fairness
- Trade-off: Pode reduzir performance geral
- Testar em validação antes de aplicar em produção

---

## Mitigação de Viés

### 1. Pré-processamento (Antes do Treinamento)

#### Re-balanceamento

```python
from imblearn.over_sampling import SMOTE

# Re-balancear por grupo
for group in df['protected_attr'].unique():
    group_data = df[df['protected_attr'] == group]
    # Aplicar SMOTE ou undersampling
```

#### Remoção de Proxies

```python
# Identificar e remover features correlacionadas
high_corr_features = ['zip_code', 'school_type']  # Exemplo
X_clean = X.drop(columns=high_corr_features)
```

#### Reweighting

```python
from sklearn.utils.class_weight import compute_sample_weight

# Computar pesos para balancear grupos
sample_weights = compute_sample_weight(
    class_weight='balanced',
    y=df['protected_attr']
)

model.fit(X, y, sample_weight=sample_weights)
```

---

### 2. In-processing (Durante o Treinamento)

#### Fairness Constraints

```python
# Exemplo conceitual (requer bibliotecas específicas)
# AIF360, Fairlearn, etc.

from fairlearn.reductions import ExponentiatedGradient, DemographicParity

mitigator = ExponentiatedGradient(
    estimator=base_model,
    constraints=DemographicParity()
)

mitigator.fit(X, y, sensitive_features=df['protected_attr'])
```

#### Adversarial Debiasing

```python
# Treinar com adversarial network
# que tenta prever atributo protegido
# (força o modelo a ser independente)
```

---

### 3. Pós-processamento (Após o Treinamento)

#### Threshold Adjustment

```python
# Usar threshold analysis para encontrar threshold ótimo
if 'threshold_analysis' in results:
    optimal_threshold = results['threshold_analysis']['optimal_threshold']

    # Aplicar threshold customizado
    y_pred_fair = (y_pred_proba >= optimal_threshold).astype(int)
```

#### Calibração por Grupo

```python
from sklearn.calibration import CalibratedClassifierCV

# Calibrar probabilidades separadamente por grupo
calibrated_models = {}

for group in df['protected_attr'].unique():
    group_mask = df['protected_attr'] == group

    calibrated = CalibratedClassifierCV(model, cv=5)
    calibrated.fit(X[group_mask], y[group_mask])

    calibrated_models[group] = calibrated
```

---

### Comparação de Abordagens

| Abordagem | Vantagens | Desvantagens | Quando Usar |
|-----------|-----------|--------------|-------------|
| **Pré-processamento** | Simples, independente do modelo | Pode perder informação | Dados desbalanceados |
| **In-processing** | Integrado no treinamento | Requer modelos específicos | Novo desenvolvimento |
| **Pós-processamento** | Aplicável a modelos existentes | Pode reduzir performance | Modelos já em produção |

---

## Monitoramento Contínuo

### 1. Frequência de Re-avaliação

| Cenário | Frequência | Configuração |
|---------|-----------|--------------|
| **Alto Risco** (crédito, saúde) | Semanal/Mensal | config='full' |
| **Médio Risco** (recrutamento) | Mensal/Trimestral | config='medium' |
| **Baixo Risco** (recomendações) | Trimestral/Anual | config='quick' |

### 2. Pipeline de Monitoramento

```python
def monitor_fairness(model, new_data, protected_attrs):
    """
    Pipeline de monitoramento contínuo de fairness.
    """
    # 1. Criar dataset
    dataset = DBDataset(data=new_data, target_column='target', model=model)

    # 2. Executar análise
    experiment = Experiment(
        dataset=dataset,
        experiment_type="binary_classification",
        tests=["fairness"],
        protected_attributes=protected_attrs
    )

    result = experiment.run_fairness_tests(config='medium')

    # 3. Verificar degradação
    if result.overall_fairness_score < 0.75:
        send_alert(f"Fairness score dropped to {result.overall_fairness_score:.3f}")

    # 4. Gerar relatório
    result.save_html(
        f'monitoring/fairness_{datetime.now().strftime("%Y%m%d")}.html',
        model_name=f'Model Monitoring - {datetime.now().strftime("%Y-%m-%d")}'
    )

    return result
```

### 3. Alertas e Thresholds

```python
# Configurar alertas
ALERT_THRESHOLDS = {
    'overall_fairness_score': 0.75,
    'disparate_impact_min': 0.80,
    'statistical_parity_max': 0.15,
    'critical_issues_max': 0
}

def check_alerts(result):
    alerts = []

    if result.overall_fairness_score < ALERT_THRESHOLDS['overall_fairness_score']:
        alerts.append(f"Overall score: {result.overall_fairness_score:.3f}")

    if len(result.critical_issues) > ALERT_THRESHOLDS['critical_issues_max']:
        alerts.append(f"Critical issues: {len(result.critical_issues)}")

    return alerts
```

---

## Considerações Legais e Éticas

### 1. Regulamentações por Região

#### Estados Unidos

- **Equal Employment Opportunity Commission (EEOC)**: 80% rule
- **Fair Credit Reporting Act (FCRA)**: Transparência em decisões de crédito
- **Fair Housing Act**: Proibição de discriminação em habitação

#### Europa

- **GDPR (General Data Protection Regulation)**:
  - Art. 22: Direito a explicações sobre decisões automatizadas
  - Art. 9: Proibição de processar dados sensíveis sem consentimento

#### Brasil

- **LGPD (Lei Geral de Proteção de Dados)**:
  - Art. 20: Direito de revisão de decisões automatizadas
  - Proibição de discriminação ilícita

### 2. Documentação Legal

Sempre documente:

```markdown
# Documentação de Fairness - [Nome do Modelo]

## 1. Atributos Protegidos
- Gender (justificativa: EEOC protected class)
- Race (justificativa: EEOC protected class)
- Age (justificativa: ADEA - Age Discrimination in Employment Act)

## 2. Métricas e Thresholds
- Disparate Impact ≥ 0.80 (EEOC compliance)
- Statistical Parity ≤ 0.10 (internal policy)

## 3. Resultados
- Overall Fairness Score: 0.85
- Disparate Impact: 0.82 ✓ (EEOC compliant)
- Critical Issues: 0

## 4. Mitigações Aplicadas
- Re-balanceamento de dados por grupo
- Threshold adjustment (0.45 → 0.42)

## 5. Limitações Conhecidas
- Grupos < 5% da população: Asian (3.2%)
- Dados de treinamento: 2020-2023 (pode não refletir mudanças recentes)

## 6. Responsáveis
- Data Scientist: [Nome]
- Legal Review: [Nome]
- Aprovação: [Nome, Data]
```

### 3. Explicabilidade

Combine fairness com explicabilidade:

```python
# SHAP values por grupo
import shap

for group in protected_attributes:
    explainer = shap.TreeExplainer(model)

    group_data = df[df[group] == 'specific_value']
    shap_values = explainer.shap_values(group_data[feature_cols])

    shap.summary_plot(shap_values, group_data[feature_cols])
```

---

## Checklist de Validação

### ✅ Antes do Deploy

- [ ] **Análise completa executada** (config='full')
- [ ] **Overall Fairness Score ≥ 0.80**
- [ ] **Zero critical issues**
- [ ] **Disparate Impact ≥ 0.80** (se aplicável EEOC)
- [ ] **Documentação legal completa**
- [ ] **Aprovação de stakeholders legais/éticos**
- [ ] **Plano de monitoramento definido**
- [ ] **Processo de re-treinamento documentado**

### ✅ Em Produção

- [ ] **Monitoramento ativo** (frequência definida)
- [ ] **Alertas configurados**
- [ ] **Relatórios arquivados**
- [ ] **Logs de decisões mantidos**
- [ ] **Processo de revisão humana disponível**
- [ ] **Canal para reportar viés**

### ✅ Manutenção

- [ ] **Re-análise após cada re-treinamento**
- [ ] **Revisão trimestral de métricas**
- [ ] **Atualização de documentação**
- [ ] **Auditoria anual completa**

---

## Recursos Adicionais

### Bibliotecas Complementares

- **AIF360** (IBM): Técnicas de mitigação
- **Fairlearn** (Microsoft): Fairness-aware learning
- **What-If Tool** (Google): Análise interativa
- **SHAP**: Explicabilidade

### Referências Acadêmicas

1. **Barocas, S., Hardt, M., & Narayanan, A.** (2019). *Fairness and Machine Learning*
2. **Mehrabi, N., et al.** (2021). *A Survey on Bias and Fairness in Machine Learning*
3. **IEEE P7003**: *Algorithmic Bias Considerations*

### Frameworks de Governança

- **EU AI Act**: Regulamentação de IA de alto risco
- **NIST AI Risk Management Framework**
- **ISO/IEC 23894**: *AI Risk Management*

---

## Conclusão

**Princípios-chave para lembrar**:

1. 🎯 **Fairness é um processo, não um destino** - Monitoramento contínuo é essencial
2. 📊 **Múltiplas métricas são necessárias** - Nenhuma métrica única captura tudo
3. 🤝 **Envolva stakeholders** - Decisões de fairness são sociotécnicas
4. 📝 **Documente tudo** - Transparência é fundamental
5. ⚖️ **Trade-offs são inevitáveis** - Balance fairness, performance e complexidade

**Lembre-se**: Tecnologia sozinha não resolve problemas de fairness. É necessário combinar ferramentas técnicas com processos organizacionais, supervisão humana e governança adequada.

---

**Versão**: 1.0
**Última atualização**: 2025-11-03
**Autores**: DeepBridge Team
