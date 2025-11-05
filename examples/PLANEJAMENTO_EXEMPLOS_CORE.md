# PLANEJAMENTO DE EXEMPLOS - MÓDULO CORE

**Documento**: Planejamento Completo de Exemplos do Módulo CORE
**Data**: 04 de Novembro de 2025
**Versão**: 1.0

---

## 📋 Sumário Executivo

Este documento mapeia **todas as funcionalidades do módulo CORE** e propõe exemplos práticos para demonstrar cada capacidade da biblioteca DeepBridge. Os exemplos são organizados por complexidade e prioridade.

---

## 🎯 Objetivo dos Exemplos

Os exemplos devem:
1. ✅ **Demonstrar funcionalidades reais** - Não apenas código de brinquedo
2. ✅ **Cobrir diferentes casos de uso** - Classificação, regressão, diferentes domínios
3. ✅ **Mostrar progressão** - Do básico ao avançado
4. ✅ **Ser reproduzíveis** - Com datasets públicos ou sintéticos incluídos
5. ✅ **Documentar boas práticas** - Comentários explicativos

---

## 📁 Estrutura de Funcionalidades do CORE

### Componentes Principais

```
CORE Module
│
├── 1. DBDataset (Gerenciamento de Dados)
│   ├── Carregamento de dados
│   ├── Train/test splits
│   ├── Integração com modelos
│   ├── Gerenciamento de features
│   └── Predições e probabilidades
│
├── 2. Experiment (Orquestração)
│   ├── Tipos de experimento
│   ├── Execução de testes
│   ├── Comparação de modelos
│   └── Geração de relatórios
│
├── 3. Test Managers
│   ├── RobustnessManager
│   ├── UncertaintyManager
│   ├── ResilienceManager
│   └── HyperparameterManager
│
└── 4. Report System
    ├── Relatórios interativos
    ├── Relatórios estáticos
    └── Customização
```

---

## 📊 PARTE 1: DBDataset - Exemplos Propostos

### 1.1 Básico - Primeiros Passos

#### Exemplo 1.1.1: Carregamento Simples de Dados
**Arquivo**: `01_dbdataset_basic_loading.py`

**Funcionalidades demonstradas**:
- Criar DBDataset com split automático
- Acessar train/test data
- Visualizar features categóricas e numéricas

**Código conceitual**:
```python
"""
Exemplo básico: Carregar dados e criar DBDataset
Dataset: Iris (sklearn)
Objetivo: Mostrar o uso mais simples possível
"""
from sklearn.datasets import load_iris
import pandas as pd
from deepbridge import DBDataset

# Carregar dados
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Criar DBDataset (split automático)
dataset = DBDataset(
    data=df,
    target_column='target',
    test_size=0.2,
    random_state=42
)

# Explorar
print(f"Train: {len(dataset.train_data)} samples")
print(f"Test: {len(dataset.test_data)} samples")
print(f"Features: {dataset.features}")
print(f"Categorical: {dataset.categorical_features}")
print(f"Numerical: {dataset.numerical_features}")
```

**Importância**: 🔴 ALTA - Primeiro contato com a biblioteca

---

#### Exemplo 1.1.2: Carregamento com Train/Test Pré-separados
**Arquivo**: `01_dbdataset_presplit_data.py`

**Funcionalidades demonstradas**:
- Usar datasets já separados
- Validação de consistência
- Comparar com exemplo anterior

**Dataset**: Titanic (Kaggle format - train.csv, test.csv)

**Importância**: 🔴 ALTA - Caso de uso comum

---

### 1.2 Intermediário - Integração com Modelos

#### Exemplo 1.2.1: DBDataset com Modelo em Memória
**Arquivo**: `02_dbdataset_with_model.py`

**Funcionalidades demonstradas**:
- Treinar modelo (sklearn RandomForest)
- Integrar modelo treinado com DBDataset
- Acessar predições automáticas
- Visualizar probabilidades

**Dataset**: Credit Card Default

**Importância**: 🔴 ALTA - Workflow típico de ML

---

#### Exemplo 1.2.2: DBDataset com Modelo Salvo
**Arquivo**: `02_dbdataset_load_model.py`

**Funcionalidades demonstradas**:
- Salvar modelo em .pkl
- Carregar modelo via model_path
- Reproduzir resultados
- Validação de modelo de produção

**Dataset**: Credit Card Default (mesmo do anterior)

**Importância**: 🔴 ALTA - Validação de modelos em produção

---

#### Exemplo 1.2.3: DBDataset com Probabilidades Pré-computadas
**Arquivo**: `02_dbdataset_precomputed_probs.py`

**Funcionalidades demonstradas**:
- Usar prob_cols para economizar tempo
- Trabalhar com predições existentes
- Validar sem re-executar modelo pesado

**Dataset**: Large dataset simulation

**Importância**: 🟡 MÉDIA - Otimização para modelos pesados

---

### 1.3 Avançado - Features e Customização

#### Exemplo 1.3.1: Seleção e Engenharia de Features
**Arquivo**: `03_dbdataset_feature_selection.py`

**Funcionalidades demonstradas**:
- Especificar subset de features
- Comparar modelos com diferentes features
- Feature importance
- Categorical features customizadas

**Dataset**: Adult Income (UCI)

**Importância**: 🟡 MÉDIA - Feature engineering

---

#### Exemplo 1.3.2: Inferência Automática de Features Categóricas
**Arquivo**: `03_dbdataset_categorical_inference.py`

**Funcionalidades demonstradas**:
- Auto-detecção de categóricas
- Controlar max_categories
- Comparar auto vs manual
- Impacto em performance

**Dataset**: Mixed types dataset

**Importância**: 🟢 BAIXA - Funcionalidade auxiliar

---

## 📊 PARTE 2: Experiment - Exemplos Propostos

### 2.1 Básico - Workflow Completo

#### Exemplo 2.1.1: Primeiro Experimento - Classificação Binária
**Arquivo**: `04_experiment_binary_classification.py`

**Funcionalidades demonstradas**:
- Criar Experiment completo
- Executar run_tests() com config='quick'
- Visualizar métricas iniciais
- Salvar relatório HTML básico

**Dataset**: Credit Scoring

**Importância**: 🔴 ALTA - Demonstração principal da biblioteca

**Estrutura**:
```python
"""
Exemplo completo: Workflow de validação de modelo
Dataset: Credit Scoring
Objetivo: Mostrar todo o pipeline DeepBridge
"""
from deepbridge import DBDataset, Experiment
from sklearn.ensemble import RandomForestClassifier

# 1. Preparar dados
dataset = DBDataset(...)

# 2. Treinar modelo
clf = RandomForestClassifier()
clf.fit(...)
dataset.set_model(clf)

# 3. Criar experimento
exp = Experiment(
    dataset=dataset,
    experiment_type='binary_classification'
)

# 4. Executar testes
results = exp.run_tests(config_name='quick')

# 5. Salvar relatórios
exp.save_html('robustness', 'robustness_report.html')
exp.save_html('uncertainty', 'uncertainty_report.html')

# 6. Análise
print("Initial metrics:", exp.initial_results)
```

---

#### Exemplo 2.1.2: Experimento de Regressão
**Arquivo**: `04_experiment_regression.py`

**Funcionalidades demonstradas**:
- experiment_type='regression'
- Métricas específicas de regressão (R², RMSE, MAE)
- Comparar com classificação

**Dataset**: House Prices

**Importância**: 🔴 ALTA - Mostrar versatilidade

---

### 2.2 Intermediário - Testes Específicos

#### Exemplo 2.2.1: Análise de Robustez em Profundidade
**Arquivo**: `05_experiment_robustness_deep.py`

**Funcionalidades demonstradas**:
- run_test('robustness', config_name='full')
- Diferentes métodos de perturbação
- Análise de degradação de performance
- Identificar features sensíveis

**Dataset**: Medical Diagnosis

**Importância**: 🔴 ALTA - Teste crítico

**Conteúdo**:
```python
"""
Análise profunda de robustez
Dataset: Medical Diagnosis
Objetivo: Validar robustez para aplicação crítica
"""

# Executar teste completo de robustez
rob_results = exp.run_test(
    'robustness',
    config_name='full',
    perturbation_methods=['raw', 'quantile', 'adversarial'],
    n_iterations=100
)

# Análise detalhada
print("Robustness Score:", rob_results['robustness_score'])
print("Degradation:", rob_results['degradation'])
print("Most sensitive features:", rob_results['sensitive_features'])

# Relatório detalhado
exp.save_html('robustness', 'robustness_detailed.html', 'Medical Model')
```

---

#### Exemplo 2.2.2: Quantificação de Incerteza
**Arquivo**: `05_experiment_uncertainty.py`

**Funcionalidades demonstradas**:
- run_test('uncertainty')
- CRQR (Conformalized Quantile Regression)
- Intervalos de confiança
- Calibração de probabilidades

**Dataset**: Customer Churn

**Importância**: 🟡 MÉDIA - Importante para decisões críticas

---

#### Exemplo 2.2.3: Análise de Resiliência a Drift
**Arquivo**: `05_experiment_resilience.py`

**Funcionalidades demonstradas**:
- run_test('resilience')
- Tipos de drift (covariate, label, concept)
- Degradação temporal
- Recomendações de re-treino

**Dataset**: Time-series fraud detection

**Importância**: 🟡 MÉDIA - Importante para modelos em produção

---

#### Exemplo 2.2.4: Importância de Hiperparâmetros
**Arquivo**: `05_experiment_hyperparameter.py`

**Funcionalidades demonstradas**:
- run_test('hyperparameter')
- Optuna optimization
- Feature importance vs hyperparameter importance
- Sensibilidade

**Dataset**: Generic classification

**Importância**: 🟢 BAIXA - Mais para tunning

---

### 2.3 Avançado - Fairness e Comparação

#### Exemplo 2.3.1: Análise Completa de Fairness
**Arquivo**: `06_experiment_fairness_complete.py`

**Funcionalidades demonstradas**:
- Auto-detecção de atributos sensíveis
- run_fairness_tests(config='full')
- 15 métricas de fairness
- Conformidade EEOC
- Análise de threshold
- Mitigação de bias

**Dataset**: COMPAS Recidivism / Credit Lending

**Importância**: 🔴 ALTA - Crítico para aplicações reguladas

**Estrutura completa**:
```python
"""
Análise Completa de Fairness
Dataset: Credit Lending
Objetivo: Garantir compliance com regulações
"""

# 1. Detectar atributos sensíveis
sensitive_attrs = Experiment.detect_sensitive_attributes(
    dataset,
    threshold=0.7
)
print(f"Detected sensitive attributes: {sensitive_attrs}")

# 2. Criar experimento com fairness
exp = Experiment(
    dataset=dataset,
    experiment_type='binary_classification',
    protected_attributes=sensitive_attrs
)

# 3. Executar análise completa
fairness_result = exp.run_fairness_tests(config='full')

# 4. Verificar conformidade
if fairness_result.passes_eeoc_compliance():
    print("✓ Model passes EEOC 80% rule")
else:
    print("✗ Model FAILS EEOC compliance")
    print("Action required!")

# 5. Análise detalhada por grupo
for attr in sensitive_attrs:
    metrics = fairness_result.get_metrics_by_attribute(attr)
    print(f"\n{attr} Analysis:")
    print(f"  Demographic Parity: {metrics['demographic_parity']:.3f}")
    print(f"  Equal Opportunity: {metrics['equal_opportunity']:.3f}")
    print(f"  Calibration: {metrics['calibration']:.3f}")

# 6. Salvar relatório detalhado
fairness_result.save_html(
    'fairness_report.html',
    model_name='Credit Lending Model v2.1'
)
```

---

#### Exemplo 2.3.2: Comparação de Múltiplos Modelos
**Arquivo**: `06_experiment_model_comparison.py`

**Funcionalidades demonstradas**:
- Alternative models automáticos
- compare_all_models()
- Benchmark de performance
- Benchmark de robustez
- Trade-offs (accuracy vs fairness)

**Dataset**: Generic classification

**Importância**: 🔴 ALTA - Seleção de modelos

---

#### Exemplo 2.3.3: Análise Multi-Teste Integrada
**Arquivo**: `06_experiment_multiteste_integrated.py`

**Funcionalidades demonstradas**:
- Executar TODOS os testes (run_tests)
- Análise holística de modelo
- Dashboard de métricas
- Decisão de deployment

**Dataset**: High-stakes application (medical/financial)

**Importância**: 🔴 ALTA - Caso de uso real completo

---

## 📊 PARTE 3: Test Managers - Exemplos Propostos

### 3.1 Uso Direto de Managers (Avançado)

#### Exemplo 3.1.1: RobustnessManager Standalone
**Arquivo**: `07_manager_robustness_standalone.py`

**Funcionalidades demonstradas**:
- Usar RobustnessManager diretamente
- Customizar testes
- Comparar diferentes configurações
- Análise granular

**Importância**: 🟢 BAIXA - Para usuários avançados

---

#### Exemplo 3.1.2: Custom Manager Implementation
**Arquivo**: `07_manager_custom_implementation.py`

**Funcionalidades demonstradas**:
- Criar manager customizado
- Herdar de BaseManager
- Registrar novo tipo de teste
- Integrar com Experiment

**Importância**: 🟢 BAIXA - Extensibilidade

---

## 📊 PARTE 4: Report System - Exemplos Propostos

### 4.1 Customização de Relatórios

#### Exemplo 4.1.1: Relatórios Interativos vs Estáticos
**Arquivo**: `08_reports_interactive_vs_static.py`

**Funcionalidades demonstradas**:
- Gerar relatório interativo (Plotly)
- Gerar relatório estático (PNG)
- Comparar tamanhos de arquivo
- Casos de uso de cada tipo

**Importância**: 🟡 MÉDIA - Flexibilidade

---

#### Exemplo 4.1.2: Customização de Templates
**Arquivo**: `08_reports_custom_templates.py`

**Funcionalidades demonstradas**:
- Modificar templates Jinja2
- Adicionar seções customizadas
- Branding corporativo

**Importância**: 🟢 BAIXA - Personalização avançada

---

## 📊 PARTE 5: Casos de Uso Completos (End-to-End)

### 5.1 Casos de Uso por Domínio

#### Exemplo 5.1.1: Credit Scoring - Análise Regulatória Completa
**Arquivo**: `09_usecase_credit_scoring.py`

**Funcionalidades demonstradas**:
- Pipeline completo de validação
- Fairness obrigatório
- Robustez crítica
- Relatórios para auditoria
- Documentação de compliance

**Importância**: 🔴 ALTA - Caso real de negócio

**Estrutura**:
```python
"""
Caso de Uso Completo: Credit Scoring
Contexto: Modelo para aprovação de crédito
Requisitos:
  - Compliance com Fair Lending Laws
  - Robustez contra manipulação
  - Explicabilidade
  - Auditoria completa
"""

# FASE 1: Preparação de dados
dataset = DBDataset(...)

# FASE 2: Treinamento
model = train_credit_model(dataset)

# FASE 3: Validação Regulatória
exp = Experiment(dataset, 'binary_classification')

# 3.1 Fairness (OBRIGATÓRIO)
fairness_result = exp.run_fairness_tests(config='full')
assert fairness_result.passes_eeoc_compliance(), "FAIL: EEOC compliance"

# 3.2 Robustez (contra fraude)
rob_result = exp.run_test('robustness', config_name='full')
assert rob_result['robustness_score'] > 0.85, "FAIL: Low robustness"

# 3.3 Incerteza (decisões críticas)
unc_result = exp.run_test('uncertainty', config_name='full')

# FASE 4: Relatórios para auditoria
exp.save_html('fairness', 'audit/fairness_compliance.html')
exp.save_html('robustness', 'audit/robustness_analysis.html')
exp.save_html('uncertainty', 'audit/uncertainty_quantification.html')

# FASE 5: Aprovação para deploy
print("✓ All compliance checks passed")
print("✓ Model approved for production")
```

---

#### Exemplo 5.1.2: Medical Diagnosis - Validação de Alta Criticidade
**Arquivo**: `09_usecase_medical_diagnosis.py`

**Funcionalidades demonstradas**:
- Validação extremamente rigorosa
- Incerteza obrigatória
- Robustez crítica
- Análise de falsos negativos
- Relatórios médicos

**Importância**: 🔴 ALTA - Aplicação crítica

---

#### Exemplo 5.1.3: E-commerce - Recomendação e Churn
**Arquivo**: `09_usecase_ecommerce_churn.py`

**Funcionalidades demonstradas**:
- Resiliência a drift temporal
- Calibração de probabilidades
- A/B testing framework
- Monitoramento contínuo

**Importância**: 🟡 MÉDIA - Caso comercial

---

#### Exemplo 5.1.4: Fraud Detection - Tempo Real
**Arquivo**: `09_usecase_fraud_detection.py`

**Funcionalidades demonstradas**:
- Robustez contra adversários
- Latência de predição
- Drift adaptation
- Falsos positivos vs negativos

**Importância**: 🟡 MÉDIA - Sistema crítico

---

## 📊 PARTE 6: Exemplos Especiais

### 6.1 Performance e Otimização

#### Exemplo 6.1.1: Otimização para Grandes Datasets
**Arquivo**: `10_optimization_large_datasets.py`

**Funcionalidades demonstradas**:
- Usar prob_cols para economizar tempo
- Lazy loading de alternative models
- Sampling estratégico
- Métricas de tempo

**Importância**: 🟡 MÉDIA - Escalabilidade

---

#### Exemplo 6.1.2: Pipeline de Produção Completo
**Arquivo**: `10_production_pipeline.py`

**Funcionalidades demonstradas**:
- CI/CD integration
- Versionamento de modelos
- Validação automática
- Rollback criteria

**Importância**: 🟡 MÉDIA - DevOps/MLOps

---

### 6.2 Comparações e Benchmarks

#### Exemplo 6.2.1: DeepBridge vs Manual Validation
**Arquivo**: `11_comparison_manual_vs_deepbridge.py`

**Funcionalidades demonstradas**:
- Comparar tempo de desenvolvimento
- Comparar cobertura de testes
- Mostrar valor agregado
- ROI da biblioteca

**Importância**: 🟡 MÉDIA - Marketing/educação

---

## 📋 RESUMO DE PRIORIZAÇÃO

### 🔴 PRIORIDADE ALTA (Desenvolver Primeiro)

1. **01_dbdataset_basic_loading.py** - Primeiro contato
2. **01_dbdataset_presplit_data.py** - Caso comum
3. **02_dbdataset_with_model.py** - Workflow típico
4. **02_dbdataset_load_model.py** - Produção
5. **04_experiment_binary_classification.py** - Demo principal
6. **04_experiment_regression.py** - Versatilidade
7. **05_experiment_robustness_deep.py** - Teste crítico
8. **06_experiment_fairness_complete.py** - Compliance
9. **06_experiment_model_comparison.py** - Seleção
10. **06_experiment_multiteste_integrated.py** - Caso completo
11. **09_usecase_credit_scoring.py** - Caso real
12. **09_usecase_medical_diagnosis.py** - Aplicação crítica

**Total**: 12 exemplos essenciais

---

### 🟡 PRIORIDADE MÉDIA (Desenvolver Depois)

1. **02_dbdataset_precomputed_probs.py** - Otimização
2. **03_dbdataset_feature_selection.py** - Feature engineering
3. **05_experiment_uncertainty.py** - Decisões críticas
4. **05_experiment_resilience.py** - Produção
5. **08_reports_interactive_vs_static.py** - Flexibilidade
6. **09_usecase_ecommerce_churn.py** - Comercial
7. **09_usecase_fraud_detection.py** - Sistema crítico
8. **10_optimization_large_datasets.py** - Escalabilidade
9. **10_production_pipeline.py** - MLOps
10. **11_comparison_manual_vs_deepbridge.py** - Marketing

**Total**: 10 exemplos complementares

---

### 🟢 PRIORIDADE BAIXA (Desenvolver Por Último)

1. **03_dbdataset_categorical_inference.py** - Auxiliar
2. **05_experiment_hyperparameter.py** - Tunning
3. **07_manager_robustness_standalone.py** - Avançado
4. **07_manager_custom_implementation.py** - Extensibilidade
5. **08_reports_custom_templates.py** - Personalização

**Total**: 5 exemplos avançados

---

## 📊 Matriz de Cobertura

| Componente | Básico | Intermediário | Avançado | Total |
|------------|--------|---------------|----------|-------|
| **DBDataset** | 2 | 3 | 2 | 7 |
| **Experiment** | 2 | 4 | 3 | 9 |
| **Managers** | 0 | 0 | 2 | 2 |
| **Reports** | 0 | 1 | 1 | 2 |
| **Use Cases** | 0 | 2 | 2 | 4 |
| **Special** | 0 | 3 | 0 | 3 |
| **TOTAL** | **4** | **13** | **10** | **27** |

---

## 🎯 Datasets Necessários

### Datasets Públicos
1. **Iris** - sklearn (básico)
2. **Titanic** - Kaggle (train/test split)
3. **Credit Card Default** - UCI
4. **Adult Income** - UCI (fairness)
5. **COMPAS** - ProPublica (fairness)
6. **House Prices** - Kaggle (regressão)

### Datasets Sintéticos (Criar)
1. **Credit Scoring Synthetic** - Para uso completo
2. **Medical Diagnosis Synthetic** - Aplicação crítica
3. **Large Dataset** - Performance testing

---

## 📝 Template de Exemplo

Cada exemplo deve seguir este template:

```python
"""
TÍTULO DO EXEMPLO
================

Dataset: [Nome do dataset]
Tipo de Problema: [Classificação/Regressão/etc]
Nível: [Básico/Intermediário/Avançado]

Objetivo:
    [Descrever o que este exemplo demonstra]

Funcionalidades Demonstradas:
    - Funcionalidade 1
    - Funcionalidade 2
    - ...

Pré-requisitos:
    - Conhecimento de ...
    - Bibliotecas: ...

Tempo de Execução Estimado: [X minutos]

Autor: DeepBridge Team
Data: [Data]
"""

# ============================================================================
# IMPORTS
# ============================================================================
import pandas as pd
import numpy as np
from deepbridge import DBDataset, Experiment
from sklearn.ensemble import RandomForestClassifier

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ============================================================================
# PARTE 1: CARREGAMENTO DE DADOS
# ============================================================================
print("="*80)
print("PARTE 1: Carregamento de Dados")
print("="*80)

# ... código comentado ...

# ============================================================================
# PARTE 2: PREPARAÇÃO
# ============================================================================
print("\n" + "="*80)
print("PARTE 2: Preparação")
print("="*80)

# ... código comentado ...

# ============================================================================
# PARTE 3: EXECUÇÃO
# ============================================================================
print("\n" + "="*80)
print("PARTE 3: Execução")
print("="*80)

# ... código comentado ...

# ============================================================================
# PARTE 4: ANÁLISE DE RESULTADOS
# ============================================================================
print("\n" + "="*80)
print("PARTE 4: Análise de Resultados")
print("="*80)

# ... código comentado ...

# ============================================================================
# CONCLUSÃO
# ============================================================================
print("\n" + "="*80)
print("CONCLUSÃO")
print("="*80)
print("""
Este exemplo demonstrou:
1. [Ponto 1]
2. [Ponto 2]
...

Próximos passos sugeridos:
- Explorar exemplo X
- Modificar parâmetros Y
""")
```

---

## 🚀 Roadmap de Implementação

### Fase 1: Fundação (Semana 1-2)
**Objetivo**: Ter exemplos básicos funcionando

- [ ] 01_dbdataset_basic_loading.py
- [ ] 01_dbdataset_presplit_data.py
- [ ] 02_dbdataset_with_model.py
- [ ] 04_experiment_binary_classification.py

**Entrega**: 4 exemplos básicos funcionais

---

### Fase 2: Expansão Core (Semana 3-4)
**Objetivo**: Cobrir funcionalidades principais

- [ ] 02_dbdataset_load_model.py
- [ ] 04_experiment_regression.py
- [ ] 05_experiment_robustness_deep.py
- [ ] 05_experiment_uncertainty.py

**Entrega**: 8 exemplos cobrindo DBDataset e Experiment básico

---

### Fase 3: Funcionalidades Avançadas (Semana 5-6)
**Objetivo**: Fairness e análises avançadas

- [ ] 06_experiment_fairness_complete.py
- [ ] 06_experiment_model_comparison.py
- [ ] 06_experiment_multiteste_integrated.py
- [ ] 05_experiment_resilience.py

**Entrega**: 12 exemplos com funcionalidades avançadas

---

### Fase 4: Casos de Uso Reais (Semana 7-8)
**Objetivo**: Demonstrar aplicações práticas

- [ ] 09_usecase_credit_scoring.py
- [ ] 09_usecase_medical_diagnosis.py
- [ ] 09_usecase_ecommerce_churn.py
- [ ] 09_usecase_fraud_detection.py

**Entrega**: 16 exemplos incluindo casos reais

---

### Fase 5: Otimização e Complementos (Semana 9-10)
**Objetivo**: Completar cobertura

- [ ] Todos os exemplos de prioridade média
- [ ] Documentação adicional
- [ ] README para cada exemplo
- [ ] Notebooks Jupyter (opcionais)

**Entrega**: 27 exemplos completos

---

## 📚 Documentação Complementar

Cada exemplo deve ter:

1. **README.md** no diretório
2. **requirements.txt** específico
3. **Dados incluídos** ou script de download
4. **Output esperado** (screenshots de relatórios)
5. **Troubleshooting** seção

---

## 🎓 Guias Adicionais Sugeridos

Além dos exemplos de código, criar:

1. **Tutorial em Vídeo** - Para exemplo principal
2. **Jupyter Notebooks** - Versões interativas
3. **Cheat Sheet** - Referência rápida
4. **FAQ** - Perguntas comuns
5. **Best Practices Guide** - Recomendações

---

## ✅ Critérios de Qualidade

Cada exemplo deve:

- [ ] **Executar sem erros** em ambiente limpo
- [ ] **Ter comentários explicativos** em português
- [ ] **Gerar saída visual** (prints, relatórios)
- [ ] **Tempo de execução** < 5 minutos (exceto 'full')
- [ ] **Dataset incluído** ou facilmente obtível
- [ ] **Seguir template** padrão
- [ ] **Testar funcionalidades** sem quebrar
- [ ] **Documentar edge cases** conhecidos

---

## 📞 Próximos Passos Imediatos

1. **Validar este planejamento** com equipe
2. **Selecionar datasets** e preparar
3. **Implementar Fase 1** (4 exemplos básicos)
4. **Testar em ambiente limpo**
5. **Iterar baseado em feedback**

---

**Última Atualização**: 04 de Novembro de 2025
**Mantido por**: Equipe DeepBridge
**Versão**: 1.0
**Status**: 📋 PLANEJAMENTO
