# 🧪 Testes de Validação

Aprenda a executar e interpretar cada tipo de teste de validação do DeepBridge.

---

## 📓 Notebooks desta Pasta

| # | Notebook | Tempo | Descrição | Prioridade |
|---|----------|-------|-----------|------------|
| 1 | `01_introducao_testes.ipynb` | 15 min | Visão geral dos 5 tipos de testes | 🔴 ALTA |
| 2 | `02_robustez_completa.ipynb` | 25 min | Análise profunda de robustez | 🔴 ALTA |
| 3 | `03_incerteza.ipynb` | 20 min | Quantificação de incerteza (CRQR) | 🟡 MÉDIA |

**Tempo Total**: ~60 minutos

---

## 🎯 Ordem Recomendada

### Para Iniciantes
1. **Comece aqui:** `01_introducao_testes.ipynb`
   - Entender os 5 tipos de testes
   - Executar todos os testes juntos
   - Comparar configurações (quick/medium/full)

2. **Continue com:** `02_robustez_completa.ipynb`
   - Deep dive em robustez
   - Métodos de perturbação
   - Features sensíveis
   - Relatórios HTML

3. **Aprofunde:** `03_incerteza.ipynb`
   - Quantificação de incerteza
   - Intervalos de confiança
   - Coverage analysis
   - Decisões baseadas em incerteza

---

## 📖 O que Você Vai Aprender

### Notebook 1: Introdução aos Testes
- ✅ Os 5 tipos de testes disponíveis
- ✅ Executar todos os testes simultaneamente
- ✅ Configurações: quick vs medium vs full
- ✅ Interpretar resultados agregados
- ✅ Quando usar cada tipo de teste

### Notebook 2: Robustez Completa
- ✅ Conceito de robustez em ML
- ✅ Métodos de perturbação (Gaussian, Dropout, Scaling, Adversarial)
- ✅ Interpretar Robustness Score (0 a 1)
- ✅ Identificar features sensíveis
- ✅ Visualizar degradação de performance
- ✅ Gerar relatórios HTML
- ✅ Técnicas para melhorar robustez

### Notebook 3: Incerteza
- ✅ Por que incerteza importa (medicina, finanças, segurança)
- ✅ CRQR (Conformalized Quantile Regression)
- ✅ Gerar intervalos de confiança
- ✅ Coverage analysis (calibração)
- ✅ Tomar decisões baseadas em incerteza
- ✅ Aplicações práticas

---

## 🎓 Pré-requisitos

- Completar `01_introducao/` (recomendado)
- Conhecimento de métricas de ML (accuracy, precision, recall)
- Familiaridade com validação de modelos

### Instalação
```bash
pip install deepbridge jupyter pandas numpy matplotlib seaborn scikit-learn
```

---

## 🚀 Como Executar

```bash
# 1. Navegar até a pasta
cd /home/guhaase/projetos/DeepBridge/examples/notebooks/03_testes_validacao

# 2. Iniciar Jupyter
jupyter notebook

# 3. Abrir o primeiro notebook
# 01_introducao_testes.ipynb
```

---

## 💡 Principais Conceitos

### Os 5 Tipos de Testes

```python
from deepbridge import Experiment

# Executar TODOS os testes
results = exp.run_tests(config_name='quick')

# Ou executar individualmente
robustness = exp.run_test('robustness', config_name='quick')
uncertainty = exp.run_test('uncertainty', config_name='quick')
resilience = exp.run_test('resilience', config_name='quick')
hyperparameter = exp.run_test('hyperparameter', config_name='quick')
# Fairness tem método próprio
fairness = exp.run_fairness_tests(config='quick')
```

### Configurações

| Config | Tempo | Uso | Cobertura |
|--------|-------|-----|-----------|
| `quick` | Segundos-minutos | Desenvolvimento rápido | Básica |
| `medium` | Minutos | Validação intermediária | Moderada |
| `full` | Pode demorar | Validação final pré-produção | Completa |

---

## 🎯 Decisão: Qual Teste Usar?

| Seu Objetivo | Teste Recomendado |
|--------------|-------------------|
| Modelo resistente a ruído | `01_introducao` + `02_robustez` |
| Quantificar confiança das predições | `03_incerteza` |
| Detectar mudanças nos dados | `04_resiliencia_drift` (futuro) |
| Otimizar hiperparâmetros | `05_hiperparametros` (futuro) |
| Comparar múltiplos modelos | `06_comparacao_modelos` (futuro) |
| Garantir fairness | `../04_fairness/` |

---

## 📊 Status de Implementação

- ✅ **Fase 2 Completa** (3/6 notebooks) - Disponível agora!
  - ✅ 01_introducao_testes.ipynb
  - ✅ 02_robustez_completa.ipynb
  - ✅ 03_incerteza.ipynb
- 🔄 **Fase 4-5** (3 notebooks restantes) - Planejado
  - 04_resiliencia_drift.ipynb
  - 05_hiperparametros.ipynb
  - 06_comparacao_modelos.ipynb

---

## 🎯 Próximos Passos

Depois de dominar os testes de validação, continue para:

📁 **04_fairness/** - Análise de fairness e compliance
- 15 métricas de fairness
- EEOC compliance
- Mitigação de bias

📁 **05_casos_uso/** - Aplicações reais end-to-end
- Credit Scoring completo
- Diagnóstico médico
- Churn prediction
- Fraud detection

---

## 💡 Dicas Importantes

### 1. Sempre Teste Antes de Produção
```python
# NUNCA faça isso:
model.fit(X_train, y_train)
# Deploy direto ❌

# SEMPRE faça isso:
model.fit(X_train, y_train)
exp = Experiment(dataset, ...)
results = exp.run_tests(config_name='full')  # ✅
# Analisar resultados
# Depois deploy
```

### 2. Use config='full' Antes de Deploy
- `quick`: Desenvolvimento
- `medium`: Iteração
- `full`: **Validação final obrigatória**

### 3. Salve Relatórios HTML
```python
# Documentação para auditoria
exp.save_report('robustness', 'robustness_report.html')
exp.save_report('uncertainty', 'uncertainty_report.html')
```

---

## 📞 Precisa de Ajuda?

- 📖 [Documentação Completa](../../planejamento_doc/1-CORE/)
- 💻 [Código Fonte](https://github.com/DeepBridge-Validation/DeepBridge)
- ❓ [Issues](https://github.com/DeepBridge-Validation/DeepBridge/issues)

---

## 🔍 Comparação com Outras Bibliotecas

| Funcionalidade | DeepBridge | Scikit-learn | TensorFlow |
|----------------|------------|--------------|------------|
| Robustez automática | ✅ | ❌ | ❌ |
| Quantificação de incerteza | ✅ | Parcial | Parcial |
| Detecção de drift | ✅ | ❌ | ❌ |
| Análise de HPM | ✅ | ❌ | ❌ |
| Fairness (15 métricas) | ✅ | ❌ | ❌ |
| Relatórios HTML | ✅ | ❌ | ❌ |
| **Todos em um lugar** | ✅ | ❌ | ❌ |

---

**Última Atualização**: 04 de Novembro de 2025
**Status**: ✅ Fase 2 Completa (3/6 notebooks)
**Próxima Implementação**: Fase 4-5 (notebooks 4-6)
