# 📓 ÍNDICE DE NOTEBOOKS - DeepBridge

**Referência Rápida de Todos os 27 Notebooks**

---

## 📁 1. INTRODUÇÃO (3 notebooks)

| # | Notebook | Tempo | Prioridade | Descrição |
|---|----------|-------|------------|-----------|
| 1.1 | `01_primeiros_passos.ipynb` | 15 min | 🔴 | Instalação, Iris, primeiro DBDataset |
| 1.2 | `02_conceitos_basicos.ipynb` | 20 min | 🔴 | Arquitetura, conceitos fundamentais |
| 1.3 | `03_workflow_completo.ipynb` ⭐ | 30 min | 🔴 | **DEMO PRINCIPAL** - End-to-end completo |

**Objetivo**: Onboarding de novos usuários
**Tempo Total**: ~65 minutos

---

## 📁 2. DBDATASET (7 notebooks)

| # | Notebook | Tempo | Prioridade | Descrição |
|---|----------|-------|------------|-----------|
| 2.1 | `01_carregamento_simples.ipynb` | 10 min | 🔴 | Split automático básico |
| 2.2 | `02_dados_pre_separados.ipynb` | 10 min | 🔴 | Train/test separados (Kaggle style) |
| 2.3 | `03_integracao_modelos.ipynb` | 15 min | 🔴 | Modelo em memória, predições auto |
| 2.4 | `04_modelos_salvos.ipynb` | 15 min | 🔴 | Carregar .pkl, produção |
| 2.5 | `05_probabilidades_precomputadas.ipynb` | 15 min | 🟡 | prob_cols, otimização |
| 2.6 | `06_selecao_features.ipynb` | 20 min | 🟡 | Subset features, engineering |
| 2.7 | `07_features_categoricas.ipynb` | 15 min | 🟢 | Auto-detecção categóricas |

**Objetivo**: Dominar gerenciamento de dados
**Tempo Total**: ~100 minutos

---

## 📁 3. TESTES DE VALIDAÇÃO (6 notebooks)

| # | Notebook | Tempo | Prioridade | Descrição |
|---|----------|-------|------------|-----------|
| 3.1 | `01_introducao_testes.ipynb` | 15 min | 🔴 | Visão geral, run_tests() |
| 3.2 | `02_robustez_completa.ipynb` | 25 min | 🔴 | Perturbações, degradação |
| 3.3 | `03_incerteza.ipynb` | 20 min | 🟡 | CRQR, intervalos confiança |
| 3.4 | `04_resiliencia_drift.ipynb` | 20 min | 🟡 | 4 tipos de drift |
| 3.5 | `05_hiperparametros.ipynb` | 20 min | 🟢 | Optuna, importância HPM |
| 3.6 | `06_comparacao_modelos.ipynb` | 25 min | 🔴 | Benchmark automático |

**Objetivo**: Dominar testes de validação
**Tempo Total**: ~125 minutos

---

## 📁 4. FAIRNESS (3 notebooks)

| # | Notebook | Tempo | Prioridade | Descrição |
|---|----------|-------|------------|-----------|
| 4.1 | `01_introducao_fairness.ipynb` | 20 min | 🔴 | Conceitos, auto-detecção |
| 4.2 | `02_analise_completa_fairness.ipynb` ⭐⭐ | 35 min | 🔴 | **15 métricas, EEOC, compliance** |
| 4.3 | `03_mitigacao_bias.ipynb` | 25 min | 🟡 | Técnicas mitigação |

**Objetivo**: Garantir fairness e compliance
**Tempo Total**: ~80 minutos

---

## 📁 5. CASOS DE USO (5 notebooks)

| # | Notebook | Tempo | Prioridade | Descrição |
|---|----------|-------|------------|-----------|
| 5.1 | `01_credit_scoring.ipynb` ⭐⭐⭐ | 60 min | 🔴 | **CASO REAL** - Compliance completo |
| 5.2 | `02_diagnostico_medico.ipynb` | 40 min | 🔴 | Aplicação crítica, incerteza |
| 5.3 | `03_churn_prediction.ipynb` | 30 min | 🟡 | E-commerce, drift temporal |
| 5.4 | `04_fraud_detection.ipynb` | 30 min | 🟡 | Robustez adversarial |
| 5.5 | `05_regressao_precos.ipynb` | 25 min | 🟡 | Regressão, House Prices |

**Objetivo**: Aplicações reais completas
**Tempo Total**: ~185 minutos

---

## 📁 6. AVANÇADO (3 notebooks)

| # | Notebook | Tempo | Prioridade | Descrição |
|---|----------|-------|------------|-----------|
| 6.1 | `01_otimizacao_performance.ipynb` | 25 min | 🟡 | Grandes datasets, otimização |
| 6.2 | `02_customizacao_relatorios.ipynb` | 20 min | 🟢 | Templates, branding |
| 6.3 | `03_extensibilidade.ipynb` | 30 min | 🟢 | Criar managers customizados |

**Objetivo**: Usuários avançados
**Tempo Total**: ~75 minutos

---

## 📊 Resumo Geral

### Por Prioridade

| Prioridade | Notebooks | Tempo Total |
|------------|-----------|-------------|
| 🔴 **ALTA** | 12 | ~400 min (~6.5h) |
| 🟡 **MÉDIA** | 10 | ~220 min (~3.5h) |
| 🟢 **BAIXA** | 5 | ~95 min (~1.5h) |
| **TOTAL** | **27** | **~715 min (~12h)** |

### Por Pasta

| Pasta | Notebooks | Tempo |
|-------|-----------|-------|
| 01_introducao | 3 | 65 min |
| 02_dbdataset | 7 | 100 min |
| 03_testes_validacao | 6 | 125 min |
| 04_fairness | 3 | 80 min |
| 05_casos_uso | 5 | 185 min |
| 06_avancado | 3 | 75 min |

---

## 🎯 Top 5 Notebooks Essenciais

### 1. 🥇 `01_introducao/03_workflow_completo.ipynb`
**Por quê**: Demo principal - mostra todo o poder da biblioteca
**Tempo**: 30 min
**Prioridade**: 🔴 MÁXIMA

### 2. 🥈 `04_fairness/02_analise_completa_fairness.ipynb`
**Por quê**: Diferencial competitivo - 15 métricas + compliance
**Tempo**: 35 min
**Prioridade**: 🔴 CRÍTICA

### 3. 🥉 `05_casos_uso/01_credit_scoring.ipynb`
**Por quê**: Caso real end-to-end completo
**Tempo**: 60 min
**Prioridade**: 🔴 ESSENCIAL

### 4. `03_testes_validacao/02_robustez_completa.ipynb`
**Por quê**: Robustez é crítica para produção
**Tempo**: 25 min
**Prioridade**: 🔴 ALTA

### 5. `03_testes_validacao/06_comparacao_modelos.ipynb`
**Por quê**: Benchmark automático economiza tempo
**Tempo**: 25 min
**Prioridade**: 🔴 ALTA

---

## 🎓 Trilhas de Aprendizado

### 👤 Trilha 1: Iniciante Completo
**Objetivo**: Do zero ao uso básico
**Tempo**: ~2 horas

1. `01_introducao/01_primeiros_passos.ipynb` (15 min)
2. `01_introducao/02_conceitos_basicos.ipynb` (20 min)
3. `01_introducao/03_workflow_completo.ipynb` ⭐ (30 min)
4. `02_dbdataset/01_carregamento_simples.ipynb` (10 min)
5. `02_dbdataset/03_integracao_modelos.ipynb` (15 min)
6. `03_testes_validacao/01_introducao_testes.ipynb` (15 min)

**Resultado**: Consegue usar DeepBridge para validar modelos

---

### 👤 Trilha 2: ML Engineer (Produção)
**Objetivo**: Validar modelos para deploy
**Tempo**: ~3 horas

1. `01_introducao/03_workflow_completo.ipynb` ⭐ (30 min)
2. `02_dbdataset/04_modelos_salvos.ipynb` (15 min)
3. `03_testes_validacao/02_robustez_completa.ipynb` (25 min)
4. `04_fairness/02_analise_completa_fairness.ipynb` ⭐⭐ (35 min)
5. `03_testes_validacao/04_resiliencia_drift.ipynb` (20 min)
6. `03_testes_validacao/06_comparacao_modelos.ipynb` (25 min)
7. `05_casos_uso/01_credit_scoring.ipynb` ⭐⭐⭐ (60 min)

**Resultado**: Deploy com confiança e compliance

---

### 👤 Trilha 3: Compliance Officer
**Objetivo**: Garantir fairness e regulação
**Tempo**: ~2 horas

1. `01_introducao/03_workflow_completo.ipynb` ⭐ (30 min)
2. `04_fairness/01_introducao_fairness.ipynb` (20 min)
3. `04_fairness/02_analise_completa_fairness.ipynb` ⭐⭐ (35 min)
4. `05_casos_uso/01_credit_scoring.ipynb` ⭐⭐⭐ (60 min)

**Resultado**: Validar compliance completo

---

### 👤 Trilha 4: Pesquisador/Avançado
**Objetivo**: Estender e customizar
**Tempo**: ~2.5 horas

1. `01_introducao/03_workflow_completo.ipynb` ⭐ (30 min)
2. `03_testes_validacao/*` - Todos (125 min)
3. `06_avancado/03_extensibilidade.ipynb` (30 min)

**Resultado**: Criar componentes customizados

---

## 📅 Cronograma de Desenvolvimento

### Semana 1-2: Fundação ✅
**Meta**: 6 notebooks críticos

- [ ] `01_introducao/01_primeiros_passos.ipynb`
- [ ] `01_introducao/02_conceitos_basicos.ipynb`
- [ ] `01_introducao/03_workflow_completo.ipynb` ⭐
- [ ] `02_dbdataset/01_carregamento_simples.ipynb`
- [ ] `02_dbdataset/02_dados_pre_separados.ipynb`
- [ ] `02_dbdataset/03_integracao_modelos.ipynb`

---

### Semana 3-4: Testes e Fairness
**Meta**: +6 notebooks (total: 12)

- [ ] `02_dbdataset/04_modelos_salvos.ipynb`
- [ ] `03_testes_validacao/01_introducao_testes.ipynb`
- [ ] `03_testes_validacao/02_robustez_completa.ipynb`
- [ ] `03_testes_validacao/06_comparacao_modelos.ipynb`
- [ ] `04_fairness/01_introducao_fairness.ipynb`
- [ ] `04_fairness/02_analise_completa_fairness.ipynb` ⭐⭐

---

### Semana 5-6: Casos de Uso
**Meta**: +3 notebooks (total: 15)

- [ ] `05_casos_uso/01_credit_scoring.ipynb` ⭐⭐⭐
- [ ] `05_casos_uso/02_diagnostico_medico.ipynb`
- [ ] `05_casos_uso/05_regressao_precos.ipynb`

---

### Semana 7-8: Testes Adicionais
**Meta**: +4 notebooks (total: 19)

- [ ] `03_testes_validacao/03_incerteza.ipynb`
- [ ] `03_testes_validacao/04_resiliencia_drift.ipynb`
- [ ] `05_casos_uso/03_churn_prediction.ipynb`
- [ ] `05_casos_uso/04_fraud_detection.ipynb`

---

### Semana 9-10: Completar
**Meta**: +8 notebooks (total: 27) ✅

- [ ] `02_dbdataset/05_probabilidades_precomputadas.ipynb`
- [ ] `02_dbdataset/06_selecao_features.ipynb`
- [ ] `02_dbdataset/07_features_categoricas.ipynb`
- [ ] `03_testes_validacao/05_hiperparametros.ipynb`
- [ ] `04_fairness/03_mitigacao_bias.ipynb`
- [ ] `06_avancado/01_otimizacao_performance.ipynb`
- [ ] `06_avancado/02_customizacao_relatorios.ipynb`
- [ ] `06_avancado/03_extensibilidade.ipynb`

---

## 📦 Datasets Necessários

| Dataset | Usado em | Status | Prioridade |
|---------|----------|--------|------------|
| **Iris** | Introdução, DBDataset | ✅ Disponível | 🔴 |
| **Titanic** | Conceitos Básicos | ⬜ Download | 🔴 |
| **Credit Scoring Synthetic** | Workflow, Credit Scoring | ⬜ Criar | 🔴 |
| **Credit Card Default** | DBDataset | ⬜ Download | 🔴 |
| **Medical Diagnosis Synthetic** | Diagnóstico Médico | ⬜ Criar | 🔴 |
| **Adult Income** | Fairness | ⬜ Download | 🔴 |
| **House Prices** | Regressão | ⬜ Download | 🟡 |
| **COMPAS** | Fairness Avançado | ⬜ Download | 🟡 |
| **Churn Dataset** | Churn Prediction | ⬜ Obter | 🟡 |
| **Fraud Dataset** | Fraud Detection | ⬜ Obter | 🟡 |

---

## 🎨 Convenções de Nomenclatura

### Arquivos
- `XX_nome_descritivo.ipynb` (XX = número sequencial)
- Nomes em minúsculas
- Underscores para separar palavras
- Em português

### Pastas
- `XX_nome_pasta/` (XX = número)
- Nomes descritivos
- Em minúsculas

### Dentro do Notebook
- **Títulos**: Emoji + Título em PT-BR
- **Seções**: Numeradas (1, 2, 3...)
- **Anchors**: `<a id="secao"></a>`

---

## ✅ Checklist Rápido

Para cada notebook:
- [ ] Segue template padrão
- [ ] Metadados completos (nível, tempo, pré-requisitos)
- [ ] Objetivos claros
- [ ] Índice de navegação
- [ ] Todas as células executam
- [ ] Outputs salvos
- [ ] Alerts visuais (info, warning, success)
- [ ] Conclusão e próximos passos
- [ ] Tempo < 10 min (exceto casos de uso)
- [ ] README.md na pasta
- [ ] requirements.txt na pasta

---

## 🎯 KPIs de Sucesso

### Engajamento
- Taxa de conclusão por notebook > 70%
- Tempo médio = tempo estimado ± 20%
- Feedback positivo > 85%

### Aprendizado
- 90% dos usuários completam trilha iniciante
- 70% aplicam em projeto próprio
- 50% completam caso de uso

### Técnico
- 0 erros de execução
- Compatível com Jupyter Lab, VS Code, Colab
- Tempo de carregamento < 5s

---

## 📞 Próxima Ação

**Começar por:**
1. ✅ Validar este índice
2. Preparar datasets
3. Criar `03_workflow_completo.ipynb` ⭐ (DEMO PRINCIPAL)
4. Testar e iterar
5. Implementar Fase 1 completa

---

**Última Atualização**: 04 de Novembro de 2025
**Versão**: 1.0
**Total de Notebooks**: 27
**Tempo Total de Conteúdo**: ~12 horas
