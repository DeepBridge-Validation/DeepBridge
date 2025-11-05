# 📚 Exemplos DeepBridge - Módulo CORE

Bem-vindo aos exemplos oficiais da biblioteca **DeepBridge**!

Este diretório contém exemplos práticos que demonstram todas as funcionalidades do módulo CORE.

---

## 🚀 Início Rápido

### Novo no DeepBridge? Comece aqui:

1. **[01_dbdataset/basic/01_basic_loading.py](./01_dbdataset/basic/01_basic_loading.py)**
   - Primeiro contato com a biblioteca
   - Carregamento básico de dados
   - ⏱️ 2 minutos

2. **[02_experiment/basic/01_binary_classification.py](./02_experiment/basic/01_binary_classification.py)** ⭐
   - **DEMO PRINCIPAL** da biblioteca
   - Workflow completo de validação
   - ⏱️ 5 minutos

3. **[02_experiment/advanced/01_fairness_complete.py](./02_experiment/advanced/01_fairness_complete.py)** ⭐⭐
   - Análise completa de fairness
   - 15 métricas + compliance EEOC
   - ⏱️ 8 minutos

---

## 📂 Estrutura de Exemplos

### 📁 01_dbdataset/ - Gerenciamento de Dados
Aprenda a usar o `DBDataset`, o container fundamental de dados.

**Básico** (comece aqui):
- `01_basic_loading.py` - Carregamento simples 🔴
- `02_presplit_data.py` - Train/test pré-separados 🔴

**Intermediário**:
- `01_with_model.py` - Modelo em memória 🔴
- `02_load_model.py` - Carregar modelo salvo 🔴
- `03_precomputed_probs.py` - Otimização 🟡

**Avançado**:
- `01_feature_selection.py` - Seleção de features 🟡
- `02_categorical_inference.py` - Auto-detecção 🟢

---

### 📁 02_experiment/ - Orquestração de Testes
Aprenda a usar o `Experiment` para validar modelos.

**Básico** (essencial):
- `01_binary_classification.py` ⭐ - **DEMO PRINCIPAL** 🔴
- `02_regression.py` - Problemas de regressão 🔴

**Intermediário** (testes específicos):
- `01_robustness_deep.py` - Robustez em profundidade 🔴
- `02_uncertainty.py` - Quantificação de incerteza 🟡
- `03_resilience.py` - Detecção de drift 🟡
- `04_hyperparameter.py` - Importância de HPM 🟢

**Avançado** (análises completas):
- `01_fairness_complete.py` ⭐⭐ - **FAIRNESS COMPLETO** 🔴
- `02_model_comparison.py` - Benchmark de modelos 🔴
- `03_multiteste_integrated.py` - Todos os testes 🔴

---

### 📁 03_managers/ - Uso Avançado de Managers
Para usuários avançados que querem controle fino.

**Avançado**:
- `01_robustness_standalone.py` - Usar manager diretamente 🟢
- `02_custom_implementation.py` - Criar manager customizado 🟢

---

### 📁 04_reports/ - Sistema de Relatórios
Personalize e otimize relatórios HTML.

**Intermediário**:
- `01_interactive_vs_static.py` - Comparar tipos 🟡

**Avançado**:
- `01_custom_templates.py` - Personalizar templates 🟢

---

### 📁 05_use_cases/ - Casos de Uso Completos
Aplicações reais end-to-end.

**⭐⭐⭐ Casos de Uso Principais**:

1. **credit_scoring/** 🔴
   - Análise regulatória completa
   - Fairness + Robustez + Compliance
   - Exemplo de produção

2. **medical_diagnosis/** 🔴
   - Aplicação crítica
   - Incerteza + Robustez máximas
   - Validação rigorosa

3. **ecommerce_churn/** 🟡
   - Detecção de churn
   - Drift temporal
   - Calibração de probabilidades

4. **fraud_detection/** 🟡
   - Detecção de fraude
   - Robustez adversarial
   - Tempo real

---

### 📁 06_special/ - Exemplos Especiais
Tópicos específicos e otimizações.

**Otimização**:
- `large_datasets.py` - Escalabilidade 🟡

**Produção**:
- `production_pipeline.py` - CI/CD + MLOps 🟡

**Comparação**:
- `manual_vs_deepbridge.py` - ROI da biblioteca 🟡

---

## 🎯 Recomendações por Perfil

### 👤 Cientista de Dados (Iniciante)
**Objetivo**: Aprender a usar DeepBridge

1. Start: `01_dbdataset/basic/01_basic_loading.py`
2. Core: `02_experiment/basic/01_binary_classification.py` ⭐
3. Practice: `02_experiment/basic/02_regression.py`
4. Next: Explorar testes específicos (robustness, uncertainty)

---

### 👤 ML Engineer (Produção)
**Objetivo**: Validar modelos para deploy

1. Start: `01_dbdataset/intermediate/02_load_model.py`
2. Core: `02_experiment/intermediate/01_robustness_deep.py`
3. Critical: `02_experiment/advanced/01_fairness_complete.py` ⭐⭐
4. Integration: `06_special/production/production_pipeline.py`

---

### 👤 Compliance/Risk Officer
**Objetivo**: Garantir conformidade regulatória

1. **Must Read**: `05_use_cases/credit_scoring/` ⭐⭐⭐
2. **Fairness Deep Dive**: `02_experiment/advanced/01_fairness_complete.py`
3. **Robustness**: `02_experiment/intermediate/01_robustness_deep.py`
4. **Documentation**: Relatórios HTML gerados

---

### 👤 Pesquisador/Desenvolvedor Avançado
**Objetivo**: Estender funcionalidades

1. Architecture: Ler documentação em `planejamento_doc/1-CORE/`
2. Managers: `03_managers/advanced/01_robustness_standalone.py`
3. Custom: `03_managers/advanced/02_custom_implementation.py`
4. Reports: `04_reports/advanced/01_custom_templates.py`

---

## 📊 Legenda de Prioridade

- 🔴 **ALTA** - Exemplos essenciais, começar por aqui
- 🟡 **MÉDIA** - Funcionalidades importantes, explorar depois
- 🟢 **BAIXA** - Funcionalidades avançadas/opcionais

---

## 🎓 Níveis de Complexidade

- **Básico** ⚪ - Primeiros passos, conceitos fundamentais
- **Intermediário** 🔵 - Uso prático, casos comuns
- **Avançado** 🟣 - Customização, extensibilidade

---

## 📦 Datasets Utilizados

### Incluídos na Biblioteca
- **Iris** (sklearn) - Classificação básica
- **Datasets Sintéticos** - Criados para exemplos específicos

### Downloadables (scripts fornecidos)
- **Titanic** (Kaggle)
- **Adult Income** (UCI)
- **Credit Card Default** (UCI)
- **House Prices** (Kaggle)

### Datasets Customizados
- **Credit Scoring Synthetic** - Para caso de uso completo
- **Medical Diagnosis Synthetic** - Para aplicação crítica

---

## 🛠️ Instalação e Setup

### 1. Instalar DeepBridge
```bash
pip install deepbridge
```

### 2. Instalar Dependências dos Exemplos
```bash
cd examples
pip install -r requirements.txt
```

### 3. Executar um Exemplo
```bash
python 02_experiment/basic/01_binary_classification.py
```

### 4. Ver Relatórios Gerados
Os relatórios HTML serão salvos em `./reports/` (ou conforme especificado no exemplo).

---

## 📚 Documentação Relacionada

### Planejamento Completo
- **[PLANEJAMENTO_EXEMPLOS_CORE.md](./PLANEJAMENTO_EXEMPLOS_CORE.md)** - Documento detalhado de planejamento
- **[SUMARIO_EXEMPLOS_CORE.md](./SUMARIO_EXEMPLOS_CORE.md)** - Sumário executivo

### Documentação Técnica
- **[planejamento_doc/1-CORE/](../planejamento_doc/1-CORE/)** - Documentação completa do módulo CORE
  - `INDEX.md` - Visão geral
  - `01-DBDATASET.md` - DBDataset detalhado
  - `02-EXPERIMENT.md` - Experiment detalhado
  - `03-MANAGERS.md` - Test Managers
  - `04-REPORT-SYSTEM.md` - Sistema de relatórios
  - `07-ARQUITETURA.md` - Arquitetura do sistema

---

## 🚀 Roadmap de Aprendizado Sugerido

### Semana 1: Fundamentos
**Objetivo**: Entender conceitos básicos

- [ ] Ler documentação de DBDataset
- [ ] Executar `01_dbdataset/basic/01_basic_loading.py`
- [ ] Executar `01_dbdataset/basic/02_presplit_data.py`
- [ ] Executar `02_experiment/basic/01_binary_classification.py` ⭐

**Resultado**: Conseguir criar experimentos básicos

---

### Semana 2: Integração com Modelos
**Objetivo**: Validar modelos próprios

- [ ] Treinar modelo próprio
- [ ] Executar `01_dbdataset/intermediate/01_with_model.py`
- [ ] Executar `01_dbdataset/intermediate/02_load_model.py`
- [ ] Adaptar para seu dataset

**Resultado**: Validar modelos de produção

---

### Semana 3: Testes Específicos
**Objetivo**: Análises profundas

- [ ] Executar `02_experiment/intermediate/01_robustness_deep.py`
- [ ] Executar `02_experiment/intermediate/02_uncertainty.py`
- [ ] Executar `02_experiment/intermediate/03_resilience.py`
- [ ] Analisar relatórios HTML

**Resultado**: Entender cada tipo de teste

---

### Semana 4: Fairness e Compliance
**Objetivo**: Garantir conformidade

- [ ] Ler sobre fairness em ML
- [ ] Executar `02_experiment/advanced/01_fairness_complete.py` ⭐⭐
- [ ] Estudar métricas de fairness
- [ ] Executar `05_use_cases/credit_scoring/` ⭐⭐⭐

**Resultado**: Validar fairness em modelos

---

### Semana 5+: Casos de Uso e Avançado
**Objetivo**: Aplicar em projetos reais

- [ ] Escolher caso de uso mais próximo do seu domínio
- [ ] Adaptar para seu problema
- [ ] Explorar customizações avançadas
- [ ] Integrar em pipeline de produção

**Resultado**: DeepBridge em produção

---

## 💡 Dicas e Melhores Práticas

### ✅ Boas Práticas
1. **Sempre começar com config='quick'** para validar pipeline
2. **Usar config='medium'** para validação regular
3. **config='full' apenas para modelos críticos** (demora mais)
4. **Salvar modelos treinados** para economizar tempo
5. **Usar prob_cols** quando possível para datasets grandes
6. **Documentar protected_attributes** em análises de fairness

### ❌ Erros Comuns
1. ❌ Não validar consistência de train/test
2. ❌ Esquecer de especificar target_column
3. ❌ Fornecer model_path E model (mutuamente exclusivos)
4. ❌ Executar config='full' em modelos pequenos (desperdício)
5. ❌ Não verificar compliance antes de deployment

---

## 🐛 Troubleshooting

### Erro: "Cannot provide both 'data' and 'train_data'"
**Solução**: Escolha UM método de fornecer dados:
```python
# Opção 1: Dataset único
dataset = DBDataset(data=df, ...)

# Opção 2: Train/test separados
dataset = DBDataset(train_data=train, test_data=test, ...)
```

### Erro: "Model não tem método predict_proba"
**Solução**: Certifique-se que está usando classificador (não regressor):
```python
# ✅ Para classificação
from sklearn.ensemble import RandomForestClassifier

# ❌ Para regressão (não tem predict_proba)
from sklearn.ensemble import RandomForestRegressor
```

### Relatório HTML não abre
**Solução**: Verifique caminho do arquivo e permissões:
```python
# Use caminho absoluto ou relativo correto
import os
output_path = os.path.abspath('reports/report.html')
exp.save_html('robustness', output_path)
```

### Performance muito lenta
**Solução**: Use config='quick' ou prob_cols pré-computadas:
```python
# Opção 1: Config rápida
results = exp.run_tests(config_name='quick')

# Opção 2: Pre-computar probabilidades
dataset = DBDataset(..., prob_cols=['prob_0', 'prob_1'])
```

---

## 📞 Suporte e Comunidade

### Tem dúvidas?
- **Issues**: [GitHub Issues](https://github.com/DeepBridge-Validation/DeepBridge/issues)
- **Discussões**: [GitHub Discussions](https://github.com/DeepBridge-Validation/DeepBridge/discussions)
- **Documentação**: [Docs](../planejamento_doc/)

### Quer contribuir?
- Sugerir novos exemplos
- Reportar bugs em exemplos
- Melhorar documentação
- Compartilhar casos de uso

---

## 📈 Status de Implementação

### ✅ Implementado (Total: 0/27)

_Nenhum exemplo implementado ainda - em planejamento_

### 🚧 Em Desenvolvimento (Total: 0/27)

_Aguardando início do desenvolvimento_

### 📋 Planejado (Total: 27/27)

Todos os 27 exemplos estão planejados. Consulte:
- [PLANEJAMENTO_EXEMPLOS_CORE.md](./PLANEJAMENTO_EXEMPLOS_CORE.md) para detalhes
- [SUMARIO_EXEMPLOS_CORE.md](./SUMARIO_EXEMPLOS_CORE.md) para visão rápida

---

## 🎯 Próximos Passos

### Para Usuários
1. ⬜ Escolher exemplo do seu nível
2. ⬜ Executar e entender o código
3. ⬜ Adaptar para seu dataset
4. ⬜ Compartilhar feedback

### Para Desenvolvedores
1. ⬜ Revisar planejamento
2. ⬜ Implementar Fase 1 (4 exemplos básicos)
3. ⬜ Testar em ambiente limpo
4. ⬜ Iterar baseado em feedback

---

## 📄 Licença

Todos os exemplos são fornecidos sob a mesma licença do DeepBridge.

---

## 🙏 Agradecimentos

Exemplos desenvolvidos pela equipe DeepBridge com contribuições da comunidade.

---

**Última Atualização**: 04 de Novembro de 2025
**Versão**: 1.0
**Status**: 📋 PLANEJAMENTO

---

## 🌟 Destaques

### Exemplo Mais Popular
🥇 **binary_classification** - O exemplo mais executado

### Exemplo Mais Crítico
🔒 **fairness_complete** - Essencial para compliance

### Exemplo Mais Completo
📊 **credit_scoring** - Caso de uso end-to-end

---

**Happy Validating! 🚀**

Para começar, execute:
```bash
python 02_experiment/basic/01_binary_classification.py
```
