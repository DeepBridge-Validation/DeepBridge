# 📚 Introdução ao DeepBridge

Bem-vindo aos notebooks de introdução! Comece aqui se você é novo no DeepBridge.

---

## 📓 Notebooks desta Pasta

| # | Notebook | Tempo | Descrição | Prioridade |
|---|----------|-------|-----------|------------|
| 1 | `01_primeiros_passos.ipynb` | 15 min | Primeiro contato, Iris dataset, DBDataset básico | 🔴 ALTA |
| 2 | `02_conceitos_basicos.ipynb` | 20 min | Arquitetura, Experiment, tipos de testes | 🔴 ALTA |
| 3 | `03_workflow_completo.ipynb` ⭐ | 30 min | **DEMO PRINCIPAL** - Workflow end-to-end completo | 🔴 ALTA |

**Tempo Total**: ~65 minutos

---

## 🎯 Ordem Recomendada

### Para Iniciantes
1. **Comece aqui:** `01_primeiros_passos.ipynb`
   - Instalação e setup
   - Criar primeiro DBDataset
   - Explorar propriedades básicas

2. **Continue com:** `02_conceitos_basicos.ipynb`
   - Entender arquitetura
   - Conhecer o Experiment
   - Tipos de testes disponíveis

3. **Finalize com:** `03_workflow_completo.ipynb` ⭐
   - **NOTEBOOK MAIS IMPORTANTE!**
   - Ver todo o poder do DeepBridge
   - Validação completa de modelo
   - Geração de relatórios

### Atalho para Usuários Experientes
- Pule direto para `03_workflow_completo.ipynb` ⭐
- Ver demonstração completa em 30 minutos

---

## 📖 O que Você Vai Aprender

### Notebook 1: Primeiros Passos
- ✅ Instalar DeepBridge
- ✅ Carregar dataset (Iris)
- ✅ Criar DBDataset
- ✅ Explorar features categóricas/numéricas
- ✅ Visualizações básicas

### Notebook 2: Conceitos Básicos
- ✅ Arquitetura do DeepBridge
- ✅ DBDataset vs Experiment
- ✅ 5 tipos de testes:
  - Robustness
  - Uncertainty
  - Resilience
  - Hyperparameter
  - Fairness
- ✅ Configurações (quick/medium/full)

### Notebook 3: Workflow Completo ⭐
- ✅ **Caso real: Credit Scoring**
- ✅ Preparação de dados
- ✅ Treinamento de modelo
- ✅ Validação completa (Robustez + Fairness)
- ✅ Geração de relatórios HTML
- ✅ Decisão de deploy

---

## 🎓 Pré-requisitos

- Python 3.8+
- Jupyter Notebook ou JupyterLab
- Conhecimento básico de:
  - Python
  - Pandas
  - Scikit-learn (básico)

---

## 🚀 Como Executar

```bash
# 1. Ativar ambiente
source venv/bin/activate  # ou ative seu ambiente

# 2. Instalar dependências
pip install deepbridge jupyter pandas numpy matplotlib seaborn scikit-learn

# 3. Iniciar Jupyter
jupyter notebook

# 4. Navegar até esta pasta e abrir o notebook
```

---

## 💡 Dicas

1. **Execute célula por célula** - Não pule células!
2. **Leia os comentários** - Explicações importantes
3. **Experimente!** - Modifique parâmetros e veja o que acontece
4. **Tempo de execução** - Alguns testes demoram minutos

---

## 🎯 Próximos Passos

Depois de completar esta pasta, continue para:

📁 **02_dbdataset/** - Aprofunde-se no DBDataset
- Diferentes formas de carregar dados
- Integração avançada com modelos
- Otimizações

📁 **03_testes_validacao/** - Domine os testes
- Robustez em profundidade
- Quantificação de incerteza
- Detecção de drift

📁 **04_fairness/** - Fairness e compliance
- 15 métricas de fairness
- Verificação EEOC
- Mitigação de bias

---

## 📞 Precisa de Ajuda?

- 📖 [Documentação Completa](../../planejamento_doc/1-CORE/)
- 💻 [Código Fonte](https://github.com/DeepBridge-Validation/DeepBridge)
- ❓ [Issues](https://github.com/DeepBridge-Validation/DeepBridge/issues)

---

**Última Atualização**: 04 de Novembro de 2025
**Status**: ✅ Completo (3/3 notebooks)
