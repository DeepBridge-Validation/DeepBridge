# 📓 DeepBridge Jupyter Notebooks

Bem-vindo aos **exemplos interativos** do DeepBridge!

---

## 🚀 Início Rápido

### Novo no DeepBridge? Comece aqui:

1. **[01_introducao/01_primeiros_passos.ipynb](./01_introducao/01_primeiros_passos.ipynb)** (15 min)
   - Primeiro contato com a biblioteca

2. **[01_introducao/03_workflow_completo.ipynb](./01_introducao/03_workflow_completo.ipynb)** ⭐ (30 min)
   - **DEMO PRINCIPAL** - Ver todo o poder do DeepBridge!

---

## 📁 Estrutura dos Notebooks

```
notebooks/
│
├── 📁 01_introducao/           ✅ Completo (3/3 notebooks)
│   ├── 01_primeiros_passos.ipynb       [15 min]
│   ├── 02_conceitos_basicos.ipynb      [20 min]
│   └── 03_workflow_completo.ipynb ⭐    [30 min] DEMO PRINCIPAL
│
├── 📁 02_dbdataset/            ✅ Fase 1 (3/7 notebooks)
│   ├── 01_carregamento_simples.ipynb   [10 min]
│   ├── 02_dados_pre_separados.ipynb    [10 min]
│   ├── 03_integracao_modelos.ipynb     [15 min]
│   ├── 04_modelos_salvos.ipynb         [15 min] 🔄 Futuro
│   ├── 05_probabilidades_precomputadas.ipynb [15 min] 🔄 Futuro
│   ├── 06_selecao_features.ipynb       [20 min] 🔄 Futuro
│   └── 07_features_categoricas.ipynb   [15 min] 🔄 Futuro
│
├── 📁 03_testes_validacao/     🔄 Fase 2 (Planejado)
│   ├── 01_introducao_testes.ipynb
│   ├── 02_robustez_completa.ipynb
│   ├── 03_incerteza.ipynb
│   ├── 04_resiliencia_drift.ipynb
│   ├── 05_hiperparametros.ipynb
│   └── 06_comparacao_modelos.ipynb
│
├── 📁 04_fairness/             🔄 Fase 2 (Planejado)
│   ├── 01_introducao_fairness.ipynb
│   ├── 02_analise_completa_fairness.ipynb ⭐⭐
│   └── 03_mitigacao_bias.ipynb
│
├── 📁 05_casos_uso/            🔄 Fase 3 (Planejado)
│   ├── 01_credit_scoring.ipynb ⭐⭐⭐
│   ├── 02_diagnostico_medico.ipynb
│   ├── 03_churn_prediction.ipynb
│   ├── 04_fraud_detection.ipynb
│   └── 05_regressao_precos.ipynb
│
└── 📁 06_avancado/              🔄 Fase 5 (Planejado)
    ├── 01_otimizacao_performance.ipynb
    ├── 02_customizacao_relatorios.ipynb
    └── 03_extensibilidade.ipynb
```

---

## 📊 Status de Implementação

### ✅ Fase 1 - Fundação (Completa!)

**6 notebooks implementados** (~100 minutos)
- ✅ 01_introducao/ (3 notebooks)
- ✅ 02_dbdataset/ (3 notebooks)

**Resultado**: Usuários conseguem entender e usar a biblioteca!

### 🔄 Fase 2 - Testes e Fairness (Planejado)

**6 notebooks** (~210 minutos)
- 03_testes_validacao/ (3 notebooks)
- 04_fairness/ (3 notebooks)

**Entrega**: Testes principais e compliance

### 🔄 Fase 3 - Casos de Uso (Planejado)

**3 notebooks** (~125 minutos)
- 05_casos_uso/ (3 notebooks críticos)

**Entrega**: Aplicações reais

### 🔄 Fases 4-5 - Completar (Planejado)

**12 notebooks** (~400 minutos)
- Notebooks restantes de todas as pastas

**Entrega**: 27 notebooks completos

---

## 🎯 Trilhas de Aprendizado

### 👤 Trilha 1: Iniciante Completo (2 horas)

**Objetivo**: Do zero ao uso básico

1. `01_introducao/01_primeiros_passos.ipynb` (15 min)
2. `01_introducao/02_conceitos_basicos.ipynb` (20 min)
3. `01_introducao/03_workflow_completo.ipynb` ⭐ (30 min)
4. `02_dbdataset/01_carregamento_simples.ipynb` (10 min)
5. `02_dbdataset/03_integracao_modelos.ipynb` (15 min)

**Resultado**: Consegue validar modelos com DeepBridge ✅

---

### 👤 Trilha 2: ML Engineer (Disponível na Fase 2)

**Objetivo**: Validar modelos para deploy

Notebooks futuros de:
- Robustez
- Fairness
- Resiliência
- Credit Scoring

---

### 👤 Trilha 3: Compliance Officer (Disponível na Fase 2)

**Objetivo**: Garantir fairness e regulação

Notebooks futuros de:
- Fairness completo
- Credit Scoring
- Compliance

---

## 🎓 Pré-requisitos

### Software
- Python 3.8+
- Jupyter Notebook ou JupyterLab
- Git (para clonar o repositório)

### Conhecimento
- Python básico
- Pandas básico
- Scikit-learn (recomendado)

### Instalação

```bash
# 1. Clonar repositório
git clone https://github.com/DeepBridge-Validation/DeepBridge.git
cd DeepBridge/examples/notebooks

# 2. Criar ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# 3. Instalar dependências
pip install deepbridge jupyter pandas numpy matplotlib seaborn scikit-learn

# 4. Iniciar Jupyter
jupyter notebook
```

---

## 💡 Como Usar Estes Notebooks

### Dicas Importantes

1. **Execute em ordem** - Cada notebook se baseia no anterior
2. **Leia os comentários** - Explicações importantes estão nos comentários
3. **Execute célula por célula** - Não pule células!
4. **Experimente!** - Modifique parâmetros e veja o que acontece
5. **Tempo de execução** - Alguns testes demoram minutos (indicado no notebook)

### Navegação

- Cada pasta tem seu próprio `README.md` com detalhes
- Use índices nos notebooks para navegar
- Links entre notebooks facilitam o fluxo

---

## 📊 Estatísticas

### Implementados (Fase 1)

- **Total**: 6 notebooks
- **Tempo**: ~100 minutos de conteúdo
- **Pastas**: 2 (Introdução, DBDataset parcial)
- **Status**: ✅ Completo e testado

### Planejados (Total)

- **Total Final**: 27 notebooks
- **Tempo Total**: ~12 horas de conteúdo
- **Pastas**: 6
- **Status**: 22% implementado

---

## 🎯 Notebooks Mais Importantes

### 🥇 Top 1: `01_introducao/03_workflow_completo.ipynb` ⭐

**Por quê**: Demo principal - mostra TODO o poder do DeepBridge
**Tempo**: 30 min
**O que faz**: Validação completa end-to-end de um modelo

### 🥈 Top 2: `04_fairness/02_analise_completa_fairness.ipynb` (Futuro)

**Por quê**: 15 métricas de fairness + EEOC compliance
**Diferencial**: Único no mercado

### 🥉 Top 3: `05_casos_uso/01_credit_scoring.ipynb` (Futuro)

**Por quê**: Caso real completo com compliance regulatório
**Aplicação**: Produção real

---

## 🔧 Troubleshooting

### Jupyter não inicia
```bash
pip install --upgrade jupyter
jupyter notebook --no-browser  # Abrir manualmente
```

### Imports falham
```bash
pip install --upgrade deepbridge
# ou
pip install -r requirements.txt
```

### Kernel morre durante execução
- Alguns testes são pesados
- Aumente memória disponível
- Use config='quick' ao invés de 'full'

---

## 📚 Recursos Adicionais

### Documentação
- 📖 [Planejamento Completo](../NOTEBOOKS_PLANEJAMENTO.md)
- 📖 [Índice Rápido](../NOTEBOOKS_INDEX.md)
- 📖 [Documentação Técnica](../../planejamento_doc/1-CORE/)

### Comunidade
- 💻 [GitHub](https://github.com/DeepBridge-Validation/DeepBridge)
- ❓ [Issues](https://github.com/DeepBridge-Validation/DeepBridge/issues)
- 💬 [Discussions](https://github.com/DeepBridge-Validation/DeepBridge/discussions)

---

## 🎉 Começe Agora!

```bash
# Abrir o primeiro notebook
jupyter notebook 01_introducao/01_primeiros_passos.ipynb
```

Ou pule direto para o **DEMO PRINCIPAL**:
```bash
jupyter notebook 01_introducao/03_workflow_completo.ipynb
```

---

## 🗺️ Roadmap

- ✅ **Fase 1** (Nov 2025): Introdução + DBDataset básico - **COMPLETA!**
- 🔄 **Fase 2** (Dez 2025): Testes de validação + Fairness
- 🔄 **Fase 3** (Jan 2026): Casos de uso reais
- 🔄 **Fase 4-5** (Fev 2026): Completar todos os 27 notebooks

---

<div style="background-color: #e3f2fd; padding: 20px; border-radius: 10px; text-align: center;">
<h2 style="color: #1976d2;">🎊 Bem-vindo ao DeepBridge! 🎊</h2>
<p style="font-size: 18px; color: #1565c0;">
Validação profissional de modelos em minutos, não semanas.<br>
Comece agora e veja a diferença!
</p>
</div>

---

**Última Atualização**: 04 de Novembro de 2025
**Versão**: 1.0 (Fase 1)
**Status**: ✅ 6/27 notebooks implementados (22%)
**Próxima Release**: Fase 2 (Dez 2025)
