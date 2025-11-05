# 📦 DBDataset - Gerenciamento de Dados

Aprenda todas as formas de usar o DBDataset, o container fundamental do DeepBridge.

---

## 📓 Notebooks desta Pasta (Fase 1: 3/7 implementados)

### ✅ Implementados

| # | Notebook | Tempo | Descrição | Prioridade |
|---|----------|-------|-----------|------------|
| 1 | `01_carregamento_simples.ipynb` | 10 min | Split automático, test_size, random_state | 🔴 ALTA |
| 2 | `02_dados_pre_separados.ipynb` | 10 min | Train/test já separados (estilo Kaggle) | 🔴 ALTA |
| 3 | `03_integracao_modelos.ipynb` | 15 min | Modelo em memória, predições automáticas | 🔴 ALTA |

### 🔄 A Implementar (Fases 2-5)

| # | Notebook | Tempo | Descrição | Prioridade |
|---|----------|-------|-----------|------------|
| 4 | `04_modelos_salvos.ipynb` | 15 min | Carregar .pkl, .joblib, produção | 🔴 ALTA |
| 5 | `05_probabilidades_precomputadas.ipynb` | 15 min | prob_cols, otimização | 🟡 MÉDIA |
| 6 | `06_selecao_features.ipynb` | 20 min | Subset features, engineering | 🟡 MÉDIA |
| 7 | `07_features_categoricas.ipynb` | 15 min | Auto-detecção, max_categories | 🟢 BAIXA |

**Tempo Total (quando completo)**: ~100 minutos

---

## 🎯 Ordem Recomendada

### Básico (Fase 1 - Disponível Agora!)
1. `01_carregamento_simples.ipynb` - Comece aqui
2. `02_dados_pre_separados.ipynb` - Caso comum
3. `03_integracao_modelos.ipynb` - Adicionar modelos

### Intermediário (Fases futuras)
4. `04_modelos_salvos.ipynb` - Produção
5. `05_probabilidades_precomputadas.ipynb` - Otimização
6. `06_selecao_features.ipynb` - Feature engineering

### Avançado (Fase futura)
7. `07_features_categoricas.ipynb` - Detecção automática

---

## 📖 O que Você Vai Aprender

### 1. Carregamento Simples
- ✅ Split automático (80/20, 70/30, etc.)
- ✅ random_state para reproducibilidade
- ✅ Explorar propriedades básicas

### 2. Dados Pré-separados
- ✅ usar train_data + test_data
- ✅ Validações automáticas de consistência
- ✅ Quando usar cada abordagem

### 3. Integração com Modelos
- ✅ Passar modelo com `model=`
- ✅ Predições automáticas
- ✅ Diferentes tipos de modelos (sklearn, xgboost, etc.)
- ✅ Acessar `.train_predictions`, `.test_predictions`

### 4. Modelos Salvos (futuro)
- Carregar de .pkl, .joblib, .h5, .onnx
- Validar modelos de produção
- Economizar tempo de treinamento

### 5. Probabilidades Pré-computadas (futuro)
- Usar `prob_cols=`
- Economizar tempo em modelos pesados
- Benchmark de performance

### 6. Seleção de Features (futuro)
- Especificar `features=`
- Comparar modelos com diferentes features
- Feature engineering

### 7. Features Categóricas (futuro)
- Auto-detecção inteligente
- Controlar com `max_categories`
- Manual vs automático

---

## 🎓 Pré-requisitos

**Para notebooks da Fase 1:**
- Completar `01_introducao/` (recomendado)
- Conhecimento básico de Pandas
- Familiaridade com datasets e splits

**Instalação:**
```bash
pip install deepbridge jupyter pandas numpy scikit-learn matplotlib seaborn
```

---

## 🚀 Como Executar

```bash
# 1. Navegar até a pasta
cd /home/guhaase/projetos/DeepBridge/examples/notebooks/02_dbdataset

# 2. Iniciar Jupyter
jupyter notebook

# 3. Abrir o primeiro notebook
# 01_carregamento_simples.ipynb
```

---

## 💡 Principais Conceitos

### DBDataset - O Container Fundamental

```python
from deepbridge import DBDataset

# Forma 1: Split automático
dataset = DBDataset(
    data=df,
    target_column='target',
    test_size=0.2,
    random_state=42
)

# Forma 2: Train/test pré-separados
dataset = DBDataset(
    train_data=train_df,
    test_data=test_df,
    target_column='target'
)

# Forma 3: Com modelo
dataset = DBDataset(
    data=df,
    target_column='target',
    model=clf  # Predições automáticas!
)
```

### Propriedades Importantes

```python
dataset.features                # Lista de features
dataset.categorical_features    # Features categóricas
dataset.numerical_features      # Features numéricas
dataset.train_data             # Dados de treino
dataset.test_data              # Dados de teste
dataset.train_predictions      # Predições de treino
dataset.test_predictions       # Predições de teste
```

---

## 🎯 Decisão: Qual Notebook Usar?

| Seu Cenário | Notebook Recomendado |
|-------------|---------------------|
| Tenho um DataFrame único | `01_carregamento_simples` |
| Tenho train.csv e test.csv | `02_dados_pre_separados` |
| Tenho modelo treinado em memória | `03_integracao_modelos` |
| Tenho modelo salvo (.pkl) | `04_modelos_salvos` (futuro) |
| Modelo é muito pesado | `05_probabilidades_precomputadas` (futuro) |
| Quero testar subset de features | `06_selecao_features` (futuro) |

---

## 🔄 Status de Implementação

- ✅ **Fase 1 Completa** (3/3 notebooks) - Disponível agora!
- ⏳ **Fase 2-5** (4 notebooks restantes) - Em planejamento

---

## 🎯 Próximos Passos

Depois de dominar DBDataset, continue para:

📁 **03_testes_validacao/** - Testes de validação
- Robustez completa
- Incerteza
- Resiliência
- Comparação de modelos

📁 **04_fairness/** - Análise de fairness
- 15 métricas
- EEOC compliance
- Mitigação de bias

---

## 📞 Precisa de Ajuda?

- 📖 [Documentação DBDataset](../../planejamento_doc/1-CORE/01-DBDATASET.md)
- 💻 [Código Fonte](https://github.com/DeepBridge-Validation/DeepBridge)
- ❓ [Issues](https://github.com/DeepBridge-Validation/DeepBridge/issues)

---

**Última Atualização**: 04 de Novembro de 2025
**Status**: ✅ Fase 1 Completa (3/7 notebooks)
**Próxima Implementação**: Fase 2 (notebooks 4-7)
