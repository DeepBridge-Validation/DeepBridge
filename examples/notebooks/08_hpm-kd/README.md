# HPM-KD: Hierarchical Progressive Multi-Teacher Knowledge Distillation

Este diretório contém notebooks de demonstração do framework HPM-KD implementado na biblioteca DeepBridge.

## 📚 Conteúdo

### `hpmkd_demo.ipynb`

Demonstração completa do uso do HPM-KD, incluindo:

1. **Configuração e Importações**: Setup do ambiente e bibliotecas necessárias
2. **Preparação de Dados**: Carregamento e divisão do dataset Digits
3. **Treinamento do Professor**: Criação de um modelo grande e complexo (Random Forest)
4. **Baseline**: Treinamento direto de um modelo pequeno para comparação
5. **HPM-KD em Ação**: Destilação com configuração automática
6. **Avaliação**: Comparação de métricas entre Professor, Baseline e HPM-KD
7. **Visualizações**: Gráficos comparativos de acurácia e compressão
8. **Análise de Componentes**: Verificação dos componentes ativos do HPM-KD

## 🎯 Objetivo

Demonstrar o uso simplificado do HPM-KD conforme apresentado no **Listing 1** do paper:

```python
from deepbridge.distillation.techniques.hpm import HPMDistiller, HPMConfig

# Configuração automática via meta-learning
hpmkd = HPMDistiller(
    teacher_model=teacher,
    student_model_type=ModelType.DECISION_TREE,
    config=HPMConfig(use_progressive=True, use_multi_teacher=True)
)

# Destilação progressiva multi-professor
hpmkd.fit(X_train, y_train, X_val, y_val)

# Avaliar estudante comprimido
student_acc = accuracy_score(y_test, hpmkd.predict(X_test))
```

## 🔧 Componentes do HPM-KD

O framework integra 6 componentes sinérgicos:

1. **Adaptive Configuration Manager**: Meta-aprendizado para seleção automática de hiperparâmetros
2. **Progressive Distillation Chain**: Cadeia hierárquica de modelos intermediários
3. **Multi-Teacher Ensemble**: Ensemble com atenção aprendida para ponderação dinâmica
4. **Meta Temperature Scheduler**: Ajuste adaptativo da temperatura durante treinamento
5. **Parallel Processing Pipeline**: Distribuição eficiente de tarefas de destilação
6. **Shared Optimization Memory**: Caching cross-experimento para reutilização

## 📊 Resultados Esperados

- **Compressão**: 10×-15× redução no tamanho do modelo
- **Retenção**: 85%+ da acurácia do professor
- **Eficiência**: Configuração automática sem ajuste manual
- **Superioridade**: Ganhos sobre baseline de treinamento direto

## 🚀 Como Executar

### Pré-requisitos

```bash
pip install deepbridge scikit-learn matplotlib seaborn pandas numpy
```

### Executando o Notebook

```bash
jupyter notebook hpmkd_demo.ipynb
```

Ou use o Jupyter Lab:

```bash
jupyter lab hpmkd_demo.ipynb
```

## 📖 Referências

- **Paper**: HPM-KD: Hierarchical Progressive Multi-Teacher Knowledge Distillation Framework
- **Documentação DeepBridge**: https://deepbridge.readthedocs.io
- **Repositório**: https://github.com/DeepBridge-Validation/DeepBridge

## 💡 Dicas

- O notebook usa o dataset **Digits** do scikit-learn para facilitar execução rápida
- Para experimentos maiores (CIFAR-10, CIFAR-100), ajuste os parâmetros de configuração
- O tempo de execução depende da configuração `n_trials` no HPMConfig
- Para processamento paralelo, ajuste `use_parallel=True` e `parallel_workers`

## 🔍 Troubleshooting

### Erro de Importação

Se encontrar erro ao importar HPMDistiller:

```python
# Verifique se o DeepBridge está instalado
pip install -e /path/to/DeepBridge

# Ou adicione ao PYTHONPATH
import sys
sys.path.append('/path/to/DeepBridge')
```

### Problemas de Memória

Para datasets grandes, ajuste:

```python
config = HPMConfig(
    cache_memory_gb=1.0,  # Reduzir cache
    use_parallel=False,   # Desabilitar paralelização
    n_trials=3            # Menos trials
)
```

## 📝 Notas

- O código foi validado contra a implementação do paper
- A API segue o padrão scikit-learn para facilitar integração
- Todos os componentes podem ser habilitados/desabilitados individualmente

---

**Autores**: Gustavo Coelho Haase, Paulo Henrique Dourado da Silva
**Instituições**: Universidade Católica de Brasília, Universidade de São Paulo
**Licença**: MIT
