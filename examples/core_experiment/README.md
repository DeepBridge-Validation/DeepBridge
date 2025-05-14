# Core Experiment Module Examples

O módulo `core.experiment` do DeepBridge fornece classes e funcionalidades para executar experimentos de validação de modelos de machine learning. Estes exemplos demonstram como usar os diferentes componentes deste módulo para avaliar e validar modelos.

## Conteúdo

- [Basic](basic/): Exemplos básicos do uso das classes principais
  - [01_experiment_setup.md](basic/01_experiment_setup.md): Configuração básica de um experimento
  - [02_model_evaluation.md](basic/02_model_evaluation.md): Avaliação de modelos
  - [03_results_handling.md](basic/03_results_handling.md): Manipulação de resultados
  
- [Tests](tests/): Exemplos de execução dos diferentes tipos de testes
  - [01_robustness_tests.md](tests/01_robustness_tests.md): Testes de robustez
  - [02_uncertainty_tests.md](tests/02_uncertainty_tests.md): Testes de incerteza
  - [03_resilience_tests.md](tests/03_resilience_tests.md): Testes de resiliência
  - [04_hyperparameter_tests.md](tests/04_hyperparameter_tests.md): Testes de hiperparâmetros
  
- [Managers](managers/): Exemplos dos gerenciadores específicos
  - [01_model_manager.md](managers/01_model_manager.md): Gerenciamento de modelos alternativos
  - [02_data_manager.md](managers/02_data_manager.md): Gerenciamento de dados de experimento
  - [03_specialized_managers.md](managers/03_specialized_managers.md): Uso de gerenciadores especializados

- [04_report_generation.md](04_report_generation.md): Exemplos de geração de relatórios a partir dos resultados dos experimentos

## Conceitos Principais

O módulo `core.experiment` é estruturado em torno de alguns conceitos-chave:

1. **Experiment**: Classe central que coordena a execução de experimentos de validação
2. **TestRunner**: Responsável por executar os testes específicos nos modelos
3. **Managers**: Classes especializadas que gerenciam aspectos específicos do experimento
4. **DataManager**: Gerencia os dados usados nos experimentos
5. **ModelEvaluation**: Avalia o desempenho dos modelos com diferentes métricas
6. **Relatórios**: Geração de relatórios visuais dos resultados dos testes

## Guia Rápido

### Uso Básico

```python
from deepbridge.core.db_data import DBDataset
from deepbridge.core.experiment.experiment import Experiment
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Preparar dados
df = pd.read_csv('data.csv')

# Criar modelo e dataset
model = RandomForestClassifier()
dataset = DBDataset(data=df, target_column='target', model=model)

# Criar experimento
experiment = Experiment(name="meu_experimento", dataset=dataset)

# Executar testes
experiment.run_robustness_tests()
experiment.run_uncertainty_tests()

# Gerar relatório
experiment.generate_report(output_dir="./resultados")
```

## Workflows Típicos

1. **Avaliação de Modelo Único**:
   - Configurar `DBDataset` com dados e modelo
   - Criar `Experiment`
   - Executar testes específicos
   - Gerar relatório

2. **Comparação de Modelos**:
   - Configurar `DBDataset` com dados e modelo principal
   - Adicionar modelos alternativos ao `Experiment`
   - Executar métodos de comparação de modelos
   - Gerar relatório com comparações

3. **Análise Personalizada**:
   - Usar gerenciadores especializados diretamente
   - Customizar parâmetros e configurações
   - Implementar lógica de avaliação personalizada

Estes exemplos demonstram uma variedade de casos de uso e padrões para ajudar você a tirar o máximo proveito do sistema de experimentação do DeepBridge.