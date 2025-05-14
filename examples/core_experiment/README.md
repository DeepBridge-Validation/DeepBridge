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
  
- [Advanced](advanced/): Exemplos avançados e casos de uso específicos
  - [01_custom_test_strategies.md](advanced/01_custom_test_strategies.md): Implementando estratégias de teste personalizadas
  - [02_advanced_distillation.md](advanced/02_advanced_distillation.md): Técnicas avançadas de destilação de modelos
  - [03_report_customization.md](advanced/03_report_customization.md): Personalização de relatórios

- [Managers](managers/): Exemplos dos gerenciadores específicos
  - [01_model_manager.md](managers/01_model_manager.md): Gerenciamento de modelos alternativos
  - [02_data_manager.md](managers/02_data_manager.md): Gerenciamento de dados de experimento
  - [03_specialized_managers.md](managers/03_specialized_managers.md): Uso de gerenciadores especializados

## Conceitos Principais

O módulo `core.experiment` é estruturado em torno de alguns conceitos-chave:

1. **Experiment**: Classe central que coordena a execução de experimentos de validação
2. **TestRunner**: Responsável por executar os testes específicos nos modelos
3. **Managers**: Classes especializadas que gerenciam aspectos específicos do experimento
4. **DataManager**: Gerencia os dados usados nos experimentos
5. **ModelEvaluation**: Avalia o desempenho dos modelos com diferentes métricas
6. **Relatórios**: Geração de relatórios visuais dos resultados dos testes

Estes exemplos mostram como utilizar estas componentes para diferentes casos de uso e tipos de modelos.