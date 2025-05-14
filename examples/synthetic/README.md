# Synthetic Data Generation Examples

O módulo `synthetic` do DeepBridge oferece ferramentas robustas para geração de dados sintéticos a partir de dados reais, preservando as características estatísticas importantes dos dados originais. Estes exemplos demonstram como usar as diferentes classes e funcionalidades do módulo.

## Conteúdo

- [Basic](basic/): Exemplos básicos de uso da classe `Synthesize`
  - [01_basic_usage.md](basic/01_basic_usage.md): Uso básico da classe Synthesize
  - [02_quality_metrics.md](basic/02_quality_metrics.md): Métricas de qualidade de dados sintéticos
  - [03_report_generation.md](basic/03_report_generation.md): Geração de relatórios de qualidade
  
- [Methods](methods/): Exemplos específicos de métodos de geração de dados sintéticos
  - [01_gaussian_copula.md](methods/01_gaussian_copula.md): Usando o método Gaussian Copula
  - [02_standard_generator.md](methods/02_standard_generator.md): Usando o StandardGenerator
  - [03_comparison.md](methods/03_comparison.md): Comparação entre diferentes métodos
  
- [Integration](integration/): Exemplos de integração com outros componentes do DeepBridge
  - [01_with_experiment.md](integration/01_with_experiment.md): Integração com Experiment e DBDataset
  - [02_large_datasets.md](integration/02_large_datasets.md): Processamento de grandes conjuntos de dados com Dask

## Conceitos Principais

O módulo `synthetic` é estruturado em torno de alguns conceitos-chave:

1. **Gerador de Dados Sintéticos**: A classe central `Synthesize` que oferece uma interface unificada para todos os métodos
2. **Métodos de Geração**: Diferentes abordagens para geração de dados (Gaussian Copula, KDE, GMM, etc.)
3. **Métricas de Qualidade**: Ferramentas para avaliar a qualidade dos dados sintéticos gerados
4. **Processamento Distribuído**: Suporte para processamento distribuído via Dask para grandes conjuntos de dados

Estes exemplos mostram como utilizar essas funcionalidades para diferentes casos de uso.