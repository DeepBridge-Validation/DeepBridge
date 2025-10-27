# Processo de Distilação HPM-KD (Hierarchical Progressive Multi-Teacher Knowledge Distillation)

## Visão Geral

O processo de distilação HPM-KD é ativado quando se usa `AutoDistiller` com `method='hpm'`. Este documento apresenta um fluxograma detalhado do fluxo completo de execução.

## Diagrama de Processo - Fluxograma

```mermaid
flowchart TD
    Start([Início: AutoDistiller com HPM]) --> CheckMethod{method == 'auto'?}

    CheckMethod -->|Sim| AutoDetect[_choose_best_method<br/>Analisa Dataset]
    CheckMethod -->|Não| InitHPM[method = 'hpm']

    AutoDetect --> CheckSize{Dataset > 10k<br/>ou Features > 50?}
    CheckSize -->|Sim| InitHPM
    CheckSize -->|Não| CheckSmall{Dataset < 1k?}
    CheckSmall -->|Sim| UseLegacy[Usa Legacy Method]
    CheckSmall -->|Não| InitHPM

    InitHPM --> CreateConfig[Cria HPMConfig<br/>max_configs: 16<br/>n_trials: 3<br/>use_cache: True<br/>use_progressive: True]

    CreateConfig --> CreateDistiller[Instancia HPMDistiller]

    CreateDistiller --> InitComponents[_initialize_components]

    InitComponents --> CreateManagers[Cria Componentes:<br/>- AdaptiveConfigurationManager<br/>- SharedOptimizationMemory<br/>- IntelligentCache<br/>- ProgressiveDistillationChain<br/>- AttentionWeightedMultiTeacher<br/>- MetaTemperatureScheduler]

    CreateManagers --> RunCall{distiller.run<br/>chamado?}

    RunCall -->|Sim| ExtractFeatures[Extrai características<br/>do dataset]
    RunCall -->|Não| WaitRun[Aguarda chamada<br/>de run]

    WaitRun --> RunCall

    ExtractFeatures --> Phase1[FASE 1: Seleção de Configurações<br/>_select_configurations]

    Phase1 --> SelectConfigs[AdaptiveConfigurationManager<br/>seleciona até 16 configs<br/>promissoras]

    SelectConfigs --> CheckMemory[SharedOptimizationMemory<br/>verifica configs similares<br/>anteriores]

    CheckMemory --> WarmStart{Encontrou<br/>similar?}

    WarmStart -->|Sim| UseWarmStart[Usa hiperparâmetros<br/>anteriores]
    WarmStart -->|Não| DefaultParams[Usa parâmetros<br/>padrão]

    UseWarmStart --> Phase2
    DefaultParams --> Phase2

    Phase2[FASE 2: Treino Progressivo<br/>_run_progressive_chain] --> CheckProgressive{use_progressive<br/>habilitado?}

    CheckProgressive -->|Sim| RunChain[Executa cadeia<br/>progressiva de modelos]
    CheckProgressive -->|Não| SkipChain[Pula cadeia<br/>progressiva]

    RunChain --> TempSchedule{use_adaptive_<br/>temperature?}

    TempSchedule -->|Sim| AdaptTemp[MetaTemperatureScheduler<br/>ajusta temperatura<br/>dinamicamente]
    TempSchedule -->|Não| FixedTemp[Usa temperatura<br/>fixa]

    AdaptTemp --> TrainChain[Treina modelos<br/>incrementalmente<br/>com min_improvement: 0.01]
    FixedTemp --> TrainChain

    TrainChain --> Phase3
    SkipChain --> Phase3

    Phase3[FASE 3: Treino de Modelos<br/>_run_sequential_training] --> CheckParallel{use_parallel<br/>habilitado?}

    CheckParallel -->|Sim| ParallelTrain[Treina configs<br/>em paralelo]
    CheckParallel -->|Não| SequentialTrain[Treina configs<br/>sequencialmente]

    ParallelTrain --> CollectResults[Coleta resultados<br/>de todos os modelos]
    SequentialTrain --> CollectResults

    CollectResults --> Phase4[FASE 4: Ensemble Multi-Teacher<br/>_create_multi_teacher_ensemble]

    Phase4 --> CheckMultiTeacher{use_multi_teacher<br/>e modelos OK?}

    CheckMultiTeacher -->|Sim| CreateEnsemble[AttentionWeightedMultiTeacher<br/>cria ensemble com atenção]
    CheckMultiTeacher -->|Não| SkipEnsemble[Pula criação<br/>de ensemble]

    CreateEnsemble --> OptimizeWeights[Otimiza pesos<br/>usando validação]

    OptimizeWeights --> SelectBest
    SkipEnsemble --> SelectBest

    SelectBest[FASE 5: Seleção do Melhor<br/>_select_best_model] --> CompareModels[Compara:<br/>- Modelos paralelos<br/>- Cadeia progressiva<br/>- Ensemble multi-teacher]

    CompareModels --> EvaluateMetrics[Avalia métricas:<br/>- Accuracy<br/>- F1-score<br/>- KS statistic]

    EvaluateMetrics --> StoreBest[Armazena best_model<br/>e best_metrics]

    StoreBest --> CreateResults[Cria DataFrame<br/>de resultados<br/>_create_hpm_results_df]

    CreateResults --> GenerateReport{Gerar<br/>relatório?}

    GenerateReport -->|Sim| CreateReport[generate_report<br/>Cria HTML interativo<br/>com comparações]
    GenerateReport -->|Não| ReturnResults[Retorna<br/>results_df]

    CreateReport --> ReturnResults

    ReturnResults --> Ready[Sistema Pronto:<br/>- best_model disponível<br/>- predict() habilitado<br/>- predict_proba() habilitado]

    Ready --> End([Fim])

    UseLegacy --> LegacyFlow[Fluxo Legacy<br/>Não detalhado aqui]
    LegacyFlow --> End
```

## Fluxo Detalhado

### 1. **Inicialização do AutoDistiller**

Quando você cria uma instância do `AutoDistiller` com `method='hpm'`:

```python
distiller = AutoDistiller(
    dataset=dataset_experiment,
    method='hpm'
)
```

O sistema:
- Detecta automaticamente se HPM é a melhor escolha baseado no dataset
- Cria uma configuração `HPMConfig` com parâmetros otimizados
- Instancia o `HPMDistiller` com todos os componentes especializados

### 2. **Componentes HPM Inicializados**

#### 2.1 Gerenciamento Adaptativo
- **AdaptiveConfigurationManager**: Seleciona configurações prometedoras baseadas em características do dataset
- **SharedOptimizationMemory**: Mantém histórico de otimizações para reutilização

#### 2.2 Sistema de Cache
- **IntelligentCache**: Sistema de cache inteligente que evita recálculos desnecessários
- Limite de memória configurável (padrão: 2GB)

#### 2.3 Técnicas Avançadas
- **ProgressiveDistillationChain**: Implementa cadeia progressiva de destilação
- **AttentionWeightedMultiTeacher**: Sistema de múltiplos professores com atenção
- **MetaTemperatureScheduler**: Ajusta temperatura dinamicamente durante o treino

### 3. **Execução do Processo de Distilação**

Ao chamar `distiller.run()`:

#### Fase 1: Seleção de Configurações
- Extrai características do dataset (tamanho, features, balanceamento)
- Usa o AdaptiveConfigurationManager para selecionar até 16 configurações promissoras
- Verifica SharedOptimizationMemory para configurações similares anteriores

#### Fase 2: Treino Progressivo (se habilitado)
- Executa cadeia progressiva de modelos
- Cada modelo aprende incrementalmente
- Temperature scheduling adaptativo baseado na performance

#### Fase 3: Treino de Modelos
- Treina modelos selecionados (sequencialmente por padrão)
- Cada modelo é treinado com sua configuração específica
- Coleta métricas detalhadas de cada modelo

#### Fase 4: Ensemble Multi-Teacher (se habilitado)
- Combina modelos bem-sucedidos
- Otimiza pesos usando dados de validação
- Cria ensemble final com fusão adaptativa

### 4. **Seleção e Armazenamento do Melhor Modelo**

O sistema:
- Compara todos os modelos treinados
- Seleciona o melhor baseado em métricas (padrão: KS statistic)
- Armazena como `best_model` para uso posterior

### 5. **Geração de Resultados**

- Cria DataFrame compatível com análises
- Gera relatório HTML interativo
- Permite comparação com modelo original
- Suporta exportação e salvamento do modelo

## Vantagens do HPM-KD

1. **Eficiência**: Reduz número de configurações testadas através de seleção inteligente
2. **Qualidade**: Múltiplas técnicas avançadas melhoram a qualidade da distilação
3. **Adaptabilidade**: Ajusta-se automaticamente às características do dataset
4. **Reutilização**: Aprende com experiências anteriores via SharedOptimizationMemory
5. **Escalabilidade**: Suporte para processamento paralelo (quando habilitado)

## Configuração Padrão HPM

```python
HPMConfig(
    max_configs=16,              # Número máximo de configurações
    n_trials=3,                  # Trials por configuração
    validation_split=0.2,        # Split de validação
    use_parallel=False,          # Paralelização (desabilitado por padrão)
    use_cache=True,              # Cache inteligente
    use_progressive=True,        # Cadeia progressiva
    use_multi_teacher=False,     # Multi-teacher (desabilitado até modelos treinarem)
    use_adaptive_temperature=True # Temperature scheduling adaptativo
)
```

## Uso Recomendado

O método HPM é automaticamente selecionado para:
- Datasets grandes (>10.000 amostras)
- Muitas features (>50)
- Quando qualidade é prioritária sobre velocidade

Para forçar o uso do HPM:
```python
distiller = AutoDistiller(dataset=dataset, method='hpm')
results = distiller.run()
best_model = distiller.best_model()
```