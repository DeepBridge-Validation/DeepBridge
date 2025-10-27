# Fluxo de Inicialização do HPM-KD no AutoDistiller

## Processo de Detecção Automática e Inicialização

Este documento detalha o processo de inicialização quando o AutoDistiller é criado com `method='hpm'` ou quando o sistema detecta automaticamente que HPM é a melhor escolha.

## Diagrama de Fluxo - Fluxograma

```mermaid
flowchart TD
    Start([Usuário cria AutoDistiller]) --> Input[/"dataset: DBDataset<br/>method: str<br/>outros parâmetros"/]

    Input --> CheckMethod{method == ?}

    CheckMethod -->|"'auto'"| AutoDetect[Chama _choose_best_method]
    CheckMethod -->|"'hpm'"| DirectHPM[Usa HPM diretamente]
    CheckMethod -->|"'legacy'"| UseLegacy[Inicializa modo Legacy]
    CheckMethod -->|"'hybrid'"| UseHybrid[Inicializa modo Híbrido]

    AutoDetect --> GetDatasetInfo[Extrai informações:<br/>n_samples = len(dataset.X)<br/>n_features = dataset.X.shape[1]]

    GetDatasetInfo --> CheckLarge{n_samples > 10000<br/>OU<br/>n_features > 50?}

    CheckLarge -->|Sim| SetHPM[method = 'hpm']
    CheckLarge -->|Não| CheckSmall{n_samples < 1000?}

    CheckSmall -->|Sim| SetLegacy[method = 'legacy']
    CheckSmall -->|Não| SetHPM

    SetHPM --> DirectHPM
    SetLegacy --> UseLegacy

    DirectHPM --> CallInitHPM[Chama _init_hpm()]

    CallInitHPM --> CreateHPMConfig[Cria HPMConfig]

    CreateHPMConfig --> SetAdaptiveParams[Define Parâmetros Adaptativos:<br/>max_configs = 16<br/>n_trials = max(3, n_trials // 3)<br/>validation_split = 0.2]

    SetAdaptiveParams --> SetTechniqueFlags[Define Flags de Técnicas:<br/>use_cache = True<br/>use_progressive = True<br/>use_multi_teacher = False<br/>use_adaptive_temperature = True<br/>use_parallel = False]

    SetTechniqueFlags --> SetSystemConfig[Define Configs de Sistema:<br/>random_state = 42<br/>verbose = inherited<br/>cache_memory_gb = 2.0]

    SetSystemConfig --> InstantiateHPM[Instancia HPMDistiller<br/>com HPMConfig]

    InstantiateHPM --> HPMInit[HPMDistiller.__init__()]

    HPMInit --> CallInitComponents[Chama _initialize_components()]

    CallInitComponents --> CreateACM[Cria AdaptiveConfigurationManager:<br/>max_configs = 16<br/>initial_samples = 8<br/>exploration_ratio = 0.3]

    CreateACM --> CreateSOM[Cria SharedOptimizationMemory:<br/>cache_size = 100<br/>similarity_threshold = 0.8]

    CreateSOM --> CheckCache{use_cache = True?}

    CheckCache -->|Sim| CreateCache[Cria IntelligentCache:<br/>max_memory_gb = 2.0<br/>Sistema LRU]
    CheckCache -->|Não| NoCache[cache = None]

    CreateCache --> CheckProgressive{use_progressive = True?}
    NoCache --> CheckProgressive

    CheckProgressive -->|Sim| CreateChain[Cria ProgressiveDistillationChain:<br/>use_adaptive_weights = True<br/>min_improvement = 0.01]
    CheckProgressive -->|Não| NoChain[progressive_chain = None]

    CreateChain --> CheckMultiTeacher{use_multi_teacher = True?}
    NoChain --> CheckMultiTeacher

    CheckMultiTeacher -->|Sim| CreateMT[Cria AttentionWeightedMultiTeacher:<br/>attention_type = 'learned']
    CheckMultiTeacher -->|Não| NoMT[multi_teacher = None]

    CreateMT --> CheckAdaptiveTemp{use_adaptive_temperature = True?}
    NoMT --> CheckAdaptiveTemp

    CheckAdaptiveTemp -->|Sim| CreateScheduler[Cria MetaTemperatureScheduler:<br/>initial_temperature = 3.0]
    CheckAdaptiveTemp -->|Não| NoScheduler[temp_scheduler = None]

    CreateScheduler --> CheckParallel{use_parallel = True?}
    NoScheduler --> CheckParallel

    CheckParallel -->|Sim| CreatePipeline[Cria ParallelDistillationPipeline:<br/>n_workers = auto<br/>enable_caching = True]
    CheckParallel -->|Não| NoPipeline[pipeline = None]

    CreatePipeline --> CreateMetrics[Cria Metrics Calculator:<br/>Classification()]
    NoPipeline --> CreateMetrics

    CreateMetrics --> LogSuccess[Log: "HPM components initialized"]

    LogSuccess --> SetCompatibility[Configura Compatibilidade]

    SetCompatibility --> CreateDistConfig[Cria DistillationConfig<br/>para compatibilidade<br/>com API legacy]

    CreateDistConfig --> SetAttributes[Define atributos:<br/>self.config = DistillationConfig<br/>self.hpm_distiller = HPMDistiller<br/>self.experiment_runner = None<br/>self.metrics_evaluator = None<br/>self.results_df = None]

    SetAttributes --> InitComplete[Inicialização HPM Completa]

    InitComplete --> Ready[Sistema Pronto<br/>para chamar .run()]

    Ready --> End([Fim da Inicialização])

    UseLegacy --> LegacyInit[Inicializa componentes Legacy<br/>DistillationConfig<br/>ExperimentRunner]
    UseHybrid --> HybridInit[Inicializa ambos:<br/>Legacy + HPM]

    LegacyInit --> End
    HybridInit --> End
```

## Fluxo Detalhado de Inicialização

### 1. **Entrada e Decisão Inicial**

```python
distiller = AutoDistiller(
    dataset=dataset_experiment,
    method='hpm'  # ou 'auto' para detecção automática
)
```

### 2. **Processo de Detecção Automática (se method='auto')**

O método `_choose_best_method()` analisa o dataset:

```python
def _choose_best_method(self, dataset: DBDataset) -> str:
    n_samples = len(dataset.X)
    n_features = dataset.X.shape[1]

    # Decisão baseada em heurísticas
    if n_samples > 10000 or n_features > 50:
        return 'hpm'  # Dataset grande ou complexo
    elif n_samples < 1000:
        return 'legacy'  # Dataset pequeno
    else:
        return 'hpm'  # Dataset médio, prioriza qualidade
```

### 3. **Criação da Configuração HPMConfig**

Quando HPM é selecionado, o sistema cria uma configuração otimizada:

```python
hpm_config = HPMConfig(
    # Configurações de busca
    max_configs=16,              # Limite de configurações a testar
    n_trials=max(3, n_trials//3), # Reduzido com warm start

    # Técnicas habilitadas
    use_progressive=True,         # Cadeia progressiva
    use_cache=True,              # Cache inteligente
    use_multi_teacher=False,     # Desabilitado inicialmente
    use_adaptive_temperature=True, # Temperature adaptativo
    use_parallel=False,          # Evita problemas de serialização

    # Parâmetros de sistema
    validation_split=0.2,
    random_state=42,
    verbose=verbose,
    cache_memory_gb=2.0
)
```

### 4. **Instanciação do HPMDistiller**

Com a configuração criada, o HPMDistiller é instanciado:

```python
self.hpm_distiller = HPMDistiller(config=hpm_config)
```

### 5. **Inicialização dos Componentes Especializados**

O HPMDistiller chama `_initialize_components()` que cria:

#### 5.1 **Gerenciamento Adaptativo**
- **AdaptiveConfigurationManager**: Seleciona configurações inteligentemente
- **SharedOptimizationMemory**: Mantém histórico para reutilização

#### 5.2 **Sistema de Cache**
- **IntelligentCache**: Cache LRU com limite de memória
- Evita recálculo de métricas e predições

#### 5.3 **Técnicas de Distilação**
- **ProgressiveDistillationChain**: Treino incremental de modelos
- **AttentionWeightedMultiTeacher**: Ensemble com atenção (preparado)
- **MetaTemperatureScheduler**: Ajuste dinâmico de temperatura

#### 5.4 **Pipeline de Execução**
- **ParallelDistillationPipeline**: Para processamento paralelo (desabilitado)
- **Metrics Calculator**: Avaliação consistente de modelos

### 6. **Configuração de Compatibilidade**

Para manter compatibilidade com a API do AutoDistiller:

```python
# Cria DistillationConfig para compatibilidade
self.config = DistillationConfig(
    output_dir=output_dir,
    test_size=test_size,
    random_state=random_state,
    n_trials=n_trials,
    validation_split=validation_split,
    verbose=verbose
)

# Atributos de compatibilidade
self.experiment_runner = None  # HPM gerencia internamente
self.metrics_evaluator = None
self.results_df = None
```

## Decisões de Design

### Por que HPM é escolhido automaticamente?

1. **Datasets Grandes (>10k amostras)**
   - HPM é mais eficiente com seleção adaptativa
   - Cache reduz recálculos significativamente

2. **Muitas Features (>50)**
   - HPM lida melhor com alta dimensionalidade
   - Progressive chain ajuda na convergência

3. **Datasets Médios (1k-10k amostras)**
   - Prioriza qualidade sobre velocidade
   - Técnicas avançadas melhoram resultados

### Configurações Conservadoras

- **max_configs=16**: Reduzido para evitar overfitting
- **use_parallel=False**: Evita problemas de pickle/serialização
- **use_multi_teacher=False**: Habilitado só após treino bem-sucedido

### Otimizações Automáticas

- **Warm Start**: Usa SharedOptimizationMemory para reutilizar hiperparâmetros
- **n_trials reduzido**: Com warm start, menos trials são necessários
- **Cache habilitado**: Melhora performance em iterações múltiplas

## Estado Final Após Inicialização

Após a inicialização completa:

1. **HPMDistiller está pronto** com todos os componentes inicializados
2. **AutoDistiller mantém compatibilidade** com API existente
3. **Sistema está preparado** para chamar `.run()` e executar distilação
4. **Componentes especializados** aguardam dados para processamento

## Exemplo de Uso

```python
# Criação com detecção automática
distiller = AutoDistiller(
    dataset=dataset,
    method='auto'  # Sistema decide baseado no dataset
)

# Ou forçar HPM
distiller = AutoDistiller(
    dataset=dataset,
    method='hpm'  # Força uso do HPM
)

# Após inicialização, executar distilação
results = distiller.run()
best_model = distiller.best_model()
```