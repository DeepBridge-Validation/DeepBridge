# Análise Profunda e Otimização dos Testes do DeepBridge

**Data da Análise**: 30 de Outubro de 2025
**Log Analisado**: `individual_tests_execution_20251030_104004.log`
**Tempo Total de Execução**: 512.77s (8.55 minutos)

---

## 📊 Sumário Executivo

Os testes individuais do DeepBridge estão levando **8.55 minutos** para executar com apenas **10% dos dados (1000 amostras)**. Os principais gargalos identificados são:

1. **Criação de Experimento de Resiliência**: 202.75s (39.5% do tempo total) ⚠️ **CRÍTICO**
2. **Execução do Teste de Incerteza**: 216.40s (42.2% do tempo total) ⚠️ **CRÍTICO**
3. **Criação de Experimento de Robustez**: 44.23s (8.6% do tempo total)
4. **Criação de Experimento de Incerteza**: 38.82s (7.6% do tempo total)

**Juntos, esses dois gargalos críticos representam 81.7% do tempo total de execução.**

---

## 🔍 Análise Detalhada dos Gargalos

### 1. GARGALO CRÍTICO #1: Criação do Experimento de Resiliência (202.75s)

#### Análise do Código

**Arquivo**: `deepbridge/validation/wrappers/resilience_suite.py`

**Problema Identificado** (linhas 29-140):
```python
def _get_config_templates(self):
    central_configs = {
        config_name: get_test_config(TestType.RESILIENCE.value, config_name)
        for config_name in [ConfigName.QUICK.value, ConfigName.MEDIUM.value, ConfigName.FULL.value]
    }

    # Transform the format to match what the resilience suite expects
    test_configs = {}
    for config_name, config in central_configs.items():
        tests = []
        drift_types = config.get('drift_types', [])
        drift_intensities = config.get('drift_intensities', [])

        # Create test configurations based on drift types and intensities
        for drift_type in drift_types:
            for intensity in drift_intensities:
                # Para CADA combinação de drift_type e intensity, cria um teste
                tests.append({...})
```

**Por que é Lento**:
- Para configuração "full", há múltiplos `drift_types` e `drift_intensities`
- Cada combinação gera um teste individual
- Cada teste no método `run()` (linha 1566-1696) executa:
  - `evaluate_distribution_shift()` - análise completa de shift de distribuição
  - `evaluate_worst_sample()` - identifica piores amostras
  - `evaluate_worst_cluster()` - clustering com K-means (linha 906)
  - `evaluate_outer_sample()` - detecção de outliers com IsolationForest (linha 1134-1140)
  - `evaluate_hard_sample()` - análise de discordância entre modelos

**Tempo Medido**: 202.75s para criar o experimento (antes mesmo de executar!)

#### Causa Raiz

A **criação do experimento** está demorando porque o construtor da classe `Experiment` já executa trabalho pesado:

**Arquivo**: `deepbridge/core/experiment/experiment.py` (linhas 100-177)

```python
def __init__(self, dataset, experiment_type, ...):
    # ...
    # Linha 166: Inicializa componentes
    self._initialize_components(dataset, test_size, random_state)
        # Linha 46: Prepara dados
        # Linha 52: create_alternative_models() - TREINA MÚLTIPLOS MODELOS! ⚠️

    # Linha 169: Inicializa test runner
    self._initialize_test_runner()

    # Linha 176: Calcula métricas iniciais
    self._process_initial_metrics()
        # Linha 74: run_initial_tests() - EXECUTA TESTES INICIAIS! ⚠️
```

**O problema**: No script `run_individual_tests.py`, um novo `Experiment` é criado para **cada tipo de teste**:
```python
# Linha 127-131 em run_individual_tests.py
experimento = Experiment(
    dataset=dataset_complete,
    experiment_type="binary_classification",
    tests=[test_type]  # Criado 3 vezes: robustness, uncertainty, resilience
)
```

Isso significa que:
1. **Modelos alternativos são treinados 3 vezes** (uma vez por teste)
2. **Testes iniciais são executados 3 vezes**
3. **Preparação de dados é feita 3 vezes**

---

### 2. GARGALO CRÍTICO #2: Execução do Teste de Incerteza (216.40s)

#### Análise do Código

**Arquivo**: `deepbridge/validation/wrappers/uncertainty_suite.py`

**Problema Identificado** (linhas 843-1053):

```python
class CRQR:
    def fit(self, X, y):
        # Linha 896-906: Divide dados em train/calib/test
        X_train, X_temp, y_train, y_temp = train_test_split(...)
        X_calib, X_test, y_calib, y_test = train_test_split(...)

        # Linha 914: Treina modelo base
        self.base_model.fit(X_train, y_train)

        # Linha 948-949: Treina DOIS modelos de regressão quantil
        self.quantile_model_lower.fit(X_train, residuals)
        self.quantile_model_upper.fit(X_train, residuals)
```

**Por que é Lento**:
1. **Treina 3 modelos por iteração**:
   - 1 modelo base (HistGradientBoostingRegressor)
   - 1 modelo quantil inferior (GradientBoostingRegressor)
   - 1 modelo quantil superior (GradientBoostingRegressor)

2. **Para cada feature testada** (linhas 362-374 em `uncertainty_suite.py`):
   ```python
   for feature in features_to_test:
       # Treina NOVAMENTE os 3 modelos para cada feature! ⚠️
       feature_result = self.evaluate_uncertainty(method, params, feature=feature)
   ```

3. **GradientBoostingRegressor é intrinsecamente lento**:
   - Treina árvores de decisão iterativamente
   - Cada árvore depende da anterior (não paralelizável)

**Tempo Medido**: 216.40s (3.61 minutos) apenas para executar o teste

---

### 3. Criação do Experimento de Robustez (44.23s)

**Arquivo**: `deepbridge/validation/wrappers/robustness_suite.py`

**Problema Identificado**:
- Menos crítico que os anteriores, mas ainda significativo
- Linha 52 em `experiment.py`: Criação de modelos alternativos executada novamente
- Linha 255-353 em `robustness_suite.py`: Loop de perturbações pode ser otimizado

**Tempo Medido**: 44.23s

---

### 4. Criação do Experimento de Incerteza (38.82s)

**Mesmo problema da criação de experimento de Resiliência**: Modelos alternativos sendo treinados novamente.

**Tempo Medido**: 38.82s

---

## 🚀 Sugestões de Otimização

### PRIORIDADE 1: Reutilizar Experimento Base (Ganho Estimado: 50-60%)

#### Problema
Atualmente, um novo `Experiment` é criado para cada teste:
```python
# 3 chamadas separadas = 3x overhead
experimento_robustness = Experiment(dataset, tests=["robustness"])
experimento_uncertainty = Experiment(dataset, tests=["uncertainty"])
experimento_resilience = Experiment(dataset, tests=["resilience"])
```

#### Solução Proposta

**Modificar `run_individual_tests.py`** para criar um único experimento:

```python
# Criar experimento ÚNICO com TODOS os testes
experimento = Experiment(
    dataset=dataset_complete,
    experiment_type="binary_classification",
    tests=["robustness", "uncertainty", "resilience"]  # Todos juntos!
)

# Executar cada teste individualmente usando o mesmo experimento
for test_type, test_name in test_configs:
    timings = executar_teste_individual_otimizado(
        experimento=experimento,  # Reutilizar o mesmo
        test_type=test_type,
        test_name=test_name,
        results_path=results_path
    )
```

**Implementação**:
```python
def executar_teste_individual_otimizado(experimento, test_type, test_name, results_path):
    """
    Executa um teste usando um experimento já criado (SEM recriá-lo).
    """
    print_section(f"EXECUTANDO TESTE: {test_name.upper()}")
    logger.info(f"Iniciando teste: {test_name}")

    timings = {}

    # NÃO cria novo experimento - usa o existente
    # APENAS executa o teste
    start_time_run = time.time()
    results = experimento.run_tests("full", tests=[test_type])
    timings['executar_teste'] = time.time() - start_time_run

    # Salvar resultados...
    # (resto do código permanece igual)

    return timings
```

**Ganho Estimado**:
- Eliminação de 2 criações de experimento = ~280s economizados
- Novo tempo total: ~230s (de 512s) = **55% de redução**

---

### PRIORIDADE 2: Paralelização de Testes (Ganho Estimado: 60-70%)

#### Problema
Testes são executados **sequencialmente**, um de cada vez.

#### Solução Proposta

Usar `concurrent.futures` ou `multiprocessing` para executar testes em paralelo:

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

def executar_teste_parallel(args):
    """Função auxiliar para execução paralela"""
    dataset, test_type, test_name, results_path = args
    return executar_teste_individual(dataset, test_type, test_name, results_path)

def executar_testes_individuais(artefatos_path, results_path, data_path, sample_frac=0.1):
    # ... preparação inicial ...

    test_configs = [
        ("robustness", "Robustez"),
        ("uncertainty", "Incerteza"),
        ("resilience", "Resiliência")
    ]

    # Determinar número de workers
    n_workers = min(3, multiprocessing.cpu_count())  # Máximo 3 testes

    # Preparar argumentos para cada teste
    test_args = [
        (dataset_complete, test_type, test_name, results_path)
        for test_type, test_name in test_configs
    ]

    # Executar testes em paralelo
    test_timings = {}
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_test = {
            executor.submit(executar_teste_parallel, args): args[1]
            for args in test_args
        }

        for future in as_completed(future_to_test):
            test_type = future_to_test[future]
            try:
                timings = future.result()
                test_timings[test_type] = timings
            except Exception as e:
                logger.error(f"Erro no teste {test_type}: {str(e)}")
                test_timings[test_type] = None

    return test_timings
```

**Ganho Estimado** (com 3 cores):
- Tempo do teste mais longo (Uncertainty): ~216s
- Tempo total paralelo: ~216s (vs 512s sequencial)
- **58% de redução no tempo total**

**Combinado com PRIORIDADE 1**:
- Tempo total: ~150-180s
- **65-70% de redução total**

---

### PRIORIDADE 3: Otimizar CRQR com Cache de Modelos (Ganho Estimado: 70%)

#### Problema
Para **cada feature**, o CRQR treina 3 modelos do zero (216s / feature).

#### Solução Proposta

**Modificar `uncertainty_suite.py`** para cachear modelos já treinados:

```python
class UncertaintySuite:
    def __init__(self, ...):
        # ...
        self._model_cache = {}  # Cache de modelos treinados

    def evaluate_uncertainty(self, method: str, params: Dict, feature=None):
        # ...

        if method == 'crqr':
            alpha = params.get('alpha', 0.1)
            test_size = params.get('test_size', 0.3)
            calib_ratio = params.get('calib_ratio', 1/3)

            # Chave de cache baseada nos parâmetros
            cache_key = (alpha, test_size, calib_ratio, feature is None)

            # Verificar cache
            if cache_key in self._model_cache and feature is not None:
                # Reutilizar modelo existente para análise de features
                model = self._model_cache[cache_key]

                # Apenas calcular importância da feature sem retreinar
                feature_importance = self._calculate_feature_importance_fast(
                    model, X, y, feature
                )

                # Retornar resultados usando modelo cacheado
                return {
                    'method': 'crqr',
                    'alpha': alpha,
                    'feature_importance': {feature: feature_importance},
                    'from_cache': True  # Indicador de cache
                }

            # Se não está em cache, criar e cachear
            model = self._create_crqr_model(alpha, test_size, calib_ratio)
            model.fit(X, y)

            if feature is None:
                # Cachear apenas o modelo geral (sem feature específica)
                self._model_cache[cache_key] = model

            # ... resto da lógica ...

    def _calculate_feature_importance_fast(self, model, X, y, feature):
        """
        Calcula importância de feature SEM retreinar modelos.
        Usa análise de sensibilidade ou permutação.
        """
        from sklearn.inspection import permutation_importance

        # Usar permutation importance (muito mais rápido que retreinar)
        result = permutation_importance(
            model.base_model, X, y,
            n_repeats=5,
            random_state=42,
            n_jobs=-1
        )

        # Encontrar índice da feature
        feature_idx = X.columns.get_loc(feature)
        importance = result.importances_mean[feature_idx]

        return abs(importance)
```

**Ganho Estimado**:
- Sem retreinamento: ~65s (de 216s)
- **70% de redução no tempo do teste de incerteza**

---

### PRIORIDADE 4: Reduzir Configurações do Teste de Resiliência (Ganho Estimado: 50-70%)

#### Problema
A configuração "full" do teste de resiliência gera **dezenas de testes** devido a todas as combinações de:
- drift_types: ['covariate', 'concept', 'label', 'distribution', 'statistical']
- drift_intensities: [0.1, 0.2, 0.3, 0.4, 0.5]
- test_scenarios: ['worst_sample', 'worst_cluster', 'outer_sample', 'hard_sample']

**Total de testes**: 5 × 5 × 4 = 100+ testes individuais

#### Solução Proposta

**Opção 1: Configuração "full" Mais Inteligente**

Modificar `parameter_standards.py` para reduzir combinações redundantes:

```python
# Em vez de todas as combinações (5×5=25 testes)
drift_types = ['covariate', 'concept', 'label', 'distribution', 'statistical']
drift_intensities = [0.1, 0.2, 0.3, 0.4, 0.5]

# Usar amostragem estratégica (apenas 9 testes)
test_combinations = [
    ('covariate', 0.1),   # Baixa intensidade
    ('covariate', 0.3),   # Média intensidade
    ('covariate', 0.5),   # Alta intensidade
    ('concept', 0.2),
    ('label', 0.2),
    ('distribution', 0.3),
    ('statistical', 0.3),
    ('covariate', 0.4),   # Combinação adicional para drift comum
    ('concept', 0.4),
]
```

**Opção 2: Modo "Adaptive Testing"**

Implementar teste adaptativo que executa apenas combinações relevantes:

```python
class ResilienceSuite:
    def run_adaptive(self):
        """
        Executa testes de forma adaptativa:
        1. Testa com intensidade baixa (0.1)
        2. Se impacto > threshold, testa intensidades maiores
        3. Caso contrário, pula para próximo drift_type
        """
        results = {}

        for drift_type in drift_types:
            # Sempre testa intensidade baixa
            low_result = self.evaluate_distribution_shift(drift_type, 0.1)

            if low_result['impact'] > 0.1:  # Threshold configurável
                # Se impacto significativo, testa intensidades maiores
                med_result = self.evaluate_distribution_shift(drift_type, 0.3)

                if med_result['impact'] > 0.2:
                    # Se impacto alto, testa intensidade máxima
                    high_result = self.evaluate_distribution_shift(drift_type, 0.5)
            else:
                # Se impacto baixo, pula para próximo drift_type
                logger.info(f"Impacto baixo para {drift_type}, pulando intensidades maiores")

        return results
```

**Ganho Estimado**:
- Redução de 100+ testes para 20-30 testes
- Tempo: ~60-80s (de 202s)
- **60-70% de redução**

---

### PRIORIDADE 5: Usar Algoritmos Mais Rápidos (Ganho Estimado: 30-40%)

#### Problema
`GradientBoostingRegressor` é lento por natureza (árvores sequenciais).

#### Solução Proposta

**Substituir por HistGradientBoostingRegressor** (nativo no scikit-learn >= 1.0):

```python
class CRQR:
    def fit(self, X, y):
        # Antes (lento):
        # from sklearn.ensemble import GradientBoostingRegressor
        # self.quantile_model_lower = GradientBoostingRegressor(...)

        # Depois (rápido):
        from sklearn.ensemble import HistGradientBoostingRegressor

        self.quantile_model_lower = HistGradientBoostingRegressor(
            loss='quantile',
            quantile=self.alpha/2,
            max_depth=5,
            max_iter=100,  # Limitar iterações para velocidade
            early_stopping=True,  # Parar quando não melhora mais
            random_state=self.random_state
        )

        self.quantile_model_upper = HistGradientBoostingRegressor(
            loss='quantile',
            quantile=1-self.alpha/2,
            max_depth=5,
            max_iter=100,
            early_stopping=True,
            random_state=self.random_state
        )
```

**Alternativa: LightGBM** (ainda mais rápido):

```python
try:
    import lightgbm as lgb

    self.quantile_model_lower = lgb.LGBMRegressor(
        objective='quantile',
        alpha=self.alpha/2,
        max_depth=5,
        n_estimators=100,
        n_jobs=-1,  # Paralelização automática
        random_state=self.random_state
    )
except ImportError:
    # Fallback para HistGradientBoostingRegressor
    pass
```

**Ganho Estimado**:
- HistGradientBoostingRegressor: ~150s (de 216s) = 30% mais rápido
- LightGBM: ~130s (de 216s) = 40% mais rápido

---

### PRIORIDADE 6: Lazy Evaluation e Data Sharing (Ganho Estimado: 20-30%)

#### Problema
Dados são copiados múltiplas vezes desnecessariamente.

#### Solução Proposta

**1. Usar views do Pandas em vez de cópias**:

```python
# Antes (faz cópia):
X_subset = X[feature_columns].copy()

# Depois (view sem cópia):
X_subset = X[feature_columns]  # Remove .copy()
# OU
X_subset = X.loc[:, feature_columns]  # View explícita
```

**2. Compartilhar dados entre testes**:

```python
class Experiment:
    def __init__(self, ...):
        # ...
        # Criar dados compartilhados em memória
        self._shared_data = {
            'X_train': self.X_train,
            'X_test': self.X_test,
            'y_train': self.y_train,
            'y_test': self.y_test
        }

    def run_tests(self, config, tests=None):
        # Passar referências aos dados (não cópias)
        for test_type in tests:
            test_runner = self._get_test_runner(test_type)
            test_runner.set_shared_data(self._shared_data)  # Compartilha
            results = test_runner.run(config)
```

**Ganho Estimado**: 20-30% de redução em overhead de memória e cópias

---

### PRIORIDADE 7: Implementar Progressive Testing (Ganho Estimado: 40-60% para exploração)

#### Conceito
Executar testes de forma progressiva: quick → medium → full (apenas se necessário).

#### Implementação

```python
def executar_testes_progressivos(dataset, results_path, max_time_seconds=300):
    """
    Executa testes progressivamente até atingir limite de tempo.
    """
    configs = ['quick', 'medium', 'full']
    results = {}

    start_time = time.time()

    for config in configs:
        elapsed = time.time() - start_time
        if elapsed > max_time_seconds:
            logger.info(f"Limite de tempo atingido, parando em config '{config}'")
            break

        logger.info(f"Executando config '{config}'...")
        experimento = Experiment(dataset, tests=["robustness", "uncertainty", "resilience"])

        config_results = experimento.run_tests(config)
        results[config] = config_results

        # Análise adaptativa: se resultados quick são bons, pular full
        if config == 'quick' and _resultados_satisfatorios(config_results):
            logger.info("Resultados satisfatórios em 'quick', pulando 'medium' e 'full'")
            break

    return results

def _resultados_satisfatorios(results):
    """Determina se resultados quick são suficientes"""
    # Exemplo: se todos os testes têm baixo impacto
    avg_impact = np.mean([
        results.get('robustness', {}).get('avg_overall_impact', 1.0),
        results.get('uncertainty', {}).get('avg_coverage_error', 1.0),
        results.get('resilience', {}).get('resilience_score', 0.0)
    ])

    return avg_impact < 0.15  # Threshold configurável
```

**Ganho Estimado**:
- Para casos exploratórios: 40-60% de economia
- Para validação completa: Nenhum ganho (ainda executa full)

---

## 📈 Resumo de Ganhos Esperados

### Aplicando Todas as Otimizações

| Otimização | Ganho Individual | Tempo Reduzido |
|------------|------------------|----------------|
| **Baseline** | - | 512.77s |
| 1. Reutilizar Experimento | 55% | ~230s |
| 2. Paralelização | 65% (cumulativo) | ~180s |
| 3. Cache CRQR | 70% no uncertainty | ~150s |
| 4. Reduzir Config Resilience | 60% no resilience | ~120s |
| 5. Algoritmos Rápidos | 30% adicional | ~100s |
| 6. Lazy Evaluation | 20% adicional | ~80s |
| 7. Progressive Testing | Variável | ~50-80s |

### **Tempo Final Estimado: 80-100 segundos (~1.5 minutos)**

**Redução Total: 80-85% do tempo original**

---

## 🎯 Plano de Implementação Recomendado

### Fase 1: Ganhos Rápidos (1-2 dias)
1. **Reutilizar Experimento Base** (PRIORIDADE 1)
   - Modificar `run_individual_tests.py`
   - Ganho: 55%

2. **Usar HistGradientBoostingRegressor** (PRIORIDADE 5)
   - Modificar `uncertainty_suite.py` (linha 923-945)
   - Ganho adicional: 30%

**Ganho Fase 1**: ~70% de redução (tempo: 150-180s)

### Fase 2: Otimizações Médias (3-5 dias)
3. **Cache de Modelos CRQR** (PRIORIDADE 3)
   - Modificar `uncertainty_suite.py`
   - Adicionar `_model_cache` e `_calculate_feature_importance_fast()`
   - Ganho adicional: 40-50%

4. **Reduzir Configurações Resilience** (PRIORIDADE 4)
   - Modificar `parameter_standards.py`
   - Implementar amostragem estratégica
   - Ganho adicional: 30-40%

**Ganho Fase 2**: ~80% de redução total (tempo: 80-100s)

### Fase 3: Otimizações Avançadas (1 semana)
5. **Paralelização** (PRIORIDADE 2)
   - Refatorar `run_individual_tests.py`
   - Implementar `ProcessPoolExecutor`
   - Ganho adicional: 20-30% (com overhead de paralelização)

6. **Lazy Evaluation** (PRIORIDADE 6)
   - Revisar uso de `.copy()` em todo o código
   - Implementar shared data structures
   - Ganho adicional: 10-15%

**Ganho Fase 3**: ~85% de redução total (tempo: 60-80s)

---

## 🔧 Código de Referência: Implementação Completa de PRIORIDADE 1

```python
#!/usr/bin/env python3
"""
Pipeline de Testes Individuais OTIMIZADO com Reutilização de Experimento
"""

import argparse
import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split

from deepbridge.core.db_data import DBDataset
from deepbridge.core.experiment import Experiment

# Logger global
logger = None


def executar_teste_individual_otimizado(experimento, test_type, test_name, results_path):
    """
    Executa um teste individual usando um experimento JÁ CRIADO (reutiliza).

    Esta função NÃO cria um novo experimento, apenas executa o teste específico.

    Args:
        experimento: Objeto Experiment já inicializado
        test_type: Tipo do teste (robustness, uncertainty, resilience)
        test_name: Nome do teste para exibição
        results_path: Caminho para salvar os resultados

    Returns:
        dict: Dicionário com tempos de cada etapa
    """
    print_section(f"EXECUTANDO TESTE: {test_name.upper()}")
    logger.info(f"Iniciando teste: {test_name}")
    logger.debug(f"Tipo: {test_type}")

    timings = {}

    # ========== ETAPA 1: Executar o teste ==========
    # NÃO cria experimento - apenas executa!
    print(f"\n🧪 Executando teste de {test_name}...")
    print("⏳ Aguarde...\n")

    logger.info(f"[{test_type}] Iniciando execução do teste...")
    start_time_run = time.time()

    try:
        # Executa apenas este teste específico
        results = experimento.run_tests("full", tests=[test_type])
        timings['executar_teste'] = time.time() - start_time_run

        logger.info(f"[{test_type}] Teste concluído com sucesso")
        logger.info(f"[{test_type}] ⏱️  Tempo de execução do teste: {timings['executar_teste']:.2f}s")
        print(f"✅ Teste de {test_name} concluído!")
        print(f"   ⏱️  Execução do teste: {timings['executar_teste']:.2f}s")

    except Exception as e:
        timings['executar_teste'] = time.time() - start_time_run
        logger.error(f"[{test_type}] Erro durante execução: {str(e)}", exc_info=True)
        raise

    # ========== ETAPA 2: Salvar resultados ==========
    print(f"\n💾 Salvando resultados de {test_name}...")
    logger.info(f"[{test_type}] Salvando resultados...")

    report_path = os.path.join(results_path, f'report_{test_type}_individual.html')
    json_path = os.path.join(results_path, f'{test_type}_results_individual.json')

    start_time_save = time.time()
    try:
        # Salvar HTML
        start_time_html = time.time()
        results.save_html(test_type, report_path, report_type="interactive")
        timings['salvar_html'] = time.time() - start_time_html
        logger.debug(f"[{test_type}] HTML salvo: {report_path}")

        # Salvar JSON
        start_time_json = time.time()
        results.save_json(test_type, json_path)
        timings['salvar_json'] = time.time() - start_time_json
        logger.debug(f"[{test_type}] JSON salvo: {json_path}")

        timings['salvar_total'] = time.time() - start_time_save

        if os.path.exists(report_path):
            file_size = os.path.getsize(report_path)
            print(f"  ✅ HTML: {file_size:,} bytes")

        if os.path.exists(json_path):
            json_size = os.path.getsize(json_path)
            print(f"  ✅ JSON: {json_size:,} bytes")

        print(f"  ⏱️  Tempo total de salvamento: {timings['salvar_total']:.2f}s")
        logger.info(f"[{test_type}] ⏱️  Tempo total para salvar resultados: {timings['salvar_total']:.2f}s")

    except Exception as e:
        timings['salvar_total'] = time.time() - start_time_save
        logger.warning(f"[{test_type}] Erro ao salvar resultados: {str(e)}")
        print(f"  ⚠️ Erro ao salvar: {str(e)}")

    # Calcular tempo total
    timings['total'] = timings.get('executar_teste', 0) + timings.get('salvar_total', 0)

    # Resumo de tempos
    logger.info(f"[{test_type}] ===== RESUMO DE TEMPOS =====")
    logger.info(f"[{test_type}]   1. Executar teste:     {timings.get('executar_teste', 0):8.2f}s")
    logger.info(f"[{test_type}]   2. Salvar resultados:  {timings.get('salvar_total', 0):8.2f}s")
    logger.info(f"[{test_type}]   TOTAL:                 {timings['total']:8.2f}s")
    logger.info(f"[{test_type}] ============================")

    return timings


def executar_testes_individuais_otimizado(artefatos_path, results_path, data_path, sample_frac=0.1):
    """
    Executa os testes de forma individual OTIMIZADA, reutilizando um único experimento.

    OTIMIZAÇÃO PRINCIPAL: Cria apenas UM experimento para todos os testes.
    """
    print_section("TESTES INDIVIDUAIS OTIMIZADOS - EXPERIMENTO ÚNICO")
    logger.info("Iniciando execução de testes individuais otimizados")

    prep_timings = {}

    # ========== Carregar modelo e dados (igual ao original) ==========
    # [código de carregamento idêntico ao original...]

    # ========== OTIMIZAÇÃO: Criar ÚNICO experimento com TODOS os testes ==========
    logger.info("Criando experimento único para todos os testes...")
    print(f"\n🔬 Criando experimento único para todos os testes...")

    start_time_experiment = time.time()
    try:
        # ✨ MUDANÇA PRINCIPAL: Um experimento para TODOS os testes
        experimento = Experiment(
            dataset=dataset_complete,
            experiment_type="binary_classification",
            tests=["robustness", "uncertainty", "resilience"]  # Todos juntos!
        )
        prep_timings['criar_experimento'] = time.time() - start_time_experiment

        logger.info(f"Experimento único criado com sucesso")
        logger.info(f"⏱️  Tempo para criar experimento: {prep_timings['criar_experimento']:.2f}s")
        print(f"✅ Experimento único criado")
        print(f"   ⏱️  Tempo: {prep_timings['criar_experimento']:.2f}s")

    except Exception as e:
        prep_timings['criar_experimento'] = time.time() - start_time_experiment
        logger.error(f"Erro ao criar experimento: {str(e)}", exc_info=True)
        raise

    # Resumo de tempos de preparação
    prep_total = sum(prep_timings.values())
    logger.info("===== RESUMO DE TEMPOS DE PREPARAÇÃO =====")
    logger.info(f"  Criar experimento único:   {prep_timings['criar_experimento']:8.2f}s")
    logger.info(f"  TOTAL PREPARAÇÃO:          {prep_total:8.2f}s")
    logger.info("==========================================")

    # Definir testes a serem executados
    test_configs = [
        ("robustness", "Robustez"),
        ("uncertainty", "Incerteza"),
        ("resilience", "Resiliência")
    ]

    # Dicionário para armazenar tempos de cada teste
    test_timings = {}

    # ========== Executar cada teste usando o MESMO experimento ==========
    logger.info("=" * 70)
    logger.info("INICIANDO EXECUÇÃO INDIVIDUAL DOS TESTES (EXPERIMENTO REUTILIZADO)")
    logger.info("=" * 70)

    for test_type, test_name in test_configs:
        try:
            # ✨ USA O MESMO EXPERIMENTO para cada teste
            timings = executar_teste_individual_otimizado(
                experimento=experimento,  # Reutilizar!
                test_type=test_type,
                test_name=test_name,
                results_path=results_path
            )
            test_timings[test_type] = timings

        except Exception as e:
            logger.error(f"Erro no teste {test_name}: {str(e)}", exc_info=True)
            print(f"\n❌ Erro no teste {test_name}: {str(e)}")
            test_timings[test_type] = None

    # ========== RESUMO FINAL ==========
    print_section("RESUMO COMPLETO - VERSÃO OTIMIZADA")

    # Calcular totais
    test_total = sum([t.get('total', 0) for t in test_timings.values() if t is not None])
    grand_total = prep_total + test_total

    print(f"\n📋 TEMPOS:")
    print(f"   Preparação (experimento único):  {prep_total:8.2f}s")
    print(f"   Testes (reutilizando):           {test_total:8.2f}s")
    print(f"   {'='*50}")
    print(f"   TOTAL:                           {grand_total:8.2f}s ({grand_total/60:6.2f}min)")

    logger.info("=" * 70)
    logger.info(f"TEMPO TOTAL (OTIMIZADO): {grand_total:.2f}s ({grand_total/60:.2f}min)")
    logger.info("=" * 70)

    return {'preparacao': prep_timings, 'testes': test_timings}
```

---

## 📊 Métricas de Sucesso

Para validar as otimizações, monitorar:

1. **Tempo Total de Execução**
   - Meta: < 100s (de 512s) = 80% de redução

2. **Tempo por Fase**
   - Preparação: < 10s (de 0.04s - sem mudança esperada)
   - Criação de Experimento: < 10s (de 285s total)
   - Execução de Testes: < 80s (de 227s total)

3. **Uso de Memória**
   - Meta: < 2GB RAM (evitando cópias desnecessárias)

4. **Throughput**
   - Meta: Processar 10k amostras em < 5 minutos

---

## ⚠️ Considerações e Trade-offs

### Paralelização
**Prós**:
- Ganho significativo de tempo (60-70%)
- Usa melhor recursos multi-core

**Contras**:
- Overhead de processos (~10-20%)
- Maior uso de memória (3x)
- Mais complexo para debug

### Cache de Modelos
**Prós**:
- Elimina retreinamento desnecessário
- Ganho de 70% no uncertainty

**Contras**:
- Uso de memória (modelos cacheados)
- Precisa invalidar cache corretamente

### Progressive Testing
**Prós**:
- Ótimo para exploração rápida
- Reduz tempo em 40-60%

**Contras**:
- Pode perder insights de config "full"
- Requer boa heurística de stopping

---

## 🎓 Conclusão

A análise identificou que **os gargalos principais são**:
1. Recriação desnecessária de experimentos (3x overhead)
2. Treino repetitivo de modelos CRQR para cada feature
3. Excesso de combinações de testes no resilience

**Implementando as PRIORIDADES 1, 3 e 4**:
- Ganho esperado: **75-80% de redução**
- Tempo final: **~100 segundos** (de 512s)
- Esforço: **1-2 semanas**

**Com todas as otimizações**:
- Ganho esperado: **85% de redução**
- Tempo final: **~80 segundos** (de 512s)
- Esforço: **2-3 semanas**

---

**Documento gerado automaticamente em**: 30/10/2025
**Autor**: Análise Profunda do DeepBridge
**Versão**: 1.0
