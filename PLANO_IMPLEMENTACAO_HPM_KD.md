# Plano de Implementação: HPM-KD no DeepBridge
## Estratégia de Migração e Substituição da Técnica de Destilação Atual

---

## 1. Análise da Estrutura Atual

### 1.1 Arquivos Existentes
```
deepbridge/distillation/
├── base.py                    # Classes abstratas base
├── auto_distiller.py          # Orquestrador principal
├── experiment_runner.py       # Executor de experimentos
├── techniques/
│   ├── knowledge_distillation.py  # KD tradicional
│   ├── surrogate.py               # Método surrogate
│   └── ensemble.py                # Ensemble methods
└── __init__.py                    # Exports do módulo
```

### 1.2 Pontos de Integração Identificados
- **BaseDistiller**: Classe abstrata que define interface comum
- **AutoDistiller**: Ponto de entrada principal (modificar método `__init__` e `run`)
- **ExperimentRunner**: Executa experimentos (paralelizar aqui)
- **DistillationConfig**: Configuração centralizada

---

## 2. Estratégia de Migração Incremental

### 2.1 Princípios
1. **Compatibilidade Total**: Manter API existente funcionando
2. **Opt-in Progressive**: Nova técnica como opção, não obrigatória
3. **Rollback Fácil**: Poder voltar ao método anterior se necessário
4. **Testes A/B**: Comparar performance lado a lado

### 2.2 Abordagem de Feature Flag
```python
# Em settings.py
class DistillationConfig:
    def __init__(self, ..., method='auto'):
        # 'auto': escolhe melhor método baseado no dataset
        # 'legacy': usa método tradicional
        # 'hpm': usa novo HPM-KD
        # 'hybrid': usa ambos e compara
        self.distillation_method = method
```

---

## 3. Plano de Implementação Detalhado

### FASE 0: Preparação (2 dias)

#### Dia 1: Setup e Infraestrutura
```bash
# 1. Criar branch de desenvolvimento
git checkout -b feature/hpm-distillation

# 2. Criar estrutura de diretórios
mkdir -p deepbridge/distillation/techniques/hpm
touch deepbridge/distillation/techniques/hpm/__init__.py
```

#### Dia 2: Testes Base e Benchmarks
```python
# tests/benchmark_current_distillation.py
"""
Capturar métricas baseline:
- Tempo de execução
- Uso de memória
- Métricas de qualidade
"""
```

---

### FASE 1: Componentes Base HPM (5 dias)

#### Dia 3-4: Adaptive Configuration Manager
```python
# deepbridge/distillation/techniques/hpm/adaptive_config.py
from typing import List, Dict, Tuple
import optuna
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

class AdaptiveConfigurationManager:
    """
    Gerenciador inteligente de configurações usando Bayesian Optimization.
    Reduz de 64 para 16 configurações mais promissoras.
    """

    def __init__(self, max_configs: int = 16):
        self.max_configs = max_configs
        self.gp_model = GaussianProcessRegressor()
        self.performance_history = []

    def select_promising_configs(
        self,
        model_types: List,
        temperatures: List[float],
        alphas: List[float],
        initial_samples: int = 8
    ) -> List[Dict]:
        # Implementação da seleção Bayesiana
        pass
```

#### Dia 5-6: Shared Optimization Memory
```python
# deepbridge/distillation/techniques/hpm/shared_memory.py
from collections import deque
import pickle
import hashlib

class SharedOptimizationMemory:
    """
    Cache compartilhado de hiperparâmetros otimizados.
    Reduz trials de 10 para 3-5 usando warm start.
    """

    def __init__(self, cache_size: int = 100):
        self.param_cache = deque(maxlen=cache_size)
        self.similarity_threshold = 0.8

    def get_similar_config(self, model_type, context):
        # Buscar configurações similares
        pass

    def warm_start_study(self, model_type, similar_configs):
        # Criar estudo Optuna com conhecimento prévio
        pass
```

#### Dia 7: Cache System
```python
# deepbridge/distillation/techniques/hpm/cache_system.py
from functools import lru_cache
import hashlib
import numpy as np

class IntelligentCache:
    """
    Sistema de cache inteligente para eliminar cálculos redundantes.
    """

    def __init__(self, max_memory_gb: float = 2.0):
        self.teacher_cache = {}
        self.feature_cache = {}
        self.memory_limit = max_memory_gb * 1024 * 1024 * 1024

    def get_cache_key(self, data):
        # Gerar hash único para dados
        pass

    def get_or_compute(self, key, compute_fn):
        # Cache com fallback para computação
        pass
```

---

### FASE 2: Destilação Progressiva (5 dias)

#### Dia 8-9: Progressive Chain Implementation
```python
# deepbridge/distillation/techniques/hpm/progressive_chain.py
from deepbridge.utils.model_registry import ModelType
from typing import List, Tuple

class ProgressiveDistillationChain:
    """
    Cadeia de destilação progressiva: Simple → Complex.
    Reduz knowledge gap através de transferência incremental.
    """

    def __init__(self):
        self.chain = [
            ModelType.LOGISTIC_REGRESSION,
            ModelType.DECISION_TREE,
            ModelType.GBM,
            ModelType.XGB
        ]
        self.intermediate_models = []

    def train_progressive(self, X, y, teacher_probs, **kwargs):
        """
        Treina modelos progressivamente, cada um
        ensinando o próximo.
        """
        pass
```

#### Dia 10-11: Multi-Teacher System
```python
# deepbridge/distillation/techniques/hpm/multi_teacher.py
import numpy as np
from typing import List, Dict

class AttentionWeightedMultiTeacher:
    """
    Sistema multi-teacher com mecanismo de atenção adaptativo.
    """

    def __init__(self, attention_type: str = 'learned'):
        self.attention_type = attention_type
        self.attention_weights = None

    def compute_attention(self, teachers, student_state):
        """
        Calcula pesos de atenção baseados em:
        - Concordância entre teachers
        - Confiança das predições
        - Performance histórica
        """
        pass

    def weighted_fusion(self, teacher_predictions, weights):
        """
        Fusão ponderada de conhecimento dos teachers.
        """
        pass
```

#### Dia 12: Meta Temperature Scheduler
```python
# deepbridge/distillation/techniques/hpm/meta_scheduler.py
from sklearn.neural_network import MLPRegressor

class MetaTemperatureScheduler:
    """
    Agendador adaptativo de temperatura usando meta-learning.
    """

    def __init__(self):
        self.meta_model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            max_iter=1000
        )
        self.history = []

    def adaptive_temperature(
        self,
        epoch: int,
        loss_history: List[float],
        kl_divergence: float
    ) -> float:
        """
        Prediz temperatura ótima baseada no estado atual.
        """
        pass
```

---

### FASE 3: Paralelização e Otimização (4 dias)

#### Dia 13-14: Parallel Pipeline
```python
# deepbridge/distillation/techniques/hpm/parallel_pipeline.py
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import asyncio

class ParallelDistillationPipeline:
    """
    Pipeline de processamento paralelo com balanceamento de carga.
    """

    def __init__(self, n_workers: int = None):
        self.n_workers = n_workers or (cpu_count() - 1)
        self.executor = ProcessPoolExecutor(max_workers=self.n_workers)

    def train_batch_parallel(self, configs, train_fn):
        """
        Treina múltiplas configurações em paralelo.
        """
        futures = []
        for config in configs:
            future = self.executor.submit(train_fn, config)
            futures.append(future)

        results = []
        for future in as_completed(futures):
            results.append(future.result())

        return results
```

#### Dia 15-16: HPM Distiller Principal
```python
# deepbridge/distillation/techniques/hpm/hpm_distiller.py
from deepbridge.distillation.base import BaseDistiller

class HPMDistiller(BaseDistiller):
    """
    Implementação principal do HPM-KD.
    """

    def __init__(self, dataset, config):
        super().__init__()
        self.config_manager = AdaptiveConfigurationManager(config)
        self.progressive_chain = ProgressiveDistillationChain()
        self.multi_teacher = AttentionWeightedMultiTeacher()
        self.cache = IntelligentCache()
        self.pipeline = ParallelDistillationPipeline()
        self.shared_memory = SharedOptimizationMemory()
        self.temp_scheduler = MetaTemperatureScheduler()

    def fit(self, X, y, teacher_probs=None):
        """
        Executa o processo completo de destilação HPM.
        """
        # 1. Selecionar configurações promissoras
        configs = self.config_manager.select_promising_configs(...)

        # 2. Executar destilação progressiva paralela
        results = self.pipeline.train_batch_parallel(configs, ...)

        # 3. Criar ensemble multi-teacher
        final_model = self.multi_teacher.create_ensemble(...)

        return self

    def predict(self, X):
        """Predição usando modelo destilado."""
        pass
```

---

### FASE 4: Integração com AutoDistiller (3 dias)

#### Dia 17: Modificar AutoDistiller
```python
# deepbridge/distillation/auto_distiller.py (modificado)
from deepbridge.distillation.techniques.hpm import HPMDistiller

class AutoDistiller:
    def __init__(self, dataset, method='auto', **kwargs):
        self.dataset = dataset
        self.method = method

        if method == 'auto':
            # Decisão automática baseada no dataset
            self.method = self._choose_best_method(dataset)

        if self.method == 'hpm':
            self.engine = HPMDistiller(dataset, **kwargs)
        elif self.method == 'hybrid':
            # Executar ambos e comparar
            self.engine = HybridDistiller(dataset, **kwargs)
        else:  # legacy
            self.engine = LegacyDistiller(dataset, **kwargs)

    def _choose_best_method(self, dataset):
        """
        Heurística para escolher método baseado em:
        - Tamanho do dataset
        - Número de features
        - Complexidade do problema
        """
        n_samples = len(dataset.X)
        n_features = dataset.X.shape[1]

        if n_samples > 10000 or n_features > 50:
            return 'hpm'  # HPM é melhor para datasets grandes
        return 'legacy'
```

#### Dia 18: Modificar ExperimentRunner
```python
# deepbridge/distillation/experiment_runner.py (modificado)
class ExperimentRunner:
    def run_experiments(self, use_probabilities=True, method='auto'):
        if method == 'hpm':
            return self._run_hpm_experiments(use_probabilities)
        else:
            return self._run_legacy_experiments(use_probabilities)

    def _run_hpm_experiments(self, use_probabilities):
        """Nova implementação HPM otimizada."""
        hpm_distiller = HPMDistiller(
            self.dataset,
            self.config
        )
        return hpm_distiller.progressive_distill()
```

#### Dia 19: Update Configuration
```python
# deepbridge/config/settings.py (modificado)
class DistillationConfig:
    def __init__(
        self,
        # Parâmetros existentes
        ...,
        # Novos parâmetros HPM
        distillation_method: str = 'auto',
        max_configs: int = 16,
        parallel_workers: int = None,
        use_cache: bool = True,
        progressive: bool = True,
        multi_teacher: bool = True,
        adaptive_temperature: bool = True
    ):
        # Configurações HPM
        self.distillation_method = distillation_method
        self.max_configs = max_configs
        self.parallel_workers = parallel_workers
        self.use_cache = use_cache
        self.progressive = progressive
        self.multi_teacher = multi_teacher
        self.adaptive_temperature = adaptive_temperature
```

---

### FASE 5: Testes e Validação (4 dias)

#### Dia 20-21: Unit Tests
```python
# tests/test_hpm_components.py
import unittest
from deepbridge.distillation.techniques.hpm import *

class TestAdaptiveConfig(unittest.TestCase):
    def test_config_reduction(self):
        """Testa redução de 64 para 16 configs."""
        pass

class TestProgressiveChain(unittest.TestCase):
    def test_chain_training(self):
        """Testa cadeia progressiva."""
        pass

class TestMultiTeacher(unittest.TestCase):
    def test_attention_weights(self):
        """Testa cálculo de pesos de atenção."""
        pass
```

#### Dia 22: Integration Tests
```python
# tests/test_hpm_integration.py
class TestHPMIntegration(unittest.TestCase):
    def test_full_pipeline(self):
        """Testa pipeline completo HPM."""
        distiller = AutoDistiller(
            dataset,
            method='hpm',
            max_configs=8,  # Reduzido para teste
            parallel_workers=2
        )
        results = distiller.run()

        # Verificar que funciona e é mais rápido
        self.assertIsNotNone(results)
        self.assertLess(distiller.total_time, legacy_time * 0.5)
```

#### Dia 23: Performance Benchmarks
```python
# benchmarks/compare_methods.py
def benchmark_comparison():
    """
    Compara HPM vs Legacy em múltiplos datasets.
    """
    datasets = load_benchmark_datasets()

    for dataset in datasets:
        # Legacy
        start = time.time()
        legacy_results = AutoDistiller(dataset, method='legacy').run()
        legacy_time = time.time() - start

        # HPM
        start = time.time()
        hpm_results = AutoDistiller(dataset, method='hpm').run()
        hpm_time = time.time() - start

        print(f"Dataset: {dataset.name}")
        print(f"Legacy: {legacy_time:.2f}s")
        print(f"HPM: {hpm_time:.2f}s")
        print(f"Speedup: {legacy_time/hpm_time:.2f}x")
```

---

### FASE 6: Documentação e Deploy (2 dias)

#### Dia 24: Documentation
```markdown
# docs/hpm_distillation.md
## HPM-KD: Hierarchical Progressive Multi-Teacher Distillation

### Quick Start
```python
from deepbridge.distillation import AutoDistiller

# Usar novo método HPM
distiller = AutoDistiller(
    dataset,
    method='hpm',  # Ativar HPM-KD
    max_configs=16,  # Configurações a testar
    parallel_workers=4  # Paralelização
)

results = distiller.run()
```

### Comparação de Performance
| Método | Tempo | Modelos | Qualidade |
|--------|-------|---------|-----------|
| Legacy | 4h | 640 | Baseline |
| HPM | 30min | 80 | +3% accuracy |
```

#### Dia 25: Migration Guide
```markdown
# docs/migration_to_hpm.md
## Guia de Migração para HPM-KD

### Para usuários existentes
Seu código continuará funcionando sem modificações:
```python
# Código existente - continua funcionando
distiller = AutoDistiller(dataset)
```

### Para ativar HPM
```python
# Opt-in para novo método
distiller = AutoDistiller(dataset, method='hpm')
```

### Rollback se necessário
```python
# Forçar método legacy
distiller = AutoDistiller(dataset, method='legacy')
```
```

---

## 4. Checklist de Implementação

### Preparação
- [ ] Criar branch feature/hpm-distillation
- [ ] Setup estrutura de diretórios
- [ ] Criar testes benchmark baseline

### Fase 1: Componentes Base
- [ ] AdaptiveConfigurationManager
- [ ] SharedOptimizationMemory
- [ ] IntelligentCache

### Fase 2: Core HPM
- [ ] ProgressiveDistillationChain
- [ ] AttentionWeightedMultiTeacher
- [ ] MetaTemperatureScheduler

### Fase 3: Paralelização
- [ ] ParallelDistillationPipeline
- [ ] HPMDistiller principal

### Fase 4: Integração
- [ ] Modificar AutoDistiller
- [ ] Update ExperimentRunner
- [ ] Atualizar DistillationConfig

### Fase 5: Testes
- [ ] Unit tests completos
- [ ] Integration tests
- [ ] Performance benchmarks
- [ ] Comparação A/B

### Fase 6: Documentação
- [ ] Documentação técnica
- [ ] Guia de migração
- [ ] Exemplos de uso
- [ ] Release notes

### Deploy
- [ ] Code review
- [ ] Merge para develop
- [ ] Testes em staging
- [ ] Release para produção

---

## 5. Comando de Execução para Teste

```bash
# Testar implementação HPM
python -m pytest tests/test_hpm_components.py -v

# Benchmark de performance
python benchmarks/compare_methods.py

# Exemplo de uso
python examples/hpm_distillation_example.py
```

---

## 6. Monitoramento Pós-Deploy

```python
# monitoring/track_hpm_usage.py
def track_distillation_metrics():
    """
    Monitora uso e performance do HPM em produção.
    """
    metrics = {
        'method_usage': {},  # Quantos usam HPM vs Legacy
        'time_reduction': [],  # Redução de tempo média
        'quality_improvement': [],  # Melhoria de métricas
        'error_rate': {}  # Taxa de erros por método
    }
    return metrics
```

---

## 7. Rollback Plan

Se houver problemas com HPM:

```bash
# 1. Reverter config padrão
git revert <commit_hash>

# 2. Forçar uso de legacy via env var
export DEEPBRIDGE_DISTILLATION_METHOD=legacy

# 3. Hotfix se necessário
git checkout -b hotfix/disable-hpm
# Modificar default em settings.py
# distillation_method='legacy'
```

---

**Plano preparado por: Claude (Anthropic)**
**Data: 23/09/2025**
**Tempo estimado: 25 dias úteis**
**Complexidade: Alta**
**Risco: Médio (mitigado por compatibilidade)**