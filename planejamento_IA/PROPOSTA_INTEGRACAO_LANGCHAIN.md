# Proposta de Integração LangChain + DeepBridge
## Evolução para Plataforma Inteligente de Model Governance

**Versão:** 1.0
**Data:** Dezembro 2025
**Status:** Proposta para Aprovação
**Autores:** Equipe DeepBridge

---

## 📋 Sumário Executivo

Esta proposta descreve a evolução da biblioteca **DeepBridge** através da integração com **LangChain** como camada opcional de orquestração por agentes inteligentes. O objetivo é transformar a DeepBridge de uma biblioteca puramente técnica de validação de modelos em uma **plataforma autônoma de governança, validação e auditoria de modelos de Machine Learning**.

### Principais Conclusões

✅ **VIÁVEL**: Arquitetura atual da DeepBridge está altamente preparada para integração
✅ **ESTRATÉGICO**: Posiciona DeepBridge como única plataforma com agentes + validação rigorosa
✅ **FACTÍVEL**: Roadmap de 12-14 semanas para production-ready
✅ **SEGURO**: Separação clara entre decisão (LLM) e execução (DeepBridge)

### Métricas de Impacto Esperadas

- **Redução de tempo de validação**: 80-90% (40h → 5h)
- **Custo por validação com agentes**: <$5 (LLM calls)
- **Adoption rate target**: 30% em 6 meses
- **ROI estimado**: >300% para validações regulares

---

## 📊 Tabela de Conteúdos

1. [Visão Geral](#1-visão-geral)
2. [Análise da Arquitetura Atual](#2-análise-da-arquitetura-atual)
3. [Motivação Estratégica](#3-motivação-estratégica)
4. [Princípios Arquiteturais](#4-princípios-arquiteturais)
5. [Arquitetura Proposta](#5-arquitetura-proposta)
6. [Implementação Detalhada](#6-implementação-detalhada)
7. [Casos de Uso](#7-casos-de-uso)
8. [Roadmap de Implementação](#8-roadmap-de-implementação)
9. [Análise de Riscos](#9-análise-de-riscos)
10. [Métricas de Sucesso](#10-métricas-de-sucesso)
11. [Comparação: Com vs. Sem LangChain](#11-comparação-com-vs-sem-langchain)
12. [Recomendações Finais](#12-recomendações-finais)
13. [Próximos Passos](#13-próximos-passos)

---

## 1. Visão Geral

### 1.1 Contexto

A validação tradicional de modelos de ML segue um fluxo manual ou semiautomático que requer expertise técnica profunda e consome 20-80 horas por modelo:

```
Fluxo Atual:
Usuário → Script Manual → Configuração de Testes → Execução → Interpretação Manual → Relatório
```

### 1.2 Proposta

Integrar LangChain como camada inteligente mantendo DeepBridge como executor determinístico:

```
Fluxo Proposto:
Usuário → Prompt Natural → Agente LangChain → DeepBridge (Testes) → Relatórios Automatizados
```

### 1.3 Valor Agregado

Esta evolução posiciona a DeepBridge como:

- ✅ Plataforma de **Model Risk Management (MRM)** automatizado
- ✅ Motor de **validação autônoma** de modelos
- ✅ Base tecnológica para **auditoria contínua de IA**
- ✅ Interface **democratizada** (não requer expertise técnica profunda)

---

## 2. Análise da Arquitetura Atual

### 2.1 Estrutura Modular da DeepBridge

A análise técnica da biblioteca revelou uma arquitetura **altamente preparada** para extensões:

```
deepbridge/
├── core/                    # Núcleo - Experimentos e dados
│   ├── db_data.py          # DBDataset (dados + modelo)
│   ├── experiment/         # Sistema de experimentação
│   │   ├── experiment.py   # Classe Experiment (orquestrador)
│   │   ├── test_runner.py  # Executor de testes
│   │   └── managers/       # Gerenciadores especializados
│   └── base_processor.py   # Base abstrata
│
├── validation/             # Suítes de validação
│   └── wrappers/
│       ├── robustness_suite.py
│       ├── fairness_suite.py
│       ├── uncertainty_suite.py
│       ├── resilience_suite.py
│       └── hyperparameter_suite.py
│
├── distillation/           # Destilação de conhecimento
├── synthetic/              # Geração de dados sintéticos
├── metrics/                # Métricas de avaliação
└── utils/                  # Utilitários compartilhados
```

### 2.2 Padrões Arquiteturais Identificados

**Padrões de Design:**
1. **Abstract Base Classes (ABC)**: Interfaces bem definidas
2. **Factory Pattern**: Criação de modelos, testes e estratégias
3. **Strategy Pattern**: Múltiplas estratégias de teste intercambiáveis
4. **Facade Pattern**: `Experiment` e `Suites` como fachadas simplificadas
5. **Composition over Inheritance**: Modularidade e flexibilidade

**Implicações para Integração:**
- ✅ **Extensibilidade**: Fácil adicionar novos componentes via Factories
- ✅ **Separação de Responsabilidades**: Cada módulo com função clara
- ✅ **Interfaces Estáveis**: ABCs garantem contratos bem definidos
- ✅ **Testabilidade**: Componentes isolados facilitam testes

### 2.3 Pontos de Integração Identificados

**Níveis de Integração Possíveis:**

1. **Nível de Teste** (mais simples)
   - Criar `LangChainTestStrategy` implementando `TestStrategy`
   - Integrar no `TestStrategyFactory`
   - Executar via `Experiment.run_test('langchain', config)`

2. **Nível de Suite** (intermediário)
   - Criar `LangChainSuite` similar a `RobustnessSuite`
   - API familiar: `.config('quick').run()`

3. **Nível de Manager** (mais controle)
   - Criar `LangChainManager` herdando de `BaseManager`
   - Acesso direto a dados de treino/teste

4. **Nível de Agente** (orquestração) ⭐ **RECOMENDADO**
   - Nova camada acima do `Experiment`
   - LLM interpreta prompts e roteia para testes apropriados
   - DeepBridge executa testes determinísticos

### 2.4 API Pública Atual

**Uso típico da DeepBridge:**

```python
from deepbridge import DBDataset, Experiment

# 1. Criar dataset
dataset = DBDataset(
    data=df,
    target_column='target',
    features=['f1', 'f2'],
    model=trained_model
)

# 2. Criar experimento
experiment = Experiment(
    dataset=dataset,
    experiment_type='binary_classification',
    tests=['robustness', 'uncertainty', 'fairness']
)

# 3. Executar testes (requer conhecimento técnico)
results = experiment.run_tests(config_name='medium')

# 4. Gerar relatório
results.save_html('robustness', 'report.html')
```

**Observações:**
- ✅ API bem definida e documentada
- ✅ Flexível e poderosa
- ❌ Requer expertise técnica
- ❌ Workflow manual
- ❌ Interpretação subjetiva dos resultados

---

## 3. Motivação Estratégica

### 3.1 Problema Atual

A validação de modelos ML em produção enfrenta desafios críticos:

**Desafios Técnicos:**
- ⏱️ **Tempo**: 20-80 horas por modelo para validação completa
- 💰 **Custo**: $6k-$24k por validação (expertise especializada)
- 🎯 **Expertise**: Requer conhecimento profundo em ML, estatística E regulações
- 📊 **Inconsistência**: Resultados variam entre auditores
- 🔄 **Frequência**: Validações pontuais vs. monitoramento contínuo necessário

**Desafios Regulatórios:**
- 📋 **SR 11-7** (Model Risk Management): Validação independente obrigatória
- ⚖️ **EEOC/ECOA**: Compliance em fairness para hiring/lending
- 🇪🇺 **EU AI Act**: Transparência e auditabilidade
- 🎯 **Basel III/IV**: Gestão de risco de modelos em bancos

### 3.2 Oportunidade de Mercado

**Segmentos-alvo:**

| Segmento | Tamanho (US) | Pain Point Principal | Valor Proposto |
|----------|--------------|----------------------|----------------|
| **Bancos** | 5,000+ instituições | Compliance SR 11-7 | MRM automatizado |
| **Fintechs** | 10,000+ empresas | EEOC/ECOA em lending/hiring | Validação contínua |
| **Big Tech** | 500+ empresas | Escala de validação | Automação em larga escala |
| **Consultorias** | 1,000+ firmas | Auditoria para clientes | Ferramenta profissional |
| **Reguladores** | Agencies federais/estaduais | Supervisão em escala | Padronização de auditorias |

**Tamanho de Mercado Estimado:**
- **TAM (Total Addressable Market)**: $2B+ (Model Risk Management global)
- **SAM (Serviceable Available Market)**: $500M (validação automatizada)
- **SOM (Serviceable Obtainable Market)**: $50M (3 anos, 10% market share)

### 3.3 Diferenciação Competitiva

**Comparação com Alternativas:**

| Feature | Manual Audit | AIF360 | Fairlearn | **DeepBridge + LangChain** |
|---------|--------------|--------|-----------|----------------------------|
| Tempo médio | 40h | N/A | N/A | **5h** |
| Custo | $12k | Grátis | Grátis | **<$100** |
| Interface natural | ❌ | ❌ | ❌ | **✅** |
| Compliance EEOC | Manual | Parcial | Parcial | **✅ Automatizado** |
| Compliance SR 11-7 | ✅ | ❌ | ❌ | **✅ Automatizado** |
| Stress testing | Manual | ❌ | ❌ | **✅ Automatizado** |
| Interpretação técnica | Humana | ❌ | ❌ | **✅ LLM** |
| Reproducibilidade | Baixa | Alta | Alta | **Alta** |
| Continuous monitoring | ❌ | Parcial | Parcial | **✅** |

**Diferencial Único:**
> **DeepBridge será a ÚNICA plataforma que combina rigor técnico de validação ML determinística com inteligência de agentes para automação completa de Model Risk Management.**

---

## 4. Princípios Arquiteturais

### 4.1 Princípios Fundamentais

**1. LangChain como Camada Opcional**
```
❌ NÃO: Tornar DeepBridge dependente de LLMs
✅ SIM: Adicionar agentes como feature opcional
```

A DeepBridge DEVE continuar funcionando perfeitamente sem LangChain.

**2. Separação Clara de Responsabilidades**

| Componente | Responsabilidade | O que NÃO faz |
|------------|------------------|---------------|
| **LangChain/LLM** | • Interpretar prompts<br>• Rotear para testes apropriados<br>• Gerar explicações técnicas | ❌ Calcular métricas<br>❌ Executar testes<br>❌ Decidir risco sozinho |
| **DeepBridge** | • Executar testes estatísticos<br>• Calcular métricas<br>• Gerar relatórios técnicos | ❌ Interpretar linguagem natural<br>❌ Gerar narrativas |

**3. Determinismo da Validação**

Toda execução da DeepBridge deve ser:
- ✅ **Reprodutível**: Mesmas entradas → mesmas métricas
- ✅ **Versionada**: Configs, prompts, datasets rastreados
- ✅ **Auditável**: Logs completos de execução
- ✅ **Independente**: LLM não afeta resultados de testes

**4. LLM Nunca Decide Risco Sozinho**

```
Fluxo Correto:
User Prompt → LLM Interpreta → DeepBridge Executa → Métricas Determinísticas → LLM Explica
                                                      ↑
                                            Source of Truth
```

**5. Auditabilidade Completa**

Cada execução deve logar:
```json
{
  "timestamp": "2025-12-06T10:30:00Z",
  "user_prompt": "Valide robustez",
  "llm_routing": {
    "tests_selected": ["robustness"],
    "config": "medium",
    "reasoning": "..."
  },
  "deepbridge_execution": {
    "test_type": "robustness",
    "config": "medium",
    "deterministic": true,
    "results_hash": "abc123..."
  },
  "llm_interpretation": {
    "summary": "...",
    "recommendations": [...]
  },
  "costs": {
    "llm_tokens": 1500,
    "estimated_cost_usd": 0.045
  }
}
```

### 4.2 Garantias Regulatórias

**Para Compliance:**

| Requisito Regulatório | Como Garantimos |
|----------------------|-----------------|
| **Reproducibilidade** | DeepBridge executa testes, LLM apenas interpreta |
| **Independência de Validação** | Métricas calculadas deterministicamente |
| **Rastreabilidade** | Logs completos de execução + versionamento |
| **Explicabilidade** | LLM gera narrativas, mas métricas são source of truth |
| **Auditabilidade** | Executions logs exportáveis para reguladores |

---

## 5. Arquitetura Proposta

### 5.1 Visão de Alto Nível

```
┌────────────────────────────────────────────────────────────┐
│                    CAMADA DE USUÁRIO                       │
│  Interface: Linguagem Natural (prompts) ou API Clássica    │
└────────────────────┬───────────────────────────────────────┘
                     ↓
┌────────────────────────────────────────────────────────────┐
│            CAMADA DE AGENTES (NOVA - OPCIONAL)             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Validation   │  │ StressTest   │  │Explainability│     │
│  │   Agent      │  │    Agent     │  │    Agent     │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                  │                  │             │
│         └──────────────────┴──────────────────┘             │
│                            ↓                                │
│              ┌─────────────────────────┐                   │
│              │   LangChain Tools       │                   │
│              │  (RobustnessTool, etc)  │                   │
│              └────────────┬────────────┘                   │
└───────────────────────────┼────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│         CAMADA DE ORQUESTRAÇÃO (EXISTENTE)                 │
│              ┌─────────────────┐                           │
│              │   Experiment    │  ← Orquestrador Central   │
│              └────────┬────────┘                           │
│                       ↓                                     │
│         ┌─────────────────────────┐                        │
│         │     TestRunner          │                        │
│         └────────┬────────────────┘                        │
└──────────────────┼─────────────────────────────────────────┘
                   ↓
┌────────────────────────────────────────────────────────────┐
│          CAMADA DE EXECUÇÃO (EXISTENTE)                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ Robustness  │  │  Fairness   │  │ Uncertainty │        │
│  │  Manager    │  │   Manager   │  │   Manager   │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                 │                 │               │
│         └─────────────────┴─────────────────┘               │
│                           ↓                                 │
│              ┌────────────────────┐                        │
│              │   Test Strategies  │                        │
│              └─────────┬──────────┘                        │
└────────────────────────┼───────────────────────────────────┘
                         ↓
┌────────────────────────────────────────────────────────────┐
│              CAMADA DE DADOS (EXISTENTE)                   │
│                  ┌──────────────┐                          │
│                  │   DBDataset  │                          │
│                  │  (Dados +    │                          │
│                  │   Modelo)    │                          │
│                  └──────────────┘                          │
└────────────────────────────────────────────────────────────┘
                         ↓
┌────────────────────────────────────────────────────────────┐
│           CAMADA DE SAÍDA (EXISTENTE)                      │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐      │
│  │  HTML   │  │   PDF   │  │  JSON   │  │ Jupyter │      │
│  │ Reports │  │ Reports │  │  Logs   │  │  Plots  │      │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘      │
└────────────────────────────────────────────────────────────┘
```

### 5.2 Estrutura de Módulos Proposta

```python
deepbridge/
│
├── agents/                          # 🆕 NOVO MÓDULO
│   ├── __init__.py                 # Exports públicos
│   │
│   ├── base.py                     # AgentBase (classe abstrata)
│   ├── validation_agent.py         # Agente de validação geral
│   ├── stress_test_agent.py        # Agente de stress testing
│   ├── explainability_agent.py     # Agente de explainability
│   ├── comparison_agent.py         # Agente de comparação de modelos
│   │
│   ├── tools/                      # LangChain Tools
│   │   ├── __init__.py
│   │   ├── base_tool.py           # Base para tools
│   │   ├── robustness_tool.py     # Tool: robustez
│   │   ├── fairness_tool.py       # Tool: fairness
│   │   ├── uncertainty_tool.py    # Tool: incerteza
│   │   ├── resilience_tool.py     # Tool: resiliência
│   │   ├── hyperparameter_tool.py # Tool: hiperparâmetros
│   │   ├── distillation_tool.py   # Tool: destilação
│   │   └── synthetic_tool.py      # Tool: dados sintéticos
│   │
│   ├── prompts/                    # Prompt Templates
│   │   ├── __init__.py
│   │   ├── validation_prompts.py  # Prompts para validação
│   │   ├── stress_test_prompts.py # Prompts para stress testing
│   │   ├── explain_prompts.py     # Prompts para explicabilidade
│   │   └── system_prompts.py      # System prompts base
│   │
│   ├── memory/                     # Memória de execuções
│   │   ├── __init__.py
│   │   ├── execution_log.py       # Logging de execuções
│   │   ├── cost_tracker.py        # Tracking de custos LLM
│   │   └── session_manager.py     # Gerenciamento de sessões
│   │
│   └── wrappers/                   # Wrappers para compatibilidade
│       ├── __init__.py
│       ├── llm_model_wrapper.py   # Wrapper LLM → sklearn API
│       └── chain_wrapper.py       # Wrapper LangChain chains
│
├── core/                           # ✅ EXISTENTE (inalterado)
├── validation/                     # ✅ EXISTENTE (inalterado)
├── distillation/                   # ✅ EXISTENTE (inalterado)
├── synthetic/                      # ✅ EXISTENTE (inalterado)
├── metrics/                        # ✅ EXISTENTE (inalterado)
└── utils/                          # ✅ EXISTENTE (inalterado)
```

### 5.3 Fluxo de Dados Detalhado

**Exemplo: "Valide este modelo quanto à robustez e viés"**

```
1. User Input
   ↓
   Prompt: "Valide este modelo quanto à robustez e viés"

2. ValidationAgent (LangChain Layer)
   ↓
   LLM interpreta prompt:
   {
     "intent": "validation",
     "tests_required": ["robustness", "fairness"],
     "config": "medium",  # inferido do prompt
     "priority": "both"
   }

3. Tool Selection (LangChain Executor)
   ↓
   Seleciona Tools:
   - RobustnessTool
   - FairnessTool

4. RobustnessTool.run()
   ↓
   Chama: experiment.run_test('robustness', 'medium')
   ↓
   Execução DeepBridge (DETERMINÍSTICA):
   - TestRunner → RobustnessManager → RobustnessStrategy
   - Perturbações Gaussianas
   - Cálculo de métricas
   - Detecção de weak spots
   ↓
   Retorna: {
     "robustness_score": 0.82,
     "degradation": 0.12,
     "weak_spots": [("feature_x", 0.25), ...],
     "deterministic": true
   }

5. FairnessTool.run()
   ↓
   Chama: experiment.run_fairness_tests('medium')
   ↓
   Execução DeepBridge (DETERMINÍSTICA):
   - TestRunner → FairnessManager
   - Cálculo de 15 métricas
   - Verificação EEOC (80% rule)
   - Confusion matrix por grupo
   ↓
   Retorna: {
     "eeoc_compliant": false,
     "metrics": {...},
     "violations": [...],
     "deterministic": true
   }

6. LLM Synthesis (LangChain Layer)
   ↓
   LLM recebe resultados e gera:
   """
   ANÁLISE DE VALIDAÇÃO

   Executei testes de robustez e fairness no modelo.

   ROBUSTEZ: ⚠️ ATENÇÃO
   - Score: 0.82 (aceitável, mas abaixo de ideal 0.9)
   - Degradação média: 12% sob perturbações
   - Weak spot crítico: feature_x (25% de degradação)

   FAIRNESS: ❌ NÃO-COMPLIANT
   - EEOC 80% Rule: VIOLADO
   - Impact ratio: 0.72 (abaixo de 0.80)
   - Grupo afetado: [detalhes]

   RECOMENDAÇÕES PRIORIZADAS:
   1. [CRÍTICO] Mitigar violação EEOC antes de deployment
   2. [ALTO] Investigar feature_x para robustez
   3. [MÉDIO] Considerar threshold adjustment

   PRÓXIMOS PASSOS:
   - Gerar relatório técnico completo
   - Revisar features para fairness
   - Re-executar testes após mitigações
   """

7. Output to User
   ↓
   {
     "answer": "...",  # Texto acima
     "executions": [...],  # Logs de execução
     "results": DBResults,  # Objeto com resultados completos
     "metadata": {
       "deterministic": true,
       "llm_cost_usd": 0.043,
       "duration_seconds": 45.2
     }
   }
```

---

## 6. Implementação Detalhada

### 6.1 Classe AgentBase

Classe abstrata base para todos os agentes:

```python
# deepbridge/agents/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from langchain.agents import AgentExecutor
from langchain.tools import Tool
from deepbridge.core import Experiment, DBDataset
import time
import hashlib

class AgentBase(ABC):
    """
    Classe abstrata para agentes LangChain integrados ao DeepBridge.

    Princípios:
    1. LLM apenas roteia e interpreta
    2. DeepBridge executa testes determinísticos
    3. Logging completo para auditoria
    4. Tracking de custos obrigatório para production

    Exemplo:
        >>> class MyAgent(AgentBase):
        ...     def _create_tools(self):
        ...         return [RobustnessTool(experiment=self.experiment)]
        ...
        ...     def _create_agent(self):
        ...         return initialize_agent(...)
    """

    def __init__(
        self,
        dataset: DBDataset,
        llm: Any,  # LangChain LLM
        experiment: Optional[Experiment] = None,
        verbose: bool = True,
        track_costs: bool = True,
        max_iterations: int = 10
    ):
        """
        Inicializa agente base.

        Args:
            dataset: Dataset DeepBridge
            llm: Modelo de linguagem (LangChain compatible)
            experiment: Experimento existente (opcional)
            verbose: Logging verboso
            track_costs: Rastrear custos de LLM calls
            max_iterations: Máximo de iterações do agente
        """
        self.dataset = dataset
        self.llm = llm
        self.experiment = experiment or self._create_experiment()
        self.verbose = verbose
        self.track_costs = track_costs
        self.max_iterations = max_iterations

        # Tracking
        self._execution_log: List[Dict] = []
        self._cost_tracker = CostTracker() if track_costs else None
        self._session_id = self._generate_session_id()

        # Setup LangChain
        self.tools = self._create_tools()
        self.agent = self._create_agent()
        self.executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            verbose=verbose,
            max_iterations=max_iterations,
            handle_parsing_errors=True
        )

    @abstractmethod
    def _create_tools(self) -> List[Tool]:
        """
        Criar LangChain Tools específicos do agente.

        Returns:
            Lista de Tools disponíveis para o agente
        """
        pass

    @abstractmethod
    def _create_agent(self) -> Any:
        """
        Criar agente LangChain (agent type, prompts, etc).

        Returns:
            Agente LangChain configurado
        """
        pass

    def _create_experiment(self) -> Experiment:
        """Criar Experiment padrão se não fornecido."""
        return Experiment(
            dataset=self.dataset,
            experiment_type=self._infer_experiment_type(),
            tests=['robustness', 'uncertainty', 'fairness'],
            verbose=self.verbose
        )

    def run(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Executar query através do agente.

        Args:
            query: Comando em linguagem natural
            **kwargs: Parâmetros adicionais

        Returns:
            {
                'answer': str,           # Resposta do LLM
                'executions': list,      # Testes executados
                'results': Any,          # Resultados dos testes
                'metadata': dict,        # Logs, custos, timestamps
                'deterministic': bool,   # Se execução foi determinística
                'session_id': str        # ID da sessão
            }
        """
        start_time = time.time()

        if self.verbose:
            print(f"[AgentBase] Executando query: {query[:100]}...")

        # Executar através do AgentExecutor
        try:
            response = self.executor.run(query, **kwargs)
        except Exception as e:
            return self._handle_error(e, query, start_time)

        # Compilar resultados
        result = {
            'answer': response,
            'executions': self._execution_log.copy(),
            'results': self._get_latest_results(),
            'metadata': {
                'timestamp': time.time(),
                'duration_seconds': time.time() - start_time,
                'query': query,
                'deterministic': True,  # DeepBridge sempre determinístico
                'llm_calls': len(self._execution_log),
                'session_id': self._session_id
            },
            'deterministic': True,
            'session_id': self._session_id
        }

        # Adicionar custos se tracking ativo
        if self.track_costs and self._cost_tracker:
            result['metadata']['costs'] = self._cost_tracker.get_summary()

        if self.verbose:
            print(f"[AgentBase] Concluído em {result['metadata']['duration_seconds']:.2f}s")

        return result

    def _log_execution(
        self,
        test_type: str,
        config: str,
        results: Any,
        tool_name: str = None
    ):
        """
        Log execução para auditoria.

        Args:
            test_type: Tipo de teste executado
            config: Configuração usada
            results: Resultados do teste
            tool_name: Nome da tool LangChain que executou
        """
        execution_entry = {
            'test_type': test_type,
            'config': config,
            'tool_name': tool_name,
            'results_summary': self._summarize_results(results),
            'timestamp': time.time(),
            'results_hash': self._hash_results(results),
            'deterministic': True
        }

        self._execution_log.append(execution_entry)

        if self.verbose:
            print(f"[AgentBase] Logged execution: {test_type} ({config})")

    def _get_latest_results(self) -> Any:
        """
        Obter últimos resultados de testes.

        Returns:
            Resultados do último teste executado
        """
        if not self._execution_log:
            return None

        # Retornar resultados mais recentes
        latest = self._execution_log[-1]
        return latest.get('results_summary')

    @abstractmethod
    def _summarize_results(self, results: Any) -> Dict:
        """
        Resumir resultados para logging e LLM.

        Args:
            results: Resultados completos do teste

        Returns:
            Resumo estruturado dos resultados
        """
        pass

    def _infer_experiment_type(self) -> str:
        """
        Inferir tipo de experimento do dataset.

        Returns:
            'binary_classification', 'multiclass_classification', ou 'regression'
        """
        target = self.dataset.target
        n_unique = len(target.unique())

        if n_unique == 2:
            return 'binary_classification'
        elif n_unique < 20:
            return 'multiclass_classification'
        else:
            return 'regression'

    def get_audit_trail(self) -> Dict:
        """
        Obter trilha de auditoria completa.

        Returns:
            {
                'session_id': str,
                'executions': list,
                'costs': dict,
                'dataset_hash': str,
                'experiment_config': dict
            }
        """
        return {
            'session_id': self._session_id,
            'executions': self._execution_log,
            'costs': self._cost_tracker.get_detailed() if self.track_costs else None,
            'dataset_info': {
                'n_samples': len(self.dataset.data),
                'n_features': len(self.dataset.features),
                'target_distribution': self.dataset.target.value_counts().to_dict()
            },
            'experiment_type': self.experiment.experiment_type,
            'llm_model': getattr(self.llm, 'model_name', 'unknown')
        }

    def save_audit_trail(self, filepath: str):
        """
        Salvar trilha de auditoria em arquivo JSON.

        Args:
            filepath: Caminho do arquivo de saída
        """
        import json
        audit = self.get_audit_trail()

        with open(filepath, 'w') as f:
            json.dump(audit, f, indent=2, default=str)

        if self.verbose:
            print(f"[AgentBase] Audit trail saved to {filepath}")

    def _generate_session_id(self) -> str:
        """Gerar ID único de sessão."""
        import uuid
        return f"session_{uuid.uuid4().hex[:12]}"

    def _hash_results(self, results: Any) -> str:
        """Gerar hash dos resultados para rastreabilidade."""
        import json
        results_str = json.dumps(results, sort_keys=True, default=str)
        return hashlib.md5(results_str.encode()).hexdigest()

    def _handle_error(self, error: Exception, query: str, start_time: float) -> Dict:
        """Lidar com erros durante execução."""
        return {
            'answer': f"Erro durante execução: {str(error)}",
            'executions': self._execution_log.copy(),
            'results': None,
            'metadata': {
                'timestamp': time.time(),
                'duration_seconds': time.time() - start_time,
                'query': query,
                'error': str(error),
                'deterministic': False,  # Erro = não determinístico
                'session_id': self._session_id
            },
            'deterministic': False,
            'session_id': self._session_id,
            'error': str(error)
        }


class CostTracker:
    """Rastreador de custos de LLM calls."""

    def __init__(self):
        self.calls = []

    def log_call(self, tokens: int, cost: float, model: str):
        """Log uma chamada LLM."""
        self.calls.append({
            'tokens': tokens,
            'cost': cost,
            'model': model,
            'timestamp': time.time()
        })

    def get_summary(self) -> Dict:
        """Obter resumo de custos."""
        if not self.calls:
            return {'total_calls': 0, 'total_tokens': 0, 'total_cost_usd': 0.0}

        return {
            'total_calls': len(self.calls),
            'total_tokens': sum(c['tokens'] for c in self.calls),
            'total_cost_usd': sum(c['cost'] for c in self.calls),
            'avg_cost_per_call': sum(c['cost'] for c in self.calls) / len(self.calls)
        }

    def get_detailed(self) -> List[Dict]:
        """Obter detalhes completos de todas as chamadas."""
        return self.calls.copy()
```

### 6.2 ValidationAgent

Agente principal para validação automática de modelos:

```python
# deepbridge/agents/validation_agent.py
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from deepbridge.agents.base import AgentBase
from deepbridge.agents.tools import (
    RobustnessTool,
    FairnessTool,
    UncertaintyTool,
    ResilienceTool,
    HyperparameterTool
)
from deepbridge.agents.prompts import VALIDATION_SYSTEM_PROMPT

class ValidationAgent(AgentBase):
    """
    Agente para validação automática de modelos ML.

    Este agente interpreta comandos em linguagem natural e executa
    testes de validação apropriados usando DeepBridge como executor.

    Testes Disponíveis:
    - Robustez: Testa resiliência a perturbações
    - Fairness: Verifica viés e compliance EEOC
    - Incerteza: Avalia calibração e conformal prediction
    - Resiliência: Analisa performance em hard samples
    - Hiperparâmetros: Identifica hiperparâmetros críticos

    Exemplo:
        >>> from deepbridge import DBDataset, ValidationAgent
        >>> from langchain.chat_models import ChatOpenAI
        >>>
        >>> dataset = DBDataset(data=df, target_column='target', model=model)
        >>> llm = ChatOpenAI(temperature=0)
        >>> agent = ValidationAgent(dataset=dataset, llm=llm)
        >>>
        >>> result = agent.run('''
        ... Valide este modelo quanto a:
        ... 1. Robustez (nível full)
        ... 2. Viés (verificar EEOC compliance)
        ... Gere relatório executivo.
        ... ''')
        >>>
        >>> print(result['answer'])
        >>> result['results'].save_html('robustness', 'report.html')
    """

    def __init__(self, *args, protected_attributes=None, **kwargs):
        """
        Inicializar ValidationAgent.

        Args:
            *args: Argumentos para AgentBase
            protected_attributes: Lista de atributos protegidos para fairness
            **kwargs: Kwargs para AgentBase
        """
        self.protected_attributes = protected_attributes
        super().__init__(*args, **kwargs)

    def _create_tools(self) -> list[Tool]:
        """Criar ferramentas de validação."""
        tools = [
            RobustnessTool(
                experiment=self.experiment,
                cost_tracker=self._cost_tracker,
                execution_logger=self._log_execution
            ),
            FairnessTool(
                experiment=self.experiment,
                protected_attributes=self.protected_attributes,
                cost_tracker=self._cost_tracker,
                execution_logger=self._log_execution
            ),
            UncertaintyTool(
                experiment=self.experiment,
                cost_tracker=self._cost_tracker,
                execution_logger=self._log_execution
            ),
            ResilienceTool(
                experiment=self.experiment,
                cost_tracker=self._cost_tracker,
                execution_logger=self._log_execution
            ),
            HyperparameterTool(
                experiment=self.experiment,
                cost_tracker=self._cost_tracker,
                execution_logger=self._log_execution
            )
        ]

        return tools

    def _create_agent(self):
        """Criar agente conversacional."""
        return initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=self.verbose,
            max_iterations=self.max_iterations,
            agent_kwargs={
                'prefix': VALIDATION_SYSTEM_PROMPT,
                'format_instructions': """
Use este formato:

Thought: [seu raciocínio sobre qual ferramenta usar]
Action: [nome da ferramenta]
Action Input: [parâmetros em JSON]
Observation: [resultado da ferramenta]
... (repita Thought/Action/Observation conforme necessário)
Thought: Agora tenho informação suficiente para responder
Final Answer: [resposta estruturada e completa]
""",
                'suffix': """
Lembre-se:
1. NUNCA calcule métricas manualmente
2. SEMPRE use as ferramentas para executar testes
3. Cite valores ESPECÍFICOS das métricas nos resultados
4. Forneça recomendações PRIORIZADAS (CRÍTICO/ALTO/MÉDIO/BAIXO)
5. Baseie análise de risco nos valores das métricas

Question: {input}
{agent_scratchpad}
"""
            }
        )

    def _summarize_results(self, results: Any) -> Dict:
        """Resumir resultados de testes para logging."""
        summary = {}

        # Robustness
        if hasattr(results, 'robustness_score'):
            summary['robustness'] = {
                'score': results.robustness_score,
                'status': 'PASS' if results.robustness_score > 0.8 else 'FAIL',
                'degradation': results.avg_degradation if hasattr(results, 'avg_degradation') else None,
                'weak_spots': results.weak_spots[:3] if hasattr(results, 'weak_spots') else []
            }

        # Fairness
        if hasattr(results, 'fairness_metrics'):
            summary['fairness'] = {
                'eeoc_compliant': results.eeoc_compliant if hasattr(results, 'eeoc_compliant') else None,
                'worst_metric': results.worst_metric_value if hasattr(results, 'worst_metric_value') else None,
                'violations': results.violations if hasattr(results, 'violations') else []
            }

        # Uncertainty
        if hasattr(results, 'coverage'):
            summary['uncertainty'] = {
                'coverage': results.coverage,
                'avg_interval_width': results.avg_interval_width if hasattr(results, 'avg_interval_width') else None,
                'calibration_error': results.calibration_error if hasattr(results, 'calibration_error') else None
            }

        return summary
```

*(Continua na próxima resposta devido ao limite de caracteres)*
