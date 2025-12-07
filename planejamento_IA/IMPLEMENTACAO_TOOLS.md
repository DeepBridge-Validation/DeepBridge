# Implementação de LangChain Tools para DeepBridge

**Documento Complementar:** Detalhamento técnico das Tools
**Versão:** 1.0
**Data:** Dezembro 2025

---

## 📋 Visão Geral

Este documento detalha a implementação das **LangChain Tools** que serão usadas pelos agentes para executar testes no DeepBridge.

**Princípio Fundamental:**
> Tools são a ponte entre LangChain (interpretação) e DeepBridge (execução determinística).

---

## 1. Arquitetura de Tools

### 1.1 Hierarquia de Classes

```
BaseTool (LangChain)
    ↑
    │
DeepBridgeTool (abstrata - nossa)
    ↑
    ├── RobustnessTool
    ├── FairnessTool
    ├── UncertaintyTool
    ├── ResilienceTool
    ├── HyperparameterTool
    ├── DistillationTool
    └── SyntheticTool
```

### 1.2 Base Tool DeepBridge

```python
# deepbridge/agents/tools/base_tool.py
from langchain.tools import BaseTool
from typing import Optional, Dict, Any, Callable
from deepbridge.core import Experiment
from pydantic import BaseModel, Field

class DeepBridgeTool(BaseTool):
    """
    Classe base para todas as Tools do DeepBridge.

    Garante:
    - Execução determinística via DeepBridge
    - Logging de execuções
    - Tracking de custos
    - Estruturação de outputs
    """

    experiment: Experiment = Field(description="Experimento DeepBridge")
    cost_tracker: Optional[Any] = Field(default=None, description="Rastreador de custos")
    execution_logger: Optional[Callable] = Field(default=None, description="Logger de execuções")

    class Config:
        arbitrary_types_allowed = True

    def _run(self, *args, **kwargs) -> str:
        """
        Executar tool e retornar resultado como string estruturada.

        Returns:
            JSON string com resultados
        """
        # Subclasses devem implementar
        raise NotImplementedError

    async def _arun(self, *args, **kwargs):
        """Async version (não implementado ainda)."""
        raise NotImplementedError("Async not supported yet")

    def _log_execution(self, test_type: str, config: str, results: Any):
        """Log execução se logger fornecido."""
        if self.execution_logger:
            self.execution_logger(
                test_type=test_type,
                config=config,
                results=results,
                tool_name=self.name
            )

    def _format_output(self, results: Dict) -> str:
        """
        Formatar output para LLM consumption.

        Estrutura padrão:
        {
            "test_type": str,
            "deterministic": true,
            "metrics": {...},
            "summary": str,
            "recommendations": [...]
        }
        """
        import json
        return json.dumps(results, indent=2, default=str)
```

---

## 2. RobustnessTool

```python
# deepbridge/agents/tools/robustness_tool.py
from deepbridge.agents.tools.base_tool import DeepBridgeTool
from typing import Optional, List
import json

class RobustnessTool(DeepBridgeTool):
    """
    Tool para executar testes de robustez.

    O LLM usa esta tool para avaliar resiliência do modelo a perturbações.

    Parâmetros:
    - config: 'quick' | 'medium' | 'full'
    - feature_subset: lista de features para testar (opcional)
    - perturbation_scale: escala de perturbação (default: auto)

    Retorna:
    - robustness_score: float (0-1)
    - degradation: percentual de degradação de performance
    - weak_spots: features mais vulneráveis
    - overfitting_regions: regiões com overfitting localizado

    Exemplo de uso pelo LLM:
        Action: run_robustness_test
        Action Input: {"config": "medium", "feature_subset": ["income", "age"]}
    """

    name = "run_robustness_test"
    description = """
Executa testes de robustez no modelo para avaliar resiliência a perturbações nos dados.

Use esta ferramenta quando precisar:
- Avaliar estabilidade do modelo
- Identificar features vulneráveis (weak spots)
- Detectar overfitting localizado
- Testar comportamento sob ruído

Parâmetros:
{
    "config": "quick" | "medium" | "full",  // Intensidade do teste
    "feature_subset": ["feature1", "feature2", ...],  // Opcional: testar apenas algumas features
    "perturbation_scale": float  // Opcional: escala de perturbação (default: auto)
}

Retorna JSON com:
- robustness_score: score geral (0-1, maior = mais robusto)
- avg_degradation_pct: degradação média de performance (%)
- weak_spots: top 5 features mais vulneráveis
- overfitting_detected: se detectou overfitting
- deterministic: true (sempre)

IMPORTANTE: Esta tool EXECUTA os testes. Você interpreta os resultados.
"""

    def _run(
        self,
        config: str = 'medium',
        feature_subset: Optional[List[str]] = None,
        perturbation_scale: Optional[float] = None
    ) -> str:
        """Executar teste de robustez."""

        # Validar config
        valid_configs = ['quick', 'medium', 'full']
        if config not in valid_configs:
            return self._format_output({
                'error': f"Config inválida: {config}. Use: {valid_configs}",
                'deterministic': False
            })

        # Executar via DeepBridge (DETERMINÍSTICO)
        try:
            results = self.experiment.run_test(
                test_type='robustness',
                config_name=config,
                feature_subset=feature_subset,
                perturbation_scale=perturbation_scale
            )
        except Exception as e:
            return self._format_output({
                'error': str(e),
                'deterministic': False
            })

        # Extrair métricas principais
        output = {
            'test_type': 'robustness',
            'config': config,
            'deterministic': True,
            'timestamp': results.timestamp if hasattr(results, 'timestamp') else None,

            # Métricas principais
            'robustness_score': results.robustness_score,
            'avg_degradation_pct': results.avg_degradation * 100 if hasattr(results, 'avg_degradation') else None,

            # Weak spots
            'weak_spots': [
                {'feature': f, 'impact': float(i)}
                for f, i in (results.weak_spots[:5] if hasattr(results, 'weak_spots') else [])
            ],

            # Overfitting
            'overfitting_detected': results.has_overfitting if hasattr(results, 'has_overfitting') else False,
            'overfitting_regions': results.overfitting_regions if hasattr(results, 'overfitting_regions') else [],

            # Interpretação
            'status': self._interpret_score(results.robustness_score),
            'summary': self._generate_summary(results)
        }

        # Log execução
        self._log_execution('robustness', config, results)

        return self._format_output(output)

    def _interpret_score(self, score: float) -> str:
        """Interpretar score de robustez."""
        if score >= 0.9:
            return 'EXCELLENT'
        elif score >= 0.8:
            return 'GOOD'
        elif score >= 0.7:
            return 'ACCEPTABLE'
        elif score >= 0.6:
            return 'WARNING'
        else:
            return 'CRITICAL'

    def _generate_summary(self, results) -> str:
        """Gerar resumo técnico dos resultados."""
        score = results.robustness_score
        degradation = results.avg_degradation * 100 if hasattr(results, 'avg_degradation') else 0

        summary = f"Robustness Score: {score:.3f} ({self._interpret_score(score)}). "
        summary += f"Degradação média: {degradation:.1f}%. "

        if hasattr(results, 'weak_spots') and results.weak_spots:
            top_weak = results.weak_spots[0]
            summary += f"Maior vulnerabilidade: {top_weak[0]} ({top_weak[1]:.1%} impacto). "

        if hasattr(results, 'has_overfitting') and results.has_overfitting:
            summary += "⚠️ Overfitting localizado detectado."

        return summary
```

---

## 3. FairnessTool

```python
# deepbridge/agents/tools/fairness_tool.py
from deepbridge.agents.tools.base_tool import DeepBridgeTool
from typing import Optional, List

class FairnessTool(DeepBridgeTool):
    """
    Tool para executar testes de fairness (viés).

    Avalia se o modelo cumpre requisitos de justiça e compliance regulatório.

    Parâmetros:
    - config: 'quick' | 'medium' | 'full'
    - protected_attributes: lista de atributos protegidos (opcional)
    - eeoc_check: se deve verificar EEOC 80% rule (default: True)

    Retorna:
    - eeoc_compliant: se passa no 80% rule (EEOC)
    - fairness_metrics: 15 métricas calculadas
    - violations: lista de violações detectadas
    - recommendations: recomendações de mitigação

    Exemplo de uso pelo LLM:
        Action: run_fairness_test
        Action Input: {"config": "medium", "protected_attributes": ["gender", "race"]}
    """

    name = "run_fairness_test"
    description = """
Executa testes de fairness (viés) para verificar justiça algorítmica e compliance.

Use esta ferramenta quando precisar:
- Verificar viés em relação a grupos protegidos
- Checar compliance EEOC (80% rule)
- Calcular métricas de fairness (demographic parity, equalized odds, etc.)
- Identificar disparidades entre grupos

Parâmetros:
{
    "config": "quick" | "medium" | "full",  // Intensidade do teste
    "protected_attributes": ["gender", "race", ...],  // Opcional: atributos a verificar
    "eeoc_check": true | false  // Default: true
}

Retorna JSON com:
- eeoc_compliant: true/false (se passa no 80% rule)
- metrics: objeto com 15 métricas de fairness
- violations: lista de violações detectadas
- worst_metric: pior métrica detectada
- recommendations: recomendações priorizadas
- deterministic: true (sempre)

IMPORTANTE:
- EEOC 80% rule: taxa de seleção do grupo protegido deve ser ≥80% do grupo referência
- Métricas incluem: demographic parity, equalized odds, equal opportunity, etc.
"""

    protected_attributes: Optional[List[str]] = None

    def _run(
        self,
        config: str = 'medium',
        protected_attributes: Optional[List[str]] = None,
        eeoc_check: bool = True
    ) -> str:
        """Executar teste de fairness."""

        # Usar protected_attributes fornecidos ou padrão
        attrs = protected_attributes or self.protected_attributes
        if attrs:
            self.experiment.protected_attributes = attrs

        # Executar via DeepBridge
        try:
            results = self.experiment.run_fairness_tests(
                config=config,
                eeoc_check=eeoc_check
            )
        except Exception as e:
            return self._format_output({
                'error': str(e),
                'deterministic': False
            })

        # Extrair métricas
        output = {
            'test_type': 'fairness',
            'config': config,
            'deterministic': True,
            'timestamp': results.timestamp if hasattr(results, 'timestamp') else None,

            # Compliance EEOC
            'eeoc_compliant': results.eeoc_compliant if hasattr(results, 'eeoc_compliant') else None,
            'impact_ratio': results.impact_ratio if hasattr(results, 'impact_ratio') else None,

            # Métricas detalhadas
            'metrics': results.metrics if hasattr(results, 'metrics') else {},

            # Violações
            'violations': results.violations if hasattr(results, 'violations') else [],
            'num_violations': len(results.violations) if hasattr(results, 'violations') else 0,

            # Pior métrica
            'worst_metric': {
                'name': results.worst_metric_name if hasattr(results, 'worst_metric_name') else None,
                'value': results.worst_metric_value if hasattr(results, 'worst_metric_value') else None,
                'severity': results.worst_metric_severity if hasattr(results, 'worst_metric_severity') else None
            },

            # Recomendações
            'recommendations': results.recommendations if hasattr(results, 'recommendations') else [],

            # Interpretação
            'status': self._interpret_compliance(results),
            'summary': self._generate_summary(results)
        }

        # Log execução
        self._log_execution('fairness', config, results)

        return self._format_output(output)

    def _interpret_compliance(self, results) -> str:
        """Interpretar status de compliance."""
        if not hasattr(results, 'eeoc_compliant'):
            return 'UNKNOWN'

        if results.eeoc_compliant:
            return 'COMPLIANT'
        else:
            # Severidade baseada em impact ratio
            if hasattr(results, 'impact_ratio'):
                if results.impact_ratio < 0.6:
                    return 'CRITICAL_VIOLATION'
                elif results.impact_ratio < 0.7:
                    return 'SEVERE_VIOLATION'
                else:
                    return 'MINOR_VIOLATION'
            return 'NON_COMPLIANT'

    def _generate_summary(self, results) -> str:
        """Gerar resumo técnico."""
        summary = ""

        if hasattr(results, 'eeoc_compliant'):
            if results.eeoc_compliant:
                summary += "✓ EEOC Compliant (80% rule). "
            else:
                impact = results.impact_ratio if hasattr(results, 'impact_ratio') else 0
                summary += f"✗ EEOC Violation (impact ratio: {impact:.2f}, required: ≥0.80). "

        if hasattr(results, 'violations') and results.violations:
            summary += f"{len(results.violations)} violação(ões) detectada(s). "

        if hasattr(results, 'worst_metric_name'):
            summary += f"Pior métrica: {results.worst_metric_name} ({results.worst_metric_value:.3f})."

        return summary
```

---

## 4. UncertaintyTool

```python
# deepbridge/agents/tools/uncertainty_tool.py
from deepbridge.agents.tools.base_tool import DeepBridgeTool

class UncertaintyTool(DeepBridgeTool):
    """
    Tool para análise de incerteza em predições.

    Avalia calibração de probabilidades e conformal prediction.

    Parâmetros:
    - config: 'quick' | 'medium' | 'full'
    - confidence_level: nível de confiança para intervalos (default: 0.95)
    - method: 'conformal' | 'calibration' | 'both'

    Retorna:
    - coverage: cobertura real dos intervalos
    - avg_interval_width: largura média dos intervalos
    - calibration_error: erro de calibração
    - reliability_diagram_data: dados para diagrama de confiabilidade
    """

    name = "run_uncertainty_test"
    description = """
Executa análise de incerteza nas predições do modelo.

Use esta ferramenta quando precisar:
- Avaliar calibração de probabilidades
- Gerar intervalos de predição (conformal prediction)
- Verificar confiabilidade das predições
- Analisar coverage vs. interval width trade-off

Parâmetros:
{
    "config": "quick" | "medium" | "full",
    "confidence_level": 0.90 | 0.95 | 0.99,  // Default: 0.95
    "method": "conformal" | "calibration" | "both"  // Default: "both"
}

Retorna JSON com:
- coverage: cobertura real (ideal: ≈ confidence_level)
- avg_interval_width: largura média dos intervalos
- calibration_error: Expected Calibration Error (ECE)
- miscalibration_severity: CRITICAL | HIGH | MEDIUM | LOW
- deterministic: true (sempre)

IMPORTANTE:
- Coverage próximo ao confidence_level indica boa calibração
- Intervalos muito largos indicam alta incerteza
- ECE < 0.05 é considerado bem calibrado
"""

    def _run(
        self,
        config: str = 'medium',
        confidence_level: float = 0.95,
        method: str = 'both'
    ) -> str:
        """Executar análise de incerteza."""

        # Executar via DeepBridge
        try:
            results = self.experiment.run_test(
                test_type='uncertainty',
                config_name=config,
                confidence_level=confidence_level,
                method=method
            )
        except Exception as e:
            return self._format_output({
                'error': str(e),
                'deterministic': False
            })

        # Extrair métricas
        output = {
            'test_type': 'uncertainty',
            'config': config,
            'deterministic': True,
            'confidence_level': confidence_level,

            # Conformal Prediction
            'coverage': results.coverage if hasattr(results, 'coverage') else None,
            'coverage_gap': abs(results.coverage - confidence_level) if hasattr(results, 'coverage') else None,
            'avg_interval_width': results.avg_interval_width if hasattr(results, 'avg_interval_width') else None,

            # Calibration
            'calibration_error': results.calibration_error if hasattr(results, 'calibration_error') else None,
            'miscalibration_severity': self._interpret_calibration(
                results.calibration_error if hasattr(results, 'calibration_error') else None
            ),

            # Interpretação
            'status': self._interpret_coverage(
                results.coverage if hasattr(results, 'coverage') else None,
                confidence_level
            ),
            'summary': self._generate_summary(results, confidence_level)
        }

        # Log execução
        self._log_execution('uncertainty', config, results)

        return self._format_output(output)

    def _interpret_calibration(self, ece: float) -> str:
        """Interpretar erro de calibração."""
        if ece is None:
            return 'UNKNOWN'
        if ece < 0.05:
            return 'WELL_CALIBRATED'
        elif ece < 0.10:
            return 'ACCEPTABLE'
        elif ece < 0.15:
            return 'POOR'
        else:
            return 'SEVERELY_MISCALIBRATED'

    def _interpret_coverage(self, coverage: float, target: float) -> str:
        """Interpretar coverage."""
        if coverage is None:
            return 'UNKNOWN'

        gap = abs(coverage - target)
        if gap < 0.02:
            return 'EXCELLENT'
        elif gap < 0.05:
            return 'GOOD'
        elif gap < 0.10:
            return 'ACCEPTABLE'
        else:
            return 'POOR'

    def _generate_summary(self, results, target_coverage: float) -> str:
        """Gerar resumo técnico."""
        summary = ""

        if hasattr(results, 'coverage'):
            gap = abs(results.coverage - target_coverage)
            summary += f"Coverage: {results.coverage:.1%} (target: {target_coverage:.1%}, gap: {gap:.1%}). "

        if hasattr(results, 'avg_interval_width'):
            summary += f"Largura média de intervalo: {results.avg_interval_width:.3f}. "

        if hasattr(results, 'calibration_error'):
            summary += f"Calibration error (ECE): {results.calibration_error:.3f}."

        return summary
```

---

## 5. Demais Tools (Resumido)

### 5.1 ResilienceTool

```python
class ResilienceTool(DeepBridgeTool):
    """
    Testa resiliência do modelo em samples difíceis (hard samples).

    Avalia:
    - Performance em samples misclassified
    - Impacto de distribution shift
    - Comparação com modelos alternativos
    """
    name = "run_resilience_test"
    # Implementação similar aos anteriores...
```

### 5.2 HyperparameterTool

```python
class HyperparameterTool(DeepBridgeTool):
    """
    Analisa importância de hiperparâmetros.

    Identifica:
    - Hiperparâmetros críticos para performance
    - Sensibilidade a mudanças
    - Ranges seguros de valores
    """
    name = "run_hyperparameter_analysis"
    # Implementação similar...
```

### 5.3 DistillationTool

```python
class DistillationTool(DeepBridgeTool):
    """
    Executa destilação de conhecimento.

    Cria modelos simplificados mantendo performance.
    """
    name = "run_model_distillation"
    # Implementação similar...
```

### 5.4 SyntheticTool

```python
class SyntheticTool(DeepBridgeTool):
    """
    Gera dados sintéticos para augmentation ou testes.

    Usa Gaussian Copula ou outros métodos.
    """
    name = "generate_synthetic_data"
    # Implementação similar...
```

---

## 6. Registro de Tools

### 6.1 Factory de Tools

```python
# deepbridge/agents/tools/__init__.py
from .robustness_tool import RobustnessTool
from .fairness_tool import FairnessTool
from .uncertainty_tool import UncertaintyTool
from .resilience_tool import ResilienceTool
from .hyperparameter_tool import HyperparameterTool
from .distillation_tool import DistillationTool
from .synthetic_tool import SyntheticTool

__all__ = [
    'RobustnessTool',
    'FairnessTool',
    'UncertaintyTool',
    'ResilienceTool',
    'HyperparameterTool',
    'DistillationTool',
    'SyntheticTool'
]

# Factory para criação de tools
class ToolFactory:
    """Factory para criar tools com configurações padrão."""

    @staticmethod
    def create_all_tools(experiment, cost_tracker=None, execution_logger=None):
        """Criar todas as tools disponíveis."""
        return [
            RobustnessTool(
                experiment=experiment,
                cost_tracker=cost_tracker,
                execution_logger=execution_logger
            ),
            FairnessTool(
                experiment=experiment,
                cost_tracker=cost_tracker,
                execution_logger=execution_logger
            ),
            UncertaintyTool(
                experiment=experiment,
                cost_tracker=cost_tracker,
                execution_logger=execution_logger
            ),
            ResilienceTool(
                experiment=experiment,
                cost_tracker=cost_tracker,
                execution_logger=execution_logger
            ),
            HyperparameterTool(
                experiment=experiment,
                cost_tracker=cost_tracker,
                execution_logger=execution_logger
            ),
        ]

    @staticmethod
    def create_validation_tools(experiment, **kwargs):
        """Criar apenas tools de validação básica."""
        return [
            RobustnessTool(experiment=experiment, **kwargs),
            FairnessTool(experiment=experiment, **kwargs),
            UncertaintyTool(experiment=experiment, **kwargs),
        ]
```

---

## 7. Testes Unitários

### 7.1 Exemplo de Teste

```python
# tests/agents/tools/test_robustness_tool.py
import pytest
from deepbridge import DBDataset, Experiment
from deepbridge.agents.tools import RobustnessTool

def test_robustness_tool_execution(sample_dataset, sample_model):
    """Testar execução básica da RobustnessTool."""
    # Setup
    dataset = DBDataset(
        data=sample_dataset,
        target_column='target',
        model=sample_model
    )
    experiment = Experiment(
        dataset=dataset,
        experiment_type='binary_classification'
    )

    tool = RobustnessTool(experiment=experiment)

    # Execute
    result = tool._run(config='quick')

    # Assert
    assert result is not None
    assert 'robustness_score' in result
    assert 'deterministic' in result
    import json
    parsed = json.loads(result)
    assert parsed['deterministic'] == True
    assert 0 <= parsed['robustness_score'] <= 1

def test_robustness_tool_invalid_config(sample_experiment):
    """Testar comportamento com config inválida."""
    tool = RobustnessTool(experiment=sample_experiment)
    result = tool._run(config='invalid_config')

    import json
    parsed = json.loads(result)
    assert 'error' in parsed
    assert parsed['deterministic'] == False
```

---

## 8. Documentação para Usuários

### 8.1 Guia de Uso das Tools

**Para Desenvolvedores de Agentes:**

```python
# Criar tool standalone
from deepbridge.agents.tools import RobustnessTool

tool = RobustnessTool(
    experiment=experiment,
    cost_tracker=my_tracker,
    execution_logger=my_logger
)

# Executar diretamente
result_str = tool._run(config='medium', feature_subset=['income', 'age'])

# Parse result
import json
result = json.loads(result_str)
print(f"Robustness Score: {result['robustness_score']}")
```

**Para LLMs (via prompt):**

```
Você tem acesso às seguintes ferramentas:

1. run_robustness_test: Testa robustez do modelo
   Input: {"config": "medium", "feature_subset": ["feature1", ...]}

2. run_fairness_test: Testa fairness/viés
   Input: {"config": "medium", "protected_attributes": ["gender", ...]}

3. run_uncertainty_test: Analisa incerteza
   Input: {"config": "medium", "confidence_level": 0.95}

IMPORTANTE: Use as ferramentas para executar testes.
NUNCA calcule métricas manualmente.
```

---

## 9. Próximos Passos

### Implementação Incremental

**Fase 1 - MVP (1 semana):**
- [ ] Implementar `DeepBridgeTool` (base)
- [ ] Implementar `RobustnessTool` completa
- [ ] Testes unitários
- [ ] Documentação básica

**Fase 2 - Core Tools (2 semanas):**
- [ ] `FairnessTool`
- [ ] `UncertaintyTool`
- [ ] `ResilienceTool`
- [ ] `HyperparameterTool`
- [ ] Testes de integração

**Fase 3 - Advanced Tools (1 semana):**
- [ ] `DistillationTool`
- [ ] `SyntheticTool`
- [ ] ToolFactory completo
- [ ] Documentação completa

---

## 10. Checklist de Qualidade

Para cada Tool implementada, verificar:

- [ ] Herda de `DeepBridgeTool`
- [ ] Implementa `_run()` com tipagem correta
- [ ] Retorna JSON string estruturado
- [ ] Inclui campo `deterministic: true`
- [ ] Loga execução via `_log_execution()`
- [ ] Trata erros gracefully
- [ ] Tem description detalhada para LLM
- [ ] Tem testes unitários (>80% coverage)
- [ ] Tem exemplo de uso em docstring
- [ ] Documentada em README

---

**Conclusão:**

A implementação de Tools segue padrões consistentes que garantem:
1. ✅ Determinismo (DeepBridge executa, LLM interpreta)
2. ✅ Rastreabilidade (logging + metadata)
3. ✅ Estruturação (outputs padronizados)
4. ✅ Testabilidade (tools isoladas e testáveis)

Próximo documento: **Casos de Uso Detalhados**
