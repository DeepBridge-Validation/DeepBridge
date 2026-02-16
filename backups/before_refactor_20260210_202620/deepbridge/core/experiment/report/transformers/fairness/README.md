# Fairness Report Transformers - Refactored Module

Este módulo foi refatorado para melhor manutenibilidade e extensibilidade.

## 📁 Estrutura

```
fairness/
├── __init__.py                     # Exports principais
├── README.md                       # Esta documentação
├── data_transformer.py             # Transformação de dados (~260 linhas)
├── chart_factory.py                # Factory para criação de charts (~230 linhas)
├── utils.py                        # Utilitários e constantes (~150 linhas)
├── charts/                         # Módulos de visualização
│   ├── __init__.py
│   ├── base_chart.py              # Classe abstrata base (~90 linhas)
│   ├── posttrain_charts.py        # 3 charts pós-treino (~300 linhas)
│   ├── pretrain_charts.py         # 3 charts pré-treino (~250 linhas)
│   ├── complementary_charts.py    # 3 charts complementares (~300 linhas)
│   ├── distribution_charts.py     # 2 charts de distribuição (~140 linhas)
│   └── legacy_charts.py           # Wrapper com deprecation warnings
└── deprecated/                     # Charts legados (backward compatibility)
    ├── __init__.py
    └── legacy_charts.py           # Implementações originais (~280 linhas)
```

## 🎯 Uso

### Transformação de Dados

```python
from deepbridge.core.experiment.report.transformers.fairness import FairnessDataTransformer

transformer = FairnessDataTransformer()
report_data = transformer.transform(fairness_results, model_name="My Model")
```

### Uso de Charts Individuais

```python
from deepbridge.core.experiment.report.transformers.fairness.charts import (
    DisparateImpactGaugeChart,
    PretrainMetricsOverviewChart
)

# Criar chart específico
gauge = DisparateImpactGaugeChart()
json_chart = gauge.create({
    'posttrain_metrics': metrics,
    'protected_attrs': ['gender', 'age']
})
```

### Factory de Charts

```python
from deepbridge.core.experiment.report.transformers.fairness import ChartFactory

factory = ChartFactory()
all_charts = factory.create_all_charts(fairness_results)
```

## 📊 Classes de Charts

### Post-Training (posttrain_charts.py)
- **DisparateImpactGaugeChart**: Gauge chart para EEOC 80% Rule
- **DisparityComparisonChart**: Barra divergente para disparidade
- **ComplianceStatusMatrixChart**: Matriz de status de conformidade

### Pre-Training (pretrain_charts.py)
- **PretrainMetricsOverviewChart**: Overview de 4 métricas pré-treino
- **GroupSizesChart**: Distribuição de tamanhos de grupos
- **ConceptBalanceChart**: Comparação de taxas positivas

### Complementary (complementary_charts.py)
- **PrecisionAccuracyComparisonChart**: Comparação de precisão e acurácia
- **TreatmentEqualityScatterChart**: Scatter de FN vs FP rates
- **ComplementaryMetricsRadarChart**: Radar de 6 métricas complementares

### Distribution (distribution_charts.py)
- **ProtectedAttributesDistributionChart**: Distribuição de atributos protegidos
- **TargetDistributionChart**: Distribuição da variável target

## ⚠️ Charts Legados (DEPRECATED)

Os seguintes charts estão **deprecados** e serão removidos em versão futura:
- `MetricsComparisonChart` → Use `posttrain_charts`
- `FairnessRadarChart` → Use `ComplementaryMetricsRadarChart`
- `ConfusionMatricesChart` → Use `complementary_charts`
- `ThresholdAnalysisChart` → Use `posttrain_charts`

Quando instanciados, esses charts emitem `DeprecationWarning`.

## 🧪 Testes

```bash
# Executar todos os testes do módulo
poetry run pytest tests/core/experiment/report/transformers/fairness/ -v

# Executar testes de uma classe específica
poetry run pytest tests/core/experiment/report/transformers/fairness/test_posttrain_charts.py -v

# Com cobertura
poetry run pytest tests/core/experiment/report/transformers/fairness/ --cov=deepbridge.core.experiment.report.transformers.fairness
```

### Cobertura de Testes

- ✅ **51 testes** implementados
- ✅ Cobertura de todas as classes de charts
- ✅ Testes de edge cases (dados vazios, métricas ausentes)
- ✅ Validação de JSON Plotly
- ✅ Testes de integração com data transformer

## 🔧 Extensibilidade

### Adicionar Novo Chart

1. **Criar classe** que herda de `BaseChart`:

```python
# Em charts/my_new_charts.py
from .base_chart import BaseChart
from typing import Dict, Any

class MyNewChart(BaseChart):
    """Descrição do chart."""

    def create(self, data: Dict[str, Any]) -> str:
        """Cria o chart e retorna JSON."""
        # Implementação
        fig = go.Figure(...)
        self._apply_common_layout(fig, title='My Chart')
        return self._to_json(fig)
```

2. **Registrar no ChartFactory**:

```python
# Em chart_factory.py
from .charts.my_new_charts import MyNewChart

class ChartFactory:
    def __init__(self):
        # ...
        self.my_new_chart = MyNewChart()

    def create_all_charts(self, results):
        # ...
        charts['my_new_chart'] = self.my_new_chart.create(data)
```

3. **Adicionar testes**:

```python
# Em tests/fairness/test_my_new_charts.py
def test_my_new_chart_creation(plotly_validator):
    chart = MyNewChart()
    result = chart.create({'data': ...})
    assert plotly_validator(result)
```

## 📚 Utilitários

### utils.py

Fornece:
- **Funções**: `get_status_from_interpretation()`, `get_assessment_text()`, `format_metric_name()`, etc.
- **Constantes**: `POSTTRAIN_MAIN_METRICS`, `POSTTRAIN_COMPLEMENTARY_METRICS`, `PRETRAIN_METRICS`
- **Labels**: `METRIC_LABELS`, `METRIC_SHORT_LABELS`

## 🔄 Backward Compatibility

O arquivo original `fairness_simple.py` foi mantido e agora **delega** para a nova implementação:

```python
# fairness_simple.py (LEGACY)
from .fairness import FairnessDataTransformer as RefactoredTransformer

class FairnessDataTransformerSimple:
    def __init__(self):
        self._transformer = RefactoredTransformer()

    def transform(self, results, model_name="Model"):
        return self._transformer.transform(results, model_name)
```

## 📈 Benefícios do Refatoramento

| Aspecto | Antes | Depois |
|---------|-------|--------|
| **Linhas por arquivo** | 1.848 | ~200 (média) |
| **Arquivos** | 1 | 11 modulares |
| **Testabilidade** | ❌ Difícil | ✅ 51 testes |
| **Extensibilidade** | ❌ Baixa | ✅ Alta |
| **Manutenibilidade** | ❌ Baixa | ✅ Alta |
| **Responsabilidades** | ❌ Misturadas | ✅ Separadas |

## 📝 Notas

- Todos os charts retornam string JSON do Plotly
- BaseChart fornece cores, layouts e utilitários comuns
- ChartFactory trata exceções e logs
- Deprecation warnings são emitidos para charts legados
- 100% de compatibilidade com código existente
