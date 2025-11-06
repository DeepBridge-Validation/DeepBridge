# Padrão de Refatoração: Static Renderers → ChartRegistry

**Phase 3 Sprint 11** - Documentação do padrão de refatoração para eliminar código duplicado.

## 📊 Resultados

### Static Uncertainty Renderer
- **Antes:** 1,602 linhas
- **Depois:** 402 linhas  
- **Redução:** -1,200 linhas (**-75%**)

### Padrão Aplicável aos Outros Renderers
- `static_robustness_renderer.py`: 546 → ~150 linhas (est. -73%)
- `static_resilience_renderer.py`: 1,226 → ~300 linhas (est. -75%)

**Total Estimado:** 3,374 → ~850 linhas (**-2,524 linhas, -75%**)

---

## 🎯 Padrão de Refatoração

### ANTES: Código Complexo e Duplicado

```python
# 800+ linhas de geração de charts
def _generate_charts(self, report_data):
    # Import específico
    from deepbridge.templates.report_types.uncertainty.static.charts import UncertaintyChartGenerator
    
    # Configuração complexa
    chart_generator = UncertaintyChartGenerator(self.chart_generator)
    
    # Logging verboso
    logger.info("DADOS PARA COVERAGE VS EXPECTED CHART:")
    # 50+ linhas de validação e logging
    
    # Conversão manual de dados
    if hasattr(alpha_values, 'tolist'):
        alpha_values = alpha_values.tolist()
    # Repetido para cada campo
    
    # Geração individual
    coverage_chart = chart_generator.generate_coverage_vs_expected(report_data)
    # 100+ linhas por chart
    
    # Salvamento manual
    if save_chart:
        # 50+ linhas de código de I/O
```

### DEPOIS: Código Limpo com ChartRegistry

```python
# ~150 linhas de geração de charts
def _generate_charts(self, report_data, save_chart=False):
    charts = {}
    charts_dir = self._setup_charts_directory() if save_chart else None
    
    # Chart 1: Coverage vs Expected (5 linhas)
    if self._has_data(report_data, ['calibration_results']):
        chart_data = self._prepare_coverage_data(report_data)
        result = self.chart_registry.generate('coverage_chart', chart_data)
        
        if result.is_success:
            charts['coverage_vs_expected'] = self._process_chart_result(
                result, 'coverage_vs_expected', charts_dir
            )
    
    # Chart 2, 3, 4... (mesmo padrão)
    
    return charts
```

---

## 🔧 Métodos Helper Reutilizáveis

### 1. Preparação de Dados

```python
def _prepare_coverage_data(self, report_data):
    """5-10 linhas em vez de 50+"""
    calib = report_data['calibration_results']
    
    return {
        'alphas': self._to_list(calib.get('alpha_values', [])),
        'coverage': self._to_list(calib.get('coverage_values', [])),
        'expected': self._to_list(calib.get('expected_coverages', []))
    }
```

### 2. Conversão de Tipos

```python
def _to_list(self, data):
    """Converte numpy arrays para listas"""
    if hasattr(data, 'tolist'):
        return data.tolist()
    return list(data) if isinstance(data, (list, tuple)) else []
```

### 3. Processamento de Resultados

```python
def _process_chart_result(self, result, chart_name, charts_dir=None):
    """Salva arquivo OU retorna base64"""
    if charts_dir:
        file_path = os.path.join(charts_dir, f"{chart_name}.png")
        with open(file_path, 'wb') as f:
            f.write(base64.b64decode(result.content))
        return f"{os.path.basename(charts_dir)}/{chart_name}.png"
    else:
        return result.content
```

---

## 📋 Checklist de Refatoração

### Para Cada Renderer

- [ ] **1. Adicionar import do ChartRegistry**
  ```python
  from ...charts import ChartRegistry
  self.chart_registry = ChartRegistry
  ```

- [ ] **2. Simplificar __init__**
  - Remover imports de chart generators específicos
  - Manter apenas transformers necessários

- [ ] **3. Refatorar _generate_charts()**
  - Identificar quais charts são gerados
  - Mapear para charts do ChartRegistry
  - Criar métodos `_prepare_*_data()` para cada chart

- [ ] **4. Adicionar métodos helper**
  - `_setup_charts_directory()`
  - `_has_data()`
  - `_to_list()`
  - `_process_chart_result()`

- [ ] **5. Remover código obsoleto**
  - Imports de chart generators antigos
  - Logging verboso
  - Validações complexas (ChartRegistry já valida)
  - Código de conversão manual

- [ ] **6. Testar**
  - Verificar que charts são gerados
  - Verificar save_chart=True e False
  - Verificar formato de saída

---

## 🎯 Mapeamento de Charts

### Uncertainty → ChartRegistry
| Chart Antigo | ChartRegistry | Status |
|--------------|---------------|--------|
| coverage_vs_expected | `coverage_chart` | ✅ |
| width_vs_coverage | `width_vs_coverage_static` | ✅ |
| calibration_error | `calibration_error` | ✅ |
| alternative_methods | `alternative_methods_comparison` | ✅ |

### Robustness → ChartRegistry
| Chart Antigo | ChartRegistry | Status |
|--------------|---------------|--------|
| perturbation_impact | `perturbation_impact_static` | ✅ |
| feature_robustness | `feature_robustness` | ✅ |

### Resilience → ChartRegistry
| Chart Antigo | ChartRegistry | Status |
|--------------|---------------|--------|
| test_type_comparison | `test_type_comparison` | ✅ |
| scenario_degradation | `scenario_degradation` | ✅ |

---

## ✅ Benefícios da Refatoração

### 1. Redução Massiva de Código
- **-75%** de linhas de código
- **-1,200 linhas** apenas no UncertaintyRenderer
- **-2,500 linhas** estimado para todos os 3 renderers

### 2. Eliminação de Duplicação
- Lógica de chart generation centralizada
- Validação automática (ChartRegistry)
- Error handling consistente

### 3. Manutenibilidade
- Atualizar chart = atualiza todos os renderers
- Código mais legível e testável
- Menos bugs

### 4. Consistência
- Mesmo estilo visual em todos os reports
- Mesma API para todos os charts
- Comportamento previsível

### 5. Performance
- Charts testados e otimizados
- Sem código redundante
- Cache futuro (Phase 3 Sprint 17)

---

## 🚀 Próximos Passos

### Aplicar Padrão aos Outros Renderers

**1. Static Robustness Renderer** (546 → ~150 linhas)
- Charts: `perturbation_impact_static`, `feature_robustness`
- Tempo estimado: 2 horas
- Redução: ~400 linhas

**2. Static Resilience Renderer** (1,226 → ~300 linhas)
- Charts: `test_type_comparison`, `scenario_degradation`
- Tempo estimado: 3 horas
- Redução: ~900 linhas

**Total:** ~2,500 linhas eliminadas com o padrão estabelecido

---

**Status:** ✅ Padrão definido e demonstrado no UncertaintyRenderer
**Próximo:** Aplicar aos RobustnessRenderer e ResilienceRenderer
