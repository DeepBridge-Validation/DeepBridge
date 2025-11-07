# Correção: Reports Interativos vs Estáticos

## Problema Identificado

Os dois tipos de reports (interactive com Plotly e static com Matplotlib/Seaborn) estavam sendo gerados **idênticos**, ambos usando Plotly (interativo).

### Causa Raiz

O método `Experiment.save_html()` **não aceitava** o parâmetro `report_type`, sempre usando o valor padrão de `ExperimentResult.save_html()` que era `"static"`. Como ambas as chamadas no notebook eram idênticas (sem passar `report_type`), os dois reports gerados eram iguais.

## Arquivos Modificados

### 1. `/deepbridge/core/experiment/experiment.py`

**Linha 673** - Adicionado parâmetro `report_type`:
```python
# ANTES
def save_html(self, test_type: str, file_path: str, model_name: str = None) -> str:

# DEPOIS
def save_html(self, test_type: str, file_path: str, model_name: str = None, report_type: str = "interactive") -> str:
```

**Linhas 708 e 720** - Passando `report_type` para `ExperimentResult.save_html()`:
```python
# ANTES
return self._experiment_result.save_html(test_type, file_path, model_name or self._get_model_name())

# DEPOIS
return self._experiment_result.save_html(test_type, file_path, model_name or self._get_model_name(), report_type=report_type)
```

### 2. `/deepbridge/core/experiment/results.py`

Adicionado método `save_html()` em três classes:

#### `UncertaintyResult` (linha 148)
```python
def save_html(self, file_path: str, model_name: str = "Model", report_type: str = "interactive") -> str:
    """Generate HTML report for uncertainty analysis."""
    from deepbridge.core.experiment.report.report_manager import ReportManager

    report_manager = ReportManager()
    report_path = report_manager.generate_report(
        test_type='uncertainty',
        results=self._results,
        file_path=file_path,
        model_name=model_name,
        report_type=report_type
    )
    return report_path
```

#### `RobustnessResult` (linha 109)
Implementação similar para robustness

#### `ResilienceResult` (linha 188)
Implementação similar para resilience

## Como Usar

### Opção 1: Via resultado direto
```python
uncertainty_result = exp.run_test('uncertainty')

# Report interativo (Plotly)
uncertainty_result.save_html(
    'report_interactive.html',
    'My Model',
    report_type='interactive'
)

# Report estático (Matplotlib)
uncertainty_result.save_html(
    'report_static.html',
    'My Model',
    report_type='static'
)
```

### Opção 2: Via Experiment
```python
# Armazenar resultado
exp._test_results = {'uncertainty': uncertainty_result}

# Report interativo
exp.save_html(
    'uncertainty',
    'report_interactive.html',
    'My Model',
    report_type='interactive'
)

# Report estático
exp.save_html(
    'uncertainty',
    'report_static.html',
    'My Model',
    report_type='static'
)
```

## Diferenças entre os tipos

### Interactive (report_type='interactive')
- ✅ Usa **Plotly** para gráficos
- ✅ Gráficos **interativos** (zoom, hover, pan)
- ✅ Melhor para **exploração de dados**
- ⚠️ Arquivo maior (~340KB)
- ⚠️ Requer JavaScript habilitado

### Static (report_type='static')
- ✅ Usa **Matplotlib/Seaborn** para gráficos
- ✅ Gráficos em **PNG embutidos** (base64)
- ✅ Melhor para **relatórios formais e impressão**
- ✅ Funciona sem JavaScript
- ⚠️ Sem interatividade

## Valores Padrão

- `UncertaintyResult.save_html()`: padrão = `"interactive"`
- `RobustnessResult.save_html()`: padrão = `"interactive"`
- `ResilienceResult.save_html()`: padrão = `"interactive"`
- `FairnessResult.save_html()`: padrão = `"interactive"`
- `Experiment.save_html()`: padrão = `"interactive"`
- `ExperimentResult.save_html()`: padrão = `"static"` (mantido para compatibilidade)

## Verificação

Para verificar se os reports são diferentes:

```bash
# Tamanho dos arquivos
ls -lh outputs/uncertainty_reports/

# Buscar por Plotly (interactive) ou matplotlib (static)
grep -c "plotly" uncertainty_report_interactive.html
grep -c "plotly" uncertainty_report_static.html
```

Se a correção estiver funcionando:
- Interactive: deve ter 5+ referências a "plotly"
- Static: deve ter 0 referências a "plotly" e imagens PNG em base64

## Status da Refatoração

✅ **Sim, estava previsto ter os dois tipos** no planejamento:

- `/planejamento_report/ANALISE_ARQUITETURA_REPORTS.md:12` confirma: "O sistema é capaz de gerar reports HTML **interativos e estáticos**"
- Dois tipos de renderers existem:
  - **Simple Renderers** → usa Plotly (interativo)
  - **Static Renderers** → usa Matplotlib (estático)

A infraestrutura estava correta, mas faltava expor o parâmetro `report_type` nas APIs públicas.

## Data da Correção

2025-11-06 20:30
