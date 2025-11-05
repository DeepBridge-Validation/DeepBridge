# Padrões de Código - Sistema de Reports

**Versão:** 1.0
**Data:** 05/11/2025
**Status:** ✅ Ativo

---

## 📋 Índice

1. [Visão Geral](#visão-geral)
2. [Estrutura de Renderer](#estrutura-de-renderer)
3. [Uso de CSSManager](#uso-de-cssmanager)
4. [Serialização JSON](#serialização-json)
5. [Padrões de Nomenclatura](#padrões-de-nomenclatura)
6. [Tratamento de Erros](#tratamento-de-erros)
7. [Logging](#logging)
8. [Exemplos Completos](#exemplos-completos)

---

## Visão Geral

Este documento define os padrões de código para o sistema de geração de reports do DeepBridge. Todos os desenvolvedores devem seguir estes padrões ao criar ou modificar renderers.

### Objetivos

- ✅ Código consistente e previsível
- ✅ Reduzir duplicação
- ✅ Facilitar manutenção
- ✅ Melhorar performance
- ✅ Simplificar onboarding

---

## Estrutura de Renderer

### 1. Template Básico de Renderer

Todos os renderers devem herdar de `BaseRenderer` e seguir esta estrutura:

```python
"""
Renderer para [tipo de report].
Descrição clara do propósito do renderer.
"""

import os
import logging
from typing import Dict, Any

logger = logging.getLogger("deepbridge.reports")

# Imports específicos do tipo de report
from ..transformers.[tipo]_simple import [Tipo]DataTransformerSimple


class [Tipo]RendererSimple:
    """
    Simple renderer para [tipo] experiment reports.
    """

    def __init__(self, template_manager, asset_manager):
        """
        Initialize the [tipo] renderer.

        Parameters:
        -----------
        template_manager : TemplateManager
            Manager for templates
        asset_manager : AssetManager
            Manager for assets (CSS, JS, images)
        """
        self.template_manager = template_manager
        self.asset_manager = asset_manager

        # Initialize CSS Manager (herdado ou direto)
        from ..css_manager import CSSManager
        self.css_manager = CSSManager()

        # Import data transformer
        self.data_transformer = [Tipo]DataTransformerSimple()

    def render(self, results: Dict[str, Any], file_path: str,
               model_name: str = "Model", report_type: str = "interactive",
               save_chart: bool = False) -> str:
        """
        Render [tipo] report from results data.

        Parameters:
        -----------
        results : Dict[str, Any]
            [Tipo] experiment results
        file_path : str
            Path where the HTML report will be saved
        model_name : str, optional
            Name for the report title
        report_type : str, optional
            Type of report ('interactive' or 'static')
        save_chart : bool, optional
            Whether to save charts as separate files

        Returns:
        --------
        str : Path to the generated report
        """
        logger.info(f"Generating [tipo] report to: {file_path}")

        try:
            # 1. Transform data
            report_data = self.data_transformer.transform(results, model_name)

            # 2. Load template
            template_path = self._find_template()
            template = self.template_manager.load_template(template_path)

            # 3. Get CSS content
            css_content = self._get_css_content()

            # 4. Get JS content
            js_content = self._get_js_content()

            # 5. Create context
            context = self._create_context(report_data, css_content, js_content)

            # 6. Render template
            html = self.template_manager.render_template(template, context)

            # 7. Write to file
            return self._write_html(html, file_path)

        except Exception as e:
            logger.error(f"Error generating [tipo] report: {str(e)}")
            raise

    def _find_template(self) -> str:
        """Find the template for this report type."""
        # Implementação específica
        pass

    def _get_css_content(self) -> str:
        """Get compiled CSS using CSSManager."""
        return self.css_manager.get_compiled_css('[tipo]')

    def _get_js_content(self) -> str:
        """Get inline JS content."""
        # Implementação específica
        pass

    def _create_context(self, report_data: Dict, css_content: str,
                       js_content: str) -> Dict:
        """Create template context."""
        # Implementação específica
        pass
```

### 2. Padrão para Static Renderers

Static renderers herdam de `BaseStaticRenderer`:

```python
from .base_static_renderer import BaseStaticRenderer

class Static[Tipo]Renderer:
    """
    Renderer for static [tipo] test reports using Seaborn charts.
    """

    def __init__(self, template_manager, asset_manager):
        # Initialize base static renderer
        self.base_renderer = BaseStaticRenderer(template_manager, asset_manager)
        self.template_manager = template_manager
        self.asset_manager = asset_manager

        # Import transformers
        from ...transformers.[tipo] import [Tipo]DataTransformer
        self.data_transformer = [Tipo]DataTransformer()

        # Import chart utilities
        from ...utils.seaborn_utils import SeabornChartGenerator
        self.chart_generator = SeabornChartGenerator()

    def render(self, results: Dict[str, Any], file_path: str,
               model_name: str = "Model", report_type: str = "static",
               save_chart: bool = False) -> str:
        """Render static report."""
        logger.info(f"Generating static [tipo] report to: {file_path}")

        try:
            # Find template
            template_path = self._find_template()
            template = self.template_manager.load_template(template_path)

            # Get CSS using CSSManager via base_renderer
            css_content = self.base_renderer._load_static_css_content('[tipo]')

            # Transform data
            report_data = self.data_transformer.transform(results, model_name)

            # Generate charts
            charts = self._generate_charts(report_data, save_chart)

            # Create context
            context = self.base_renderer._create_static_context(
                report_data, "[tipo]", css_content
            )
            context['charts'] = charts

            # Render and write
            html = self.template_manager.render_template(template, context)
            return self.base_renderer._write_report(html, file_path)

        except Exception as e:
            logger.error(f"Error generating static [tipo] report: {str(e)}")
            raise
```

---

## Uso de CSSManager

### ✅ Padrão Correto

**Para Simple Renderers:**
```python
# No __init__
from ..css_manager import CSSManager
self.css_manager = CSSManager()

# No método _get_css_content
def _get_css_content(self) -> str:
    """Get compiled CSS using CSSManager."""
    return self.css_manager.get_compiled_css('uncertainty')
```

**Para Static Renderers:**
```python
# Via base_renderer (já tem css_manager)
css_content = self.base_renderer._load_static_css_content('uncertainty')
```

### ❌ Padrão Incorreto (Deprecado)

```python
# NÃO USAR - Deprecado
css_content = self.asset_manager.get_combined_css_content()
css_content = self.asset_manager.get_css_content(css_dir)
```

### Fallback para Erros

Sempre tenha um fallback caso CSSManager falhe:

```python
def _get_css_content(self) -> str:
    """Get compiled CSS with fallback."""
    try:
        return self.css_manager.get_compiled_css('uncertainty')
    except Exception as e:
        logger.error(f"Error loading CSS: {e}")
        logger.warning("Using fallback minimal CSS")
        return """
        :root {
            --primary-color: #1b78de;
            --background-color: #f8f9fa;
        }
        body {
            font-family: sans-serif;
            background-color: var(--background-color);
        }
        """
```

---

## Serialização JSON

### ✅ Padrão Correto

Use sempre as utilities do módulo `json_utils.py`:

```python
from ..utils.json_utils import safe_json_dumps, format_for_javascript

# Para uso geral
json_str = safe_json_dumps(data, indent=2)

# Para embedding em JavaScript
js_data = format_for_javascript(data)
context['report_data_json'] = js_data
```

### Usando BaseRenderer

Se herdar de `BaseRenderer`, use o método herdado:

```python
# Método _safe_json_dumps já disponível
json_str = self._safe_json_dumps(report_data)
```

### ❌ Padrão Incorreto

```python
# NÃO USAR - Pode falhar com NaN/Inf
import json
json_str = json.dumps(data)

# NÃO USAR - Código duplicado
def _safe_json_dumps(self, data):
    def handler(obj):
        # ... código duplicado
    return json.dumps(data, default=handler)
```

### Tratamento de Valores Especiais

Os métodos em `json_utils.py` automaticamente tratam:

- ✅ `float('nan')` → `null`
- ✅ `float('inf')` → `null`
- ✅ `datetime` → ISO format string
- ✅ Numpy types → Python natives
- ✅ Non-serializable → `None` ou string

---

## Padrões de Nomenclatura

### Arquivos

```
[tipo]_renderer_simple.py      # Simple/interactive renderer
static_[tipo]_renderer.py      # Static renderer
[tipo]_transformer.py          # Data transformer
[tipo]_simple.py               # Simple transformer
```

### Classes

```python
class UncertaintyRendererSimple    # Simple renderer
class StaticUncertaintyRenderer    # Static renderer
class UncertaintyDataTransformer   # Transformer
```

### Métodos

```python
# Públicos
def render(...)                    # Método principal
def transform(...)                 # Transformação de dados

# Privados (prefixo _)
def _find_template(...)            # Template lookup
def _get_css_content(...)          # CSS compilation
def _get_js_content(...)           # JS content
def _create_context(...)           # Context building
def _generate_charts(...)          # Chart generation
def _write_html(...)               # File writing
```

### Variáveis

```python
# Descritivas e em snake_case
report_data = {...}                # Dados do report
css_content = "..."                # CSS compilado
template_path = "/path/to/..."     # Caminho do template
context = {...}                    # Contexto Jinja2
```

---

## Tratamento de Erros

### Padrão de Logging e Exceptions

```python
def render(self, results, file_path, model_name="Model"):
    """Render report."""
    logger.info(f"Generating {self.report_type} report to: {file_path}")

    try:
        # Lógica principal
        report_data = self.data_transformer.transform(results, model_name)
        # ...

    except FileNotFoundError as e:
        logger.error(f"Template or asset not found: {str(e)}")
        raise

    except ValueError as e:
        logger.error(f"Invalid data provided: {str(e)}")
        raise ValueError(f"Failed to generate report: {str(e)}")

    except Exception as e:
        logger.error(f"Unexpected error generating report: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise RuntimeError(f"Report generation failed: {str(e)}")
```

### Validação de Inputs

```python
def render(self, results, file_path, model_name="Model"):
    """Render report."""
    # Validar inputs críticos
    if not results:
        raise ValueError("Results dictionary cannot be empty")

    if not file_path:
        raise ValueError("File path must be specified")

    # Validar estrutura esperada
    if 'test_results' not in results:
        logger.warning("Missing 'test_results' in results data")
        # Continuar com fallback ou raise
```

---

## Logging

### Níveis de Log

```python
# INFO - Operações normais
logger.info(f"Generating uncertainty report to: {file_path}")
logger.info(f"CSS compiled successfully: {len(css)} chars")

# WARNING - Situações anormais mas recuperáveis
logger.warning(f"Template not found at {path}, trying fallback")
logger.warning("Using minimal CSS fallback")

# ERROR - Erros que impedem a operação
logger.error(f"Error generating report: {str(e)}")
logger.error(f"Required data missing: {missing_keys}")

# DEBUG - Informações detalhadas para debugging
logger.debug(f"Report data keys: {list(report_data.keys())}")
logger.debug(f"Template path resolved to: {template_path}")
```

### Formato de Mensagens

```python
# ✅ Bom - Mensagens descritivas com contexto
logger.info(f"Rendering {report_type} report for model '{model_name}'")
logger.error(f"Failed to load template at {template_path}: {str(e)}")

# ❌ Ruim - Mensagens genéricas
logger.info("Rendering report")
logger.error("Error occurred")
```

---

## Exemplos Completos

### Exemplo 1: Simple Renderer Completo

Ver arquivo de referência: `uncertainty_renderer_simple.py`

```python
"""
Simple renderer for uncertainty reports.
Uses Plotly for visualizations.
"""

import os
import logging
from typing import Dict, Any

logger = logging.getLogger("deepbridge.reports")

from ..css_manager import CSSManager


class UncertaintyRendererSimple:
    """Simple renderer for uncertainty experiment reports."""

    def __init__(self, template_manager, asset_manager):
        self.template_manager = template_manager
        self.asset_manager = asset_manager
        self.css_manager = CSSManager()

        from ..transformers.uncertainty_simple import UncertaintyDataTransformerSimple
        self.data_transformer = UncertaintyDataTransformerSimple()

    def render(self, results: Dict[str, Any], file_path: str,
               model_name: str = "Model", report_type: str = "interactive",
               save_chart: bool = False) -> str:
        """Render uncertainty report."""
        logger.info(f"Generating uncertainty report to: {file_path}")

        try:
            # 1. Transform data
            report_data = self.data_transformer.transform(results, model_name)

            # 2. Load template
            template_path = self._find_template()
            template = self.template_manager.load_template(template_path)

            # 3. Get CSS
            css_content = self._get_css_content()

            # 4. Get JS
            js_content = self._get_js_content()

            # 5. Create context
            context = {
                'model_name': report_data['model_name'],
                'report_data_json': self._safe_json_dumps(report_data),
                'css_content': css_content,
                'js_content': js_content,
                # ... outros campos
            }

            # 6. Render
            html = self.template_manager.render_template(template, context)

            # 7. Write
            output_dir = os.path.dirname(file_path)
            os.makedirs(output_dir, exist_ok=True)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html)

            logger.info(f"Report saved to: {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            raise

    def _find_template(self) -> str:
        """Find template."""
        template_path = os.path.join(
            self.template_manager.templates_dir,
            'report_types', 'uncertainty', 'interactive',
            'index_simple.html'
        )
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template not found: {template_path}")
        return template_path

    def _get_css_content(self) -> str:
        """Get CSS using CSSManager."""
        return self.css_manager.get_compiled_css('uncertainty')

    def _get_js_content(self) -> str:
        """Get inline JS."""
        return """
        // Tab navigation
        function initTabs() { /* ... */ }

        document.addEventListener('DOMContentLoaded', function() {
            initTabs();
        });
        """

    def _safe_json_dumps(self, data: Dict) -> str:
        """Safe JSON serialization."""
        from ..utils.json_utils import format_for_javascript
        return format_for_javascript(data)
```

---

## Checklist de Code Review

Ao revisar código de renderers, verifique:

- [ ] Herda de `BaseRenderer` ou `BaseStaticRenderer`?
- [ ] Usa `CSSManager` em vez de métodos legados?
- [ ] Usa `json_utils` para serialização JSON?
- [ ] Tem logging adequado (INFO, WARNING, ERROR)?
- [ ] Trata erros com try/except apropriados?
- [ ] Documentação (docstrings) completa?
- [ ] Segue nomenclatura padrão (snake_case, prefixo _)?
- [ ] Valida inputs críticos?
- [ ] Tem fallbacks para situações de erro?
- [ ] Código DRY (Don't Repeat Yourself)?

---

## Referências

- **BaseRenderer:** `deepbridge/core/experiment/report/renderers/base_renderer.py`
- **CSSManager:** `deepbridge/core/experiment/report/css_manager.py`
- **JSON Utils:** `deepbridge/core/experiment/report/utils/json_utils.py`
- **Exemplo Simple:** `uncertainty_renderer_simple.py`
- **Exemplo Static:** `static/static_uncertainty_renderer.py`

---

**Última Atualização:** 05/11/2025
**Mantido por:** Tech Lead
**Versão:** 1.0
