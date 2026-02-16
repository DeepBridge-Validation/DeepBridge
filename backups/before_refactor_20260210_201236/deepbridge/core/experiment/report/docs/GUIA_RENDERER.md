# Guia Passo-a-Passo: Como Criar um Novo Renderer

**Versão:** 1.0
**Data:** 05/11/2025
**Tempo estimado:** 2-4 horas

---

## 📋 Índice

1. [Visão Geral](#visão-geral)
2. [Pré-requisitos](#pré-requisitos)
3. [Passo 1: Planejar o Renderer](#passo-1-planejar-o-renderer)
4. [Passo 2: Criar o Transformer](#passo-2-criar-o-transformer)
5. [Passo 3: Criar o Template](#passo-3-criar-o-template)
6. [Passo 4: Implementar o Renderer](#passo-4-implementar-o-renderer)
7. [Passo 5: Testar](#passo-5-testar)
8. [Passo 6: Integrar](#passo-6-integrar)
9. [Troubleshooting](#troubleshooting)

---

## Visão Geral

Este guia mostra como criar um novo tipo de renderer para reports no DeepBridge. Usaremos como exemplo a criação de um renderer hipotético chamado "ExplainabilityRenderer".

### O que você vai criar

1. **Transformer** - Transforma dados brutos em formato para template
2. **Template HTML** - Define a estrutura visual do report
3. **Renderer** - Orquestra tudo e gera o HTML final
4. **Testes** - Valida que tudo funciona

---

## Pré-requisitos

Antes de começar, certifique-se de ter:

- [ ] Python 3.8+ instalado
- [ ] DeepBridge clonado e configurado
- [ ] Familiaridade com Python e Jinja2
- [ ] Lido o `PADROES_CODIGO.md`

### Estrutura do Sistema

```
deepbridge/core/experiment/report/
├── renderers/
│   ├── base_renderer.py              # Base class
│   ├── [novo]_renderer_simple.py     # Seu renderer aqui
│   └── static/
│       └── static_[novo]_renderer.py # Static version
├── transformers/
│   └── [novo]_simple.py              # Seu transformer aqui
├── css_manager.py                    # Gerenciador de CSS
└── utils/
    └── json_utils.py                 # Utilities JSON

templates/
└── report_types/
    └── [novo]/                       # Seus templates aqui
        └── interactive/
            ├── index_simple.html
            └── [novo]_custom.css
```

---

## Passo 1: Planejar o Renderer

### 1.1 Definir Requisitos

Responda estas perguntas:

1. **Qual o propósito do report?**
   - Exemplo: "Mostrar métricas de explainability do modelo"

2. **Quais dados são necessários?**
   - Exemplo: "SHAP values, feature importance, lime explanations"

3. **Quais visualizações?**
   - Exemplo: "Feature importance chart, SHAP waterfall, decision plot"

4. **Tipo de report?**
   - [ ] Interactive (Plotly)
   - [ ] Static (Matplotlib/Seaborn)
   - [ ] Ambos

### 1.2 Criar Checklist

```markdown
- [ ] Transformer para processar dados
- [ ] Template HTML Jinja2
- [ ] CSS customizado (opcional)
- [ ] JavaScript para interatividade (se interactive)
- [ ] Renderer simple
- [ ] Renderer static (opcional)
- [ ] Testes unitários
- [ ] Testes de integração
```

---

## Passo 2: Criar o Transformer

### 2.1 Criar o Arquivo

Crie `deepbridge/core/experiment/report/transformers/explainability_simple.py`:

```python
"""
Data transformer for Explainability reports.
Transforms raw explainability results into template-ready format.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger("deepbridge.reports")


class ExplainabilityDataTransformerSimple:
    """
    Transformer for explainability experiment results.
    """

    def transform(self, results: Dict[str, Any], model_name: str = "Model") -> Dict[str, Any]:
        """
        Transform explainability results into report data.

        Parameters:
        -----------
        results : Dict[str, Any]
            Raw experiment results containing:
            - shap_values: SHAP values for model
            - feature_importance: Feature importance scores
            - lime_explanations: LIME explanations
        model_name : str, optional
            Name of the model

        Returns:
        --------
        Dict[str, Any] : Transformed data for template
        """
        logger.info("Transforming explainability data for report")

        try:
            # Extract data from results
            shap_values = results.get('shap_values', {})
            feature_importance = results.get('feature_importance', {})
            lime_explanations = results.get('lime_explanations', {})

            # Transform to template format
            report_data = {
                # Metadata
                'model_name': model_name,
                'model_type': results.get('model_type', 'Unknown'),
                'timestamp': self._get_timestamp(),

                # SHAP data
                'shap_values': self._process_shap_values(shap_values),

                # Feature importance
                'feature_importance': self._process_feature_importance(feature_importance),

                # LIME
                'lime_explanations': self._process_lime(lime_explanations),

                # Charts (format for Plotly/template)
                'charts': self._prepare_charts(
                    shap_values, feature_importance, lime_explanations
                )
            }

            logger.info(f"Transformation complete for model: {model_name}")
            return report_data

        except Exception as e:
            logger.error(f"Error transforming explainability data: {str(e)}")
            raise ValueError(f"Failed to transform data: {str(e)}")

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def _process_shap_values(self, shap_values: Dict) -> Dict:
        """Process SHAP values for template."""
        if not shap_values:
            return {}

        # Transform SHAP values to template format
        return {
            'values': shap_values.get('values', []),
            'base_value': shap_values.get('base_value', 0.0),
            'features': shap_values.get('features', [])
        }

    def _process_feature_importance(self, feature_importance: Dict) -> Dict:
        """Process feature importance for template."""
        if not feature_importance:
            return {}

        # Sort by importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        return {
            'features': [f[0] for f in sorted_features],
            'importances': [f[1] for f in sorted_features]
        }

    def _process_lime(self, lime_explanations: Dict) -> Dict:
        """Process LIME explanations for template."""
        # Implementation specific to your needs
        return lime_explanations

    def _prepare_charts(self, shap_values, feature_importance, lime) -> Dict:
        """Prepare chart data in Plotly format."""
        charts = {}

        # Feature importance chart
        if feature_importance:
            fi_processed = self._process_feature_importance(feature_importance)
            charts['feature_importance'] = {
                'data': [{
                    'type': 'bar',
                    'x': fi_processed.get('importances', []),
                    'y': fi_processed.get('features', []),
                    'orientation': 'h'
                }],
                'layout': {
                    'title': 'Feature Importance',
                    'xaxis': {'title': 'Importance'},
                    'yaxis': {'title': 'Features'}
                }
            }

        # Add more charts as needed...

        return charts
```

### 2.2 Testar o Transformer

Crie um teste rápido:

```python
# test_transformer.py
from transformers.explainability_simple import ExplainabilityDataTransformerSimple

transformer = ExplainabilityDataTransformerSimple()

# Mock data
results = {
    'shap_values': {'values': [0.5, -0.3, 0.8], 'features': ['A', 'B', 'C']},
    'feature_importance': {'A': 0.5, 'B': 0.3, 'C': 0.2},
    'model_type': 'RandomForest'
}

report_data = transformer.transform(results, 'MyModel')
print(report_data.keys())
```

---

## Passo 3: Criar o Template

### 3.1 Estrutura de Diretórios

Crie a estrutura de templates:

```bash
mkdir -p templates/report_types/explainability/interactive
touch templates/report_types/explainability/interactive/index_simple.html
touch templates/report_types/explainability/interactive/explainability_custom.css
```

### 3.2 Template HTML Básico

Crie `templates/report_types/explainability/interactive/index_simple.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Explainability Report - {{ model_name }}</title>

    <!-- Favicon -->
    <link rel="icon" type="image/x-icon" href="data:image/x-icon;base64,{{ favicon_base64 }}">

    <!-- Plotly for charts -->
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>

    <!-- Inline CSS -->
    <style>
        {{ css_content|safe }}
    </style>
</head>
<body>
    <!-- Header -->
    <header class="header">
        <div class="logo-container">
            <img src="data:image/png;base64,{{ logo }}" alt="DeepBridge Logo" class="logo">
        </div>
        <h1>Explainability Analysis Report</h1>
        <p class="subtitle">{{ model_name }} - {{ timestamp }}</p>
    </header>

    <!-- Main Content -->
    <div class="report-container">
        <!-- Summary Section -->
        <section class="section">
            <h2 class="section-title">Summary</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>Model Type</h3>
                    <p class="metric-value">{{ report_data.model_type }}</p>
                </div>
                <div class="metric-card">
                    <h3>Features Analyzed</h3>
                    <p class="metric-value">{{ report_data.feature_importance.features|length }}</p>
                </div>
            </div>
        </section>

        <!-- Feature Importance Chart -->
        <section class="section">
            <h2 class="section-title">Feature Importance</h2>
            <div id="chart-feature-importance" class="chart-container"></div>
        </section>

        <!-- SHAP Values -->
        <section class="section">
            <h2 class="section-title">SHAP Analysis</h2>
            <div id="chart-shap" class="chart-container"></div>
        </section>
    </div>

    <!-- Footer -->
    <footer class="footer">
        <p>Generated with DeepBridge - {{ current_year }}</p>
    </footer>

    <!-- Inline JavaScript -->
    <script>
        {{ js_content|safe }}

        // Report data
        const reportData = {{ report_data_json|safe }};

        // Render charts on load
        document.addEventListener('DOMContentLoaded', function() {
            renderCharts(reportData.charts);
        });

        function renderCharts(charts) {
            // Feature importance
            if (charts.feature_importance) {
                Plotly.newPlot(
                    'chart-feature-importance',
                    charts.feature_importance.data,
                    charts.feature_importance.layout,
                    {responsive: true}
                );
            }

            // Add more charts as needed...
        }
    </script>
</body>
</html>
```

### 3.3 CSS Customizado (Opcional)

Crie `templates/report_types/explainability/interactive/explainability_custom.css`:

```css
/* Custom styles for Explainability reports */

.explainability-specific {
    /* Seus estilos específicos aqui */
}

.shap-waterfall {
    max-width: 100%;
    margin: 2rem 0;
}

.feature-importance-bar {
    transition: opacity 0.3s;
}

.feature-importance-bar:hover {
    opacity: 0.8;
}
```

---

## Passo 4: Implementar o Renderer

### 4.1 Criar o Arquivo

Crie `deepbridge/core/experiment/report/renderers/explainability_renderer_simple.py`:

```python
"""
Simple renderer for explainability reports.
Uses Plotly for visualizations and single-page template approach.
"""

import os
import logging
from typing import Dict, Any

logger = logging.getLogger("deepbridge.reports")

# Import CSS Manager
from ..css_manager import CSSManager


class ExplainabilityRendererSimple:
    """
    Simple renderer for explainability experiment reports.
    """

    def __init__(self, template_manager, asset_manager):
        """
        Initialize the explainability renderer.

        Parameters:
        -----------
        template_manager : TemplateManager
            Manager for templates
        asset_manager : AssetManager
            Manager for assets (CSS, JS, images)
        """
        self.template_manager = template_manager
        self.asset_manager = asset_manager

        # Initialize CSS Manager
        self.css_manager = CSSManager()

        # Import data transformer
        from ..transformers.explainability_simple import ExplainabilityDataTransformerSimple
        self.data_transformer = ExplainabilityDataTransformerSimple()

    def render(self, results: Dict[str, Any], file_path: str,
               model_name: str = "Model", report_type: str = "interactive",
               save_chart: bool = False) -> str:
        """
        Render explainability report from results data.

        Parameters:
        -----------
        results : Dict[str, Any]
            Explainability experiment results
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
        logger.info(f"Generating explainability report to: {file_path}")

        try:
            # 1. Transform the data
            report_data = self.data_transformer.transform(results, model_name)

            # 2. Load template
            template_path = self._find_template()
            logger.info(f"Using template: {template_path}")
            template = self.template_manager.load_template(template_path)

            # 3. Get CSS content
            css_content = self._get_css_content()

            # 4. Get JS content
            js_content = self._get_js_content()

            # 5. Prepare context for template
            context = {
                'model_name': report_data['model_name'],
                'model_type': report_data['model_type'],
                'timestamp': report_data['timestamp'],
                'report_title': 'Explainability Analysis Report',

                # Data as JSON for JavaScript access
                'report_data': report_data,
                'report_data_json': self._safe_json_dumps(report_data),

                # Assets
                'css_content': css_content,
                'js_content': js_content,
                'logo': self.asset_manager.get_logo_base64(),
                'favicon_base64': self.asset_manager.get_favicon_base64(),

                # Metadata
                'current_year': datetime.datetime.now().year
            }

            # 6. Render template
            html = self.template_manager.render_template(template, context)

            # 7. Write to file
            output_dir = os.path.dirname(file_path)
            os.makedirs(output_dir, exist_ok=True)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html)

            logger.info(f"Explainability report saved to: {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"Error generating explainability report: {str(e)}")
            raise ValueError(f"Failed to generate report: {str(e)}")

    def _find_template(self) -> str:
        """Find the simple template."""
        template_path = os.path.join(
            self.template_manager.templates_dir,
            'report_types',
            'explainability',
            'interactive',
            'index_simple.html'
        )

        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template not found: {template_path}")

        return template_path

    def _get_css_content(self) -> str:
        """
        Get CSS content using CSSManager.

        Returns:
        --------
        str : Compiled CSS (base + components + custom)
        """
        try:
            # Use CSSManager to compile CSS layers
            compiled_css = self.css_manager.get_compiled_css('explainability')
            logger.info(f"CSS compiled successfully: {len(compiled_css)} chars")
            return compiled_css
        except Exception as e:
            logger.error(f"Error loading CSS: {str(e)}")
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

    def _get_js_content(self) -> str:
        """Get inline JS content (minimal - just tab navigation if needed)."""
        js = """
        // Initialize on load
        console.log('Explainability report initialized');

        // Add any custom JavaScript here
        """
        return js

    def _safe_json_dumps(self, data: Dict) -> str:
        """Safely serialize data to JSON."""
        from ..utils.json_utils import format_for_javascript
        return format_for_javascript(data)
```

---

## Passo 5: Testar

### 5.1 Teste Manual

Crie um script de teste `test_explainability_renderer.py`:

```python
"""
Test script for ExplainabilityRenderer.
"""

import os
import sys

# Add project to path
sys.path.insert(0, os.path.abspath('../../../..'))

from deepbridge.core.experiment.report.renderers.explainability_renderer_simple import ExplainabilityRendererSimple
from deepbridge.core.experiment.report.template_manager import TemplateManager
from deepbridge.core.experiment.report.asset_manager import AssetManager


def test_explainability_renderer():
    """Test explainability renderer with mock data."""

    # Setup managers
    templates_dir = os.path.abspath('../../../../templates')
    template_manager = TemplateManager(templates_dir)
    asset_manager = AssetManager(templates_dir)

    # Create renderer
    renderer = ExplainabilityRendererSimple(template_manager, asset_manager)

    # Mock results
    mock_results = {
        'shap_values': {
            'values': [0.5, -0.3, 0.8, 0.2],
            'features': ['Feature A', 'Feature B', 'Feature C', 'Feature D'],
            'base_value': 0.5
        },
        'feature_importance': {
            'Feature A': 0.35,
            'Feature B': 0.25,
            'Feature C': 0.20,
            'Feature D': 0.20
        },
        'lime_explanations': {},
        'model_type': 'RandomForestClassifier'
    }

    # Generate report
    output_path = '/tmp/test_explainability_report.html'
    result = renderer.render(mock_results, output_path, model_name='TestModel')

    print(f"✅ Report generated successfully: {result}")
    print(f"📂 Open file://{result} to view")

    return result


if __name__ == '__main__':
    test_explainability_renderer()
```

Execute o teste:

```bash
cd deepbridge/core/experiment/report/renderers
python test_explainability_renderer.py
```

### 5.2 Testes Unitários

Crie testes unitários em `tests/report/test_explainability_renderer.py`:

```python
import pytest
import os
from unittest.mock import Mock, patch
from deepbridge.core.experiment.report.renderers.explainability_renderer_simple import ExplainabilityRendererSimple


class TestExplainabilityRenderer:
    """Tests for ExplainabilityRenderer."""

    @pytest.fixture
    def renderer(self):
        """Create renderer instance with mocks."""
        mock_template_mgr = Mock()
        mock_asset_mgr = Mock()
        mock_asset_mgr.get_logo_base64.return_value = "fake_logo_b64"
        mock_asset_mgr.get_favicon_base64.return_value = "fake_favicon_b64"

        return ExplainabilityRendererSimple(mock_template_mgr, mock_asset_mgr)

    def test_init(self, renderer):
        """Test renderer initialization."""
        assert renderer.template_manager is not None
        assert renderer.asset_manager is not None
        assert renderer.css_manager is not None
        assert renderer.data_transformer is not None

    def test_render_success(self, renderer, tmp_path):
        """Test successful report rendering."""
        # Mock data
        results = {
            'shap_values': {'values': [0.5], 'features': ['A']},
            'feature_importance': {'A': 0.5},
            'model_type': 'RF'
        }

        output_file = tmp_path / "report.html"

        # Mock methods
        with patch.object(renderer, '_find_template', return_value='mock.html'):
            with patch.object(renderer.template_manager, 'load_template'):
                with patch.object(renderer.template_manager, 'render_template', return_value='<html></html>'):
                    result = renderer.render(results, str(output_file))

        assert os.path.exists(result)
        assert result == str(output_file)

    def test_render_missing_data(self, renderer, tmp_path):
        """Test rendering with missing data."""
        results = {}  # Empty results
        output_file = tmp_path / "report.html"

        with pytest.raises((ValueError, KeyError)):
            renderer.render(results, str(output_file))
```

Execute os testes:

```bash
pytest tests/report/test_explainability_renderer.py -v
```

---

## Passo 6: Integrar

### 6.1 Registrar no ReportManager

Atualize `report_manager.py` para incluir o novo renderer:

```python
# Em report_manager.py

# Import the new renderer
from .renderers.explainability_renderer_simple import ExplainabilityRendererSimple

class ReportManager:
    def __init__(self, templates_dir: str):
        # ... existing code ...

        # Register explainability renderer
        self.renderers['explainability'] = {
            'simple': ExplainabilityRendererSimple(
                self.template_manager,
                self.asset_manager
            )
        }
```

### 6.2 Atualizar __init__.py

Adicione ao `__init__.py`:

```python
# deepbridge/core/experiment/report/renderers/__init__.py

from .explainability_renderer_simple import ExplainabilityRendererSimple

__all__ = [
    # ... existing ...
    'ExplainabilityRendererSimple',
]
```

### 6.3 Documentar

Adicione ao README ou documentação do projeto:

```markdown
## Explainability Reports

O DeepBridge agora suporta reports de explainability!

### Uso

```python
from deepbridge.core.experiment.report import ReportManager

# Initialize
report_mgr = ReportManager('/path/to/templates')

# Generate explainability report
results = {
    'shap_values': {...},
    'feature_importance': {...}
}

report_mgr.generate_report(
    report_type='explainability',
    results=results,
    output_path='explainability_report.html',
    model_name='MyModel'
)
```
```

---

## Troubleshooting

### Problema: Template não encontrado

**Erro:**
```
FileNotFoundError: Template not found: /path/to/templates/.../index_simple.html
```

**Solução:**
1. Verifique se criou o diretório correto: `templates/report_types/explainability/interactive/`
2. Verifique o nome do arquivo: deve ser exatamente `index_simple.html`
3. Use caminhos absolutos para debug:
   ```python
   print(f"Looking for template at: {template_path}")
   print(f"File exists: {os.path.exists(template_path)}")
   ```

### Problema: CSS não carrega

**Erro:**
```
CSS compiled with 0 chars / No styles applied
```

**Solução:**
1. Verifique se `base_styles.css` e `report_components.css` existem
2. Crie arquivo custom CSS mesmo que vazio:
   ```bash
   touch templates/report_types/explainability/interactive/explainability_custom.css
   ```
3. Use fallback CSS durante desenvolvimento

### Problema: JSON serialization error

**Erro:**
```
TypeError: Object of type 'float64' is not JSON serializable
```

**Solução:**
Use sempre `json_utils`:
```python
from ..utils.json_utils import format_for_javascript
json_str = format_for_javascript(data)
```

### Problema: Charts não aparecem

**Possíveis causas:**
1. JavaScript com erro (check browser console)
2. Dados no formato errado para Plotly
3. Div com ID incorreto

**Debug:**
```javascript
console.log('Report data:', reportData);
console.log('Charts:', reportData.charts);
console.log('Chart element:', document.getElementById('chart-feature-importance'));
```

---

## Checklist Final

Antes de considerar completo, verifique:

- [ ] Transformer criado e testado
- [ ] Template HTML criado
- [ ] CSS customizado (se necessário)
- [ ] Renderer implementado seguindo `PADROES_CODIGO.md`
- [ ] Testes unitários escritos
- [ ] Teste manual executado e report gerado
- [ ] Integrado no ReportManager
- [ ] Documentação atualizada
- [ ] Code review solicitado
- [ ] PR criado

---

## Próximos Passos

Após criar o renderer simple, você pode:

1. **Criar versão static** (usando Matplotlib/Seaborn)
2. **Adicionar mais visualizações**
3. **Otimizar performance**
4. **Adicionar testes de integração**
5. **Criar documentação de usuário**

---

## Recursos Adicionais

- **Exemplos de referência:**
  - `uncertainty_renderer_simple.py`
  - `robustness_renderer_simple.py`
  - `resilience_renderer_simple.py`

- **Documentação:**
  - `PADROES_CODIGO.md` - Padrões de código
  - `ANALISE_ARQUITETURA_REPORTS.md` - Arquitetura do sistema

- **Ferramentas:**
  - [Plotly Documentation](https://plotly.com/python/)
  - [Jinja2 Template Designer](https://jinja.palletsprojects.com/)

---

**Última Atualização:** 05/11/2025
**Mantido por:** Tech Lead
**Versão:** 1.0

**Dúvidas?** Consulte o Tech Lead ou abra uma issue no GitHub.
