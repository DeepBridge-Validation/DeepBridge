# 📋 Plano de Refatoração - Relatório de Resiliência

## 🎯 Objetivo
Refatorar completamente o relatório de resiliência para seguir o padrão do relatório de destilação que está funcionando corretamente.

---

## 📊 Análise do Padrão Atual (Destilação)

### Estrutura que Funciona:
```
1. AutoDistiller.generate_report()
   ↓
2. Prepara report_data dict
   ↓
3. DistillationRenderer.render(report_data)
   ↓
4. DistillationDataTransformer.transform(results)
   ↓
5. Template único: index.html + partials/
   ↓
6. CSS/JS embutidos (sem arquivos externos)
   ↓
7. Dados como JSON via context['report_data_json']
   ↓
8. Plotly para visualizações
```

### Características-chave:
- ✅ **Sem scripts inline** nos partials
- ✅ **Dados centralizados** em `report_data_json`
- ✅ **Inicialização única** no index.html
- ✅ **CSS/JS embutidos** no HTML final
- ✅ **Template simples** sem lógica complexa
- ✅ **Transformador de dados** separa lógica de apresentação

---

## 🔴 Problemas Atuais (Resiliência)

### 1. Múltiplas inicializações
- Scripts inline em overview.html, details.html
- Race conditions entre componentes
- DOMContentLoaded executando antes dos componentes carregarem

### 2. Dados fragmentados
- `window.reportData` e `window.reportConfig`
- JavaScript buscando de lugares diferentes
- feature_importance vazio por buscar em reportConfig primeiro

### 3. Complexidade desnecessária
- Muitos arquivos JS separados (details.js, overview.js, features.js, etc.)
- Controllers duplicados
- Inicialização em múltiplos lugares

---

## ✅ Solução: Refatoração Completa

### Fase 1: Criar Novo Transformador (Simples)
**Arquivo:** `/deepbridge/core/experiment/report/transformers/resilience_new.py`

**Responsabilidades:**
- Receber `results` dict do Experiment
- Extrair dados de `initial_model_evaluation`
- Extrair dados de `test_results.primary_model`
- Estruturar dados para o template
- Calcular estatísticas (médias, máximos, mínimos)
- Preparar dados para gráficos Plotly

**Saída:**
```python
{
    'model_name': 'Model',
    'model_type': 'LogisticRegression',
    'resilience_score': 1.0,
    'summary': {
        'total_scenarios': 20,
        'valid_scenarios': 12,
        'avg_performance_gap': 0.0
    },
    'scenarios': [
        {
            'name': 'Scenario 1',
            'alpha': 0.01,
            'metric': 'PSI',
            'performance_gap': null,
            'baseline': null,
            'target': 1.0
        },
        ...
    ],
    'features': {
        'total': 199,
        'importance': {
            'feature1': 0.05,
            ...
        },
        'model_importance': {
            'feature1': 0.03,
            ...
        },
        'top_10': [
            {'name': 'feature1', 'importance': 0.05, 'model_importance': 0.03},
            ...
        ]
    },
    'charts': {
        'overview': {...},  # Dados prontos para Plotly
        'scenarios_by_alpha': {...},
        'feature_importance': {...},
        'boxplot': {...}
    }
}
```

---

### Fase 2: Simplificar Renderer
**Arquivo:** `/deepbridge/core/experiment/report/renderers/resilience_renderer_new.py`

**Mudanças:**
- Seguir padrão de `distillation_renderer.py`
- Usar o novo transformador
- Preparar dados para charts no backend (não no frontend!)
- Context simples para o template

**Código base:**
```python
class ResilienceRendererNew:
    def __init__(self, template_manager, asset_manager):
        self.template_manager = template_manager
        self.asset_manager = asset_manager
        self.data_transformer = ResilienceDataTransformerNew()

    def render(self, results, file_path, model_name="Model", report_type="interactive"):
        # 1. Transform data
        report_data = self.data_transformer.transform(results)

        # 2. Load template
        template = self.template_manager.load_template("resilience/interactive/index.html")

        # 3. Load CSS/JS
        css_content = self._load_css()
        js_content = self._load_js()

        # 4. Create context
        context = {
            'report_data_json': json.dumps(report_data),
            'css_content': css_content,
            'js_content': js_content,
            'model_name': model_name,
            ...
        }

        # 5. Render
        html = template.render(context)

        # 6. Save
        with open(file_path, 'w') as f:
            f.write(html)

        return file_path
```

---

### Fase 3: Novo Template Simples
**Arquivo:** `/deepbridge/templates/report_types/resilience/interactive/index_new.html`

**Estrutura:**
```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Resilience Report - {{ model_name }}</title>
    <script src="https://cdn.plot.ly/plotly-2.29.1.min.js"></script>
    <style>{{ css_content }}</style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        {% include 'common/header.html' %}

        <!-- Navigation tabs -->
        {% include 'common/navigation.html' %}

        <!-- Tab: Overview -->
        <div id="overview" class="tab-content active">
            <h2>Overview</h2>
            <div id="overview-summary"></div>
            <div id="overview-chart"></div>
        </div>

        <!-- Tab: Scenarios -->
        <div id="scenarios" class="tab-content">
            <h2>Shift Scenarios</h2>
            <div id="scenarios-table"></div>
            <div id="scenarios-chart"></div>
        </div>

        <!-- Tab: Features -->
        <div id="features" class="tab-content">
            <h2>Feature Importance</h2>
            <div id="features-table"></div>
            <div id="features-chart"></div>
        </div>

        <!-- Footer -->
        {% include 'common/footer.html' %}
    </div>

    <!-- DADOS CENTRALIZADOS -->
    <script>
        window.reportData = {{ report_data_json|safe }};
    </script>

    <!-- INICIALIZAÇÃO ÚNICA -->
    <script>{{ js_content }}</script>

    <script>
        // Inicializar apenas DEPOIS que tudo estiver carregado
        document.addEventListener('DOMContentLoaded', function() {
            // Aguardar para garantir que Plotly e dados estão prontos
            setTimeout(function() {
                if (typeof Plotly !== 'undefined' && window.reportData) {
                    // Inicializar abas
                    initializeTabs();

                    // Renderizar cada aba
                    renderOverview();
                    renderScenarios();
                    renderFeatures();
                } else {
                    console.error('Plotly or reportData not available');
                }
            }, 100);
        });

        function initializeTabs() {
            // Lógica simples de tabs
            const tabButtons = document.querySelectorAll('.tab-btn');
            tabButtons.forEach(btn => {
                btn.addEventListener('click', () => {
                    const targetId = btn.getAttribute('data-tab');
                    showTab(targetId);
                });
            });
        }

        function showTab(tabId) {
            // Esconder todas
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            // Mostrar a selecionada
            document.getElementById(tabId).classList.add('active');
        }

        function renderOverview() {
            const data = window.reportData;

            // Summary cards
            const summaryHtml = `
                <div class="metrics-grid">
                    <div class="metric-card">
                        <span class="label">Resilience Score</span>
                        <span class="value">${data.resilience_score.toFixed(4)}</span>
                    </div>
                    <div class="metric-card">
                        <span class="label">Total Scenarios</span>
                        <span class="value">${data.summary.total_scenarios}</span>
                    </div>
                    <div class="metric-card">
                        <span class="label">Valid Scenarios</span>
                        <span class="value">${data.summary.valid_scenarios}</span>
                    </div>
                </div>
            `;
            document.getElementById('overview-summary').innerHTML = summaryHtml;

            // Chart using Plotly
            Plotly.newPlot('overview-chart', data.charts.overview.data, data.charts.overview.layout);
        }

        function renderScenarios() {
            // Tabela de scenarios
            let tableHtml = '<table><thead><tr><th>Name</th><th>Alpha</th><th>Metric</th><th>Gap</th></tr></thead><tbody>';
            window.reportData.scenarios.forEach(sc => {
                tableHtml += `<tr>
                    <td>${sc.name}</td>
                    <td>${sc.alpha}</td>
                    <td>${sc.metric}</td>
                    <td>${sc.performance_gap !== null ? sc.performance_gap.toFixed(4) : 'N/A'}</td>
                </tr>`;
            });
            tableHtml += '</tbody></table>';
            document.getElementById('scenarios-table').innerHTML = tableHtml;

            // Chart
            Plotly.newPlot('scenarios-chart',
                window.reportData.charts.scenarios_by_alpha.data,
                window.reportData.charts.scenarios_by_alpha.layout
            );
        }

        function renderFeatures() {
            // Tabela de features
            let tableHtml = '<table><thead><tr><th>Feature</th><th>Importance</th></tr></thead><tbody>';
            window.reportData.features.top_10.forEach(feat => {
                tableHtml += `<tr>
                    <td>${feat.name}</td>
                    <td>${feat.importance.toFixed(4)}</td>
                </tr>`;
            });
            tableHtml += '</tbody></table>';
            document.getElementById('features-table').innerHTML = tableHtml;

            // Chart
            Plotly.newPlot('features-chart',
                window.reportData.charts.feature_importance.data,
                window.reportData.charts.feature_importance.layout
            );
        }
    </script>
</body>
</html>
```

---

### Fase 4: CSS Simples e Limpo
**Arquivo:** CSS embutido no template

**Características:**
- Grid responsivo
- Cards para métricas
- Tabelas estilizadas
- Tabs simples
- **Sem complexidade desnecessária**

---

### Fase 5: Integração com Experiment
**Arquivo:** `/deepbridge/core/experiment/experiment.py`

**Adicionar método:**
```python
def generate_resilience_report(self, output_path, model_name="Model"):
    """
    Generate resilience report using new refactored renderer.
    """
    # Get resilience results
    resilience_result = self.results.get('resilience')
    if not resilience_result:
        raise ValueError("No resilience results available")

    # Prepare data structure
    report_data = {
        'test_results': resilience_result.results,
        'initial_model_evaluation': self.initial_results
    }

    # Use new renderer
    from deepbridge.core.experiment.report.renderers.resilience_renderer_new import ResilienceRendererNew
    from deepbridge.core.experiment.report.template_manager import TemplateManager
    from deepbridge.core.experiment.report.asset_manager import AssetManager

    templates_dir = ...  # path to templates
    template_manager = TemplateManager(templates_dir)
    asset_manager = AssetManager(templates_dir)
    renderer = ResilienceRendererNew(template_manager, asset_manager)

    return renderer.render(report_data, output_path, model_name)
```

---

## 📅 Cronograma de Implementação

### Etapa 1: Transformador (30 min)
- [x] Criar `resilience_new_transformer.py`
- [x] Implementar método `transform()`
- [x] Testar com dados reais
- [x] Validar estrutura de saída

### Etapa 2: Renderer (20 min)
- [ ] Criar `resilience_renderer_new.py`
- [ ] Seguir padrão de distillation_renderer
- [ ] Integrar com transformador
- [ ] Testar geração básica

### Etapa 3: Template (40 min)
- [ ] Criar `index_new.html` simples
- [ ] Implementar tabs básicas
- [ ] Adicionar visualizações Plotly
- [ ] CSS inline simples

### Etapa 4: Testes (20 min)
- [ ] Gerar relatório de teste
- [ ] Verificar todas as abas
- [ ] Validar gráficos Plotly
- [ ] Comparar com distillation report

### Etapa 5: Integração (10 min)
- [ ] Substituir chamada antiga por nova
- [ ] Atualizar run_pipeline.py
- [ ] Testar pipeline completo
- [ ] Documentar mudanças

**Tempo total estimado:** 2 horas

---

## ✅ Critérios de Sucesso

1. ✅ Relatório gerado sem erros
2. ✅ Todas as abas renderizando dados
3. ✅ Gráficos Plotly funcionando
4. ✅ 199 features visíveis
5. ✅ 20 scenarios visíveis
6. ✅ Overview com cards de métricas
7. ✅ Sem "Loading..." permanente
8. ✅ Sem race conditions
9. ✅ Código limpo e maintível
10. ✅ Seguindo padrão do distillation

---

## 🚀 Começar?

Pronto para começar a implementação?
Digite "sim" para iniciar pela Etapa 1: Transformador

---

**Data:** 2025-10-29
**Autor:** Claude Code
**Status:** Planejamento completo ✅
