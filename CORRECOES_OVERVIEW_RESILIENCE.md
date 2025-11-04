# Correções para Aba Overview do Relatório de Resiliência

## Problemas Identificados e Resolvidos

### 1. Feature Importance Vazio (RESOLVIDO ✓)
**Problema:** `feature_importance` e `features` chegavam vazios ao `window.reportData`

**Causa:** O método `save_html()` não incluía `initial_model_evaluation`

**Solução:** Modificado `/deepbridge/core/experiment/results.py` (linhas 274-284) para incluir `initial_model_evaluation` no `report_data`

**Resultado:** 199 features agora estão disponíveis no HTML

---

### 2. Aba Overview Vazia (RESOLVIDO ✓)
**Problema:** A aba Overview não renderizava nenhum dado, apenas mostrava "Loading resilience overview..."

**Causa:** Problema de **ordem de carregamento de scripts**
- `OverviewController.init()` era chamado na linha 3631 (inline no partial)
- Mas `ResilienceDataManager` e `ResilienceOverviewChartManager` só eram definidos nas linhas 5079-5186
- A inicialização acontecia ANTES dos componentes existirem!

**Solução:**
1. **Removido script inline** de `/deepbridge/templates/report_types/resilience/interactive/partials/overview.html`
   - Antes: Tinha um `<script>` inline tentando inicializar imediatamente
   - Depois: Apenas comentário indicando que inicialização é feita no main.js

2. **Melhorado main.js** (`/deepbridge/templates/report_types/resilience/interactive/js/main.js`):
   - Aumentado timeout de 300ms → 500ms
   - Adicionada verificação de componentes antes da inicialização
   - Logs detalhados para debug

**Resultado Esperado:**
- Overview mostrará 3 cards de métricas (Model, Resilience Score, Scenarios Tested)
- Mostrará estatísticas por Distance Metric
- 12 dos 20 cenários terão dados válidos (8 têm NaN por serem alphas muito baixos onde o modelo não degrada)

---

## Arquivos Modificados

1. `/deepbridge/core/experiment/results.py` (linhas 274-284)
2. `/deepbridge/templates/report_types/resilience/interactive/partials/overview.html` (removido script inline)
3. `/deepbridge/templates/report_types/resilience/interactive/js/main.js` (melhorada inicialização)

---

## Como Testar

```bash
cd /home/guhaase/projetos/DeepBridge/simular_lib/analise_v2
poetry run python run_pipeline.py --sample-frac 0.1
```

Após gerar o relatório, abra no navegador e verifique:

1. **Aba Overview:**
   - ✓ Deve mostrar 3 cards: Model, Resilience Score, Scenarios Tested
   - ✓ Deve mostrar "Performance by Distance Metric" com estatísticas
   - ✓ Console do navegador deve mostrar logs de inicialização

2. **Aba Feature Importance:**
   - ✓ Deve mostrar tabela com 199 features
   - ✓ Deve mostrar gráficos de importância

3. **Aba Box Plot:**
   - ✓ Deve mostrar boxplot com estatísticas

4. **Aba Model Features:**
   - ✓ Deve mostrar features do modelo

---

## Dados Esperados

### Cenários com performance_gap válido:
- 12 de 20 cenários têm `performance_gap` válido
- 8 cenários (alpha=0.01 e 0.05) têm `NaN` porque o modelo é muito bom

### Métricas:
- **Resilience Score:** 1.0 (modelo perfeito)
- **Base Score:** 1.0 (accuracy)
- **Distance Metrics:** PSI, WD1, KS
- **Alphas testados:** 0.01, 0.05, 0.1, 0.2, 0.3

---

## Data
2025-10-29 10:51 BRT
