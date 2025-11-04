# Problema Resolvido: Abas Vazias em Relatórios de Resiliência

## Problema
As abas Overview, Box Plot, Feature Importance e Model Features dos relatórios de resiliência não renderizavam dados porque `feature_importance` e `features` chegavam vazios ao `window.reportData`.

## Causa Raiz
O método `save_html()` em `/deepbridge/core/experiment/results.py` (linha 268) pegava apenas `result.to_dict()['results']`, que contém os `test_results`, MAS não incluía o `initial_model_evaluation` que contém os dados de `feature_importance`.

## Solução
Modificado o método `save_html()` (linhas 274-284) para:
1. Buscar `initial_results` de `self.results['initial_results']`  
2. Incluir como `initial_model_evaluation` no `report_data`
3. O transformer então consegue extrair os 199 features corretamente

## Arquivos Modificados
- `/home/guhaase/projetos/DeepBridge/deepbridge/core/experiment/results.py` (linhas 274-284)

## Como Usar
Simplesmente execute o pipeline novamente:

```bash
cd /home/guhaase/projetos/DeepBridge/simular_lib/analise_v2
poetry run python run_pipeline.py --sample-frac 0.1
```

Os relatórios serão gerados CORRETAMENTE com todos os dados!

## Testes Realizados
✓ Teste direto do transformer: 199 features extraídos  
✓ Teste com save_html: 199 features no HTML  
✓ report_resilience_SAVEHTML_TEST.html: SUCESSO  

## Data
2025-10-29 10:01 BRT
