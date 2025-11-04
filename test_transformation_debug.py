#!/usr/bin/env python3
"""
Script de debug para testar a transformação de dados de resiliência.
"""
import sys
import json
sys.path.insert(0, '/home/guhaase/projetos/DeepBridge')

from deepbridge.core.experiment.report.transformers.resilience import ResilienceDataTransformer
from deepbridge.core.experiment.report.renderers.resilience_renderer import ResilienceRenderer
from deepbridge.core.experiment.report.template_manager import TemplateManager
from deepbridge.core.experiment.report.asset_manager import AssetManager

print("="*70)
print("TESTE DE TRANSFORMAÇÃO DE DADOS - RESILIÊNCIA")
print("="*70)

# Carregar o JSON de resultados
json_path = '/home/guhaase/projetos/DeepBridge/simular_lib/analise_v2/results/resilience_results_pixpj.json'
print(f"\n1. Carregando JSON: {json_path}")

with open(json_path) as f:
    results = json.load(f)

print(f"   ✓ JSON carregado")
print(f"   Keys: {list(results.keys())}")

# Testar o transformer diretamente
print(f"\n2. Testando transformer diretamente...")
transformer = ResilienceDataTransformer()
report_data = transformer.transform(results, "Model")

print(f"\n3. Dados transformados:")
print(f"   feature_importance: {len(report_data.get('feature_importance', {}))} items")
print(f"   model_feature_importance: {len(report_data.get('model_feature_importance', {}))} items")
print(f"   features: {len(report_data.get('features', []))} items")

if report_data.get('feature_importance'):
    print(f"   ✓ feature_importance tem dados!")
    print(f"     Primeiras 5: {list(report_data['feature_importance'].keys())[:5]}")
else:
    print(f"   ✗ feature_importance está VAZIO!")

# Testar o renderer completo
print(f"\n4. Testando renderer completo...")
templates_dir = '/home/guhaase/projetos/DeepBridge/deepbridge/templates'
template_manager = TemplateManager(templates_dir)
asset_manager = AssetManager(templates_dir)
renderer = ResilienceRenderer(template_manager, asset_manager)

output_path = '/home/guhaase/projetos/DeepBridge/simular_lib/analise_v2/results/report_resilience_DEBUG.html'
print(f"   Gerando em: {output_path}")

result_path = renderer.render(
    results=results,
    file_path=output_path,
    model_name="Model",
    report_type="interactive"
)

print(f"\n5. Verificando HTML gerado...")
with open(result_path, 'r') as f:
    html = f.read()

# Verificar window.reportData no HTML
import re
match = re.search(r'window\.reportData\s*=\s*(\{.+?\});', html, re.DOTALL)
if match:
    data_str = match.group(1)
    try:
        report_data_in_html = json.loads(data_str)
        fi_len = len(report_data_in_html.get('feature_importance', {}))
        features_len = len(report_data_in_html.get('features', []))

        print(f"   window.reportData.feature_importance: {fi_len} items")
        print(f"   window.reportData.features: {features_len} items")

        if fi_len > 0:
            print(f"\n   ✓ ✓ ✓ SUCESSO! Os dados foram incluídos corretamente!")
        else:
            print(f"\n   ✗ ✗ ✗ FALHA! Os dados NÃO foram incluídos!")

    except Exception as e:
        print(f"   Erro ao parsear: {e}")
else:
    print(f"   ✗ window.reportData não encontrado!")

print("\n" + "="*70)
print(f"Relatório de debug salvo em: {output_path}")
print("="*70)
