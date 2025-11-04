#!/usr/bin/env python3
"""
Teste para verificar se o save_html funciona corretamente após o fix.
"""
import sys
import json
sys.path.insert(0, '/home/guhaase/projetos/DeepBridge')

from deepbridge.core.experiment.results import ExperimentResult, ResilienceResult

print("="*70)
print("TESTE DO save_html APÓS FIX")
print("="*70)

# Carregar o JSON salvo
json_path = '/home/guhaase/projetos/DeepBridge/simular_lib/analise_v2/results/resilience_results_pixpj.json'
with open(json_path) as f:
    saved_data = json.load(f)

print(f"\n1. JSON carregado: {json_path}")
print(f"   Keys: {list(saved_data.keys())}")

# Criar ExperimentResult como o pipeline faz
experiment_result = ExperimentResult(
    experiment_type='binary_classification',
    config=saved_data.get('experiment_info', {}).get('config', {})
)

print(f"\n2. ExperimentResult criado")

# Adicionar o resultado de resiliência
resilience_result = ResilienceResult(
    results=saved_data['test_results'],
    metadata=saved_data.get('experiment_info', {})
)
experiment_result.add_result(resilience_result)

print(f"   Resilience result adicionado")

# CRÍTICO: Adicionar initial_results como o Experiment faz
# Linha 443 do experiment.py: experiment_result.results['initial_results'] = self.initial_results
if 'initial_model_evaluation' in saved_data:
    experiment_result.results['initial_results'] = saved_data['initial_model_evaluation']
    print(f"   ✓ initial_results adicionado ao experiment_result.results")
else:
    print(f"   ✗ initial_model_evaluation não encontrado no JSON!")

# Chamar save_html
output_path = '/home/guhaase/projetos/DeepBridge/simular_lib/analise_v2/results/report_resilience_SAVEHTML_TEST.html'
print(f"\n3. Chamando save_html...")
print(f"   Output: {output_path}")

try:
    result_path = experiment_result.save_html(
        test_type='resilience',
        file_path=output_path,
        model_name='Model',
        report_type='interactive'
    )
    print(f"   ✓ save_html executado com sucesso!")

    # Verificar o HTML gerado
    print(f"\n4. Verificando HTML gerado...")
    with open(result_path, 'r') as f:
        html = f.read()

    import re
    match = re.search(r'window\.reportData\s*=\s*(\{.+?\});', html, re.DOTALL)
    if match:
        data_str = match.group(1)
        report_data_in_html = json.loads(data_str)

        fi_len = len(report_data_in_html.get('feature_importance', {}))
        features_len = len(report_data_in_html.get('features', []))

        print(f"   window.reportData.feature_importance: {fi_len} items")
        print(f"   window.reportData.features: {features_len} items")

        if fi_len > 0 and features_len > 0:
            print(f"\n   ✓ ✓ ✓ SUCESSO! O FIX FUNCIONOU!")
            print(f"   As abas Overview, Box Plot, Feature Importance e Model Features")
            print(f"   devem renderizar corretamente agora!")
        else:
            print(f"\n   ✗ ✗ ✗ FALHA! Os dados ainda estão vazios!")
    else:
        print(f"   ✗ window.reportData não encontrado!")

except Exception as e:
    print(f"   ✗ Erro ao executar save_html: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
