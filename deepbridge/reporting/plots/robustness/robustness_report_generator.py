"""
Módulo gerador de relatórios de robustez para DeepBridge.
Gera relatórios HTML a partir dos resultados dos testes de robustez.
"""

import os
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import numpy as np
from jinja2 import Template
import traceback # Para melhor relatório de erros

class RobustnessReportGenerator:
    """
    Gera relatórios de análise de robustez com base nos resultados dos testes.
    Utiliza Plotly.js para visualizações interativas.
    """

    def __init__(self):
        """Inicializa o gerador de relatórios de robustez."""
        try:
            # Define o caminho para o template HTML relativo a este arquivo
            self.template_path = os.path.join(
                os.path.dirname(__file__),
                "robustness_report_template.html"
            )
            # Verifica se o template existe durante a inicialização
            if not os.path.exists(self.template_path):
                 raise FileNotFoundError(f"Arquivo de template não encontrado em: {self.template_path}")
        except Exception as e:
            print(f"Erro ao encontrar o caminho do template: {e}")
            # Indica que o template está ausente
            self.template_path = None


    def generate_report(self,
                        results: Dict[str, Any],
                        output_path: str,
                        model_name: str = "Primary Model",
                        experiment_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Gera um relatório de robustez com base nos resultados dos testes.

        Parâmetros:
        -----------
        results : Dict[str, Any]
            Os resultados do teste de robustez (e.g., de experiment._test_results['robustness'])
        output_path : str
            Caminho para salvar o relatório gerado
        model_name : str
            Nome do modelo primário sendo analisado
        experiment_info : Dict[str, Any], opcional
            Informações adicionais do experimento (e.g., de experiment.experiment_info mesclado
            com métricas de comparação)

        Retorna:
        --------
        str : Caminho para o relatório salvo ou uma mensagem de erro
        """
        if self.template_path is None:
             return f"Erro: O template do relatório de robustez não foi encontrado durante a inicialização."
        if experiment_info is None: experiment_info = {}

        try:
            # Extrai resultados para modelos primários e alternativos do dicionário de resultados de robustez
            if 'primary_model' in results:
                # Estrutura aninhada (provavelmente de Experiment.run_tests)
                primary_results = results.get('primary_model', {})
                alternative_models_results = results.get('alternative_models', {})
            else:
                # Assume estrutura plana (e.g., chamada direta de RobustnessSuite para um modelo)
                primary_results = results
                alternative_models_results = {} # Sem alternativas nesta estrutura

            # Garante que primary_results seja um dicionário
            if not isinstance(primary_results, dict):
                 print(f"Aviso: primary_results não é um dict ({type(primary_results)}). Usando dict vazio.")
                 primary_results = {}

            # Prepara os dados do template usando o método refinado
            template_data = self._prepare_template_data(
                primary_results,
                model_name, # Passa o nome específico para o modelo primário
                alternative_models_results,
                experiment_info
            )

            # Renderiza o template
            report_html = self._render_template(template_data)

            # Cria o diretório de saída se necessário
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            # Salva o relatório
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_html)

            return output_path

        except Exception as e:
            error_msg = f"Erro durante a geração do relatório de robustez: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            # Opcionalmente, cria um HTML de erro básico
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(f"<html><body><h1>Erro na Geração do Relatório</h1><pre>{error_msg}</pre></body></html>")
                return f"Relatório de erro salvo em {output_path}"
            except Exception as save_err:
                print(f"Adicionalmente, falha ao salvar relatório de erro: {save_err}")
                return f"Erro ao gerar relatório: {e}. Falha ao salvar detalhes do erro."


    def _prepare_template_data(
        self,
        results: Dict[str, Any], # Resultados do teste de robustez para o modelo primário
        model_name_param: str, # Renomeado para evitar conflito com variável de loop
        alternative_models_results: Optional[Dict[str, Dict[str, Any]]] = None, # Resultados de robustez para alternativas
        experiment_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Prepara dados para o template do relatório, priorizando clareza e consistência.
        Depende fortemente de experiment_info['comparison_metrics_df'] (ou _list) para métricas de desempenho do modelo.
        Usa 'results' e 'alternative_models_results' para detalhes específicos de robustez.
        """
        if experiment_info is None: experiment_info = {}
        if alternative_models_results is None: alternative_models_results = {}

        # --- 1. Informações Básicas de Robustez (do 'results' de robustez do modelo primário) ---
        primary_robustness_score = 1.0 - results.get('avg_overall_impact', 0)
        primary_base_score = results.get('base_score', 0) # AUC usado durante o teste de robustez
        template_data = {
            'model_name': model_name_param,
            'model_type': results.get('model_type', 'Unknown'),
            'base_score': primary_base_score,
            'robustness_score': primary_robustness_score,
            'avg_raw_impact': results.get('avg_raw_impact', 0),
            'avg_quantile_impact': results.get('avg_quantile_impact', 0),
            'n_iterations': results.get('n_iterations', 1),
            'feature_subset': results.get('feature_subset', None),
            'metric_name': results.get('metric', 'AUC') # Métrica usada para base_score/impacto
        }

        # --- 2. Configuração do Experimento (de 'experiment_info') ---
        config = experiment_info.get('config', {})
        dataset_info = config.get('dataset_info', {})
        template_data.update({
            'experiment_type': config.get('experiment_type', experiment_info.get('experiment_type', 'Unknown')),
            'test_size': config.get('test_size', 0.2),
            'random_state': config.get('random_state', 42),
            'n_samples': dataset_info.get('n_samples'),
            'n_features': dataset_info.get('n_features')
        })

        # --- 3. Métricas Comparativas de Modelos (Fonte Primária: experiment_info) ---
        comparison_records_raw = []
        all_model_metrics = {} # Armazena dict de métricas chaveado pelo nome do modelo
        primary_metrics = {} # Armazena dict de métricas do modelo primário

        # Verifica formato preferido primeiro (lista de dicts, provavelmente pré-processado em Experiment)
        if 'comparison_metrics_list' in experiment_info and isinstance(experiment_info['comparison_metrics_list'], list):
            comparison_records_raw = experiment_info['comparison_metrics_list']
        # Verifica formato DataFrame em seguida (precisa de conversão)
        elif 'comparison_metrics' in experiment_info and hasattr(experiment_info['comparison_metrics'], 'iterrows'):
             try:
                 comparison_df = experiment_info['comparison_metrics']
                 # Converte DataFrame para lista de dicts para tratamento consistente
                 temp_records = []
                 for _, row in comparison_df.iterrows():
                     record = {}
                     for col in comparison_df.columns:
                         value = row[col]
                         # Trata valores NaN/Inf do DataFrame antes da serialização JSON posterior
                         if isinstance(value, float):
                             if np.isnan(value): value = None
                             elif np.isinf(value): value = None # Ou trate apropriadamente
                         record[col] = value
                     temp_records.append(record)
                 comparison_records_raw = temp_records
             except Exception as e:
                 print(f"Aviso: Não foi possível processar o DataFrame comparison_metrics: {e}")
                 comparison_records_raw = []
        else:
             print("Aviso: Nenhuma 'comparison_metrics_list' ou DataFrame 'comparison_metrics' encontrado em experiment_info. Tabela/gráfico de métricas pode ser limitado.")
             # Fallback básico: tentar usar o dict 'models' de experiment_info (provavelmente resultados iniciais)
             initial_models_info = experiment_info.get('models', {})
             temp_records = []
             for m_name, m_data in initial_models_info.items():
                 if isinstance(m_data, dict):
                     # Cria um registro básico com nome, tipo e métricas
                     record = {'model_name': m_name, 'model_type': m_data.get('model_type', 'unknown')}
                     record.update(m_data.get('metrics', {})) # Adiciona métricas se existirem
                     temp_records.append(record)
             comparison_records_raw = temp_records


        # Processa registros: Extrai métricas, garante AUC, injeta pontuação de robustez
        processed_records = []
        alternative_models_impact_map = {} # Armazena impacto calculado para alternativas

        # Calcula e armazena impactos de modelos alternativos primeiro
        if alternative_models_results:
            for alt_model_name, alt_model_res in alternative_models_results.items():
                 # Garante que os impactos sejam numéricos antes de calcular
                 avg_raw_impact = alt_model_res.get('avg_raw_impact', 0) if isinstance(alt_model_res.get('avg_raw_impact'), (int, float)) else 0
                 avg_quantile_impact = alt_model_res.get('avg_quantile_impact', 0) if isinstance(alt_model_res.get('avg_quantile_impact'), (int, float)) else 0
                 # Evita erros se os impactos forem None
                 if avg_raw_impact is None: avg_raw_impact = 0
                 if avg_quantile_impact is None: avg_quantile_impact = 0
                 avg_impact = (avg_raw_impact + avg_quantile_impact) / 2
                 alternative_models_impact_map[alt_model_name] = avg_impact

        for record_raw in comparison_records_raw:
            # Faz uma cópia para evitar modificar o dict original se veio de experiment_info
            record = record_raw.copy()
            current_model_name = record.get('model_name')
            # Extrai métricas disponíveis no registro
            model_metrics = {k: v for k, v in record.items() if k not in ['model_name', 'model_type']}

            # --- Garante a métrica AUC padrão ('roc_auc') ---
            auc_val = model_metrics.get('roc_auc')
            # Verifica se é None, NaN ou zero
            if auc_val is None or (isinstance(auc_val, float) and (np.isnan(auc_val) or auc_val == 0)):
                 auc_val = model_metrics.get('auc') # Tenta 'auc'
                 # Verifica novamente
                 if auc_val is None or (isinstance(auc_val, float) and (np.isnan(auc_val) or auc_val == 0)):
                      # Fallback para base_score *apenas* para o resultado de robustez do modelo relevante
                      if current_model_name == model_name_param:
                           auc_val = primary_base_score
                      elif current_model_name in alternative_models_results:
                           # Usa base_score do resultado de robustez do modelo alternativo, se disponível
                           auc_val = alternative_models_results[current_model_name].get('base_score', 0)
                      else:
                           auc_val = 0 # Fallback final se nenhum base_score estiver disponível

            # Garante que roc_auc seja numérico e atualiza o registro e o dict de métricas
            model_metrics['roc_auc'] = auc_val if auc_val is not None else 0
            record['roc_auc'] = model_metrics['roc_auc']
            # Garante que 'auc' corresponda a 'roc_auc' por consistência se 'auc' estava ausente/diferente
            if model_metrics.get('auc') != model_metrics['roc_auc']:
                 model_metrics['auc'] = model_metrics['roc_auc']
                 record['auc'] = model_metrics['roc_auc']

            # --- Injeta Pontuação de Robustez ---
            robustness_val = None
            if current_model_name == model_name_param:
                robustness_val = primary_robustness_score
                primary_metrics = model_metrics # Armazena métricas primárias processadas
            elif current_model_name in alternative_models_impact_map:
                # Pontuação de robustez = 1 - impacto médio
                robustness_val = 1.0 - alternative_models_impact_map[current_model_name]

            record['robustness'] = robustness_val # Pode ser None se não for primário nem alternativo com impacto

            # --- Garante valores padrão para tabela ---
            for metric_key in ['accuracy', 'roc_auc', 'f1', 'precision', 'recall']:
                 # Garante que o valor exista e seja numérico, caso contrário, 0
                 current_val = record.get(metric_key)
                 record[metric_key] = current_val if isinstance(current_val, (int, float)) and not np.isnan(current_val) else 0

            # Armazena métricas processadas e adiciona registro para a tabela
            all_model_metrics[current_model_name] = model_metrics
            processed_records.append(record)

        # Passa a lista de dicionários processados para o template (para a tabela de métricas)
        template_data['comparison_metrics_df'] = processed_records
        template_data['all_model_metrics'] = all_model_metrics
        template_data['primary_metrics'] = primary_metrics

        # --- 4. Dados de Perturbação (do 'results' de robustez do modelo primário) ---
        perturbation_levels = []
        raw_mean_scores, raw_worst_scores, quantile_mean_scores, quantile_worst_scores = [], [], [], []
        raw_results_table, quantile_results_table = [], []
        raw_distributions, quantile_distributions = {}, {}

        # Processa resultados de perturbação raw com segurança
        raw_data = results.get('raw', {}).get('by_level', {})
        if isinstance(raw_data, dict):
            for level_str, level_data in sorted(raw_data.items(), key=lambda x: float(x[0])):
                try: level = float(level_str)
                except ValueError: continue # Pula se o nível não for um número

                perturbation_levels.append(level)
                overall_result = level_data.get('overall_result', {}) if isinstance(level_data, dict) else {}
                # Determina a chave de execução (geralmente 'all_features')
                run_key = 'all_features' if 'all_features' in overall_result else (next(iter(overall_result)) if overall_result else None)

                mean_score, worst_score, std_score, impact = None, None, None, None
                if run_key and run_key in overall_result and isinstance(overall_result[run_key], dict):
                    run_data = overall_result[run_key]
                    mean_score = run_data.get('mean_score')
                    worst_score = run_data.get('worst_score')
                    std_score = run_data.get('std_score')
                    impact = run_data.get('impact')

                    # Extrai distribuições
                    runs_data = level_data.get('runs', {}).get(run_key, [])
                    # Verifica a estrutura esperada antes de acessar
                    if runs_data and isinstance(runs_data, list) and len(runs_data) > 0 and \
                       isinstance(runs_data[0], dict) and 'iterations' in runs_data[0] and \
                       isinstance(runs_data[0]['iterations'], dict):
                        scores = runs_data[0]['iterations'].get('scores', [])
                        if scores: raw_distributions[level] = scores # Armazena lista de scores para o boxplot

                # Adiciona valores (ou None) às listas para plotagem
                raw_mean_scores.append(mean_score)
                raw_worst_scores.append(worst_score)
                # Adiciona dados para a tabela detalhada
                raw_results_table.append((level, {'mean_score': mean_score, 'std_score': std_score, 'worst_score': worst_score, 'impact': impact}))

        # Processa resultados de perturbação quantile (similarmente)
        quantile_data = results.get('quantile', {}).get('by_level', {})
        processed_quantile_levels = set() # Rastreia níveis processados aqui
        temp_quantile_mean_scores = {}
        temp_quantile_worst_scores = {}
        temp_quantile_results_table = {}

        if isinstance(quantile_data, dict):
             for level_str, level_data in sorted(quantile_data.items(), key=lambda x: float(x[0])):
                 try: level = float(level_str)
                 except ValueError: continue
                 processed_quantile_levels.add(level)
                 if level not in perturbation_levels: perturbation_levels.append(level); perturbation_levels.sort() # Adiciona se novo

                 overall_result = level_data.get('overall_result', {}) if isinstance(level_data, dict) else {}
                 run_key = 'all_features' if 'all_features' in overall_result else (next(iter(overall_result)) if overall_result else None)

                 mean_score, worst_score, std_score, impact = None, None, None, None
                 if run_key and run_key in overall_result and isinstance(overall_result[run_key], dict):
                    run_data = overall_result[run_key]
                    mean_score = run_data.get('mean_score')
                    worst_score = run_data.get('worst_score')
                    std_score = run_data.get('std_score')
                    impact = run_data.get('impact')

                    # Extrai distribuições
                    runs_data = level_data.get('runs', {}).get(run_key, [])
                    if runs_data and isinstance(runs_data, list) and len(runs_data) > 0 and \
                       isinstance(runs_data[0], dict) and 'iterations' in runs_data[0] and \
                       isinstance(runs_data[0]['iterations'], dict):
                        scores = runs_data[0]['iterations'].get('scores', [])
                        if scores: quantile_distributions[level] = scores

                 temp_quantile_mean_scores[level] = mean_score
                 temp_quantile_worst_scores[level] = worst_score
                 temp_quantile_results_table[level] = (level, {'mean_score': mean_score, 'std_score': std_score, 'worst_score': worst_score, 'impact': impact})

        # Garante que os scores quantile estejam alinhados com perturbation_levels finais
        quantile_mean_scores = [temp_quantile_mean_scores.get(lvl) for lvl in perturbation_levels]
        quantile_worst_scores = [temp_quantile_worst_scores.get(lvl) for lvl in perturbation_levels]
        quantile_results_table = [temp_quantile_results_table.get(lvl, (lvl, {})) for lvl in perturbation_levels]


        template_data.update({
            'perturbation_levels': perturbation_levels,
            'raw_mean_scores': raw_mean_scores,
            'raw_worst_scores': raw_worst_scores,
            'quantile_mean_scores': quantile_mean_scores,
            'quantile_worst_scores': quantile_worst_scores,
            'raw_results': raw_results_table,
            'quantile_results': quantile_results_table,
            'raw_distributions': raw_distributions,
            'quantile_distributions': quantile_distributions,
        })

        # --- 5. Comparação de Subconjunto de Features (se aplicável) ---
        feature_subset_comparison = None
        # Usa raw_data já processado e seguro
        if template_data['feature_subset'] and isinstance(raw_data, dict):
            all_features_scores = []
            subset_features_scores = []
            raw_by_level_map = {float(k): v for k, v in raw_data.items() if isinstance(v, dict)} # Mapeia level para dados
            for level in perturbation_levels: # Alinha com os níveis finais
                level_data = raw_by_level_map.get(level, {})
                overall = level_data.get('overall_result', {}) if isinstance(level_data, dict) else {}
                # Obtém scores ou None se ausente
                all_features_scores.append(overall.get('all_features', {}).get('mean_score'))
                subset_features_scores.append(overall.get('feature_subset', {}).get('mean_score'))

            # Inclui apenas se ambas as listas tiverem dados válidos (não apenas Nones)
            if any(s is not None for s in all_features_scores) and any(s is not None for s in subset_features_scores):
                 feature_subset_comparison = {
                     'all_features': all_features_scores,
                     'feature_subset': subset_features_scores
                 }
        # Garante que a chave exista com valores padrão para o template
        template_data['feature_subset_comparison'] = feature_subset_comparison or {'all_features': [], 'feature_subset': []}


        # --- 6. Importância de Features (do 'results' de robustez do modelo primário) ---
        feature_names_imp = []
        feature_importance_values_imp = []
        feature_importance_items_imp = []
        feature_importance_pct_values_imp = []
        feature_importance_pct_imp = []

        feature_importance_data = results.get('feature_importance', {})
        if isinstance(feature_importance_data, dict):
            # Filtra chaves internas como '_detailed_results'
            feature_importance = {k: v for k, v in feature_importance_data.items() if not k.startswith('_')}

            if feature_importance:
                # Ordena por importância absoluta, tratando valores não numéricos graciosamente
                try:
                    sorted_features = sorted(
                        feature_importance.items(),
                        key=lambda item: abs(item[1]) if isinstance(item[1], (int, float)) else 0,
                        reverse=True
                    )
                except Exception as sort_err: # Captura erros de ordenação inesperados
                     print(f"Aviso: Erro ao ordenar feature_importance: {sort_err}. Usando ordem original.")
                     sorted_features = list(feature_importance.items())


                feature_names_imp = [f[0] for f in sorted_features]
                # Armazena valores originais, incluindo possíveis não numéricos se necessário em outro lugar
                feature_importance_values_imp = [f[1] for f in sorted_features]
                feature_importance_items_imp = sorted_features # Lista de tuplas (nome, valor) ordenada

                # Calcula porcentagens apenas de valores numéricos
                numeric_importance_values = [v for v in feature_importance_values_imp if isinstance(v, (int, float)) and not np.isnan(v)]
                total_abs_importance = sum(abs(v) for v in numeric_importance_values)

                if total_abs_importance > 0:
                    # Valores percentuais para plot (baseado em importâncias numéricas)
                    feature_importance_pct_values_imp = [(100 * abs(v) / total_abs_importance) for v in numeric_importance_values]
                    # Itens percentuais para tabela (nome, valor_pct), filtrado para numéricos
                    feature_importance_pct_imp = [(name, 100 * abs(val) / total_abs_importance)
                                                 for name, val in sorted_features if isinstance(val, (int, float)) and not np.isnan(val)]

        template_data.update({
            'feature_names': feature_names_imp, # Nomes ordenados por importância
            'feature_importance_values': feature_importance_values_imp, # Valores originais
            'feature_importance': feature_importance_items_imp, # Itens (nome, valor) ordenados
            'feature_importance_pct_values': feature_importance_pct_values_imp, # Valores % para plot
            'feature_importance_pct': feature_importance_pct_imp, # Itens % (nome, pct) para tabela
        })


        # --- 7. Dados para Plotagem de Modelos Alternativos (de 'alternative_models_results') ---
        alt_model_names_plot = list(alternative_models_results.keys()) # Nomes baseados nos resultados de robustez
        raw_model_scores_plot = []
        quantile_model_scores_plot = []
        alternative_models_impact_plot = [] # Para o gráfico de barras de impacto


        if alternative_models_results:
             # Usa o mapa calculado anteriormente para impacto
             alternative_models_impact_plot = [alternative_models_impact_map.get(name, 0) for name in alt_model_names_plot]

             for alt_model_name in alt_model_names_plot:
                 alt_results = alternative_models_results.get(alt_model_name, {}) # Obtem com segurança
                 alt_raw_scores = []
                 alt_quantile_scores = []

                 # Obtem dados por nível com segurança
                 alt_raw_by_level = alt_results.get('raw', {}).get('by_level', {}) if isinstance(alt_results.get('raw'), dict) else {}
                 alt_quantile_by_level = alt_results.get('quantile', {}).get('by_level', {}) if isinstance(alt_results.get('quantile'), dict) else {}

                 # Extrai scores alinhados com os perturbation_levels principais
                 for level in perturbation_levels:
                     level_str = str(level)
                     # Raw
                     level_data_raw = alt_raw_by_level.get(level_str, {})
                     overall_raw = level_data_raw.get('overall_result', {}) if isinstance(level_data_raw, dict) else {}
                     run_key_raw = 'all_features' if 'all_features' in overall_raw else (next(iter(overall_raw)) if overall_raw else None)
                     alt_raw_scores.append(overall_raw.get(run_key_raw, {}).get('mean_score') if run_key_raw else None)
                     # Quantile
                     level_data_quantile = alt_quantile_by_level.get(level_str, {})
                     overall_quantile = level_data_quantile.get('overall_result', {}) if isinstance(level_data_quantile, dict) else {}
                     run_key_quantile = 'all_features' if 'all_features' in overall_quantile else (next(iter(overall_quantile)) if overall_quantile else None)
                     alt_quantile_scores.append(overall_quantile.get(run_key_quantile, {}).get('mean_score') if run_key_quantile else None)

                 raw_model_scores_plot.append(alt_raw_scores)
                 quantile_model_scores_plot.append(alt_quantile_scores)


        template_data.update({
            'model_names': alt_model_names_plot, # Nomes dos modelos alternativos *testados para robustez*
            'raw_model_scores': raw_model_scores_plot,
            'quantile_model_scores': quantile_model_scores_plot,
            'alternative_models_impact': alternative_models_impact_plot, # Impactos para gráfico de barras
        })

        # --- 8. Prepara Lista de Métricas para Plotagem (derivado de processed_records) ---
        # Inclui apenas modelo primário + alternativas *testadas para robustez* no gráfico de métricas
        metrics_plot_model_names = [model_name_param] + alt_model_names_plot
        primary_metrics_values_plot = [0] * 5 # Padrão para 5 métricas (Acc, AUC, F1, Prec, Recall)
        alt_metrics_values_plot = [] # Lista de listas para modelos alt

        # Encontra o registro processado do modelo primário
        primary_rec = next((r for r in processed_records if r.get('model_name') == model_name_param), None)
        if primary_rec:
             # Usa os valores já verificados e padronizados do registro processado
             primary_metrics_values_plot = [
                 primary_rec.get('accuracy', 0),
                 primary_rec.get('roc_auc', 0),
                 primary_rec.get('f1', 0),
                 primary_rec.get('precision', 0),
                 primary_rec.get('recall', 0)
             ]

        # Extrai métricas *apenas* para os modelos alternativos testados para robustez
        for alt_name in alt_model_names_plot: # Itera apenas sobre modelos em alt_model_names_plot
             alt_rec = next((r for r in processed_records if r.get('model_name') == alt_name), None)
             if alt_rec:
                 # Usa os valores já verificados e padronizados do registro processado
                 alt_metrics_values_plot.append([
                     alt_rec.get('accuracy', 0),
                     alt_rec.get('roc_auc', 0),
                     alt_rec.get('f1', 0),
                     alt_rec.get('precision', 0),
                     alt_rec.get('recall', 0)
                 ])
             else: # Adiciona padrões se as métricas estiverem ausentes para este modelo alt
                 alt_metrics_values_plot.append([0] * 5)


        template_data['primary_metrics_values'] = primary_metrics_values_plot
        template_data['alt_metrics_values'] = alt_metrics_values_plot


        # --- 9. Limpeza Final para Serialização JSON ---
        # Garante que NaN/Inf sejam convertidos para None antes de renderizar
        def clean_for_json(data):
            if isinstance(data, dict):
                return {k: clean_for_json(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [clean_for_json(item) for item in data]
            elif isinstance(data, tuple):
                 return tuple(clean_for_json(item) for item in data) # Limpa tuplas também
            elif isinstance(data, float): # Trata float nativo do Python
                if np.isnan(data) or np.isinf(data):
                    return None # Representa como null em JSON
                return data
            # Trata tipos numpy se passarem
            elif isinstance(data, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                 return int(data)
            # CORREÇÃO APLICADA AQUI: np.float_ removido
            elif isinstance(data, (np.float16, np.float32, np.float64)):
                 # Verifica NaN/Inf para os tipos float específicos do NumPy
                 return None if np.isnan(data) or np.isinf(data) else float(data)
            elif isinstance(data, (np.ndarray,)): # Trata arrays numpy
                 return clean_for_json(data.tolist()) # Converte para lista
            elif isinstance(data, (np.bool_)):
                 return bool(data)
            elif isinstance(data, (np.void)): # Como de structured arrays
                 return None # Ou outra representação apropriada
            return data

        template_data_cleaned = clean_for_json(template_data)

        return template_data_cleaned


    def _render_template(self, template_data: Dict[str, Any]) -> str:
        """
        Renderiza o template HTML com os dados fornecidos.
        """
        if self.template_path is None:
             raise RuntimeError("Caminho do template não foi definido corretamente durante a inicialização.")

        try:
            with open(self.template_path, 'r', encoding='utf-8') as f:
                template_content = f.read()

            template = Template(template_content)

            # Adiciona um filtro 'tojson' seguro para embutir dados no <script>
            # Usando json.dumps com um manipulador para tipos não serializáveis padrão
            def safe_json_dumps(obj, **kwargs):
                 # Função auxiliar para lidar com tipos não serializáveis
                 def default_serializer(o):
                      if isinstance(o, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                           return int(o)
                      elif isinstance(o, (np.float16, np.float32, np.float64)):
                           # Já deve ter sido tratado por clean_for_json, mas como fallback
                           return float(o) if not (np.isnan(o) or np.isinf(o)) else None
                      elif isinstance(o, (np.bool_)):
                           return bool(o)
                      elif isinstance(o, (np.void)):
                           return None
                      # Último recurso: converte para string
                      try:
                          return str(o)
                      except Exception: # Caso str(o) falhe
                          return repr(o) # Usa repr como fallback final

                 return json.dumps(obj, default=default_serializer, **kwargs)

            template.globals['tojson'] = safe_json_dumps
            return template.render(**template_data)
        except FileNotFoundError:
             raise FileNotFoundError(f"Não foi possível encontrar o arquivo de template em: {self.template_path}")
        except Exception as e:
             # Adiciona mais detalhes ao erro de renderização
             detailed_error = f"Falha ao renderizar template Jinja2: {e}\n"
             # Tenta obter informações sobre qual parte dos dados pode ter causado o problema
             try:
                  # Tenta serializar para ver se há erro, usando o mesmo serializador seguro
                  safe_json_dumps(template_data)
             except Exception as json_err:
                  detailed_error += f"Possível problema de serialização JSON nos dados: {json_err}\n"
             detailed_error += traceback.format_exc()
             raise RuntimeError(detailed_error)


# Função wrapper standalone permanece similar, mas se beneficia
# do tratamento de erros aprimorado dentro da classe.
def generate_robustness_report(
    results: Dict[str, Any],
    output_path: str,
    model_name: str = "Primary Model",
    experiment_info: Optional[Dict[str, Any]] = None
) -> str:
    """
    Gera um relatório de robustez a partir dos resultados do teste usando RobustnessReportGenerator.

    Parâmetros:
    -----------
    results : Dict[str, Any]
        Dicionário de resultados do teste de robustez, potencialmente aninhado.
    output_path : str
        Caminho para salvar o relatório.
    model_name : str, opcional
        Nome do modelo primário.
    experiment_info : Dict[str, Any], opcional
        Dicionário de informações adicionais do experimento.

    Retorna:
    --------
    str : Caminho para o relatório salvo ou uma mensagem de erro.
    """
    try:
        # Validação básica de entrada
        if not isinstance(results, dict):
            # Se não for dict, provavelmente indica um erro na chamada anterior (Experiment)
            # É mais seguro lançar um erro aqui do que tentar adivinhar a estrutura.
            raise TypeError(f"Input 'results' deve ser um dicionário, mas recebeu {type(results)}")

        if experiment_info is not None and not isinstance(experiment_info, dict):
             print(f"Aviso: experiment_info não é um dicionário ({type(experiment_info)}). Usando dict vazio.")
             experiment_info = {}
        elif experiment_info is None:
             experiment_info = {} # Garante que seja um dict


        # Cria gerador (inicialização verifica o caminho do template)
        generator = RobustnessReportGenerator()
        if generator.template_path is None: # Verifica se a inicialização falhou
             raise FileNotFoundError("Caminho do template não pôde ser determinado.")

        # Gera relatório (lógica principal e tratamento de erros agora estão dentro do método da classe)
        report_path_or_error = generator.generate_report(
            results,
            output_path,
            model_name,
            experiment_info
        )
        return report_path_or_error

    except FileNotFoundError as e:
         # Tratamento específico para erro de template não encontrado da __init__ ou _render_template
         print(f"Geração do relatório falhou: {e}")
         return f"Erro: Não foi possível gerar o relatório porque o arquivo de template está ausente. Detalhes: {e}"
    except Exception as e:
        # Captura erros que ocorrem *antes* ou *durante* a chamada para generate_report
        error_msg = f"Erro crítico configurando a geração do relatório de robustez: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        # Tenta salvar um relatório de erro mínimo
        try:
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                 f.write(f"<html><body><h1>Erro na Geração do Relatório</h1><pre>{error_msg}</pre></body></html>")
            return f"Erro crítico impediu a geração do relatório. Relatório de erro salvo em {output_path}"
        except Exception as save_err:
            print(f"Adicionalmente, falha ao salvar relatório de erro crítico: {save_err}")
            return f"Erro crítico ao gerar relatório: {e}. Falha ao salvar detalhes do erro."